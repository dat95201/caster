#!/usr/bin/env bash
#
# A fleet: N shards of one stage, launched into the background on one machine,
# and one place to ask where they are.
#
#   bash scripts/fleet.sh launch --shards 6 -- --policies km,natural --budget 20
#   bash scripts/fleet.sh status
#   bash scripts/fleet.sh tail 3          # or `tail` for every shard's last lines
#   bash scripts/fleet.sh wait            # blocks; the only command that does
#   bash scripts/fleet.sh stop
#
# Launching returns immediately — that is the whole point on a Colab runtime,
# where a blocking cell is a cell you cannot use. Nothing here cuts a corpus or
# writes a row: every shard is the same `caster evaluate` invocation you would
# have typed by hand. What this script owns is the three things that go wrong
# when you type six of those into a notebook cell.
#
#   ranges     Contiguous, non-overlapping, covering the universe exactly, with
#              the remainder on the FIRST shards. Hand-typed ranges are off by
#              one eventually and neither mistake announces itself: a skipped
#              index turns up in the merge audit days later, and an overlapped
#              one is silently re-run work whose only symptom is time.
#
#   the cap    A per-process spend cap is a real ceiling of N× it. The GATEWAY
#              owns one global ledger and one global cap; when a fleet
#              is launched without one, the cap is divided here and the
#              arithmetic is printed, because the alternative is pushing it onto
#              the operator.
#
#   the books  One log per shard under logs/<run>/fleet/, plus a manifest saying
#              which pid is walking which range under which command.
#              `output2.log` beside `output11.log` stops being a record of
#              anything the moment the session that named them is gone — and on
#              Colab that is hours.
#
# THE UNIVERSE SIZE IS NEVER WRITTEN DOWN HERE. It is read out of
# `caster evaluate --dry-run`, which is the one authority on how big a stage is.
# Two scripts holding two opinions about that cut ranges past the end of it.
#
# Sizing a fleet. Phase 2 is CPU-bound, so the useful shard count is bounded by
# CORES — measured, not assumed: throughput was still climbing at 8 concurrent
# and flat by 12. Phase 1 is bounded by the GATEWAY instead. And phase-2
# parallelism must be PROCESSES, never threads inside one driver, because
# getrusage(RUSAGE_CHILDREN) is process-global — which is exactly what a
# fleet of separate processes gives you.
#
# One fleet at a time: a second launch refuses rather than interleaving two
# manifests. `stop` first, or `wait`.
#
# Not a substitute for the merge audit. Every shard reporting `done` says every
# process exited 0, nothing more. Coverage is `caster merge`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=run_dir_paths.sh
. scripts/run_dir_paths.sh

FLEET_ROOT="$RUN_LOGS/fleet"
# A file, not a symlink: logs/ is often a Drive mount, and Drive and symlinks
# are a bad pair.
MANIFEST="$FLEET_ROOT/manifest.json"
STAGGER="${STAGGER:-5}"
CASTER="${CASTER:-python3 -m caster.cli}"

die() { echo "fleet: $*" >&2; exit 2; }

usage() {
  sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

  launch  --shards N [--from A --to B] [--stagger S] [-- ...]

      --from/--to   1-based inclusive, over the stage's own universe. Omit both
                    to take all of it; the size is ASKED of the stage rather
                    than written down here.
      --shards N    how many background shards to cut the range into.
      --stagger S   seconds between launches (default 5). Not cosmetic: N shards
                    starting in the same second all read the same done-index and
                    open the same corpus at once.
      --            everything after it is forwarded verbatim to `caster
                    evaluate`. This script deliberately does not know that list.

  status            one line per shard: alive, progress, elapsed.
  tail [N]          follow shard N, or the last lines of all.
  wait              block until the fleet drains, then print the board.
  stop              SIGTERM the fleet. Shards resume on an identical re-run.
USAGE
}

# ── where a universe size comes from ─────────────────────────────────────────
# Not from here. A dry run also proves the ARGUMENTS are good before six
# processes discover otherwise one at a time.
universe_size() {
  local out n
  out="$($CASTER evaluate --dry-run "$@" 2>&1)" || {
    echo "$out" >&2
    die "caster evaluate --dry-run refused these arguments; fix that before fanning out"
  }
  n="$(sed -n 's/^universe  *\([0-9][0-9]*\) cells.*/\1/p' <<<"$out" | head -1)"
  [[ "$n" =~ ^[0-9]+$ ]] || { echo "$out" >&2; die "could not read the universe size out of --dry-run"; }
  echo "$n"
}

cmd_launch() {
  local shards=0 from=0 to=0 stagger="$STAGGER"
  local -a forward=()
  while (($#)); do
    case "$1" in
      --shards) shards="$2"; shift 2 ;;
      --from)   from="$2";   shift 2 ;;
      --to)     to="$2";     shift 2 ;;
      --stagger) stagger="$2"; shift 2 ;;
      --) shift; forward=("$@"); break ;;
      *) die "unknown argument $1 (did you mean to put it after --?)" ;;
    esac
  done
  ((shards >= 1)) || die "--shards must be at least 1"

  local universe
  universe="$(universe_size "${forward[@]}")"
  ((from == 0)) && from=1
  ((to == 0)) && to="$universe"
  ((to <= universe)) || die "--to $to is past the universe ($universe cells)"

  mkdir -p "$FLEET_ROOT"
  python3 - "$MANIFEST" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
if path.is_file():
    doc = json.loads(path.read_text())
    alive = []
    for e in doc.get("shards", []):
        try:
            os.kill(int(e["pid"]), 0)
            alive.append(e["pid"])
        except (OSError, ValueError, TypeError, KeyError):
            pass
    if alive:
        print(f"fleet: {len(alive)} shard(s) still running (pids {alive}). "
              f"`stop` or `wait` first: a second launch would interleave two "
              f"manifests and neither would be a record of anything.",
              file=sys.stderr)
        raise SystemExit(2)
PY

  # The cap. One global ledger lives in the gateway; without one, divide and
  # SHOW THE ARITHMETIC rather than leaving a ceiling of N× the number in .env.
  local cap="${BUDGET_USD_CAP:-}"
  if [[ -n "$cap" && "${LLM_BASE_URL:-}" != *":1153"* ]]; then
    local per
    per="$(python3 -c "print(round(float('$cap')/$shards, 4))")"
    echo "fleet: BUDGET_USD_CAP=$cap / $shards shards = $per each"
    echo "       (a per-process cap is a real ceiling of ${shards}× it — run"
    echo "        \`caster serve gateway\` for one global ledger instead)"
    export BUDGET_USD_CAP="$per"
  fi

  # NOT `mapfile < <(python3 …)`: process substitution does not trip
  # `pipefail`, so a ValueError out of plan_ranges (e.g. --shards > span) would
  # leave `ranges` empty, the loop body would never run, and a 0-shard manifest
  # would be written as a success. Capture, check the status, THEN split.
  local ranges_raw
  if ! ranges_raw="$(python3 -c "
from caster.run.shard import plan_ranges
for lo, hi in plan_ranges($from, $to, $shards):
    print(lo, hi)
" 2>&1)"; then
    echo "$ranges_raw" >&2
    die "could not cut [$from..$to] into $shards shard(s)"
  fi
  local -a ranges=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && ranges+=("$line")
  done <<<"$ranges_raw"
  (( ${#ranges[@]} == shards )) || die "cut $from..$to into ${#ranges[@]} range(s), expected $shards"

  local -a entries=()
  local i=0
  for r in "${ranges[@]}"; do
    read -r lo hi <<<"$r"
    local log="$FLEET_ROOT/shard-$(printf '%02d' "$i").log"
    : >"$log"
    # shellcheck disable=SC2086
    nohup $CASTER evaluate --shard "$i" --first "$lo" --last "$hi" \
          "${forward[@]}" >>"$log" 2>&1 &
    local pid=$!
    entries+=("$(python3 -c "
import json,sys
print(json.dumps({'index': $i, 'first': $lo, 'last': $hi, 'pid': $pid,
                  'command': ['caster','evaluate','--shard','$i','--first','$lo','--last','$hi'],
                  'log': '$log'}))")")
    echo "fleet: shard $i  [$lo..$hi]  pid $pid  -> $log"
    i=$((i + 1))
    sleep "$stagger"
  done

  python3 - "$MANIFEST" "$universe" "${entries[@]}" <<'PY'
import json, pathlib, sys, time
path, universe, *rows = sys.argv[1:]
doc = {"universe": int(universe), "at": time.time(),
       "shards": [json.loads(r) for r in rows]}
pathlib.Path(path).write_text(json.dumps(doc, indent=2))
PY
  echo "fleet: manifest -> $MANIFEST"
  echo "fleet: \`bash scripts/fleet.sh status\` · \`… wait\` · \`… stop\`"
}

cmd_status() {
  [[ -f "$MANIFEST" ]] || die "no manifest at $MANIFEST — nothing launched from here"
  python3 - "$MANIFEST" <<'PY'
import json, os, pathlib, sys, time
doc = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(f"universe {doc['universe']} cells · launched "
      f"{time.strftime('%H:%M:%S', time.localtime(doc['at']))}")
for e in doc["shards"]:
    try:
        os.kill(int(e["pid"]), 0)
        state = "alive"
    except (OSError, ValueError):
        state = "done "
    log = pathlib.Path(e["log"])
    last = ""
    if log.is_file():
        lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
        last = lines[-1][:88] if lines else ""
    print(f"  s{e['index']:02d} {state} [{e['first']}..{e['last']}] pid {e['pid']}  {last}")
PY
}

cmd_tail() {
  [[ -f "$MANIFEST" ]] || die "no manifest at $MANIFEST"
  if [[ $# -ge 1 ]]; then
    tail -f "$FLEET_ROOT/shard-$(printf '%02d' "$1").log"
  else
    for f in "$FLEET_ROOT"/shard-*.log; do
      echo "── $f"; tail -n 5 "$f"; echo
    done
  fi
}

cmd_wait() {
  [[ -f "$MANIFEST" ]] || die "no manifest at $MANIFEST"
  while python3 - "$MANIFEST" <<'PY'
import json, os, pathlib, sys
doc = json.loads(pathlib.Path(sys.argv[1]).read_text())
alive = 0
for e in doc["shards"]:
    try:
        os.kill(int(e["pid"]), 0); alive += 1
    except (OSError, ValueError):
        pass
raise SystemExit(0 if alive else 1)
PY
  do sleep 10; done
  cmd_status
  echo
  echo "fleet: every process exited — which says EXACTLY that and nothing more."
  echo "       Coverage is \`caster merge\` (Invariant 12)."
}

cmd_stop() {
  [[ -f "$MANIFEST" ]] || die "no manifest at $MANIFEST"
  python3 - "$MANIFEST" <<'PY'
import json, os, pathlib, signal, sys
doc = json.loads(pathlib.Path(sys.argv[1]).read_text())
for e in doc["shards"]:
    try:
        os.kill(int(e["pid"]), signal.SIGTERM)
        print(f"fleet: SIGTERM -> s{e['index']:02d} pid {e['pid']}")
    except (OSError, ValueError):
        pass
PY
  echo "fleet: re-issue the identical launch to resume — a cell is written whole"
  echo "       or not at all, so there is nothing half-finished to clean up."
}

case "${1:-}" in
  launch) shift; cmd_launch "$@" ;;
  status) cmd_status ;;
  tail)   shift; cmd_tail "$@" ;;
  wait)   cmd_wait ;;
  stop)   cmd_stop ;;
  ""|-h|--help|help) usage ;;
  *) usage; die "unknown command ${1}" ;;
esac
