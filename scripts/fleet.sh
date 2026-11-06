#!/usr/bin/env bash
#
# A fleet: N shards of one stage, launched into the background on one machine,
# and one place to ask where they are.
#
#   bash scripts/fleet.sh launch --stage propose --shards 6 -- --policies no-mem
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

  launch  --shards N [--stage propose|evaluate] [--shard-base B]
          [--from A --to B] [--stagger S] [-- ...]
      --shard-base  offset for the shard INDEX, default 0. Shards write to
                    data/<run>/shards/sNN, so two machines launching into one
                    Drive-backed run directory with the default both write
                    s00 — same rounds.jsonl, same cells.done, same meta.json,
                    and `meta_conflict` cannot see it because the protocol is
                    identical. Give the second machine `--shard-base 10` and
                    the two never touch the same directory; `merge` globs
                    `shards/*` and picks both up.

      --stage       which verb the shards run. `evaluate` (phase 2) is the
                    default. `propose` is PHASE 1 and it is not the same stage
                    wearing a flag: it writes one level deeper, where the
                    merge cannot reach it, and it resumes against its own
                    done-index. Running phase 1 as `evaluate --force-full-budget`
                    — which is what this script used to force — put phase-1
                    rows in the evidence namespace and marked those cells done,
                    so phase 2 skipped the arm and merged phase-1 rows in its
                    place.
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
  local stage="$1"; shift
  local out n
  out="$($CASTER "$stage" --dry-run "$@" 2>&1)" || {
    echo "$out" >&2
    die "caster $stage --dry-run refused these arguments; fix that before fanning out"
  }
  n="$(sed -n 's/^universe  *\([0-9][0-9]*\) cells.*/\1/p' <<<"$out" | head -1)"
  [[ "$n" =~ ^[0-9]+$ ]] || { echo "$out" >&2; die "could not read the universe size out of --dry-run"; }
  echo "$n"
}

cmd_launch() {
  local shards=0 from=0 to=0 stagger="$STAGGER" stage="evaluate" base=0
  local -a forward=()
  while (($#)); do
    case "$1" in
      --stage)  stage="$2";  shift 2 ;;
      --shard-base) base="$2"; shift 2 ;;
      --shards) shards="$2"; shift 2 ;;
      --from)   from="$2";   shift 2 ;;
      --to)     to="$2";     shift 2 ;;
      --stagger) stagger="$2"; shift 2 ;;
      --) shift; forward=("$@"); break ;;
      *) die "unknown argument $1 (did you mean to put it after --?)" ;;
    esac
  done
  ((shards >= 1)) || die "--shards must be at least 1"
  [[ "$stage" == "propose" || "$stage" == "evaluate" ]] \
    || die "--stage must be propose or evaluate, got '$stage'"

  local universe
  universe="$(universe_size "$stage" "${forward[@]}")"
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
    local log="$FLEET_ROOT/shard-$(printf '%02d' "$((base + i))").log"
    local rc="$log.rc"
    : >"$log"
    rm -f "$rc"
    # The shard runs inside a subshell that records its EXIT CODE. Without this
    # `status` can only ask whether the pid is alive, so a shard killed at its
    # first cell and a shard that walked every cell both render as "done" —
    # which is not a record of anything. The rc file is written whole, after
    # the process ends, so its absence means "still running or killed by a
    # signal the shell never saw" and is reported as such rather than as
    # success.
    # shellcheck disable=SC2086
    nohup bash -c '"$@" >>"'"$log"'" 2>&1; printf "%s" "$?" > "'"$rc"'"' \
          _ $CASTER "$stage" --shard "$((base + i))" --first "$lo" --last "$hi" \
          "${forward[@]}" >/dev/null 2>&1 &
    local pid=$!
    entries+=("$(python3 -c "
import json,sys
print(json.dumps({'index': $i, 'first': $lo, 'last': $hi, 'pid': $pid,
                  'command': ['caster','$stage','--shard','$i','--first','$lo','--last','$hi'],
                  'log': '$log', 'rc': '$rc'}))")")
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
import json, os, pathlib, re, sys, time
doc = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(f"universe {doc['universe']} cells · launched "
      f"{time.strftime('%H:%M:%S', time.localtime(doc['at']))}")

PROGRESS = re.compile(r"^\s*\[(\d+)/(\d+)\]")
worst = 0
for e in doc["shards"]:
    alive = True
    try:
        os.kill(int(e["pid"]), 0)
    except (OSError, ValueError):
        alive = False

    # The exit code is the whole point: a shard killed at its first cell and a
    # shard that walked every cell are indistinguishable by pid liveness alone.
    rc_path = pathlib.Path(e.get("rc") or (e["log"] + ".rc"))
    rc = None
    if rc_path.is_file():
        try:
            rc = int(rc_path.read_text().strip() or "-1")
        except ValueError:
            rc = -1

    if alive:
        state = "alive  "
    elif rc == 0:
        state = "OK     "
    elif rc is None:
        # No rc file and no process: the shell never saw an exit. On a notebook
        # runtime that is almost always the OOM killer or a lost session.
        state = "KILLED "
        worst = 1
    else:
        state = "FAIL:%-2d" % rc
        worst = 1

    walked = total = 0
    log = pathlib.Path(e["log"])
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            m = PROGRESS.match(line)
            if m:
                walked, total = int(m.group(1)), int(m.group(2))
    cells = "%d/%d cells" % (walked, total) if total else "no cell started"
    note = ""
    if not alive and total and walked < total:
        note = "  <-- STOPPED %d cells SHORT" % (total - walked)
        worst = 1
    if not alive and not total:
        note = "  <-- died before the first cell; read the log"
        worst = 1
    print("  s%02d %s [%s..%s] pid %s  %s%s"
          % (e["index"], state, e["first"], e["last"], e["pid"], cells, note))

raise SystemExit(worst)
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
  # A shard that skipped cells on a cache miss exits 3, and `status` prints
  # that code — but the list of what it owed is in the log, and a fleet log is
  # read once. Surface it here, where the operator is already looking.
  local skipped
  skipped="$(grep -h "SKIPPED on a cache miss" "$FLEET_ROOT"/shard-*.log 2>/dev/null || true)"
  if [[ -n "$skipped" ]]; then
    echo
    echo "fleet: cells were SKIPPED on a cache miss — phase 1 owes them:"
    sed 's/^/       /' <<<"$skipped"
    echo "       They are unwritten and unindexed; re-issue this launch once"
    echo "       phase 1 (or \`caster audit\`) has filled the gap."
  fi
  if cmd_status; then
    echo
    echo "fleet: every shard exited 0 and walked its whole range."
    echo "       That is still not coverage — \`caster merge\` is (Invariant 12)."
  else
    echo
    echo "fleet: AT LEAST ONE SHARD DID NOT FINISH. Nothing below this line is a"
    echo "       complete grid. Read the log of every shard not marked OK:"
    echo "         bash scripts/fleet.sh tail N"
    echo "       Re-issuing the identical launch resumes from cells.done."
    return 1
  fi
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
