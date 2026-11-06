"""Sharding: range cutting, the manifest, and the six merge audits.

Four rules, each of which costs a run to learn the other way:

* **contiguous, non-overlapping ranges covering `[first, last]` exactly**, with
  the remainder going to the **first** shards, one index each, never all onto
  the last. A fat tail is the shard still running when the others finished.
* **one log per shard**, plus a manifest of pid → range → command. `output2.log`
  beside `output11.log` stops being a record of anything the moment the session
  that named them is gone — and on a Colab runtime that is hours.
* **one fleet at a time.** A second launch refuses rather than interleaving
  manifests.
* **the universe size is computed in exactly one place** — by asking the driver
  for a plan (`driver.plan`) and reading the count out of it. The fleet never
  hardcodes a universe size: two scripts holding two opinions about how big a
  stage is cut ranges past the end of it.
* **the spend cap is divided by the shard count** when shards do not go through
  the gateway, and the arithmetic is printed. N shards each honouring a cap of
  C is a real ceiling of C·N. (With the gateway there is one ledger and one cap
  — and this division is skipped.)

`done` from every shard means "every process exited 0" and **nothing more**.
Coverage is a separate merge audit, and there are **six** of them. The two that
are easy to leave out are the ones that catch instrument drift:

| audit | what it means |
|---|---|
| protocol disagreement | a shard ran under a different model or policy protocol. Rows would never pool. **Hard stop** |
| **runtime disagreement** | the shards' `.meta.json` differ on served context window, model digest, temperature, sandbox timeout, backend or reasoning effort. **None of these reaches the cache key**, so those shards re-judged each other's draws against a different instrument. **Hard stop** |
| no protocol record | a shard log with no `.meta.json` — a hand-run driver. Reported, not fatal |
| `GAPS` / `TRUNCATED` | cells missing from an arm; cells that neither accepted nor reached the budget |
| `DISAGREEMENT` | the same (cell, round) collected twice with different results — usually a wall-clock timeout firing on the slower machine. Fix and re-run both; **do not pick a winner** |
| **`FOREIGN`** | a task not in the universe — a shard cut from a different corpus |

Invariant 14: **smoke output is never mergeable.** `inputs()` globs
`shards/*/rounds.jsonl`, and `caster smoke` writes under `data/<run>/smoke/`,
which that glob cannot match — by construction, not by a naming convention an
operator has to remember.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import time
from typing import Any, Iterable, Sequence

from caster import paths
from caster.report.records import RoundRecord, collapse, read_rows
from caster.run import policies

#: Fields that must agree across shards or the rows cannot pool. "Hard stop".
#:
#: `key_version` is here because the driver's own docstring already promised
#: it ("`shard.merge_audit` refuses to join across a disagreement") and nothing
#: delivered it: the ROW does not carry the key version — it carries the
#: CellSpec fields, which are identical across versions — so v3 and v4 rows
#: differ only in `episode_id` and pooled silently, which is the one failure
#: the version exists to prevent.
PROTOCOL_FIELDS = ("model", "temperature", "reasoning_effort", "key_version")
#: The dangerous half — none of these reaches the cache key (Invariant 13).
RUNTIME_FIELDS = ("context_tokens", "sandbox_timeout_sec", "interpreter",
                  "recursion_limit", "normalisation", "backend", "cpu_source")


def looks_like_alias(model: str) -> bool:
    """A model id with no version in it. A hosted model has to be pinned.

    Heuristic on purpose, and non-fatal on purpose: it cannot know every
    provider's naming, so it reports rather than blocks. `gpt-4o-mini` is an
    alias; `gpt-4o-mini-2024-07-18` and `qwen2.5-coder:7b` are pinned. A local
    tag counts as pinned because the digest travels in the runtime record.
    """
    if not model:
        return False
    if ":" in model or "@" in model or "/" in model:
        return False            # a local tag or a digest-qualified id
    return not re.search(r"-(\d{4}-\d{2}-\d{2}|\d{6,})$", model)


# ── ranges ───────────────────────────────────────────────────────────────────


def plan_ranges(first: int, last: int, shards: int) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping, covering `[first, last]` exactly.

    The remainder goes to the FIRST shards, one index each, never all onto the
    last: a fat tail is the shard still going when the others have finished,
    which is the one thing a fleet exists to avoid.
    """
    span = last - first + 1
    if span < 1:
        raise ValueError(f"empty range: first={first} last={last}")
    if shards < 1:
        raise ValueError("shards must be at least 1")
    if shards > span:
        raise ValueError(f"--shards {shards} over a range of {span} index(es) "
                         f"— at most one each")
    base, extra = divmod(span, shards)
    out: list[tuple[int, int]] = []
    cursor = first
    for i in range(shards):
        size = base + (1 if i < extra else 0)
        out.append((cursor, cursor + size - 1))
        cursor += size
    assert out[0][0] == first and out[-1][1] == last
    return out


# ── the manifest ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ShardEntry:
    index: int
    first: int
    last: int
    pid: int | None
    command: list[str]
    log: str
    started_at: float = dataclasses.field(default_factory=time.time)


def manifest_path(run_dir: pathlib.Path | None = None) -> pathlib.Path:
    return (run_dir or paths.LOGS_DIR) / "fleet" / "manifest.json"


def write_manifest(entries: Sequence[ShardEntry], universe: int,
                   run_dir: pathlib.Path | None = None) -> pathlib.Path:
    """One fleet at a time: refuses while a previous fleet is still alive."""
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        alive = [e for e in old.get("shards", []) if _alive(e.get("pid"))]
        if alive:
            raise RuntimeError(
                f"{len(alive)} shard(s) from a previous fleet are still running "
                f"(pids {[e['pid'] for e in alive]}). Stop or wait for them: a "
                f"second launch would interleave two manifests and neither would "
                f"be a record of anything."
            )
    doc = {"universe": universe, "at": time.time(),
           "shards": [dataclasses.asdict(e) for e in entries]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _alive(pid: Any) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


#: The phase whose rows ARE the evidence. It keeps the flat `shards/sNN`
#: layout, so every glob below — and every run collected before phases were
#: separated — still resolves.
EVIDENCE_PHASE = "evaluate"


def shard_dir(index: int | str, run_dir: pathlib.Path | None = None,
              phase: str | None = None) -> pathlib.Path:
    """Where one shard of one PHASE writes its three files.

    Phase 1 and phase 2 used to share `shards/sNN`, and nothing in the cell key
    says which phase produced a row — `CellSpec` has no such field, and
    `episode_id` is a hash of that key. Three things followed, and all three
    were silent:

    * `evaluate` read `propose`'s `cells.done` and skipped every cell phase 1
      had already walked, so the arm it shared was never evaluated;
    * `meta.json` was replaced by whichever phase ran last, destroying the
      other's protocol record (Invariant 13);
    * phase-1 rows, drawn at full budget against a cold store, merged beside
      phase-2 replay rows and tripped the soundness audit.

    `propose` now writes one level deeper, under `shards/propose/sNN`. The
    globs below are deliberately single-level, so phase-1 rows are NOT evidence
    and cannot reach a merge: filling the response cache is phase 1's job, and
    the floor arm is walked by `evaluate --force-full-budget` like every other
    arm.
    """
    root = (run_dir or paths.DATA_DIR) / "shards"
    if phase and phase != EVIDENCE_PHASE:
        root = root / phase
    return root / f"s{int(index):02d}"


def inputs(run_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every EVIDENCE shard's row file.

    Single-level by construction: smoke output cannot match it (Invariant 14)
    and neither can a non-evidence phase under `shards/<phase>/sNN`.
    """
    root = (run_dir or paths.DATA_DIR) / "shards"
    return sorted(root.glob("*/rounds.jsonl"))


def done_inputs(run_dir: pathlib.Path | None = None,
                phase: str | None = None) -> list[pathlib.Path]:
    """The done-index a phase may resume from — its own, never another's."""
    root = (run_dir or paths.DATA_DIR) / "shards"
    if phase and phase != EVIDENCE_PHASE:
        root = root / phase
    return sorted(root.glob("*/cells.done"))


def phase_inputs(phase: str, run_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """One NON-evidence phase's row files. Deliberately not part of `inputs()`.

    `caster audit` needs phase 1's — the patch sequence it scores is the one
    phase 1 drew, and before phase 2 runs that sequence exists nowhere else —
    so it asks for both. `merge` asks for `inputs()` only.
    """
    root = (run_dir or paths.DATA_DIR) / "shards"
    if phase and phase != EVIDENCE_PHASE:
        root = root / phase
    return sorted(root.glob("*/rounds.jsonl"))


def metas(run_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    root = (run_dir or paths.DATA_DIR) / "shards"
    return sorted(root.glob("*/meta.json"))


def meta_records(run_dir: pathlib.Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    """EVERY protocol record a shard directory holds, not just the last one.

    `meta.json` is rewritten by each launch into the same shard index, so a
    second launch erased the first's record — and the two audits that exist to
    catch instrument drift then had nothing to compare. The rows from the
    erased launch were still in `rounds.jsonl`, pooling with rows collected at
    a different sandbox timeout or a different served window, and the audit
    said `ok`. So every launch also APPENDS to `meta.jsonl`, and the audits
    read the history.
    """
    root = (run_dir or paths.DATA_DIR) / "shards"
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(root.glob("*/meta.json")):
        try:
            out.append((path.parent.name, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue
    # The current `meta.json` is also the LAST line of this shard's history,
    # so appending the history unfiltered reports the latest launch twice. It
    # never invented a disagreement — a record cannot differ from itself — but
    # it doubled one entry in every list the audits print, and an audit that
    # miscounts what it found is one the operator learns to re-read rather
    # than trust.
    seen = {json.dumps(b, sort_keys=True) for _, b in out}
    for path in sorted(root.glob("*/meta.jsonl")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                continue
            canon = json.dumps(blob, sort_keys=True)
            if canon in seen:
                continue
            seen.add(canon)
            out.append((f"{path.parent.name}#{i}", blob))
    return out


def meta_conflict(doc: dict[str, Any], out_dir: pathlib.Path) -> str:
    """Why this launch may not write into a shard directory that already has rows.

    Returns "" when it may. A launch that reuses a shard index under a
    different model, key version, sandbox timeout or served window is two
    instruments appending to one file, and `meta.json` being overwritten is
    what used to hide it until the numbers were already published.
    """
    rows = out_dir / "rounds.jsonl"
    prior = out_dir / "meta.json"
    if not (rows.is_file() and rows.stat().st_size and prior.is_file()):
        return ""
    try:
        old = json.loads(prior.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    bad: list[str] = []
    for section, fields in (("protocol", PROTOCOL_FIELDS), ("runtime", RUNTIME_FIELDS)):
        for field in fields:
            a = (old.get(section) or {}).get(field)
            b = (doc.get(section) or {}).get(field)
            if a != b:
                bad.append(f"{section}.{field}: {a!r} -> {b!r}")
    if not bad:
        return ""
    return (
        f"{out_dir} already holds rows collected under a different protocol:\n"
        "       " + "\n       ".join(bad) + "\n"
        f"       Appending would pool two instruments into one file, and "
        f"overwriting {prior.name} would erase the evidence that they differ "
        f"(Invariant 13). Use a fresh --run-dir, a different --shard index, or "
        f"move {out_dir} aside."
    )


# ── the six audits ───────────────────────────────────────────────────────────


@dataclasses.dataclass
class Audit:
    name: str
    fatal: bool
    ok: bool
    detail: str = ""
    items: list[Any] = dataclasses.field(default_factory=list)

    def render(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.fatal else "warn")
        head = f"[{mark}] {self.name}"
        return head if self.ok else f"{head}\n       {self.detail}"


def _disagreements(blobs: list[tuple[str, dict[str, Any]]], section: str,
                   fields: Sequence[str]) -> list[str]:
    out: list[str] = []
    for field in fields:
        seen: dict[Any, list[str]] = {}
        for name, blob in blobs:
            value = (blob.get(section) or {}).get(field)
            seen.setdefault(json.dumps(value, sort_keys=True), []).append(name)
        if len(seen) > 1:
            rendered = "; ".join(f"{v}={', '.join(sorted(n))}" for v, n in sorted(seen.items()))
            out.append(f"{section}.{field}: {rendered}")
    return out


def audit_protocol(meta_paths: Sequence[pathlib.Path],
                   shard_dirs: Sequence[pathlib.Path] = (),
                   records: Sequence[tuple[str, dict[str, Any]]] | None = None,
                   ) -> list[Audit]:
    """Audits 1–3: protocol, runtime, and the missing record.

    `records` is every protocol record the run holds — `meta_records()`, which
    reads the per-launch history as well as the current `meta.json`. Reading
    only the current one made a shard directory reused by a second launch
    exempt from both drift audits.
    """
    blobs: list[tuple[str, dict[str, Any]]] = []
    if records is not None:
        blobs = list(records)
    else:
        for p in meta_paths:
            try:
                blobs.append((p.parent.name, json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError):
                continue

    # A MIXED set is the dangerous case and the easy one to miss: some shards
    # recorded their protocol, some did not, and the un-recorded ones are
    # silently exempt from the two audits that catch instrument drift. Firing
    # only when NO shard wrote a meta would wave exactly that through.
    recorded = {name.split("#", 1)[0] for name, _ in blobs}
    unrecorded = sorted(d.name for d in shard_dirs if d.name not in recorded)
    out: list[Audit] = [Audit(
        "no protocol record", fatal=False, ok=not unrecorded and bool(blobs),
        detail=(f"{len(unrecorded)} shard(s) wrote rows but no meta.json — a "
                f"hand-run driver: {', '.join(unrecorded[:8])}. Reported, not "
                f"fatal, but nothing checks instrument drift for those rows."
                if unrecorded else
                "no shard wrote a meta.json — a hand-run driver. Reported, not "
                "fatal, but nothing checks instrument drift for these rows."),
        items=unrecorded)]
    if not blobs:
        return out
    proto = _disagreements(blobs, "protocol", PROTOCOL_FIELDS)
    out.append(Audit(
        "protocol disagreement", fatal=True, ok=not proto,
        detail="\n       ".join(proto) + (
            "\n       Rows collected under different protocols would never pool."
            if proto else ""),
        items=proto))
    aliases = sorted({
        str((b.get("protocol") or {}).get("model") or "")
        for _, b in blobs
        if looks_like_alias(str((b.get("protocol") or {}).get("model") or ""))
    } - {""})
    out.append(Audit(
        "model pinned by snapshot", fatal=False, ok=not aliases,
        detail=(f"{', '.join(aliases)} looks like a bare ALIAS, not an exact "
                f"snapshot id. A hosted model silently re-pointed is a different "
                f"instrument with nothing to say so — pin it."),
        items=aliases))
    # The control arm walks a DIFFERENT universe and is reported apart, so it
    # being live while the main grid replays is not drift between comparable
    # rows — and a permanent [warn] after every control pass would train the
    # operator to ignore the one audit that names a real soundness cause.
    collection = _disagreements(
        [(n, b) for n, b in blobs
         if b.get("collection") is not None
         and not (b.get("collection") or {}).get("control")],
        "collection", ("replay_only", "force_full_budget"))
    out.append(Audit(
        "collection mode", fatal=False, ok=not collection,
        detail="\n       ".join(collection) + (
            "\n       These shards' rows are in ONE namespace but were not "
            "collected the same way. A cell walked live against a cold "
            "verdict store is judged by a different oracle state than one "
            "replayed after `caster audit` — a probe that timed out under "
            "load refutes a patch the store later says passes. If one arm is "
            "the odd one out in SOUNDNESS, this is where to look first: "
            "phase 1 belongs in `caster propose`, which writes outside the "
            "evidence namespace." if collection else ""),
        items=collection))
    runtime = _disagreements(blobs, "runtime", RUNTIME_FIELDS)
    out.append(Audit(
        "runtime disagreement", fatal=True, ok=not runtime,
        detail="\n       ".join(runtime) + (
            "\n       None of these reaches the cache key, so these shards "
            "re-judged each other's draws against a different instrument "
            "(Invariant 13)." if runtime else ""),
        items=runtime))
    return out


def _listing(items: Sequence[str], cap: int) -> str:
    """Render at most `cap` items and SAY SO when there are more.

    A bare `items[:12]` reads as the whole finding. Arms are listed
    alphabetically, so a truncated GAPS block shows the first few arms and
    hides the rest — which reads as "four arms are short" when the truth is
    "every arm is short".
    """
    shown = "\n       ".join(items[:cap])
    if len(items) > cap:
        shown += (f"\n       ... and {len(items) - cap} more (the full list is "
                  f"in this audit's `items`; arms are listed alphabetically, so "
                  f"this is NOT the complete set of affected arms)")
    return shown


def audit_coverage(rows: Sequence[RoundRecord], universe: Sequence[str],
                   budget: int, arms: Sequence[str] = (),
                   seeds: Sequence[int] = (),
                   control: Sequence[str] = ()) -> list[Audit]:
    """Audits 4 and 6: GAPS / TRUNCATED, and FOREIGN.

    `arms` and `seeds` are what the shards were ASKED for (their `meta.json`
    protocol records). Deriving them from the rows instead — which is what
    this did — makes an arm that produced NOTHING invisible to the one audit
    whose job is to notice that: zero rows, zero keys in `by_cell`, zero
    mentions in the gap list, and a grid reported complete.

    `control` is the unfiltered control arm's faults, read off the frozen
    corpus. They are deliberately **not** in `pool_order` — that is what keeps
    a corpus-wide average from picking them up — and `caster evaluate
    --control` writes them into this same namespace. Without this argument
    FOREIGN is fatal on every one of them, so `caster merge` refused for good
    the moment the control arm ran, with "a shard cut from a different
    corpus" as the reason. A declared control fault is not a foreign one: it
    is counted, reported separately, and excluded from the gap arithmetic,
    which is what "reported separately, whatever it shows" requires.
    """
    known = set(universe)
    declared = set(control)
    seen_tasks = {r.task for r in rows}
    foreign = sorted(seen_tasks - known - declared) if known else []
    control_seen = sorted(seen_tasks & declared)

    by_cell: dict[tuple[str, int, str], list[RoundRecord]] = {}
    for r in rows:
        by_cell.setdefault((r.task, r.seed, r.policy), []).append(r)

    want_arms = sorted(set(arms) | {r.policy for r in rows})
    want_seeds = sorted(set(seeds) | {r.seed for r in rows})
    seen_arms = {r.policy for r in rows}
    gaps: list[str] = []
    silent: list[str] = []
    for arm_id in want_arms:
        if arm_id not in seen_arms:
            silent.append(arm_id)
        for seed in want_seeds:
            present = {t for (t, s, p) in by_cell if p == arm_id and s == seed}
            missing = sorted(known - present) if known else []
            if missing:
                gaps.append(f"{arm_id} seed {seed}: {len(missing)} task(s) missing "
                            f"(e.g. {', '.join(missing[:4])})")

    truncated: list[str] = []
    for (task, seed, arm_id), rs in sorted(by_cell.items()):
        accepted = any(r.accept for r in rs)
        rounds = len({r.round_index for r in rs})
        if not accepted and rounds < budget:
            truncated.append(f"{arm_id} {task} seed {seed}: {rounds}/{budget} rounds, "
                             f"no accept")

    return [
        Audit("FOREIGN", fatal=True, ok=not foreign,
              detail=f"{len(foreign)} task(s) not in the universe — a shard cut "
                     f"from a different corpus: {', '.join(foreign[:6])}",
              items=foreign),
        # ok=True ALWAYS: a declared control fault is intended, and a [warn]
        # beside eleven [ok]s reads as a defect. The count rides on the NAME,
        # because `Audit.render` only prints `detail` when the audit is not ok.
        Audit(f"CONTROL ARM ({len(control_seen)} fault(s), reported separately)",
              fatal=False, ok=True, detail="", items=control_seen),
        Audit("NO ROWS", fatal=False, ok=not silent,
              detail=f"{len(silent)} arm(s) were requested and produced no row "
                     f"at all: {', '.join(silent)}. An arm with zero rows "
                     f"cannot appear in GAPS — it has no cells to be missing "
                     f"from.",
              items=silent),
        Audit("GAPS", fatal=False, ok=not gaps,
              detail=f"{len(gaps)} arm/seed pair(s) short of the universe.\n       "
                     + _listing(gaps, 12), items=gaps),
        Audit("TRUNCATED", fatal=False, ok=not truncated,
              detail=f"{len(truncated)} cell(s) neither accepted nor reached the "
                     f"budget. Atomicity solves torn rows, not short episodes "
                     f".\n       " + _listing(truncated, 8),
              items=truncated),
    ]


def audit_conflicts(all_rows: Iterable[RoundRecord]) -> Audit:
    """Audit 5: the same (cell, round, candidate) collected twice, differently.

    **Do not pick a winner.** It is usually a wall-clock timeout firing on the
    slower machine; the fix is to re-run both under one protocol, not to choose.
    That is also why a timeout verdict is quarantined rather than silently
    reconciled.
    """
    seen: dict[tuple[str, int, int], RoundRecord] = {}
    conflicts: list[str] = []
    for r in all_rows:
        prior = seen.get(r.row_key)
        if prior is None:
            seen[r.row_key] = r
            continue
        if (prior.accept, prior.killed_by, prior.patch_sha) != (r.accept, r.killed_by, r.patch_sha):
            conflicts.append(
                f"{r.policy} {r.task} seed {r.seed} round {r.round_index}"
                f"#{r.candidate_index}: {prior.killed_by!r}/{prior.accept} vs "
                f"{r.killed_by!r}/{r.accept}"
            )
    return Audit("DISAGREEMENT", fatal=True, ok=not conflicts,
                 detail=f"{len(conflicts)} (cell, round) collected twice with "
                        f"different results. Fix and re-run both; do not pick a "
                        f"winner.\n       " + "\n       ".join(conflicts[:8]),
                 items=conflicts)


def audit_soundness(rows: Sequence[RoundRecord]) -> Audit:
    """Invariant 2 — on EVERY cell, not on a sample.

    Under CRN every Block A arm sees the identical patch sequence and ordering
    cannot change acceptance, so `success@B` and the first-accept round must be
    equal across all of them. "A gap is a soundness bug, not a finding."

    And when a violation is found it goes **in the result**, not only on stdout:
    "a warning in a multi-day shard log is a warning nobody reads, and the row
    is what the analysis sees."
    """
    block_a = set(policies.SOUNDNESS_EQUIVALENT)
    by_cell: dict[tuple[str, int], dict[str, tuple[bool, int | None]]] = {}
    for r in rows:
        if r.policy not in block_a:
            continue
        slot = by_cell.setdefault((r.task, r.seed), {})
        accepted, first = slot.get(r.policy, (False, None))
        if r.accept and (first is None or r.round_index < first):
            accepted, first = True, r.round_index
        slot[r.policy] = (accepted or r.accept, first)

    violations: list[str] = []
    for (task, seed), per_arm in sorted(by_cell.items()):
        outcomes = set(per_arm.values())
        if len(outcomes) > 1:
            rendered = ", ".join(f"{p}={v[0]}@{v[1]}" for p, v in sorted(per_arm.items()))
            violations.append(f"{task} seed {seed}: {rendered}")
    return Audit("SOUNDNESS (Invariant 2)", fatal=True, ok=not violations,
                 detail=f"{len(violations)} cell(s) where Block A arms disagree on "
                        f"accept or on the first-accept round. Ordering cannot "
                        f"change acceptance: this is a soundness bug, not a "
                        f"finding.\n       " + _listing(violations, 8) +
                        "\n       An arm that is the ODD ONE OUT on every "
                        "listed cell is usually not a scheduling bug: check "
                        "first that its rows were collected in the same phase, "
                        "against the same verdict store, as the arms it "
                        "disagrees with.",
                 items=violations)


def audit_invariant9(rows: Sequence[RoundRecord]) -> Audit:
    """Invariant 9, over the whole run rather than per scan.

    `(sandbox_runs − timeout_reruns) + verdict_hits == Σ test_calls`. It is an
    assertion, not a log line.
    """
    executed = sum(r.sandbox_runs - r.timeout_reruns for r in rows)
    hits = sum(r.verdict_hits for r in rows)
    calls = sum(r.test_calls for r in rows)
    ok = executed + hits == calls
    return Audit("INVARIANT 9 (no (patch, probe) run twice)", fatal=True, ok=ok,
                 detail=f"(sandbox_runs {executed + sum(r.timeout_reruns for r in rows)} "
                        f"− timeout_reruns {sum(r.timeout_reruns for r in rows)}) + "
                        f"verdict_hits {hits} = {executed + hits}, but Σ test_calls "
                        f"= {calls}.")


def merge_audit(
    *,
    run_dir: pathlib.Path | None = None,
    universe: Sequence[str] = (),
    budget: int = 20,
    control: Sequence[str] = (),
) -> tuple[list[Audit], list[RoundRecord]]:
    """All six audits plus the two invariants, then the collapsed rows.

    **Merge audit before analysis** (Invariant 12). A merged file is not
    evidence that the grid is complete.
    """
    row_files = inputs(run_dir)
    raw = list(read_rows(row_files))
    records = meta_records(run_dir)
    audits: list[Audit] = []
    audits.extend(audit_protocol(
        metas(run_dir),
        shard_dirs=sorted(p.parent for p in row_files),
        records=records,
    ))
    audits.append(audit_conflicts(raw))
    rows = collapse(raw)
    asked_arms: set[str] = set()
    asked_seeds: set[int] = set()
    for _, blob in records:
        proto = blob.get("protocol") or {}
        asked_arms.update(str(a) for a in (proto.get("policies") or ()))
        for sd in (proto.get("seeds") or ()):
            try:
                asked_seeds.add(int(sd))
            except (TypeError, ValueError):
                continue
    audits.extend(audit_coverage(rows, universe, budget,
                                 arms=sorted(asked_arms),
                                 seeds=sorted(asked_seeds),
                                 control=control))
    audits.append(audit_soundness(rows))
    audits.append(audit_invariant9(rows))
    return audits, rows


def fatal(audits: Sequence[Audit]) -> list[Audit]:
    return [a for a in audits if a.fatal and not a.ok]
