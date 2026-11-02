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
PROTOCOL_FIELDS = ("model", "temperature", "reasoning_effort")
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


def shard_dir(index: int | str, run_dir: pathlib.Path | None = None) -> pathlib.Path:
    return (run_dir or paths.DATA_DIR) / "shards" / f"s{int(index):02d}"


def inputs(run_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every shard's row file. Smoke output cannot match this glob (Invariant 14)."""
    root = (run_dir or paths.DATA_DIR) / "shards"
    return sorted(root.glob("*/rounds.jsonl"))


def done_inputs(run_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    root = (run_dir or paths.DATA_DIR) / "shards"
    return sorted(root.glob("*/cells.done"))


def metas(run_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    root = (run_dir or paths.DATA_DIR) / "shards"
    return sorted(root.glob("*/meta.json"))


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
                   shard_dirs: Sequence[pathlib.Path] = ()) -> list[Audit]:
    """Audits 1–3: protocol, runtime, and the missing record."""
    blobs: list[tuple[str, dict[str, Any]]] = []
    for p in meta_paths:
        try:
            blobs.append((p.parent.name, json.loads(p.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue

    # A MIXED set is the dangerous case and the easy one to miss: some shards
    # recorded their protocol, some did not, and the un-recorded ones are
    # silently exempt from the two audits that catch instrument drift. Firing
    # only when NO shard wrote a meta would wave exactly that through.
    recorded = {name for name, _ in blobs}
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
    runtime = _disagreements(blobs, "runtime", RUNTIME_FIELDS)
    out.append(Audit(
        "runtime disagreement", fatal=True, ok=not runtime,
        detail="\n       ".join(runtime) + (
            "\n       None of these reaches the cache key, so these shards "
            "re-judged each other's draws against a different instrument "
            "(Invariant 13)." if runtime else ""),
        items=runtime))
    return out


def audit_coverage(rows: Sequence[RoundRecord], universe: Sequence[str],
                   budget: int) -> list[Audit]:
    """Audits 4 and 6: GAPS / TRUNCATED, and FOREIGN."""
    known = set(universe)
    seen_tasks = {r.task for r in rows}
    foreign = sorted(seen_tasks - known) if known else []

    by_cell: dict[tuple[str, int, str], list[RoundRecord]] = {}
    for r in rows:
        by_cell.setdefault((r.task, r.seed, r.policy), []).append(r)

    arms = sorted({r.policy for r in rows})
    seeds = sorted({r.seed for r in rows})
    gaps: list[str] = []
    for arm_id in arms:
        for seed in seeds:
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
        Audit("GAPS", fatal=False, ok=not gaps,
              detail="\n       ".join(gaps[:12]), items=gaps),
        Audit("TRUNCATED", fatal=False, ok=not truncated,
              detail=f"{len(truncated)} cell(s) neither accepted nor reached the "
                     f"budget. Atomicity solves torn rows, not short episodes "
                     f".\n       " + "\n       ".join(truncated[:8]),
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
                        f"finding.\n       " + "\n       ".join(violations[:8]),
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
) -> tuple[list[Audit], list[RoundRecord]]:
    """All six audits plus the two invariants, then the collapsed rows.

    **Merge audit before analysis** (Invariant 12). A merged file is not
    evidence that the grid is complete.
    """
    row_files = inputs(run_dir)
    raw = list(read_rows(row_files))
    audits: list[Audit] = []
    audits.extend(audit_protocol(
        metas(run_dir),
        shard_dirs=sorted(p.parent for p in row_files),
    ))
    audits.append(audit_conflicts(raw))
    rows = collapse(raw)
    audits.extend(audit_coverage(rows, universe, budget))
    audits.append(audit_soundness(rows))
    audits.append(audit_invariant9(rows))
    return audits, rows


def fatal(audits: Sequence[Audit]) -> list[Audit]:
    return [a for a in audits if a.fatal and not a.ok]
