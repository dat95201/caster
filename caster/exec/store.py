"""Never run the same (patch, case) twice. The verdict store and the patch store.

This is a correctness-shaped performance rule: the same `(patch, case)` under the
same sandbox protocol has **one** answer, so computing it twice is pure waste,
and computing it twice with *different* answers is a bug that must surface.

    key    = (patch_sha256, case_name, protocol_sha)
    value  = { verdict: pass|wrong|crash|timeout,
               output_sha256, cpu_s, measured_at, host, source, concurrency }

Three scopes of reuse, in increasing size:

| scope             | what it saves                                   |
|-------------------|-------------------------------------------------|
| within a round    | a probe run twice against one patch             |
| within an episode | a case the same candidate already faced (VS)    |
| **across arms**   | **most of the grid**                            |

The third is the big one. Under CRN every Block A arm sees the same patch
sequence, so after the first arm the rest are near-free: they differ only in
*which* cases they choose to run, and almost every choice has already been
answered. Combined with `caster audit` (the dedicated full-scan pass), every
subsequent policy on that task is a pure replay with zero sandbox runs.

**The store is shared; the ledger is not.** Warming a ledger's (n,k) counts from
the store is forbidden — see `caster.exec.ledger.seed_from`. Reusing *verdicts*
is sound because a verdict is a fact about a (patch, case); reusing *statistics*
leaks the outcome of one cell into the treatment of another.

Two decisions this module implements, both recorded in `docs/DECISIONS.md`:

**Timeouts are quarantined** (fix 2). `protocol_sha` covers the
interpreter, the timeout value, the recursion ceiling and the normalisation
rules. It *cannot* cover machine speed or concurrent load, which are the actual
inputs to a wall-clock timeout — and the store is cross-run and cross-host by
design, so a verdict measured on a fast idle box would be replayed against a
slow loaded one. So a `timeout` verdict is written, but **never served**:
`get()` returns None for it, `--from-cache-only` cannot be satisfied by it, a
flip is appended to `jitter.jsonl`, and only a *stable* disagreement across
`REPEATS_BEFORE_HALT` re-runs on one host halts the run.

Invariant 9's arithmetic has to name those repeats, or it stops being an
identity. The spec writes it `sandbox_runs + verdict_hits == Σ scan lengths`;
the quarantine makes the exact form

    (sandbox_runs − timeout_reruns) + verdict_hits == Σ test_calls

because a repeat is real CPU but not an extra *verdict*. `ScanResult` asserts it
per scan and `run.shard.audit_invariant9` over the whole run. `docs/DECISIONS.md`
`docs/DECISIONS.md` records the amendment.

**The cost table is not written by the arms**. The first arm to touch a
(patch, case) would otherwise write the cost every later arm inherits, making
the shared table a function of which arm ran first and of the concurrency it ran
under. So `caster audit` populates it in a dedicated pass at a declared
concurrency, before any arm runs, and `put()` refuses to overwrite an existing
entry's `cpu_s` from an arm scan. The table is a stamped artifact with its own
digest, and the arm comparison is a replay against a fixed instrument.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import socket
import time
from typing import Any, Iterable, Literal

from caster import paths
from caster.errors import CacheMiss  # noqa: F401 - re-exported; see caster/errors.py

Verdict = Literal["pass", "wrong", "crash", "timeout"]

#: A `timeout` verdict is re-executed rather than served. A disagreement is
#: only a halt condition when it is stable across this many re-runs on one host;
#: a flip below it is jitter and goes to `jitter.jsonl`.
REPEATS_BEFORE_HALT = 3


class VerdictDivergence(RuntimeError):
    """A fresh execution disagreed with a stored verdict. The run stops.

    It means the protocol hash is incomplete (a hidden dependency on the
    environment) or the program is nondeterministic. Both are findings. The key,
    both verdicts and both hosts are in the message; nothing is overwritten.

    Unqualified this would halt the grid on a known, measured, non-zero base
    rate — so it is raised only for non-`timeout` verdicts, and for a
    `timeout` only when the disagreement is stable (see REPEATS_BEFORE_HALT).
    """


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protocol_sha(fields: dict[str, Any]) -> str:
    """Digest of everything that can change a verdict without changing the patch.

    A protocol change invalidates the store by construction instead of silently
    mixing instruments. `caster.exec.sandbox.protocol_fields` builds the input.
    """
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


#: How much of a refuted candidate's stdout the store keeps. The contract block
#: renders "your last attempt printed `…`" from it, and without it a replay
#: against a warm store would build a DIFFERENT prompt from the run that filled
#: the store — a different draw, with nothing in any key to say so. 240 is the
#: hard quoting cap on contest data in a prompt (`steer.render`), so keeping
#: more could never be rendered anyway.
OUTPUT_HEAD_CHARS = 240


@dataclasses.dataclass(frozen=True)
class StoredVerdict:
    verdict: Verdict
    output_sha256: str
    cpu_s: float
    measured_at: float
    host: str
    source: str = "scan"          # "audit" | "scan" | "remeasure"
    concurrency: int | None = None
    #: The head of what the candidate printed, and how it failed. Steering-only;
    #: see OUTPUT_HEAD_CHARS.
    output_head: str | None = None
    output_len: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StoredVerdict":
        return cls(
            verdict=d["verdict"],
            output_sha256=d.get("output_sha256", ""),
            cpu_s=float(d.get("cpu_s", 0.0)),
            measured_at=float(d.get("measured_at", 0.0)),
            host=d.get("host", ""),
            source=d.get("source", "scan"),
            concurrency=d.get("concurrency"),
            output_head=d.get("output_head"),
            output_len=d.get("output_len"),
            error_type=d.get("error_type"),
            error_message=d.get("error_message"),
        )


class PatchStore:
    """Content-addressed patch text. the second gap.

    "The text lives in the cache" is not true as written: the response cache is
    keyed by `cache_key` and holds the model's *reply*, not the extracted
    program, and `cache_key` is nullable on a RoundRecord (a round can fail
    before any call). The verdict store is keyed by `patch_sha256`, so phase 2
    and `caster audit` both need sha → source.

    **Who garbage-collects it:** nobody, automatically. `caster merge --strict`
    reports orphans (a sha in no surviving record) and `caster report` prints
    the store's size; deleting is an explicit operator action, because a patch
    deleted while a record still references it makes that record unreplayable.
    """

    def __init__(self, root: pathlib.Path | None = None) -> None:
        self.root = pathlib.Path(root or paths.PATCH_DIR)

    def _path(self, sha: str) -> pathlib.Path:
        return self.root / sha[:2] / f"{sha}.py"

    def put(self, source: str) -> str:
        sha = sha256_text(source)
        path = self._path(sha)
        if path.exists():
            return sha
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, source)
        return sha

    def get(self, sha: str) -> str | None:
        path = self._path(sha)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def require(self, sha: str) -> str:
        source = self.get(sha)
        if source is None:
            raise CacheMiss(
                f"patch {sha[:12]} is not in the patch store at {self.root}. "
                f"Phase 2 cannot replay a round whose patch text is gone — "
                f"re-run phase 1 for that cell, or install the artifacts."
            )
        return source

    def shas(self) -> Iterable[str]:
        for p in sorted(self.root.glob("*/*.py")):
            yield p.stem


def _atomic_write(path: pathlib.Path, text: str) -> None:
    """Write via a same-directory temp + os.replace.

    Two shards running against one store will both miss the same key and both
    write it. `os.replace` is atomic on POSIX, so a reader can only ever see the
    whole file or no file at all — never the half-written JSON a plain
    `write_text` can expose.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class VerdictStore:
    """Cross-run, cross-host store of (patch, case) verdicts.

    On-disk layout — one file per (protocol, patch), holding every case:

        verdicts/<protocol_sha>/<patch_sha[:2]>/<patch_sha>.json
        { "<case name>": {verdict, output_sha256, cpu_s, measured_at, host, ...} }

    One file per patch rather than per (patch, case) because the dominant access
    pattern is the full scan — every case of one patch — and because a pool of
    1,244 tasks × ~42 cases × tens of thousands of patches is tens of millions
    of inodes otherwise.

    Concurrency: two processes that both learn a verdict for the same patch
    merge-then-replace, so the loser's entries can be lost. That is safe and not
    silent corruption — the invariant is that the two agreed — and the lost
    entries are simply recomputed. It is also why `caster audit` runs as the
    dedicated pass: one writer, one concurrency, one cost table.
    """

    def __init__(self, protocol: dict[str, Any], root: pathlib.Path | None = None) -> None:
        self.protocol = dict(protocol)
        self.protocol_sha = protocol_sha(self.protocol)
        self.root = pathlib.Path(root or paths.VERDICT_DIR) / self.protocol_sha
        self.host = socket.gethostname()
        # Invariant 9's two counters. They are reported separately and their sum
        # is asserted against the scan length.
        self.verdict_hits = 0
        self.sandbox_runs = 0
        self.timeout_reexecutions = 0
        self._mem: dict[str, dict[str, StoredVerdict]] = {}

    # -- paths ---------------------------------------------------------------

    def _path(self, patch_sha: str) -> pathlib.Path:
        return self.root / patch_sha[:2] / f"{patch_sha}.json"

    def _load(self, patch_sha: str) -> dict[str, StoredVerdict]:
        cached = self._mem.get(patch_sha)
        if cached is not None:
            return cached
        path = self._path(patch_sha)
        out: dict[str, StoredVerdict] = {}
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            for name, blob in raw.items():
                try:
                    out[name] = StoredVerdict.from_dict(blob)
                except (KeyError, TypeError, ValueError):
                    continue
        self._mem[patch_sha] = out
        return out

    def _flush(self, patch_sha: str) -> None:
        """Merge this process's view with what is on disk, then replace."""
        mine = self._mem.get(patch_sha) or {}
        path = self._path(patch_sha)
        merged: dict[str, Any] = {}
        if path.is_file():
            try:
                merged = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                merged = {}
        for name, sv in mine.items():
            existing = merged.get(name)
            # Never let an arm scan overwrite the audit pass's cost —
            # unless the verdict itself changed, in which case the audited cost
            # belongs to a different outcome. See `put`.
            if (existing and existing.get("source") == "audit"
                    and sv.source == "scan" and existing.get("verdict") == sv.verdict):
                continue
            merged[name] = sv.as_dict()
        _atomic_write(path, json.dumps(merged, sort_keys=True, separators=(",", ":")))

    # -- reads ---------------------------------------------------------------

    def peek(self, patch_sha: str, case_name: str) -> StoredVerdict | None:
        """The raw entry, timeouts included. For audits and the cost table only."""
        return self._load(patch_sha).get(case_name)

    def get(self, patch_sha: str, case_name: str) -> StoredVerdict | None:
        """A *servable* entry, or None.

        A `timeout` entry is never servable — see the module docstring. This is
        the only read a scan may use, and it increments `verdict_hits` so
        Invariant 9's arithmetic stays honest.
        """
        sv = self._load(patch_sha).get(case_name)
        if sv is None:
            return None
        if sv.verdict == "timeout":
            return None
        self.verdict_hits += 1
        return sv

    def has_servable(self, patch_sha: str, case_name: str) -> bool:
        """Like `get` but without counting a hit. For `--dry-run` planning."""
        sv = self._load(patch_sha).get(case_name)
        return sv is not None and sv.verdict != "timeout"

    # -- writes --------------------------------------------------------------

    def put(
        self,
        patch_sha: str,
        case_name: str,
        *,
        verdict: Verdict,
        output_sha256: str,
        cpu_s: float,
        source: str = "scan",
        concurrency: int | None = None,
        output_head: str | None = None,
        output_len: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        flush: bool = True,
    ) -> StoredVerdict:
        """Record a freshly measured verdict, checking for divergence on the way in.

        Raises `VerdictDivergence` when a non-timeout stored verdict disagrees.
        A timeout disagreement is jitter until it is stable — the caller
        (`caster.exec.verify`) owns the repeat loop and calls
        `note_timeout_flip` for the unstable case.
        """
        existing = self._load(patch_sha).get(case_name)
        if existing is not None and existing.verdict != "timeout" and verdict != "timeout":
            if existing.verdict != verdict or existing.output_sha256 != output_sha256:
                raise VerdictDivergence(
                    f"(patch {patch_sha[:12]}, case {case_name}, protocol "
                    f"{self.protocol_sha}) was {existing.verdict}"
                    f"/{existing.output_sha256[:12]} on host {existing.host!r} and is "
                    f"{verdict}/{output_sha256[:12]} on host {self.host!r}. "
                    f"The protocol hash is incomplete or the program is "
                    f"nondeterministic — both are findings. Nothing was overwritten."
                )
        sv = StoredVerdict(
            verdict=verdict,
            output_sha256=output_sha256,
            cpu_s=float(cpu_s),
            measured_at=time.time(),
            host=self.host,
            source=source,
            concurrency=concurrency,
            output_head=output_head,
            output_len=output_len,
            error_type=error_type,
            error_message=error_message,
        )
        # An arm scan does not get to restate an audited cost — but ONLY
        # when it is restating the same verdict. An audited `timeout` whose
        # re-execution comes back `pass` is a probe the audit mismeasured: its
        # 10-second wall-clock kill is not the cost of the millisecond run that
        # actually happened, and inheriting it would poison the frozen cost
        # table on exactly the probes most likely to flip. The fresh one wins
        # and stays `source="scan"`, so `cost_digest()` is unchanged by it.
        if (existing is not None and existing.source == "audit" and source == "scan"
                and existing.verdict == verdict):
            sv = dataclasses.replace(sv, cpu_s=existing.cpu_s, source="audit",
                                     concurrency=existing.concurrency,
                                     measured_at=existing.measured_at,
                                     host=existing.host)
        self._load(patch_sha)[case_name] = sv
        if flush:
            self._flush(patch_sha)
        return sv

    def flush(self, patch_sha: str | None = None) -> None:
        for sha in ([patch_sha] if patch_sha else list(self._mem)):
            self._flush(sha)

    def note_timeout_flip(self, patch_sha: str, case_name: str,
                          verdicts: list[str], host: str | None = None,
                          *, stable: bool = False) -> None:
        """A `timeout` key that did not reproduce. Logged, counted, not fatal.

        The threats table is built from this file. The run continues, because
        the measured base rate is 0.09% and halting on it would stop the grid
        for a known property of a wall-clock budget.

        `stable` marks the case where EVERY repeat disagreed with the store.
        It used to halt the run. It no longer does (DECISIONS 27): the repeats
        are back to back and therefore sample one machine state, so on a loaded
        box "stable" is a statement about the load and not about the protocol.
        Keeping the two apart in this file is what turns the halt into the
        measurement DECISIONS 2 asked for -- the flip rate at the grid's own
        target concurrency, which is the number that was missing.
        """
        paths.ensure(paths.DATA_DIR)
        line = json.dumps({
            "patch_sha": patch_sha, "case": case_name,
            "protocol_sha": self.protocol_sha,
            "verdicts": verdicts, "host": host or self.host, "at": time.time(),
            "stable": bool(stable),
        })
        with (paths.DATA_DIR / "jitter.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -- the cost table ------------------------------------------------------

    def cost_digest(self) -> str:
        """Digest of every audited cost under this protocol.

        Stamped into the run manifest so a CPU figure can name the table it was
        reconstructed from. Arms-written costs are excluded by construction:
        only `source == "audit"` entries enter.
        """
        h = hashlib.sha256()
        h.update(self.protocol_sha.encode())
        for path in sorted(self.root.glob("*/*.json")):
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for case, entry in sorted(blob.items()):
                if entry.get("source") != "audit":
                    continue
                if entry.get("verdict") == "timeout":
                    # A timeout's `cpu_s` is the wall-clock kill, not a
                    # measurement of the work, and the verdict is quarantined
                    # anyway. Letting it into the stamped cost table would make
                    # the digest move the moment the probe was re-executed and
                    # came back conclusive — which is exactly when it should
                    # NOT move.
                    continue
                h.update(f"{path.stem}|{case}|{entry.get('cpu_s')}".encode())
        return h.hexdigest()[:16]

    def stats(self) -> dict[str, Any]:
        return {
            "protocol_sha": self.protocol_sha,
            "verdict_hits": self.verdict_hits,
            "sandbox_runs": self.sandbox_runs,
            "timeout_reexecutions": self.timeout_reexecutions,
            "root": str(self.root),
        }
