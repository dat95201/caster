"""The round record, and atomic cells.

One row per **candidate considered**, not per draw: a round that produces *n*
verbalized-sampling candidates emits *n* rows sharing one `cache_key`, told
apart by `candidate_index` (the schema consequence, decided in scope). A row
that never reached verification carries `charged=False`, so a `success@B` curve
reads only the rows that cost budget while a cost table still sums over all of
them.

Four things the obvious schema leaves out, each of which stops an analysis dead:

1. **`killed_by: None` was ambiguous three ways** — accepted, no patch built,
   and inconclusive scan all produced it. So `accept: bool` sits **beside**
   `oracle_error`, and the four terminal states are named and exhaustive:
   `accepted`, `refuted`, `proposal_failure`, `oracle_failure`. An inconclusive
   round is a spent round with no refutation, never an acceptance.
2. **Nothing resolved `patch_sha` to a patch.** "The text lives in the cache" is
   not true as written: the response cache holds the model's *reply*, not the
   extracted program, and `cache_key` is nullable on this record. So phase 1
   writes a content-addressed patch store beside the cache
   (`caster.exec.store.PatchStore`), and `patch_sha` resolves there.
3. **No `finish_reason`, no charged/uncharged bit.** "Ran out of output budget"
   and "answered in the wrong format" both reach the parser as a missing closing
   fence and neither is visible in the text. And a harness-failure round is
   charged against the budget deliberately — leaving it free lets a task whose
   proposer keeps truncating draw until a cap — so which rounds were charged has
   to be on the row, or `success@B` is a curve of an unknown B.
4. **One round, one patch** — closed by `candidate_index`.

`scan_order` is **not debug output**. It is what lets a different policy be
replayed offline against the same patches, and it is how the full kill matrix is
reconstructed. Do not drop it to save space.

`cell` carries `CellSpec.as_dict()` verbatim, which is what makes the rule
bind *consumers*: every analysis filter and the frozen summary read fields that
the key constructor produced, never a hand-maintained list.

## Atomic cells

**A cell is written atomically or not at all.** Rows are buffered in memory,
written in one `write()` + `flush()`, and only then is the cell recorded in a
separate `cells.done` index. Resume reads `cells.done`, **never** the row file.
A crash leaves at most one torn row, which the loader drops.

Two things the atomic rule must not lose, and both are here:

* **Extending a budget.** `budget` is not in the cell key, so a cell
  already done at B=12 is "done" when the grid is re-issued at B=20. So
  `cells.done` records `rounds_written` **and the budget it was produced
  under**, and resume re-walks a cell whose recorded budget is below the
  requested one, re-emitting the whole cell. The loader collapses on
  `(episode_id, round_index, candidate_index)`, last write wins, so the rewrite
  means what it looks like.
* **Truncated cells are still possible.** A cell can end early and legitimately
  — a circuit breaker, a context-overflow run, a draw cap — and is then written
  atomically and looks complete. `TRUNCATED` in the merge audit is what catches
  those. Atomicity solves torn rows, not short episodes.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import time
from typing import Any, Iterable, Iterator, Literal

Terminal = Literal["accepted", "refuted", "proposal_failure", "oracle_failure"]

SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class RoundRecord:
    # ── identity ────────────────────────────────────────────────────────────
    run_id: str
    episode_id: str
    task: str
    seed: int
    round_index: int
    policy: str
    candidate_index: int = 0          # 0 when the round emitted one patch
    cell: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ── proposal ────────────────────────────────────────────────────────────
    patch_sha: str | None = None      # resolves in the patch store, not the cache
    cache_key: str | None = None      # the join key to the call ledger
    proposal_error: str | None = None  # backend_unreachable|context_overflow|truncated_response
    finish_reason: str | None = None
    charged: bool = True              # did this row cost a unit of budget?
    probability: float | None = None  # VS self-report
    prob_stated: bool = False         # False ⇒ the model declined to verbalize
    history_restarts: int = 0         # chat arm only

    # ── verification ────────────────────────────────────────────────────────
    accept: bool = False
    n_cases: int = 0                  # usable pool size
    test_calls: int = 0
    cpu_s: float = 0.0                # getrusage(RUSAGE_CHILDREN), never wall clock
    killed_by: str | None = None
    scan_order: tuple[str, ...] = ()
    oracle_error: str | None = None
    sandbox_runs: int = 0
    verdict_hits: int = 0
    timeout_reruns: int = 0
    reference_runs: int = 0           # the own bucket; never folded into cpu_s

    # ── steering (null when steer_on is False) ──────────────────────────────
    facts_shown: tuple[str, ...] = ()
    k_facts: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    llm_sec: float | None = None
    usd: float = 0.0
    backend: str | None = None

    # ── provenance ──────────────────────────────────────────────────────────
    at: float = dataclasses.field(default_factory=time.time)
    schema: int = SCHEMA_VERSION

    @property
    def terminal(self) -> Terminal:
        """The four states, disjoint and exhaustive. See gap 1."""
        if self.proposal_error is not None:
            return "proposal_failure"
        if self.oracle_error is not None:
            return "oracle_failure"
        if self.accept:
            return "accepted"
        return "refuted"

    @property
    def row_key(self) -> tuple[str, int, int]:
        return (self.episode_id, self.round_index, self.candidate_index)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["scan_order"] = list(self.scan_order)
        d["facts_shown"] = list(self.facts_shown)
        d["terminal"] = self.terminal
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RoundRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        kw["scan_order"] = tuple(kw.get("scan_order") or ())
        kw["facts_shown"] = tuple(kw.get("facts_shown") or ())
        return cls(**kw)


# ── atomic cell writing ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, kw_only=True)
class CellDone:
    """One line of `cells.done`. Resume reads THIS, never the row file.

    Keyword-only on purpose: `rounds_written` (rows) and `rounds` (rounds) are
    adjacent integers that mean different things, and `budget` sits beside them.
    Positional construction is how one of those quietly becomes another.
    """

    cell_key: str
    episode_id: str
    task: str
    seed: int
    policy: str
    rounds_written: int     # ROWS. A VS cell writes k of them per round.
    #: ROUNDS — which is the unit `budget` is in, and therefore the one the
    #: top-up rule compares against. `rounds_written` is rows and would read 20
    #: for a 4-round cell at k=5.
    rounds: int = 0
    budget: int = 0         # the budget this cell was produced under

    accepted: bool = False
    truncated: bool = False  # ended early without accepting and without the budget
    at: float = dataclasses.field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class CellWriter:
    """Buffer a cell's rows, then write them in one call and index the cell.

    The order is load-bearing: rows first, `flush()`, *then* the done-index. A
    crash between the two re-walks the cell, which is free and correct; a crash
    the other way around would make resume skip a cell that was never written.
    """

    def __init__(self, rows_path: pathlib.Path, done_path: pathlib.Path) -> None:
        self.rows_path = pathlib.Path(rows_path)
        self.done_path = pathlib.Path(done_path)
        self.rows_path.parent.mkdir(parents=True, exist_ok=True)
        self.done_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[RoundRecord] = []

    def add(self, record: RoundRecord) -> None:
        self._buffer.append(record)

    def commit(self, done: CellDone) -> int:
        payload = "".join(
            json.dumps(r.as_dict(), separators=(",", ":")) + "\n" for r in self._buffer
        )
        n = len(self._buffer)
        self._buffer.clear()
        if payload:
            with self.rows_path.open("a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
        with self.done_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.replace(done, rounds_written=n).as_dict(),
                               separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return n

    def abandon(self) -> None:
        """Drop the buffer without writing. A cell is written whole or not at all."""
        self._buffer.clear()


def load_done(path: pathlib.Path) -> dict[str, CellDone]:
    """The resume index. Last entry per cell key wins (a re-walk rewrites it)."""
    out: dict[str, CellDone] = {}
    if not pathlib.Path(path).is_file():
        return out
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue   # a torn last line; the loader drops it
        try:
            out[d["cell_key"]] = CellDone(**d)
        except TypeError:
            continue
    return out


def needs_walk(done: dict[str, CellDone], cell_key: str, budget: int) -> bool:
    """A cell done at a LOWER budget must be re-walked and re-emitted.

    `budget` is deliberately not in the cell key, so without this check topping
    a grid up from B=12 to B=20 would silently draw nothing.
    """
    entry = done.get(cell_key)
    if entry is None:
        return True
    return entry.budget < budget


def read_rows(paths: Iterable[pathlib.Path]) -> Iterator[RoundRecord]:
    """Stream rows, dropping torn lines. Does NOT de-duplicate — see `collapse`."""
    for p in paths:
        p = pathlib.Path(p)
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue   # the one torn row an interrupted write can leave
                try:
                    yield RoundRecord.from_dict(d)
                except TypeError:
                    continue


def collapse(rows: Iterable[RoundRecord]) -> list[RoundRecord]:
    """Collapse on `(episode_id, round_index, candidate_index)`.

    Last write wins, so a re-walked cell **rewrites itself** rather than
    appending a duplicate — which is what makes topping a cell up from 12 rounds
    to 20 mean what it looks like.
    """
    by_key: dict[tuple[str, int, int], RoundRecord] = {}
    for r in rows:
        by_key[r.row_key] = r
    return sorted(by_key.values(), key=lambda r: (r.episode_id, r.round_index,
                                                  r.candidate_index))


def write_rows(rows: Iterable[RoundRecord], path: pathlib.Path) -> int:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.as_dict(), separators=(",", ":")) + "\n")
            n += 1
    tmp.replace(path)
    return n
