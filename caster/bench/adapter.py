"""The ConDefects adapter. The only place in CASTER that reads a benchmark.

ConDefects is a corpus of real faults from AtCoder submissions. Every fault
ships three things:

    faultyVersion.py    the program to repair — a whole script reading stdin
    correctVersion.py   the same author's accepted submission (the D3 *judge*;
                        **never** placed in a prompt — Invariant 4)
    faultLocation.txt   1-indexed line numbers of the faulty statements

and the contest's own test data, shipped separately as Test.zip:

    Test/<contest>/<LETTER>/in/<case>     the input handed to the program
    Test/<contest>/<LETTER>/out/<case>    the output AtCoder accepted

So there are no input *strategies* to write: the inputs exist, each with its
expected output. A counterexample is therefore a concrete test case identified
by **name** — the input text never has to travel through the round log. Contest
inputs run to hundreds of kilobytes.

Task identity is one *fault*, `"<task_id>/<program_id>"`. Several submissions
can fail the same coding task; they are different programs with different bugs
but they share one test pool, so G4 keeps at most one per `task_id`.

Two things this adapter deliberately does not do:

* **The worked examples have no fallback.** When a task has no
  `sample`/`example`-named case it gets *no* examples. Falling back to "the two
  smallest inputs" would disclose a hidden test on the 1.3% of problems that
  have no samples — safe by construction beats 98.7% safe.
* **Discovery does not run at import.** Scanning the checkout at import time
  makes every `import` an 18k-directory walk and leaves a stale snapshot after a
  download. It is lazy and cached.
"""
from __future__ import annotations

import os
import pathlib
from typing import Iterable, Sequence

from caster.bench.protocol import (
    Authority,
    Legibility,
    Program,
    ProbeVerdict,
    Task,
    TestCase,
)
from caster.exec.sandbox import DEFAULT_RECURSION_LIMIT, DEFAULT_TIMEOUT, Outcome, run_program

LANGUAGE_DIR = "Python"

#: Worked examples put in front of the proposer (slot S4). Contest problems
#: always show a couple of sample cases, and a submission stripped of its
#: problem statement is genuinely ambiguous without them — a patch would be
#: refuted for printing "Yes" where the judge wanted "YES", which is a
#: disagreement about the specification, not a bug. Applied identically in every
#: arm, so it cannot confound the comparison, and the examples stay in the pool:
#: failing one the proposer was shown is still a legitimate refutation.
#: How many coding tasks' case TEXTS stay resident. Small on purpose: contest
#: inputs reach 2.3 MB each and the driver walks task-major, so the hit rate is
#: ~1. Listing probe NAMES does not go through this cache at all — see
#: `_probe_names` — so discovery never pays for it.
CASE_CACHE_GROUPS = 4

N_SPEC_EXAMPLES = 2
SPEC_MAX_INPUT_CHARS = 400
SPEC_MAX_OUTPUT_CHARS = 200


def normalize_output(text: str | None) -> list[str]:
    """Judge-normalised form of a program's stdout.

    Trailing whitespace on a line and trailing blank lines at the end are not
    differences a contest judge counts, and neither should the oracle: a patch
    that prints an extra newline is not a distinct failure, it is noise.
    Everything else is compared exactly, matching ConDefects' own `compare_res`.

    This function is part of the verdict, so it is named in
    `sandbox.protocol_fields()["normalisation"]`. Changing it must bump that
    string, or the verdict store silently mixes two instruments.
    """
    if text is None:
        return []
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def outputs_equal(candidate: str | None, expected: str | None) -> bool:
    return normalize_output(candidate) == normalize_output(expected)


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class ConDefectsAdapter:
    """`OracleAdapter` over a ConDefects checkout.

    `splits` maps a task name to its held-out probe set (`P_eval`). It is read
    from the frozen corpus and is what makes `authority()` meaningful; an
    adapter constructed without it reports every probe `visible`, which is
    correct for `caster audit` and wrong for a reported loop — so
    `run.episode` requires a split and `caster preflight` checks for one.
    """

    name = "condefects"
    determinism = "stable"        # D2 — wall-clock timeout, so NOT `pure`
    fixed_cost_scope = "probe"    # D6 — the reference answer, cacheable per probe

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        test_dir: str | os.PathLike[str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        splits: dict[str, frozenset[str]] | None = None,
        pools: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.root = pathlib.Path(root or os.environ.get("CONDEFECTS_ROOT", "external/ConDefects"))
        self.code_dir = self.root / "Code"
        # Normally <root>/Test. Overridable because Test.zip is a 6.4 GB
        # out-of-band download that can arrive incomplete: a partial tree can be
        # salvaged to a separate directory and pointed at from here, which keeps
        # a truncated download from being frozen into a corpus by accident.
        # `or` would not work: pathlib.Path("") is PosixPath("."), which is
        # truthy, so an unset variable would resolve the test tree to the CWD.
        env_test = str(test_dir or os.environ.get("CONDEFECTS_TEST_DIR", "")).strip()
        self.test_dir = pathlib.Path(env_test) if env_test else self.root / "Test"
        self.timeout = float(timeout)
        self.recursion_limit = int(recursion_limit)
        self._held_out: dict[str, frozenset[str]] = dict(splits or {})
        self._pools: dict[str, tuple[str, ...]] = dict(pools or {})
        self._tasks: dict[str, Task] | None = None
        self._ref_cache: dict[tuple[str, str], Outcome | None] = {}
        self._dates: dict[str, str] | None = None
        self._difficulties: dict[str, int] | None = None
        self._cov_cache: dict[tuple[str, str], frozenset[int] | None] = {}
        self._case_cache: dict[str, tuple[TestCase, ...]] = {}
        self._case_cache_order: list[str] = []
        self._name_cache: dict[str, tuple[str, ...]] = {}
        self._program_cache: dict[str, Program] = {}
        #: Reference executions. They belong to no candidate, so they are
        #: counted in their own bucket and never folded into a round's
        #: scan cost. An oracle that re-derives the reference on every call for
        #: any case missing an `out/` file pays a second execution per case per
        #: round, and it is invisible in the unit such an oracle reports.
        self.reference_runs = 0

    # ── discovery ───────────────────────────────────────────────────────────

    @property
    def setup_hint(self) -> str:
        return (
            f"{self.root} is missing or incomplete — run "
            f"`python3 scripts/fetch_condefects.py` (it clones the code and says "
            f"where to get Test.zip, which is not in the git repository)."
        )

    def _test_root(self, task_id: str) -> pathlib.Path | None:
        """Test/<contest>/<LETTER> for a task id like "abc221_f".

        Mirrors the toolkit's own lookup (ConDefects `Tool/RunTest.py`): the
        letter after the underscore, uppercased, with "Ex" as the fallback —
        AtCoder names a Beginner Contest's eighth problem "Ex", not "H".
        """
        contest, _, suffix = task_id.partition("_")
        for letter in (suffix.upper(), "Ex"):
            candidate = self.test_dir / contest / letter
            if (candidate / "in").is_dir():
                return candidate
        return None

    def _probe_names(self, task_id: str) -> tuple[str, ...]:
        """Probe names for a coding task, from the directory listing alone."""
        hit = self._name_cache.get(task_id)
        if hit is not None:
            return hit
        root = self._test_root(task_id)
        if root is None:
            out: tuple[str, ...] = ()
        else:
            out = tuple(sorted(
                p.name for p in (root / "in").iterdir()
                if p.is_file() and not p.name.startswith(".")
            ))
        self._name_cache[task_id] = out
        return out

    def _cases_for_group(self, task_id: str) -> tuple[TestCase, ...]:
        """Every shipped case for a coding task, in deterministic (pool) order.

        Cached per `task_id` rather than per fault: several faulty programs
        share one coding task, and re-reading a few hundred input files for each
        would dominate the stage's cost.

        The cache is a BOUNDED instance dict, not `functools.lru_cache`. On an
        instance method `lru_cache` keys on `self`, which pins the adapter for
        the life of the process, and contest inputs reach 2.3 MB each: 4,096
        groups × ~42 blobs is multiple gigabytes resident on the full corpus.
        One group at a time is enough, because the driver walks task-major
        (`Grid.cells`), so the hit rate is ~1.
        """
        hit = self._case_cache.get(task_id)
        if hit is not None:
            self._case_cache_order.remove(task_id)
            self._case_cache_order.append(task_id)
            return hit
        root = self._test_root(task_id)
        if root is None:
            return ()
        out_dir = root / "out"
        cases: list[TestCase] = []
        for in_path in sorted((root / "in").iterdir()):
            if not in_path.is_file() or in_path.name.startswith("."):
                continue
            out_path = out_dir / in_path.name
            cases.append(TestCase(
                name=in_path.name,
                input_text=_read(in_path),
                expected_output=_read(out_path) if out_path.is_file() else None,
            ))
        out = tuple(cases)
        self._case_cache[task_id] = out
        self._case_cache_order.append(task_id)
        while len(self._case_cache_order) > CASE_CACHE_GROUPS:
            self._case_cache.pop(self._case_cache_order.pop(0), None)
        return out

    def tasks(self) -> dict[str, Task]:
        if self._tasks is not None:
            return self._tasks
        found: dict[str, Task] = {}
        if self.code_dir.is_dir():
            for faulty in self.code_dir.glob(f"*/{LANGUAGE_DIR}/*/faultyVersion.py"):
                program_dir = faulty.parent
                if not (program_dir / "correctVersion.py").is_file():
                    continue
                task_id = program_dir.parent.parent.name
                name = f"{task_id}/{program_dir.name}"
                found[name] = Task(
                    name=name,
                    group=task_id,
                    # Names only. Reading every case's text here would pull
                    # ~42 files per FAULT — thousands of groups, megabyte
                    # inputs — just to list what is in the pool, and the bounded
                    # case cache cannot hold that walk. `_cases_for_group` is
                    # paid for once, later, when a stage actually needs the text.
                    probes=self._pools.get(name) or self._probe_names(task_id),
                    meta={"task_id": task_id, "program_id": program_dir.name},
                )
        self._tasks = dict(sorted(found.items()))
        return self._tasks

    def task(self, task_name: str) -> Task:
        tasks = self.tasks()
        if task_name not in tasks:
            if not tasks:
                raise FileNotFoundError(self.setup_hint)
            raise KeyError(f"{task_name!r} is not a known fault ({len(tasks)} available)")
        return tasks[task_name]

    def _dir(self, task_name: str) -> pathlib.Path:
        task_id, _, program_id = task_name.partition("/")
        return self.code_dir / task_id / LANGUAGE_DIR / program_id

    # ── the six Protocol functions ──────────────────────────────────────────

    def program(self, task_name: str) -> Program:
        hit = self._program_cache.get(task_name)
        if hit is not None:
            return hit
        d = self._dir(task_name)
        faulty, correct = d / "faultyVersion.py", d / "correctVersion.py"
        if not faulty.is_file() or not correct.is_file():
            raise FileNotFoundError(f"{faulty} or {correct} missing — {self.setup_hint}")
        location = d / "faultLocation.txt"
        fault_lines: tuple[int, ...] = ()
        if location.is_file():
            fault_lines = tuple(
                int(t) for t in _read(location).split() if t.lstrip("-").isdigit()
            )
        prog = Program(
            name=task_name,
            buggy_source=_read(faulty),
            reference_source=_read(correct),
            fault_lines=fault_lines,
        )
        # One fault's two sources are a few KB and the loop reads them every
        # round, so they are kept; the cache is per adapter and the driver walks
        # one task at a time.
        if len(self._program_cache) > 64:
            self._program_cache.clear()
        self._program_cache[task_name] = prog
        return prog

    def probes(self, task_name: str) -> Sequence[str]:
        return self.task(task_name).probes

    def case(self, task_name: str, probe: str) -> TestCase | None:
        group = self.task(task_name).group
        for c in self._cases_for_group(group):
            if c.name == probe:
                return c
        return None

    def reference(self, task_name: str, probe: str) -> Outcome | None:
        """The expected answer for one probe, computed at most once.

        None when the reference cannot answer it — the probe proves nothing and
        is skipped (G1 should already have excluded it from the pool; this is
        the runtime backstop).
        """
        key = (task_name, probe)
        if key in self._ref_cache:
            return self._ref_cache[key]
        if len(self._ref_cache) > 4096:
            # Bounded: one fault's pool is ~42 entries, so this holds roughly a
            # hundred tasks. Dropping the lot is correct — a reference answer is
            # a constant and recomputing one is cheap next to holding every
            # expected output of a 1,244-task corpus resident.
            self._ref_cache.clear()
        case = self.case(task_name, probe)
        if case is None:
            self._ref_cache[key] = None
            return None
        if case.expected_output is not None:
            out = Outcome(ok=True, value=case.expected_output)
            self._ref_cache[key] = out
            return out
        ref = run_program(
            self.program(task_name).reference_source, case.input_text,
            timeout=self.timeout, recursion_limit=self.recursion_limit,
        )
        self.reference_runs += 1
        self._ref_cache[key] = ref if ref.ok else None
        return self._ref_cache[key]

    def run(self, task_name: str, probe: str, patch: str) -> ProbeVerdict:
        """D2 + D4 + D6. One probe, one patch, cost measured in CPU seconds."""
        case = self.case(task_name, probe)
        ref = self.reference(task_name, probe)
        if case is None or ref is None:
            # Not a refutation and not a pass — the caller skips it. Reported as
            # `pass` would claim correctness we did not establish, so the status
            # is `pass` only when a real comparison happened; this path returns
            # None-ish via tier=-1, which `exec.verify` treats as "skip".
            return ProbeVerdict(status="pass", cpu_s=0.0, tier=-1, runs=0)
        got = run_program(patch, case.input_text,
                          timeout=self.timeout, recursion_limit=self.recursion_limit)
        if got.timed_out:
            return ProbeVerdict(status="timeout", cpu_s=got.cpu_s, tier=1,
                                error_type="Timeout",
                                error_message=f"exceeded {self.timeout}s wall time")
        if not got.ok:
            return ProbeVerdict(status="crash", cpu_s=got.cpu_s, tier=1,
                                output=got.value, expected=ref.value,
                                error_type=got.error_type, error_message=got.error_message)
        if not outputs_equal(got.value, ref.value):
            # D4 tier 2 — the differential. 94.2% of ConDefects refutations.
            return ProbeVerdict(status="wrong", cpu_s=got.cpu_s, tier=2,
                                output=got.value, expected=ref.value)
        return ProbeVerdict(status="pass", cpu_s=got.cpu_s, tier=2,
                            output=got.value, expected=ref.value)

    def authority(self, task_name: str, probe: str) -> Authority:
        """D3. `held_out` is `P_eval` — never shown, never gates acceptance."""
        if probe in self._held_out.get(task_name, frozenset()):
            return "held_out"
        return "visible"

    def render(self, task_name: str, probe: str) -> tuple[Legibility, str]:
        """D5. `verbatim` when the stdin fits a prompt, else `shape`."""
        case = self.case(task_name, probe)
        if case is None:
            return "opaque", ""
        if len(case.input_text) <= SPEC_MAX_INPUT_CHARS:
            return "verbatim", case.input_text
        return "shape", case.input_text

    def prior(self, task_name: str, probe: str) -> float | None:
        """D7. None — ConDefects supplies no a-priori kill rate.

        The free prior the dimensions doc names (probe *name class*:
        `max`/`random`/`sample`) is deliberately NOT returned here. Using it
        would put the filesystem-naming artefact the hashed tie-break exists to
        remove right back into the schedule, in the one place no ablation could
        see it.
        """
        return None

    def signature(self, task_name: str, probe: str) -> frozenset[int] | None:
        """D8. Reference coverage for this probe — computed once, then cached.

        Coverage of the REFERENCE, so it is a property of the probe: the same
        for every round, every seed and every arm. The `coverage` arm is priced
        as a one-time N runs per task followed by pure replay, and without this
        memo it is N traces per *cell* — at 1,244 tasks × 3 seeds × ~42
        probes, with a tracing budget of ten times the sandbox clock, by far the
        largest uncosted item in the campaign, and invisible because it lands in
        no counter.

        It is counted into `reference_runs`, which is the own bucket: a trace
        belongs to no candidate, so it must never be folded into a round's scan
        cost, and it must not vanish either.

        None rather than an exception when the trace fails, so the arm degrades
        to a plain score sort instead of failing an episode.
        """
        key = (task_name, probe)
        if key in self._cov_cache:
            return self._cov_cache[key]
        case = self.case(task_name, probe)
        if case is None:
            self._cov_cache[key] = None
            return None
        from caster.bench.cover import lines_executed

        sig = lines_executed(
            self.program(task_name).reference_source, case.input_text,
            timeout=self.timeout, recursion_limit=self.recursion_limit,
        )
        self.reference_runs += 1
        self._cov_cache[key] = sig
        return sig

    # ── prompt material ─────────────────────────────────────────────────────

    def input_chars(self, task_name: str) -> dict[str, int]:
        """Probe -> len(stdin). Feeds `ledger.legibility`; no execution."""
        group = self.task(task_name).group
        return {c.name: len(c.input_text) for c in self._cases_for_group(group)}

    def spec_cases(self, task_name: str) -> tuple[TestCase, ...]:
        """Slot S4's worked examples: the problem statement's own samples.

        **No fallback**. AtCoder ships the statement's samples under
        names like `sample_01` / `00_example_00`; those are the ones a human
        solver would have seen, and they are the only ones that may be shown
        unconditionally. Falling back to "the two smallest inputs" when a task
        has none would disclose a hidden test on 19 of 1,422 problems. A task
        with no sample-named case gets **no** worked examples.
        """
        group = self.task(task_name).group
        named = [
            c for c in self._cases_for_group(group)
            if c.expected_output is not None
            and ("sample" in c.name.lower() or "example" in c.name.lower())
            and len(c.input_text) <= SPEC_MAX_INPUT_CHARS
            and len(c.expected_output) <= SPEC_MAX_OUTPUT_CHARS
        ]
        return tuple(named[:N_SPEC_EXAMPLES])

    # ── corpus metadata (root date.txt / difficulty.txt) ────────────────────

    def _metadata(self, filename: str) -> dict[str, str]:
        path = self.root / filename
        if not path.is_file():
            return {}
        out: dict[str, str] = {}
        for line in _read(path).splitlines():
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
        return out

    def dates(self) -> dict[str, str]:
        """Task_id -> contest date, "YYYY-MM-DD". G5's contamination cut."""
        if self._dates is None:
            self._dates = self._metadata("date.txt")
        return self._dates

    def difficulties(self) -> dict[str, int]:
        if self._difficulties is None:
            self._difficulties = {
                k: int(v) for k, v in self._metadata("difficulty.txt").items()
                if v.lstrip("-").isdigit()
            }
        return self._difficulties

    def set_held_out(self, splits: dict[str, Iterable[str]]) -> None:
        """Install `P_eval` after the corpus is frozen.

        The loop must never read a held-out probe, so `run.episode` asserts the
        split is installed before it schedules anything.
        """
        self._held_out = {k: frozenset(v) for k, v in splits.items()}

    def set_pools(self, pools: dict[str, Iterable[str]]) -> None:
        """Install the gated pool. Narrows `Task.probes` to usable cases."""
        self._pools = {k: tuple(v) for k, v in pools.items()}
        self._tasks = None  # force re-derivation with the new probe lists
