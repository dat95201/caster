"""Prompt assembly from slots. Never string concatenation at a call site.

```
S1 role + I/O contract        always
S2 program label              always
S3 current source             always
S4 worked examples            when the task has sample/example-named probes
S5 contract: k requirements   when the ledger holds ≥1 proven probe
S6 progress line              with S5 — a COUNT and a pass/fail pattern, never a bare name
S7 output format              always, always last
S8 sampling instruction       VS branch only, round ≥3, replaces S7's tail
```

Six hard rules, each tied to a measurement or a paper finding:

1. **No instruction names a behaviour to avoid.** Not a ban on the token `not` —
   "the 9 that do not pass" is a description, not an instruction. The failure
   mode is *instruction + named target*. Measured on the frozen log: a class the
   model would otherwise emit 5.4% of the time was emitted 14.7% once named
   (2.7×, the priming inversion), and the instruction cut class repetition only
   82.1% → 70.9% while growing the prompt 462 → 1,263 tokens. S7's "nothing
   else" / "no prose" stay: those are format constraints, identical in every
   arm, and changing them breaks CRN round-1 identity.
2. **Never name a probe whose content is not in the prompt.** Satisfied by
   construction here: `render` never writes a case name at all.
3. **k = 3, non-accumulating** — `CellSpec.k_facts` and `_STEER` both say 3
   (DECISIONS 34); this line said 2 until the pilot write-up caught it. The
   fact-selection plateau is 2–5 and measured saturation is 2.57 distinct
   refuters per episode, so 2 was defensible; the default moved to 3 because
   `ledger.evidence` now DEDUPLICATES, which frees a slot. It moved on that
   argument, not on a measurement: leakage was 7.0% at k=2 against 16.8% at
   k=3, and whether dropping near-duplicates pulls that back is what
   `km+steer-k2` vs `km+steer` has to show. Read any sweep against
   `k_facts_rendered`, never the requested k.
4. **Evidence ranks by `K × legibility`, not `K/m̂`** — a different cost
   currency: the scan pays CPU, the prompt pays tokens. The **raw K** is what
   reaches the prompt, because the number there is a claim about the world
   ("an estimated 43% of attempted fixes get this input wrong") and
   must stay true. `ledger.evidence`
   owns this.
5. **Only proven probes.** A probe with a prior and no record is a guess, and a
   wrong guess primes.
6. **Inputs > 400 chars render as a shape**, first line verbatim. `steer.render`.

And Invariant 4: **the reference implementation never enters a prompt.**
Reference *values* for a probe are allowed — the oracle has always been allowed
to say what output was expected. `tests/test_invariants.py` greps
every rendered prompt for the reference source.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Sequence

from caster.steer.render import (
    render_input,
    render_observation,
    render_required,
    snippet,
)

#: disclosure is a first-class, switchable axis. Measured:
#:
#: | disclosed set                                    | exploitability |
#: |--------------------------------------------------|----------------|
#: | `sample-only` — fixed, chosen **by name**         | **0.0%** (0/273) |
#: | `full` — contract top-2, chosen **by kill rate**  | **7.0%** (19/273) |
#:
#: `no-required-line` shows the inputs and withholds the required outputs, which
#: separates "knowing which input matters" from "being handed the answer".
DISCLOSURES = ("full", "sample-only", "no-required-line")

#: S8's threshold schedule.
#:
#:   rounds 1–2  no VS       keeps CRN round-1 identity intact (Invariant 1)
#:   rounds 3–5  threshold None
#:   rounds 6–10 threshold 0.10
#:   rounds 11+  threshold 0.01
#:
#: Lowered late because the cognitive burden of a tight threshold costs quality;
#: the tail is worth asking for only once the mode has been spent.
VS_FIRST_ROUND = 3
VS_THRESHOLD_SCHEDULE: tuple[tuple[int, float | None], ...] = (
    (VS_FIRST_ROUND, None), (6, 0.10), (11, 0.01),
)

# Output-budget sizing. 3.5 chars/token measured on this corpus's Python, 15%
# margin for the model's own formatting. The proposer rewrites a WHOLE program,
# so a reply cut off mid-block is a SyntaxError — and a SyntaxError refutes the
# *harness*, not the proposal. A flat 2048 did exactly that on the 8 corpus
# programs longer than ~7 KB; one 68 KB submission carrying a precomputed table
# needs 22k.
#
# Sized per program rather than raised globally for one reason: `max_tokens` is
# part of the cache key, so a global raise invalidates every cached call, while
# a per-program budget changes the key only for the programs whose budget
# actually moves.
CHARS_PER_TOKEN = 3.5
BUDGET_MARGIN = 1.15
MIN_BUDGET = 2048
#: The chat ceiling. Raised 16_000 -> 32_000 when G6 (`--lines-max`) was
#: switched off: with no source-length screen the corpus admits programs the
#: old ceiling could not rewrite, and a budget below what the program needs
#: produces a reply cut off mid-block on EVERY round of EVERY seed of exactly
#: those tasks — a `truncated_response` that is a property of this constant,
#: not of the model. Measured over ConDefects' 2,864 Python faults:
#:
#:     out-tokens needed = ceil(len(source) / 3.5 * 1.15)
#:     median 152 · p95 1,298 · p99 5,871 · max 63,153 (one 192 KB submission)
#:     > 16,000: 4 programs (0.14%)   > 32,000: 1 program
#:
#: So 32_000 is the smallest round ceiling that stops this constant being the
#: binding constraint on all but one fault in the benchmark. It is NOT free:
#: `max_tokens` is in the cache key, so the three programs whose budget moves
#: are re-keyed and redrawn. Nothing else is.
MAX_BUDGET = 32_000
# A reasoning model spends hidden tokens out of this same budget before writing
# a character of the answer, so the chat ceiling would cut the program off
# mid-rewrite on exactly the longest tasks.
MAX_BUDGET_REASONING = 48_000
REASONING_HEADROOM = 24_000
_REASONING_PREFIXES = ("o1", "o3", "o4", "o5")


def is_reasoning_model(model: str) -> bool:
    return bool(model) and model.split("-", 1)[0] in _REASONING_PREFIXES


def output_ceiling(model: str | None = None) -> int:
    """The largest output budget any single call may request, for this model.

    One function, because the ceiling was written down twice — here and as a
    bare `16_000` inside `propose.draw` — and the second copy silently clipped
    a k-candidate draw back to the chat ceiling even on a reasoning model.
    A ceiling in two places is a ceiling that disagrees with itself.
    """
    return MAX_BUDGET_REASONING if (model and is_reasoning_model(model)) else MAX_BUDGET


def prompt_tokens_estimate(text: str) -> int:
    """The same estimate `llm.client.check_window` applies to a built prompt."""
    return math.ceil(len(text) / CHARS_PER_TOKEN * BUDGET_MARGIN)


def window_needed(source: str, model: str | None = None,
                  overhead_tokens: int = 1200) -> int:
    """Served context a task needs: whole source in, whole rewrite out.

    Slot S3 carries the entire current source and the reply carries an entire
    rewrite of it, so the window has to hold BOTH — which is why raising
    `MAX_BUDGET` alone does not admit a long program. `overhead_tokens` is the
    rest of the prompt (role, contract, examples, the k-fact block), measured
    at well under 1,200 tokens on this corpus's longest rendered prompts.

    `caster preflight` reports the admitted tasks whose number exceeds
    `LLM_CONTEXT_TOKENS`: with G6 off that list is the whole of what "the
    source does not fit a prompt" now means, and it is a number before the run
    rather than a `context_overflow` discovered in round 1 of every seed.
    """
    return (prompt_tokens_estimate(source) + overhead_tokens
            + budget_for_source(source, model=model))


def budget_for_source(source: str, model: str | None = None) -> int:
    """Output tokens needed to rewrite `source`, floored at MIN_BUDGET.

    `model` only ever *raises* the ceiling, and only for a reasoning model — a
    chat model gets the identical number it got before this parameter existed,
    which is what keeps the response cache valid across the change.
    """
    estimate = math.ceil(len(source) / CHARS_PER_TOKEN * BUDGET_MARGIN)
    if model and is_reasoning_model(model):
        return max(MIN_BUDGET, min(estimate + REASONING_HEADROOM, MAX_BUDGET_REASONING))
    return max(MIN_BUDGET, min(estimate, MAX_BUDGET))


def vs_threshold_for(round_index: int) -> float | None:
    """The VS probability ceiling for this round, or None for "no ceiling"."""
    if round_index < VS_FIRST_ROUND:
        return None
    chosen: float | None = None
    for start, thr in VS_THRESHOLD_SCHEDULE:
        if round_index >= start:
            chosen = thr
    return chosen


def vs_active(round_index: int, k_candidates: int) -> bool:
    """Whether slot S8 replaces S7 this round.

    Rounds 1–2 carry no VS whatever `k_candidates` says, and that is what keeps
    the CRN round-1 identity intact against the non-VS arms (Invariant 1). The
    `vs_schedule` flag is a separate question — it decides whether S8 carries a
    probability *ceiling*, not whether S8 appears.
    """
    return k_candidates > 1 and round_index >= VS_FIRST_ROUND


@dataclasses.dataclass(frozen=True)
class Fact:
    """One requirement the patch must meet. Positive by construction.

    `kill_rate` is the **raw K** (rule 4). `input_text` is the probe's stdin;
    `render` decides verbatim vs shape. `required` is the reference *value* —
    allowed; the reference *source* is not (Invariant 4). `last_*` is what the
    most recent candidate did on this probe, shown only when it exists.
    """

    input_text: str
    required: str | None
    kill_rate: float
    last_status: str | None = None
    last_output: str | None = None
    last_output_len: int | None = None
    last_error_type: str | None = None
    last_error_message: str | None = None


@dataclasses.dataclass(frozen=True)
class Example:
    """A worked example from the problem statement (slot S4)."""

    input_text: str
    output_text: str


@dataclasses.dataclass(frozen=True)
class Prompt:
    """The assembled prompt plus what it is made of, for the record."""

    text: str
    slots: tuple[str, ...]
    k_facts_rendered: int
    vs_threshold: float | None


# ── the slots ────────────────────────────────────────────────────────────────


def s1_role() -> str:
    return (
        "You are repairing a single buggy Python program. It reads its input "
        "from standard input and writes its answer to standard output."
    )


def s2_label(program_label: str) -> str:
    return f"Program: `{program_label}`"


def s3_source(buggy_source: str) -> str:
    return f"Current source:\n```python\n{buggy_source}\n```"


def s4_examples(examples: Sequence[Example]) -> str:
    """Worked examples of the intended behaviour.

    Empty when the task has no `sample`/`example`-named probe — there is **no
    fallback to "the smallest inputs"**, which is the deleted defect: it can
    disclose a hidden test. `bench.adapter.spec_cases` enforces that upstream;
    this slot just renders what it is given.
    """
    if not examples:
        return ""
    blocks = [
        f"Example {i}\ninput:\n{e.input_text.rstrip()}\n"
        f"expected output:\n{e.output_text.rstrip()}"
        for i, e in enumerate(examples, 1)
    ]
    return (
        "The program reads from standard input and writes to standard output. "
        "Worked examples of the intended behaviour:\n\n" + "\n\n".join(blocks)
    )


def s5_contract(facts: Sequence[Fact], *, disclosure: str = "full",
                show_kill_rate: bool = True,
                show_last_output: bool = True) -> str:
    """The counterexample contract. Every line is a requirement the patch must MEET.

    There is no "do not", no "already eliminated", no failure-class name, and
    nothing about what the previous patch *was* (rule 1). `your last attempt …`
    is a fact about an observation, not an instruction about a class.

    `show_kill_rate` is an ablation in its own right: it is the only thing in
    the prompt that says which requirement is load-bearing, it is cheap (one
    float per line), and it has no analogue in the fact-selection literature —
    so it must be reported as an addition rather than smuggled in.
    """
    if disclosure == "sample-only":
        # The contract is withheld entirely; S4's by-name samples are the whole
        # disclosure. Measured exploitability 0.0% (0/273).
        return ""
    if not facts:
        return ""
    lines = ["Your patch must produce exactly this output on each of these inputs:"]
    if disclosure == "no-required-line":
        lines = ["Your patch is checked on each of these inputs:"]
    for i, f in enumerate(facts, 1):
        # "kills N% of attempts" was test-prioritisation jargon aimed at a
        # reader who already knew the field, and "attempts" named nothing the
        # model could see. Spelled out instead — but still as an ESTIMATE, not
        # a tally, because the number is the Beta posterior and not the
        # observed frequency: k=1,n=1 shows 43% where the tally is 100%, and
        # k=4,n=8 shows 43% where the tally is 50%. "N% of your attempts got
        # this wrong" would be false in both. See prompt rule 4.
        weight = (f" — an estimated {f.kill_rate:.0%} of attempted fixes "
                  f"get this input wrong" if show_kill_rate else "")
        lines.append(f"\n  Input #{i}{weight}:")
        lines.append(render_input(f.input_text))
        if disclosure != "no-required-line":
            lines.append(render_required(f.required))
        if f.last_status is not None and show_last_output:
            # The one line in S5 that puts the WRONG answer, verbatim, into the
            # context immediately before generation. It is an observation and
            # not a prohibition, so prompt rule 1 does not forbid it — but Rana
            # (arXiv 2601.08070), measuring Qwen2.5-7B-Instruct, finds 87.5% of
            # negative-constraint failures are PRIMING: naming the target
            # activates it, and failures attend more to the target mention
            # (TMF 0.205) than to the negation cue (NF 0.141). That mechanism
            # is about token activation in context, not about a sentence's
            # logical form, so it plausibly reaches this line too. "Plausibly"
            # is why `show_last_output` is an ARM (`km+steer-nolast`) and not a
            # deletion: the paper measures instructions, this is not one, and
            # the extrapolation has to be priced rather than assumed.
            lines.append(
                f"    your last attempt "
                f"{render_observation(f.last_status, f.last_output, f.last_error_type, f.last_error_message, f.last_output_len)}"
            )
    return "\n".join(lines) if len(lines) > 1 else ""


def s5_exclusion(facts: Sequence[Fact]) -> str:
    """Arm 13 only — the negative block, rendered so it can be **compared**.

    This deliberately breaks prompt rule 1. It exists because arm 13 is the
    thing the contract block replaces, and a comparison needs it on the page.
    Nothing else may call it, and `tests/test_invariants.py` says so.

    Measured, on 927 CRN-paired cells: the instruction cut class repetition only
    82.1% → 70.9%; at LOW semantic pressure it ran **backwards** — a class the
    model would otherwise emit 5.4% of the time was emitted 14.7% once named
    (2.7×); at high pressure the forbidden class still came back 63.7% of the
    time; the benefit decayed to zero by |E| = 5 and inverted by |E| = 8; and
    the prompt grew 462 → 1,263 tokens buying it.
    """
    if not facts:
        return ""
    lines = ["Do not propose a patch that repeats any of these already-eliminated "
             "behaviours:"]
    for i, f in enumerate(facts, 1):
        lines.append(f"\n  On input #{i}:")
        lines.append(render_input(f.input_text))
        if f.last_status is not None:
            lines.append(
                f"    your previous patch "
                f"{render_observation(f.last_status, f.last_output, f.last_error_type, f.last_error_message, f.last_output_len)}"
                f" — do not do that again"
            )
    return "\n".join(lines) if len(lines) > 1 else ""


def s6_progress(facts: Sequence[Fact], surviving: int = 0) -> str:
    """A COUNT and a pass/fail pattern of the shown inputs. Never a bare name (rule 2).

    The `surviving` half is **off by default**: it renders only for
    `km+steer-pass`, the arm that exists to price it. The obvious implementation
    is dead code — fill it on `accept`, and the episode ends before it can
    render — and an oracle that stops at the first divergence throws away
    everything a refuted candidate passed. This scan knows them, so "these
    inputs already produced the right answer" is a positive, zero-leak,
    zero-extra-cost fact. It is still the one block here that no measurement
    prices, which is why it is an arm and not a default.
    """
    shown = [f for f in facts if f.last_status is not None]
    if not shown and surviving <= 0:
        return ""
    bits: list[str] = []
    if shown:
        pattern = " ".join("pass" if f.last_status == "pass" else "fail" for f in shown)
        where = "the input above" if len(shown) == 1 else f"the {len(shown)} inputs above"
        bits.append(f"On {where}, your last attempt scored: {pattern}.")
    if surviving > 0:
        bits.append(
            f"{surviving} further input{'s' if surviving != 1 else ''} already "
            f"produced the right answer and must keep doing so."
        )
    return " ".join(bits)


def s7_output_format() -> str:
    """Format constraints only — identical in every arm (rule 1)."""
    return (
        "Return the corrected source for the whole program and nothing else: "
        "a single ```python fenced code block, no prose before or after it."
    )


def s8_sampling(k_candidates: int, threshold: float | None) -> str:
    """Verbalized sampling, in its **CoT** form. Replaces S7's tail.

    The reasoning paragraph is not decoration either. Zhang et al. (ICML 2026,
    arXiv 2510.01171) measure a "cognitive burden": asking a SMALL model for a
    list — plain Sequence, Multi-turn, or bare VS-Standard — costs output
    quality, and only VS-CoT and VS-Multi recover it, while the diversity gain
    itself runs 1.5-2x larger on large models than on small ones. The proposer
    here is a 7B coder, smaller than either model class in that figure, so the
    bare VS-Standard this used to render was the one variant their Figure 3f
    says a model this size cannot afford.

    Ask for a *distribution*, not an instance. The probability is not
    decoration — it is what makes the request distribution-level; asking for a
    list without one loses the diversity gain. This is the only exploration
    control the loop has: the alternative is a negative instruction, which
    barely moves it (duplicate-patch rate 0.235 → 0.192) and primes besides.
    """
    tail = ""
    if threshold is not None:
        tail = f" Only include candidates whose probability is below {threshold:g}."
    return (
        f"First work out, in one short paragraph, what could be wrong with the "
        f"program and which distinct repairs are plausible. Then generate "
        f"{k_candidates} different candidate corrections of the whole program, "
        f"each with the probability you would assign to it being the correct "
        f"fix.{tail}\n"
        f"Put the paragraph first, then the candidates, formatted exactly like "
        f"this and with nothing else after them:\n"
        f"### Candidate <n> (probability: <p>)\n"
        f"```python\n<the whole corrected program>\n```"
    )


def s9_transcript(history: Sequence[str], *, cap_tokens: int) -> str:
    """ChatRepair-style full transcript. Not in the default grid.

    It needs a restart policy: a transcript cap and a restart after N
    consecutive failures, both in the cell key, with
    `history_restarts` on the record. Without a cap this is the one arm that can
    hit the context ceiling, so it is also the arm whose
    `proposal_error="context_overflow"` count must be reported before any of its
    numbers are. It is already measured to be bad on this corpus (budgeted
    success 0.50 against no-memory's 0.63).
    """
    if not history:
        return ""
    cap_chars = int(cap_tokens * CHARS_PER_TOKEN)
    kept: list[str] = []
    used = 0
    for entry in reversed(history):     # keep the most recent turns
        if used + len(entry) > cap_chars:
            break
        kept.append(entry)
        used += len(entry)
    kept.reverse()
    if not kept:
        return ""
    return "Previous attempts in this session:\n\n" + "\n\n".join(kept)


# ── assembly ─────────────────────────────────────────────────────────────────


def build(
    *,
    program_label: str,
    buggy_source: str,
    examples: Sequence[Example] = (),
    facts: Sequence[Fact] = (),
    surviving: int = 0,
    round_index: int = 1,
    steer_on: bool = False,
    disclosure: str = "full",
    show_kill_rate: bool = True,
    show_last_output: bool = True,
    k_candidates: int = 1,
    vs_schedule: bool = False,
    steer_mode: str = "contract",
    transcript: Sequence[str] = (),
    transcript_cap_tokens: int = 6000,
) -> Prompt:
    """One prompt builder for every arm.

    `steer_on=False` is the ablation: S5 and S6 disappear and the prompt becomes
    the unconditioned one. **Round 1 is byte-identical to it in every arm**
    whenever the ledger holds no proven probe, which is what preserves CRN
    round-1 identity and the shared response cache (Invariant 1).
    `tests/test_invariants.py` asserts it over the whole arm roster.
    """
    if disclosure not in DISCLOSURES:
        raise ValueError(f"unknown disclosure {disclosure!r}; one of {DISCLOSURES}")

    used: list[str] = ["S1", "S2", "S3"]
    sections = [s1_role(), s2_label(program_label), s3_source(buggy_source)]

    s4 = s4_examples(examples)
    if s4:
        sections.append(s4)
        used.append("S4")

    rendered_facts = 0
    if steer_on:
        if steer_mode == "exclusion":
            s5 = s5_exclusion(facts)
            tag = "S5x"
        else:
            s5 = s5_contract(facts, disclosure=disclosure,
                             show_kill_rate=show_kill_rate,
                             show_last_output=show_last_output)
            tag = "S5"
        if s5:
            sections.append(s5)
            used.append(tag)
            rendered_facts = len(facts)
            # S6 rides with S5 only for the positive block: a progress line
            # under an exclusion block would restate the forbidden behaviour,
            # which is the priming mechanism, not a report of it.
            if steer_mode != "exclusion":
                s6 = s6_progress(facts, surviving)
                if s6:
                    sections.append(s6)
                    used.append("S6")

    if transcript:
        s9 = s9_transcript(transcript, cap_tokens=transcript_cap_tokens)
        if s9:
            # Before S7, because S7 is always last.
            sections.append(s9)
            used.append("S9")

    threshold: float | None = None
    if vs_active(round_index, k_candidates):
        threshold = vs_threshold_for(round_index) if vs_schedule else None
        sections.append(s8_sampling(k_candidates, threshold))
        used.append("S8")
    else:
        sections.append(s7_output_format())
        used.append("S7")

    return Prompt(
        text="\n\n".join(sections),
        slots=tuple(used),
        k_facts_rendered=rendered_facts,
        vs_threshold=threshold,
    )
