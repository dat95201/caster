"""The draw: one model call → one or more candidate patches.

Split out of `episode.py` rather than folded into it (a small deviation from
the module map, recorded in `docs/DECISIONS.md`) because parsing a reply is the
one part of the loop that is worth testing without a model, a sandbox or a
benchmark — and it is where a weak proposer's failures actually land.

**Parsing is deliberately forgiving.** A model that ignores the verbalized-
sampling format and returns one fenced block is not a failed round: it is k=1,
and the loop treats it as such. A model that returns blocks without
probabilities gets a uniform prior, and `stated=False` keeps saying so, because
the VS mechanism depends on the model actually producing the distribution and
the rate at which it refuses is a number the VS arm has to report.

The only unrecoverable reply is one with **no closed fence at all**, which is
`TruncatedResponse` — and it is re-raised unchanged so the loop's handling does
not have to fork. The tempting fallback — return the raw reply, prose and
opening fence included, as though it were a patch — always produces a candidate
that fails, so a harness defect enters memory as evidence inside the very
mechanism under study.
"""
from __future__ import annotations

import dataclasses
import re

from caster.llm.client import CallMeta, LLMClient, TruncatedResponse
from caster.steer.template import budget_for_source

# "### Candidate 2 (probability: 0.17)" — tolerant of the heading level, of
# "prob"/"p"/"probability", of = or :, and of a trailing %.
_HEAD = re.compile(
    r"^\s{0,3}#{0,6}\s*candidate\s*\#?\s*(\d+)?\s*"
    r"(?:\(|\[)?\s*(?:probability|prob|p)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)",
    re.IGNORECASE | re.MULTILINE,
)
_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

CANDIDATE_POLICIES = ("exploit", "explore", "as_given")


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One patch from one draw."""

    source: str
    probability: float       # self-reported; uniform when the model gave none
    index: int               # position in the reply, 1-based
    stated: bool             # False when the probability was imputed


def parse_candidates(text: str) -> list[Candidate]:
    """Split a reply into candidates.

    Anchors on the fenced blocks rather than on the headings, because the block
    is the thing we need and the heading is the thing a model drops. Each block
    takes the probability from the nearest heading that precedes it.
    """
    blocks = list(_FENCE.finditer(text))
    if not blocks:
        raise TruncatedResponse(
            f"no closed ```python block in a {len(text)}-char reply — either the "
            f"output budget cut it off or the model answered in prose; "
            f"finish_reason on the record says which (ends: {text[-80:]!r})"
        )
    heads = [(m.start(), m.group(2), m.group(3)) for m in _HEAD.finditer(text)]
    out: list[Candidate] = []
    for i, blk in enumerate(blocks, 1):
        prob: float | None = None
        stated = False
        for pos, raw, pct in heads:
            if pos >= blk.start():
                break
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            prob, stated = (value / 100.0 if pct else value), True
        source = blk.group(1).strip()
        if not source:
            continue
        out.append(Candidate(source=source + "\n",
                             probability=prob if prob is not None else 0.0,
                             index=i, stated=stated))
    if not out:
        raise TruncatedResponse(
            f"every ```python block in a {len(text)}-char reply was empty")
    if not any(c.stated for c in out):
        u = 1.0 / len(out)
        out = [dataclasses.replace(c, probability=u) for c in out]
    return out


def order_candidates(cands: list[Candidate], policy: str) -> list[Candidate]:
    """Which candidate to spend the scan on first.

    `exploit`  — highest self-reported probability. The mode. Right early:
                 P(accept) at round 1 is 0.309 and falls to 0.059 after.
    `explore`  — lowest. Diversity tuning applied at *selection* time rather
                 than at generation time, so it can be swept independently of
                 the prompt threshold.
    `as_given` — the reply's own order. The null: it isolates how much of VS's
                 effect is the k draws and how much is the ranking over them.
    """
    if policy == "exploit":
        return sorted(cands, key=lambda c: (-c.probability, c.index))
    if policy == "explore":
        return sorted(cands, key=lambda c: (c.probability, c.index))
    if policy == "as_given":
        return list(cands)
    raise ValueError(f"unknown candidate policy {policy!r}; one of {CANDIDATE_POLICIES}")


def draw(
    client: LLMClient,
    prompt: str,
    buggy_source: str,
    *,
    k: int,
    nonce: str,
    model: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    meta: CallMeta | None = None,
    from_cache_only: bool = False,
) -> list[Candidate]:
    """One model call → up to k candidates.

    The output budget is k whole programs, not one, then re-clipped by
    `budget_for_source`'s own ceiling — which exists because the backend's
    context window has to hold the prompt too. On the few corpus programs over
    ~7 KB this is the binding constraint on k, and the loop must be prepared for
    a reply that carries fewer than k blocks.
    """
    per = budget_for_source(buggy_source, model=model or client.model)
    max_tokens = min(per * max(1, k), 16_000)
    text = client.complete(
        prompt, nonce=nonce, max_tokens=max_tokens, model=model,
        temperature=temperature, reasoning_effort=reasoning_effort, meta=meta,
        from_cache_only=from_cache_only,
    )
    return parse_candidates(text)


def duplicate_rate(sources: list[str]) -> float:
    """RQ7's measurable-today half: distinct patches per episode.

    Measured on this corpus: 23.5% of unsteered proposals are byte-identical
    repeats within an episode, and a negative instruction barely moves it
    (0.235 → 0.192). This needs no new arm and no model calls.
    """
    if not sources:
        return 0.0
    return 1.0 - (len(set(sources)) / len(sources))
