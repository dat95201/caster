"""How a probe's input is shown — verbatim when it fits, as a *shape* when it does not.

Two constants, not one (prompt rule 6):

* **400** is the legibility threshold (`ledger.SHOW_INPUT_CHARS`), the limit the
  worked examples already use. Above it, a probe is ranked down as *evidence*
  even while it stays a good *schedule* entry.
* **240** is the hard quoting cap on any single piece of contest data that does
  reach a prompt. One AtCoder input runs to hundreds of kilobytes and would
  otherwise blow the window on exactly the tasks whose inputs are largest.

The shape renderer exists rather than a truncator because truncating 2.3 MB of
digits to 240 characters and then asserting a required output for it is a
**false fact**. The incident that produced this: the top kill-rate probe for
`abc236_b` is a 2.3 MB stress input. As a schedule entry it is the right probe —
cheap to run, kills almost everything. As a fact it is worthless: the model is
told to produce `66368` on an input it cannot read.

The first line is kept verbatim because on AtCoder it almost always carries N,
the single most useful number about a large case: it says whether this is a
complexity failure or an edge case.

**Case names are never rendered.** A filename is a label the model cannot act
on, and worse, AtCoder's naming is *itself* information: the
`sample`/`example`-named cases are the public, small, highly discriminating
ones. Putting the name in the prompt
leaks the name class, which is the very artefact the hashed tie-break exists to
keep out of the schedule. Rule 2 is then satisfied by construction: nothing in a
prompt names a case whose content is not also in it.
"""
from __future__ import annotations

from caster.exec.ledger import SHOW_INPUT_CHARS

#: The hard quoting cap on any single piece of contest data in a prompt.
MAX_SNIPPET_CHARS = 240

#: First line of a shaped input, so the model sees N without seeing the body.
MAX_HEAD_CHARS = 80


def snippet(text: str | None, limit: int = MAX_SNIPPET_CHARS,
            total: int | None = None) -> str:
    """One-line form of a short piece of program I/O, hard-capped.

    `total` is the TRUE length of the thing `text` is a head of. It exists so a
    value that has already been truncated once — by the verdict store, which
    keeps 240 characters — still reports the size it really had. Without it, a
    warm replay would print "[truncated, 240 chars total]" where the cold run
    printed "[truncated, 8,890 chars total]", and two byte-different prompts
    would differ only by whether the store was warm.
    """
    if not text:
        return "<no output>"
    size = len(text) if total is None else total
    joined = " / ".join(line.strip() for line in text.strip().splitlines())
    if len(joined) > limit or size > len(text):
        return f"{joined[:limit]}… [truncated, {size:,} chars total]"
    return joined


def is_legible(input_text: str, threshold: int = SHOW_INPUT_CHARS) -> bool:
    return len(input_text) <= threshold


def _block(label: str, text: str) -> str:
    """A fenced, verbatim block. Line structure preserved, nothing truncated.

    `snippet()` must never touch something the contract then asserts a required
    output for. It joins lines with `" / "` and cuts at 240 characters, so a
    legible 300-char three-line stdin would reach the model as one mangled,
    truncated line under the heading "produce exactly this output on each of
    these inputs" — which is the **false fact** prompt rule 6 exists to prevent, and
    it would hit every multi-line input at any size.
    """
    body = text.rstrip("\n")
    n = len(body.splitlines())
    # Explicit markers rather than a ``` fence: S7 and S8 ask for the answer in
    # a ```python block, and a second fence in the prompt is the one thing that
    # can make a model close the wrong one. These markers cannot collide with
    # it, and the bytes between them are exact.
    return f"    {label} (verbatim, {n} line{'s' if n != 1 else ''})\n<<<\n{body}\n>>>"


def render_input(input_text: str, *, threshold: int = SHOW_INPUT_CHARS) -> str:
    """The stdin block for one probe. Verbatim when legible, else a shape.

    Legible means "fits in a prompt" (`threshold`, 400 chars), and a legible
    input is shown WHOLE — line breaks and all. An illegible one is never shown
    at all: it becomes a shape, because truncating 2.3 MB of digits to 240
    characters and then asserting a required output for it is a false fact.
    """
    if is_legible(input_text, threshold):
        return _block("stdin:", input_text)
    lines = input_text.splitlines()
    head = lines[0].strip() if lines else ""
    return (
        f"    stdin:    <large input: {len(lines):,} lines, {len(input_text):,} "
        f"chars; first line `{snippet(head, MAX_HEAD_CHARS)}`>"
    )


def render_required(expected: str | None, *, threshold: int = SHOW_INPUT_CHARS) -> str:
    """The required-output block. Same rule, and for the same reason.

    An expected output too large to show is withheld rather than truncated: a
    required value the model cannot read is not a requirement, it is noise with
    an assertion attached.
    """
    text = expected or ""
    if not text.strip():
        return "    required: <empty output>"
    if len(text) <= threshold:
        return _block("required:", text)
    lines = text.splitlines()
    return (f"    required: <large output: {len(lines):,} lines, {len(text):,} "
            f"chars — withheld rather than truncated>")


def render_observation(status: str, output: str | None,
                       error_type: str | None = None,
                       error_message: str | None = None,
                       output_len: int | None = None) -> str:
    """"your last attempt …" — a fact about an observation, never an instruction.

    `last_outcomes` is deliberately LAST-attempt rather than all-attempts: an
    accumulating record grows the prompt roughly 3× for a benefit measured to
    be zero by |E| = 5.
    """
    if status == "timeout":
        return "exceeded the time limit"
    if status == "crash":
        return f"crashed with {error_type}: {error_message}"
    if status == "pass":
        return "produced the required output"
    return f"printed `{snippet(output, total=output_len)}`"
