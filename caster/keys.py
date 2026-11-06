"""Three identity keys, three jobs, one constructor each.

Two call sites building a cell key from separate tuple literals will drift, and
the symptom is a silently re-run grid rather than an error. So: **one constructor
each, no literals anywhere else.**

| key          | = f(...)                                                          | breaks if |
|--------------|-------------------------------------------------------------------|-----------|
| `cache_key`  | prompt, model, temperature, max_tokens, nonce, reasoning_effort    | any prompt byte changes |
| `cell_key`   | task, seed, policy, every knob that changes what the cell *is*     | a knob is added to the CLI but not to the key → cells silently collide |
| `episode_id` | `sha256(cell_key)[:12]`                                           | — |

Two exclusions are load-bearing enough to name, and both are tested:

* **`budget` is not in the key.** A B-round episode is the prefix of a longer
  one — same task, same seed, same nonces, so round *k* is the identical draw at
  either budget. Putting it in gives one cell two episode ids, so topping a cell
  up from 12 rounds to 20 appends a *second* episode instead of rewriting the
  first, and every round-averaged estimator counts the first 12 draws twice.
  `CellSpec` has no `budget` field at all, so it cannot be passed by accident;
  `report.records.needs_walk` handles the top-up instead.
* **Retry / giving-up caps are not in the key** (`MAX_CONSECUTIVE_BACKEND_FAILURES`
  and its kind). Raising one extends an episode that hit it; it never changes a
  draw already made. It goes on the *record*, so an analysis can see which cells
  were truncated by it.

The *universe list* is likewise out: the same (task, seed, policy) is the same
cell whichever list named it.

**The rule binds consumers, not just producers.** Two ways an analysis layer
breaks it, both of which produce a plausible-looking table:

* re-deriving a cell key by hand from a few pinned fields and keying on
  `(task, seed)`. Every arm differing only in an unpinned field collapses onto
  one key, last-write-wins, and because episodes arrive sorted by `episode_id`
  (a sha256 prefix) the winner is arbitrary — so an arm that blocks nearly every
  patch can take the baseline's place, and every reported divergence is then
  about something else.
* adding arm-identity fields to a frozen summary by hand and missing one. A key
  never written reads `False`, so a whole arm passes every main-grid filter and
  pools into the wrong condition.

Therefore `CellSpec.as_dict()` and `CellSpec.key()` read the **same dataclass
fields**, so the frozen summary and every analysis filter derive their fields
from the key constructor by construction, never from a hand-maintained list.
`tests/test_keys.py` asserts that the two agree field for field.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Callable, Iterable, Sequence

# ── Invariant 1: the draw nonce ──────────────────────────────────────────────
# Round 1 carries no facts, so the steered prompt is byte-identical to the
# unsteered one and must share its cache entry. That only works if the nonce is
# a function of (task, seed, round) and **deliberately not** of the policy, the
# arm, or any ablation flag.
#
# Two conditions whose prompt is byte-identical then share one draw; the moment
# the prompts diverge, the prompt itself separates the cache keys, so nothing is
# shared that should not be. An engineer who derives the nonce from `cell_key` —
# which does contain the policy — destroys CRN silently and buys every
# completion once per arm.
#
# Corollary: the nonce is NOT a join key to the call ledger. One nonce maps to
# several ledger rows the moment arms diverge; `cache_key` is that key.


def draw_nonce(task: str, seed: int, round_index: int) -> str:
    """The draw nonce. NOT a function of policy, arm, or any ablation flag."""
    return f"{task}|seed{seed}|r{round_index}"


def aux_nonce(namespace: str, *parts: object) -> str:
    """A nonce for a draw that is NOT a reported round.

    A selection draw gets its own namespace, so no value that *selects* can be
    a value that is *reported*. The namespace is mandatory and must not be
    empty, because an unnamespaced auxiliary draw is indistinguishable from a
    reported one in the call ledger.
    """
    if not namespace or "|" in namespace:
        raise ValueError(f"aux nonce namespace must be non-empty and '|'-free, got {namespace!r}")
    return "|".join([namespace, *(str(p) for p in parts)])


def cache_key(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    nonce: str,
    reasoning_effort: str = "",
) -> str:
    """Identity of one model call.

    `reasoning_effort` is *appended* rather than always present, and that is
    deliberate rather than tidy: it is empty for every chat model, an empty
    field still changes a hash, and writing it into the tuple unconditionally
    would re-key every chat-model response already on disk. The two key spaces
    cannot collide, because a run that sets an effort is by construction on a
    reasoning model that no chat-model key was ever built for.
    """
    parts = [model, repr(float(temperature)), str(int(max_tokens))]
    if reasoning_effort:
        parts.append(f"effort={reasoning_effort}")
    parts += [nonce, prompt]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


# ── the cell key ─────────────────────────────────────────────────────────────

#: Cold-start estimators for m̄. `max` is the shipped default and
#: the pessimistic one; the other two are what arm 9 sweeps.
COLD_STARTS = ("max", "median", "q75")

#: the two defensible priors, shipped as a pre-registered pair. The
#: default is the measured one: `Beta(½,2)` has prior mean 0.2, chosen to sit
#: above this corpus's base rate (≈0.03) without reaching ½. Jeffreys
#: `Beta(½,½)` puts an unseen case at K=0.5, which outranks a proven refuter
#: whose record has decayed below it (k=2,n=10 ⇒ K=0.227) — under homogeneous
#: per-case costs that is a full-pool sweep.
PRIORS: dict[str, tuple[float, float]] = {
    "shipped": (0.5, 2.0),
    "jeffreys": (0.5, 0.5),
}
DEFAULT_PRIOR = "shipped"

#: disclosure is a first-class, switchable axis, not a constant.
DISCLOSURES = ("full", "sample-only", "no-required-line")


@dataclasses.dataclass(frozen=True, order=True)
class CellSpec:
    """Everything that changes what a cell *is*. One field, one knob.

    Adding a knob to the CLI means adding a field here in the same commit, and
    `tests/test_invariants.py` fails the build otherwise.

    What is NOT here, and must never be added:
      * `budget` — a B-round episode is the prefix of a longer one.
      * the universe list — the same (task, seed, policy) is the same cell
        whichever list named it.
      * retry/giving-up caps — they extend an episode, they do not change a draw.
      * `--resume`, `--shards`, `--from-cache-only`, `--dry-run`, and every
        post-hoc audit — they change how the cell is *produced or inspected*.
    """

    task: str
    seed: int
    policy: str

    # scheduler knobs
    prior: str = DEFAULT_PRIOR
    cold_start: str = "max"
    explore_p: float = 0.15
    #: Containment at which a probe is deferred to the back of the round's
    #: queue because something that already PASSED refutes the same patches.
    #: 0.0 disables the rule, which is what arm `km-ov0` is. See
    #: `caster.exec.ledger` for why 0.75 and not ⅔.
    overlap_thr: float = 0.75
    #: Refutations a probe needs on record before its rejection set may
    #: suppress anything.
    overlap_min_rej: int = 1
    #: "rank" (the default) | "name" | "hash". See `schedule_tie_rank`.
    #: Whether a CONFIRMED timeout (REPEATS_BEFORE_HALT consecutive timeouts
    #: of one (patch, probe) on one host) may be served from the verdict store
    #: instead of re-executed. ConDefects comes from AtCoder, where exceeding
    #: the limit is a non-accepting verdict, so a confirmed timeout refuting a
    #: patch is the benchmark's own semantics. It is a CELL field and not a
    #: protocol field on purpose: the verdict on disk is identical either way,
    #: so this must not change `protocol_sha` and discard a valid store — but
    #: it DOES change measured results, so it must keep two regimes from
    #: pooling. False restores the blanket quarantine.
    serve_stable_timeouts: bool = True
    tie_break: str = "rank"

    # steering knobs; all inert when the policy is unsteered
    steer_on: bool = False
    #: 3, not 2. The contract block now DEDUPLICATES: two facts that refute the
    #: same patches spend two slots to say one thing, so the near-copy is
    #: dropped and the next distinct refuter takes the slot. Asking for 3 under
    #: deduplication renders about what asking for 2 rendered before, and the
    #: number that any k-sweep must be read against is `k_facts_rendered`, not
    #: this one. Measured on the pilot: 55% of episodes end with fewer than 2
    #: distinct proven refuters, so k is a ceiling far more often than a count.
    k_facts: int = 3
    disclosure: str = "full"
    show_kill_rate: bool = True
    #: Whether S5 shows what the last attempt actually printed. True is
    #: the shipped block; `km+steer-nolast` sets it False. It is an arm
    #: rather than a default because the evidence against it is an
    #: EXTRAPOLATION — Rana measures negative INSTRUCTIONS priming their
    #: own target, and this line is an observation, not an instruction.
    show_last_output: bool = True
    k_candidates: int = 1            # >1 ⇒ verbalized sampling
    vs_schedule: bool = False
    candidate_policy: str = "exploit"
    passing_block: bool = False
    #: "contract" is the positive block. "exclusion" is the negative block it
    #: replaces, present only because arm 13 is the comparison and a comparison
    #: needs it rendered. "chat" is ChatRepair-style full transcript.
    steer_mode: str = "contract"
    transcript_cap_tokens: int = 6000
    transcript_restart_after: int = 5

    # proposer protocol
    model: str = ""
    temperature: float = 1.0
    reasoning_effort: str = ""

    def __post_init__(self) -> None:
        if self.prior not in PRIORS:
            raise ValueError(f"unknown prior {self.prior!r}; one of {sorted(PRIORS)}")
        if self.cold_start not in COLD_STARTS:
            raise ValueError(f"unknown cold_start {self.cold_start!r}; one of {COLD_STARTS}")
        if self.disclosure not in DISCLOSURES:
            raise ValueError(f"unknown disclosure {self.disclosure!r}; one of {DISCLOSURES}")
        if self.tie_break not in TIE_BREAKS:
            raise ValueError(f"unknown tie_break {self.tie_break!r}")
        if self.candidate_policy not in ("exploit", "explore", "as_given"):
            raise ValueError(f"unknown candidate_policy {self.candidate_policy!r}")
        if self.steer_mode not in ("contract", "exclusion", "chat"):
            raise ValueError(f"unknown steer_mode {self.steer_mode!r}")
        if self.k_candidates < 1:
            raise ValueError("k_candidates must be >= 1")
        if not 0.0 <= self.explore_p <= 1.0:
            raise ValueError("explore_p must be in [0, 1]")
        if not 0.0 <= self.overlap_thr <= 1.0:
            raise ValueError("overlap_thr must be in [0, 1]")
        if self.overlap_min_rej < 1:
            raise ValueError("overlap_min_rej must be >= 1")
        if self.k_facts < 0:
            raise ValueError("k_facts must be >= 0")

    # -- the one source of truth for "what are this cell's fields" ------------

    def as_dict(self) -> dict[str, Any]:
        """The cell's identity as a flat dict.

        The frozen summary and every analysis filter read THIS, so they cannot
        drift from `key()` — both walk `dataclasses.fields(self)`.
        """
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def key(self) -> str:
        """Canonical cell key. Stable field order, explicit types.

        Every field is written, including defaults. Appending optional knobs
        only when set would avoid renaming every existing cell when a flag is
        added — a real convenience, and the cause of the second failure in the
        module docstring, because a key never written reads `False` in an
        analysis and an arm pools into the wrong condition. The key is versioned
        instead: `KEY_VERSION` bumps when a field is added, every cell is
        re-keyed at once, and nothing silently reads as a default.
        """
        parts = [f"v{KEY_VERSION}"]
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool):
                rendered = "1" if value else "0"
            elif isinstance(value, float):
                rendered = repr(float(value))
            else:
                rendered = str(value)
            parts.append(f"{f.name}={rendered}")
        return "|".join(parts)

    @property
    def episode_id(self) -> str:
        return hashlib.sha256(self.key().encode()).hexdigest()[:12]

    def replace(self, **kw: Any) -> "CellSpec":
        return dataclasses.replace(self, **kw)


#: Bumped whenever a field is added to `CellSpec`, and whenever the MEANING
#: of an existing key changes — Invariant 13 is "every value that changes a
#: measurement is in a key or in `.meta.json`", and a silent re-meaning is the
#: one way to break it without adding a field. v2: the schedule tie-break is
#: seeded on (task, seed) instead of the cell key, so round 1 is identical
#: across arms. v3: that was not enough — a shared neutral permutation is
#: still a claim at a round where no arm has anything to claim, so the
#: default tie-break is now POOL RANK and every arm but `random` walks the
#: natural order at round 1. Same cells, same nonces, different results — the
#: bump is what keeps rows of different versions from pooling. v4: `model` is
#: now RESOLVED before the key is built (`cli._model`). The field always
#: existed; the CLI left it empty and the client fell back to `$MODEL`, so
#: every v3 key claims `model=` for a cell drawn by a named model, and two
#: models shared one `episode_id`. The meaning of every existing key therefore
#: changed, which is exactly what this constant is bumped for — and
#: `shard.PROTOCOL_FIELDS` now carries `key_version`, so a v3 shard and a v4
#: shard are a hard stop at merge rather than a silent pool.
#: v7: `serve_stable_timeouts` joined the spec (DECISIONS 38).
#: v6: `show_last_output` joined the spec (DECISIONS 37). v5: `overlap_thr`
#: and `overlap_min_rej` joined the spec (DECISIONS 33) and
#: `k_facts` defaulted 2 → 3 (DECISIONS 34). All three change what a cell IS,
#: so every cell is re-keyed at once rather than letting an unwritten field
#: read as a default in an analysis. Every pilot row is v4 or older and does
#: not pool with anything produced from here on.
KEY_VERSION = 7


#: How the schedule breaks a tie between probes of EQUAL score.
#:
#: "rank" — pool position. The default, and the only one that is not a claim.
#: "name" — the case name. On ConDefects pool order IS `sorted(filenames)`, so
#:          this equals "rank"; it is kept for an adapter whose pool order is
#:          not alphabetical, where the two come apart.
#: "hash" — a neutral permutation. The ABLATION, not the default: it is what
#:          measures how much the benchmark's own file order is worth.
TIE_BREAKS = ("rank", "name", "hash")


def schedule_tie_rank(pool: Sequence[str]) -> Callable[[str], int]:
    """Tie-break on POOL POSITION. The default, because of what round 1 is.

    At round 1 every ledger is empty, so every probe carries the identical `K`
    and the identical `m̂`, so every score ties and the tie-break IS the whole
    order. A policy has nothing to say at that point — and a tie-break that
    says something anyway is not neutrality, it is a fabricated difference
    between arms that hold identical information.

    `sorted` is stable, so ranking on pool position is exactly "leave it
    alone": `natural`, `no-mem`, `fault-record`, `km-bucket`, `cost-only`,
    `kill-only` and `km` all walk the SAME order at round 1, and any later
    difference is evidence the ledger actually acquired. `random` is excluded
    by construction — a reshuffled control is its entire definition.

    This replaces a hashed default whose cost was measured: it put the arm's
    NAME into the order (`CellSpec.key()` writes every field), so renaming `km`
    to `km2` changed its cost, and the same logic under two seeds of that hash
    moved an arm's CPU by 34% with nothing else changed. `fault-record` and
    `km-bucket` never used it — their no-evidence bucket already keeps pool
    order — so three arms did one thing and three the opposite.
    """
    rank = {name: i for i, name in enumerate(pool)}
    return lambda n: rank.get(n, len(rank))


def schedule_tie_seed(task: str, seed: int) -> str:
    """The CRN unit, for `tie_break="hash"` — the ABLATION, not the default.

    Deliberately the same shape as `draw_nonce`, and for the same reason: at
    round 1 every Block A arm holds an empty ledger, so every arm carries the
    identical `K` and the identical `m̂`, so every score ties and the tie-break
    IS the order. If that tie-break is keyed on the cell, it is keyed on
    `policy` — `CellSpec.key()` writes every field — and the arm's NAME decides
    which probe runs first at the one moment the arm has nothing to decide
    with. Renaming `km` to `km2` then changes its cost.

    That also contradicted the design's own CRN rule one line over: the draw
    nonce is policy-free so all 13 Block A arms share round 1's candidate, and
    the schedule then split that shared candidate back apart by arm name.
    `fault-record` and `km-bucket` never did — their no-evidence bucket keeps
    pool order — so three arms did one thing and three the opposite.
    """
    return f"{task}|seed{seed}"


def schedule_tie_break(tie_seed: str, case_name: str) -> str:
    """The tie-break decides the whole tail, so it may not be the case name.

    Unseen cases all carry the same K and the same m̂, so they carry the
    identical score: the tie-break is not a tie-break, it is the total order of
    the entire unexplored pool, every round, in every `km` arm. Breaking on case
    name makes that order the filesystem order — and that order is measured to
    be non-neutral: 78.4% of `natural`-order rounds begin on a `sample`/
    `example` case, because AtCoder's naming sorts the public, small, highly
    discriminating cases first.

    `tie_seed` is `schedule_tie_seed(task, seed)` — NOT the cell key. Keying it
    on the cell put `policy` in the hash; see that function. `tie_break="name"`
    reproduces the filesystem order deliberately, for the audit subset that
    measures its size: pool order IS `sorted(filenames)`, so "name" is pool
    order exactly, not an approximation of it.
    """
    return hashlib.sha256(f"{tie_seed}|{case_name}".encode()).hexdigest()


# ── Invariant 6: every CLI flag is in exactly one declared list ──────────────
# "Every flag is in the key" is the wrong rule. Flags split in two, and the
# split is itself declared data:
#
#   key flags     — they change what the cell IS.
#   non-key flags — they change how the cell is PRODUCED OR INSPECTED.
#
# Post-hoc audit flags stay out of the key deliberately, so they can be switched
# on without invalidating finished cells. A test that only checked the key would
# pass while a flag went undeclared entirely; this one asserts something stronger
# and buildable — a flag in neither list fails the build.

KEY_FLAGS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(CellSpec)
) | frozenset({
    # Aliases the CLI exposes that map onto a CellSpec field.
    "policies",   # plural on the CLI, `policy` per cell
    "seeds",      # plural on the CLI, `seed` per cell
    "tasks",      # the subset of the frozen pool to walk; each task is its own cell
    # The CLI exposes the NEGATIVE of `serve_stable_timeouts`, because the
    # default is to serve and a flag that does nothing by default is clearer
    # written as the opt-out. It is still the same cell field.
    "quarantine_timeouts",
})

NONKEY_FLAGS: frozenset[str] = frozenset({
    # production — how the cell is MADE
    "budget", "resume", "no_resume", "shards", "shard", "shard_index",
    "from_cache_only", "replay_only", "on_cache_miss",
    "dry_run", "jobs", "run_dir", "run_id",
    "force_full_budget", "max_consecutive_backend_failures", "timeout",
    "backend", "gateway_url", "local_replicas", "max_replicas", "keep_serving",
    "context_tokens", "context", "port", "spend_cap", "queue_timeout",
    "service_timeout", "concurrency", "num_parallel", "vram", "model_bytes",
    "limit", "out", "title", "verbose", "quiet", "first", "last",
    "cloud_url", "tok_per_sec", "tok_budget",
    # The UNIVERSE is deliberately out of the key: the same
    # (task, seed, policy) is the same cell whichever list named it. `--control`
    # selects a different universe — the unfiltered control arm — and nothing
    # else, so it belongs here and not in `CellSpec`.
    "control",
    # inspection / post-hoc audit — switchable WITHOUT invalidating a finished
    # cell, which is exactly why the audit flags stay out of the key.
    "full_scan", "check_regression", "check_overfit", "remeasure_cost",
    "remeasure_repeats", "remeasure_seed", "bootstrap", "live", "universe", "freeze_rates",
    "format", "open_browser", "since", "strict",
    # freeze-stage knobs: they define the POOL, which is stamped into
    # `pool_sha256` and recorded in the manifest — not into a per-cell key. The
    # same (task, seed, policy) is the same cell whichever list named it.
    "condefects_root", "test_dir", "overwrite", "date_min", "lines_max",
    "min_pool", "holdout", "control_size", "pool_order", "pool_seed",
    "assemble",
    "dedup_per_group",
})


def check_flag_partition(dests: Iterable[str]) -> list[str]:
    """Return the flags that are in neither declared list, or in both.

    `caster preflight` and `tests/test_invariants.py` both call this. A flag in
    neither fails the build — that is the whole point of Invariant 6.
    """
    problems: list[str] = []
    for dest in sorted(set(dests)):
        if dest in ("help", "command", "subcommand"):
            continue
        in_key = dest in KEY_FLAGS
        in_nonkey = dest in NONKEY_FLAGS
        if in_key and in_nonkey:
            problems.append(f"{dest}: declared in BOTH KEY_FLAGS and NONKEY_FLAGS")
        elif not in_key and not in_nonkey:
            problems.append(
                f"{dest}: in NEITHER declared list. Add it to KEY_FLAGS (and to "
                f"CellSpec, in this same commit) if it changes what the cell IS, "
                f"or to NONKEY_FLAGS if it changes how the cell is produced or "
                f"inspected."
            )
    return problems
