"""Invariant 5 — one path resolver. No module computes its own data path.

The alternative — each module computing

    DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

beside a CWD-relative `pathlib.Path("data/rounds.jsonl")` somewhere else — agrees
only as long as every entry point does `cd "$ROOT"` first. Two resolution bases
for one concept is how a run writes half its output to one directory and half to
another without erroring once.

`RUN_DIR` moves every artifact of a run under a folder of that name:

    RUN_DIR=campaign-7b   ->  data/campaign-7b/...  logs/campaign-7b/...

Deliberately NOT moved, and each for a stated reason:

  cache/      the response cache is a cross-run asset — Invariant 8. It is what
              makes a re-run free; scoping it per run would cold-start every
              sweep. `CACHE_DIR` overrides it for the rare case you want that.
  verdicts/   the verdict store, for the same reason ("like the response
              cache, cross-run and never scoped per run").
  patches/    content-addressed patch store. `patch_sha` on a
              RoundRecord resolves here, and phase 2 of a LATER run must still
              resolve a sha phase 1 of an earlier one wrote.
  external/   read-only benchmark input, not output of a run.

Three rules for callers:

  1. Always `ensure()` before writing. Under a nested RUN_DIR the intermediate
     directory may not exist, and a bare `mkdir(exist_ok=True)` raises
     FileNotFoundError rather than creating it.
  2. Record `str(path)` in provenance blobs, never a hardcoded "data/..."
     string, or two runs produce byte-identical provenance for different data.
  3. Bash gets the same value from `scripts/run_dir_paths.sh`; do not re-derive
     it. `tests/test_invariants.py` asserts the two agree.
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(
    os.environ.get("CASTER_ROOT") or pathlib.Path(__file__).resolve().parent.parent
)

# Stripped of surrounding slashes so RUN_DIR=/foo/ and RUN_DIR=foo agree, and so
# an accidental leading slash cannot escape the repo via pathlib's absolute-path
# join semantics (ROOT / "/etc" is "/etc", not ROOT/etc).
RUN_SLUG: str = os.environ.get("RUN_DIR", "").strip().strip("/")

DATA_DIR: pathlib.Path = (ROOT / "data" / RUN_SLUG) if RUN_SLUG else (ROOT / "data")
LOGS_DIR: pathlib.Path = (ROOT / "logs" / RUN_SLUG) if RUN_SLUG else (ROOT / "logs")

# The frozen corpus is a sub-namespace of the run, not a sibling of it: the
# RUN_DIR segment goes BETWEEN data and pool, or a run would write its pool into
# the shared one. data/<run>/pool/tasks.json, never data/pool/<run>/tasks.json.
POOL_DIR: pathlib.Path = DATA_DIR / "pool"

# The merged round log, and the done-index resume reads instead of it.
ROUNDS: pathlib.Path = DATA_DIR / "rounds.jsonl"
CELLS_DONE: pathlib.Path = DATA_DIR / "cells.done"
SHARD_DIR: pathlib.Path = DATA_DIR / "shards"

# Invariant 14: smoke output is never mergeable. `caster merge` globs
# shards/*/rounds.jsonl; smoke writes HERE, which that glob cannot match by
# construction rather than by a naming convention an operator has to remember.
SMOKE_DIR: pathlib.Path = DATA_DIR / "smoke"

# Cross-run stores. Never under RUN_DIR — see the module docstring.
CACHE_DIR: pathlib.Path = pathlib.Path(os.environ.get("CACHE_DIR", "") or (ROOT / "cache"))
VERDICT_DIR: pathlib.Path = pathlib.Path(os.environ.get("VERDICT_DIR", "") or (ROOT / "verdicts"))
PATCH_DIR: pathlib.Path = pathlib.Path(os.environ.get("PATCH_DIR", "") or (ROOT / "patches"))

# The call ledger IS output of a run, so it follows RUN_DIR. The gateway owns
# the single global one; a shard that bypasses the gateway gets its own
# and the spend cap is then per process, which is the bug one global ledger
# exists to fix.
CALLS_LOG: pathlib.Path = DATA_DIR / "calls.jsonl"

# Relative forms, for the handful of places that must emit a string a shell or a
# provenance blob reads back (bash entry points all `cd "$ROOT"` first).
DATA_REL: str = f"data/{RUN_SLUG}" if RUN_SLUG else "data"
LOGS_REL: str = f"logs/{RUN_SLUG}" if RUN_SLUG else "logs"


def ensure(*dirs: pathlib.Path) -> None:
    """Mkdir -p, for every directory a stage is about to write into."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def banner(stage: str) -> str:
    """One line naming the stage and where its output is going.

    Printed at the top of every stage so a run that silently wrote to the wrong
    directory is visible in the first line of its log rather than discovered at
    merge time.
    """
    run = RUN_SLUG or "(default — RUN_DIR unset)"
    return f"[{stage}] run={run}  ->  {DATA_REL}/"


def announce(stage: str) -> None:
    """Banner(), to stderr, so it survives `> out.json` redirection.

    Several stages write their payload to stdout. A banner on stdout would end
    up inside the JSON.
    """
    print(banner(stage), file=sys.stderr, flush=True)


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(banner("paths"))
    for name in ("ROOT", "RUN_SLUG", "DATA_DIR", "LOGS_DIR", "POOL_DIR", "ROUNDS",
                 "CELLS_DONE", "SHARD_DIR", "SMOKE_DIR", "CACHE_DIR", "VERDICT_DIR",
                 "PATCH_DIR", "CALLS_LOG"):
        print(f"  {name:12s} {globals()[name]}")
