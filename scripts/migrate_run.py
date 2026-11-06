#!/usr/bin/env python3
"""Start a new RUN_DIR from an old one's frozen corpus. Nothing is deleted.

Why this exists rather than "just re-run it in place".

`caster.keys.KEY_VERSION` is now **5**: it went 3 -> 4 when `model` started
being resolved before the cell key is built, and 4 -> 5 when the overlap knobs
joined `CellSpec` and `k_facts` changed default. Every cell key changed meaning
at each step, so:

  * `shards/sNN/meta.json` under the old run records `key_version: 3`, and
    `shard.meta_conflict` now REFUSES a v5 launch into a directory that
    already holds older rows — appending would pool two key spaces into one file;
  * `collapse()` keys on `episode_id`, which is a hash of the cell key, so rows of
    different versions would not overwrite each other but WOULD both reach
    `analyse.cells()`, which groups on `(task, seed, policy)` and would pool
    them into one cell.

A new run directory is the clean cut, and it is cheap: `cache/`, `verdicts/`
and `patches/` are cross-run by design (Invariant 8), so every draw phase 1
already made is still a cache hit and the kill matrix is still servable. What
is copied here is the only thing that must NOT be regenerated — the frozen
corpus, whose `pool_sha256` is what makes the two runs comparable at all.

    python3 scripts/migrate_run.py --from pilot --to pilot-v5

Then set RUN_DIR=pilot-v5 and re-run `propose` (cache hits, no model calls),
`audit` if the kill matrix did not come across, and `evaluate`.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Copied because regenerating them would change a digest the two runs are
#: compared on. Everything else a run produces is derived and is re-derived.
CARRY = ("pool", "kill_matrix.json", "cost_table.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", required=True, help="existing RUN_DIR slug")
    ap.add_argument("--to", dest="dst", required=True, help="new RUN_DIR slug")
    ap.add_argument("--force", action="store_true",
                    help="overwrite files already present in the destination")
    args = ap.parse_args(argv)

    src = ROOT / "data" / args.src
    dst = ROOT / "data" / args.dst
    if not src.is_dir():
        print(f"no such run: {src}", file=sys.stderr)
        return 2
    if src.resolve() == dst.resolve():
        print("--from and --to are the same run", file=sys.stderr)
        return 2
    dst.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in CARRY:
        a, b = src / name, dst / name
        if not a.exists():
            continue
        if b.exists() and not args.force:
            print(f"  skip {name} (already in {dst.name}; --force to replace)")
            continue
        if b.exists():
            shutil.rmtree(b) if b.is_dir() else b.unlink()
        shutil.copytree(a, b) if a.is_dir() else shutil.copy2(a, b)
        copied.append(name)

    print(f"copied {', '.join(copied) or 'nothing'} -> data/{args.dst}/")
    print()
    print(f"  export RUN_DIR={args.dst}")
    print( "  caster preflight")
    print( "  caster propose  --policies no-mem --budget 20   # cache hits; writes under data/<run>/propose/")
    print( "  caster audit    --concurrency 4                 # only if kill_matrix.json did not come across")
    print( "  caster evaluate --policies <block A> --from-cache-only --replay-only")
    print()
    print(f"data/{args.src}/ is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
