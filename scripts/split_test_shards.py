#!/usr/bin/env python3
"""Cut `Test/` into one archive per freeze shard.

`Test/` is 6.4 GB. A cloud runner that only walks faults 717–1432 has no use
for the other three quarters, and downloading and unpacking the whole tree to
reach them is most of the wall clock of a short shard.

The cut is **not** a quarter of the directories. `caster freeze` shards over
FAULTS (`abc265_d/45075464`), sorted; `Test/` is organised by CODING TASK
(`Test/abc265/D`), and several faults share one task's test data. So this walks
the same candidate list `freeze` walks, slices it the same way, and packages
the test directories the faults in each slice actually need. A task straddling
a boundary lands in both archives, which costs a few megabytes and keeps every
archive self-sufficient.

Both halves of that come from the adapter — `tasks()` for the candidate order
and `test_root()` for the lookup — so this cannot drift from what the gate
expects to find.

    python3 scripts/split_test_shards.py --shards 4

writes `external/ConDefects/Test_shard0.zip` … `Test_shard3.zip`, each
unpacking to `Test/...` exactly like the original:

    unzip -q Test_shard1.zip -d external/ConDefects/
    caster freeze --first 717 --last 1432 --shard 1
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from caster.bench.adapter import ConDefectsAdapter  # noqa: E402


def ranges(n: int, shards: int) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping, covering 1..n exactly, remainder first.

    The same arithmetic `scripts/fleet.sh` uses, for the same reason: a
    hand-typed range is off by one eventually, and neither mistake announces
    itself — a skipped fault turns up as a gap at `--assemble`, an overlapped
    one is silently redone work whose only symptom is time.
    """
    base, extra = divmod(n, shards)
    out, lo = [], 1
    for i in range(shards):
        size = base + (1 if i < extra else 0)
        if size == 0:
            break
        out.append((lo, lo + size - 1))
        lo += size
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--condefects-root", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="defaults to the ConDefects root, beside Test/")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-make an archive that already exists")
    ap.add_argument("--store", action="store_true",
                    help="no compression — faster to write, much bigger to move")
    args = ap.parse_args()

    ad = ConDefectsAdapter(root=args.condefects_root)
    root = ad.root
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    names = sorted(ad.tasks())
    if not names:
        print(ad.setup_hint, file=sys.stderr)
        return 1
    print(f"{len(names)} candidates  ({time.time() - t0:.0f}s)", flush=True)
    print("   this must equal `caster freeze --dry-run`; if it does not, the "
          "archives would be cut for a different corpus.", flush=True)

    method = zipfile.ZIP_STORED if args.store else zipfile.ZIP_DEFLATED
    rows = []
    for shard, (lo, hi) in enumerate(ranges(len(names), args.shards)):
        chunk = names[lo - 1:hi]
        groups = sorted({n.split("/")[0] for n in chunk})
        dirs, missing = [], []
        for g in groups:
            d = ad.test_root(g)
            (dirs if d is not None else missing).append(d if d is not None else g)

        out = out_dir / f"Test_shard{shard}.zip"
        if out.exists() and not args.overwrite:
            print(f"[s{shard}] {out.name} exists — skipping (--overwrite to redo)",
                  flush=True)
            rows.append((shard, lo, hi, len(chunk), len(dirs),
                         out.stat().st_size / 2 ** 30, len(missing)))
            continue

        print(f"[s{shard}] faults {lo}-{hi}: {len(groups)} coding tasks, "
              f"{len(dirs)} test dirs"
              + (f", {len(missing)} WITHOUT test data" if missing else ""),
              flush=True)
        t1 = time.time()
        tmp = out.with_suffix(".zip.part")
        n_files = 0
        with zipfile.ZipFile(tmp, "w", method, allowZip64=True) as z:
            for i, d in enumerate(dirs, 1):
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(root))
                        n_files += 1
                if i % 50 == 0 or i == len(dirs):
                    print(f"   {i}/{len(dirs)} dirs, {n_files} files, "
                          f"{time.time() - t1:.0f}s", flush=True)
        tmp.replace(out)        # whole or not at all
        size = out.stat().st_size / 2 ** 30
        print(f"[s{shard}] {out.name}  {size:.2f} GB  ({time.time() - t1:.0f}s)",
              flush=True)
        rows.append((shard, lo, hi, len(chunk), len(dirs), size, len(missing)))

    print("\n" + "=" * 72)
    for shard, lo, hi, nf, nd, size, nmiss in rows:
        print(f"Test_shard{shard}.zip   {size:5.2f} GB   {nf} faults, {nd} test dirs"
              + (f"   ({nmiss} tasks have no test data)" if nmiss else ""))
    print("=" * 72)
    print("\nOn the runner, per shard:\n")
    for shard, lo, hi, *_ in rows:
        print(f"  unzip -q Test_shard{shard}.zip -d external/ConDefects/ && \\\n"
              f"  caster freeze --first {lo} --last {hi} --shard {shard}")
    print("\nThen, once every shard's `pool/gate/sNN.jsonl` is back in one "
          "place:\n\n  caster freeze --assemble\n")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
