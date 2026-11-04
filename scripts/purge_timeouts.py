#!/usr/bin/env python3
"""Drop quarantined `timeout` verdicts from the verdict store.

A stored `timeout` is NEVER served -- `VerdictStore.get()` returns None for it
and `--from-cache-only` cannot be satisfied by it (store.py's module
docstring). The only thing such an entry can still do is arm the stable-
disagreement halt in `verify._measure`. On a shared / loaded box a wall-clock
timeout is not reproducible, so dropping these entries loses no information the
run uses and removes a halt that is an artifact of machine load.

    python3 purge_timeouts.py                      # dry run, whole store
    python3 purge_timeouts.py --protocol 2dcd47fd  # one protocol only
    python3 purge_timeouts.py --patch 0ad7c8a537a5 --probe 003.txt
    python3 purge_timeouts.py --apply              # actually write
"""
from __future__ import annotations
import argparse, json, pathlib, sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="verdicts", help="verdict store root")
ap.add_argument("--protocol", default=None, help="protocol sha prefix")
ap.add_argument("--patch", default=None, help="patch sha prefix")
ap.add_argument("--probe", default=None, help="exact case name, e.g. 003.txt")
ap.add_argument("--apply", action="store_true", help="write; default is dry run")
a = ap.parse_args()

root = pathlib.Path(a.root)
if not root.is_dir():
    sys.exit(f"no verdict store at {root.resolve()}")

files = scanned = dropped = 0
touched: list[str] = []
for proto in sorted(p for p in root.iterdir() if p.is_dir()):
    if a.protocol and not proto.name.startswith(a.protocol):
        continue
    for shard in sorted(p for p in proto.iterdir() if p.is_dir()):
        for f in sorted(shard.glob("*.json")):
            if a.patch and not f.stem.startswith(a.patch):
                continue
            files += 1
            try:
                blob = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            keep, lost = {}, []
            for case, rec in blob.items():
                scanned += 1
                is_to = isinstance(rec, dict) and rec.get("verdict") == "timeout"
                if is_to and (a.probe is None or case == a.probe):
                    lost.append(case)
                else:
                    keep[case] = rec
            if not lost:
                continue
            dropped += len(lost)
            touched.append(f"{proto.name}/{f.stem[:12]}  ->  {', '.join(lost[:6])}"
                           + (f" (+{len(lost)-6})" if len(lost) > 6 else ""))
            if a.apply:
                if keep:
                    tmp = f.with_suffix(".tmp")
                    tmp.write_text(json.dumps(keep), encoding="utf-8")
                    tmp.replace(f)
                else:
                    f.unlink()

for line in touched[:25]:
    print("  " + line)
if len(touched) > 25:
    print(f"  ... and {len(touched)-25} more patch files")
print(f"\n{'APPLIED' if a.apply else 'DRY RUN'}: {files} patch files, {scanned} verdicts scanned, "
      f"{dropped} timeout entries {'dropped' if a.apply else 'would be dropped'} "
      f"across {len(touched)} files")
if not a.apply and dropped:
    print("re-run with --apply to write")
