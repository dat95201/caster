#!/usr/bin/env python3
"""Fetch the ConDefects checkout. The code clones; the test data does not.

`Code/` is a ~125 MB git clone. `Test/` is a 6.4 GB archive distributed out of
band, and that asymmetry is the whole reason this script exists as more than one
`git clone` line: **a partial Test/ tree is the one failure that corrupts a
corpus quietly.** A truncated download leaves some tasks with fewer probes than
the contest shipped, G3 then admits or rejects them on a pool size that is an
artefact of the download, and nothing in the run says so.

So this script separates the two, verifies what it can, and refuses to pretend
the second half is present when it is not. `caster freeze` writes `pool_sha256`
over the probe lists it actually found, which is the backstop: two freezes over
two different downloads produce two different digests.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

REPO = "https://github.com/appmlk/ConDefects.git"
DEFAULT_ROOT = pathlib.Path(os.environ.get("CONDEFECTS_ROOT", "external/ConDefects"))


def run(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def clone(root: pathlib.Path) -> None:
    if (root / "Code").is_dir():
        print(f"{root}/Code already present — leaving it alone.")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and any(root.iterdir()):
        run("git", "-C", str(root), "pull", "--ff-only")
    else:
        run("git", "clone", "--depth", "1", REPO, str(root))


def unpack(root: pathlib.Path, archive: pathlib.Path) -> None:
    dest = root / "Test"
    if dest.is_dir() and any(dest.iterdir()):
        print(f"{dest} already present — leaving it alone. Delete it to re-unpack.")
        return
    if not archive.is_file():
        raise SystemExit(
            f"{archive} not found.\n\n"
            f"Test.zip is a 6.4 GB out-of-band download and is NOT in the git\n"
            f"repository. Get it from the ConDefects release page, put it\n"
            f"somewhere this machine can read, and pass --test-zip.\n\n"
            f"On Colab, keep it on Drive and point --test-zip at it: a runtime is\n"
            f"reclaimed in hours and re-downloading 6.4 GB each session is the\n"
            f"single largest avoidable cost in the pipeline."
        )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"unpacking {archive} -> {dest} (this takes minutes)", flush=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest.parent if _has_top_level_test(z) else dest)


def _has_top_level_test(z: zipfile.ZipFile) -> bool:
    """Some releases wrap the tree in `Test/`, some do not."""
    return any(name.split("/")[0] == "Test" for name in z.namelist()[:64])


def audit(root: pathlib.Path) -> int:
    """Report what is present. The numbers are the point, not the checkmarks."""
    code, test = root / "Code", root / "Test"
    env_test = os.environ.get("CONDEFECTS_TEST_DIR", "").strip()
    if env_test:
        test = pathlib.Path(env_test)

    faults = list(code.glob("*/Python/*/faultyVersion.py")) if code.is_dir() else []
    with_ref = [p for p in faults if (p.parent / "correctVersion.py").is_file()]
    groups = {p.parent.parent.parent.name for p in with_ref}
    contests = sorted(d.name for d in test.iterdir()
                      if d.is_dir() and not d.name.startswith(".")) if test.is_dir() else []

    print()
    print(f"  Code/   {len(with_ref)} Python faults over {len(groups)} coding tasks")
    print(f"  Test/   {len(contests)} contest directories at {test}")

    if not with_ref:
        print(f"\n  Code/ is missing or empty at {code}.")
        return 1
    if not contests:
        print(
            f"\n  Test/ is missing at {test}.\n"
            f"  Every fault will fail G2 ('no test data'), so `caster freeze`\n"
            f"  would admit nothing. Pass --test-zip, or set CONDEFECTS_TEST_DIR\n"
            f"  if you salvaged a partial tree elsewhere."
        )
        return 1

    # The partial-download check. A fault whose contest directory is absent is
    # not a gate failure to be discovered 40 minutes into a freeze.
    missing = sorted({
        p.parent.parent.parent.name.partition("_")[0] for p in with_ref
    } - set(contests))
    if missing:
        print(
            f"\n  {len(missing)} contest(s) in Code/ have no directory under Test/:\n"
            f"    {', '.join(missing[:12])}{' …' if len(missing) > 12 else ''}\n"
            f"  That is what a truncated Test.zip looks like — and also what ONE\n"
            f"  SHARD of `scripts/split_test_shards.py` looks like, which this\n"
            f"  check cannot tell apart.\n"
            f"\n"
            f"  If you unpacked a Test_shardN.zip, this is expected: pass that\n"
            f"  shard's `--first/--last/--shard` to `caster freeze`. With a shard\n"
            f"  of the test tree the range is NOT an optimisation, it is what\n"
            f"  keeps the gate from recording every other fault as a G2 reject.\n"
            f"\n"
            f"  Otherwise the download is incomplete. Those faults would be\n"
            f"  excluded by G2 with reason 'no test data' — which is correct, but\n"
            f"  the resulting corpus is a property of your download, not of the\n"
            f"  benchmark. Finish the download before freezing."
        )
        return 1

    print("\n  Both halves present. Next: `caster preflight`, then `caster freeze`.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    ap.add_argument("--test-zip", type=pathlib.Path, default=None,
                    help="path to Test.zip (6.4 GB, out of band)")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    if not args.audit_only:
        if shutil.which("git") is None:
            raise SystemExit("no `git` on PATH")
        clone(args.root)
        if args.test_zip is not None:
            unpack(args.root, args.test_zip)
    return audit(args.root)


if __name__ == "__main__":
    sys.exit(main())
