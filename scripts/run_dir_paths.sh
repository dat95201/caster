#!/usr/bin/env bash
# Where this run's artifacts live — the bash half of `caster/paths.py`.
#
# Source it; do not execute it:
#
#     . scripts/run_dir_paths.sh
#     echo "$RUN_DATA"      # data  or  data/<RUN_DIR>
#
# This file and `caster/paths.py` must agree, and `tests/test_invariants.py
# ::test_bash_and_python_agree_on_the_run_dir` says so. Two resolution bases for
# one concept is how a run writes half its output to one directory and half to
# another without erroring (Invariant 5).
#
# The slug is stripped of surrounding slashes so RUN_DIR=/foo/ and RUN_DIR=foo
# agree, and so a leading slash cannot escape the repo.

_caster_slug="${RUN_DIR:-}"
_caster_slug="${_caster_slug#"${_caster_slug%%[!/]*}"}"   # strip leading  /
_caster_slug="${_caster_slug%"${_caster_slug##*[!/]}"}"   # strip trailing /
_caster_slug="$(printf '%s' "$_caster_slug" | tr -d '[:space:]')"

if [ -n "$_caster_slug" ]; then
  RUN_DATA="data/$_caster_slug"
  RUN_LOGS="logs/$_caster_slug"
else
  RUN_DATA="data"
  RUN_LOGS="logs"
fi

# Cross-run stores. Deliberately NOT under RUN_DIR: the response cache is what
# makes a re-run free (Invariant 8), and the verdict and patch stores are its
# CPU-side siblings.
RUN_CACHE="${CACHE_DIR:-cache}"
RUN_VERDICTS="${VERDICT_DIR:-verdicts}"
RUN_PATCHES="${PATCH_DIR:-patches}"

export RUN_DATA RUN_LOGS RUN_CACHE RUN_VERDICTS RUN_PATCHES
unset _caster_slug
