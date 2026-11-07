#!/usr/bin/env python3
"""Fail if a number printed in Chapter 6 has drifted from the analysis output.

    python3 verify_pilot_numbers.py

Every claim below was typed by hand into `chapters/06-results.tex`. This script
re-reads `data/pilot_v1.json` and asserts the match, so a re-run that moves a
number cannot leave a stale one in the report.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
D  = json.load(open(os.path.join(HERE, '..', 'data', 'pilot_v1.json')))
TEX = open(os.path.join(HERE, '..', 'chapters', '06-results.tex')).read()
ABS = open(os.path.join(HERE, '..', 'frontmatter', 'abstract.tex')).read()

fails: list[str] = []

def check(name, got, want, tol=0.05):
    if abs(got - want) > tol:
        fails.append(f'{name}: analysis says {got:.4g}, report says {want:.4g}')

def in_tex(name, needle, where=TEX):
    if needle not in where:
        fails.append(f'{name}: {needle!r} not found in the chapter')

A = D['arms']; H = {(h['base'], h['new']): h for h in D['head_to_head']}
L = {(h['base'], h['new']): h for h in D['ladder']}

# ---- the head-to-head table ------------------------------------------------
for arm, cpu, ep, ex, ce in [
        ('natural', 7383, 25.37, 7.16, 0.298), ('random', 8331, 28.63, 6.08, 0.397),
        ('coverage', 5420, 18.63, 5.78, 0.271), ('frtp', 2979, 10.24, 2.15, 0.401),
        ('sched', 2893, 9.94, 2.43, 0.344)]:
    check(f'{arm} total CPU',   A[arm]['cpu_total'], cpu, 1.0)
    check(f'{arm} CPU/session', A[arm]['cpu_per_episode'], ep, 0.01)
    check(f'{arm} runs/round',  A[arm]['exec_per_round'], ex, 0.01)
    check(f'{arm} CPU/run',     A[arm]['cpu_per_exec'], ce, 0.001)

for (b, n), cpu, lo, hi, ex in [
        (('natural', 'sched'),  -60.8, -73.3, -36.3, -66.0),
        (('random', 'sched'),   -65.3, -77.4, -45.3, -59.9),
        (('coverage', 'sched'), -46.6, -62.3, -20.4, -57.9),
        (('frtp', 'sched'),      -2.9, -11.0,  +4.9, +13.1),
        (('natural', 'frtp'),   -59.7, -73.3, -33.0, -70.0),
        (('natural', 'coverage'), -26.6, -56.7, +7.4, -19.3),
        (('natural', 'random'), +12.8, -36.8, +97.6, -15.2)]:
    h = H[(b, n)]
    check(f'{n} vs {b} dCPU', h['cpu_pct'], cpu, 0.06)
    check(f'{n} vs {b} CI lo', h['ci'][0], lo, 0.6)
    check(f'{n} vs {b} CI hi', h['ci'][1], hi, 0.6)
    check(f'{n} vs {b} dexec', h['exec_pct'], ex, 0.1)

# ---- the frtp split --------------------------------------------------------
h = H[('frtp', 'sched')]
check('frtp split: same',    h['same'], 34, 0)
check('frtp split: frtp win', h['worse'], 45, 0)
check('frtp split: sched win', h['cheaper'], 18, 0)
check('frtp split: saved',   h['saved_on_wins'], 188, 1.0)
check('frtp split: lost',    h['lost_on_losses'], 102, 1.0)

# ---- the ladder ------------------------------------------------------------
for (b, n), pct in [(('natural', 'cost-only'), -47.8), (('natural', 'kill-only'), -59.1),
                    (('natural', 'frtp'), -59.7), (('frtp', 'kill-only'), +1.4),
                    (('cost-only', 'sched'), -25.0), (('kill-only', 'sched'), -4.2)]:
    check(f'ladder {b}->{n}', L[(b, n)]['cpu_pct'], pct, 0.06)

# ---- the by-block contrast -------------------------------------------------
for i, (pct, lo, hi, base, new, cheap, diff) in enumerate([
        (+2.1,  -1.5, +7.5, 1444.8, 1474.5, 23, 62),
        (-5.2, -16.6, +5.0,  604.4,  573.1, 14, 55),
        (-10.6, -23.0, +6.2, 434.1,  388.3, 12, 49),
        (-7.8, -19.2, +3.8,  495.4,  456.8, 14, 49)]):
    b = D['by_block']['sched_vs_frtp'][i]
    check(f'block {i} dCPU', b['cpu_pct'], pct, 0.06)
    check(f'block {i} CI lo', b['ci'][0], lo, 0.5)
    check(f'block {i} CI hi', b['ci'][1], hi, 0.5)
    check(f'block {i} frtp s', b['base_cpu'], base, 0.1)
    check(f'block {i} sched s', b['new_cpu'], new, 0.1)
    check(f'block {i} cheaper', b['cheaper'], cheap, 0)
    check(f'block {i} differing', b['differing'], diff, 0)

import numpy as _np
_d = _np.array(D['cumulative']['sched']) - _np.array(D['cumulative']['frtp'])
check('crossover round', next(i + 1 for i in range(1, 20) if _d[i] <= 0 < _d[i - 1]), 10, 0)
check('peak deficit', _d.max(), 34, 0.5)
check('final surplus', _d[-1], -86, 0.5)
for arm, cum in [('natural', 7383), ('random', 8331), ('coverage', 5420),
                 ('frtp', 2979), ('sched', 2893)]:
    check(f'cumulative {arm}', D['cumulative'][arm][-1], cum, 1.0)

# ---- the corpus table ------------------------------------------------------
C = D['corpus']
check('cost median', C['cost_median'], 0.036, 0.001)
check('cost p90', C['cost_p90'], 1.41, 0.01)
check('cost p99', C['cost_p99'], 6.32, 0.01)
check('cost max', C['cost_max'], 10.14, 0.01)
check('within-task CV', C['cost_cv_within_task_median'], 0.159, 0.001)
check('kill mean', C['kill_mean'], 0.376, 0.001)
check('never-kill share', C['never_kill_share'] * 100, 31.2, 0.1)
check('corr', C['corr_logcost_kill_median'], 0.11, 0.01)
check('complete tasks', C['complete'], 97, 0)
check('tests costed', C['tests_costed'], 2378, 0)
check('pool total', C['pool_total'], 2521, 0)
check('patches', C['patches'], 1167, 0)
check('p99 / median ratio', C['cost_p99'] / C['cost_median'], 177, 1.0)

# ---- the headline numbers also appear in the abstract ----------------------
for needle in ('60.8', '59.7', '2.9', '2.1', '10.6'):
    if needle not in ABS:
        fails.append(f'abstract: {needle!r} missing')

if fails:
    print('MISMATCH:'); [print(' ', f) for f in fails]; sys.exit(1)
print('all pilot numbers in Chapter 6 and the abstract match the analysis output')
