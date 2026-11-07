#!/usr/bin/env python3
"""Price the within-round coverage check on its own, three ways.

The scheduler's last clause asks, inside one round: *is this test's rejection
record already contained in the record of a test that has just PASSED on this
candidate?*  If it is, the test is deferred to the back of the round.

The clause is cheap to state and expensive to feed, so the question is how much
evidence it has to work with.  Three settings, same tasks, same candidates:

    off        the clause disabled (tau = 0)
    session    the clause on, fed only by what this session has seen
    seeded     the clause on, fed from the audit pass's complete rejection sets

`session` is what the pilot's `sched` arm actually runs; `seeded` is the
headroom that a precomputed table would buy, and is NOT implementable online.

    PYTHONPATH=<repo> python3 coverage_check.py [run-dir]
    -> writes ../data/coverage_check.json
"""
from __future__ import annotations
import json, os, random, sys
from collections import defaultdict
import numpy as np

from caster.exec.ledger import CaseLedger
from caster.run import policies

HERE = os.path.dirname(os.path.abspath(__file__))
RUN  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'data', 'pilot-v1')
OUT  = os.path.join(HERE, '..', 'data', 'coverage_check.json')

rows   = [json.loads(l) for l in open(os.path.join(RUN, 'rounds.jsonl'))]
killm  = json.load(open(os.path.join(RUN, 'kill_matrix.json')))
tasksj = json.load(open(os.path.join(RUN, 'pool', 'tasks.json')))['tasks']

# the frozen per-test cost table, recovered from the replayed rounds
eqs = defaultdict(set)
for r in rows:
    if r['scan_order']: eqs[r['task']].add((tuple(r['scan_order']), round(r['cpu_s'], 9)))
cost = defaultdict(dict)
for t, es in eqs.items():
    es = sorted(es)          # a set's iteration order is not stable across
                             # processes, and lstsq on this system is
                             # ill-conditioned enough for the row order to
                             # move the solution in the third decimal.
    names = sorted({p for so, _ in es for p in so}); idx = {p: i for i, p in enumerate(names)}
    A = np.zeros((len(es), len(names))); y = np.zeros(len(es))
    for i, (so, c) in enumerate(es):
        for p in so: A[i, idx[p]] += 1
        y[i] = c
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    for p, i in idx.items(): cost[t][p] = float(max(x[i], 0.0))

# the candidate sequence each (task, seed) actually drew, under CRN
seq = defaultdict(list)
for r in sorted(rows, key=lambda z: (z['task'], z['seed'], z['round_index'])):
    if r['policy'] == 'km': seq[(r['task'], r['seed'])].append(r['patch_sha'])

# the audit's complete rejection sets, per task
audit_rej = {}
for t, patches in killm.items():
    R = defaultdict(set)
    for sha, v in patches.items():
        for probe, verdict in v.items():
            if verdict != 'pass': R[probe].add(sha)
    audit_rej[t] = {p: s for p, s in R.items() if s}


def ctx_for(led, pool, ri, task, seed):
    return policies.PolicyContext(ledger=led, pool=pool, round_index=ri,
                                  cell_key=f'{task}|{seed}', coverage={})


def replay(task, seed, shas, mode):
    pool = tuple(tasksj[task]['p_loop'])
    thr  = 0.0 if mode == 'off' else 0.75
    led  = CaseLedger(pool, prior='shipped', cold_start='max', explore_p=0.15,
                      overlap_thr=thr, overlap_min_rej=1,
                      tie_key=lambda n: pool.index(n) if n in pool else len(pool),
                      rng=random.Random(f'{task}|{seed}'))
    if mode == 'seeded':
        led.seed_overlap({p: s for p, s in audit_rej.get(task, {}).items() if p in pool})
    cpu = 0.0; calls = 0; defers = 0
    for ri, sha in enumerate(shas, 1):
        verdicts = killm.get(task, {}).get(sha, {})
        # the deferral walk of `caster.exec.verify.scan`: the overlap rule
        # reorders inside the round's queue and never removes from it, and a
        # test may be pushed back at most once.
        pending = list(policies.schedule('km', ctx_for(led, pool, ri, task, seed)))
        passed, deferred_once, seen = [], set(), set()
        defer_on = thr > 0.0
        while pending:
            probe = None
            if defer_on and passed:
                for i, cand in enumerate(pending):
                    if cand in deferred_once: continue
                    if led.redundant(cand, passed):
                        deferred_once.add(cand); defers += 1; continue
                    probe = pending.pop(i); break
            if probe is None: probe = pending.pop(0)
            if probe in seen: continue
            seen.add(probe)
            status = verdicts.get(probe)
            if status is None: continue
            c = float(cost.get(task, {}).get(probe, 0.0))
            cpu += c; calls += 1
            led.update(probe, refuted=status != 'pass', cpu_s=c, round_index=ri)
            if status != 'pass': break
            passed.append(probe)
    return cpu, calls, defers


tasks = sorted({t for t, _ in seq})
per  = {m: defaultdict(float) for m in ('off', 'session', 'seeded')}
pere = {m: defaultdict(int)   for m in ('off', 'session', 'seeded')}
perd = {m: 0 for m in ('off', 'session', 'seeded')}
for (t, s), shas in seq.items():
    for m in ('off', 'session', 'seeded'):
        c, e, d = replay(t, s, shas, m)
        per[m][t] += c; pere[m][t] += e; perd[m] += d

RNG = np.random.default_rng(20260101)
def boot(a, b, iters=10000):
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    pt = (b.sum() - a.sum()) / a.sum() * 100
    i = RNG.integers(0, n, (iters, n)); sa = a[i].sum(1); sb = b[i].sum(1)
    d = (sb - sa) / sa * 100
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

base = [per['off'][t] for t in tasks]
out = {'tasks': len(tasks), 'episodes': len(seq), 'modes': {}}
for m in ('off', 'session', 'seeded'):
    pt, lo, hi = boot(base, [per[m][t] for t in tasks])
    out['modes'][m] = dict(cpu=sum(per[m].values()), execs=sum(pere[m].values()),
                           deferrals=perd[m], pct_vs_off=pt, ci=[lo, hi])
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
