#!/usr/bin/env python3
"""Everything the report prints about the Block A pilot, from one pass.

    python3 pilot_v1.py <run-dir>        # default: ../../data/pilot-v1
    -> writes ../data/pilot_v1.json

The run directory is a finished `caster merge` output: `rounds.jsonl`,
`kill_matrix.json` and `pool/tasks.json`.  Nothing here reads the model, the
sandbox or the benchmark; it is arithmetic over rows that already exist.
"""
from __future__ import annotations
import json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'data', 'pilot-v1')
OUT  = os.path.join(HERE, '..', 'data', 'pilot_v1.json')

#: the five configurations the report compares head to head, under the names
#: the report uses for them.
ALIAS = {'natural': 'natural', 'random': 'random', 'fault-record': 'frtp',
         'coverage': 'coverage', 'km': 'sched',
         'cost-only': 'cost-only', 'kill-only': 'kill-only', 'km-ov0': 'sched-ov0'}
HEAD = ['natural', 'random', 'coverage', 'fault-record', 'km']

rows   = [json.loads(l) for l in open(os.path.join(RUN, 'rounds.jsonl'))]
tasksj = json.load(open(os.path.join(RUN, 'pool', 'tasks.json')))
killm  = json.load(open(os.path.join(RUN, 'kill_matrix.json')))
SEEDS  = sorted({r['seed'] for r in rows})

# ----------------------------------------------------------------- cells ----
cpu   = defaultdict(float); calls = defaultdict(int)
rounds_n = defaultdict(int); accept_at = {}
for r in rows:
    k = (r['policy'], r['task'], r['seed'])
    cpu[k] += r['cpu_s']; calls[k] += r['test_calls']; rounds_n[k] += 1
    if r['accept'] and k not in accept_at: accept_at[k] = r['round_index']

POLS   = sorted({p for p, _, _ in cpu})
SEEN   = sorted({t for _, t, _ in cpu})
COMMON = [t for t in SEEN if all((p, t, s) in cpu for p in POLS for s in SEEDS)]

tot  = {p: {t: sum(cpu[(p, t, s)]   for s in SEEDS) for t in COMMON} for p in POLS}
tote = {p: {t: sum(calls[(p, t, s)] for s in SEEDS) for t in COMMON} for p in POLS}

# ------------------------------------------------- bootstrap and wilcoxon ---
RNG = np.random.default_rng(20260101)

def boot(a, b, iters=10000):
    """Percentage change from a to b, cluster bootstrap resampling tasks."""
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    pt = (b.sum() - a.sum()) / a.sum() * 100
    i = RNG.integers(0, n, (iters, n)); sa = a[i].sum(1); sb = b[i].sum(1)
    d = (sb - sa) / sa * 100
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def wilcoxon(a, b):
    d = np.asarray(b, float) - np.asarray(a, float); d = d[d != 0]; n = len(d)
    if n < 6: return float('nan')
    ab = np.abs(d); order = np.argsort(ab); srt = ab[order]; rank = np.empty(n)
    i = 0; tie = 0.0
    while i < n:
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]: j += 1
        rank[order[i:j + 1]] = (i + j) / 2.0 + 1
        c = j - i + 1; tie += c ** 3 - c; i = j + 1
    W  = rank[d > 0].sum()
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    return float(math.erfc(abs((W - mu) / sd) / math.sqrt(2)))

def holm(ps):
    m = len(ps); out = [0.0] * m; prev = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: ps[i])):
        prev = max(min(1.0, (m - rank) * ps[i]), prev); out[i] = prev
    return out

def contrast(base, new):
    a = [tot[base][t]  for t in COMMON]; b = [tot[new][t]  for t in COMMON]
    x = [tote[base][t] for t in COMMON]; y = [tote[new][t] for t in COMMON]
    pt, lo, hi = boot(a, b); ept, elo, ehi = boot(x, y)
    d = np.asarray(b) - np.asarray(a)
    return dict(base=ALIAS[base], new=ALIAS[new], n=len(COMMON),
                cpu_pct=pt, ci=[lo, hi], exec_pct=ept, exec_ci=[elo, ehi],
                p=wilcoxon(a, b),
                cheaper=int((d < 0).sum()), worse=int((d > 0).sum()),
                same=int((d == 0).sum()),
                saved_on_wins=float(-d[d < 0].sum()), lost_on_losses=float(d[d > 0].sum()))

# ------------------------------------------------------------------ arms ----
arms = {}
for p in POLS:
    c = sum(tot[p].values()); e = sum(tote[p].values())
    nr = sum(rounds_n[(p, t, s)] for t in COMMON for s in SEEDS)
    ne = len(COMMON) * len(SEEDS)
    acc = [accept_at[(p, t, s)] for t in COMMON for s in SEEDS if (p, t, s) in accept_at]
    arms[ALIAS.get(p, p)] = dict(
        cpu_total=c, cpu_per_episode=c / ne, cpu_per_round=c / nr,
        exec_total=e, exec_per_round=e / nr, cpu_per_exec=c / e,
        rounds=nr, episodes=ne, success_at_b=len(acc) / ne,
        first_accept_mean=float(np.mean(acc)) if acc else None)

PAIRS = [('natural', 'km'), ('random', 'km'), ('coverage', 'km'), ('fault-record', 'km'),
         ('natural', 'fault-record'), ('natural', 'coverage'), ('natural', 'random')]
head = [contrast(a, b) for a, b in PAIRS]
for h, p in zip(head, holm([h['p'] for h in head])): h['p_holm'] = p

LADDER = [('natural', 'cost-only'), ('natural', 'kill-only'), ('natural', 'fault-record'),
          ('fault-record', 'kill-only'), ('kill-only', 'km'), ('cost-only', 'km')]
ladder = [contrast(a, b) for a, b in LADDER]

# ------------------------------------------ contrast by block of rounds -----
# `K` converges on a handful of observations; `m-hat` needs more, so the cost
# term cannot act until the cost table fills in. Splitting the budget into
# blocks is what makes that visible -- the whole-session total averages the
# early rounds, where the cost term has nothing, together with the late ones.
BLOCKS = [(1, 5), (6, 10), (11, 15), (16, 20)]
blk = defaultdict(lambda: defaultdict(float))
for r in rows:
    if r['task'] not in set(COMMON): continue
    for i, (a, b) in enumerate(BLOCKS):
        if a <= r['round_index'] <= b: blk[(r['policy'], i)][r['task']] += r['cpu_s']

def block_contrast(base, new_):
    out = []
    for i, (a, b) in enumerate(BLOCKS):
        x = [blk[(base, i)][t] for t in COMMON]; y = [blk[(new_, i)][t] for t in COMMON]
        pt, lo, hi = boot(x, y)
        d = np.asarray(y) - np.asarray(x)
        out.append(dict(first=a, last=b, base_cpu=float(sum(x)), new_cpu=float(sum(y)),
                        cpu_pct=pt, ci=[lo, hi],
                        cheaper=int((d < 0).sum()), differing=int((d != 0).sum())))
    return out

by_block = {f'sched_vs_{ALIAS[b]}': block_contrast(b, 'km')
            for b in ('fault-record', 'coverage', 'natural')}

# ----------------------------------------------------------- round curve ----
curve_cpu = defaultdict(lambda: defaultdict(list)); curve_ex = defaultdict(lambda: defaultdict(list))
cs = set(COMMON)
for r in rows:
    if r['task'] in cs:
        curve_cpu[r['policy']][r['round_index']].append(r['cpu_s'])
        curve_ex[r['policy']][r['round_index']].append(r['test_calls'])
B = max(r['round_index'] for r in rows)
curve = {ALIAS[p]: dict(
            cpu=[float(np.mean(curve_cpu[p][i])) if curve_cpu[p].get(i) else None for i in range(1, B + 1)],
            ex =[float(np.mean(curve_ex[p][i]))  if curve_ex[p].get(i)  else None for i in range(1, B + 1)])
         for p in HEAD}

# ------------------------------- the frozen per-test cost table, recovered ---
# In replay a round's CPU is the sum of the frozen costs of the tests it ran,
# so the per-test table is the least-squares solution of those equations.
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

kill = defaultdict(dict)
for t, patches in killm.items():
    tally = defaultdict(lambda: [0, 0])
    for sha, v in patches.items():
        for probe, verdict in v.items():
            tally[probe][1] += 1
            if verdict != 'pass': tally[probe][0] += 1
    for probe, (k, n) in tally.items(): kill[t][probe] = k / n if n else 0.0

allc = np.array([c for d in cost.values() for c in d.values()])
allk = np.array([v for d in kill.values() for v in d.values()])
cv, corr = {}, []
for t, d in cost.items():
    v = np.array(list(d.values()))
    cv[t] = float(v.std(ddof=1) / v.mean()) if len(v) >= 3 and v.mean() > 0 else 0.0
    ks = kill.get(t, {}); pr = [p for p in d if p in ks]
    if len(pr) >= 5:
        lx = np.log(np.maximum([d[p] for p in pr], 1e-6)); ly = np.array([ks[p] for p in pr])
        if lx.std() > 0 and ly.std() > 0: corr.append(float(np.corrcoef(lx, ly)[0, 1]))
pool = np.array([len(v['p_loop']) for v in tasksj['tasks'].values()])

corpus = dict(
    admitted=len(tasksj['tasks']), ran=len(SEEN), complete=len(COMMON), seeds=len(SEEDS),
    budget=max(r['round_index'] for r in rows),
    cells=len(COMMON) * len(SEEDS) * len(POLS),
    pool_total=int(pool.sum()), pool_median=float(np.median(pool)),
    pool_mean=float(pool.mean()), pool_min=int(pool.min()), pool_max=int(pool.max()),
    patches=int(sum(len(v) for v in killm.values())),
    tests_costed=int(len(allc)),
    cost_mean=float(allc.mean()), cost_median=float(np.median(allc)),
    cost_p90=float(np.percentile(allc, 90)), cost_p99=float(np.percentile(allc, 99)),
    cost_max=float(allc.max()),
    cost_cv_within_task_median=float(np.median([v for v in cv.values() if v > 0])),
    kill_mean=float(allk.mean()), kill_median=float(np.median(allk)),
    never_kill_share=float((allk == 0).mean()),
    corr_logcost_kill_median=float(np.median(corr)), corr_n=len(corr))

# ------------------------------------------- the within-round coverage check -
ep = defaultdict(list)
for r in rows:
    if r['policy'] == 'km': ep[(r['task'], r['seed'])].append(r)
distinct, co_ep, co_pairs = [], 0, 0
pre_rounds = pre_eps = km_rounds = 0
for rs in ep.values():
    refs = defaultdict(set)
    for r in rs:
        if r['killed_by']: refs[r['patch_sha']].add(r['killed_by'])
    distinct.append(len({p for s in refs.values() for p in s}))
    rec = sum(1 for s in refs.values() if len(s) >= 2)
    if rec: co_ep += 1; co_pairs += rec
    # the precondition of the clause: a test that PASSES this round must
    # already have a rejection of its own on record, or the overlap is 0.
    R = defaultdict(set); hit = False
    for r in sorted(rs, key=lambda z: z['round_index']):
        km_rounds += 1
        passed = [p for p in r['scan_order'] if p != r['killed_by']]
        if any(R[p] for p in passed): pre_rounds += 1; hit = True
        if r['killed_by']: R[r['killed_by']].add(r['patch_sha'])
    if hit: pre_eps += 1
distinct = np.array(distinct)

sub = []
for t, patches in killm.items():
    R = defaultdict(set)
    for sha, v in patches.items():
        for probe, verdict in v.items():
            if verdict != 'pass': R[probe].add(sha)
    names = [p for p in R if R[p]]; n = len(names); hit = tot_pairs = 0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            tot_pairs += 1
            if len(R[names[i]] & R[names[j]]) / len(R[names[i]]) >= 0.75: hit += 1
    if tot_pairs: sub.append(hit / tot_pairs)

coverage_check = dict(
    episodes=len(ep), mean_distinct_refuters=float(distinct.mean()),
    share_lt2=float((distinct < 2).mean()),
    episodes_with_a_copair=co_ep, share_with_a_copair=co_ep / len(ep),
    copairs=co_pairs,
    audit_subsumed_share_median=float(np.median(sub)),
    audit_subsumed_share_mean=float(np.mean(sub)),
    rounds=km_rounds, precondition_rounds=pre_rounds,
    precondition_round_share=pre_rounds / km_rounds,
    precondition_episodes=pre_eps, precondition_episode_share=pre_eps / len(ep))

# --------------------------------------------- where the cost term earns it --
cvv = np.array([cv.get(t, 0.0) for t in COMMON])
q = np.quantile(cvv, [1/3, 2/3])
terciles = []
for lab, m in [('low', cvv <= q[0]), ('mid', (cvv > q[0]) & (cvv <= q[1])), ('high', cvv > q[1])]:
    ts = [t for t, mm in zip(COMMON, m) if mm]
    a = sum(tot['fault-record'][t] for t in ts); b = sum(tot['km'][t] for t in ts)
    pt, lo, hi = boot([tot['fault-record'][t] for t in ts], [tot['km'][t] for t in ts])
    terciles.append(dict(band=lab, tasks=len(ts), cv_lo=float(cvv[m].min()), cv_hi=float(cvv[m].max()),
                         frtp=a, sched=b, pct=pt, ci=[lo, hi]))

# cumulative CPU per round, summed over every cell (not a mean)
cum = defaultdict(lambda: np.zeros(B + 1))
for r in rows:
    if r['task'] in set(COMMON): cum[r['policy']][r['round_index']] += r['cpu_s']
cumulative = {ALIAS[p]: np.cumsum(cum[p][1:]).tolist() for p in HEAD}

json.dump(dict(corpus=corpus, arms=arms, head_to_head=head, ladder=ladder,
               round_curve=curve, cumulative=cumulative, by_block=by_block,
               coverage_check=coverage_check,
               cost_spread_terciles=terciles,
               per_task={ALIAS[p]: {t: tot[p][t] for t in COMMON} for p in HEAD},
               per_task_cv={t: cv.get(t, 0.0) for t in COMMON}),
          open(OUT, 'w'), indent=1)
print('wrote', os.path.abspath(OUT))
