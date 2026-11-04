"""Two questions the grid cannot answer on its own:
   1. what is the best exponent on cost, holding the rest fixed?
   2. does g add anything AT ALL once a and c are held fixed?
Both are answered paired, same seeds, bootstrapped over tasks."""
import json, statistics, numpy as np
from run_base import PREP
from sim import simulate

SEEDS = tuple(range(1, 21))
tasks = list(PREP)


def ev(a, b, c):
    row = {}
    for t, (pool, patches, cost, cb, U) in PREP.items():
        r = [simulate(pool, patches, cost, cb, U, "exp", s, a=a, b=b, c=c) for s in SEEDS]
        row[t] = (statistics.mean(x[0] for x in r),
                  statistics.mean(x[1] for x in r) / len(patches))
    return row


def boot(x, y, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    X = np.array([x[t][0] for t in tasks]); Y = np.array([y[t][0] for t in tasks])
    d = (X.sum() - Y.sum()) / Y.sum() * 100
    bs = [(X[i].sum() - Y[i].sum()) / Y[i].sum() * 100
          for i in (rng.integers(0, len(tasks), len(tasks)) for _ in range(n))]
    return d, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


res = {"c_sweep": {}, "g_isolation": {}}
print("cost exponent sweep at a=1, b=0   (reference: c=1, i.e. km)")
cs = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25]
rows = {c: ev(1.0, 0.0, c) for c in cs}
km = rows[1.0]
for c in cs:
    tot = sum(v[0] for v in rows[c].values())
    pr = statistics.mean(v[1] for v in rows[c].values())
    d, lo, hi = boot(rows[c], km)
    res["c_sweep"][c] = {"total_cpu": tot, "probes": pr, "delta_vs_km": d,
                         "ci_lo": lo, "ci_hi": hi}
    print("  c=%.3f  CPU %8.1f  probes %5.3f  %+7.1f%%  CI [%+6.1f,%+6.1f]%s"
          % (c, tot, pr, d, lo, hi, "  *" if hi < 0 else ""))

print("\ndoes g help?  paired, same (a,c), b=0 vs b>0")
for a, c in [(1.0, 0.5), (1.0, 0.25), (0.5, 0.25), (1.0, 1.0), (2.0, 0.5)]:
    b0 = ev(a, 0.0, c)
    for b in [0.25, 0.5, 1.0]:
        bb = ev(a, b, c)
        d, lo, hi = boot(bb, b0)
        res["g_isolation"]["a=%.2f,c=%.2f,b=%.2f" % (a, c, b)] = {
            "total_cpu": sum(v[0] for v in bb.values()),
            "delta_vs_b0": d, "ci_lo": lo, "ci_hi": hi}
        print("  a=%.1f c=%.2f  b=%.2f vs b=0: %+6.2f%%  CI [%+6.2f,%+6.2f]%s"
              % (a, c, b, d, lo, hi, "  *" if hi < 0 or lo > 0 else ""))

json.dump(res, open("iso.json", "w"))
