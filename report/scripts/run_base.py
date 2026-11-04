import json, statistics, time
from arena import build
from sim import simulate

A = build()
PREP = {}
for t, (pool, patches, cost, cov) in A.items():
    lines = sorted(set().union(*cov.values()))
    bit = {l: 1 << i for i, l in enumerate(lines)}
    cb = [sum(bit[l] for l in cov[x]) for x in pool]
    U = (1 << len(lines)) - 1
    PREP[t] = (pool, patches, cost, cb, U)

SEEDS = (1, 2, 3, 4, 5)


def ev(policy, **kw):
    per = {}
    for t, (pool, patches, cost, cb, U) in PREP.items():
        r = [simulate(pool, patches, cost, cb, U, policy, s, **kw) for s in SEEDS]
        per[t] = (statistics.mean(x[0] for x in r),
                  statistics.mean(x[1] for x in r) / len(patches))
    return per


if __name__ == "__main__":
    res = {}
    t0 = time.time()
    for p in ["natural", "random", "kill-only", "cost-only", "coverage", "km", "km-cov"]:
        per = ev(p)
        tot = sum(v[0] for v in per.values())
        pr = statistics.mean(v[1] for v in per.values())
        res[p] = {"total_cpu": tot, "probes_per_round": pr,
                  "per_task": {k: v[0] for k, v in per.items()}}
        print("%-10s  CPU %9.1f s   probes/round %5.2f   (%.0fs)"
              % (p, tot, pr, time.time() - t0))
    json.dump(res, open("baselines.json", "w"))
