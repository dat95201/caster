import json, statistics, numpy as np, time
from arena import build
from run_base import PREP
from sim import simulate

SEEDS = tuple(range(1, 21))
CAND = [
    ("natural (pool order)",        "natural", {}),
    ("random",                      "random",  {}),
    ("coverage (TP_add)",           "coverage", {}),
    ("cost-only  1/m",              "cost-only", {}),
    ("kill-only  K",                "exp", dict(a=1.0, b=0.0, c=0.0)),
    ("km  K/m",                     "exp", dict(a=1.0, b=0.0, c=1.0)),
    ("K/m^0.25",                    "exp", dict(a=1.0, b=0.0, c=0.25)),
    ("K/m^0.5   (proposed)",        "exp", dict(a=1.0, b=0.0, c=0.5)),
    ("K^0.5 g^0.25 /m^0.25",        "exp", dict(a=0.5, b=0.25, c=0.25)),
    ("K g /m^0.5",                  "exp", dict(a=1.0, b=1.0, c=0.5)),
    ("K g /m   (1,1,1)",            "exp", dict(a=1.0, b=1.0, c=1.0)),
]
tasks = sorted(PREP); per = {}
t0=time.time()
for name, pol, kw in CAND:
    row = {}
    for t in tasks:
        pool, patches, cost, cb, U = PREP[t]
        r = [simulate(pool, patches, cost, cb, U, pol, s, **kw) for s in SEEDS]
        row[t] = (statistics.mean(x[0] for x in r),
                  statistics.mean(x[1] for x in r) / len(patches))
    per[name] = row
    print(f"  {name:28s} {time.time()-t0:6.0f}s", flush=True)

rng = np.random.default_rng(7)
km = np.array([per["km  K/m"][t][0] for t in tasks])
nat = sum(per["natural (pool order)"][t][0] for t in tasks)
res = []
for name, _, _ in CAND:
    x = np.array([per[name][t][0] for t in tasks])
    d = (x.sum() - km.sum()) / km.sum() * 100
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(tasks), len(tasks))
        bs.append((x[i].sum() - km[i].sum()) / km[i].sum() * 100)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res.append(dict(policy=name, cpu=float(x.sum()),
                    probes=float(statistics.mean(per[name][t][1] for t in tasks)),
                    vs_natural=float(x.sum()/nat), delta_km=float(d),
                    ci_lo=float(lo), ci_hi=float(hi),
                    wins=int((x < km).sum()), n=len(tasks), sig=bool(hi < 0 or lo > 0)))
json.dump({"head":res,"per_task":{n:{t:per[n][t][0] for t in tasks} for n,_,_ in CAND}}, open("head_verify.json","w"))
print("\n%-26s %10s %7s %8s %9s %16s %6s" % ("policy","CPU s","probes","vs nat","Δ vs km%","95% CI","wins"))
for r in res:
    print("%-26s %10.1f %7.3f %8.4f %+9.2f  [%+6.2f,%+6.2f] %4d %s" % (
        r["policy"],r["cpu"],r["probes"],r["vs_natural"],r["delta_km"],r["ci_lo"],r["ci_hi"],r["wins"],"*" if r["sig"] else ""))
