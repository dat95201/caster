"""Paired per-task analysis of the MEASURED engine runs (rounds.csv), restricted
to the 68-task arena. Cluster bootstrap over tasks + Wilcoxon signed-rank."""
import csv, collections, json, statistics
import numpy as np
from run_base import PREP

ARENA = set(PREP)
rows = list(csv.DictReader(open("rounds.csv")))
POL = ["alphabet", "fault-record", "km"]

cpu = collections.defaultdict(lambda: collections.defaultdict(float))
calls = collections.defaultdict(lambda: collections.defaultdict(int))
nr = collections.defaultdict(lambda: collections.defaultdict(int))
accepts = collections.defaultdict(int)
per_round = collections.defaultdict(lambda: collections.defaultdict(list))
allrounds = collections.defaultdict(int)

for r in rows:
    t, p = r["task"], r["policy"]
    allrounds[p] += 1
    if t not in ARENA: continue
    cpu[p][t] += float(r["cpu_s"]); calls[p][t] += int(r["test_calls"]); nr[p][t] += 1
    accepts[p] += int(r["accept"])
    per_round[p][int(r["round"])].append(float(r["cpu_s"]))

tasks = sorted(ARENA & set(cpu["km"]))
print(f"rounds.csv: {len(rows)} rows; rounds/policy overall {dict(allrounds)}")
print(f"arena tasks with measured rounds: {len(tasks)}  rounds/policy in arena: "
      f"{ {p: sum(nr[p].values()) for p in POL} }\n")

def boot(x, y, n=10000, seed=20260920):
    """% difference of x vs y, cluster-bootstrapped over tasks."""
    rng = np.random.default_rng(seed)
    X, Y = np.asarray(x), np.asarray(y)
    d = (X.sum() - Y.sum()) / Y.sum() * 100
    bs = np.empty(n)
    for j in range(n):
        i = rng.integers(0, len(X), len(X))
        bs[j] = (X[i].sum() - Y[i].sum()) / Y[i].sum() * 100
    return d, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def wilcoxon(a, b):
    from scipy.stats import wilcoxon as W
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0): return 1.0
    return float(W(a, b, zero_method="wilcox").pvalue)

V = {p: np.array([cpu[p][t] for t in tasks]) for p in POL}
C = {p: np.array([calls[p][t] for t in tasks]) for p in POL}
R = np.array([nr["km"][t] for t in tasks])

out = {"n_tasks": len(tasks), "rounds_per_policy": int(sum(nr["km"].values())),
       "policies": {}}
print("%-14s %11s %9s %9s %10s" % ("policy", "CPU s", "probes/rd", "CPU/rd", "accepts"))
for p in POL:
    out["policies"][p] = dict(cpu_total=float(V[p].sum()),
                              probes_per_round=float(C[p].sum()/R.sum()),
                              cpu_per_round=float(V[p].sum()/R.sum()),
                              accepts=accepts[p])
    print("%-14s %11.1f %9.3f %9.4f %10d" % (p, V[p].sum(), C[p].sum()/R.sum(),
                                             V[p].sum()/R.sum(), accepts[p]))

print("\npaired contrasts (cluster bootstrap over %d tasks, 10k resamples)" % len(tasks))
pairs = [("km","alphabet"), ("km","fault-record"), ("fault-record","alphabet")]
out["contrasts"] = {}
for a,b in pairs:
    d, lo, hi = boot(V[a], V[b])
    pw = wilcoxon(V[a], V[b])
    wins = int((V[a] < V[b]).sum())
    dc, cl, ch = boot(C[a].astype(float), C[b].astype(float))
    out["contrasts"][f"{a}_vs_{b}"] = dict(cpu_delta_pct=d, ci_lo=lo, ci_hi=hi,
        wilcoxon_p=pw, task_wins=wins, n=len(tasks),
        calls_delta_pct=dc, calls_ci_lo=cl, calls_ci_hi=ch)
    print("  %-14s vs %-14s CPU %+7.2f%%  CI [%+7.2f,%+7.2f]  p=%.2e  wins %2d/%d | probes %+7.2f%%"
          % (a, b, d, lo, hi, pw, wins, len(tasks), dc))

# per-round learning curve (measured)
out["per_round_cpu"] = {p: [round(statistics.mean(per_round[p][r]),4) if per_round[p][r] else None
                            for r in range(1,21)] for p in POL}
print("\nmeasured CPU per round (mean s over arena cells)")
for p in POL:
    v = out["per_round_cpu"][p]
    print("  %-14s r1 %7.3f  r5 %7.3f  r10 %7.3f  r20 %7.3f" % (p, v[0], v[4], v[9], v[19]))

json.dump(out, open("measured_stats.json","w"), indent=1)
