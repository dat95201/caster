#!/usr/bin/env python3
"""Verification gate: every pilot number asserted in the report's LaTeX must
match the generated artefact it came from.

Run from report/:  python3 scripts/verify_numbers.py
Exit 0 iff every assertion below holds against data/*.json.
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "data"
arena = json.load(open(D / "arena_stats.json"))
meas = json.load(open(D / "measured_stats.json"))
iso = json.load(open(D / "iso.json"))
head = {r["policy"]: r for r in json.load(open(D / "head_verify.json"))["head"]}

TEX = "\n".join(p.read_text() for p in
                sorted((ROOT / "chapters").glob("*.tex")) +
                sorted((ROOT / "frontmatter").glob("*.tex")))

fails, checks = [], 0


def claim(label, printed, actual, tol=5e-3):
    """`printed` is what the report says; `actual` is what the data says."""
    global checks
    checks += 1
    if abs(printed - actual) > tol * max(abs(actual), 1e-9) + 1e-9:
        fails.append(f"{label}: report says {printed}, data says {actual}")


def intext(s, label=None):
    """Assert a literal string really occurs in the LaTeX."""
    global checks
    checks += 1
    if s not in TEX:
        fails.append(f"string not found in report: {label or s!r}")


P = meas["policies"]
alpha, frtp, km = P["alphabet"], P["fault-record"], P["km"]
C = meas["contrasts"]

# ---- arena ----
claim("arena tasks", 68, arena["arena_tasks"])
claim("arena cases", 2523, arena["arena_cases"])
claim("arena patches", 4670, arena["arena_patches"])
claim("cost median", 0.0763, arena["cost_median"], 1e-2)
claim("cost p90", 1.740, arena["cost_p90"], 1e-2)
claim("cost p99", 4.476, arena["cost_p99"], 1e-2)
claim("cost max", 7.502, arena["cost_max"], 1e-2)
claim("p99/median ratio", 58.7, arena["cost_p99"] / arena["cost_median"], 1e-2)
claim("within-task cost CV", 0.571, arena["cost_cv_within_task_median"], 1e-2)
claim("killrate mean", 0.463, arena["killrate_mean"], 1e-2)
claim("killrate median", 0.306, arena["killrate_median"], 1e-2)
claim("never-kill share", 0.168, arena["never_kill_share"], 1e-2)
claim("corr(log cost, K)", 0.172, arena["corr_logcost_killrate_median"], 1e-2)
claim("corpus tasks", 309, arena["corpus_tasks"])
claim("pool median", 34, arena["pool_median"])
claim("pool mean", 38.9, arena["pool_mean"], 1e-2)
claim("pool min", 15, arena["pool_min"])
claim("pool max", 144, arena["pool_max"])

# ---- measured head-to-head ----
claim("rounds/policy", 6800, meas["rounds_per_policy"])
claim("alphabet CPU", 14842.8, alpha["cpu_total"], 1e-4)
claim("frtp CPU", 4189.8, frtp["cpu_total"], 1e-4)
claim("km CPU", 3531.9, km["cpu_total"], 1e-4)
claim("alphabet probes/rd", 6.493, alpha["probes_per_round"], 1e-3)
claim("frtp probes/rd", 1.641, frtp["probes_per_round"], 1e-3)
claim("km probes/rd", 2.024, km["probes_per_round"], 1e-3)
claim("alphabet CPU/rd", 2.1828, alpha["cpu_per_round"], 1e-3)
claim("frtp CPU/rd", 0.6162, frtp["cpu_per_round"], 1e-3)
claim("km CPU/rd", 0.5194, km["cpu_per_round"], 1e-3)
claim("alphabet CPU/probe", 0.3362, alpha["cpu_per_round"] / alpha["probes_per_round"], 1e-3)
claim("frtp CPU/probe", 0.3754, frtp["cpu_per_round"] / frtp["probes_per_round"], 1e-3)
claim("km CPU/probe", 0.2567, km["cpu_per_round"] / km["probes_per_round"], 1e-3)
claim("km cheaper/probe than frtp by", 31.6,
      100 * (1 - (km["cpu_per_round"] / km["probes_per_round"]) /
             (frtp["cpu_per_round"] / frtp["probes_per_round"])), 2e-2)

# ---- contrasts ----
ka, kf, fa = C["km_vs_alphabet"], C["km_vs_fault-record"], C["fault-record_vs_alphabet"]
claim("km vs alpha dCPU", -76.20, ka["cpu_delta_pct"], 1e-3)
claim("km vs alpha ci_lo", -84.94, ka["ci_lo"], 1e-2)
claim("km vs alpha ci_hi", -61.17, ka["ci_hi"], 1e-2)
claim("km vs alpha wins", 63, ka["task_wins"])
claim("km vs alpha dprobes", -68.84, ka["calls_delta_pct"], 1e-3)
claim("km vs frtp dCPU", -15.70, kf["cpu_delta_pct"], 1e-3)
claim("km vs frtp ci_lo", -28.93, kf["ci_lo"], 1e-2)
claim("km vs frtp ci_hi", -5.06, kf["ci_hi"], 2e-2)
claim("km vs frtp wins", 42, kf["task_wins"])
claim("km vs frtp dprobes", +23.30, kf["calls_delta_pct"], 1e-3)
claim("frtp vs alpha dCPU", -71.77, fa["cpu_delta_pct"], 1e-3)
claim("frtp vs alpha wins", 62, fa["task_wins"])
claim("frtp vs alpha dprobes", -74.73, fa["calls_delta_pct"], 1e-3)
assert kf["wilcoxon_p"] < 0.05, "km vs frtp no longer significant"
assert ka["wilcoxon_p"] < 1e-9, "km vs alphabet p drifted"

# ---- learning curve endpoints quoted in the text ----
claim("km round-20 CPU", 0.359, meas["per_round_cpu"]["km"][19], 1e-2)
claim("frtp round-20 CPU", 0.502, meas["per_round_cpu"]["fault-record"][19], 1e-2)

# ---- replay table ----
claim("replay natural CPU", 18297.1, head["natural (pool order)"]["cpu"], 1e-4)
claim("replay random CPU", 13275.8, head["random"]["cpu"], 1e-4)
claim("replay coverage CPU", 16611.3, head["coverage (TP_add)"]["cpu"], 1e-4)
claim("replay cost-only CPU", 8810.6, head["cost-only  1/m"]["cpu"], 1e-4)
claim("replay kill-only CPU", 2800.9, head["kill-only  K"]["cpu"], 1e-4)
claim("replay km CPU", 2654.2, head["km  K/m"]["cpu"], 1e-4)
claim("replay K/m^0.5 CPU", 2409.4, head["K/m^0.5   (proposed)"]["cpu"], 1e-4)
claim("replay K^.5 g^.25 /m^.25 CPU", 2366.4, head["K^0.5 g^0.25 /m^0.25"]["cpu"], 1e-4)

# ---- cost-exponent sweep ----
cs = iso["c_sweep"]
best = min(cs, key=lambda k: cs[k]["delta_vs_km"])
claim("best c", 0.5, float(best))
claim("best c delta", -9.22, cs[best]["delta_vs_km"], 1e-2)
claim("best c ci_lo", -15.03, cs[best]["ci_lo"], 2e-2)
claim("best c ci_hi", -1.56, cs[best]["ci_hi"], 5e-2)
claim("c=0 delta", +5.53, cs["0.0"]["delta_vs_km"], 2e-2)
claim("c=1.25 delta", +2.04, cs["1.25"]["delta_vs_km"], 3e-2)
assert cs[best]["ci_hi"] < 0, "c=0.5 is no longer significant vs km"

# ---- coverage isolation: the report claims 12 of 15 straddle zero ----
g = iso["g_isolation"]
straddle = sum(1 for v in g.values() if v["ci_lo"] < 0 < v["ci_hi"])
claim("g combos", 15, len(g))
claim("g combos straddling zero", 12, straddle)
claim("g at a=1,c=0.5,b=0.25", -2.16, g["a=1.00,c=0.50,b=0.25"]["delta_vs_b0"], 1e-2)

# ---- soundness ----
claim("accepts equal across policies", 1,
      1 if alpha["accepts"] == frtp["accepts"] == km["accepts"] else 0)
claim("accepts count", 6, km["accepts"])

# ---- strings that must literally appear, so a silent edit is caught ----
for s in ["$58.7\\times$", "$-76.20\\,\\%$", "$-15.70\\,\\%$", "\\bm{+23.30\\,\\%}",
          "0.3754", "0.2567", "$\\bm{+0.172}$", "$0.571$", "$c = 0.5$"]:
    intext(s)

print(f"{checks} checks run")
if fails:
    print("\nFAILED:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all pilot numbers in the report match the generated data")
