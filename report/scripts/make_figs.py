import json, csv, collections, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 160, "savefig.bbox": "tight",
    "legend.frameon": False, "xtick.direction": "out", "ytick.direction": "out",
})
M = json.load(open("measured_stats.json"))
POL = ["alphabet", "fault-record", "km"]
LAB = {"alphabet": "natural (pool order)", "fault-record": "fault-record (FRTP)", "km": r"$K/\hat{m}$ (ours)"}
COL = {"alphabet": BLUE, "fault-record": ORANGE, "km": AQUA}

# ---------- Fig 1: two panels, CPU and probes (never a dual axis) ----------
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5))
for ax, key, title, unit in [
        (axs[0], "cpu_total", "Oracle CPU over the arena", "CPU seconds"),
        (axs[1], "probes_per_round", "Probes paid per round", "probes / round")]:
    vals = [M["policies"][p][key] for p in POL]
    y = np.arange(len(POL))[::-1]
    ax.barh(y, vals, height=0.52, color=[COL[p] for p in POL], zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + max(vals) * 0.025, yy, f"{v:,.0f}" if v > 50 else f"{v:.2f}",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([LAB[p] for p in POL], fontsize=8.5)
    ax.set_xlabel(unit); ax.set_title(title, fontsize=9, color=INK, pad=6)
    ax.set_xlim(0, max(vals) * 1.22); ax.grid(axis="y", visible=False)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=2.2); fig.savefig("figs/fig-measured.pdf"); plt.close(fig)

# ---------- Fig 2: measured CPU per round ----------
fig, ax = plt.subplots(figsize=(5.4, 2.6))
R = np.arange(1, 21)
for p in POL:
    v = M["per_round_cpu"][p]
    ax.plot(R, v, color=COL[p], lw=2.0, label=LAB[p], solid_capstyle="round", zorder=3)
    ax.annotate(LAB[p], (R[-1], v[-1]), xytext=(5, 0), textcoords="offset points",
                color=INK2, fontsize=7.6, va="center", annotation_clip=False)
ax.set_xlabel("repair round"); ax.set_ylabel("mean oracle CPU per round (s)")
ax.set_xlim(1, 20); ax.set_ylim(0, 2.75); ax.set_xticks([1, 5, 10, 15, 20])
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(fontsize=7.6, loc="upper center", bbox_to_anchor=(0.46, -0.26), ncol=3,
          handlelength=1.6, columnspacing=1.4)
fig.subplots_adjust(right=0.70)
fig.savefig("figs/fig-round-curve.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: cost-exponent sweep ----------
iso = json.load(open("iso.json"))["c_sweep"]
cs = sorted(float(k) for k in iso)
d = [iso[str(c) if str(c) in iso else repr(c)]["delta_vs_km"] for c in cs]
lo = [iso[str(c) if str(c) in iso else repr(c)]["ci_lo"] for c in cs]
hi = [iso[str(c) if str(c) in iso else repr(c)]["ci_hi"] for c in cs]
fig, ax = plt.subplots(figsize=(4.4, 2.5))
ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
ax.fill_between(cs, lo, hi, color=AQUA, alpha=0.16, lw=0, zorder=2)
ax.plot(cs, d, color=AQUA, lw=2.0, zorder=3)
best = cs[int(np.argmin(d))]
ax.scatter([best], [min(d)], s=34, color=AQUA, edgecolor="#fcfcfb", lw=1.4, zorder=4)
ax.annotate(f"c = {best:g}\n{min(d):+.1f}%", (best, min(d)), xytext=(6, -16),
            textcoords="offset points", fontsize=7.8, color=INK)
ax.annotate(r"$c\!=\!1$: plain $K/\hat{m}$", (1.0, 0), xytext=(-4, 8),
            textcoords="offset points", fontsize=7.6, color=INK2, ha="right")
ax.set_xlabel(r"cost exponent $c$ in $K/\hat{m}^{\,c}$")
ax.set_ylabel(r"CPU vs. $K/\hat{m}$ (%)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("figs/fig-c-sweep.pdf"); plt.close(fig)

# ---------- Fig 4: per-task, km vs fault-record (measured) ----------
rows = list(csv.DictReader(open("rounds.csv")))
from run_base import PREP
ARENA = set(PREP)
cpu = collections.defaultdict(lambda: collections.defaultdict(float))
for r in rows:
    if r["task"] in ARENA: cpu[r["policy"]][r["task"]] += float(r["cpu_s"])
tasks = sorted(ARENA)
x = np.array([cpu["fault-record"][t] for t in tasks])
y = np.array([cpu["km"][t] for t in tasks])
fig, ax = plt.subplots(figsize=(3.5, 3.3))
lim = (min(x.min(), y.min()) * 0.5, max(x.max(), y.max()) * 2)
ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
win = y < x
ax.scatter(x[win], y[win], s=22, color=AQUA, edgecolor="#fcfcfb", lw=0.7, zorder=3,
           label=f"$K/\\hat{{m}}$ cheaper ({win.sum()})")
ax.scatter(x[~win], y[~win], s=22, color=ORANGE, edgecolor="#fcfcfb", lw=0.7, zorder=3,
           label=f"FRTP cheaper ({(~win).sum()})")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("fault-record CPU per task (s)"); ax.set_ylabel(r"$K/\hat{m}$ CPU per task (s)")
ax.legend(fontsize=7.4, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("figs/fig-per-task.pdf"); plt.close(fig)
print("wrote 4 figures")
print("per-task wins km<frtp:", int(win.sum()), "/", len(tasks))
