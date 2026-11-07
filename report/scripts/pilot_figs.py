#!/usr/bin/env python3
"""The report's four pilot figures, from data/pilot_v1.json.

    python3 pilot_figs.py        -> ../figures/*.pdf

Print first: every series carries a line style and a marker as well as a
colour, so the figures survive a greyscale printer.  The five hues were
checked for colour-vision separation on a white surface before use.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D    = json.load(open(os.path.join(HERE, '..', 'data', 'pilot_v1.json')))
CC   = json.load(open(os.path.join(HERE, '..', 'data', 'coverage_check.json')))
FIG  = os.path.join(HERE, '..', 'figures')

C = {'natural': '#8E6FBF', 'random': '#C96A0E', 'coverage': '#0E8F6F',
     'frtp': '#C0392B', 'sched': '#2B5FC9'}
LS = {'natural': (0, (4, 2)), 'random': (0, (1, 1.6)), 'coverage': (0, (5, 1.5, 1, 1.5)),
      'frtp': (0, (2.5, 1.2)), 'sched': '-'}
MK = {'natural': 's', 'random': '^', 'coverage': 'D', 'frtp': 'o', 'sched': 'o'}
LB = {'natural': 'natural (incumbent)', 'random': 'random (control)',
      'coverage': 'coverage', 'frtp': 'frtp', 'sched': 'sched (ours)'}
ORDER = ['natural', 'random', 'coverage', 'frtp', 'sched']

plt.rcParams.update({
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'legend.fontsize': 7.6, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#8a8a8a', 'axes.linewidth': 0.7,
    'grid.color': '#dcdcdc', 'grid.linewidth': 0.6,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
    'text.color': '#222222', 'axes.labelcolor': '#222222',
    'xtick.color': '#555555', 'ytick.color': '#555555'})


def style(ax):
    ax.grid(True, axis='y', zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)


# -------------------------------------------- part 1: is it worth doing? ----
BASE4 = ['natural', 'random', 'coverage', 'sched']


def _curve(ax, group, key, lab, ymax=None):
    cur = D['round_curve']; B = len(cur['sched']['cpu']); x = np.arange(1, B + 1)
    for p in group:
        ax.plot(x, cur[p][key], color=C[p], ls=LS[p], marker=MK[p], ms=2.8,
                lw=1.7 if p == 'sched' else 1.2, label=LB[p],
                zorder=3 if p == 'sched' else 2)
    ax.set_xlabel('round'); ax.set_ylabel(lab)
    ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlim(0.5, B + 0.5)
    ax.set_ylim(0, ymax); style(ax)


def vs_baselines():
    """natural / random / sched. Does scheduling the oracle pay at all?"""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    _curve(axes[0], BASE4, 'cpu', 'CPU seconds per round')
    cum = D['cumulative']; B = len(cum['sched']); x = np.arange(1, B + 1)
    for p in BASE4:
        axes[1].plot(x, cum[p], color=C[p], ls=LS[p], marker=MK[p], ms=2.8,
                     lw=1.7 if p == 'sched' else 1.2, zorder=3 if p == 'sched' else 2)
        axes[1].annotate(f'{cum[p][-1]:,.0f}\u2009s', (x[-1], cum[p][-1]), xytext=(3, 0),
                         textcoords='offset points', va='center', fontsize=7.2, color=C[p])
    axes[1].set_xlabel('round'); axes[1].set_ylabel('cumulative CPU seconds')
    axes[1].set_xticks([1, 5, 10, 15, 20]); axes[1].set_xlim(0.5, B + 4.5)
    axes[1].set_ylim(bottom=0); style(axes[1])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, handlelength=2.2, loc='lower center',
               bbox_to_anchor=(0.5, -0.1), columnspacing=1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig-vs-baselines.pdf')); plt.close(fig)


# ------------------------------- part 2: against the published prioritisers --
def vs_prioritisers():
    """sched against frtp, as DIFFERENCES only.

    Plotting the two as absolute curves is what hides the result: both sit near
    0.4 s per round on an axis that has to reach 2.5, so a 10% gap is a line
    width. Both panels here are differences, where a 10% gap is 10% of the
    plotted quantity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))

    # (a) the running difference in seconds
    cum = D['cumulative']; B = len(cum['sched']); xr = np.arange(1, B + 1)
    diff = np.array(cum['sched']) - np.array(cum['frtp'])
    ax = axes[0]
    ax.axhline(0, color='#9a9a9a', lw=0.9, zorder=1)
    ax.plot(xr, diff, color=C['sched'], lw=1.8, marker='o', ms=3, zorder=3)
    ax.fill_between(xr, 0, diff, where=diff <= 0, color=C['sched'], alpha=0.16, zorder=2)
    ax.fill_between(xr, 0, diff, where=diff >= 0, color=C['frtp'], alpha=0.16, zorder=2)
    cross = next((i for i in range(1, B) if diff[i] <= 0 < diff[i - 1]), None)
    if cross is not None:
        ax.axvline(xr[cross], color='#9a9a9a', lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.annotate(f'sched overtakes\nat round {xr[cross]}', (xr[cross], 0),
                    xytext=(5, 16), textcoords='offset points', fontsize=7.2,
                    color='#555555')
    ax.annotate(f'{diff.max():+.0f}\u2009s', (xr[int(diff.argmax())], diff.max()),
                xytext=(0, 5), textcoords='offset points', ha='center',
                fontsize=7.2, color=C['frtp'])
    ax.annotate(f'{diff[-1]:+,.0f}\u2009s', (xr[-1], diff[-1]), xytext=(-4, 7),
                textcoords='offset points', ha='right', fontsize=7.4, color=C['sched'])
    ax.set_ylim(diff.min() * 1.22, diff.max() * 1.9)
    ax.set_xlabel('round')
    ax.set_ylabel('cumulative CPU,\nsched minus frtp (s)')
    ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlim(0.5, B + 0.5)
    style(ax)

    # (b) the same split into blocks, with intervals
    blk = D['by_block']['sched_vs_frtp']
    xb = np.arange(len(blk))
    val = [b['cpu_pct'] for b in blk]
    lo = [v - b['ci'][0] for v, b in zip(val, blk)]
    hi = [b['ci'][1] - v for v, b in zip(val, blk)]
    ax = axes[1]
    ax.axhline(0, color='#9a9a9a', lw=0.9, zorder=1)
    ax.errorbar(xb, val, yerr=[lo, hi], fmt='o', ms=5.5, lw=1.4, capsize=3,
                color=C['sched'], ecolor=C['sched'], zorder=3)
    for x_, v, b in zip(xb, val, blk):
        ax.annotate(f'{v:+.1f}%', (x_, b['ci'][1]), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=7.4,
                    color=C['sched'])
    top = max(b['ci'][1] for b in blk); bot = min(b['ci'][0] for b in blk)
    ax.set_ylim(bot - 0.06 * (top - bot), top + 0.22 * (top - bot))
    ax.set_xticks(xb); ax.set_xticklabels([f"{b['first']}\u2013{b['last']}" for b in blk])
    ax.set_xlabel('rounds'); ax.set_ylabel('sched vs frtp, % CPU')
    ax.set_xlim(-0.5, len(blk) - 0.5)
    style(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig-vs-prioritisers.pdf')); plt.close(fig)


# ------------------------------------------------------------- per task ----
def _scatter(ax, base, fig):
    pt = D['per_task']; cv = D['per_task_cv']; tasks = sorted(pt['sched'])
    x = np.array([pt[base][t] for t in tasks]); y = np.array([pt['sched'][t] for t in tasks])
    m = (x > 0) & (y > 0); c = np.array([cv[t] for t in tasks])[m]
    lo = min(x[m].min(), y[m].min()) * 0.6; hi = max(x[m].max(), y[m].max()) * 1.6
    ax.plot([lo, hi], [lo, hi], color='#9a9a9a', lw=0.8, zorder=1)
    s = ax.scatter(x[m], y[m], c=c, cmap='PuBu', vmin=0, vmax=1.0, s=20,
                   edgecolors='#555555', linewidths=0.4, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f'{LB[base]} \u2014 CPU seconds per task')
    ax.set_ylabel('sched \u2014 CPU seconds per task')
    ax.text(0.97, 0.06, 'below the line:\nsched cheaper', transform=ax.transAxes,
            va='bottom', ha='right', fontsize=7.2, color=C['sched'])
    ax.grid(True, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=3, width=0.7)
    return s


def per_task_baseline():
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    s = _scatter(ax, 'natural', fig)
    cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label('cost spread within the task (CV)', fontsize=7.4)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig-per-task-baseline.pdf')); plt.close(fig)


def per_task_prioritiser():
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    s = _scatter(ax, 'frtp', fig)
    cb = fig.colorbar(s, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label('cost spread within the task (CV)', fontsize=7.4)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig-per-task-prioritiser.pdf')); plt.close(fig)


# ---------------------------------------------------------- cost spread ----
def spread():
    """Only the within-task CV curve. The per-test cost quantiles live in the
    corpus table, and a bar chart of four numbers already in a table is a
    figure that earns nothing."""
    cv = np.array(sorted(D['per_task_cv'].values()))
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    ax.plot(cv, np.arange(1, len(cv) + 1) / len(cv), color='#2B5FC9', lw=1.8, zorder=3)
    ax.axvline(0.3, color='#C0392B', lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(0.33, 0.10, 'spread the cost term\ncan act on', fontsize=7.2, color='#C0392B')
    ax.set_xlabel('spread of test costs within one task (CV)')
    ax.set_ylabel('share of tasks\nat or below')
    ax.set_xlim(0, min(1.6, cv.max())); ax.set_ylim(0, 1.02)
    ax.grid(True, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=3, width=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig-cost-spread.pdf')); plt.close(fig)


# ------------------------------------------------------ coverage check -----
def coverage():
    m = CC['modes']
    labels = ['clause off\n($\\tau = 0$)', 'clause on,\nfed by the session',
              'clause on,\nfed by the audit table']
    vals = [m['off']['pct_vs_off'], m['session']['pct_vs_off'], m['seeded']['pct_vs_off']]
    err = [[v - m[k]['ci'][0] for v, k in zip(vals, ('off', 'session', 'seeded'))],
           [m[k]['ci'][1] - v for v, k in zip(vals, ('off', 'session', 'seeded'))]]
    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    y = np.arange(3)[::-1]
    ax.axvline(0, color='#9a9a9a', lw=0.8, zorder=1)
    for i, (v, yy) in enumerate(zip(vals, y)):
        col = '#8a8a8a' if i == 0 else '#2B5FC9'
        ax.errorbar([v], [yy], xerr=[[err[0][i]], [err[1][i]]], fmt='o', ms=5, lw=1.3,
                    capsize=3, color=col, ecolor=col, zorder=3)
        ax.text(v, yy + 0.22, f'{v:+.1f}%' if i else 'reference',
                ha='center', fontsize=7.4, color=col)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('change in validation CPU against the clause switched off')
    ax.set_ylim(-0.6, 2.6)
    ax.grid(True, axis='x', zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig-coverage-check.pdf')); plt.close(fig)


for f in (vs_baselines, vs_prioritisers, per_task_baseline,
          per_task_prioritiser, spread, coverage):
    f(); print('ok', f.__name__)
