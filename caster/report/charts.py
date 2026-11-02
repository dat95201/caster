"""The HTML report. One self-contained file, no network, no build step.

Deliberately inline SVG and plain CSS rather than a charting library: the report
is written on a Colab runtime and read on a laptop, often offline, and a CDN
script tag is one more thing that can be missing at the moment someone wants to
read a number. It also means the file can be committed next to the run.

The report leads with the three things that must be read before any scheduler
number — proposal-error rates per arm and per model, the realised
`k_facts` per steered arm, and the repair-rate bands — because a reader who
scrolls past them to the CPU table is reading a number whose denominator they
have not seen.
"""
from __future__ import annotations

import html
import json
import pathlib
from typing import Any, Sequence

_CSS = """
:root { --fg:#111; --mut:#666; --line:#e4e4e7; --bg:#fff; --accent:#2563eb;
        --warn:#b45309; --bad:#b91c1c; --ok:#15803d; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e7e7ea; --mut:#9a9aa2; --line:#2a2a30; --bg:#121214;
          --accent:#7aa2f7; --warn:#d9a441; --bad:#f87171; --ok:#4ade80; }
}
* { box-sizing: border-box; }
body { margin:0; padding:32px 16px 72px; background:var(--bg); color:var(--fg);
       font:15px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }
h2 { font-size:18px; margin:40px 0 8px; letter-spacing:-0.01em; }
p.sub { color:var(--mut); margin:0 0 24px; }
p.note { color:var(--mut); font-size:13px; margin:6px 0 14px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; margin:8px 0 4px; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line);
        white-space:nowrap; }
th { color:var(--mut); font-weight:600; font-size:12px; text-transform:uppercase;
     letter-spacing:.04em; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.pill { display:inline-block; padding:1px 7px; border-radius:999px; font-size:11.5px;
        border:1px solid var(--line); color:var(--mut); }
.bad { color:var(--bad); font-weight:600; }
.warn { color:var(--warn); font-weight:600; }
.ok { color:var(--ok); }
.bar { height:9px; border-radius:999px; background:var(--accent); display:inline-block;
       vertical-align:middle; }
.scroll { overflow-x:auto; }
footer { margin-top:56px; color:var(--mut); font-size:12.5px; }
code { font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;
       background:color-mix(in srgb, var(--fg) 7%, transparent);
       padding:1px 5px; border-radius:4px; }
"""


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
           numeric: Sequence[int] = ()) -> str:
    th = "".join(
        f'<th class="num">{html.escape(str(h))}</th>' if i in numeric
        else f"<th>{html.escape(str(h))}</th>"
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        tds = "".join(
            f'<td class="num">{cell}</td>' if i in numeric else f"<td>{cell}</td>"
            for i, cell in enumerate(row)
        )
        body.append(f"<tr>{tds}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _sparkline(values: Sequence[float], width: int = 260, height: int = 44) -> str:
    if not values:
        return ""
    top = max(values) or 1.0
    step = width / max(1, len(values) - 1) if len(values) > 1 else width
    pts = " ".join(
        f"{i * step:.1f},{height - (v / top) * (height - 6) - 3:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="success@b curve">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linejoin="round" points="{pts}"/></svg>'
    )


def render(summary: dict[str, Any], *, title: str = "CASTER run report",
           meta: dict[str, Any] | None = None) -> str:
    budget = summary.get("budget", 0)
    parts: list[str] = []

    # ── threats first ───────────────────────────────────────────────────────
    errs = summary.get("proposal_errors", [])
    rows = [[
        html.escape(r["policy"]), html.escape(r["model"] or "—"), r["cells"], r["rounds"],
        r.get("backend_unreachable", 0), r.get("context_overflow", 0),
        r.get("truncated_response", 0),
        (f'<span class="bad">{r["error_rate"]:.1%}</span>' if r.get("threat")
         else f'{r["error_rate"]:.1%}'),
    ] for r in errs]
    parts.append("<h2>Proposal failures, per arm and per model</h2>")
    parts.append('<p class="note">Reported before any scheduler number. The steered '
                 'arms carry the longest prompts, so every failure here lands '
                 'differentially on exactly the comparison the experiment exists to '
                 'make. Above ~20% a model is a threat to validity, not a rung on a '
                 'ladder.</p>')
    parts.append(_table(
        ["arm", "model", "cells", "rounds", "unreachable", "overflow", "truncated", "rate"],
        rows, numeric=(2, 3, 4, 5, 6, 7)))

    kf = summary.get("k_facts", [])
    if kf:
        parts.append("<h2>Realised evidence budget</h2>")
        parts.append('<p class="note">A steering arm whose realised mean '
                     '<code>k_facts</code> is below the configured <code>k</code> is '
                     'reporting a different treatment than its label says — which is '
                     'what happens when <code>explore_p</code> is off and one probe '
                     'kills everything.</p>')
        parts.append(_table(
            ["arm", "configured k", "realised k", "cells", ""],
            [[html.escape(r["policy"]), r["configured_k"], r["realised_k_mean"], r["cells"],
              '<span class="bad">mislabelled</span>' if r["mislabelled"]
              else '<span class="ok">ok</span>']
             for r in kf], numeric=(1, 2, 3)))

    bands = summary.get("bands", {})
    if bands:
        counts: dict[str, int] = {}
        for band in bands.values():
            counts[band] = counts.get(band, 0) + 1
        total = sum(counts.values()) or 1
        parts.append("<h2>Repair-rate bands (post hoc)</h2>")
        parts.append('<p class="note">Computed from the floor arm after the run. '
                     'Repair rate is an outcome, so it may not select the corpus — '
                     'a pre-treatment screen on a difficulty proxy puts a covariate '
                     'upstream of the thing under study.</p>')
        parts.append(_table(
            ["band", "(task, model) pairs", "share", ""],
            [[b, n, f"{n / total:.1%}",
              f'<span class="bar" style="width:{max(2, int(180 * n / total))}px"></span>']
             for b, n in sorted(counts.items(), key=lambda kv: -kv[1])],
            numeric=(1, 2)))

    by_pool = summary.get("arms_by_pool_band", {})
    if by_pool:
        parts.append("<h2>Pool-size bands (post hoc)</h2>")
        parts.append('<p class="note">G3 used to screen on <code>|P_loop| &ge; 20</code> '
                     '<em>before</em> the run, which deleted 23.2% of the corpus and '
                     'left a population nobody could weight back to the benchmark. '
                     'The grid now runs every sound fault and the screen is a band: '
                     '<code>standard</code> is exactly the old corpus, and the rest is '
                     'reported beside it. At <code>|P_loop|</code> = 1 every policy is '
                     'the same schedule by construction, so <code>degenerate</code> '
                     'dilutes a corpus-wide average and is shown apart.</p>')
        parts.append(_table(
            ["band", "tasks", "cells", "median |P_loop|", "best arm by cpu/episode"],
            [[b, d["tasks"], d["cells"], d["pool_median"],
              html.escape(min(d["arms"], key=lambda r: r["cpu_s_mean"])["policy"])
              if d["arms"] else "—"]
             for b, d in by_pool.items()], numeric=(1, 2, 3)))
        for b, d in by_pool.items():
            parts.append(f"<h3>Arms — {html.escape(b)} "
                         f'(|P_loop| median {d["pool_median"]}, {d["tasks"]} tasks)</h3>')
            parts.append(_table(
                ["arm", "block", "cells", "cpu/episode", "probes/episode",
                 "success@B"],
                [[html.escape(r["policy"]), f'<span class="pill">{r["block"]}</span>',
                  r["cells"], f'{r["cpu_s_mean"]:.3f}', f'{r["test_calls_mean"]:.2f}',
                  f'{r["success_at_b"]:.1%}']
                 for r in d["arms"]], numeric=(2, 3, 4, 5)))

    # ── the main tables ─────────────────────────────────────────────────────
    arms = summary.get("arms", [])
    survival = summary.get("survival", {})
    parts.append("<h2>Arms — CPU seconds per episode</h2>")
    parts.append('<p class="note">RQ1\'s primary metric. CPU is '
                 '<code>getrusage(RUSAGE_CHILDREN)</code>, never wall clock '
                 '(Invariant 3): wall clock is contaminated by sibling shards and '
                 'would make m̂ a function of load.</p>')
    parts.append(_table(
        ["arm", "block", "cells", "cpu/episode", "probes/episode", "sandbox runs",
         "store hits", "success@B", "first accept", f"success@b curve (B={budget})"],
        [[html.escape(r["policy"]), f'<span class="pill">{r["block"]}</span>',
          r["cells"], f'{r["cpu_s_mean"]:.3f}', f'{r["test_calls_mean"]:.2f}',
          r["sandbox_runs"], r["verdict_hits"], f'{r["success_at_b"]:.1%}',
          (f'{r["first_accept_mean"]:.2f}' if r["first_accept_mean"] else "—"),
          _sparkline(survival.get(r["policy"], []))]
         for r in arms], numeric=(2, 3, 4, 5, 6, 7, 8)))

    dec = summary.get("decomposition", [])
    if dec:
        parts.append("<h2>RQ2 — the stepwise decomposition</h2>")
        parts.append('<p class="note">Paired per cell under CRN. One thing changes '
                     'per step: <code>fault-record → km</code> changes two (a '
                     'membership bit becomes a rate, and the rate is divided by '
                     'cost), so it cannot attribute; this chain can.</p>')
        parts.append(_table(
            ["step", "paired cells", "ΔCPU mean", "ΔCPU median"],
            [[html.escape(r["step"]), r["paired_cells"],
              (f'{r["cpu_delta_mean_pct"]:+.2f}%' if r["cpu_delta_mean_pct"] is not None else "—"),
              (f'{r["cpu_delta_median_pct"]:+.2f}%' if r["cpu_delta_median_pct"] is not None else "—")]
             for r in dec], numeric=(1, 2, 3)))

    head = (
        f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<div class=\"wrap\"><h1>{html.escape(title)}</h1>"
        f"<p class=\"sub\">{summary.get('cells', 0)} cells · budget {budget} · "
        f"merge audit must pass before any of this is evidence (Invariant 12)</p>"
    )
    foot = (
        "<footer>Generated by <code>caster report</code>. "
        "CPU is getrusage(RUSAGE_CHILDREN). Timeout verdicts are quarantined and "
        "never served from the verdict store. "
        f"{'<br>' + html.escape(json.dumps(meta, sort_keys=True)) if meta else ''}"
        "</footer></div></body></html>"
    )
    return head + "".join(parts) + foot


def write(summary: dict[str, Any], path: pathlib.Path, **kw: Any) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(summary, **kw), encoding="utf-8")
    return path
