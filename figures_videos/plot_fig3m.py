#!/usr/bin/env python3
"""Fig. 3m plot body (axes, points, 95% CI, mean line and +-s.d. band; no text labels).

Input: SourceData_Fig3m_*.csv (one row per fluorescent bolus). The open grey symbol marks a bolus whose
speed could not be estimated reliably (speed_resolved = "no"); its height is arbitrary.

usage:  python plot_fig3m.py SourceData_Fig3m_lymph_pulses.csv --out Fig3m_plot_body
"""
import argparse
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out", default="Fig3m_plot_body")
ap.add_argument("--ylim", type=float, nargs=2, default=(375, 1475)); ap.add_argument("--xlim", type=float, nargs=2, default=(0, 852.5))
a = ap.parse_args()
m = pd.read_csv(a.csv); inc = m[m.speed_resolved == "yes"]; mean, sd = inc.speed_um_s.mean(), inc.speed_um_s.std(ddof=1)
CM = 1 / 2.54; fig = plt.figure(figsize=(9.831 * CM, 9.491 * CM)); left, bottom = 8.8 / 578, (558 - 545.5) / 558
ax = fig.add_axes([left, bottom, 1 - left, 1 - bottom]); BLUE, RED, GREY = (11/255, 111/255, 164/255), (198/255, 64/255, 39/255), (149/255,) * 3
ax.axhspan(mean - sd, mean + sd, color=(249/255, 235/255, 232/255), lw=0, zorder=0); ax.axhline(mean, color=RED, lw=1.9, zorder=1)
ax.errorbar(inc.time_s, inc.speed_um_s, yerr=[inc.speed_um_s - inc.ci95_lower_um_s, inc.ci95_upper_um_s - inc.speed_um_s],
            fmt="o", ms=6.75, color=BLUE, ecolor=BLUE, elinewidth=1.45, capsize=6.3, capthick=1.45, zorder=3, clip_on=False)
ymark = a.ylim[0] + 0.078 * (a.ylim[1] - a.ylim[0])
for t in m[m.speed_resolved != "yes"].time_s:
    ax.plot([t, t], [a.ylim[0], ymark], color=GREY, lw=1.0, zorder=2); ax.plot([t], [ymark], "o", ms=6.2, mfc="white", mec=GREY, mew=1.3, zorder=3)
ax.set_xlim(*a.xlim); ax.set_ylim(*a.ylim); ax.set_xticks([0, 200, 400, 600, 800]); ax.set_yticks(range(400, 1401, 200)); ax.set_xticklabels([]); ax.set_yticklabels([])
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_linewidth(1.45)
ax.tick_params(axis="x", direction="out", length=6.0, width=1.45); ax.tick_params(axis="y", direction="out", length=3.9, width=1.45)
fig.savefig(a.out + ".png", dpi=600); fig.savefig(a.out + ".svg"); print(f"mean +- s.d. = {mean:.1f} +- {sd:.1f} um/s (n = {len(inc)})")
