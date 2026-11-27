"""Q5 bi-objective trade-off and one-at-a-time sensitivity.

(a) epsilon-constraint Pareto front of priority-weighted unmet share f1 vs
weighted delivery time f2; served population as point sample; recommended plan,
knee and the 07-04 plan re-optimized under 07-16 inputs.
(b) one-at-a-time exact re-solves, expressed as the change in optimal f1*
relative to the adopted plan.  Positive (red) bars mean a higher unmet share;
negative (blue) bars mean a lower unmet share.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
OUT_PDF = os.path.join(FIG_DIR, "fig_q5_pareto.pdf")
OUT_PNG = os.path.join(FIG_DIR, "fig_q5_pareto_preview.png")

# A small, self-contained version of the manuscript palette.
BLUE = "#2F6D96"
BLUE_DARK = "#214E6B"
RED = "#B64035"
RED_DARK = "#8E2F27"
GOLD = "#D8A02B"
INK = "#252525"
GREY = "#777777"
GREY_LIGHT = "#D8D8D8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.3,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load(name):
    for d in (FIG_DIR, os.path.join(ROOT, "output")):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(name)


q5 = load("problem_5_results.json")
P = q5["pareto"]
f1 = np.array([p["f1"] for p in P])
f2 = np.array([p["f2"] for p in P])
sp = np.array([p["served_pop"] for p in P])
assert all(p["optimal"] for p in P)
kn = q5["knee_index"]
rec = q5["plan_0716"]
rc = q5["recourse_0704_to_0716"]

fig = plt.figure(figsize=(6.6, 3.65))
gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.12], wspace=0.26)

# ------------------------------------------------------------------ panel a
ax = fig.add_subplot(gs[0, 0])
ax.plot(f1, f2, color=GREY, lw=0.9, zorder=1)
ax.scatter(f1, f2, color=BLUE, s=24, edgecolor=INK, lw=0.45, zorder=3,
           label=r"Pareto point ($\varepsilon$-constraint)")
ax.scatter(rec["f1"], rec["f2"], marker="*", s=140, color=RED,
           edgecolor="white", lw=0.55, zorder=5,
           label=f"Recommended, {rec['served_pop']:.0f} served")
ax.scatter(f1[kn], f2[kn], marker="D", s=42, facecolor="none",
           edgecolor=INK, lw=1.2, zorder=5,
           label=f"Knee, {sp[kn]:.0f} served")
ax.scatter(rc["f1"], rc["f2"], marker="^", s=44, color=GOLD,
           edgecolor=INK, lw=0.5, zorder=5,
           label=f"07-04 depots re-used, {rc['served_pop']:.0f} served")

ax.set_xlim(0.555, 1.015)
ax.set_ylim(-0.2, 13.7)
ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
ax.set_xlabel(r"Weighted unmet share $f_1$")
ax.set_ylabel(r"Weighted delivery time $f_2$ (min)")
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.14, 1.02, "a", transform=ax.transAxes, fontsize=9,
        fontweight="bold", va="bottom", ha="left")

# shared legend above the panels
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.985),
           ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.0,
           borderaxespad=0.0)

# ------------------------------------------------------------------ panel b
ax2 = fig.add_subplot(gs[0, 1])
base = float(rec["f1"])
groups = [
    ("buffer", "Road buffer (m)"),
    ("facility", "Candidate set"),
    ("t_max", "Time limit (min)"),
    ("K", "Depots K"),
    ("speed", "Speed factor"),
    ("q", "Ration (kg/day)"),
    ("cap_scale", "Capacity factor"),
    ("n_veh", "Fleet size"),
]
rows = []
for key, name in groups:
    vals = np.array([r["f1"] for r in q5["sensitivity"] if r["param"] == key]) - base
    assert len(vals) > 0
    rows.append((name, vals.min(), vals.max(), vals))

# sort most sensitive first (largest absolute change at the top)
rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])), reverse=True)

y = np.arange(len(rows))[::-1]
bar_height = 0.62
ZERO = 3e-4
for yy, (name, lo, hi, devs) in zip(y, rows):
    if hi > ZERO:
        ax2.barh(yy, hi, height=bar_height, color=RED, edgecolor="none", zorder=3)
        ax2.text(hi + 0.0018, yy, f"{hi:+.3f}", va="center", ha="left",
                 fontsize=6.4, color=RED_DARK, zorder=5)
    if lo < -ZERO:
        ax2.barh(yy, lo, height=bar_height, color=BLUE, edgecolor="none", zorder=3)
        ax2.text(lo - 0.0018, yy, f"{lo:+.3f}", va="center", ha="right",
                 fontsize=6.4, color=BLUE_DARK, zorder=5)
    if abs(lo) <= ZERO and abs(hi) <= ZERO:
        ax2.text(0.0022, yy, "0.000", va="center", ha="left",
                 fontsize=6.4, color=GREY, zorder=5)
    ax2.scatter(devs, np.full(len(devs), yy), s=5.5, facecolor="white",
                edgecolor=INK, linewidth=0.3, zorder=6, clip_on=False)

ax2.axvline(0, color=INK, lw=0.75, zorder=2)
ax2.set_yticks(y, [r[0] for r in rows])
ax2.set_ylim(-0.35, len(rows) - 0.45)
ax2.set_xlim(-0.03, 0.091)
ax2.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08])
ax2.set_xlabel(r"$\Delta f_1^*$ from adopted plan")
ax2.tick_params(axis="y", length=0, pad=4)
ax2.tick_params(axis="x", pad=2)
ax2.spines[["top", "right"]].set_visible(False)
ax2.xaxis.grid(True, color="#ECECEC", lw=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.text(-0.24, 1.02, "b", transform=ax2.transAxes, fontsize=9,
        fontweight="bold", va="bottom", ha="left")
ax2.legend(
    handles=[Patch(facecolor=RED, edgecolor="none"),
             Patch(facecolor=BLUE, edgecolor="none")],
    labels=[r"higher $f_1^*$", r"lower $f_1^*$"],
    loc="lower right", frameon=False, fontsize=6.4,
    handlelength=1.0, handletextpad=0.4, borderaxespad=0.1,
)

fig.subplots_adjust(left=0.095, right=0.985, bottom=0.16, top=0.82)

fig.savefig(OUT_PDF, dpi=300)
fig.savefig(OUT_PNG, dpi=300)
plt.close(fig)
print(OUT_PDF)
print(OUT_PNG)
