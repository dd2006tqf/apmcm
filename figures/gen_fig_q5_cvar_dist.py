"""Figure: frozen-plan coverage under 600 demand and network scenarios (Problem 5)."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.frameon": False,
})

d = json.load(open("experiments/results/e15_q5_cvar.json"))
sc = d["scenarios"]
base = d["base"]["f1"]
cvar = d["all"]["f1"]["cvar95"]
stress = d["stress"]["f1"]

GROUPS = [("demand", "Demand", "#9dbcd4"), ("network", "Network", "#4a7ba7"), ("both", "Both", "#1f4e79")]
rng = np.random.default_rng(7)

fig, ax = plt.subplots(figsize=(3.35, 2.62))
fig.subplots_adjust(left=0.145, right=0.975, bottom=0.255, top=0.965)

for xi, (key, lab, col) in enumerate(GROUPS):
    v = np.array([s["f1"] for s in sc if s["kind"] == key])
    jit = rng.uniform(-0.155, 0.155, len(v))
    ax.scatter(np.full(len(v), xi) + jit, v, s=2.2, color=col, alpha=0.55, lw=0, zorder=2)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([xi - 0.214, xi + 0.214], [med, med], color=col, lw=1.0, zorder=4)
    ax.plot([xi, xi], [q1, q3], color=col, lw=0.8, zorder=3)

ax.axhline(base, color="0.25", lw=1.0, ls="-", zorder=5)
ax.axhline(cvar, color="#a63d00", lw=1.2, ls=(0, (3.4, 1.6)), zorder=5)
ax.scatter([2.0], [stress], marker="D", s=17, facecolors="none", edgecolors="#a63d00", lw=0.9, zorder=6)

ax.set_xlim(-0.42, 2.42)
ax.set_ylim(0.547, 0.6815)
ax.set_xticks(range(3))
ax.set_xticklabels([g[1] for g in GROUPS], fontsize=6.2)
ax.set_ylabel("Weighted unmet share $f_1$", labelpad=2)
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", length=2, pad=1.5, labelsize=6)

handles = [
    Line2D([0], [0], color="0.25", lw=1.0, label="Recommended 0.570"),
    Line2D([0], [0], color="#a63d00", lw=1.2, ls=(0, (3.4, 1.6)), label="CVaR$_{95}$ 0.651"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="none",
           markeredgecolor="#a63d00", markeredgewidth=0.9, markersize=3.6,
           label="Stress 0.670"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, -0.135),
          ncol=3, fontsize=5.5, handlelength=1.7, columnspacing=1.3, handletextpad=0.55)

fig.savefig("figures/fig_q5_cvar.pdf", bbox_inches="tight")
fig.savefig("figures/fig_q5_cvar.png", dpi=400, bbox_inches="tight")
print("saved")
