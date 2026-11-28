"""Figure: feasibility of a minimum service floor over depot budgets (Problem 5)."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.frameon": False,
})

rows = json.load(open("experiments/results/e18_q5_equity_grid.json"))["rows"]
Ks = sorted({r["K"] for r in rows})
qs = sorted({r["q"] for r in rows})
M = {(r["K"], r["q"]): r for r in rows}

FIG_W, FIG_H = 3.05, 2.72
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.135, right=0.975, bottom=0.135, top=0.965)

C_OK = "#c9dced"
C_NO = "#ffffff"
for xi, K in enumerate(Ks):
    for yi, q in enumerate(qs):
        r = M[(K, q)]
        if r["feasible"]:
            ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=C_OK, edgecolor="white", lw=0.9))
        else:
            ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor="#e9e9e9", edgecolor="white", lw=0.9))
            ax.text(xi + 0.5, yi + 0.5, "\u00d7", ha="center", va="center",
                    fontsize=6.4, color="0.42", zorder=4)
        if r["feasible"] and xi < 3:
            ax.text(xi + 0.5, yi + 0.5, f"{r['f1']:.3f}", ha="center", va="center",
                    fontsize=5.4, color="#14406b", zorder=4)

ax.set_xlim(0, len(Ks))
ax.set_ylim(0, len(qs))
ax.set_xticks(np.arange(len(Ks)) + 0.5)
ax.set_xticklabels(Ks, fontsize=6)
ax.set_yticks(np.arange(len(qs)) + 0.5)
ax.set_yticklabels([f"{q:g}" for q in qs], fontsize=6)
ax.tick_params(length=0, pad=2.2)
ax.set_xlabel("Depot budget $K$", labelpad=3)
ax.set_ylabel("Minimum service floor $q$", labelpad=3)

for s in ax.spines.values():
    s.set_linewidth(0.7)
    s.set_color("0.35")

ax.axvline(3.0, color="#14406b", lw=1.7, zorder=5)
ax.axvline(3.0, color="#14406b", lw=1.7, alpha=0.0)

ax.annotate("", xy=(3.0, len(qs) + 0.06), xytext=(3.0, len(qs) + 0.06),
            arrowprops=dict(arrowstyle="-", lw=0))
ax.text(1.5, len(qs) + 0.10, "no positive floor is feasible", ha="center", va="bottom",
        fontsize=5.6, color="#333333")
ax.text(4.5, len(qs) + 0.10, "floor already met at $q=1$", ha="center", va="bottom",
        fontsize=5.6, color="#14406b")

ax.text(3.06, 5.52, "$K=6$", ha="left", va="center", fontsize=6.2, color="#14406b",
        fontweight="bold")

fig.savefig("figures/fig_q5_equity.pdf", bbox_inches="tight")
fig.savefig("figures/fig_q5_equity.png", dpi=400, bbox_inches="tight")
print("saved")
