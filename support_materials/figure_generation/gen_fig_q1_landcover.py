"""Figure: area error of the Problem 1 model decomposed by pre-flood land cover."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

# LightGBM area error by pre-flood land cover (km^2): name, reference F, FP, FN, net
ROWS = [
    ("Tree", 33.46, 79.80, 4.93, +74.87),
    ("Cropland", 379.62, 105.59, 59.09, +46.50),
    ("Built-up", 8.79, 19.38, 1.81, +17.57),
    ("Shrub / grass", 53.92, 23.43, 7.65, +15.77),
    ("Bare", 7.97, 7.70, 1.15, +6.54),
    ("Permanent water", 156.06, 26.80, 50.91, -24.11),
    ("Wetland", 293.59, 25.46, 76.89, -51.43),
]

C_FP = "#c0504d"
C_FN = "#3b6ea5"
C_REF = "0.50"

fig, ax = plt.subplots(figsize=(3.35, 2.94))
fig.subplots_adjust(left=0.235, right=0.972, bottom=0.235, top=0.955)

n = len(ROWS)
for i, (name, ref, fp, fn, net) in enumerate(ROWS):
    y = n - 1 - i
    ax.barh(y + 0.185, fp, height=0.29, color=C_FP, lw=0, zorder=2)
    ax.barh(y - 0.185, fn, height=0.29, color=C_FN, lw=0, zorder=2)
    ax.text(fp + 1.8, y + 0.19, f"{fp:.1f}", va="center", fontsize=5.2, color=C_FP, zorder=6)
    ax.text(fn + 1.8, y - 0.19, f"{fn:.1f}", va="center", fontsize=5.2, color=C_FN, zorder=6)
    if name == "Tree":
        ax.plot([ref, ref], [y - 0.42, y + 0.42], color=C_REF, lw=1.0, ls=(0, (2.0, 1.3)), zorder=4)
        ax.text(ref, y + 0.50, f"real {ref:.1f}", ha="center", va="bottom", fontsize=5.0,
                color=C_REF, zorder=6)

ax.set_yticks(np.arange(n))
ax.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=6.2)
ax.set_ylim(-0.5, n - 0.42)
ax.set_xlim(0, 118)
ax.set_xlabel("Area (km$^2$)", labelpad=2)
ax.tick_params(axis="y", length=0, pad=3)
ax.tick_params(axis="x", length=2, pad=1.5, labelsize=6)

handles = [Patch(facecolor=C_FP, label="False alarm (FP)"),
           Patch(facecolor=C_FN, label="Missed (FN)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, -0.155),
          ncol=2, fontsize=5.6, handlelength=1.2, columnspacing=1.6, handletextpad=0.6)

fig.savefig("figures/fig_q1_landcover.pdf", bbox_inches="tight")
fig.savefig("figures/fig_q1_landcover.png", dpi=400, bbox_inches="tight")
print("saved")
