"""Q1 error structure, left-right layout: (a) FP/FN by pre-flood WorldCover stratum, with net
M0 and M6; (b) TreeSHAP beeswarm of tile-level |area error| drivers."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 7,
    "axes.labelsize": 7.4,
    "legend.fontsize": 6.9,
})

C = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#7FA8D9",
    "red_strong": "#B64342",
    "red_2": "#D98B8B",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "neutral_light": "#D9D9D9",
    "neutral_mid": "#9A9A9A",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#1A1A1A",
}


def load(name):
    with open(os.path.join(FIG, name), encoding="utf-8") as f:
        return json.load(f)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=2.5)
    ax.yaxis.set_minor_locator(plt.NullLocator())


q1 = load("problem_1_results.json")
q2 = load("problem_2_results.json")
rows = [r for r in q1["models"]["B"]["fpfn_worldcover"]["rows"] if r["ref_km2"] > 0]
m6 = {r["stratum"]: r for r in q2["fpfn_M6"]["rows"]}
rows = sorted(rows, key=lambda r: r["net_km2"])
names = [r["stratum"] for r in rows]
FP = np.array([r["FP_km2"] for r in rows])
FN = np.array([r["FN_km2"] for r in rows])
NET = np.array([r["net_km2"] for r in rows])
NET6 = np.array([m6[n]["net_km2"] for n in names])
assert np.allclose(FP - FN, NET)

LABEL = {
    "rho_ref": "Reference flood share", "sar_mean": "SAR mean", "sar_std": "SAR std (raw)",
    "ashman_share": "Bimodality share", "enl": "ENL", "ndvi_mean": "NDVI mean",
    "share_built": "Built-up share", "share_cropland": "Cropland share", "share_tree": "Tree share",
    "share_prewater": "Pre-flood water share", "opt_invalid": "Optical invalid share",
    "boundary_density": "Label boundary density",
}
dr = q1["drivers"]
desc = dr["descriptors"]
S = np.asarray(dr["abs"]["shap_values"], float)
X = np.asarray(dr["X"], float)
order = np.argsort(np.abs(S).mean(0))[::-1][:8]

fig = plt.figure(figsize=(7.2, 4.45), layout="constrained")
fig.get_layout_engine().set(rect=(0.005, 0.005, 0.99, 0.99))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.18, 0.035], height_ratios=[1, 0.12])

# ---- (a) error decomposition
ax = fig.add_subplot(gs[0, 0])
yy = np.arange(len(rows))
ax.barh(yy, FP, color=C["red_2"], edgecolor=C["red_strong"], lw=0.6, height=0.62, zorder=2)
ax.barh(yy, -FN, color=C["blue_secondary"], alpha=0.55, edgecolor=C["blue_main"],
        lw=0.6, height=0.62, zorder=2)
ax.scatter(NET, yy, marker="D", s=18, color=C["neutral_black"], zorder=3)
ax.scatter(NET6, yy, marker="o", s=20, facecolor="white", edgecolor=C["teal"], lw=1.1, zorder=3)
ax.axvline(0, color=C["neutral_dark"], lw=0.8)
ax.set_yticks(yy, names)
ax.set_xlabel("Area (km$^2$)   FN $\\leftarrow$ | $\\rightarrow$ FP")
xmin = min(-FN.min(), NET.min(), NET6.min(), 0)
xmax = max(FP.max(), NET6.max(), 0)
span = xmax - xmin
ax.set_xlim(xmin - 0.05 * span, xmax + 0.06 * span)
ax.set_ylim(-0.55, len(rows) - 0.45)
style(ax)
ax.set_title("a", loc="left", fontweight="bold", fontsize=9, pad=3)

# ---- (b) SHAP beeswarm
ax2 = fig.add_subplot(gs[0, 1])
cmap = LinearSegmentedColormap.from_list("feat", [C["blue_main"], "#F7F7F7", C["red_strong"]])
rng = np.random.default_rng(2026)
for r, j in enumerate(order):
    s, x = S[:, j], X[:, j]
    lo, hi = np.nanpercentile(x, [5, 95])
    xn = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
    hist, edges = np.histogram(s, 40)
    dens = hist[np.clip(np.digitize(s, edges) - 1, 0, 39)] / max(hist.max(), 1)
    yj = len(order) - 1 - r + rng.uniform(-1, 1, len(s)) * 0.36 * dens
    sc = ax2.scatter(s, yj, c=xn, cmap=cmap, vmin=0, vmax=1, s=3.2, alpha=0.85,
                     edgecolors="none", rasterized=True, zorder=2)
ax2.axvline(0, color=C["neutral_mid"], lw=0.7, zorder=1)
ax2.set_yticks(np.arange(len(order))[::-1], [LABEL[desc[j]] for j in order])
ax2.set_xlabel("SHAP value for $|e_t|$ (km$^2$)")
smin, smax = np.nanpercentile(S[:, order], [1, 99])
ax2.set_xlim(smin - 0.04 * (smax - smin), smax + 0.04 * (smax - smin))
ax2.set_ylim(-0.55, len(order) - 0.45)
style(ax2)
ax2.tick_params(axis="y", labelsize=6.9, pad=2)
ax2.set_title("b", loc="left", fontweight="bold", fontsize=9, pad=3)

cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(sc, cax=cax, ticks=[0, 1])
cb.ax.set_yticklabels(["Low", "High"])
cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5)
cb.set_label("Descriptor value", fontsize=6.4, labelpad=4)

# ---- shared legend, below both panels
lax = fig.add_subplot(gs[1, 0:2])
lax.set_axis_off()
handles = [
    Patch(facecolor=C["red_2"], edgecolor=C["red_strong"], lw=0.6, label="FP (M0)"),
    Patch(facecolor=C["blue_secondary"], alpha=0.55, edgecolor=C["blue_main"], lw=0.6, label="FN (M0)"),
    Line2D([], [], marker="D", ls="", color=C["neutral_black"], markersize=4.2, label="Net M0"),
    Line2D([], [], marker="o", ls="", mfc="white", mec=C["teal"], mew=1.1, markersize=4.6, label="Net M6"),
]
lax.legend(handles=handles, loc="center", ncol=4, frameon=False, handlelength=1.1,
           columnspacing=1.2, handletextpad=0.5)

fig.savefig(os.path.join(FIG, "fig_q1_error_drivers.pdf"), dpi=300, bbox_inches="tight",
            pad_inches=0.02)
print("wrote", os.path.join(FIG, "fig_q1_error_drivers.pdf"))
