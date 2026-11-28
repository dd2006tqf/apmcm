"""Figure: 1 km exposure layer for Problem 4 (new inundation vs flooded population)."""
import sys
sys.path.insert(0, r"C:\Users\40461\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.user_skills\nature-figure\scripts")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

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

d = np.load("_tmp/v2/grid1km.npz")
A, P, land, ext = d["A"], d["P"], d["land"], d["extent"]
A = np.where(land > 0, A, np.nan)
P = np.where(land > 0, P, np.nan)
x0, x1, y0, y1 = ext / 1000.0

cm1 = plt.get_cmap("Blues").copy(); cm1.set_under("#f0f0f0")
cm2 = plt.get_cmap("OrRd").copy(); cm2.set_under("#f0f0f0")

fig = plt.figure(figsize=(6.30, 2.52))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.20,
                      left=0.075, right=0.985, bottom=0.30, top=0.91)

ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(np.where(np.isnan(A), np.nan, np.maximum(A, 1e-9)),
                 extent=[x0, x1, y0, y1], origin="upper", cmap=cm1,
                 norm=LogNorm(vmin=0.004, vmax=0.36), interpolation="nearest")
ax1.set_aspect("equal")
ax1.set_xlabel("Easting (km)", labelpad=1.2)
ax1.set_ylabel("Northing (km)", labelpad=1.2)
ax1.set_xticks([280, 300, 320]); ax1.set_yticks([2490, 2510, 2530, 2550])
ax1.tick_params(length=2, pad=1.2, labelsize=6)
ax1.set_xlim(x0, x1); ax1.set_ylim(y0, y1); ax1.set_autoscale_on(False)
ax1.text(-0.14, 1.00, "(A)", transform=ax1.transAxes, fontsize=9, fontweight="bold", va="bottom")
ax1.add_patch(Rectangle((x0 + 1.5, y0 + 3.5), 10, 1.3, facecolor="white", edgecolor="0.25", lw=0.5, zorder=5))
ax1.text(x0 + 12.5, y0 + 4.15, "10 km", ha="left", va="center", fontsize=5.5, zorder=6)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(np.where(np.isnan(P), np.nan, np.maximum(P, 1e-9)),
                 extent=[x0, x1, y0, y1], origin="upper", cmap=cm2,
                 norm=LogNorm(vmin=1, vmax=470), interpolation="nearest")
ax2.set_aspect("equal")
ax2.set_xlabel("Easting (km)", labelpad=1.2)
ax2.set_ylabel("Northing (km)", labelpad=1.2, color="white")
ax2.set_xticks([280, 300, 320]); ax2.set_yticks([])
ax2.tick_params(length=2, pad=1.2, labelsize=6)
ax2.set_xlim(x0, x1); ax2.set_ylim(y0, y1); ax2.set_autoscale_on(False)
ax2.text(-0.14, 1.00, "(B)", transform=ax2.transAxes, fontsize=9, fontweight="bold", va="bottom")

ax3 = fig.add_subplot(gs[0, 2])
m = np.isfinite(A) & (A > 0) & np.isfinite(P) & (P > 0)
av, pv = A[m], P[m]
ax3.scatter(av, pv, s=3.2, c="#6b6b6b", alpha=0.45, lw=0)
topA = np.argsort(av)[-10:]
topP = np.argsort(pv)[-10:]
ax3.scatter(av[topA], pv[topA], s=20, facecolors="none", edgecolors="#14406b", lw=0.9, zorder=4)
ax3.scatter(av[topP], pv[topP], s=20, facecolors="none", edgecolors="#a63d00", lw=0.9, zorder=5)
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.set_xlim(0.004, 0.5); ax3.set_ylim(0.8, 700)
ax3.set_xlabel("New water (km$^2$ / cell)", labelpad=1.2)
ax3.set_ylabel("People / cell", labelpad=1.2)
ax3.text(-0.19, 1.00, "(C)", transform=ax3.transAxes, fontsize=9, fontweight="bold", va="bottom")
ax3.text(0.97, 0.06, r"$\rho_S = 0.36$", transform=ax3.transAxes, fontsize=7, ha="right")
ax3.tick_params(length=2, pad=1.2, labelsize=6)

fig.canvas.draw()
p1, p2 = ax1.get_position(), ax2.get_position()
cax1 = fig.add_axes([p1.x0, p1.y0 - 0.115, p1.width, 0.018])
cax2 = fig.add_axes([p2.x0, p2.y0 - 0.115, p2.width, 0.018])
cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal", ticks=[0.005, 0.35])
cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal", ticks=[2, 450])
for cb, lab in ((cb1, ("0.005", "0.36")), (cb2, ("2", "470"))):
    cb.ax.set_xticklabels(lab, fontsize=5.2)
    cb.ax.tick_params(length=1.5, pad=1, labelsize=5.2)
    cb.outline.set_linewidth(0.5)
cb1.set_label("New water (km$^2$ per cell)", fontsize=5.5, labelpad=1)
cb2.set_label("People per flooded cell", fontsize=5.5, labelpad=1)

from audit_panel_alignment import require_matplotlib_panel_alignment, matplotlib_layout_manifest
mf = matplotlib_layout_manifest(fig, exclude_axes=(cax1, cax2))
for p in mf["panels"]:
    print(p["id"], [round(v, 1) for v in p["bbox_pt"]])
require_matplotlib_panel_alignment(
    fig, json_out="_tmp/v2/fig_q4_grid1km.alignment.json",
    overlay_svg="_tmp/v2/fig_q4_grid1km.alignment.svg",
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False,
    exclude_axes=(cax1, cax2),
    row_groups=[{"id": "maps", "panels": ["a", "b"]}],
    exemptions=[{
        "panels": ["c"], "checks": ["panel-width", "horizontal-gutter", "vertical-gutter"],
        "reason": "scatter panel has a different inferential role (joint distribution, "
                  "not a map) and cannot share the equal-aspect map width",
    }],
)
fig.savefig("figures/fig_q4_grid1km.pdf", bbox_inches="tight")
fig.savefig("figures/fig_q4_grid1km.png", dpi=400, bbox_inches="tight")
print("saved")
