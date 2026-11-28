"""Figure: 3D exposure pillars over the 1 km layer of Problem 4.

Bar height is new water per cell; bar colour is the population living on that
new water. The window is the 36 x 36 km block carrying most of the new water.
Cells are aggregated to 2 km (block sum, totals preserved) for legibility.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
    "legend.frameon": False,
})

d = np.load("_tmp/v2/grid1km.npz")
A, P, land, ext = d["A"], d["P"], d["land"], d["extent"]
A = np.where(land > 0, np.nan_to_num(A), 0.0)
P = np.where(land > 0, np.nan_to_num(P), 0.0)

ny, nx = A.shape
ny2, nx2 = ny // 2, nx // 2
A2 = A[:ny2 * 2, :nx2 * 2].reshape(ny2, 2, nx2, 2).sum(axis=(1, 3))
P2 = P[:ny2 * 2, :nx2 * 2].reshape(ny2, 2, nx2, 2).sum(axis=(1, 3))

x0, x1, y0, y1 = ext / 1000.0
dx = (x1 - x0) / nx2
dy = (y1 - y0) / ny2

W = 14
if ny2 >= W and nx2 >= W:
    sw = sliding_window_view(A2, (W, W)).sum(axis=(2, 3))
    i0, j0 = np.unravel_index(sw.argmax(), sw.shape)
else:
    i0 = j0 = 0
    W = min(ny2, nx2)
A3 = A2[i0:i0 + W, j0:j0 + W]
P3 = P2[i0:i0 + W, j0:j0 + W]

xs = x0 + (np.arange(j0, j0 + W) + 0.5) * dx
ys = y1 - (np.arange(i0, i0 + W) + 0.5) * dy
XX, YY = np.meshgrid(xs, ys)

m = A3 > 0.015
xv, yv = XX[m], YY[m]
av, pv = A3[m], P3[m]

HMAX = 2.45
zv = HMAX * np.sqrt(av / av.max())
cmap = plt.get_cmap("OrRd")
norm = LogNorm(vmin=1.0, vmax=max(pv.max(), 2.0))
col = cmap(0.05 + 0.90 * norm(np.maximum(pv, 1.0)))

fig = plt.figure(figsize=(3.30, 3.05))
ax = fig.add_subplot(111, projection="3d")
ax.set_position([0.005, 0.02, 0.795, 0.96])

ax.bar3d(xs[0] - dx * 0.55, ys[-1] - dy * 0.55, -0.006,
         (xs[-1] - xs[0]) + dx * 1.1, (ys[0] - ys[-1]) + dy * 1.1, 0.006,
         color="0.915", shade=False, edgecolor="0.80", linewidth=0.25, zsort="min")

ax.bar3d(xv - dx * 0.44, yv - dy * 0.44, 0.0,
         dx * 0.88, dy * 0.88, zv,
         color=col, shade=False, edgecolor="0.30", linewidth=0.18, zsort="max")

ax.set_xlim(xs[0] - dx * 0.75, xs[-1] + dx * 0.75)
ax.set_ylim(ys[-1] - dy * 0.75, ys[0] + dy * 0.75)
ax.set_zlim(0, HMAX * 1.04)
ax.view_init(elev=41, azim=-61)
ax.set_box_aspect((1.18, 1.0, 0.56))

ax.set_xlabel("Easting (km)", labelpad=-2)
ax.set_ylabel("Northing (km)", labelpad=-2)
ax.set_xticks([280, 300, 320])
ax.set_yticks([2500, 2520, 2540])
ax.set_zticks([0, 1, 2])
ax.tick_params(pad=-1.5, labelsize=5.2)

for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)
ax.grid(False)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = fig.add_axes([0.845, 0.265, 0.020, 0.415])
cb = fig.colorbar(sm, cax=cax, ticks=[1, 10, 100, 400])
cb.ax.set_yticklabels(["1", "10", "100", "400"], fontsize=5.0)
cb.set_label("People per 2 km cell", fontsize=5.4, labelpad=3)
cb.ax.tick_params(length=1.4, pad=1, labelsize=5.0)
cb.outline.set_linewidth(0.5)

ax.text2D(0.03, 0.972, "tall $\\neq$ populated", transform=ax.transAxes,
          fontsize=6.4, color="0.25", va="top")

fig.savefig("figures/fig_q4_exposure3d.pdf", bbox_inches="tight")
fig.savefig("figures/fig_q4_exposure3d.png", dpi=400, bbox_inches="tight")
print("saved", A3.shape, int(m.sum()))
