# -*- coding: utf-8 -*-
"""Compact valley cross-section scene: dry vs flood (road cut, island village)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.font_manager as fm

serif = "Times New Roman" if any("Times New Roman" in f.name for f in fm.fontManager.ttflist) else "DejaVu Serif"
plt.rcParams.update({"font.family": serif, "mathtext.fontset": "stix", "pdf.fonttype": 42})

LAND = "#f2ede0"; LAND_EDGE = "#b3a78b"
WATER = "#14487d"; FLOOD = "#79aede"
ROAD = "#2f2f2f"; DEPOT = "#b8860b"; RED = "#b00000"

T = np.array([
    (0.00, 2.95), (0.30, 2.62), (0.60, 2.14), (0.90, 1.52), (1.15, 1.10),
    (1.45, 0.92), (2.10, 0.84), (2.55, 0.84), (2.80, 0.36), (3.00, 0.16),
    (3.20, 0.36), (3.45, 0.84), (4.05, 0.86), (4.35, 0.98), (4.55, 1.36),
    (4.72, 1.58), (4.88, 1.36), (5.05, 0.98), (5.40, 0.94), (5.60, 1.10),
    (5.80, 1.90), (5.95, 2.50), (6.00, 2.80)
])

def ty(x): return np.interp(x, T[:, 0], T[:, 1])

def panel(ax, title):
    ax.set_xlim(-0.02, 6.02); ax.set_ylim(0.02, 3.10); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.04, 3.06, title, fontsize=11.3, ha="left", va="top", weight="bold")

def land(ax):
    xs = np.linspace(0, 6, 900); ys = ty(xs)
    ax.add_patch(Polygon(np.vstack([np.column_stack([xs, ys]), [6, 0.02], [0, 0.02]]),
                         closed=True, facecolor=LAND, edgecolor=LAND_EDGE, lw=0.8, zorder=1))

def water(ax, level):
    xs = np.linspace(0, 6, 1500); ys = ty(xs)
    m = ys < level
    if not m.any(): return
    # split into contiguous runs so that islands are not bridged by a chord
    idx = np.where(m)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for r in runs:
        if len(r) < 2: continue
        poly = np.vstack([np.column_stack([xs[r], ys[r]]),
                          [xs[r][-1], level], [xs[r][0], level]])
        ax.add_patch(Polygon(poly, closed=True, facecolor=FLOOD, alpha=0.66,
                             edgecolor="none", zorder=2))
        ax.plot([xs[r][0], xs[r][-1]], [level, level], color=WATER, lw=1.3, zorder=3)

def road(ax, y, x0=1.05, x1=5.45, z=5):
    ax.plot([x0, x1], [y, y], color=ROAD, lw=3.2, solid_capstyle="butt", zorder=z)
    ax.plot([x0, x1], [y, y], color="#ededed", lw=1.5, solid_capstyle="butt", zorder=z + 0.5)

def house(ax, x, y, s=0.07):
    ax.add_patch(Rectangle((x - s, y), 2 * s, 1.2 * s, facecolor="white", edgecolor="#2f2f2f", lw=0.6, zorder=8))

fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.36))

# -------- (a) --------
ax = axs[0]
panel(ax, "(a) Before the flood")
land(ax); water(ax, 0.40); road(ax, 0.92); house(ax, 4.72, 1.58)
ax.add_patch(Rectangle((0.70, 2.02), 0.17, 0.13, facecolor=DEPOT, alpha=0.9, edgecolor=DEPOT, lw=0.7, zorder=8))
ax.annotate("channel $W_0$", xy=(3.00, 0.42), xytext=(2.15, 1.55), fontsize=8.3, color="#0f3a66",
            arrowprops=dict(arrowstyle="-", color=WATER, lw=0.55))
ax.annotate("valley road", xy=(1.40, 0.92), xytext=(1.05, 0.32), fontsize=8.1, color="#222222",
            arrowprops=dict(arrowstyle="-", color="#222222", lw=0.5))
ax.annotate("terrace settlement", xy=(4.72, 1.68), xytext=(5.02, 2.30), fontsize=8.1, color="#222222",
            ha="right", arrowprops=dict(arrowstyle="-", color="#222222", lw=0.5))
ax.annotate("depot", xy=(0.79, 2.16), xytext=(1.02, 2.62), fontsize=8.1, color=DEPOT,
            arrowprops=dict(arrowstyle="-", color=DEPOT, lw=0.5))
ax.text(5.97, 0.12, "passable", fontsize=7.6, color="#555555", ha="right")

# -------- (b) --------
ax = axs[1]
panel(ax, "(b) During the flood")
land(ax); road(ax, 0.92, z=1.4); water(ax, 1.28); house(ax, 4.72, 1.58)
ax.add_patch(Rectangle((0.70, 2.02), 0.17, 0.13, facecolor=DEPOT, alpha=0.9, edgecolor=DEPOT, lw=0.7, zorder=8))
ax.plot([3.56, 3.72], [0.84, 1.00], color=RED, lw=1.5, zorder=10)
ax.plot([3.56, 3.72], [1.00, 0.84], color=RED, lw=1.5, zorder=10)
ax.add_patch(Circle((4.72, 1.64), 0.28, facecolor="none", edgecolor=RED, lw=1.0, ls=(0, (3, 2)), zorder=10))
ax.annotate("new inundation $F$", xy=(2.30, 1.10), xytext=(1.55, 0.30), fontsize=8.5, color="#0f3a66",
            arrowprops=dict(arrowstyle="-", color=WATER, lw=0.5))
ax.annotate("road cut", xy=(3.64, 0.92), xytext=(3.30, 0.16), fontsize=8.0, color=RED,
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.5))
ax.annotate("island village:\nboat only", xy=(4.94, 1.46), xytext=(5.32, 0.26), fontsize=7.9,
            color=RED, ha="center", arrowprops=dict(arrowstyle="-", color=RED, lw=0.5))
boat = Path([(1.14, 1.31), (2.90, 1.34), (4.56, 1.55)],
            [Path.MOVETO, Path.CURVE3, Path.CURVE3])
ax.add_patch(FancyArrowPatch(path=boat, arrowstyle="-|>", mutation_scale=7, color="#0f3a66",
                             lw=0.9, ls=(0, (4, 2)), zorder=11))

plt.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004, wspace=0.03)
out = r"E:\华数数据\workspaces\f7ba95ca861e\figures\fig_scenario.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.replace(".pdf", "_preview.png"), dpi=260, bbox_inches="tight", pad_inches=0.02)
print("saved")
