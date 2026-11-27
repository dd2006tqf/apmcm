# -*- coding: utf-8 -*-
"""Figure 1: 3D cut-away valley block, dry vs flood."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Ellipse
from matplotlib.path import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

# Muted, print-safe palette
LAND_TOP = {"left": "#EADFC7", "slope": "#E2D3B6", "bed": "#D8D0C1",
            "right": "#E7D9BE", "hill": "#D9C9AA"}
LAND_FRONT = "#D5C3A5"
LAND_EDGE = "#9E8A6B"
WATER = "#2E6DA4"
FLOOD_FILL = "#8FBCE3"
ROAD = "#2B2B2B"; ROAD_LIGHT = "#F2F2F2"
DEPOT = "#C58B2A"
RED = "#B64342"
HOUSE = "#FFFFFF"; HOUSE_EDGE = "#353535"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.linewidth": 0.7,
})

T = np.array([
    (0.00, 2.95), (0.30, 2.62), (0.60, 2.14), (0.90, 1.52), (1.15, 1.10),
    (1.45, 0.92), (2.10, 0.84), (2.55, 0.84), (2.80, 0.36), (3.00, 0.16),
    (3.20, 0.36), (3.45, 0.84), (4.05, 0.86), (4.35, 0.98), (4.55, 1.36),
    (4.72, 1.58), (4.88, 1.36), (5.05, 0.98), (5.40, 0.94), (5.60, 1.10),
    (5.80, 1.90), (5.95, 2.50), (6.00, 2.80)
])

# oblique projection: y is depth into the page, rising up-right
KX, KY, YD, ZS = 0.36, 0.45, 3.60, 0.70


def P(x, y, z):
    return (x + KX * y, z * ZS + KY * y)


def ty(x):
    return np.interp(x, T[:, 0], T[:, 1])


def poly(ax, pts, fc, ec="none", lw=0.0, z=1, alpha=1.0):
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=z, alpha=alpha))


def front_face(ax):
    xs = np.linspace(0, 6, 500)
    zz = ty(xs)
    pts = [P(x, 0, z) for x, z in zip(xs, zz)]
    pts += [P(6, 0, 0), P(0, 0, 0)]
    poly(ax, pts, LAND_FRONT, LAND_EDGE, 0.8, z=4)
    # one subtle bedding line on the cut face
    for z0 in (0.65, 1.20):
        ax.plot([P(0, 0, z0)[0], P(6, 0, z0)[0]],
                [P(0, 0, z0)[1], P(6, 0, z0)[1]],
                color="#BBA98A", lw=0.35, alpha=0.55, zorder=4.1)


def side_wall(ax, x=6.0, z=1.7):
    pts = [P(x, 0, 0), P(x, YD, 0), P(x, YD, ty(x)), P(x, 0, ty(x))]
    poly(ax, pts, "#D8C7A8", LAND_EDGE, 0.55, z=z)
    ax.plot(*zip(*[P(x, YD, 0), P(x, YD, ty(x))]),
            color=LAND_EDGE, lw=0.5, zorder=z + 0.1)


def top_surface(ax):
    zones = [(0.0, 1.6, LAND_TOP["left"]),
             (1.6, 2.6, LAND_TOP["slope"]),
             (2.6, 3.4, LAND_TOP["bed"]),
             (3.4, 5.5, LAND_TOP["right"]),
             (5.5, 6.0, LAND_TOP["hill"])]
    for a, b, fc in zones:
        xs = np.linspace(a, b, 90)
        front = [P(x, 0, ty(x)) for x in xs]
        back = [P(x, YD, ty(x)) for x in xs]
        poly(ax, front + back[::-1], fc, z=2)
    # top outlines: front, back and the two end edges
    back = [P(x, YD, ty(x)) for x in np.linspace(0, 6, 300)]
    ax.plot(*zip(*back), color=LAND_EDGE, lw=0.55, alpha=0.75, zorder=2.2)
    front = [P(x, 0, ty(x)) for x in np.linspace(0, 6, 300)]
    ax.plot(*zip(*front), color=LAND_EDGE, lw=0.7, zorder=4.2)
    for x in (0, 6):
        ax.plot(*zip(*[P(x, 0, ty(x)), P(x, YD, ty(x))]),
                color=LAND_EDGE, lw=0.55, alpha=0.7, zorder=2.2)


def water_plane(ax, level, z=5.0):
    xs = np.linspace(0, 6, 1200)
    m = ty(xs) < level
    idx = np.where(m)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for r in runs:
        if len(r) < 2:
            continue
        x0, x1 = xs[r[0]], xs[r[-1]]
        # vertical water face at the cut front
        poly(ax, [P(x0, 0, ty(x0)), P(x1, 0, ty(x1)),
                  P(x1, 0, level), P(x0, 0, level)],
             "#6FA8D6", WATER, 0.45, z=z - 0.35, alpha=0.82)
        # horizontal water surface
        poly(ax, [P(x0, 0, level), P(x1, 0, level),
                  P(x1, YD, level), P(x0, YD, level)],
             FLOOD_FILL, WATER, 0.7, z=z, alpha=0.58)
        # waterline at the front edge
        ax.plot([P(x0, 0, level)[0], P(x1, 0, level)[0]],
                [P(x0, 0, level)[1], P(x1, 0, level)[1]],
                color=WATER, lw=1.3, zorder=z + 0.2)


def road(ax, z_road, y0=0.25, y1=3.15, z=6.0):
    ys = np.linspace(y0, y1, 200)
    pts = [P(2.15, y, z_road) for y in ys]
    ax.plot(*zip(*pts), color=ROAD, lw=3.5, solid_capstyle="butt", zorder=z)
    ax.plot(*zip(*pts), color=ROAD_LIGHT, lw=1.5, solid_capstyle="butt", zorder=z + 0.1)


def box(ax, x, y, z, hx, hy, h, fc, ec, lw=0.6, zo=9, roof=False):
    b = [P(x - hx, y - hy, z), P(x + hx, y - hy, z),
         P(x + hx, y + hy, z), P(x - hx, y + hy, z)]
    t = [P(x - hx, y - hy, z + h), P(x + hx, y - hy, z + h),
         P(x + hx, y + hy, z + h), P(x - hx, y + hy, z + h)]
    poly(ax, [b[0], b[1], t[1], t[0]], fc, ec, lw, z=zo)
    poly(ax, [b[1], b[2], t[2], t[1]], "#E6E6E6", ec, lw, z=zo)
    poly(ax, [t[0], t[1], t[2], t[3]], "#F7F7F7", ec, lw, z=zo + 0.1)
    if roof:
        ridge = [P(x, y - hy, z + h + 0.13), P(x, y + hy, z + h + 0.13)]
        poly(ax, [t[0], t[1], ridge[0]], "#8B5E3C", ec, lw, z=zo + 0.2)
        poly(ax, [t[3], t[2], ridge[1]], "#A87650", ec, lw, z=zo + 0.2)
        ax.plot([ridge[0][0], ridge[1][0]], [ridge[0][1], ridge[1][1]],
                color=ec, lw=lw, zorder=zo + 0.25)


def settlement(ax, z0):
    for (x, y, s, h, roof) in [(4.66, 1.82, 0.09, 0.17, True),
                               (4.82, 2.03, 0.07, 0.14, True),
                               (4.58, 2.12, 0.06, 0.12, False)]:
        box(ax, x, y, z0, s, s * 1.1, h, HOUSE, HOUSE_EDGE, 0.55, 9, roof)


def depot(ax, z0):
    box(ax, 0.80, 2.55, z0, 0.15, 0.11, 0.18, DEPOT, "#7A571E", 0.6, 9)
    ax.plot(*zip(*[P(0.66, 2.44, z0 + 0.18), P(0.94, 2.66, z0 + 0.18)]),
            color="white", lw=0.8, zorder=9.2)


def panel(ax, letter, title):
    ax.set_xlim(-0.12, 7.55)
    ax.set_ylim(-0.12, 4.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.00, 4.00, letter, fontsize=10.2, ha="left", va="top",
            weight="bold", color="#1A1A1A")
    ax.text(0.58, 4.00, title, fontsize=9.4, ha="left", va="top",
            weight="normal", color="#1A1A1A")


def labels_before(ax):
    ax.text(*P(2.70, 2.80, 2.90), "channel $W_0$", fontsize=8.2,
            color="#17466F", ha="center", va="bottom", zorder=12)
    ax.plot([P(2.70, 2.80, 2.90)[0], P(3.00, 1.70, 0.41)[0]],
            [P(2.70, 2.80, 2.90)[1] - 0.07, P(3.00, 1.70, 0.41)[1]],
            color="#17466F", lw=0.5, zorder=11.5)
    ax.annotate("valley road", xy=P(2.15, 1.10, 0.86), xytext=P(1.30, 0.75, 1.15),
                fontsize=8.0, color="#222222", ha="center", zorder=12,
                arrowprops=dict(arrowstyle="-", color="#222222", lw=0.5,
                                shrinkA=2, shrinkB=1, zorder=12))
    ax.annotate("terrace\nsettlement", xy=P(4.72, 1.95, 1.72), xytext=P(4.20, 2.20, 2.95),
                fontsize=7.9, color="#222222", ha="center", va="bottom", zorder=12,
                arrowprops=dict(arrowstyle="-", color="#222222", lw=0.5,
                                shrinkA=2, shrinkB=1, zorder=12))
    ax.text(*P(1.70, 2.80, 3.05), "depot", fontsize=7.9, color="#7A571E",
            ha="center", va="bottom", zorder=12)
    ax.plot([P(1.70, 2.80, 3.05)[0], P(0.80, 2.55, 2.33)[0]],
            [P(1.70, 2.80, 3.05)[1] - 0.06, P(0.80, 2.55, 2.33)[1]],
            color="#7A571E", lw=0.5, zorder=11.5)


def labels_flood(ax):
    halo = [pe.withStroke(linewidth=1.1, foreground="white")]
    ax.annotate("new inundation $F$", xy=P(2.10, 0.60, 1.42), xytext=P(1.75, 1.20, 2.45),
                fontsize=8.2, color="#17466F", ha="center", zorder=12, path_effects=halo,
                arrowprops=dict(arrowstyle="-", color="#17466F", lw=0.55,
                                shrinkA=2, shrinkB=1, zorder=12))
    ax.annotate("road cut", xy=P(2.15, 1.55, 1.42), xytext=P(2.95, 1.65, 2.50),
                fontsize=7.9, color=RED, ha="center", zorder=12, path_effects=halo,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5,
                                shrinkA=2, shrinkB=1, zorder=12))
    ax.annotate("island village:\nboat only", xy=P(4.75, 1.92, 1.66),
                xytext=P(4.95, 2.35, 3.25), fontsize=7.7, color=RED,
                ha="center", va="bottom", zorder=12, path_effects=halo,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5,
                                shrinkA=2, shrinkB=1, zorder=12))
    # boat path on the water surface
    start, end = P(2.55, 0.35, 1.42), P(4.35, 1.75, 1.42)
    ax.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=-0.10",
                                 arrowstyle="-|>", mutation_scale=7,
                                 color="#17466F", lw=0.8, ls=(0, (4, 2)), zorder=11))
    ax.text(*P(3.55, 0.25, 1.30), "boat", fontsize=7.3, color="#17466F",
            ha="center", zorder=11, path_effects=halo)
    ax.add_patch(Ellipse(P(4.75, 1.92, 1.66), width=0.82, height=0.34,
                         facecolor="none", edgecolor=RED, lw=1.0,
                         ls=(0, (3, 2)), zorder=10.5))


def draw_panel(ax, letter, title, flood=False):
    panel(ax, letter, title)
    front_face(ax)
    side_wall(ax)
    top_surface(ax)
    depot(ax, 2.16)
    z_road = ty(2.15)
    road(ax, z_road)
    if flood:
        water_plane(ax, 1.42, z=6.2)
        # red cross where the road is cut
        c = P(2.15, 1.55, 1.42)
        s = 0.10
        ax.plot([c[0] - s, c[0] + s], [c[1] - s * 0.65, c[1] + s * 0.65],
                color=RED, lw=1.4, zorder=12)
        ax.plot([c[0] - s, c[0] + s], [c[1] + s * 0.65, c[1] - s * 0.65],
                color=RED, lw=1.4, zorder=12)
        settlement(ax, ty(4.72))
        labels_flood(ax)
    else:
        water_plane(ax, 0.40, z=6.2)
        settlement(ax, ty(4.72))
        labels_before(ax)


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.75))
fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.02)
draw_panel(axs[0], "a", "Before the flood", flood=False)
draw_panel(axs[1], "b", "During the flood", flood=True)

# Render-time multi-panel alignment gate (optional locally; never blocks drawing
# on machines that do not carry the Nature-figure skill package).
skill_scripts = os.path.join(
    os.environ.get("USERPROFILE", ""),
    r"AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace"
    r"\.user_skills\nature-figure\scripts")
if os.path.isdir(skill_scripts):
    sys.path.insert(0, skill_scripts)
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig,
            json_out=os.path.join(FIG, "fig_scenario.alignment.json"),
            overlay_svg=os.path.join(FIG, "fig_scenario.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    except Exception as exc:
        print(f"alignment gate: {exc}")

out = os.path.join(FIG, "fig_scenario.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.replace(".pdf", ".svg"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.replace(".pdf", "_preview.png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
print("saved")

