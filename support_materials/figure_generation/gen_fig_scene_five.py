"""Single flood-peak scene annotated with the five problems.

No chained boxes or arrows between problems: the scene carries five numbered
markers, and a side key states what each problem looks at.  The detailed
technical route stays in Fig. 2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "figures" / "scenario_ai"
OUT = ROOT / "figures" / "scenario_options"
OUT.mkdir(exist_ok=True)

INK = "#24313A"
MUTED = "#5A6266"
AMBER = "#A46A2E"
BLUE = "#356F8E"
RED = "#B65441"
GRID = "#FFFFFF"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 5.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


def white_background(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    arr = np.array(im).astype(np.int16)
    near = (arr.min(axis=2) >= 224) & ((arr.max(axis=2) - arr.min(axis=2)) <= 16)
    lab, _ = ndimage.label(near)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    ids = set(np.unique(border)) - {0}
    arr[np.isin(lab, list(ids))] = 255
    return Image.fromarray(arr.astype(np.uint8))


def badge(ax, x, y, text, color, r=24):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white",
                        linewidth=1.1, zorder=14))
    ax.text(x, y - 2, text, ha="center", va="center", fontsize=5.4,
            color="white", fontweight="bold", zorder=15)


def draw_scene(ax):
    im = white_background(SUB / "after_flat.png")
    w, h = im.size
    ax.imshow(im, extent=(0, w, h, 0), interpolation="lanczos", zorder=1)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # P1: a 3 x 3 tile patch on the new-water margin
    x0, y0, cell = 1290, 540, 34
    for i in range(4):
        ax.plot([x0 + i * cell, x0 + i * cell], [y0, y0 + 3 * cell],
                color=GRID, lw=0.55, alpha=0.95, zorder=10)
        ax.plot([x0, x0 + 3 * cell], [y0 + i * cell, y0 + i * cell],
                color=GRID, lw=0.55, alpha=0.95, zorder=10)
    ax.add_patch(Rectangle((x0 + cell, y0 + cell), cell, cell,
                           facecolor=AMBER, edgecolor="none", alpha=0.22, zorder=9))
    badge(ax, 1315, 505, "P1", AMBER)

    # P2: four intensity classes as a compact strip on the flood
    gx, gy, gw, gh = 1180, 650, 28, 24
    gcols = ["#E7D2B1", "#D8B27A", "#BE8548", "#8F5524"]
    for i, c in enumerate(gcols):
        ax.add_patch(Rectangle((gx + i * (gw + 2), gy), gw, gh,
                               facecolor=c, edgecolor="white", linewidth=0.45, zorder=10))
    badge(ax, 1150, 622, "P2", AMBER)

    # P3: Sentinel-1 footprint frame and pass dates
    ax.add_patch(Rectangle((192, 205), 1495, 865, fill=False,
                           edgecolor="#73797C", linewidth=0.55,
                           linestyle=(0, (3.2, 3.2)), zorder=8))
    ax.text(760, 190, "S1 footprint · 27 Jun–16 Jul", fontsize=5.1, color=MUTED,
            ha="center", va="bottom", zorder=10)
    badge(ax, 215, 190, "P3", BLUE)

    # P4: exposure dots around the island village and terrace road
    pts = [(1040, 505), (1070, 525), (1095, 545), (1120, 565),
           (1150, 530), (1185, 550), (1215, 515), (1240, 540),
           (1065, 570), (1110, 590), (1175, 585), (1225, 575)]
    for x, y in pts:
        ax.add_patch(Circle((x, y), 5.2, facecolor=AMBER, edgecolor="white",
                            linewidth=0.45, zorder=11))
    badge(ax, 1080, 455, "P4", BLUE)

    # P5: depot to boat-only village, road leg and boat leg
    route = [(655, 620), (760, 606), (845, 585), (920, 555), (980, 515), (1060, 500)]
    ax.plot([p[0] for p in route], [p[1] for p in route],
            color="#5E666A", lw=0.85, ls=(0, (2.6, 2.0)), zorder=12)
    ax.plot([920, 980], [555, 515], color=BLUE, lw=0.95, ls=(0, (2.6, 2.0)), zorder=12)
    ax.add_patch(FancyArrowPatch((1040, 503), (1062, 500),
                                 arrowstyle="-|>", mutation_scale=6,
                                 color=BLUE, lw=0.8, zorder=12))
    badge(ax, 630, 590, "P5", BLUE)


def draw_key(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.97, "The five problems", fontsize=6.8, fontweight="bold",
            color=INK, va="top")

    entries = [
        ("P1", AMBER, "Recognise new water", "tiles → $F$"),
        ("P2", AMBER, "Fuse and grade", "four grades · $\\tau_A$"),
        ("P3", BLUE, "Transfer to Hengzhou", "frozen model · no labels"),
        ("P4", BLUE, "Assess exposure", "people · roads · facilities"),
        ("P5", BLUE, "Allocate relief", "cut roads · depot · boat"),
    ]
    ys = [0.845, 0.695, 0.545, 0.395, 0.245]
    for (num, col, title, sub), y in zip(entries, ys):
        ax.scatter([0.055], [y], s=86, marker="o", color=col, edgecolors="none",
                   transform=ax.transAxes, clip_on=False, zorder=5)
        ax.text(0.055, y - 0.004, num, transform=ax.transAxes,
                ha="center", va="center", fontsize=5.5, color="white",
                fontweight="bold")
        ax.text(0.145, y + 0.018, title, transform=ax.transAxes,
                fontsize=6.0, color=INK, fontweight="bold", va="center")
        ax.text(0.145, y - 0.032, sub, transform=ax.transAxes,
                fontsize=5.3, color=MUTED, va="center")


def main():
    fig_w_mm, fig_h_mm = 183.0, 100.0
    scene_w_mm, scene_h_mm = 140.0, 93.3
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    fig.patch.set_facecolor("white")

    ax_scene = fig.add_axes([1.5 / fig_w_mm, 2.0 / fig_h_mm,
                             scene_w_mm / fig_w_mm, scene_h_mm / fig_h_mm])
    draw_scene(ax_scene)

    ax_key = fig.add_axes([145.0 / fig_w_mm, 5.0 / fig_h_mm,
                           36.0 / fig_w_mm, 88.0 / fig_h_mm])
    draw_key(ax_key)

    out_pdf = OUT / "fig_scene_five.pdf"
    out_png = OUT / "fig_scene_five_preview.png"
    fig.savefig(out_pdf, dpi=600, facecolor="white")
    fig.savefig(out_png, dpi=300, facecolor="white")
    print("wrote", out_pdf, out_png)


if __name__ == "__main__":
    main()
