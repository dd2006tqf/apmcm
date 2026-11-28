"""Figure 1: AI pictorial substrate + vector label overlay.

Substrate: two matched isometric cutaway scenes of the same valley
(before flood / during flood), generated as textless raster art.
Overlay: all panel letters, labels, leader lines and the road-cut mark
are drawn as vector text/geometry with matplotlib, so the final PDF has
editable, resolution-independent typography.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "figures" / "scenario_ai"

# Canvas / crop (substrate pixels). Crop keeps the block large and leaves
# enough breathing room for the outer labels.
CROP = (110, 145, 1870, 1125)

PANELS = [
    ("before_raw.png", "a", "Before the flood", "before"),
    ("after_v2_raw.png", "b", "During the flood", "after"),
]

INK = "#24313A"
WATER = "#2E5F77"
ALERT = "#A5433B"
AMBER = "#96551F"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 5.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def _leader(ax, text, xytext, xy, color=INK, fontsize=5.9, ha="left", va="center"):
    """One label with a thin leader arrow and a white legibility halo."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="data",
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        zorder=6,
        linespacing=1.15,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=0.55,
            shrinkA=2.5,
            shrinkB=3.0,
            mutation_scale=5.0,
            connectionstyle="arc3,rad=0.0",
        ),
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.35", fc="white", ec="none", alpha=0.90),
    )


def _plain(ax, text, xy, color=INK, fontsize=5.9, ha="left", va="center", weight="normal"):
    ax.text(
        xy[0],
        xy[1],
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        zorder=7,
        linespacing=1.15,
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.35", fc="white", ec="none", alpha=0.90),
    )


def _road_cut(ax, x, y):
    """Small dashed red cross on the washed-out road segment."""
    for dx, dy in ((1, 1), (1, -1)):
        ax.plot(
            [x - 13 * dx, x + 13 * dx],
            [y - 13 * dy, y + 13 * dy],
            color=ALERT,
            lw=1.0,
            ls=(0, (2.0, 1.6)),
            zorder=8,
            solid_capstyle="round",
        )


def _draw_panel(ax, image_path: Path, state: str):
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    ax.imshow(im, extent=(0, w, h, 0), interpolation="lanczos", zorder=1)
    ax.set_xlim(CROP[0], CROP[2])
    ax.set_ylim(CROP[3], CROP[1])  # inverted y: pixel coordinates read downward
    ax.set_aspect("equal")
    ax.axis("off")

    if state == "before":
        # Channel width: label in the upper-left air, leader to the narrow upstream reach.
        _leader(ax, "channel $W_0$", (380, 300), (575, 462), color=WATER)
        # Valley road: label in the open lower meadow, leader to the main road near the depot.
        _leader(ax, "valley road", (880, 735), (825, 680), color=INK)
        # Terrace settlement: label along the upper-right slope, leader to the house cluster.
        _leader(ax, "terrace settlement", (1350, 390), (1235, 505), color=INK)
        # Depot: label left of the amber-roofed building.
        _leader(ax, "depot", (450, 545), (575, 600), color=AMBER)
    else:
        # New inundation: label on the dry right bank, leader to the expanded water margin.
        _leader(ax, "new inundation $F$", (1415, 720), (1295, 665), color=WATER)
        # Road cut: label on the lower meadow, leader to the washed-out bridge approach.
        _leader(ax, "road cut", (920, 775), (800, 640), color=ALERT)
        _road_cut(ax, 792, 630)
        # Island village label on the water; two lines keep it compact.
        _leader(
            ax,
            "island village\nboat only",
            (1010, 405),
            (1130, 485),
            color=INK,
            ha="left",
            va="center",
        )
        # Boat label, short leader to the tiny rowboat left of the village.
        _leader(ax, "boat", (850, 500), (940, 525), color=WATER)


def main():
    fig_w_mm = 183.0
    panel_w_mm = 89.0
    gutter_mm = 3.0
    side_mm = 1.0
    bottom_mm = 2.0

    crop_w = CROP[2] - CROP[0]
    crop_h = CROP[3] - CROP[1]
    panel_h_mm = panel_w_mm * crop_h / crop_w
    top_pad_mm = 4.2
    fig_h_mm = bottom_mm + panel_h_mm + top_pad_mm

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    for i, (fname, letter, title, state) in enumerate(PANELS):
        x_mm = side_mm + i * (panel_w_mm + gutter_mm)
        y_mm = bottom_mm
        ax = fig.add_axes(
            [
                x_mm / fig_w_mm,
                y_mm / fig_h_mm,
                panel_w_mm / fig_w_mm,
                panel_h_mm / fig_h_mm,
            ]
        )
        _draw_panel(ax, SUB / fname, state)

        # Panel letter + short state title sit in the white strip above each panel.
        fig.text(
            (x_mm + 1.0) / fig_w_mm,
            (y_mm + panel_h_mm + 2.0) / fig_h_mm,
            letter,
            fontsize=8.5,
            fontweight="bold",
            color=INK,
            ha="left",
            va="center",
        )
        fig.text(
            (x_mm + 5.0) / fig_w_mm,
            (y_mm + panel_h_mm + 2.0) / fig_h_mm,
            title,
            fontsize=6.8,
            color=INK,
            ha="left",
            va="center",
        )

    out_pdf = SUB / "fig_scenario_ai.pdf"
    out_png = SUB / "fig_scenario_ai_preview.png"
    fig.savefig(out_pdf, dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(out_png, dpi=300, bbox_inches=None, pad_inches=0)
    print(f"wrote {out_pdf} ({fig_w_mm:.1f} x {fig_h_mm:.1f} mm)")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
