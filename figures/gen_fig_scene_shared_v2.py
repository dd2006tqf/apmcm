"""Option 1 Figure 1: flat axonometric cutaway with margin callouts.

Substrate: flat-vector-look isometric scene (no embedded text).
Labels: vector text in the white margins, with thin leader lines; no
white label plates over the drawing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Circle
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "scenario_ai"
OUT = ROOT / "scenario_options"
OUT.mkdir(exist_ok=True)

INK = "#24313A"
WATER = "#2E5F77"
ALERT = "#A5433B"
AMBER = "#96551F"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 5.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def leader(ax, text, xytext, xy, color=INK, fontsize=5.8, ha="left", va="center"):
    ax.annotate(
        text, xy=xy, xytext=xytext,
        ha=ha, va=va, fontsize=fontsize, color=color, zorder=8,
        linespacing=1.15,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.50,
                        shrinkA=2.0, shrinkB=2.5, mutation_scale=4.5),
    )


def _white_background(image_path):
    """Replace the border-connected off-white paper with pure white."""
    im = Image.open(image_path).convert("RGB")
    arr = np.array(im).astype(np.int16)
    near = (arr.min(axis=2) >= 224) & ((arr.max(axis=2) - arr.min(axis=2)) <= 16)
    lab, n = ndimage.label(near)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    ids = set(np.unique(border)) - {0}
    mask = np.isin(lab, list(ids))
    arr[mask] = 255
    return Image.fromarray(arr.astype(np.uint8))


def draw_panel(ax, image_path, state):
    im = _white_background(image_path)
    w, h = im.size
    ax.imshow(im, extent=(0, w, h, 0), interpolation="lanczos", zorder=1)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    if state == "before":
        leader(ax, "channel $W_0$", (80, 150), (575, 462), color=WATER)
        leader(ax, "terrace settlement", (1345, 150), (1235, 505), color=INK)
        leader(ax, "depot", (90, 610), (648, 620), color=AMBER)
        leader(ax, "valley road", (865, 1180), (1045, 752), color=INK)
    else:
        leader(ax, "new inundation $F$", (1465, 1180), (1282, 568), color=WATER)
        leader(ax, "road cut", (790, 1180), (835, 655), color=ALERT)
        # clean ring around the washed-out road segment
        ax.add_patch(Circle((838, 655), 28, fill=False, edgecolor=ALERT,
                            lw=0.85, zorder=9))
        leader(ax, "island village\nboat only", (1040, 115), (1130, 485), color=INK)


def main():
    fig_w_mm, panel_w_mm, gutter_mm, side_mm = 183.0, 89.0, 3.0, 1.0
    w, h = Image.open(SUB / "before_flat.png").size
    panel_h_mm = panel_w_mm * h / w
    top_mm, bottom_mm = 4.0, 1.5
    fig_h_mm = top_mm + panel_h_mm + bottom_mm

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    fig.patch.set_facecolor("white")
    for i, (fname, letter, title, state) in enumerate([
        ("before_flat.png", "a", "Before the flood", "before"),
        ("after_flat.png", "b", "During the flood", "after"),
    ]):
        x_mm = side_mm + i * (panel_w_mm + gutter_mm)
        ax = fig.add_axes([x_mm / fig_w_mm, bottom_mm / fig_h_mm,
                          panel_w_mm / fig_w_mm, panel_h_mm / fig_h_mm])
        draw_panel(ax, SUB / fname, state)
        fig.text((x_mm + 1.0) / fig_w_mm,
                 (bottom_mm + panel_h_mm + 2.0) / fig_h_mm,
                 letter, fontsize=8.5, fontweight="bold", color=INK, va="center")
        fig.text((x_mm + 5.0) / fig_w_mm,
                 (bottom_mm + panel_h_mm + 2.0) / fig_h_mm,
                 title, fontsize=7.0, fontweight="bold", color=INK, va="center")

    out_pdf = OUT / "scene_shared_v2.pdf"
    out_png = OUT / "scene_shared_v2_preview.png"
    fig.savefig(out_pdf, dpi=600, facecolor="white")
    fig.savefig(out_png, dpi=300, facecolor="white")
    print("wrote", out_pdf, out_png)


if __name__ == "__main__":
    main()
