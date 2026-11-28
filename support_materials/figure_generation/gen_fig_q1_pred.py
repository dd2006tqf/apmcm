# -*- coding: utf-8 -*-
"""Contact sheet of the exported per-tile masks in output/q1_pred.

Five held-out tiles of increasing flood share, baseline A (top) and the
Problem 1 LightGBM model (bottom).  Land light grey, new water blue, the
same palette as fig_q1_visual_cmp.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
from PIL import Image

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["mathtext.fontset"] = "stix"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.size"] = 8

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "figures")

CMAP = ListedColormap(["#dedede", "#2f6f9f"])

res = json.load(open(os.path.join(OUT, "q1_results.json")))
ids = [int(i) for i in res["tile_ids"]]
share = {t: float(s) for t, s in zip(ids, res["tile_flood_share"])}

TILES = [1720, 1252, 1077, 1972, 1426]
ROWS = [("A", "Baseline A"), ("B", "LightGBM M0")]

fig, axes = plt.subplots(2, 5, figsize=(6.45, 2.95))
fig.subplots_adjust(left=0.085, right=0.995, top=0.845, bottom=0.115,
                    wspace=0.10, hspace=0.30)

for j, t in enumerate(TILES):
    for i, (pre, name) in enumerate(ROWS):
        ax = axes[i, j]
        img = np.array(Image.open(os.path.join(OUT, "q1_pred", f"{pre}_{t}.png")))
        wet = float((img > 127).mean())
        ax.imshow(img, cmap=CMAP, vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color("0.55")
        ax.text(0.5, -0.07, f"$\\hat\\rho$ = {wet * 100:.1f}%",
                transform=ax.transAxes, ha="center", va="top", fontsize=8)
        if i == 0:
            ax.set_title(f"tile {t}\n$\\rho$ = {share[t] * 100:.1f}%",
                         fontsize=9, pad=3)

for i, (pre, name) in enumerate(ROWS):
    axes[i, 0].text(-0.32, 0.5, name, transform=axes[i, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=9)

fig.savefig(os.path.join(FIG, "fig_q1_pred_examples.pdf"))
fig.savefig(os.path.join(FIG, "fig_q1_pred_examples.png"), dpi=200)
print("written")
