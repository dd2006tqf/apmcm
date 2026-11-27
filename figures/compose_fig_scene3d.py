# -*- coding: utf-8 -*-
"""Compose the two terrain panels into Figure 1: no legend, no axes."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42,
})


def bbox(path, pad=8):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    bg = np.median(a.reshape(-1, 3), axis=0)
    mask = (np.abs(a - bg).max(axis=2) > 12)
    ys, xs = np.where(mask)
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
            min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))


def crop(path, box):
    return Image.open(path).convert("RGB").crop(box)


def fit(img, cw, ch):
    canvas = Image.new("RGB", (cw, ch), "white")
    k = min(cw / img.width, ch / img.height)
    new = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
    canvas.paste(new, ((cw - new.width) // 2, (ch - new.height) // 2))
    arr = np.asarray(canvas).astype(int)
    sat = arr.max(axis=2) - arr.min(axis=2)
    near_white = (arr.min(axis=2) >= 240) & (sat <= 14)
    arr[near_white] = 255
    arr[arr.min(axis=2) >= 250] = 255
    return Image.fromarray(arr.astype(np.uint8))


CW, CH = 1600, 1080
A = os.path.join(FIG, "scene_a_new.png")
B = os.path.join(FIG, "scene_b_new.png")
box = bbox(A)
pa = fit(crop(A, box), CW, CH)
pb = fit(crop(B, box), CW, CH)

fig = plt.figure(figsize=(6.3, 2.5))
for img, x0, label, title in ((pa, 0.012, "(a)", "Before the flood"),
                              (pb, 0.512, "(b)", "During the flood")):
    ax = fig.add_axes([x0, 0.015, 0.476, 0.90])
    ax.imshow(np.asarray(img))
    ax.axis("off")
    fig.text(x0 + 0.004, 0.945, label, fontsize=10.5, weight="bold", va="top")
    fig.text(x0 + 0.047, 0.945, title, fontsize=9.6, color="#111111", va="top")

# ---- leader-line annotations, in canvas pixels (canvas = 1890 x 750 px) ----
W_PX, H_PX = 1890.0, 750.0
ov = fig.add_axes([0, 0, 1, 1])
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")


TIPS = []


def lab(txt, text_xy, tip_xy, ha="left", va="center"):
    TIPS.append(tip_xy)
    ov.annotate(txt,
                xy=(tip_xy[0] / W_PX, 1 - tip_xy[1] / H_PX),
                xytext=(text_xy[0] / W_PX, 1 - text_xy[1] / H_PX),
                ha=ha, va=va, fontsize=9.6, color="#111111",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                arrowprops=dict(arrowstyle="-", lw=0.85, color="#2b2b2b",
                                shrinkA=2, shrinkB=3))


lab("channel $W_0$", (80, 185), (333, 378))
lab("terrace settlement", (560, 145), (534, 367))
lab("depot", (700, 722), (596, 423))
lab("valley road", (415, 722), (548, 420))

lab("island village\nboat only", (1395, 132), (1477, 370))
lab("road cut by $F$", (1370, 722), (1497, 416))
lab("new inundation $F$", (975, 722), (1175, 362))

if os.environ.get("DBG") == "1":
    for tx, ty in TIPS:
        ov.plot(tx / W_PX, 1 - ty / H_PX, marker="o", ms=3.2, color="#d40000", zorder=10)

out = os.path.join(FIG, "fig_scene3d.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=300)
fig_fn = out.replace(".pdf", ".png")
dbg = os.path.join(FIG, "fig_scene3d_dbg.png")
Image.open(fig_fn).save(dbg)
print("saved", out)















