"""Qualitative comparison on four official tiles: post-flood SAR, pre-flood optical, reference,
and confusion maps of baseline A, LightGBM (M0) and the final fused model M6."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
import rasterio
from PIL import Image
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(ROOT, 'code'))
import config as CF

q1 = load('problem_1_results.json')
folds = load('folds.json')
ids = folds['ids']
assert ids == sorted(ids)
pos = {t: i for i, t in enumerate(ids)}
tiles = q1['example_tiles']
NT, H = len(ids), 256
m6 = np.memmap(os.path.join(CACHE_Q1, 'mask_M6.u8'), dtype=np.uint8, mode='r', shape=(NT, H, H))
valid_all = np.memmap(os.path.join(CACHE_Q1, 'valid.u8'), dtype=np.uint8, mode='r', shape=(NT, H, H))
y_all = np.memmap(os.path.join(CACHE_Q1, 'y.u8'), dtype=np.uint8, mode='r', shape=(NT, H, H))


def stretch(a, valid):
    v = a[valid] if valid.any() else a.ravel()
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def read(tid):
    with rasterio.open(os.path.join(CF.OFFICIAL_SAR_DIR, f'after_gf3_hh_{tid}.TIF')) as d:
        sar = d.read(1).astype(float)
    with rasterio.open(os.path.join(CF.OFFICIAL_OPT_DIR, f'before_s2_{tid}.TIF')) as d:
        opt = d.read().astype(float)
    return sar, opt


def cm_map(pred, y, valid):
    """0 TN, 1 TP, 2 FP, 3 FN, 4 invalid"""
    out = np.zeros(y.shape, np.uint8)
    out[(pred == 1) & (y == 1)] = 1
    out[(pred == 1) & (y == 0)] = 2
    out[(pred == 0) & (y == 1)] = 3
    out[~valid] = 4
    return out


def scores(pred, y, valid):
    p, t = pred[valid] == 1, y[valid] == 1
    inter, union = np.sum(p & t), np.sum(p | t)
    iou = inter / union if union else np.nan
    rae = (p.sum() - t.sum()) / t.sum() if t.sum() else np.nan
    return iou, rae


cmap_cm = ListedColormap([C['neutral_light'], C['blue_main'], C['red_strong'], C['gold'], '#FFFFFF'])
cols = [['Post-flood', 'SAR'], ['Pre-flood', 'optical'], ['Reference', '$F$'], ['A', 'split-Otsu'], ['M0', 'LightGBM'], ['M6', 'fused (ours)']]

fig = plt.figure(figsize=(6.4, 6.2), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(len(tiles) + 1, 6, height_ratios=[1] * len(tiles) + [0.12])
for r, tid in enumerate(tiles):
    i = pos[tid]
    sar, opt = read(tid)
    valid = valid_all[i].astype(bool)
    y = y_all[i].astype(np.uint8)
    pa = (np.array(Image.open(os.path.join(OUT_DIR, 'q1_pred', f'A_{tid}.png'))) > 0).astype(np.uint8)
    pb = (np.array(Image.open(os.path.join(OUT_DIR, 'q1_pred', f'B_{tid}.png'))) > 0).astype(np.uint8)
    p6 = m6[i].astype(np.uint8)
    rgb = np.dstack([stretch(opt[k], opt[k] > 0) for k in (2, 1, 0)])
    for c in range(6):
        ax = fig.add_subplot(gs[r, c])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_frame_on(False)
        if c == 0:
            ax.imshow(stretch(sar, sar > 0), cmap='gray', interpolation='nearest', rasterized=True)
            ax.set_ylabel(f'Tile {tid}\n$\\rho$={y[valid].mean():.2f}')
        elif c == 1:
            ax.imshow(rgb, interpolation='nearest', rasterized=True)
        elif c == 2:
            ref = np.where(valid, y, 4).astype(np.uint8)
            ref = np.where(ref == 0, 0, np.where(ref == 1, 1, 4))
            ax.imshow(ref, cmap=cmap_cm, vmin=-0.5, vmax=4.5, interpolation='nearest', rasterized=True)
        else:
            pred = [pa, pb, p6][c - 3]
            ax.imshow(cm_map(pred, y, valid), cmap=cmap_cm, vmin=-0.5, vmax=4.5,
                      interpolation='nearest', rasterized=True)
            iou, rae = scores(pred, y, valid)
            ax.set_xlabel(f'IoU {iou:.2f}\nRAE {rae:+.2f}', labelpad=2,
                          fontweight='bold' if c == 5 else 'normal')
        if r == 0:
            ax.set_title(chr(10).join(cols[c]), fontsize=plt.rcParams['axes.labelsize'],
                         fontweight='bold' if c == 5 else 'normal', pad=3)
lax = fig.add_subplot(gs[-1, :])
lax.set_axis_off()
handles = [Patch(facecolor=C['blue_main'], label='TP (new inundation hit)'),
           Patch(facecolor=C['red_strong'], label='FP (false alarm)'),
           Patch(facecolor=C['gold'], label='FN (missed)'),
           Patch(facecolor=C['neutral_light'], label='TN')]
lax.legend(handles=handles, loc='center', ncol=4, frameon=False)
finish(fig, 'fig_q1_visual_cmp')
