"""E8 — Q1 post-processing grid on the existing LightGBM masks (no retraining).

Variants: morphological closing (disk r = 0/1/2 px) followed by a minimum-mapping-unit filter
(remove 8-connected components smaller than 0/5/10/25/50/100 px). Pooled IoU and RAE are computed
with WGS-84 ellipsoidal pixel areas, exactly as the paper's Q1 metrics.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
import config as C
import common as U

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)


def disk(r):
    if r <= 0:
        return None
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def mmu(mask, thr):
    if thr <= 0:
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= thr
    keep[0] = False
    return keep[lab]


def main():
    q1 = json.load(open(C.OUT / "q1_results.json"))
    ids = [int(i) for i in q1["tile_ids"]]
    rows = []
    for tid in ids:
        with rasterio.open(C.OFFICIAL_LABEL_DIR / f"label_{tid}.tif") as d:
            T, H, W = d.transform, d.height, d.width
            ref = d.read(1) > 0
        px = np.repeat(U.pixel_area_rows(T, H)[:, None], W, axis=1)
        with rasterio.open(C.OUT / "q1_pred" / f"B_{tid}.png") as d:
            pred = d.read(1) > 0
        rows.append(dict(tid=tid, px_sum=px.sum(), ref_km2=float(px[ref].sum() / 1e6),
                         ref_px=int(ref.sum())))
    df = pd.DataFrame(rows)
    print("tiles", len(df), "ref", round(df.ref_km2.sum(), 1))

    grid = []
    for r in (0, 1, 2):
        for thr in (0, 5, 10, 25, 50, 100):
            grid.append((r, thr))
    out = []
    for r, thr in grid:
        st = disk(r)
        tp = fp = fn = 0
        hat = 0.0
        for _, row in df.iterrows():
            with rasterio.open(C.OFFICIAL_LABEL_DIR / f"label_{int(row.tid)}.tif") as d:
                T, H, W = d.transform, d.height, d.width
                ref = d.read(1) > 0
            with rasterio.open(C.OUT / "q1_pred" / f"B_{int(row.tid)}.png") as d:
                pred = d.read(1) > 0
            if st is not None:
                pred = ndi.binary_closing(pred, structure=st)
            pred = mmu(pred, thr)
            px = np.repeat(U.pixel_area_rows(T, H)[:, None], W, axis=1)
            tp += int((pred & ref).sum()); fp += int((pred & ~ref).sum()); fn += int((~pred & ref).sum())
            hat += float(px[pred].sum() / 1e6)
        iou = tp / max(tp + fp + fn, 1)
        rae = (hat - df.ref_km2.sum()) / df.ref_km2.sum()
        out.append(dict(closing=r, mmu_px=thr, iou=float(iou), rae=float(rae),
                        hat_km2=float(hat), tp=int(tp), fp=int(fp), fn=int(fn)))
        print(f"[E8] close={r} mmu={thr:3d}: IoU={iou:.4f} RAE={rae:+.3f}")
    best_iou = max(out, key=lambda x: x["iou"])
    best_abs_rae = min(out, key=lambda x: abs(x["rae"]))
    json.dump(dict(meta=dict(n_tiles=len(df), ref_km2=float(df.ref_km2.sum())), grid=out,
                   best_iou=best_iou, best_abs_rae=best_abs_rae), open(os.path.join(RES, "e8_q1_postprocess.json"), "w"), indent=1)
    print("best IoU", best_iou)
    print("best |RAE|", best_abs_rae)


if __name__ == "__main__":
    main()
