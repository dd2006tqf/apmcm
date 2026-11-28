"""E13 — Q1 MMU sweep for baseline A and LightGBM B (no closing), for the paper figure."""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import numpy as np
import rasterio
from scipy import ndimage as ndi
import config as C
import common as U

RES = os.path.join(_HERE, "results")


def mmu(m, thr):
    if thr <= 0:
        return m
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel()); keep = sizes >= thr; keep[0] = False
    return keep[lab]


def main():
    q1 = json.load(open(C.OUT / "q1_results.json"))
    ids = [int(i) for i in q1["tile_ids"]]
    thresholds = [0, 10, 25, 50, 100]
    specs = [("A", t) for t in thresholds] + [("B", t) for t in thresholds]
    acc = {s: dict(tp=0, fp=0, fn=0) for s in specs}
    ref_tot = 0.0
    hat = {s: 0.0 for s in specs}
    for tid in ids:
        with rasterio.open(C.OFFICIAL_LABEL_DIR / f"label_{tid}.tif") as d:
            T, H, W = d.transform, d.height, d.width
            ref = d.read(1) > 0
        px = np.repeat(U.pixel_area_rows(T, H)[:, None], W, axis=1)
        ref_tot += float(px[ref].sum() / 1e6)
        for src, thr in specs:
            with rasterio.open(C.OUT / "q1_pred" / f"{src}_{tid}.png") as d:
                p = d.read(1) > 0
            p = mmu(p, thr)
            a = acc[(src, thr)]
            a["tp"] += int((p & ref).sum()); a["fp"] += int((p & ~ref).sum()); a["fn"] += int((~p & ref).sum())
            hat[(src, thr)] += float(px[p].sum() / 1e6)
    rows = []
    for src, thr in specs:
        a = acc[(src, thr)]
        iou = a["tp"] / max(a["tp"] + a["fp"] + a["fn"], 1)
        rows.append(dict(src=src, mmu=thr, iou=float(iou), rae=float((hat[(src, thr)] - ref_tot) / ref_tot),
                         hat_km2=float(hat[(src, thr)]), tp=a["tp"], fp=a["fp"], fn=a["fn"]))
        print(f"[E13] {src} mmu={thr:3d} IoU={iou:.4f} RAE={(hat[(src, thr)] - ref_tot) / ref_tot:+.4f}")
    json.dump(dict(meta=dict(n_tiles=len(ids), ref_km2=float(ref_tot)), rows=rows),
              open(os.path.join(RES, "e13_q1_mmu_grid.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
