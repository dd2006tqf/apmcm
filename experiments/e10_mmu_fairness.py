"""E10 — fairness check: MMU 50 px applied to every Q1/Q2 model, plus hard-set metrics.

Q1 masks: A_/B_*.png. Q2 masks: cached M1..M6. M0 (Q2 physical baseline) is the Q1 LightGBM mask.
"""
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
os.makedirs(RES, exist_ok=True)
CACHE = C.CACHE / "q1"


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
    pos = {t: i for i, t in enumerate(ids)}
    hard = [pos[int(t)] for t in q1["hard_tiles"]]
    M = {n: np.memmap(CACHE / f"mask_{n}.u8", dtype=np.uint8, mode="r", shape=(1000, 256, 256)) for n in ("M1", "M2", "M3", "M4", "M5", "M6")}

    specs = [("Q1_A", "A", 0), ("Q1_A_mmu50", "A", 50), ("Q1_B", "B", 0), ("Q1_B_mmu50", "B", 50),
             ("Q2_M0", "B", 0), ("Q2_M0_mmu50", "B", 50),
             ("Q2_M2", "M2", 0), ("Q2_M2_mmu50", "M2", 50),
             ("Q2_M6", "M6", 0), ("Q2_M6_mmu50", "M6", 50)]
    acc = {n: dict(tp=0, fp=0, fn=0, hat=0.0, tp_h=0, fp_h=0, fn_h=0, hat_h=0.0) for n, _, _ in specs}
    ref_tot = 0.0; ref_h = 0.0
    for idx, tid in enumerate(ids):
        with rasterio.open(C.OFFICIAL_LABEL_DIR / f"label_{tid}.tif") as d:
            T, H, W = d.transform, d.height, d.width
            ref = d.read(1) > 0
        px = np.repeat(U.pixel_area_rows(T, H)[:, None], W, axis=1)
        ref_tot += float(px[ref].sum() / 1e6)
        is_h = idx in set(hard)
        if is_h:
            ref_h += float(px[ref].sum() / 1e6)
        for name, src, thr in specs:
            if src in ("A", "B"):
                with rasterio.open(C.OUT / "q1_pred" / f"{src}_{tid}.png") as d:
                    p = d.read(1) > 0
            else:
                p = M[src][idx] > 0
            p = mmu(p, thr)
            a = acc[name]
            a["tp"] += int((p & ref).sum()); a["fp"] += int((p & ~ref).sum()); a["fn"] += int((~p & ref).sum())
            a["hat"] += float(px[p].sum() / 1e6)
            if is_h:
                a["tp_h"] += int((p & ref).sum()); a["fp_h"] += int((p & ~ref).sum()); a["fn_h"] += int((~p & ref).sum())
                a["hat_h"] += float(px[p].sum() / 1e6)
    out = []
    for name, _, _ in specs:
        a = acc[name]
        iou = a["tp"] / max(a["tp"] + a["fp"] + a["fn"], 1)
        rae = (a["hat"] - ref_tot) / ref_tot
        iou_h = a["tp_h"] / max(a["tp_h"] + a["fp_h"] + a["fn_h"], 1)
        rae_h = (a["hat_h"] - ref_h) / ref_h if ref_h else None
        out.append(dict(name=name, iou=float(iou), rae=float(rae), iou_hard=float(iou_h), rae_hard=float(rae_h),
                        hat_km2=float(a["hat"])))
        print(f"[E10] {name:14s} IoU={iou:.4f} RAE={rae:+.3f} | hard IoU={iou_h:.4f} RAE={rae_h:+.4f}")
    json.dump(dict(meta=dict(n_tiles=len(ids), n_hard=len(hard), ref_km2=float(ref_tot), ref_hard_km2=float(ref_h)),
                   rows=out), open(os.path.join(RES, "e10_mmu_fairness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
