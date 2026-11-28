"""E9 — Q2 mask-level combinations from the cached variant masks (no retraining).

The cache stores per-tile binary masks for M1..M6 under _tmp/cache/q1/mask_M*.u8 (1000 x 256 x 256).
We test intersections/unions/votes and MMU post-processing of M2 and M6, with the paper's pooled
IoU and ellipsoidal-area RAE.
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


def mask_mm(name):
    return np.memmap(CACHE / f"mask_{name}.u8", dtype=np.uint8, mode="r", shape=(1000, 256, 256))


def mmu(m, thr):
    if thr <= 0:
        return m
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel()); keep = sizes >= thr; keep[0] = False
    return keep[lab]


def add_large(m6, m2, thr):
    """Add M2-only components of at least thr pixels to M6."""
    lab, n = ndi.label(m2 & ~m6, structure=np.ones((3, 3)))
    if n == 0:
        return m6
    sizes = np.bincount(lab.ravel()); keep = sizes >= thr; keep[0] = False
    return m6 | keep[lab]


def main():
    q1 = json.load(open(C.OUT / "q1_results.json"))
    ids = [int(i) for i in q1["tile_ids"]]
    M = {n: mask_mm(n) for n in ("M1", "M2", "M3", "M4", "M5", "M6")}

    variants = {
        "M2": lambda i, m: m["M2"],
        "M6": lambda i, m: m["M6"],
        "M2_and_M6": lambda i, m: m["M2"] & m["M6"],
        "M2_or_M6": lambda i, m: m["M2"] | m["M6"],
        "M2_M3_M6_vote": lambda i, m: (m["M2"].astype(np.uint8) + m["M3"] + m["M6"]) >= 2,
        "M6_plus_M2_large50": lambda i, m: add_large(m["M6"], m["M2"], 50),
        "M6_plus_M2_large100": lambda i, m: add_large(m["M6"], m["M2"], 100),
        "M2_mmu50": lambda i, m: mmu(m["M2"], 50),
        "M6_mmu50": lambda i, m: mmu(m["M6"], 50),
        "M2orM6_mmu50": lambda i, m: mmu(m["M2"] | m["M6"], 50),
        "M2_and_M6_mmu50": lambda i, m: mmu(m["M2"] & m["M6"], 50),
    }
    acc = {k: dict(tp=0, fp=0, fn=0, hat=0.0) for k in variants}
    ref_tot = 0.0
    for idx, tid in enumerate(ids):
        with rasterio.open(C.OFFICIAL_LABEL_DIR / f"label_{tid}.tif") as d:
            T, H, W = d.transform, d.height, d.width
            ref = d.read(1) > 0
        px = np.repeat(U.pixel_area_rows(T, H)[:, None], W, axis=1)
        ref_tot += float(px[ref].sum() / 1e6)
        mm = {n: M[n][idx] > 0 for n in M}
        for k, f in variants.items():
            p = f(idx, mm)
            a = acc[k]
            a["tp"] += int((p & ref).sum()); a["fp"] += int((p & ~ref).sum()); a["fn"] += int((~p & ref).sum())
            a["hat"] += float(px[p].sum() / 1e6)
    out = []
    for k, a in acc.items():
        iou = a["tp"] / max(a["tp"] + a["fp"] + a["fn"], 1)
        rae = (a["hat"] - ref_tot) / ref_tot
        out.append(dict(name=k, iou=float(iou), rae=float(rae), hat_km2=float(a["hat"]),
                        tp=int(a["tp"]), fp=int(a["fp"]), fn=int(a["fn"])))
        print(f"[E9] {k:20s} IoU={iou:.4f} RAE={rae:+.3f} A_hat={a['hat']:.1f}")
    best = max(out, key=lambda x: x["iou"] - 2 * abs(x["rae"]))
    json.dump(dict(meta=dict(n_tiles=len(ids), ref_km2=float(ref_tot)), variants=out, best=best),
              open(os.path.join(RES, "e9_q2_mask_combos.json"), "w"), indent=1)
    print("best", best)


if __name__ == "__main__":
    main()
