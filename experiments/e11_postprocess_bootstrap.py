"""E11 — paired block bootstrap for the two post-processing gains.

Q1: LightGBM mask with 50 px MMU vs the raw mask.
Q2: M2 + 50 px MMU vs the paper's M6.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import numpy as np
import rasterio
from scipy import ndimage as ndi
from pyproj import Transformer
import config as C
import common as U

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)
CACHE = C.CACHE / "q1"
BOOT = 2000


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
    M2 = np.memmap(CACHE / "mask_M2.u8", dtype=np.uint8, mode="r", shape=(1000, 256, 256))
    M6 = np.memmap(CACHE / "mask_M6.u8", dtype=np.uint8, mode="r", shape=(1000, 256, 256))
    rows = []
    for i, tid in enumerate(ids):
        with rasterio.open(C.OFFICIAL_LABEL_DIR / f"label_{tid}.tif") as d:
            T, H, W = d.transform, d.height, d.width
            ref = d.read(1) > 0
            b = d.bounds
            tr = Transformer.from_crs(d.crs, "EPSG:32649", always_xy=True)
            xs, ys = zip(*[tr.transform(x, y) for x, y in [(b.left, b.bottom), (b.right, b.bottom), (b.right, b.top), (b.left, b.top)]])
        px = np.repeat(U.pixel_area_rows(T, H)[:, None], W, axis=1)
        with rasterio.open(C.OUT / "q1_pred" / f"B_{tid}.png") as d:
            B = d.read(1) > 0
        variants = {"Q1_B": B, "Q1_B_mmu": mmu(B, 50), "Q2_M2": M2[i] > 0, "Q2_M2_mmu": mmu(M2[i] > 0, 50), "Q2_M6": M6[i] > 0}
        r = dict(tid=tid, cx=float((min(xs) + max(xs)) / 2), cy=float((min(ys) + max(ys)) / 2),
                 ref_km2=float(px[ref].sum() / 1e6))
        for k, p in variants.items():
            r[f"tp_{k}"] = int((p & ref).sum()); r[f"fp_{k}"] = int((p & ~ref).sum()); r[f"fn_{k}"] = int((~p & ref).sum())
            r[f"hat_{k}"] = float(px[p].sum() / 1e6)
        rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows)
    df["block"] = ((df.cx - df.cx.min()) // 10000).astype(int).astype(str) + "_" + ((df.cy - df.cy.min()) // 10000).astype(int).astype(str)
    blocks = sorted(df.block.unique()); groups = [df.index[df.block == b].values for b in blocks]
    ref_sum = df.ref_km2.sum()

    def met(sub, k):
        tp, fp, fn = sub[f"tp_{k}"].sum(), sub[f"fp_{k}"].sum(), sub[f"fn_{k}"].sum()
        return tp / max(tp + fp + fn, 1), (sub[f"hat_{k}"].sum() - sub.ref_km2.sum()) / sub.ref_km2.sum()

    rng = np.random.default_rng(C.SEED)
    pairs = [("Q1_B", "Q1_B_mmu"), ("Q2_M2_mmu", "Q2_M6")]
    out = {}
    for a, b in pairs:
        ia, ra = met(df, a); ib, rb = met(df, b)
        di, dr = [], []
        for _ in range(BOOT):
            pick = rng.integers(0, len(blocks), len(blocks))
            sub = df.iloc[np.concatenate([groups[i] for i in pick])]
            x = met(sub, a); y = met(sub, b)
            di.append(x[0] - y[0]); dr.append(x[1] - y[1])
        out[f"{a}_vs_{b}"] = dict(iou_a=float(ia), iou_b=float(ib), rae_a=float(ra), rae_b=float(rb),
                                  diou=float(ia - ib), diou_ci=[float(np.percentile(di, 2.5)), float(np.percentile(di, 97.5))],
                                  drae=float(ra - rb), drae_ci=[float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
                                  p_diou_le_0=float(np.mean(np.array(di) <= 0)), p_drae_ge_0=float(np.mean(np.array(dr) >= 0)))
        print(a, "vs", b, json.dumps(out[f"{a}_vs_{b}"], indent=1))
    json.dump(dict(meta=dict(n_tiles=len(df), n_blocks=len(blocks), boot=BOOT, ref_km2=float(ref_sum)), tests=out),
              open(os.path.join(RES, "e11_postprocess_bootstrap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
