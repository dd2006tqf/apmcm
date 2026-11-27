"""E1 — Q1 uncertainty and patch-size error structure (v2, ellipsoidal pixel areas).

1) block bootstrap CI for pooled IoU and RAE of Q1 model A (threshold baseline) and B (LightGBM),
   resampling 10 km spatial blocks;
2) patch-size stratified detection rate at the physical thresholds 0.005 km2 (50 px) and 0.05 km2;
3) size distribution of false-positive blobs.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from pyproj import Transformer
import config as C
import common as U

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)
BOOT = 2000


def tile_info(path):
    with rasterio.open(path) as d:
        T = d.transform
        H = d.height
        px = np.repeat(U.pixel_area_rows(T, H)[:, None], d.width, axis=1)  # m2 per pixel
        b = d.bounds
        tr = Transformer.from_crs(d.crs, "EPSG:32649", always_xy=True)
        xs, ys = [], []
        for x, y in [(b.left, b.bottom), (b.right, b.bottom), (b.right, b.top), (b.left, b.top)]:
            xx, yy = tr.transform(x, y)
            xs.append(xx); ys.append(yy)
        return px, (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def load_mask(path):
    with rasterio.open(path) as d:
        return d.read(1) > 0


def patch_stats(ref, pred, px_area):
    """ref/pred boolean, px_area 2D m2. Returns per-bucket and FP blob stats."""
    flat_area = px_area.ravel()
    lab, n = ndi.label(ref, structure=np.ones((3, 3)))
    out = []
    if n:
        labf = lab.ravel()
        sizes = np.bincount(labf)[1:]
        area = np.bincount(labf, weights=flat_area)[1:]
        det = np.bincount(labf, weights=flat_area * pred.ravel().astype(float))[1:]
        km2 = area / 1e6
        for lo, hi in [(0, 0.005), (0.005, 0.05), (0.05, np.inf)]:
            m = (km2 >= lo) & (km2 < hi)
            out.append(dict(lo=lo, hi=hi, n_patches=int(m.sum()), area_km2=float(km2[m].sum()),
                            det_area_km2=float(det[m].sum() / 1e6),
                            det_rate_area=float(det[m].sum() / max(area[m].sum(), 1e-9))))
    flab, fn = ndi.label(pred & ~ref, structure=np.ones((3, 3)))
    if fn:
        fa = np.bincount(flab.ravel(), weights=flat_area)[1:] / 1e6
        fp = dict(n=int(fn), area_km2=float(fa.sum()),
                  frac_area_lt_0p005=float(fa[fa < 0.005].sum() / max(fa.sum(), 1e-12)),
                  frac_count_lt_0p005=float((fa < 0.005).mean()))
    else:
        fp = dict(n=0, area_km2=0.0, frac_area_lt_0p005=0.0, frac_count_lt_0p005=0.0)
    return out, fp


def main():
    q1 = json.load(open(C.OUT / "q1_results.json"))
    ids = [int(i) for i in q1["tile_ids"]]
    rows = []
    for tid in ids:
        lab_p = C.OFFICIAL_LABEL_DIR / f"label_{tid}.tif"
        px, cx, cy = tile_info(lab_p)
        ref = load_mask(lab_p)
        mA = load_mask(C.OUT / "q1_pred" / f"A_{tid}.png")
        mB = load_mask(C.OUT / "q1_pred" / f"B_{tid}.png")
        r = dict(tid=tid, cx=cx, cy=cy, tile_area_km2=float(px.sum() / 1e6),
                 ref_km2=float(px[ref].sum() / 1e6), ref_px=int(ref.sum()))
        for name, m in (("A", mA), ("B", mB)):
            tp = int((m & ref).sum()); fp = int((m & ~ref).sum()); fnc = int((~m & ref).sum())
            r[f"tp_{name}"] = tp; r[f"fp_{name}"] = fp; r[f"fn_{name}"] = fnc
            r[f"hat_km2_{name}"] = float(px[m].sum() / 1e6)
        rows.append(r)
    df = pd.DataFrame(rows)
    print("tiles", len(df), "ref km2", round(df.ref_km2.sum(), 2), "tile km2", round(df.tile_area_km2.sum(), 1))

    x0, y0 = df.cx.min(), df.cy.min()
    df["block"] = ((df.cx - x0) // 10000).astype(int).astype(str) + "_" + ((df.cy - y0) // 10000).astype(int).astype(str)
    blocks = sorted(df.block.unique())
    print("blocks", len(blocks))
    groups = [df.index[df.block == b].values for b in blocks]

    def metrics(sub):
        o = {}
        for name in ("A", "B"):
            tp, fp, fn = sub[f"tp_{name}"].sum(), sub[f"fp_{name}"].sum(), sub[f"fn_{name}"].sum()
            o[name] = dict(iou=float(tp / max(tp + fp + fn, 1)),
                           rae=float((sub[f"hat_km2_{name}"].sum() - sub.ref_km2.sum()) / max(sub.ref_km2.sum(), 1e-9)))
        return o

    point = metrics(df)
    rng = np.random.default_rng(C.SEED)
    boot = {k: {"iou": [], "rae": []} for k in ("A", "B")}
    for _ in range(BOOT):
        pick = rng.integers(0, len(blocks), len(blocks))
        sub = df.iloc[np.concatenate([groups[i] for i in pick])]
        m = metrics(sub)
        for k in ("A", "B"):
            boot[k]["iou"].append(m[k]["iou"]); boot[k]["rae"].append(m[k]["rae"])

    summary = {}
    for k in ("A", "B"):
        summary[k] = dict(iou=point[k]["iou"], rae=point[k]["rae"],
                          iou_ci=[float(np.percentile(boot[k]["iou"], 2.5)), float(np.percentile(boot[k]["iou"], 97.5))],
                          rae_ci=[float(np.percentile(boot[k]["rae"], 2.5)), float(np.percentile(boot[k]["rae"], 97.5))])

    strata, fps = {}, {}
    for name in ("A", "B"):
        agg = {}
        fpagg = dict(n=0, area=0.0, small_area=0.0, small_n=0)
        for _, row in df.iterrows():
            lab_p = C.OFFICIAL_LABEL_DIR / f"label_{int(row.tid)}.tif"
            px, _, _ = tile_info(lab_p)
            ref = load_mask(lab_p)
            pred = load_mask(C.OUT / "q1_pred" / f"{name}_{int(row.tid)}.png")
            st, fp = patch_stats(ref, pred, px)
            for s in st:
                key = f"{s['lo']:.3f}-{s['hi']}"
                a = agg.setdefault(key, dict(lo=s["lo"], hi=s["hi"], n_patches=0, area=0.0, det=0.0))
                a["n_patches"] += s["n_patches"]; a["area"] += s["area_km2"]; a["det"] += s["det_area_km2"]
            fpagg["n"] += fp["n"]; fpagg["area"] += fp["area_km2"]
            fpagg["small_area"] += fp["frac_area_lt_0p005"] * fp["area_km2"]
            fpagg["small_n"] += int(round(fp["frac_count_lt_0p005"] * fp["n"]))
        for a in agg.values():
            a["det_rate_area"] = a["det"] / max(a["area"], 1e-12)
            a["mean_patch_km2"] = a["area"] / max(a["n_patches"], 1)
        strata[name] = agg
        fps[name] = dict(n=fpagg["n"], area_km2=fpagg["area"],
                         small_blob_area_share=fpagg["small_area"] / max(fpagg["area"], 1e-12),
                         small_blob_count_share=fpagg["small_n"] / max(fpagg["n"], 1))

    out = dict(meta=dict(n_tiles=len(df), n_blocks=len(blocks), boot=BOOT, block_km=10.0,
                         thresholds_km2=[0.005, 0.05], seed=C.SEED), pooled=summary,
               patch_strata=strata, false_positive=fps)
    json.dump(out, open(os.path.join(RES, "e1_q1_uncertainty.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for name in strata:
        print(name, json.dumps(strata[name], indent=1))
        print(name, "FP", fps[name])


if __name__ == "__main__":
    main()
