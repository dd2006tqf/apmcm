"""E12 — full table metrics for Q1/Q2 raw and 50 px MMU variants.

E10 only stored IoU/RAE; the paper tables also need OA/F1/kappa/MAE/median.
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
    hard = set(int(t) for t in q1["hard_tiles"])
    M = {n: np.memmap(CACHE / f"mask_{n}.u8", dtype=np.uint8, mode="r", shape=(1000, 256, 256))
         for n in ("M2", "M6")}

    specs = [("Q1_A", "A", 0), ("Q1_A_mmu50", "A", 50),
             ("Q1_B", "B", 0), ("Q1_B_mmu50", "B", 50),
             ("Q2_M2", "M2", 0), ("Q2_M2_mmu50", "M2", 50),
             ("Q2_M6", "M6", 0), ("Q2_M6_mmu50", "M6", 50)]
    acc = {n: dict(tp=0, fp=0, fn=0, tn=0, n=0) for n, _, _ in specs}
    acc_h = {n: dict(tp=0, fp=0, fn=0) for n, _, _ in specs}
    A_ref = {n: [] for n, _, _ in specs}
    A_hat = {n: [] for n, _, _ in specs}
    A_ref_h = {n: [] for n, _, _ in specs}
    A_hat_h = {n: [] for n, _, _ in specs}

    for idx, tid in enumerate(ids):
        with rasterio.open(C.OFFICIAL_LABEL_DIR / f"label_{tid}.tif") as d:
            T, H, W = d.transform, d.height, d.width
            ref = d.read(1) > 0
        px = np.repeat(U.pixel_area_rows(T, H)[:, None], W, axis=1)
        is_h = tid in hard
        for name, src, thr in specs:
            if src in ("A", "B"):
                with rasterio.open(C.OUT / "q1_pred" / f"{src}_{tid}.png") as d:
                    p = d.read(1) > 0
            else:
                p = M[src][idx] > 0
            p = mmu(p, thr)
            a = acc[name]
            a["tp"] += int((p & ref).sum()); a["fp"] += int((p & ~ref).sum())
            a["fn"] += int((~p & ref).sum()); a["tn"] += int((~p & ~ref).sum())
            a["n"] += int(ref.size)
            ar = float(px[ref].sum() / 1e6); ah = float(px[p].sum() / 1e6)
            A_ref[name].append(ar); A_hat[name].append(ah)
            if is_h:
                A_ref_h[name].append(ar); A_hat_h[name].append(ah)
                ahc = acc_h[name]
                ahc["tp"] += int((p & ref).sum()); ahc["fp"] += int((p & ~ref).sum())
                ahc["fn"] += int((~p & ref).sum())

    def area(xr, xh):
        xr = np.asarray(xr); xh = np.asarray(xh)
        e = xh - xr
        sel = xr >= 0.05
        rt = e[sel] / xr[sel] if sel.any() else np.array([0.0])
        return dict(RAE=float(e.sum() / xr.sum()), MAE_A=float(np.abs(e).mean()),
                    median_abs_RAE_t=float(np.median(np.abs(rt))))

    out = {}
    for name, _, _ in specs:
        a = acc[name]
        tp, fp, fn, tn, n = a["tp"], a["fp"], a["fn"], a["tn"], a["n"]
        oa = (tp + tn) / n
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
        rec = dict(OA=oa, Precision=prec, Recall=rec, F1=f1, IoU=iou, kappa=kappa,
                   TP=tp, FP=fp, FN=fn, TN=tn, N=n, **area(A_ref[name], A_hat[name]))
        ah = acc_h[name]
        hiou = ah["tp"] / max(ah["tp"] + ah["fp"] + ah["fn"], 1)
        rec["hard"] = dict(IoU=float(hiou), **area(A_ref_h[name], A_hat_h[name]))
        out[name] = rec
        print(f"[E12] {name:14s} IoU={iou:.4f} F1={f1:.4f} RAE={rec['RAE']:+.4f} "
              f"MAE={rec['MAE_A']:.4f} med={rec['median_abs_RAE_t']:.2f} "
              f"hard IoU={hiou:.4f}")

    json.dump(out, open(os.path.join(RES, "e12_mmu_table_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
