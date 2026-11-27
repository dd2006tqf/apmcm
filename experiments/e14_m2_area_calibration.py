"""E14 — M2 + isotonic / area-preserving calibration, to test whether M2 can replace M6.

For each outer fold, load the M2 checkpoint, predict its calibration and test tiles, fit isotonic on
calibration probabilities, choose tau_A so the calibrated mask area matches the expected calibrated
area, then apply to test. Compare raw M2, M2+MMU50, M2+cal, M2+cal+MMU50 with M6.
"""
import os, sys, json, types
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

# problem1 imports lightgbm at module import; Q2 inference does not need it.
sys.modules.setdefault("lightgbm", types.ModuleType("lightgbm"))

import numpy as np
import torch
from scipy import ndimage as ndi
import config as C
import common as U
import problem1 as P1
import problem2 as P2

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
    col = np.array(q1["tile_col"]); row = np.array(q1["tile_row"]); share = np.array(q1["tile_flood_share"])
    fold, _, _ = P1.build_folds(col, row, share)
    sets = P1.fold_sets(fold, col, row)
    pos = {t: i for i, t in enumerate(ids)}

    M = {n: np.memmap(CACHE / f"mask_{n}.u8", dtype=np.uint8, mode="r", shape=(1000, 256, 256))
         for n in ("M2", "M6")}
    D = P2.Data()

    specs = ["M2_raw", "M2_mmu50", "M2_cal", "M2_cal_mmu50", "M6_raw", "M6_mmu50"]
    acc = {s: dict(tp=0, fp=0, fn=0, tn=0, n=0) for s in specs}
    ar = {s: [] for s in specs}; ah = {s: [] for s in specs}
    ar_h = {s: [] for s in specs}; ah_h = {s: [] for s in specs}
    taus = []

    for k, f in enumerate(sets):
        te = np.array(f["test"]); ca = np.array(f["cal"])
        model = P2.make_model("M2")
        with open(C.CACHE / "q2" / f"M2_fold{k}.pt", "rb") as fh:
            model.load_state_dict(torch.load(fh, map_location=P2.DEV))
        model.eval()
        p_c = P2.predict(model, D, ca, 6)
        p_t = P2.predict(model, D, te, 6)
        vc = D.V[ca] > 0
        wA = D.A[ca][:, :, None].expand_as(p_c) * vc
        iso = P2.fit_isotonic(p_c, D.Y[ca], D.V[ca], seed=C.SEED + k)
        expected = float((wA * P2.iso_apply(iso, p_c)).sum())
        tauA, mask_area = P2.area_preserving_tau(p_c, wA, expected)
        taus.append(dict(fold=k, tauA=float(tauA), expected=expected, mask_area=float(mask_area),
                         n_cal=len(ca), n_test=len(te)))
        p_t_iso = P2.iso_apply(iso, p_t)
        m_cal = (p_t_iso >= tauA) & (D.V[te] > 0)
        y = D.Y[te] > 0
        v = D.V[te] > 0

        for j, idx in enumerate(te.tolist()):
            tid = ids[idx]
            is_h = tid in hard
            cand = {
                "M2_raw": M["M2"][idx] > 0,
                "M2_mmu50": mmu(M["M2"][idx] > 0, 50),
                "M2_cal": m_cal[j].cpu().numpy(),
                "M2_cal_mmu50": mmu(m_cal[j].cpu().numpy(), 50),
                "M6_raw": M["M6"][idx] > 0,
                "M6_mmu50": mmu(M["M6"][idx] > 0, 50),
            }
            ref = y[j].cpu().numpy(); vv = v[j].cpu().numpy()
            for s, p in cand.items():
                p = p & vv
                tp = int((p & ref).sum()); fp = int((p & ~ref).sum()); fn = int((~p & ref).sum())
                a = acc[s]; a["tp"] += tp; a["fp"] += fp; a["fn"] += fn
                a["tn"] += int((~p & ~ref & vv).sum()); a["n"] += int(vv.sum())
                arow = D.A[idx].cpu().numpy()[:, None]
                ar[s].append(float((arow * (ref & vv)).sum()))
                ah[s].append(float((arow * p).sum()))
                if is_h:
                    ar_h[s].append(ar[s][-1]); ah_h[s].append(ah[s][-1])
        print(f"[E14] fold {k}: tauA={tauA:.4f}, expected={expected:.1f}, area={mask_area:.1f}, "
              f"IoU_cal={float((m_cal & y).sum()) / max(float(((m_cal | y) & v).sum()), 1):.4f}", flush=True)

    out = {}
    for s in specs:
        a = acc[s]
        oa = (a["tp"] + a["tn"]) / a["n"]
        prec = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0.0
        rec = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        iou = a["tp"] / (a["tp"] + a["fp"] + a["fn"]) if a["tp"] + a["fp"] + a["fn"] else 0.0
        pe = ((a["tp"] + a["fp"]) * (a["tp"] + a["fn"]) + (a["fn"] + a["tn"]) * (a["fp"] + a["tn"])) / (a["n"] ** 2)
        kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
        r = np.array(ar[s]); h = np.array(ah[s]); e = h - r
        sel = r >= 0.05
        out[s] = dict(OA=float(oa), Precision=float(prec), Recall=float(rec), F1=float(f1), IoU=float(iou),
                      kappa=float(kappa), RAE=float(e.sum() / r.sum()), MAE_A=float(np.abs(e).mean()),
                      median_abs_RAE_t=float(np.median(np.abs(e[sel] / r[sel]))),
                      A_ref_total=float(r.sum()), A_hat_total=float(h.sum()))
        rh = np.array(ar_h[s]); hh = np.array(ah_h[s]); eh = hh - rh
        out[s]["hard"] = dict(RAE=float(eh.sum() / rh.sum()), MAE_A=float(np.abs(eh).mean()),
                              A_ref_total=float(rh.sum()), A_hat_total=float(hh.sum()))
        print(f"[E14] {s:14s} IoU={iou:.4f} RAE={out[s]['RAE']:+.4f} MAE={out[s]['MAE_A']:.4f} "
              f"med={out[s]['median_abs_RAE_t']:.2f} F1={f1:.4f}")

    json.dump(dict(meta=dict(n_tiles=len(ids), ref_km2=float(sum(ar["M2_raw"]))), taus=taus, rows=out),
              open(os.path.join(RES, "e14_m2_area_calibration.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
