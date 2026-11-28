"""E5 — block-bootstrap CI for Q2 area error (M0 physics baseline vs M6 gated fusion) and the paired gain.

Uses the stored per-tile predicted areas (q2_results.per_tile) and ellipsoidal reference areas.
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import numpy as np
import rasterio
from pyproj import Transformer
import config as C

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)
BOOT = 2000


def main():
    q2 = json.load(open(C.OUT / "q2_results.json"))
    ids = [int(i) for i in json.load(open(C.OUT / "q1_results.json"))["tile_ids"]]
    A_ref = np.array(q2["A_ref_t"], float)
    A0 = np.array(q2["per_tile"]["M0"]["A_hat"], float)
    A6 = np.array(q2["per_tile"]["M6"]["A_hat"], float)
    cx, cy = [], []
    for tid in ids:
        with rasterio.open(C.OFFICIAL_LABEL_DIR / f"label_{tid}.tif") as d:
            b = d.bounds
            tr = Transformer.from_crs(d.crs, "EPSG:32649", always_xy=True)
            xs, ys = zip(*[tr.transform(x, y) for x, y in [(b.left, b.bottom), (b.right, b.bottom), (b.right, b.top), (b.left, b.top)]])
            cx.append((min(xs) + max(xs)) / 2); cy.append((min(ys) + max(ys)) / 2)
    cx, cy = np.array(cx), np.array(cy)

    def rae(a, ref):
        return float((a - ref).sum() / ref.sum())

    blocks = (cx // 10000).astype(int).astype(str) + "_" + (cy // 10000).astype(int).astype(str)
    ub = sorted(set(blocks))
    groups = [np.where(blocks == b)[0] for b in ub]

    point = dict(M0=rae(A0, A_ref), M6=rae(A6, A_ref), delta=rae(A6, A_ref) - rae(A0, A_ref))
    rng = np.random.default_rng(C.SEED)
    b0, b6, bd = [], [], []
    for _ in range(BOOT):
        pick = rng.integers(0, len(ub), len(ub))
        idx = np.concatenate([groups[i] for i in pick])
        r0, r6 = rae(A0[idx], A_ref[idx]), rae(A6[idx], A_ref[idx])
        b0.append(r0); b6.append(r6); bd.append(r6 - r0)
    out = dict(meta=dict(n_tiles=len(A_ref), n_blocks=len(ub), boot=BOOT, seed=C.SEED),
               point=point,
               ci=dict(M0=[float(np.percentile(b0, 2.5)), float(np.percentile(b0, 97.5))],
                       M6=[float(np.percentile(b6, 2.5)), float(np.percentile(b6, 97.5))],
                       delta=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                       p_delta_le_0=float(np.mean(np.array(bd) <= 0))))
    # tile-level bootstrap for comparison (ignores spatial correlation)
    t0, t6, td = [], [], []
    for _ in range(BOOT):
        idx = rng.integers(0, len(A_ref), len(A_ref))
        r0, r6 = rae(A0[idx], A_ref[idx]), rae(A6[idx], A_ref[idx])
        t0.append(r0); t6.append(r6); td.append(r6 - r0)
    out["ci_tile"] = dict(M0=[float(np.percentile(t0, 2.5)), float(np.percentile(t0, 97.5))],
                          M6=[float(np.percentile(t6, 2.5)), float(np.percentile(t6, 97.5))],
                          delta=[float(np.percentile(td, 2.5)), float(np.percentile(td, 97.5))],
                          p_delta_le_0=float(np.mean(np.array(td) <= 0)))
    json.dump(out, open(os.path.join(RES, "e5_q2_rae_ci.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

