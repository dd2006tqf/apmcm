"""E17: residual intercept of the transferred model on Hengzhou.

No target labels exist, so a full Platt intercept is not identifiable. What is
identifiable is the residual offset left after the physics guards: the model
probability is asked to agree with two physical anchors.

  positive anchor: dVV <= -6 dB and REM <= 5 m and outside W0  (new water)
  negative anchor: REM >= 30 m and dVV >= +2 dB                 (upland, brightened)

Solve sum sigmoid(logit(p) + b) = sum y for b on a sample. Report b, the shift in
probability at the operating threshold tau_A, and the raw-mask area before and
after calibration. The guards are not re-run, so this prices the calibration
alone. Output: experiments/results/e17_q3_intercept.json
"""
import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
os.chdir(os.path.join(ROOT, "code"))

import numpy as np
import config as C

S = (7500, 8700)
B = os.path.join(ROOT, "_tmp", "cache", "q3")
N_SAMPLE = 300_000
SEED = 2026
TAU_A = 0.557


def rd_u1(n):
    return np.fromfile(os.path.join(B, n), dtype=np.uint8).reshape(S) > 0


def rd_f4(n):
    return np.fromfile(os.path.join(B, n), dtype=np.float32).reshape(S)


def rd_f2(n):
    return np.fromfile(os.path.join(B, n), dtype=np.float16).reshape(S).astype(np.float32)


def main():
    t0 = time.time()
    pbar = rd_f2("pbar_20260716_db.f2")
    vv16 = rd_f4("vvlee_20260716.f4")
    vv27 = rd_f4("vvlee_20260627.f4")
    rem = np.load(os.path.join(B, "rem30.npy")).astype(np.float32)
    if rem.shape != S:
        rem = np.repeat(np.repeat(rem, 3, axis=0), 3, axis=1)
    if rem.shape != S:
        raise ValueError(f"rem shape {rem.shape}")
    w0 = rd_u1("W0_union.u1")
    bnd = rd_u1("boundary.u1")

    print(f"[e17] pbar range {np.nanmin(pbar):.4f}..{np.nanmax(pbar):.4f}", flush=True)

    eps = 1e-6
    z = np.log(np.clip(pbar, eps, 1 - eps) / (1 - np.clip(pbar, eps, 1 - eps)))
    dvv = vv16 - vv27

    pos = (dvv <= -6) & (rem <= 5) & (~w0) & bnd & np.isfinite(z)
    neg = (rem >= 30) & (dvv >= 2) & bnd & np.isfinite(z)
    print(f"[e17] anchors pos={int(pos.sum())} neg={int(neg.sum())}", flush=True)

    rng = np.random.default_rng(SEED)
    zp = z[pos]
    zn = z[neg]
    print(f"[e17] anchors used pos={len(zp)} neg={len(zn)}", flush=True)

    def objective(b):
        return float(np.sum(1.0 / (1.0 + np.exp(-(zp + b))))
                     + np.sum(1.0 / (1.0 + np.exp(-(zn + b))) - 0.0) - len(zp))

    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if objective(mid) > 0:
            hi = mid
        else:
            lo = mid
    b = 0.5 * (lo + hi)

    p_pos = float(1.0 / (1.0 + np.exp(-float(zp.mean()))))
    p_neg = float(1.0 / (1.0 + np.exp(-float(zn.mean()))))
    m_raw = pbar >= TAU_A
    m_cal = (1.0 / (1.0 + np.exp(-(z + b))) >= TAU_A) & bnd
    area_raw = float((m_raw & bnd).sum()) * 100.0 / 1e6
    area_cal = float(m_cal.sum()) * 100.0 / 1e6
    shift_at_tau = float(1.0 / (1.0 + np.exp(-(np.log(TAU_A / (1 - TAU_A)) + b))) - TAU_A)

    out = dict(b=b, n_pos=int(len(zp)), n_neg=int(len(zn)),
               mean_p_pos=p_pos, mean_p_neg=p_neg,
               area_raw=area_raw, area_cal=area_cal,
               delta_pct=100.0 * (area_cal - area_raw) / area_raw,
               p_at_tau_after=TAU_A + shift_at_tau,
               seconds=time.time() - t0)
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(ROOT, "experiments", "results", "e17_q3_intercept.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
