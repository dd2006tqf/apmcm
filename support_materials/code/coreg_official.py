"""Co-registration check of the official tiles (SAR vs pre-flood optical), the counterpart of the
Sentinel-1 translation estimate in problem3: per-tile FFT cross-correlation between the optical water
mask (NDWI > 0) and the SAR-dark mask (Lee-filtered z < 0.2) on tiles with 3-60 % optical water and
>= 95 % valid optics. Writes output/coreg_official.json."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import config as C
import common as U
import problem1 as P1


def main():
    feats = P1.mm("feats.f16", np.float16, (P1.NT, P1.NF, P1.H, P1.H))
    rows = []
    for i in range(P1.NT):
        m = feats[i, U.FI["m"]] > 0.5
        w = ((feats[i, U.FI["ndwi"]] > 0) & m).astype(float)
        if w.mean() < 0.03 or w.mean() > 0.6 or m.mean() < 0.95:
            continue
        d = (feats[i, U.FI["zt"]] < 0.2).astype(float)
        a, b = w - w.mean(), d - d.mean()
        c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
        r, cc = np.unravel_index(np.argmax(c), c.shape)
        r = int(r if r < 128 else r - 256); cc = int(cc if cc < 128 else cc - 256)
        rows.append(dict(i=i, dr=r, dc=cc, ncc=float(c.max() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))))
    s = np.array([[x["dr"], x["dc"], x["ncc"]] for x in rows])
    out = dict(n_tiles=len(rows), median_shift_px=[float(np.median(s[:, 0])), float(np.median(s[:, 1]))],
               share_within_2px=float(np.mean((np.abs(s[:, 0]) <= 2) & (np.abs(s[:, 1]) <= 2))),
               share_within_5px=float(np.mean((np.abs(s[:, 0]) <= 5) & (np.abs(s[:, 1]) <= 5))),
               median_ncc=float(np.median(s[:, 2])), tiles=rows)
    U.save_json(out, C.OUT / "coreg_official.json")
    print(f"[coreg] official tiles checked {len(rows)}: median shift {out['median_shift_px']} px, "
          f"within 2 px {out['share_within_2px']:.3f}")
    return out


if __name__ == "__main__":
    main()
