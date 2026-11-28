"""Shared preprocessing (MODELING_REPORT.md §3): I/O of official tiles, pixel area (3.1),
SAR rank normalization (3.2), Lee filter with ENL estimate (3.3), optical indices (3.4),
split-based Otsu with the Ashman test (§4.2), MMU filter and the pixel feature stack.
The same functions are reused unchanged for Hengzhou (Q3).
"""
import os, sys, json, hashlib, platform
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
import rasterio
from rasterio.transform import Affine
import config as C

FEATURES = ["z", "zt", "zt_mean3", "zt_std3", "zt_mean7", "zt_std7", "zt_mean15", "zt_std15",
            "zt_min7", "ndwi", "ndvi", "bri", "ndwi_mean7", "B", "G", "R", "N", "m", "zt_x_notwater"]
FI = {n: i for i, n in enumerate(FEATURES)}
SAR_CH = ["z", "zt", "zt_std7"]
OPT_CH = ["B", "G", "R", "N", "ndwi", "ndvi", "m"]


def set_all_seeds(seed=C.SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_meta(seed=C.SEED):
    meta = dict(seed=seed, python=platform.python_version(), numpy=np.__version__)
    for mod in ("torch", "lightgbm", "sklearn", "rasterio", "pulp", "networkx", "scipy", "geopandas"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            pass
    return meta


def file_sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(obj, path):
    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- official tiles
def _index(d):
    m = {}
    for f in os.listdir(d):
        stem = os.path.splitext(f)[0]
        try:
            m[int(stem.split("_")[-1])] = os.path.join(d, f)
        except ValueError:
            continue
    return m


def official_index():
    L, S, O = _index(C.OFFICIAL_LABEL_DIR), _index(C.OFFICIAL_SAR_DIR), _index(C.OFFICIAL_OPT_DIR)
    ids = sorted(set(L) & set(S) & set(O))
    assert len(ids) == len(L) == len(S) == len(O) == 1000, (len(ids), len(L), len(S), len(O))
    return ids, L, S, O


def resample_nearest(src, src_T, dst_T, shape, fill=0):
    """Nearest-neighbour resampling of a (..., h, w) array between two north-up grids."""
    H, W = shape
    xs = dst_T.c + (np.arange(W) + 0.5) * dst_T.a
    ys = dst_T.f + (np.arange(H) + 0.5) * dst_T.e
    cols = np.floor((xs - src_T.c) / src_T.a).astype(int)
    rows = np.floor((ys - src_T.f) / src_T.e).astype(int)
    vc = (cols >= 0) & (cols < src.shape[-1])
    vr = (rows >= 0) & (rows < src.shape[-2])
    out = np.full(src.shape[:-2] + (H, W), fill, dtype=src.dtype)
    rr, cc = np.meshgrid(np.clip(rows, 0, src.shape[-2] - 1), np.clip(cols, 0, src.shape[-1] - 1), indexing="ij")
    out[...] = src[..., rr, cc]
    inside = vr[:, None] & vc[None, :]
    out[..., ~inside] = fill
    return out, inside


def read_official_tile(tid, L, S, O):
    """Returns dict with label y (uint8), sar (uint8), opt (4,H,W uint8), valid mask, SAR transform.
    The 9 anisotropic label tiles are resampled by nearest neighbour onto the SAR grid; SAR pixels
    outside the label footprint and SAR no-data (DN=0) are invalid."""
    with rasterio.open(S[tid]) as d:
        sar = d.read(1)
        T = d.transform
    with rasterio.open(L[tid]) as d:
        lab = d.read(1)
        TL = d.transform
    with rasterio.open(O[tid]) as d:
        opt = d.read()
        TO = d.transform
    shape = sar.shape
    resampled = False
    if abs(TL.a - T.a) / T.a > 0.01 or abs(TL.e - T.e) / abs(T.e) > 0.01:
        lab, foot = resample_nearest(lab, TL, T, shape)
        resampled = True
    else:
        foot = np.ones(shape, bool)
    if abs(TO.a - T.a) / T.a > 0.01 or abs(TO.e - T.e) / abs(T.e) > 0.01:
        opt, _ = resample_nearest(opt, TO, T, shape)
    valid = foot & (sar > 0)
    return dict(y=(lab > 0).astype(np.uint8), sar=sar, opt=opt.astype(np.uint8), valid=valid,
                T=T, resampled=resampled)


def pixel_area_rows(T, H):
    """(3.1) WGS-84 ellipsoidal pixel area for each row of an EPSG:4326 grid (m^2)."""
    lat = np.deg2rad(T.f + (np.arange(H) + 0.5) * T.e)
    s2 = np.sin(lat) ** 2
    M = C.WGS84_A * (1 - C.WGS84_E2) / (1 - C.WGS84_E2 * s2) ** 1.5
    N = C.WGS84_A / np.sqrt(1 - C.WGS84_E2 * s2)
    return M * np.deg2rad(abs(T.e)) * N * np.cos(lat) * np.deg2rad(abs(T.a))


# ---------------------------------------------------------------- SAR
def rank_normalize(I, valid):
    """(3.2) per-tile 2-98 % quantile stretch on valid pixels."""
    v = I[valid]
    if v.size < 10:
        return np.zeros(I.shape, np.float32)
    q2, q98 = np.percentile(v, [2, 98])
    z = np.clip((I.astype(np.float32) - q2) / (q98 - q2 + C.EPS), 0, 1)
    z[~valid] = 0
    return z.astype(np.float32)


def local_stats(x, size):
    m = ndi.uniform_filter(x, size, mode="reflect")
    m2 = ndi.uniform_filter(x * x, size, mode="reflect")
    return m, np.maximum(m2 - m * m, 0)


def estimate_enl(z, valid):
    """ENL = median of mean^2/var over the 10 % lowest-CV 7x7 windows (homogeneous areas)."""
    m, v = local_stats(z, 7)
    vv = ndi.minimum_filter(valid.astype(np.uint8), 7) > 0
    sel = vv[3::7, 3::7] & (m[3::7, 3::7] > 0.02) & (v[3::7, 3::7] > 1e-8)
    mm, var = m[3::7, 3::7][sel], v[3::7, 3::7][sel]
    if mm.size < 10:
        return 1.0
    cv = np.sqrt(var) / mm
    k = max(1, int(0.1 * cv.size))
    idx = np.argsort(cv)[:k]
    return float(np.clip(np.median(mm[idx] ** 2 / var[idx]), 1.0, 100.0))


def lee_filter(z, valid, L=None):
    """(3.3) Lee filter, 5x5 window, Cu^2 = 1/L."""
    if L is None:
        L = estimate_enl(z, valid)
    m, v = local_stats(z, 5)
    cz2 = v / (m * m + C.EPS)
    k = np.maximum(0, 1 - (1.0 / L) / np.maximum(cz2, 1e-12))
    zt = (m + k * (z - m)).astype(np.float32)
    zt[~valid] = 0
    return zt, L


def ashman_d(a, b):
    return np.sqrt(2) * abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var() + 1e-12)


def split_otsu(zt, valid, block=64, d_min=2.0, min_share=0.1):
    """Split-based Otsu (§4.2): returns (tile threshold or None, share of sub-blocks accepted)."""
    H, W = zt.shape
    acc, n = [], 0
    for r in range(0, H, block):
        for c in range(0, W, block):
            v = zt[r:r + block, c:c + block][valid[r:r + block, c:c + block]]
            if v.size < 0.5 * block * block:
                continue
            n += 1
            if np.ptp(v) < 1e-6:
                continue
            t = threshold_otsu(v)
            lo, hi = v[v <= t], v[v > t]
            if min(lo.size, hi.size) < min_share * v.size:
                continue
            if ashman_d(lo, hi) > d_min:
                acc.append(t)
    share = len(acc) / n if n else 0.0
    return (float(np.median(acc)) if acc else None), share


def mmu_filter(mask, min_px):
    """Remove 8-connected components smaller than min_px."""
    if min_px <= 1 or not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------- optical
def optical_indices(opt, m):
    B, G, R, N = [opt[i].astype(np.float32) for i in range(4)]
    ndwi = (G - N) / (G + N + C.EPS)
    ndvi = (N - R) / (N + R + C.EPS)
    bri = 0.25 * (B + G + R + N) / 255.0
    for a in (ndwi, ndvi, bri):
        a[~m] = 0
    return ndwi, ndvi, bri


def feature_stack(z, zt, opt, m):
    """Pixel features of §3 (19 channels, float32). Uses only the pixel's own tile."""
    ndwi, ndvi, bri = optical_indices(opt, m)
    f = np.zeros((len(FEATURES),) + z.shape, np.float32)
    f[FI["z"]], f[FI["zt"]] = z, zt
    for s in (3, 7, 15):
        mu, var = local_stats(zt, s)
        f[FI[f"zt_mean{s}"]], f[FI[f"zt_std{s}"]] = mu, np.sqrt(var)
    f[FI["zt_min7"]] = ndi.minimum_filter(zt, 7)
    f[FI["ndwi"]], f[FI["ndvi"]], f[FI["bri"]] = ndwi, ndvi, bri
    f[FI["ndwi_mean7"]] = ndi.uniform_filter(ndwi, 7)
    for i, b in enumerate("BGRN"):
        f[FI[b]] = opt[i].astype(np.float32) / 255.0 * m
    f[FI["m"]] = m.astype(np.float32)
    f[FI["zt_x_notwater"]] = zt * (1 - np.maximum(ndwi, 0))
    return f


# ---------------------------------------------------------------- metrics
def confusion(yhat, y, valid, w=None):
    yh, yy = yhat[valid].astype(bool), y[valid].astype(bool)
    ww = None if w is None else w[valid]
    def s(mask):
        return float(mask.sum()) if ww is None else float(ww[mask].sum())
    return dict(tp=s(yh & yy), fp=s(yh & ~yy), fn=s(~yh & yy), tn=s(~yh & ~yy))


def pixel_metrics(cm):
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn
    oa = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    return dict(OA=oa, Precision=prec, Recall=rec, F1=f1, IoU=iou, kappa=kappa)


def area_metrics(A_ref, A_hat, floor=0.05):
    """(4.3) area metrics from per-tile areas (km^2)."""
    A_ref, A_hat = np.asarray(A_ref, float), np.asarray(A_hat, float)
    e = A_hat - A_ref
    sel = A_ref >= floor
    rae_t = e[sel] / A_ref[sel]
    return dict(A_ref_total=A_ref.sum(), A_hat_total=A_hat.sum(), DeltaA=e.sum(),
                RAE=e.sum() / A_ref.sum(), MAE_A=np.abs(e).mean(),
                median_abs_RAE_t=float(np.median(np.abs(rae_t))) if sel.any() else None,
                BA_bias=e.mean(), BA_loa_low=e.mean() - 1.96 * e.std(ddof=1),
                BA_loa_high=e.mean() + 1.96 * e.std(ddof=1), n_tiles=len(e))


def validate_constraints(results, constraints, tag=""):
    """Generic bound checker: constraints = {key: (lo, hi, desc)}. Prints conclusions only."""
    viol = []
    for key, (lo, hi, desc) in constraints.items():
        val = results.get(key)
        if val is None:
            viol.append(f"FAIL {key}: missing")
            continue
        arr = np.atleast_1d(np.asarray(val, dtype=float)).ravel()
        if arr.size == 0:
            viol.append(f"FAIL {key}: empty")
            continue
        bad = ~np.isfinite(arr) | (arr < lo) | (arr > hi)
        if bad.any():
            viol.append(f"FAIL {desc}: {int(bad.sum())} values outside [{lo},{hi}], e.g. {arr[bad][:3]}")
    print(f"[{tag}] validate_constraints PASS={not viol} n_violations={len(viol)}")
    for v in viol[:5]:
        print("   ", v)
    return not viol
