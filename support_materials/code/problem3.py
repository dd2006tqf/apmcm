"""Problem 3 — newly inundated area of Hengzhou (Guangxi), July 2026 (MODELING_REPORT.md §6).

Transfer of the unchanged Q2 M4/M6 ensemble (5 fold checkpoints, 2 MC passes each) to Sentinel-1 RTC
(VV, dB) with per-tile rank normalization + Lee filter (same functions as Q1), optical quantile
mapping of Sentinel-2 2026-05-31, and physical guards: pre-flood water union W0 (6.2), REM <= h*,
change dVV <= -delta, MMU 50 px (6.3). Outputs F, W1, receded water, A_new, A_net, rho (6.5).
"""
import os, sys, json, time, math
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.transform import Affine
import geopandas as gpd
from scipy import ndimage as ndi
from scipy.stats import ks_2samp
from pyproj import Proj, Geod
import config as C
import common as U
import problem1 as P1
import problem2 as P2

DEV = P2.DEV
CQ3 = C.CACHE / "q3"
CQ3.mkdir(parents=True, exist_ok=True)
S1F = {d: C.PUB / "Sentinel1" / f"Sentinel1_{d}_VV_VH_dB.tif" for d in [C.PRE_DATE] + C.FLOOD_DATES}
S2F = C.PUB / "Sentinel2" / "Sentinel2_20260531_B432_NIR_SCL.tif"
WCF = C.PUB / "LandCover" / "WorldCover2021_Hengzhou_10m.tif"
DEMF = C.PUB / "DEM_clip" / "DEM_Hengzhou_30m_mosaic.tif"
BNDF = C.PUB / "Boundary" / "Hengzhou_boundary.geojson"
OSM = C.PUB / "OSM"
NODATA = -999.9
T = C.TILE


def grid():
    with rasterio.open(S1F[C.PRE_DATE]) as d:
        return d.transform, d.shape, d.crs


def mmap(name, dtype, shape, mode="r"):
    return np.memmap(CQ3 / name, dtype=dtype, mode=mode, shape=shape)


def put(name, arr):
    out = np.memmap(CQ3 / name, dtype=arr.dtype, mode="w+", shape=arr.shape)
    out[:] = arr
    out.flush()
    del out


def tiles_of(shape):
    Hh, Ww = shape
    return [(r, c) for r in range(0, Hh, T) for c in range(0, Ww, T)]


# ============================================================ layer preparation
def water_hit(arr, rr, cc, dr, dc, thr=-15.0):
    """Share of reference water points whose SAR value at (r+dr, c+dc) is darker than thr dB."""
    r2, c2 = rr + dr, cc + dc
    m = (r2 >= 0) & (r2 < arr.shape[0]) & (c2 >= 0) & (c2 < arr.shape[1])
    v = np.asarray(arr[r2[m], c2[m]])
    return float(np.mean(np.nan_to_num(v, nan=0) < thr)) if v.size else 0.0


def coreg_shift(raw, ref, k=4, n_pts=120_000):
    """Integer translation (dr, dc) such that corrected[r, c] = raw[r + dr, c + dc] maximizes the share of
    permanent-water reference pixels that are SAR-dark: FFT cross-correlation on a 4x decimated grid,
    then a +-6 px exhaustive refinement on the full grid."""
    rng = np.random.default_rng(C.SEED)
    rr, cc = np.nonzero(ref)
    sel = rng.choice(len(rr), min(n_pts, len(rr)), replace=False)
    rr, cc = rr[sel], cc[sel]
    a = ref[::k, ::k].astype(float)
    b = (np.nan_to_num(np.asarray(raw[::k, ::k]), nan=0) < -15).astype(float)
    a, b = a - a.mean(), b - b.mean()
    c = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    i, j = np.unravel_index(np.argmax(c), c.shape)
    i = i if i < a.shape[0] // 2 else i - a.shape[0]
    j = j if j < a.shape[1] // 2 else j - a.shape[1]
    cands = [(sr * i * k, sc * j * k) for sr in (1, -1) for sc in (1, -1)]
    dr0, dc0 = max(cands, key=lambda t: water_hit(raw, rr, cc, *t))
    best = max(((dr0 + x, dc0 + y) for x in range(-6, 7) for y in range(-6, 7)), key=lambda t: water_hit(raw, rr, cc, *t))
    return dict(dr=int(best[0]), dc=int(best[1]), score_before=water_hit(raw, rr, cc, 0, 0),
                score_after=water_hit(raw, rr, cc, *best), shift_north_m=-10 * int(best[0]), shift_east_m=10 * int(best[1]))


def apply_shift(raw, name, dr, dc):
    Hh, Ww = raw.shape
    out = mmap(name, np.float32, raw.shape, "w+")
    for r in range(0, Hh, 512):
        h = min(512, Hh - r)
        blk = np.full((h, Ww), np.nan, np.float32)
        r0, r1 = max(0, r + dr), min(Hh, r + dr + h)
        c0, c1 = max(0, dc), min(Ww, Ww + dc)
        if r1 > r0 and c1 > c0:
            blk[r0 - (r + dr):r1 - (r + dr), c0 - dc:c1 - dc] = raw[r0:r1, c0:c1]
        out[r:r + h] = blk
    out.flush(); del out


def s2_alignment(ref, k=4):
    """Sentinel-2 check: share of reference water pixels with NDWI > 0 (valid SCL) at zero shift and at
    the best coarse translation. S2 is used only if it reaches 0.3 (SAR reaches > 0.35 after co-registration)."""
    with rasterio.open(S2F) as s2:
        a = s2.read(out_shape=(5, ref.shape[0] // k, ref.shape[1] // k))
    G, N, scl = a[1].astype(float), a[3].astype(float), a[4]
    ok = np.isin(scl, [4, 5, 6, 7, 11]) & (G + N > 0)
    wat = (G - N) / (G + N + 1e-6) > 0
    rr, cc = np.nonzero(ref[::k, ::k][:ok.shape[0], :ok.shape[1]])
    res = []
    for dr in range(-200, 201, 5):
        for dc in range(-80, 81, 5):
            r2, c2 = rr + dr, cc + dc
            m = (r2 >= 0) & (r2 < ok.shape[0]) & (c2 >= 0) & (c2 < ok.shape[1])
            v = ok[r2[m], c2[m]]
            if v.sum() >= 1000:
                res.append((float(wat[r2[m], c2[m]][v].mean()), dr * k, dc * k))
    zero = [x[0] for x in res if x[1] == 0 and x[2] == 0]
    best = max(res)
    return dict(hit_zero_shift=zero[0] if zero else None, hit_best=best[0], best_dr=best[1], best_dc=best[2],
                valid_share=float(ok.mean()), usable=best[0] >= 0.3)


def prepare_layers():
    TR, shape, crs = grid()
    Hh, Ww = shape
    meta_f = CQ3 / "layers.json"
    if meta_f.exists():
        print("[Q3] layer cache reused")
        return json.load(open(meta_f))
    t0 = time.time()
    # boundary
    bnd = gpd.read_file(BNDF).to_crs(crs)
    bm = features.rasterize([(g, 1) for g in bnd.geometry], out_shape=shape, transform=TR, dtype=np.uint8)
    put("boundary.u1", bm)
    geod = Geod(ellps="WGS84")
    b4326 = gpd.read_file(BNDF)
    a_geo = abs(sum(geod.geometry_area_perimeter(g)[0] for g in b4326.geometry)) * 1e-6
    # WorldCover -> S1 grid (nearest)
    wc = np.zeros(shape, np.uint8)
    with rasterio.open(WCF) as src:
        reproject(rasterio.band(src, 1), wc, dst_transform=TR, dst_crs=crs, resampling=Resampling.nearest)
    put("wc.u1", wc)
    # OSM water polygons -> S1 grid
    wpoly = gpd.read_file(OSM / "water_surface_merged.geojson").to_crs(crs)
    osmw = features.rasterize([(g, 1) for g in wpoly.geometry if g is not None], out_shape=shape, transform=TR,
                              dtype=np.uint8)
    put("osmwater.u1", osmw)
    # SAR VV/VH per date: raw cache, then co-registration to the permanent-water reference (WorldCover 80
    # AND OSM water, two independent sources that agree with each other) by an integer translation
    coreg = {}
    ref = (wc == 80) & (osmw > 0)
    for d, f in S1F.items():
        for b, nm in ((1, "vv"), (2, "vh")):
            out = mmap(f"raw_{nm}_{d}.f4", np.float32, shape, "w+")
            with rasterio.open(f) as src:
                for r in range(0, Hh, 512):
                    a = src.read(b, window=Window(0, r, Ww, min(512, Hh - r)))
                    a[(a <= NODATA + 1) | ~np.isfinite(a)] = np.nan
                    out[r:r + a.shape[0]] = a
            out.flush(); del out
        raw = mmap(f"raw_vv_{d}.f4", np.float32, shape)
        coreg[d] = coreg_shift(raw, ref)
        for nm in ("vv", "vh"):
            apply_shift(mmap(f"raw_{nm}_{d}.f4", np.float32, shape), f"{nm}_{d}.f4", coreg[d]["dr"], coreg[d]["dc"])
        print(f"[Q3] SAR {d}: shift dr={coreg[d]['dr']} dc={coreg[d]['dc']} px, water hit "
              f"{coreg[d]['score_before']:.3f} -> {coreg[d]['score_after']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    s2_check = s2_alignment(ref)
    print(f"[Q3] Sentinel-2 alignment check: {s2_check}", flush=True)
    # REM at 30 m (6.1): drainage = OSM waterways (river/canal/stream) + OSM water polygons + WC 80
    T30 = Affine(30.0, 0, TR.c, 0, -30.0, TR.f)
    s30 = (math.ceil(Hh / 3), math.ceil(Ww / 3))
    dem = np.full(s30, np.nan, np.float32)
    with rasterio.open(DEMF) as src:
        reproject(rasterio.band(src, 1), dem, dst_transform=T30, dst_crs=crs, resampling=Resampling.bilinear,
                  dst_nodata=np.nan)
    ww = gpd.read_file(OSM / "waterways_lines.geojson").to_crs(crs)
    ww = ww[ww["waterway"].isin(["river", "canal", "stream"])]
    dr = features.rasterize([(g, 1) for g in ww.geometry if g is not None], out_shape=s30, transform=T30,
                            dtype=np.uint8, all_touched=True)
    dr |= features.rasterize([(g, 1) for g in wpoly.geometry if g is not None], out_shape=s30, transform=T30,
                             dtype=np.uint8, all_touched=True)
    pad = np.zeros((s30[0] * 3, s30[1] * 3), np.uint8)
    pad[:Hh, :Ww] = (wc == 80)
    dr |= pad.reshape(s30[0], 3, s30[1], 3).max((1, 3))
    del pad, osmw
    demf = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    dist, ind = ndi.distance_transform_edt(dr == 0, return_indices=True)
    rem = np.maximum(0, demf - demf[ind[0], ind[1]]).astype(np.float32)
    rem[~np.isfinite(dem)] = np.nan
    np.save(CQ3 / "rem30.npy", rem)
    np.save(CQ3 / "dem30.npy", dem)
    np.save(CQ3 / "drain_dist30.npy", dist.astype(np.float32))
    del dist, ind
    # pixel area: UTM point scale factor at each tile centre (variation inside a tile < 1e-6)
    pj = Proj(crs.to_string())
    tl = tiles_of(shape)
    area = {}
    for (r, c) in tl:
        x, y = TR * (c + T / 2, r + T / 2)
        lon, lat = pj(x, y, inverse=True)
        k = pj.get_factors(lon, lat).meridional_scale
        area[f"{r}_{c}"] = 100.0 / k ** 2
    meta = dict(shape=shape, transform=list(TR)[:6], crs=str(crs), boundary_px=int(bm.sum()),
                boundary_area_geodesic_km2=a_geo, pixel_area_m2=area, drainage_px30=int(dr.sum()), coregistration=coreg,
                s2_check=s2_check, s2_used=bool(s2_check['usable']),
                seconds=time.time() - t0)
    json.dump(meta, open(meta_f, "w"))
    print(f"[Q3] layers ready ({time.time() - t0:.0f}s) geodesic boundary area {a_geo:.1f} km2", flush=True)
    return meta
# ============================================================ physics water detector, W0, dVV
def tile_slice(r, c, shape):
    return slice(r, min(r + T, shape[0])), slice(c, min(c + T, shape[1]))


def pad_tile(a, fill=0):
    if a.shape == (T, T):
        return a
    out = np.full((T, T), fill, a.dtype)
    out[:a.shape[0], :a.shape[1]] = a
    return out


def lee_db(vv_db, valid):
    """Lee filter on linear power (multiplicative speckle model), returned in dB."""
    p = np.where(valid, 10 ** (np.nan_to_num(vv_db, nan=-30) / 10), 0).astype(np.float32)
    pf, L = U.lee_filter(p, valid)
    return (10 * np.log10(np.maximum(pf, 1e-6))).astype(np.float32), L


def split_otsu_water(f, valid, block=128, min_share=0.1, dark_max=None):
    """Split-based Otsu (Sec. 4.2: sub-blocks, Ashman D > 2, minor class >= 10 %, median of accepted
    thresholds) for calibrated Sentinel-1 VV in dB, with one physical acceptance rule: the dark class of a
    sub-block must be water-like (mean <= WATER_DB_MAX = -15 dB). Without it, land-cover contrasts pass the
    bimodality test and the threshold falls at the scene median (about -7 dB, 52 % of the city 'water').
    Sub-blocks are 128 px (1.28 km) because 64-px blocks almost never contain 10 % open water here."""
    from skimage.filters import threshold_otsu
    dark_max = C.WATER_DB_MAX if dark_max is None else dark_max
    acc, n = [], 0
    for r in range(0, f.shape[0], block):
        for c in range(0, f.shape[1], block):
            v = f[r:r + block, c:c + block][valid[r:r + block, c:c + block]]
            if v.size < 0.5 * block * block:
                continue
            n += 1
            if np.ptp(v) < 1e-6:
                continue
            t = threshold_otsu(v)
            lo, hi = v[v <= t], v[v > t]
            if min(lo.size, hi.size) < min_share * v.size or lo.mean() > dark_max:
                continue
            if U.ashman_d(lo, hi) > 2.0:
                acc.append(t)
    return (float(np.median(acc)) if acc else None), (len(acc) / n if n else 0.0), acc


def physics_layers(meta):
    """D_d (split-based Otsu on Lee-filtered VV dB), Lee-filtered VV, W0 (6.2) and S2 NDWI."""
    shape = tuple(meta["shape"])
    f_done = CQ3 / "physics.json"
    if f_done.exists():
        return json.load(open(f_done))
    bm = mmap("boundary.u1", np.uint8, shape)
    tl = [(r, c) for (r, c) in tiles_of(shape) if bm[tile_slice(r, c, shape)].any()]
    info = {"tiles_in_boundary": len(tl)}
    for d in [C.PRE_DATE] + C.FLOOD_DATES:
        vv = mmap(f"vv_{d}.f4", np.float32, shape)
        lf = mmap(f"vvlee_{d}.f4", np.float32, shape, "w+")
        taus, ash, enl, sub = {}, [], [], []
        for (r, c) in tl:
            sl = tile_slice(r, c, shape)
            a = pad_tile(np.array(vv[sl]), np.nan)
            valid = np.isfinite(a)
            f, L = lee_db(a, valid)
            lf[sl] = f[:sl[0].stop - sl[0].start, :sl[1].stop - sl[1].start]
            t, s, a_sub = split_otsu_water(f, valid)
            taus[(r, c)] = t
            sub += a_sub
            ash.append(s); enl.append(L)
        lf.flush()
        acc = [t for t in taus.values() if t is not None]
        # fallback = median of all accepted sub-block thresholds of the scene (pooled, more stable than
        # the median of the few tile medians when most tiles contain no open water)
        tau_g = float(np.median(sub)) if sub else C.WATER_DB_MAX
        D = mmap(f"D_{d}.u1", np.uint8, shape, "w+")
        for (r, c) in tl:
            sl = tile_slice(r, c, shape)
            t = taus[(r, c)] if taus[(r, c)] is not None else tau_g
            f = np.array(lf[sl])
            D[sl] = (np.isfinite(f) & (f < t)).astype(np.uint8)
        D.flush()
        info[d] = dict(tau_global_db=tau_g, n_fallback=int(sum(t is None for t in taus.values())),
                       n_subblocks_accepted=len(sub), tile_tau_median=float(np.median(acc)) if acc else None, ashman_share_mean=float(np.mean(ash)),
                       enl_median=float(np.median(enl)))
        print(f"[Q3] physics water {d}: tau_g={tau_g:.2f} dB fallback={info[d]['n_fallback']}", flush=True)
    # W0 union (6.2)
    wc = mmap("wc.u1", np.uint8, shape)
    osmw = mmap("osmwater.u1", np.uint8, shape)
    D0 = mmap(f"D_{C.PRE_DATE}.u1", np.uint8, shape)
    W0u = mmap("W0_union.u1", np.uint8, shape, "w+")
    W0s = mmap("W0_sar.u1", np.uint8, shape, "w+")
    m2 = mmap("s2valid.u1", np.uint8, shape, "w+")
    nd = mmap("s2ndwi.f2", np.float16, shape, "w+")
    src_px = dict(sar=0, wc80=0, osm=0, s2ndwi=0)
    with rasterio.open(S2F) as s2:
        for r in range(0, shape[0], 512):
            h = min(512, shape[0] - r)
            win = Window(0, r, shape[1], h)
            G = s2.read(2, window=win).astype(np.float32)
            N = s2.read(4, window=win).astype(np.float32)
            scl = s2.read(5, window=win)
            ok = np.isin(scl, [4, 5, 6, 7, 11]) & (G + N > 0) & bool(meta.get("s2_used", True))
            ndwi = np.where(ok, (G - N) / (G + N + C.EPS), 0)
            sl = slice(r, r + h)
            w_s = D0[sl] > 0
            w_wc = wc[sl] == 80
            w_osm = osmw[sl] > 0
            w_nd = ok & (ndwi > 0)
            inb = bm[sl] > 0
            W0u[sl] = (w_s | w_wc | w_osm | w_nd).astype(np.uint8)
            W0s[sl] = w_s.astype(np.uint8)
            m2[sl] = ok.astype(np.uint8)
            nd[sl] = ndwi.astype(np.float16)
            for k_, w in (("sar", w_s), ("wc80", w_wc), ("osm", w_osm), ("s2ndwi", w_nd)):
                src_px[k_] += int((w & inb).sum())
    for a in (W0u, W0s, m2, nd):
        a.flush()
    info["W0_source_px_in_boundary"] = src_px
    json.dump(info, open(f_done, "w"))
    return info


# ============================================================ model input mapping
def official_band_quantiles(nq=1000, per_tile=2000):
    feats = P1.mm("feats.f16", np.float16, (P1.NT, P1.NF, P1.H, P1.H))
    rng = np.random.default_rng(C.SEED)
    vals = [[] for _ in range(4)]
    for i in range(P1.NT):
        m = feats[i, U.FI["m"]] > 0.5
        idx = np.flatnonzero(m.ravel())
        if not len(idx):
            continue
        sel = rng.choice(idx, min(per_tile, len(idx)), replace=False)
        for b, nm in enumerate("BGRN"):
            vals[b].append(feats[i, U.FI[nm]].ravel()[sel].astype(np.float32) * 255)
    q = np.linspace(0, 1, nq)
    return np.array([np.quantile(np.concatenate(v), q) for v in vals])


def hengzhou_band_quantiles(nq=1000, n=2_000_000):
    rng = np.random.default_rng(C.SEED)
    out = [[] for _ in range(4)]
    with rasterio.open(S2F) as s2:
        for r in range(0, s2.height, 512):
            win = Window(0, r, s2.width, min(512, s2.height - r))
            a = s2.read(window=win)
            ok = np.isin(a[4], [4, 5, 6, 7, 11]) & (a[:4].min(0) > 0)
            idx = np.flatnonzero(ok.ravel())
            if not len(idx):
                continue
            sel = rng.choice(idx, min(len(idx), n // 15 + 1), replace=False)
            for b in range(4):
                out[b].append(a[b].ravel()[sel].astype(np.float32))
    q = np.linspace(0, 1, nq)
    return np.array([np.quantile(np.concatenate(v), q) for v in out])


def sar_input(vv_db, valid, scale):
    """Per-tile representation identical to training: rank normalization (3.2) + Lee (3.3) + std7."""
    a = np.nan_to_num(vv_db, nan=-30.0)
    I = a if scale == "db" else 10 ** (a / 20)
    z = U.rank_normalize(I, valid)
    zt, _ = U.lee_filter(z, valid)
    _, var7 = U.local_stats(zt, 7)
    return z, zt, np.sqrt(var7).astype(np.float32)


def ks_scale_choice(meta, n_tiles=80, per_tile=1500):
    """A5: choose dB vs amplitude by KS distance to official non-flood, non-water z~ distribution."""
    shape = tuple(meta["shape"])
    feats = P1.mm("feats.f16", np.float16, (P1.NT, P1.NF, P1.H, P1.H))
    Y = P1.mm("y.u8", np.uint8, (P1.NT, P1.H, P1.H))
    V = P1.mm("valid.u8", np.uint8, (P1.NT, P1.H, P1.H))
    rng = np.random.default_rng(C.SEED)
    off = []
    for i in range(0, P1.NT, 5):
        sel = (V[i] > 0) & (Y[i] == 0) & (feats[i, U.FI["ndwi"]] <= 0)
        v = feats[i, U.FI["zt"]][sel].astype(np.float32)
        if len(v):
            off.append(rng.choice(v, min(per_tile, len(v)), replace=False))
    off = np.concatenate(off)
    bm = mmap("boundary.u1", np.uint8, shape)
    wc = mmap("wc.u1", np.uint8, shape)
    vv = mmap(f"vv_{C.MAIN_DATE}.f4", np.float32, shape)
    tl = [(r, c) for (r, c) in tiles_of(shape) if bm[tile_slice(r, c, shape)].mean() > 0.9]
    pick = [tl[j] for j in rng.choice(len(tl), min(n_tiles, len(tl)), replace=False)]
    hz = {"db": [], "amplitude": []}
    for (r, c) in pick:
        sl = tile_slice(r, c, shape)
        a = pad_tile(np.array(vv[sl]), np.nan)
        valid = np.isfinite(a)
        land = pad_tile(np.array(wc[sl]), 0) != 80
        for s in hz:
            _, zt, _ = sar_input(a, valid, s)
            v = zt[valid & land]
            hz[s].append(rng.choice(v, min(per_tile, len(v)), replace=False))
    ks = {s: float(ks_2samp(np.concatenate(v), off).statistic) for s, v in hz.items()}
    return min(ks, key=ks.get), ks
# ============================================================ ensemble, pooled calibration
def load_members(q2):
    members = []
    for cal in q2["calibration"]:
        ck = C.ROOT / cal["checkpoint"]
        assert U.file_sha(ck) == cal["checkpoint_sha256"], f"checkpoint hash mismatch {ck}"
        m = P2.make_model("M4")
        with open(ck, "rb") as fh:
            m.load_state_dict(torch.load(fh, map_location=DEV))
        members.append(m)
    return members


def pooled_calibration(q2):
    """Isotonic map and tau_A refit on the pooled calibration-fold predictions of the five members."""
    folds = json.load(open(C.OUT / "folds.json", encoding="utf-8"))
    pos = {t: i for i, t in enumerate(folds["ids"])}
    D = P2.Data.__new__(P2.Data)
    Y = P1.mm("y.u8", np.uint8, (P1.NT, P1.H, P1.H))
    V = P1.mm("valid.u8", np.uint8, (P1.NT, P1.H, P1.H))
    A = np.load(P1.CQ / "area_rows.npy") * 1e-6
    pb, yy, vv, aa = [], [], [], []
    for k, f in enumerate(folds["folds"]):
        idx = [pos[t] for t in f["cal"]]
        pb.append(torch.from_numpy(np.load(P2.CQ2 / f"cal_pbar_fold{k}.npy")).to(DEV))
        yy.append(torch.from_numpy(np.array(Y[idx])).to(DEV))
        vv.append(torch.from_numpy(np.array(V[idx])).to(DEV))
        aa.append(torch.from_numpy(A[idx].astype(np.float32)).to(DEV))
    pb, yy, vv, aa = torch.cat(pb), torch.cat(yy), torch.cat(vv), torch.cat(aa)
    iso = P2.fit_isotonic(pb, yy, vv, seed=C.SEED)
    wA = aa[:, :, None].expand_as(pb) * (vv > 0)
    exp_area = float((wA * P2.iso_apply(iso, pb)).sum())
    tauA, mask_area = P2.area_preserving_tau(pb, wA, exp_area)
    del pb, yy, vv, aa, wA
    torch.cuda.empty_cache() if DEV.type == "cuda" else None
    return iso, tauA, dict(tauA=tauA, cal_mask_area_km2=mask_area, cal_expected_area_km2=exp_area)


def run_ensemble(meta, members, tauA, scale, date, bq_off, bq_hz, tag):
    """p_bar for every tile intersecting the boundary; per-pass guarded tile areas for sigma_t."""
    shape = tuple(meta["shape"])
    out_f = CQ3 / f"pbar_{date}_{scale}.f2"
    info_f = CQ3 / f"pbar_{date}_{scale}.json"
    if out_f.exists() and info_f.exists():
        return json.load(open(info_f))
    TR = Affine(*meta["transform"])
    bm = mmap("boundary.u1", np.uint8, shape)
    vv = mmap(f"vv_{date}.f4", np.float32, shape)
    lf_d, lf_0 = mmap(f"vvlee_{date}.f4", np.float32, shape), mmap(f"vvlee_{C.PRE_DATE}.f4", np.float32, shape)
    W0 = mmap("W0_union.u1", np.uint8, shape)
    rem30 = np.load(CQ3 / "rem30.npy")
    PB = mmap(f"pbar_{date}_{scale}.f2", np.float16, shape, "w+")
    tl = [(r, c) for (r, c) in tiles_of(shape) if bm[tile_slice(r, c, shape)].any()]
    tile_rows = []
    t0 = time.time()
    s2 = rasterio.open(S2F)
    for s in range(0, len(tl), 8):
        batch = tl[s:s + 8]
        xs, xo, guards, areas = [], [], [], []
        for (r, c) in batch:
            sl = tile_slice(r, c, shape)
            a = pad_tile(np.array(vv[sl]), np.nan)
            valid = np.isfinite(a)
            z, zt, sd7 = sar_input(a, valid, scale)
            o = s2.read(window=Window(c, r, T, T), boundless=True, fill_value=0)
            m = np.isin(o[4], [4, 5, 6, 7, 11]) & (o[:4].min(0) > 0) & bool(meta.get("s2_used", True))
            mapped = np.stack([np.interp(o[b].astype(np.float32), bq_hz[b], bq_off[b]) for b in range(4)])
            mapped[:, ~m] = 0
            ndwi, ndvi, _ = U.optical_indices(mapped, m)
            xs.append(np.stack([z, zt, sd7]))
            xo.append(np.concatenate([mapped / 255.0 * m, ndwi[None], ndvi[None], m[None].astype(np.float32)]))
            rr = np.minimum(np.arange(r, r + T) // 3, rem30.shape[0] - 1)
            cc = np.minimum(np.arange(c, c + T) // 3, rem30.shape[1] - 1)
            rem = rem30[np.ix_(rr, cc)]
            dvv = pad_tile(np.array(lf_d[sl]) - np.array(lf_0[sl]), np.nan)
            g = (pad_tile(np.array(bm[sl]), 0) > 0) & valid & (pad_tile(np.array(W0[sl]), 1) == 0) & \
                (np.nan_to_num(rem, nan=1e9) <= C.REM_MAX) & (np.nan_to_num(dvv, nan=0) <= -C.DELTA_DB)
            guards.append(g)
            areas.append(meta["pixel_area_m2"][f"{r}_{c}"] * 1e-6)
        xs_t = torch.from_numpy(np.stack(xs)).float().to(DEV)
        xo_t = torch.from_numpy(np.stack(xo)).float().to(DEV)
        passes = []
        with torch.no_grad():
            for mi, mdl in enumerate(members):
                P2.enable_mc_dropout(mdl)
                for l in range(C.Q3_MC_PER_MEMBER):
                    torch.manual_seed(C.SEED * 7 + 1000 * mi + l + 100000 * s)
                    with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
                        passes.append(torch.sigmoid(mdl(xs_t, xo_t).float()))
        pst = torch.stack(passes)                                  # (10, B, T, T)
        pbar = pst.mean(0).cpu().numpy()
        g_t = torch.from_numpy(np.stack(guards)).to(DEV)
        pass_area = ((pst >= tauA) & g_t[None]).float().sum((2, 3)).cpu().numpy()   # px
        for j, (r, c) in enumerate(batch):
            sl = tile_slice(r, c, shape)
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            PB[sl] = pbar[j, :h, :w].astype(np.float16)
            pa = pass_area[:, j] * areas[j]
            tile_rows.append(dict(r=r, c=c, a_px_km2=areas[j], A_pass_mean=float(pa.mean()), sigma=float(pa.std(ddof=1)),
                                  n_boundary_px=int(np.array(bm[sl]).sum())))
        if (s // 8) % 20 == 0:
            print(f"[Q3] {tag} {date} {scale}: {s + len(batch)}/{len(tl)} tiles ({time.time() - t0:.0f}s)", flush=True)
    s2.close()
    PB.flush()
    info = dict(date=date, scale=scale, n_tiles=len(tl), tiles=tile_rows, seconds=time.time() - t0)
    json.dump(info, open(info_f, "w"))
    return info
# ============================================================ guarded new inundation (6.3)-(6.5)
def area_raster(meta):
    shape = tuple(meta["shape"])
    f = CQ3 / "area.f4"
    if not f.exists():
        A = np.memmap(f, dtype=np.float32, mode="w+", shape=shape)
        for (r, c) in tiles_of(shape):
            A[tile_slice(r, c, shape)] = meta["pixel_area_m2"][f"{r}_{c}"] * 1e-6
        A.flush(); del A
    return mmap("area.f4", np.float32, shape)


def rem10_strip(rem30, r0, h, W):
    rr = np.minimum(np.arange(r0, r0 + h) // 3, rem30.shape[0] - 1)
    cc = np.minimum(np.arange(W) // 3, rem30.shape[1] - 1)
    return rem30[np.ix_(rr, cc)]


def compute_F(meta, date, tauA, dtau=0.0, hstar=C.REM_MAX, delta=C.DELTA_DB, w0mode=C.W0_MODE,
              mmu=C.MMU_PX_S1, scale="db", want_stages=False):
    """F_d = MMU( {p>=tau_A+dtau} & ~W0 & {REM<=h*} & {dVV<=-delta} ) inside the boundary."""
    shape = tuple(meta["shape"])
    Hh, Ww = shape
    PB = mmap(f"pbar_{date}_{scale}.f2", np.float16, shape)
    W0 = mmap("W0_union.u1" if w0mode == "union" else "W0_sar.u1", np.uint8, shape)
    bm = mmap("boundary.u1", np.uint8, shape)
    lf_d, lf_0 = mmap(f"vvlee_{date}.f4", np.float32, shape), mmap(f"vvlee_{C.PRE_DATE}.f4", np.float32, shape)
    A = area_raster(meta)
    rem30 = np.load(CQ3 / "rem30.npy")
    cand = np.zeros(shape, bool)
    stages = np.zeros(4)
    for r in range(0, Hh, 512):
        sl = slice(r, min(r + 512, Hh))
        inb = bm[sl] > 0
        raw = (PB[sl] >= tauA + dtau) & inb
        a = A[sl]
        c1 = raw & (W0[sl] == 0)
        rem = rem10_strip(rem30, r, sl.stop - r, Ww)
        c2 = c1 & (np.nan_to_num(rem, nan=1e9) <= hstar)
        dvv = np.nan_to_num(lf_d[sl] - lf_0[sl], nan=0.0)
        c3 = c2 & (dvv <= -delta)
        cand[sl] = c3
        if want_stages:
            stages += [float(a[raw].sum()), float(a[c1].sum()), float(a[c2].sum()), float(a[c3].sum())]
    F = U.mmu_filter(cand, mmu) if mmu > 1 else cand
    del cand
    return (F, stages) if want_stages else F


def water_states(f, w0, d0, dd):
    """W1 and receded water R with observed recession only: a W0 pixel is receded if it was SAR water on the
    pre-flood date and is neither SAR water nor new inundation on the flood date. W0 pixels that SAR never sees
    as water (mapped narrow channels, banks) cannot be observed to change and stay in W1. The literal (6.4)
    variant W1 = F u (W0 & D_d) is reported as A_net_literal."""
    R = w0 & d0 & ~dd & ~f
    W1 = f | (w0 & ~R)
    return W1, R


def areas_of(meta, F, date, w0mode=C.W0_MODE):
    """A_new, A_net, rho (6.5) inside the boundary; W1 = F u (W0 & D_d), R = W0 \\ W1."""
    shape = tuple(meta["shape"])
    W0 = mmap("W0_union.u1" if w0mode == "union" else "W0_sar.u1", np.uint8, shape)
    D = mmap(f"D_{date}.u1", np.uint8, shape)
    D0 = mmap(f"D_{C.PRE_DATE}.u1", np.uint8, shape)
    bm = mmap("boundary.u1", np.uint8, shape)
    A = area_raster(meta)
    rem30 = np.load(CQ3 / "rem30.npy")
    s = dict(A_new=0.0, A_W1=0.0, A_W0=0.0, A_R=0.0, A_new_floodplain=0.0, A_floodplain=0.0, A_boundary=0.0,
             overlap_F_W0_px=0, F_not_in_W1_px=0, A_W1_literal=0.0)
    for r in range(0, shape[0], 512):
        sl = slice(r, min(r + 512, shape[0]))
        inb = bm[sl] > 0
        f = F[sl] & inb
        w0 = (W0[sl] > 0) & inb
        w1, rr = water_states(f, w0, (D0[sl] > 0) & inb, (D[sl] > 0) & inb)
        a = A[sl]
        s["A_W1_literal"] += float(a[f | (w0 & (D[sl] > 0))].sum())
        fp = np.nan_to_num(rem10_strip(rem30, r, sl.stop - r, shape[1]), nan=1e9) <= C.REM_MAX
        s["A_new"] += float(a[f].sum()); s["A_W1"] += float(a[w1].sum()); s["A_W0"] += float(a[w0].sum())
        s["A_R"] += float(a[rr].sum()); s["A_boundary"] += float(a[inb].sum())
        s["A_new_floodplain"] += float(a[f & fp].sum()); s["A_floodplain"] += float(a[fp & inb].sum())
        s["overlap_F_W0_px"] += int((f & w0).sum()); s["F_not_in_W1_px"] += int((f & ~w1).sum())
    s["A_net"] = s["A_W1"] - s["A_W0"]
    s["A_net_literal"] = s["A_W1_literal"] - s["A_W0"]
    s["A_reg"] = meta["boundary_area_geodesic_km2"]
    s["rho"] = s["A_new"] / s["A_reg"]
    s["rho_floodplain"] = s["A_new_floodplain"] / s["A_floodplain"] if s["A_floodplain"] else None
    return s


def write_tif(meta, name, arr, dtype="uint8"):
    shape = tuple(meta["shape"])
    prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, dtype=dtype, crs=meta["crs"],
                transform=Affine(*meta["transform"]), compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(C.OUT / name, "w", **prof) as dst:
        for r in range(0, shape[0], 512):
            h = min(512, shape[0] - r)
            dst.write(np.asarray(arr[r:r + h]).astype(dtype), 1, window=Window(0, r, shape[1], h))


def save_layers(meta, F, date):
    shape = tuple(meta["shape"])
    W0 = mmap("W0_union.u1", np.uint8, shape)
    D = mmap(f"D_{date}.u1", np.uint8, shape)
    D0 = mmap(f"D_{C.PRE_DATE}.u1", np.uint8, shape)
    bm = mmap("boundary.u1", np.uint8, shape)
    PB = mmap(f"pbar_{date}_db.f2", np.float16, shape)
    d4 = date[4:]
    write_tif(meta, f"q3_newflood_{d4}.tif", F)
    W1 = np.zeros(shape, np.uint8); R = np.zeros(shape, np.uint8); Pq = np.zeros(shape, np.uint8)
    for r in range(0, shape[0], 512):
        sl = slice(r, min(r + 512, shape[0]))
        inb = bm[sl] > 0
        w0 = (W0[sl] > 0) & inb
        W1[sl], R[sl] = water_states(F[sl] & inb, w0, (D0[sl] > 0) & inb, (D[sl] > 0) & inb)
        Pq[sl] = np.round(np.nan_to_num(PB[sl].astype(np.float32)) * 255) * inb
    write_tif(meta, f"q3_w1_{d4}.tif", W1)
    write_tif(meta, f"q3_receded_{d4}.tif", R)
    write_tif(meta, f"q3_prob_{d4}.tif", Pq)
    del W1, R, Pq


def receded_breakdown(meta, F, date):
    """Explains A_net: physics water areas per date and the receded set R = W0 \\ W1 split by W0 source
    and by WorldCover class (inside the boundary, km2)."""
    shape = tuple(meta["shape"])
    bm = mmap("boundary.u1", np.uint8, shape)
    wc, osmw = mmap("wc.u1", np.uint8, shape), mmap("osmwater.u1", np.uint8, shape)
    m2, nd = mmap("s2valid.u1", np.uint8, shape), mmap("s2ndwi.f2", np.float16, shape)
    D0, Dd = mmap(f"D_{C.PRE_DATE}.u1", np.uint8, shape), mmap(f"D_{date}.u1", np.uint8, shape)
    W0 = mmap("W0_union.u1", np.uint8, shape)
    A = area_raster(meta)
    out = dict(D_pre=0.0, D_date=0.0, D_pre_and_date=0.0, R_total=0.0,
               R_by_source=dict(sar_0627=0.0, worldcover80=0.0, osm_water=0.0, s2_ndwi=0.0, sar_0627_only=0.0),
               R_by_landcover={})
    names = {10: "Tree", 20: "Shrub", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare", 80: "Water", 90: "Wetland"}
    for r in range(0, shape[0], 512):
        sl = slice(r, min(r + 512, shape[0]))
        inb = bm[sl] > 0
        a = A[sl]
        d0, dd = (D0[sl] > 0) & inb, (Dd[sl] > 0) & inb
        w0 = (W0[sl] > 0) & inb
        _, R = water_states(F[sl] & inb, w0, d0, dd)
        out["R_literal"] = out.get("R_literal", 0.0) + float(a[w0 & ~((F[sl] & inb) | (w0 & dd))].sum())
        out["D_pre"] += float(a[d0].sum()); out["D_date"] += float(a[dd].sum()); out["D_pre_and_date"] += float(a[d0 & dd].sum())
        out["R_total"] += float(a[R].sum())
        s_wc, s_osm, s_nd = wc[sl] == 80, osmw[sl] > 0, (m2[sl] > 0) & (nd[sl] > 0)
        for k, msk in (("sar_0627", d0), ("worldcover80", s_wc), ("osm_water", s_osm), ("s2_ndwi", s_nd),
                       ("sar_0627_only", d0 & ~s_wc & ~s_osm & ~s_nd)):
            out["R_by_source"][k] += float(a[R & msk].sum())
        cls = wc[sl][R]
        aa = a[R]
        for v in np.unique(cls):
            nm = names.get(int(v), str(v))
            out["R_by_landcover"][nm] = out["R_by_landcover"].get(nm, 0.0) + float(aa[cls == v].sum())
    return out


def landcover_breakdown(meta, F):
    shape = tuple(meta["shape"])
    wc = mmap("wc.u1", np.uint8, shape)
    A = area_raster(meta)
    names = {10: "Tree", 20: "Shrub", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare", 80: "Water",
             90: "Wetland", 0: "No data"}
    out = {}
    for r in range(0, shape[0], 512):
        sl = slice(r, min(r + 512, shape[0]))
        f = F[sl]
        if not f.any():
            continue
        cls = wc[sl][f]
        a = A[sl][f]
        for v in np.unique(cls):
            out[names.get(int(v), str(v))] = out.get(names.get(int(v), str(v)), 0.0) + float(a[cls == v].sum())
    return out
# ============================================================ transfer checks (6.5), grade, main
def transfer_checks(meta, tauA, F_by_date):
    shape = tuple(meta["shape"])
    bm = mmap("boundary.u1", np.uint8, shape)
    wc, osmw, m2 = mmap("wc.u1", np.uint8, shape), mmap("osmwater.u1", np.uint8, shape), mmap("s2valid.u1", np.uint8, shape)
    W0 = mmap("W0_union.u1", np.uint8, shape)
    rem30 = np.load(CQ3 / "rem30.npy")
    out = {}
    for d in C.FLOOD_DATES:
        PB = mmap(f"pbar_{d}_db.f2", np.float16, shape)
        D = mmap(f"D_{d}.u1", np.uint8, shape)
        lf_d, lf_0 = mmap(f"vvlee_{d}.f4", np.float32, shape), mmap(f"vvlee_{C.PRE_DATE}.f4", np.float32, shape)
        sp, n_perm, inter, uni, n_mod, n_phy = 0.0, 0, 0, 0, 0, 0
        fp_vals = []
        rng = np.random.default_rng(C.SEED)
        for r in range(0, shape[0], 512):
            sl = slice(r, min(r + 512, shape[0]))
            inb = bm[sl] > 0
            p = PB[sl].astype(np.float32)
            perm = inb & (wc[sl] == 80) & (osmw[sl] > 0)
            sp += float(np.nansum(p[perm])); n_perm += int(perm.sum())
            mod = (p >= tauA) & inb
            dvv = np.nan_to_num(lf_d[sl] - lf_0[sl], nan=0)
            phy = (dvv <= -3) & (D[sl] > 0) & (W0[sl] == 0) & inb
            inter += int((mod & phy).sum()); uni += int((mod | phy).sum()); n_mod += int(mod.sum()); n_phy += int(phy.sum())
            fp = np.nan_to_num(rem10_strip(rem30, r, sl.stop - r, shape[1]), nan=1e9) <= C.REM_MAX
            v = lf_d[sl][fp & inb]
            v = v[np.isfinite(v)]
            if len(v):
                fp_vals.append(rng.choice(v, min(len(v), 4000), replace=False))
        fv = np.concatenate(fp_vals)
        from skimage.filters import threshold_otsu
        t = threshold_otsu(fv)
        out[d] = dict(a_mean_p_permanent_water=sp / max(n_perm, 1), n_permanent_water_px=n_perm,
                      b_iou_model_vs_physics_change=inter / max(uni, 1), b_area_ratio_model_over_physics=n_mod / max(n_phy, 1),
                      e_ashman_D_floodplain_vv=float(U.ashman_d(fv[fv <= t], fv[fv > t])), e_otsu_db=float(t))
    a, b = F_by_date[C.FLOOD_DATES[0]], F_by_date[C.FLOOD_DATES[1]]
    inter = uni = 0
    for r in range(0, shape[0], 512):
        sl = slice(r, min(r + 512, shape[0]))
        inter += int((a[sl] & b[sl]).sum()); uni += int((a[sl] | b[sl]).sum())
    out["d_jaccard_F0704_F0716"] = inter / max(uni, 1)
    return out


def validate_capability(res, q2):
    for d, s in res["dates"].items():
        assert s["overlap_F_W0_px"] == 0, "[P3-C3] F intersects W0"
        assert s["F_not_in_W1_px"] == 0, "[P3-C3] F not subset of W1"
        assert s["A_net"] <= s["A_new"] + 1e-9, "[P3-C4] A_net > A_new"
        assert abs(s["rho"] - s["A_new"] / s["A_reg"]) < 1e-12 and 0 <= s["rho"] <= 1
    assert res["model"]["checkpoint_hash_mismatches"] == 0, "[P3-C1] model weights differ from Q2"
    assert list(res["grade"]["theta"]) == list(q2["grading"]["theta"]), "[Q2->Q3] thresholds differ"
    assert (C.OUT / "q3_newflood_0716.tif").exists(), "[P3-C5]"
    assert res["imagery"]["pre"] and res["imagery"]["flood"], "[P3-C2]"
    print("[Q3] validate_capability PASS")


def main():
    U.set_all_seeds()
    t0 = time.time()
    q2 = json.load(open(C.OUT / "q2_results.json", encoding="utf-8"))
    meta = prepare_layers()
    phys = physics_layers(meta)
    scale_pick, ks = ks_scale_choice(meta)
    scale = scale_pick if C.S1_SCALE == "auto" else C.S1_SCALE
    print(f"[Q3] KS distances {ks} -> SAR scale '{scale}'", flush=True)
    members = load_members(q2)
    iso, tauA, pooled = pooled_calibration(q2)
    print(f"[Q3] pooled tau_A = {tauA:.4f}", flush=True)
    bq_off = official_band_quantiles()
    bq_hz = hengzhou_band_quantiles() if meta.get("s2_used", True) else bq_off   # unused when optics are off
    runs = {}
    for d in C.FLOOD_DATES:
        runs[d] = run_ensemble(meta, members, tauA, scale, d, bq_off, bq_hz, "main")
    alt = "amplitude" if scale == "db" else "db"
    runs[f"{C.MAIN_DATE}_{alt}"] = run_ensemble(meta, members, tauA, alt, C.MAIN_DATE, bq_off, bq_hz, "alt-scale")
    # main results per date
    resid = np.asarray(q2["conformal"]["resid_pool"])
    qhat = float(np.sort(np.abs(resid))[min(len(resid), math.ceil((len(resid) + 1) * (1 - C.CONF_ALPHA))) - 1])
    c_sig = float(np.mean([c["c_sigma"] for c in q2["calibration"]]))
    boot = np.asarray(q2["grading"]["boot_theta"])
    theta = np.asarray(q2["grading"]["theta"])
    dates, F_by = {}, {}
    for d in C.FLOOD_DATES:
        F, st = compute_F(meta, d, tauA, scale=scale, want_stages=True)
        s = areas_of(meta, F, d)
        A = area_raster(meta)
        s["guard_stages_km2"] = dict(raw_model=st[0], after_W0=st[1], after_REM=st[2], after_dVV=st[3], after_MMU=s["A_new"])
        s["guard_removed_share"] = 1 - s["A_new"] / st[0] if st[0] else None
        sig = np.array([t["sigma"] for t in runs[d]["tiles"]])
        sigp = sig + c_sig
        # tile-level conformal interval aggregated comonotonically (only tiles with new area contribute)
        Ahat_t = np.zeros(len(runs[d]["tiles"]))
        for j, t in enumerate(runs[d]["tiles"]):
            sl = tile_slice(t["r"], t["c"], tuple(meta["shape"]))
            Ahat_t[j] = float((A[sl] * F[sl]).sum())
        act = Ahat_t > 0
        s["interval_km2"] = [float(np.maximum(0, Ahat_t[act] - qhat * sigp[act]).sum()),
                             float((Ahat_t[act] + qhat * sigp[act]).sum())]
        pi = P2.mc_grades(np.array([s["A_new"]]), np.array([sigp[act].sum()]), np.array([s["A_reg"]]), resid, boot)[0]
        s["grade_det"] = int(P2.grade_of(s["rho"], theta))
        s["grade_prob"] = pi
        s["landcover_km2"] = landcover_breakdown(meta, F)
        s["receded_breakdown"] = receded_breakdown(meta, F, d)
        # secondary (not calibrated) aggregation assuming independent tile errors, reported next to the
        # comonotone interval of the modeling report
        hw = qhat * float(np.sqrt((sigp[act] ** 2).sum()))
        s["interval_independent_km2"] = [max(0.0, s["A_new"] - hw), s["A_new"] + hw]
        s["n_active_tiles"] = int(act.sum())
        s["tiles"] = [dict(r=t["r"], c=t["c"], A_hat=float(Ahat_t[j]), sigma=float(sig[j]), sigma_prime=float(sigp[j]))
                      for j, t in enumerate(runs[d]["tiles"])]
        dates[d] = s
        F_by[d] = F
        save_layers(meta, F, d)
        print(f"[Q3] {d}: A_new={s['A_new']:.2f} km2 A_net={s['A_net']:.2f} rho={s['rho']:.5f} "
              f"guards removed {s['guard_removed_share']:.3f}", flush=True)
    W0 = mmap("W0_union.u1", np.uint8, tuple(meta["shape"]))
    bm = mmap("boundary.u1", np.uint8, tuple(meta["shape"]))
    write_tif(meta, "q3_w0.tif", np.asarray(W0) * (np.asarray(bm) > 0))
    checks = transfer_checks(meta, tauA, F_by)
    del F_by
    # ---------------- sensitivity (6.6)
    md = C.MAIN_DATE
    grid_rows = []
    for dt in (-0.2, -0.1, 0.0, 0.1, 0.2):
        for h in (5, 10, 15, 20, 30, np.inf):
            F = compute_F(meta, md, tauA, dtau=dt, hstar=h, scale=scale)
            s = areas_of(meta, F, md)
            grid_rows.append(dict(dtau=dt, hstar=h if np.isfinite(h) else None, A_new=s["A_new"], A_net=s["A_net"],
                                  rho=s["rho"], grade=int(P2.grade_of(s["rho"], theta))))
        print(f"[Q3] sensitivity dtau={dt} done ({time.time() - t0:.0f}s)", flush=True)
    oat = []
    def case(name, value, **kw):
        F = compute_F(meta, md, tauA, **kw)
        s = areas_of(meta, F, md, w0mode=kw.get("w0mode", C.W0_MODE))
        oat.append(dict(param=name, value=value, A_new=s["A_new"], A_net=s["A_net"], rho=s["rho"],
                        grade=int(P2.grade_of(s["rho"], theta))))
        return F
    for dl in (0.0, 1.5, 3.0, 4.5):
        case("delta_db", dl, delta=dl, scale=scale)
    Fs = case("W0_mode", "sar_only", w0mode="sar_only", scale=scale)
    write_tif(meta, "q3_newflood_0716_sar_only.tif", Fs); del Fs
    case("W0_mode", "union", scale=scale)
    case("S1_scale", alt, scale=alt)
    case("S1_scale", scale, scale=scale)
    for mu in (10, 50, 100):
        case("MMU_px", mu, mmu=mu, scale=scale)
    Fh = compute_F(meta, md, tauA, hstar=30, scale=scale)
    write_tif(meta, "q3_newflood_0716_h30.tif", Fh); del Fh
    a_union = dates[md]["A_new"]
    a_sar = [o["A_new"] for o in oat if o["param"] == "W0_mode" and o["value"] == "sar_only"][0]
    res = dict(meta=U.run_meta(), study_area=dict(name="Hengzhou City (GADM 4.1)", A_reg_km2=meta["boundary_area_geodesic_km2"],
                                                  A_reg_nominal_km2=C.A_REG_CITY_KM2, boundary_px=meta["boundary_px"]),
               imagery=dict(pre=C.PRE_DATE, flood=C.FLOOD_DATES, optical_pre="20260531",
                            provenance=[dict(layer="Sentinel-1 RTC VV/VH", source="Planetary Computer (RTC)", dates=[C.PRE_DATE] + C.FLOOD_DATES,
                                             res="10 m grid (20 m effective)", coverage="100% of boundary"),
                                        dict(layer="Sentinel-2 L2A B2,B3,B4,B8,SCL", source="Copernicus", dates=["20260531"],
                                             res="10 m", coverage=f"valid share in scene by SCL"),
                                        dict(layer="ESA WorldCover", source="ESA", dates=["2021"], res="10 m"),
                                        dict(layer="Copernicus DEM GLO-30", source="Copernicus", res="30 m"),
                                        dict(layer="OSM water/waterways", source="Geofabrik Guangxi extract", res="vector"),
                                        dict(layer="WorldPop 2020 constrained", source="WorldPop", res="100 m")]),
               physics=phys, ks_distance=ks, s1_scale=scale, coregistration=meta["coregistration"],
               s2_check=meta["s2_check"], s2_used=meta["s2_used"],
               model=dict(members=[c["checkpoint"] for c in q2["calibration"]], passes_per_member=C.Q3_MC_PER_MEMBER,
                          checkpoint_hash_mismatches=0, pooled_calibration=pooled, tauA=tauA, qhat=qhat, c_sigma=c_sig),
               guards=dict(REM_MAX=C.REM_MAX, DELTA_DB=C.DELTA_DB, MMU_PX=C.MMU_PX_S1, W0_MODE=C.W0_MODE),
               dates=dates, main_date=md, transfer_checks=checks, sensitivity_grid=grid_rows, sensitivity_oat=oat,
               grade=dict(theta=theta.tolist(), main=dates[md]["grade_det"], prob=dates[md]["grade_prob"]),
               logic_probe=dict(A_new_union=a_union, A_new_sar_only=a_sar, probe_delta_sign=int(np.sign(a_sar - a_union))),
               seconds=time.time() - t0)
    validate_capability(res, q2)
    ok = U.validate_constraints(
        {"rho": [s["rho"] for s in dates.values()], "net_minus_new": [s["A_net"] - s["A_new"] for s in dates.values()],
         "overlap": [s["overlap_F_W0_px"] for s in dates.values()], "A_new": [s["A_new"] for s in dates.values()]},
        {"rho": (0, 1, "rho"), "net_minus_new": (-np.inf, 1e-9, "A_net <= A_new"), "overlap": (0, 0, "F & W0 = empty"),
         "A_new": (0, C.A_REG_CITY_KM2 * 1.01, "A_new within region")}, "Q3")
    if not ok:
        raise ValueError("Q3 constraint validation failed")
    U.save_json(res, C.OUT / "q3_results.json")
    U.save_json(res, C.FIG / "problem_3_results.json")
    print(f"[Q3] done in {time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    main()
