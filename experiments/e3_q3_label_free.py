"""E3 — label-free audits for Q3 (no JRC layer, no human interpreter in this environment).

A. Provenance of the pre-flood water mask W0: how much comes from ESA WorldCover water (class 80),
   how much from the SAR low-backscatter branch ("SAR-only"), how much OSM adds, and whether the
   optical Sentinel-2 water classes confirm the SAR-only part.
B. Cross-polarisation VH change detector vs the flood mask F, threshold sweep -2..-8 dB (VH is unused
   by the VV model, so it is an independent channel; the sweep makes the threshold choice transparent).
C. Terrain position of F relative to the boundary (Copernicus 30 m DEM).
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
import config as C

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)
S2 = C.PUB / "Sentinel2" / "Sentinel2_20260531_B432_NIR_SCL.tif"
VH0 = C.PUB / "Sentinel1" / "Sentinel1_20260627_VV_VH_dB.tif"
VH1 = C.PUB / "Sentinel1" / "Sentinel1_20260716_VV_VH_dB.tif"
DEM = C.PUB / "DEM_clip" / "DEM_Hengzhou_30m_mosaic.tif"
WC = C.PUB / "LandCover" / "WorldCover2021_Hengzhou_10m.tif"
W0 = C.OUT / "q3_w0.tif"
F = C.OUT / "q3_newflood_0716.tif"
BND = C.PUB / "Boundary" / "Hengzhou_boundary.geojson"


def main():
    with rasterio.open(W0) as d:
        shape, T, crs = (d.height, d.width), d.transform, d.crs
    bnd = gpd.read_file(BND).to_crs(crs)
    inside = rasterize([(g, 1) for g in bnd.geometry], out_shape=shape, transform=T, fill=0, dtype="uint8") > 0

    wc = np.zeros(shape, np.uint8)
    with rasterio.open(WC) as c:
        reproject(rasterio.band(c, 1), wc, dst_transform=T, dst_crs=crs, dst_nodata=0, resampling=Resampling.nearest)

    # A. provenance (counted only inside boundary)
    cnt = dict(w0=0, wc80=0, sar_only=0, s2_class6=0, s2_mndwi=0, sar_only_s2c6=0, sar_only_mndwi=0, valid=0)
    with rasterio.open(W0) as w, rasterio.open(S2) as s:
        for r in range(0, shape[0], 512):
            h = min(512, shape[0] - r)
            win = Window(0, r, shape[1], h)
            wm = w.read(1, window=win) > 0
            m = inside[r:r + h]
            w80 = wc[r:r + h] == 80
            cls = s.read(5, window=win)
            g = s.read(2, window=win).astype(np.float32)
            n = s.read(4, window=win).astype(np.float32)
            val = m & ~np.isin(cls, [0, 1, 3, 8, 9, 10]) & ((g + n) > 0)
            mnd = val & ((g - n) / np.maximum(g + n, 1e-6) > 0.3)
            c6 = m & (cls == 6)
            sar_only = wm & ~w80 & m
            cnt["w0"] += int((wm & m).sum()); cnt["wc80"] += int((w80 & m).sum())
            cnt["sar_only"] += int(sar_only.sum()); cnt["s2_class6"] += int(c6.sum()); cnt["s2_mndwi"] += int(mnd.sum())
            cnt["sar_only_s2c6"] += int((sar_only & c6).sum()); cnt["sar_only_mndwi"] += int((sar_only & mnd).sum())
            cnt["valid"] += int(val.sum())
    km2 = lambda v: v * 1e-4
    prov = dict(w0_km2=km2(cnt["w0"]), wc80_km2=km2(cnt["wc80"]), sar_only_km2=km2(cnt["sar_only"]),
                sar_only_share=cnt["sar_only"] / max(cnt["w0"], 1),
                s2_class6_km2=km2(cnt["s2_class6"]), s2_mndwi03_km2=km2(cnt["s2_mndwi"]),
                sar_only_confirmed_by_s2class6=km2(cnt["sar_only_s2c6"]),
                sar_only_confirmed_by_mndwi03=km2(cnt["sar_only_mndwi"]),
                s2_valid_km2=km2(cnt["valid"]),
                note="W0 = WorldCover water (class 80) union SAR low-backscatter union OSM water; no S2 NDWI "
                     "component was used in the run (q3_results.s2_used=false)")

    # B. VH threshold sweep
    thresholds = [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
    rows = {t: dict(tp=0, fp=0, fn=0) for t in thresholds}
    f_area = 0
    with rasterio.open(VH0) as a, rasterio.open(VH1) as b, rasterio.open(F) as fm, rasterio.open(W0) as wm_:
        for r in range(0, shape[0], 512):
            h = min(512, shape[0] - r)
            win = Window(0, r, shape[1], h)
            dv = b.read(2, window=win).astype(np.float32) - a.read(2, window=win).astype(np.float32)
            f = fm.read(1, window=win) > 0
            w0 = wm_.read(1, window=win) > 0
            base = inside[r:r + h] & ~w0 & np.isfinite(dv) & (dv > -50) & (dv < 50)
            f_area += int((f & base).sum())
            for t in thresholds:
                det = base & (dv <= t)
                rows[t]["tp"] += int((det & f).sum()); rows[t]["fp"] += int((det & ~f).sum()); rows[t]["fn"] += int((~det & f & base).sum())
    sweep = []
    for t in thresholds:
        c = rows[t]
        p = c["tp"] / max(c["tp"] + c["fp"], 1); rc = c["tp"] / max(c["tp"] + c["fn"], 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        sweep.append(dict(db=t, precision=p, recall=rc, iou=c["tp"] / max(c["tp"] + c["fp"] + c["fn"], 1),
                          f1=f1, detected_km2=km2(c["tp"] + c["fp"])))
    vh = dict(sweep=sweep, f_km2=km2(f_area),
              note="detector = VH(07-16) - VH(06-27) <= threshold, outside W0 and inside boundary")

    # C. terrain
    z = np.full(shape, np.nan, np.float32)
    with rasterio.open(DEM) as dm:
        reproject(rasterio.band(dm, 1), z, dst_transform=T, dst_crs=crs, resampling=Resampling.bilinear,
                  dst_nodata=np.nan)
    z[z < -100] = np.nan
    with rasterio.open(F) as fm:
        f = fm.read(1) > 0
    zf, zi = z[f & inside], z[inside]
    elev = dict(mean_F=float(np.nanmean(zf)), mean_in=float(np.nanmean(zi)),
                median_F=float(np.nanmedian(zf)), median_in=float(np.nanmedian(zi)),
                pct_rank_F_median=float((zi < np.nanmedian(zf)).mean()),
                p90_F=float(np.nanpercentile(zf, 90)), p90_in=float(np.nanpercentile(zi, 90)),
                valid_px=int(np.isfinite(zf).sum()))

    out = dict(meta=dict(s2=S2.name, vh=[VH0.name, VH1.name], dem=DEM.name), w0_provenance=prov,
               vh_change_sweep=vh, elevation=elev)
    json.dump(out, open(os.path.join(RES, "e3_q3_label_free.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
