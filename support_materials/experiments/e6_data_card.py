"""E6 — input data card: CRS, native resolution, grid and how each layer enters the model."""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code"))

import rasterio
import config as C

RES = os.path.join(_HERE, "results")
os.makedirs(RES, exist_ok=True)

LAYERS = [
    ("official label tiles", C.OFFICIAL_LABEL_DIR / "label_1001.tif", "Q1-Q2 target (new inundation)"),
    ("post-disaster SAR", C.OFFICIAL_SAR_DIR / "after_gf3_hh_1001.TIF", "Q1-Q2 predictor (HH, rank-normalised)"),
    ("pre-disaster optical", C.OFFICIAL_OPT_DIR / "before_s2_1001.TIF", "Q1-Q2 predictor (4 bands)"),
    ("Sentinel-1 VV/VH", C.PUB / "Sentinel1" / "Sentinel1_20260716_VV_VH_dB.tif", "Q3 VV branch; VH audit"),
    ("Sentinel-2", C.PUB / "Sentinel2" / "Sentinel2_20260531_B432_NIR_SCL.tif", "Q3 W0 optical (not used in run)"),
    ("WorldPop 100 m", C.PUB / "WorldPop" / "WorldPop_Hengzhou_2020_100m.tif", "Q4 E1 area-weighted overlay"),
    ("WorldCover 10 m", C.PUB / "LandCover" / "WorldCover2021_Hengzhou_10m.tif", "Q3 W0 water class, Q4 E2/E3, Q1 strata"),
    ("DEM 30 m", C.PUB / "DEM_clip" / "DEM_Hengzhou_30m_mosaic.tif", "Q3 REM (relative elevation)"),
    ("flood mask 07-16", C.OUT / "q3_newflood_0716.tif", "Q3 output -> Q4/Q5"),
    ("W0 water union", C.OUT / "q3_w0.tif", "Q3 physical guard"),
]


def info(p):
    if not p.exists():
        return dict(path=str(p), exists=False)
    with rasterio.open(p) as d:
        return dict(path=str(p.relative_to(C.ROOT)), exists=True, crs=str(d.crs), width=d.width, height=d.height,
                    count=d.count, dtype=str(d.dtypes[0]), nodata=d.nodata,
                    res=[abs(d.transform.a), abs(d.transform.e)], bounds=[float(v) for v in d.bounds],
                    descriptions=[x for x in (d.descriptions or []) if x])


def main():
    rows = []
    for name, p, role in LAYERS:
        r = info(p)
        r.update(name=name, role=role)
        rows.append(r)
    # subset of label tiles with anisotropic pixels (9 tiles reported by the paper)
    rows.append(dict(name="label anisotropy", role="9 tiles are resampled to the SAR grid by nearest neighbour",
                     note="see code/common.py load_tile; counted in Q1 preprocessing"))
    out = dict(layers=rows)
    json.dump(out, open(os.path.join(RES, "e6_data_card.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['name']:24s} {r.get('crs',''):14s} {str(r.get('res','')):20s} {r['role']}")
        if "path" in r:
            print("   ", r["path"])


if __name__ == "__main__":
    main()

