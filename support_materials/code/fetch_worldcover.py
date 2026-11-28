"""A4: ESA WorldCover 2021 v200 land cover for the 1000 official tiles.

Windowed HTTP reads of the public COGs (tiles N27E111, N27E114), nearest-neighbour sampled at the
centre of every SAR pixel. Result cached in output/q1_worldcover_tiles.npz (uint8, 1000x256x256).
If the download fails, problem1 falls back to the optical index rules (recorded in results).
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import rasterio
from rasterio.windows import Window
import config as C
from common import official_index

URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
OUTF = C.OUT / "q1_worldcover_tiles.npz"


def main():
    if OUTF.exists():
        print("[worldcover] cache exists:", OUTF)
        return True
    ids, L, S, O = official_index()
    grids = {}
    for t in ids:
        with rasterio.open(S[t]) as d:
            grids[t] = d.transform
    order = sorted(ids, key=lambda t: (grids[t].c, -grids[t].f))
    wc = np.zeros((len(ids), C.TILE, C.TILE), np.uint8)
    pos = {t: i for i, t in enumerate(ids)}
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
               GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=3,
               GDAL_CACHEMAX=256, VSI_CACHE=True, VSI_CACHE_SIZE=100_000_000)
    t0 = time.time()
    with rasterio.Env(**env):
        src = {k: rasterio.open(URL.format(k)) for k in ("N27E111", "N27E114")}
        for n, t in enumerate(order):
            T = grids[t]
            d = src["N27E111" if T.c + 128 * T.a < 114 else "N27E114"]
            WT = d.transform
            xs = T.c + (np.arange(C.TILE) + 0.5) * T.a
            ys = T.f + (np.arange(C.TILE) + 0.5) * T.e
            cols = np.floor((xs - WT.c) / WT.a).astype(int)
            rows = np.floor((ys - WT.f) / WT.e).astype(int)
            c0, r0 = cols.min(), rows.min()
            win = Window(c0, r0, cols.max() - c0 + 1, rows.max() - r0 + 1)
            a = d.read(1, window=win)
            wc[pos[t]] = a[np.ix_(rows - r0, cols - c0)]
            if n % 50 == 0:
                print(f"[worldcover] {n}/{len(ids)} tiles, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(OUTF, ids=np.array(ids), wc=wc)
    print("[worldcover] saved", OUTF, "classes", np.unique(wc).tolist(), f"{time.time() - t0:.0f}s")
    return True


if __name__ == "__main__":
    main()
