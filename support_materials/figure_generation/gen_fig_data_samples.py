"""Data samples used in the paper: the official triplet and the main Hengzhou layers."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, LogNorm, Normalize, BoundaryNorm
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
import geopandas as gpd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUB = os.path.join(ROOT, "user_data", "a题数据汇总", "公开数据")
OFF = os.path.join(ROOT, "user_data", "a题数据汇总", "Problem A Dataset 1", "Dataset")
OFF2 = os.path.join(ROOT, "user_data", "a题数据汇总", "Problem A Dataset 2", "Dataset")
FIG = os.path.join(ROOT, "figures")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.linewidth": 0.6,
})

BLUE = "#0F4D92"
RED = "#B64342"
GRAY = "#555555"


def read_band(path, band=1):
    with rasterio.open(path) as d:
        return d.read(band)


def read_all(path):
    with rasterio.open(path) as d:
        return d.read()


def stretch(a, lo=2, hi=98):
    a = a.astype(float)
    p1, p2 = np.percentile(a[np.isfinite(a)], [lo, hi])
    return np.clip((a - p1) / max(p2 - p1, 1e-9), 0, 1)


def panel_label(ax, letter):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))


# ---------------------------------------------------------------- official triplet
opt = read_all(os.path.join(OFF2, "pre-disaster optical images", "before_s2_1001.TIF"))
sar = read_band(os.path.join(OFF, "post-disaster SAR images", "after_gf3_hh_1001.TIF"))
lab = read_band(os.path.join(OFF, "label information", "label_1001.tif"))

rgb = np.dstack([stretch(opt[2]), stretch(opt[1]), stretch(opt[0])])

# ---------------------------------------------------------------- Hengzhou common grid
bb = gpd.read_file(os.path.join(PUB, "Boundary", "Hengzhou_boundary.geojson")).to_crs(32649)
geom = bb.geometry.iloc[0]
xmin, ymin, xmax, ymax = bb.total_bounds
res = 150.0
W = int((xmax - xmin) / res)
H = int((ymax - ymin) / res)
transform = rasterio.transform.from_origin(xmin, ymax, res, res)
crs = "EPSG:32649"

def warp_to_grid(path, band=1, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        arr = np.full((H, W), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, band),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    arr[geometry_mask([geom], out_shape=(H, W), transform=transform, invert=False)] = np.nan
    return arr

s1 = warp_to_grid(os.path.join(PUB, "Sentinel1", "Sentinel1_20260716_VV_VH_dB.tif"), 1)
wc = warp_to_grid(os.path.join(PUB, "LandCover", "WorldCover2021_Hengzhou_10m.tif"), 1,
                  resampling=Resampling.nearest)
pop = warp_to_grid(os.path.join(PUB, "WorldPop", "WorldPop_Hengzhou_2020_100m.tif"), 1)
pop[pop < 0] = np.nan

roads = gpd.read_file(os.path.join(PUB, "OSM", "roads.geojson")).to_crs(32649)
roads = roads.clip(bb)

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(7.2, 4.9))
fig.text(0.015, 0.985, "Official data (Attachment 1)", fontsize=8.5, fontweight="bold", va="top")
fig.text(0.015, 0.505, "Self-collected public layers (Hengzhou)", fontsize=8.5, fontweight="bold", va="top")

gs = fig.add_gridspec(2, 3, left=0.015, right=0.985, bottom=0.04, top=0.915,
                      wspace=0.055, hspace=0.16)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(rgb, aspect="auto")
ax.set_title("Pre-flood optical RGB", fontsize=8, pad=3)
panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
ax.imshow(sar, cmap="gray", vmin=0, vmax=255, aspect="auto")
ax.set_title("Post-flood SAR HH", fontsize=8, pad=3)
panel_label(ax, "b")

ax = fig.add_subplot(gs[0, 2])
ax.imshow(lab, cmap=ListedColormap(["#F4F4F4", RED]), vmin=0, vmax=1, aspect="auto")
ax.set_title("New-inundation label", fontsize=8, pad=3)
panel_label(ax, "c")

ext = [xmin, xmax, ymin, ymax]

ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(s1, extent=ext, cmap="gray", vmin=-20, vmax=5, origin="upper")
bb.boundary.plot(ax=ax, color="black", linewidth=0.6)
ax.set_title("Sentinel-1 VV, 16 Jul 2026", fontsize=8, pad=3)
panel_label(ax, "d")

ax = fig.add_subplot(gs[1, 1])
wc_colors = ["#F4F4F4", "#006400", "#8B8B00", "#7CFC00", "#FFD700", "#B22222",
             "#D2B48C", "#808080", "#0000FF"]
wc_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
wc_cmap = ListedColormap(wc_colors)
wc_norm = BoundaryNorm(wc_bins, ncolors=len(wc_colors))
im = ax.imshow(wc, extent=ext, cmap=wc_cmap, norm=wc_norm, origin="upper",
               interpolation="nearest")
bb.boundary.plot(ax=ax, color="black", linewidth=0.6)
ax.set_title("ESA WorldCover 2021", fontsize=8, pad=3)
panel_label(ax, "e")

ax = fig.add_subplot(gs[1, 2])
im = ax.imshow(pop, extent=ext, cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=100), origin="upper")
roads.plot(ax=ax, color="black", linewidth=0.25, alpha=0.75)
bb.boundary.plot(ax=ax, color="black", linewidth=0.6)
ax.set_title("WorldPop 2020 + OSM roads", fontsize=8, pad=3)
panel_label(ax, "f")

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)

# scale bar for the Hengzhou panels (10 km)
for ax in fig.axes[3:]:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.plot([x1 - 12000, x1 - 2000], [y0 + 1800, y0 + 1800], color="white", lw=3.0,
            solid_capstyle="butt")
    ax.text(x1 - 7000, y0 + 2800, "10 km", color="white", ha="center", va="bottom",
            fontsize=6.4, path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

fig.savefig(os.path.join(FIG, "fig_data_samples.pdf"), dpi=300, bbox_inches="tight")
print("wrote", os.path.join(FIG, "fig_data_samples.pdf"))
