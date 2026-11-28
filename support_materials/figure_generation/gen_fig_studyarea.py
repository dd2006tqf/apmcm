"""Study-area map panels for the two Figure 1 options.

Uses the real Hengzhou boundary, DEM, OSM river/roads, and the 16 July
new-inundation raster. All coordinates are projected to UTM 49N.
"""

from __future__ import annotations

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "user_data" / "a题数据汇总" / "公开数据"
OUT = ROOT / "figures" / "scenario_options"
OUT.mkdir(exist_ok=True)

CRS = "EPSG:32649"  # UTM 49N
RES = 50.0


def project_vector(path: Path, clip=None):
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g.to_crs(CRS)
    if clip is not None:
        g = gpd.clip(g, clip)
    return g


def target_grid(boundary):
    xmin, ymin, xmax, ymax = boundary.total_bounds
    pad = 1800
    xmin -= pad; ymin -= pad; xmax += pad; ymax += pad
    w = int(math.ceil((xmax - xmin) / RES))
    h = int(math.ceil((ymax - ymin) / RES))
    transform = rasterio.transform.from_origin(xmin, ymax, RES, RES)
    return (xmin, ymin, xmax, ymax), (h, w), transform


def warp_to_grid(src_path, shape, transform, resampling=Resampling.bilinear):
    with rasterio.open(src_path) as src:
        dst = np.full(shape, np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=CRS,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        return dst


def make_hillshade(dem):
    # simple Horn-style hillshade; NaN outside handled by fill
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    x, y = np.gradient(filled, RES)
    slope = np.pi / 2 - np.arctan(np.hypot(x, y))
    aspect = np.arctan2(-x, y)
    az = np.deg2rad(315.0)
    alt = np.deg2rad(45.0)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    hs = (hs + 1) / 2
    return np.clip(hs, 0, 1)


def build_map_png():
    boundary = project_vector(DATA / "Boundary" / "Hengzhou_boundary.geojson")
    extent, shape, transform = target_grid(boundary)
    dem = warp_to_grid(DATA / "DEM_clip" / "DEM_Hengzhou_30m_mosaic.tif", shape, transform)
    hs = make_hillshade(dem)

    # inside-boundary mask on the target grid
    mask = geometry_mask(
        [geom for geom in boundary.geometry],
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=True,
    )
    hs = np.where(mask, hs, np.nan)
    dem = np.where(mask, dem, np.nan)

    flood = warp_to_grid(ROOT / "output" / "q3_newflood_0716.tif", shape, transform, Resampling.nearest)
    flood = np.where((flood > 0) & mask, 1.0, np.nan)
    # aggregate to 100 m cells so the 26.9 km2 of new water is visible at page size
    if flood.shape[0] % 2:
        flood = flood[:-1, :]
    if flood.shape[1] % 2:
        flood = flood[:, :-1]
    flood100 = flood.reshape(flood.shape[0] // 2, 2, flood.shape[1] // 2, 2).mean(axis=(1, 3))

    rivers = project_vector(DATA / "OSM" / "waterways_lines.geojson", boundary)
    rivers = rivers[rivers["waterway"].isin(["river", "stream", "canal"])]
    water = project_vector(DATA / "OSM" / "water_surface_merged.geojson", boundary)
    roads = project_vector(DATA / "OSM" / "roads.geojson", boundary)
    roads = roads[roads["highway"].isin(["motorway", "trunk", "primary", "secondary"])]

    # render
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    land = LinearSegmentedColormap.from_list(
        "land", ["#EDE8D8", "#D8D6B8", "#AFC29B", "#7E9C76"]
    )
    fig_w, fig_h = 100 / 25.4, 86 / 25.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.88])
    ax.set_facecolor("white")

    ax.imshow(
        dem,
        extent=(extent[0], extent[2], extent[1], extent[3]),
        cmap=land,
        vmin=np.nanpercentile(dem, 2),
        vmax=np.nanpercentile(dem, 98),
        alpha=0.66,
        interpolation="bilinear",
        zorder=1,
    )
    ax.imshow(
        hs,
        extent=(extent[0], extent[2], extent[1], extent[3]),
        cmap="Greys",
        alpha=0.22,
        interpolation="bilinear",
        zorder=2,
    )
    water.plot(ax=ax, facecolor="#9FC4CE", edgecolor="none", alpha=0.65, zorder=3)
    rivers.plot(ax=ax, color="#356F8E", linewidth=0.34, alpha=0.65, zorder=4)
    yu_pre = rivers[rivers["name"].fillna("").str.contains("郁江")]
    yu_pre.plot(ax=ax, color="#2F6684", linewidth=0.9, zorder=4.2)
    roads.plot(ax=ax, color="#7A7A72", linewidth=0.34, alpha=0.62, zorder=5)
    ax.imshow(
        flood100,
        extent=(extent[0], extent[2], extent[1], extent[3]),
        cmap=LinearSegmentedColormap.from_list("flood", ["#C65B45", "#C65B45"]),
        alpha=0.88,
        interpolation="nearest",
        zorder=6,
    )
    boundary.boundary.plot(ax=ax, color="#3C3C36", linewidth=0.9, zorder=7)

    # town labels
    places = project_vector(DATA / "OSM" / "places.geojson", boundary)
    town_names = {
        "横州市": "Hengzhou", "石塘镇": "Shitang", "马岭镇": "Maling",
        "峦城镇": "Luancheng", "云表镇": "Yunbiao",
    }
    towns = places[places["name"].isin(town_names)].copy()
    for _, row in towns.iterrows():
        pt = row.geometry.representative_point()
        ax.plot(pt.x, pt.y, marker="o", ms=2.2, mfc="#24313A", mec="white", mew=0.35, zorder=12)
        ax.text(
            pt.x + 900, pt.y + 500, town_names[row["name"]],
            fontsize=4.8, color="#24313A", ha="left", va="bottom", zorder=12,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.82),
        )

    # Yu River label
    yu = rivers[rivers["name"].fillna("").str.contains("郁江")]
    if len(yu):
        line = max(yu.geometry, key=lambda g: g.length)
        pt = line.interpolate(0.55, normalized=True)
        ax.annotate(
            "Yu River", xy=(pt.x, pt.y), xytext=(pt.x + 6000, pt.y + 5000),
            fontsize=5.0, color="#356F8E", ha="left", va="bottom", zorder=12,
            arrowprops=dict(arrowstyle="-|>", color="#356F8E", lw=0.55, mutation_scale=4),
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.84),
        )

    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # labels
    ax.text(
        0.04, 0.96, "Hengzhou, Guangxi",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.4, fontweight="bold", color="#24313A",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9),
    )
    ax.text(
        0.04, 0.905, "Yu River floodplain · 3,440 km$^2$",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=5.6, color="#4A5558",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82),
    )

    # scale bar 10 km, bottom-left
    x0 = extent[0] + 0.04 * (extent[2] - extent[0])
    y0 = extent[1] + 0.055 * (extent[3] - extent[1])
    ax.plot([x0, x0 + 10000], [y0, y0], color="#24313A", lw=1.4, solid_capstyle="butt", zorder=10)
    ax.text(x0 + 5000, y0 + 900, "10 km", ha="center", va="bottom", fontsize=5.2, color="#24313A")

    # north arrow, top right
    xa = extent[0] + 0.92 * (extent[2] - extent[0])
    ya = extent[1] + 0.90 * (extent[3] - extent[1])
    ax.annotate(
        "", xy=(xa, ya + 4500), xytext=(xa, ya - 1500),
        arrowprops=dict(arrowstyle="-|>", color="#24313A", lw=0.8, mutation_scale=7),
        zorder=10,
    )
    ax.text(xa, ya + 5400, "N", ha="center", va="bottom", fontsize=5.2, color="#24313A")

    # legend inside lower right
    handles = [
        Patch(facecolor="#C65B45", edgecolor="none", label="New inundation, 16 Jul"),
        Line2D([0], [0], color="#356F8E", lw=0.8, label="River"),
        Line2D([0], [0], color="#7A7A72", lw=0.8, label="Main road"),
    ]
    leg = ax.legend(
        handles=handles, loc="lower right", frameon=True, fontsize=4.8,
        handlelength=1.6, borderpad=0.35, labelspacing=0.3,
        framealpha=0.9, edgecolor="none",
    )
    leg.get_frame().set_facecolor("white")

    fig.savefig(OUT / "studyarea_map_preview.png", dpi=400, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "studyarea_map_preview.png")


if __name__ == "__main__":
    build_map_png()
