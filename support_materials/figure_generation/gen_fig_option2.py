"""Option 2 Figure 1: study area + five-problem chain.

Left: real Hengzhou boundary, DEM, OSM river/roads, 16 July new water.
Right: five-stage chain distilled from Sections 4--8 of the paper.
Vector typography, flat palette, no raster labels.
"""

from __future__ import annotations

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "user_data" / "a题数据汇总" / "公开数据"
OUT = ROOT / "figures" / "scenario_options"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 5.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

CRS = "EPSG:32649"
RES = 50.0
INK = "#24313A"
MUTED = "#5A6266"
AMBER = "#A46A2E"
BLUE = "#356F8E"
RED = "#B65441"
PAPER = "#F7F5EE"
BORDER = "#D8D2C1"


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
    pad = 1600
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


def hillshade(dem):
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    x, y = np.gradient(filled, RES)
    slope = np.pi / 2 - np.arctan(np.hypot(x, y))
    aspect = np.arctan2(-x, y)
    az = np.deg2rad(315.0)
    alt = np.deg2rad(45.0)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((hs + 1) / 2, 0, 1)


def load_layers():
    boundary = project_vector(DATA / "Boundary" / "Hengzhou_boundary.geojson")
    extent, shape, transform = target_grid(boundary)
    dem = warp_to_grid(DATA / "DEM_clip" / "DEM_Hengzhou_30m_mosaic.tif", shape, transform)
    hs = hillshade(dem)
    mask = geometry_mask(
        [geom for geom in boundary.geometry],
        out_shape=shape, transform=transform,
        invert=True, all_touched=True,
    )
    hs = np.where(mask, hs, np.nan)
    dem = np.where(mask, dem, np.nan)
    flood = warp_to_grid(ROOT / "output" / "q3_newflood_0716.tif", shape, transform, Resampling.nearest)
    flood = np.where((flood > 0) & mask, 1.0, np.nan)
    # 100 m aggregate: visible at print scale without inventing area
    if flood.shape[0] % 2: flood = flood[:-1, :]
    if flood.shape[1] % 2: flood = flood[:, :-1]
    flood100 = flood.reshape(flood.shape[0] // 2, 2, flood.shape[1] // 2, 2).mean(axis=(1, 3))
    # vector footprint of the 50 m flood mask; plotted as a thin red band so
    # that the 26.9 km2 area is legible at 82 mm map width.
    fvals = np.where(np.isfinite(flood), flood, 0).astype("uint8")
    fpolys = [shp_shape(geom) for geom, val in shapes(fvals, mask=np.isfinite(flood), transform=transform) if val == 1]
    flood_poly = unary_union(fpolys).simplify(180.0) if fpolys else None

    rivers = project_vector(DATA / "OSM" / "waterways_lines.geojson", boundary)
    rivers = rivers[rivers["waterway"].isin(["river", "stream", "canal"])]
    water = project_vector(DATA / "OSM" / "water_surface_merged.geojson", boundary)
    roads = project_vector(DATA / "OSM" / "roads.geojson", boundary)
    roads = roads[roads["highway"].isin(["motorway", "trunk", "primary", "secondary"])]
    places = project_vector(DATA / "OSM" / "places.geojson", boundary)
    return dict(
        boundary=boundary, extent=extent, dem=dem, hs=hs, flood=flood100, flood_poly=flood_poly,
        rivers=rivers, water=water, roads=roads, places=places,
    )


def draw_map(ax, L):
    extent = L["extent"]
    dem, hs, flood = L["dem"], L["hs"], L["flood"]
    flood_poly = L["flood_poly"]
    land = LinearSegmentedColormap.from_list("land", ["#EFEAD9", "#DAD8BC", "#B1C49C", "#7E9C76"])
    ax.set_facecolor("white")
    ax.imshow(
        dem, extent=(extent[0], extent[2], extent[1], extent[3]),
        cmap=land, vmin=np.nanpercentile(dem, 2), vmax=np.nanpercentile(dem, 98),
        alpha=0.62, interpolation="bilinear", zorder=1,
    )
    ax.imshow(
        hs, extent=(extent[0], extent[2], extent[1], extent[3]),
        cmap="Greys", alpha=0.20, interpolation="bilinear", zorder=2,
    )
    L["water"].plot(ax=ax, facecolor="#A8C8D0", edgecolor="none", alpha=0.55, zorder=3)
    L["rivers"].plot(ax=ax, color=BLUE, linewidth=0.30, alpha=0.60, zorder=4)
    yu = L["rivers"][L["rivers"]["name"].fillna("").str.contains("郁江")]
    yu.plot(ax=ax, color="#2C627D", linewidth=0.85, zorder=4.2)
    L["roads"].plot(ax=ax, color="#797971", linewidth=0.30, alpha=0.55, zorder=5)
    # The SAR-derived 16 Jul layer is deliberately not drawn here: at 26.9 km2
    # it is a river-corridor signal that reads as speckle at 82 mm.  The flood
    # extent is shown in the dedicated Q3 map; this panel is the study area.
    L["boundary"].boundary.plot(ax=ax, color="#3B3B35", linewidth=0.8, zorder=7)

    # town labels
    town_names = {
        "横州市": "Hengzhou", "石塘镇": "Shitang", "马岭镇": "Maling",
        "峦城镇": "Luancheng", "云表镇": "Yunbiao",
    }
    towns = L["places"][L["places"]["name"].isin(town_names)].copy()
    for _, row in towns.iterrows():
        pt = row.geometry.representative_point()
        ax.plot(pt.x, pt.y, marker="o", ms=1.9, mfc=INK, mec="white", mew=0.3, zorder=12)
        ax.text(
            pt.x + 800, pt.y + 450, town_names[row["name"]],
            fontsize=4.8, color=INK, ha="left", va="bottom", zorder=12,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.80),
        )

    # river label
    if len(yu):
        line = max(yu.geometry, key=lambda g: g.length)
        pt = line.interpolate(0.55, normalized=True)
        ax.annotate(
            "Yu River", xy=(pt.x, pt.y), xytext=(pt.x + 5200, pt.y + 4200),
            fontsize=4.9, color=BLUE, ha="left", va="bottom", zorder=12,
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.5, mutation_scale=4),
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.82),
        )

    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # title block
    ax.text(
        0.035, 0.965, "Hengzhou, Guangxi",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.0, fontweight="bold", color=INK,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.90),
    )
    ax.text(
        0.035, 0.910, "Yu River floodplain · 3,440 km$^2$",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=5.2, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82),
    )

    # scale bar, white-backed
    x0 = extent[0] + 0.035 * (extent[2] - extent[0])
    y0 = extent[1] + 0.045 * (extent[3] - extent[1])
    ax.add_patch(Rectangle((x0 - 700, y0 - 2400), 11400, 6500,
                           facecolor="white", edgecolor="none", alpha=0.88, zorder=9))
    ax.plot([x0, x0 + 10000], [y0, y0], color=INK, lw=1.2, solid_capstyle="butt", zorder=10)
    ax.text(
        x0 + 5000, y0 + 700, "10 km", ha="center", va="bottom", fontsize=4.8, color=INK, zorder=10,
        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85),
    )

    # north arrow
    xa = extent[0] + 0.91 * (extent[2] - extent[0])
    ya = extent[1] + 0.88 * (extent[3] - extent[1])
    ax.annotate(
        "", xy=(xa, ya + 3800), xytext=(xa, ya - 1200),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, mutation_scale=6),
        zorder=10,
    )
    ax.text(xa, ya + 4600, "N", ha="center", va="bottom", fontsize=4.8, color=INK, zorder=10,
            bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.9))

    # legend
    handles = [
        Line2D([0], [0], color=INK, lw=0.9, label="Hengzhou boundary"),
        Line2D([0], [0], color="#2C627D", lw=0.8, label="Yu River"),
        Line2D([0], [0], color="#797971", lw=0.8, label="Main road"),
    ]
    leg = ax.legend(
        handles=handles, loc="lower right", frameon=True, fontsize=4.4,
        handlelength=1.5, borderpad=0.30, labelspacing=0.25,
        framealpha=0.90, edgecolor="none",
    )
    leg.get_frame().set_facecolor("white")


def draw_workflow(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        ("P1", "Recognize new water", "LightGBM · 1,000 tiles", "IoU 0.618", AMBER),
        ("P2", "Fuse and grade", "Gated U-Net · GMM", "IoU 0.797", AMBER),
        ("P3", "Transfer to Hengzhou", "S1 27 Jun–16 Jul · guards", "26.9 km$^2$", BLUE),
        ("P4", "Assess catchments", "393 units · TOPSIS", "13.9k people", BLUE),
        ("P5", "Allocate relief", "6 depots · MILP", "7.3k served", BLUE),
    ]

    # domain key above the chain
    ax.plot(0.035, 0.955, marker="o", ms=3.0, mfc=AMBER, mec="none")
    ax.text(0.055, 0.955, "Official tiles", fontsize=5.7, color=AMBER, fontweight="bold", va="center")
    ax.text(0.255, 0.955, "S2 · GF-3 · labels", fontsize=5.1, color=MUTED, va="center")
    ax.plot(0.035, 0.905, marker="o", ms=3.0, mfc=BLUE, mec="none")
    ax.text(0.055, 0.905, "Hengzhou layers", fontsize=5.7, color=BLUE, fontweight="bold", va="center")
    ax.text(0.340, 0.905, "S1 · DEM · OSM · WorldPop", fontsize=5.1, color=MUTED, va="center")

    top = 0.845
    h = 0.125
    gap = 0.035
    for i, (num, title, method, stat, accent) in enumerate(stages):
        y = top - i * (h + gap) - h
        card = FancyBboxPatch(
            (0.03, y), 0.94, h,
            boxstyle="square,pad=0.0",
            linewidth=0.45, edgecolor=BORDER, facecolor="white", zorder=2,
        )
        ax.add_patch(card)
        ax.add_patch(Rectangle((0.03, y), 0.014, h, facecolor=accent, edgecolor="none", zorder=3))
        ax.text(0.055, y + h * 0.68, num, fontsize=6.8, fontweight="bold", color=accent, va="center")
        ax.text(0.120, y + h * 0.68, title, fontsize=6.4, fontweight="bold", color=INK, va="center")
        ax.text(0.120, y + h * 0.28, method, fontsize=5.2, color=MUTED, va="center")
        ax.text(0.955, y + h * 0.50, stat, fontsize=5.0, color=accent, ha="right", va="center")

        if i < len(stages) - 1:
            x = 0.50
            y1 = y
            y2 = y - gap
            style = "-" if i != 1 else (0, (2.2, 1.4))
            ax.annotate(
                "", xy=(x, y2 + 0.004), xytext=(x, y1 - 0.004),
                arrowprops=dict(arrowstyle="-|>", color="#6E7478", lw=0.55,
                                linestyle=style, mutation_scale=5, shrinkA=0, shrinkB=0),
                zorder=4,
            )
            if i == 1:
                ax.text(0.525, (y1 + y2) / 2, r"frozen model · $\tau_A$ · $\theta_g$",
                        fontsize=4.6, color=RED, va="center", ha="left")

    # bottom output band
    by = 0.012
    bh = 0.044
    ax.add_patch(FancyBboxPatch(
        (0.03, by), 0.94, bh,
        boxstyle="square,pad=0.0",
        linewidth=0.45, edgecolor=BORDER, facecolor="#F7F5EE", zorder=2,
    ))
    ax.text(0.50, by + bh / 2, "F · $A_{new}$ · grade · catchment ranking · relief plan",
            fontsize=5.1, color=INK, ha="center", va="center")



def main():
    L = load_layers()
    fig_w_mm, fig_h_mm = 183.0, 91.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    fig.patch.set_facecolor("white")

    # panel a: map
    ax_a = fig.add_axes([2.0 / fig_w_mm, 2.0 / fig_h_mm, 94.0 / fig_w_mm, 82.0 / fig_h_mm])
    draw_map(ax_a, L)
    # panel b: workflow
    ax_b = fig.add_axes([103.0 / fig_w_mm, 2.0 / fig_h_mm, 78.0 / fig_w_mm, 82.0 / fig_h_mm])
    draw_workflow(ax_b)

    # panel letters / titles
    fig.text(2.0 / fig_w_mm, 87.0 / fig_h_mm, "a", fontsize=9.0, fontweight="bold", color=INK)
    fig.text(7.0 / fig_w_mm, 87.0 / fig_h_mm, "Study area", fontsize=6.8, color=INK)
    fig.text(103.0 / fig_w_mm, 87.0 / fig_h_mm, "b", fontsize=9.0, fontweight="bold", color=INK)
    fig.text(108.0 / fig_w_mm, 87.0 / fig_h_mm, "Five-problem chain", fontsize=6.8, color=INK)

    out_pdf = OUT / "option2_studyarea_workflow.pdf"
    out_png = OUT / "option2_studyarea_workflow_preview.png"
    fig.savefig(out_pdf, dpi=600, facecolor="white")
    fig.savefig(out_png, dpi=300, facecolor="white")
    print("wrote", out_pdf, out_png)


if __name__ == "__main__":
    main()
