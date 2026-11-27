"""Q3 figure: 16 July new inundation draped on the Copernicus DEM, two view angles.
Window 40 x 32 km around the densest new water; W0 and F are coloured on the terrain surface.
Colours follow the paper palette (blue_secondary #4284B6, red_strong #BD564E)."""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output')
PUB = os.path.join(ROOT, 'user_data', 'a题数据汇总', '公开数据')
FIG = os.path.join(ROOT, 'figures')

BLUE = '#4284B6'
RED = '#BD564E'
BLACK = '#343534'
GRAY_MID = '#999999'

# ---- masks ----------------------------------------------------------------
with rasterio.open(os.path.join(OUT, 'q3_newflood_0716.tif')) as r:
    F = r.read(1) > 0; T = r.transform; crs = r.crs
with rasterio.open(os.path.join(OUT, 'q3_w0.tif')) as r:
    W0 = r.read(1) > 0
with rasterio.open(os.path.join(OUT, 'q3_receded_0716.tif')) as r:
    R = r.read(1) > 0
px_km2 = abs(T.a * T.e) / 1e6

# ---- pick a 40 x 32 km window with the densest new water -------------------
win_w, win_h = 40_000, 32_000
step = 500
H, W = F.shape
gh, gw = H * T.e * -1 // step, W * T.a // step
Fs = np.zeros((int(gh), int(gw)), np.float32)
for i in range(Fs.shape[0]):
    for j in range(Fs.shape[1]):
        r0, c0 = int(i * step / -T.e), int(j * step / T.a)
        Fs[i, j] = F[r0:r0 + int(step / -T.e), c0:c0 + int(step / T.a)].sum()
from scipy.ndimage import uniform_filter
dens = uniform_filter(Fs, size=(int(win_h / step), int(win_w / step)), mode='constant')
# keep the whole window inside the raster
pad_i, pad_j = int(win_h / step) // 2, int(win_w / step) // 2
dens[:pad_i, :] = -1; dens[-pad_i:, :] = -1
dens[:, :pad_j] = -1; dens[:, -pad_j:] = -1
bi, bj = np.unravel_index(np.argmax(dens), dens.shape)
cx = T.c + (bj + win_w / step / 2) * step
cy = T.f - (bi + win_h / step / 2) * step
x0, x1 = cx - win_w / 2, cx + win_w / 2
y1, y0 = cy + win_h / 2, cy - win_h / 2

# ---- DEM on a 50 m grid over the window ------------------------------------
res = 50
nx, ny = int(win_w / res), int(win_h / res)
dst_transform = from_origin(x0, y1, res, res)
dem_src = os.path.join(PUB, 'DEM_clip', 'DEM_Hengzhou_30m_mosaic.tif')
Z = np.full((ny, nx), np.nan, np.float32)
with rasterio.open(dem_src) as src:
    reproject(source=rasterio.band(src, 1), destination=Z,
              src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
              dst_transform=dst_transform, dst_crs=crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)

# ---- masks on the 50 m grid (a cell is water if any 10 m pixel is water) --
def block_any(mask, res_m):
    b = int(res_m / 10)
    hh, ww = (mask.shape[0] // b) * b, (mask.shape[1] // b) * b
    return mask[:hh, :ww].reshape(hh // b, b, ww // b, b).any(axis=(1, 3))

Fb, W0b = block_any(F, res), block_any(W0, res)
c0n, c1n = int((x0 - T.c) / T.a), int((x1 - T.c) / T.a)
r0n, r1n = int((y1 - T.f) / T.e), int((y0 - T.f) / T.e)
Fw = Fb[r0n // 5:r1n // 5, c0n // 5:c1n // 5]
W0w = W0b[r0n // 5:r1n // 5, c0n // 5:c1n // 5]

# exact water areas in the window (from the native 10 m masks)
F_km2 = F[r0n:r1n, c0n:c1n].sum() * px_km2
W0_km2 = W0[r0n:r1n, c0n:c1n].sum() * px_km2

# ---- terrain colours -------------------------------------------------------
from scipy.ndimage import gaussian_filter
Zs = gaussian_filter(np.nan_to_num(Z, nan=np.nanmedian(Z)), 1.3)
ls = LightSource(azdeg=315, altdeg=38)
rgb = ls.shade(Zs, cmap=cm.gray, vert_exag=2.2, blend_mode='soft', vmin=0, vmax=800)
rgb[..., :3] = 0.55 + 0.45 * rgb[..., :3]
fc = rgb.copy()
fc[W0w] = matplotlib.colors.to_rgba(BLUE)
fc[Fw] = matplotlib.colors.to_rgba(RED)

X, Y = np.meshgrid((x0 - x0 + np.arange(nx) * res) / 1000,
                   (y1 - y0 - np.arange(ny) * res) / 1000)
X = X + x0 / 1000
Y = Y + y0 / 1000

# ---- draw -----------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
})


def render_view(az, el, path, show_z=True):
    """Render one 3D view on its own canvas so labels never collide."""
    fig = plt.figure(figsize=(4.3, 3.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d')
    ax.plot_surface(X, Y, Zs, facecolors=fc, rstride=2, cstride=2,
                    linewidth=0, antialiased=False, shade=False, rasterized=True)
    ax.view_init(elev=el, azim=az)
    ax.set_box_aspect((win_w, win_h, 5200), zoom=1.30)
    ax.set_xticks([x0 / 1000 + 5, x0 / 1000 + 20, x0 / 1000 + 35])
    ax.set_yticks([y0 / 1000 + 5, y0 / 1000 + 16, y0 / 1000 + 27])
    ax.set_zticks([100, 700])
    ax.tick_params(pad=-1, labelsize=7.2)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color('#C9C9C9')
        axis.line.set_linewidth(0.4)
    ax.set_xlabel('Easting (km)', labelpad=-1, fontsize=8)
    ax.set_ylabel('Northing (km)', labelpad=-2, fontsize=8)
    if show_z:
        ax.set_zlabel('Elevation (m)', labelpad=-4, fontsize=8)
    else:
        ax.zaxis.set_ticks([])
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def trim_white(path):
    """Crop uniform white margins left by the 3D axes box."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        pad = 6
        im.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                 min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))).save(path)


tmp_a = os.path.join(FIG, '_view_a.png')
tmp_b = os.path.join(FIG, '_view_b.png')
render_view(-55, 42, tmp_a, show_z=True)
render_view(125, 38, tmp_b, show_z=False)
trim_white(tmp_a); trim_white(tmp_b)

fig = plt.figure(figsize=(7.3, 2.7))
for k, p in enumerate([tmp_a, tmp_b]):
    im = plt.imread(p)
    wfrac = 0.485
    hfrac = wfrac * 7.3 * im.shape[0] / im.shape[1] / 2.7
    bottom = 0.955 - hfrac
    ax = fig.add_axes([0.008 + 0.497 * k, bottom, wfrac, hfrac])
    ax.imshow(im); ax.axis('off')
    ax.text(0.02, 0.99, f'({chr(97 + k)})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

handles = [Patch(facecolor=BLUE, label=f'Pre-flood water $W_0$ ({W0_km2:.1f} km$^2$)'),
           Patch(facecolor=RED, label=f'New inundation $F$ ({F_km2:.1f} km$^2$)'),
           Patch(facecolor='#9A9A9A', label='Low terrain'),
           Patch(facecolor='#4F4F4F', label='High terrain')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
           handlelength=1.1, columnspacing=1.2, bbox_to_anchor=(0.5, 0.01),
           fontsize=8)

fig.savefig(os.path.join(FIG, 'fig_q3_terrain.pdf'), dpi=300)
fig.savefig(os.path.join(FIG, 'fig_q3_terrain.png'), dpi=200)
print(f'window x {x0/1000:.1f}-{x1/1000:.1f} km, y {y0/1000:.1f}-{y1/1000:.1f} km; '
      f'F {F_km2:.2f} km2, W0 {W0_km2:.2f} km2; z {np.nanmin(Z):.0f}-{np.nanmax(Z):.0f} m')
