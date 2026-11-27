"""Q3 map of Hengzhou on 2026-07-16: new inundation F = W1 \\ W0, pre-flood water W0 and observed
recession, (a) whole city at 100 m display cells, (b) zoom on the 6 km window with most new area (10 m)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
import rasterio
import geopandas as gpd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

q3 = load('problem_3_results.json')
D = q3['dates']['20260716']
rd = lambda n: rasterio.open(os.path.join(OUT_DIR, n))
with rd('q3_newflood_0716.tif') as r:
    F = r.read(1) > 0; T = r.transform; crs = r.crs
with rd('q3_w0.tif') as r:
    W0 = r.read(1) > 0
with rd('q3_receded_0716.tif') as r:
    R = r.read(1) > 0
px_km2 = abs(T.a * T.e) / 1e6
bnd = gpd.read_file(os.path.join(PUB, 'Boundary', 'Hengzhou_boundary.geojson')).to_crs(crs)
roads = gpd.read_file(os.path.join(OUT_DIR, 'q5_roads_cut_0716.geojson')).to_crs(crs)
assert abs(F.sum() * px_km2 - D['A_new']) / D['A_new'] < 0.02, (F.sum() * px_km2, D['A_new'])

# display classes: 0 none, 1 W0, 2 receded, 3 new F  (F has priority; F and W0 are disjoint)
def classes(f, w, r):
    out = np.zeros(f.shape, np.uint8)
    out[w] = 1; out[r] = 2; out[f] = 3
    return out

b = 10   # 100 m display cells; a cell is shown as F if any F pixel lies in it
H, W = (F.shape[0] // b) * b, (F.shape[1] // b) * b
pool = lambda a: a[:H, :W].reshape(H // b, b, W // b, b).any(axis=(1, 3))
cls_a = classes(pool(F), pool(W0) & ~pool(F), pool(R) & ~pool(F))
from rasterio.features import geometry_mask
from rasterio.transform import Affine
inside_a = ~geometry_mask(bnd.geometry, out_shape=cls_a.shape, transform=T * Affine.scale(b))
cls_a[~inside_a] = 0
# zoom window: 600 x 600 px (6 km) with max new area
k = 600
Fs = F[:H, :W].reshape(H // 100, 100, W // 100, 100).sum(axis=(1, 3))
from scipy.ndimage import uniform_filter
best = np.unravel_index(np.argmax(uniform_filter(Fs.astype(float), 6, mode='constant')), Fs.shape)
r0 = int(np.clip(best[0] * 100 - k // 2 + 50, 0, F.shape[0] - k)); c0 = int(np.clip(best[1] * 100 - k // 2 + 50, 0, F.shape[1] - k))
win = (slice(r0, r0 + k), slice(c0, c0 + k))
cls_b = classes(F[win], W0[win], R[win])
zoom_km2 = F[win].sum() * px_km2

cmap = ListedColormap(['#FFFFFF', C['blue_secondary'], C['gold'], C['red_strong']])
x0, y0 = T.c / 1000, T.f / 1000
ext_a = [x0, x0 + W * T.a / 1000, y0 + H * T.e / 1000, y0]
ext_b = [x0 + c0 * T.a / 1000, x0 + (c0 + k) * T.a / 1000, y0 + (r0 + k) * T.e / 1000, y0 + r0 * T.e / 1000]

fig = plt.figure(figsize=(6.6, 3.9), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 0.09])
ax = fig.add_subplot(gs[0, 0])
ax.imshow(np.ma.masked_equal(cls_a, 0), cmap=cmap, vmin=-0.5, vmax=3.5, extent=ext_a,
          interpolation='nearest', rasterized=True)
bnd.boundary.scale(1e-3, 1e-3, origin=(0, 0)).plot(ax=ax, color=C['neutral_black'], lw=0.8, zorder=5)
bb = bnd.total_bounds / 1000
ax.set_xlim(bb[0] - 1, bb[2] + 1); ax.set_ylim(bb[1] - 1, bb[3] + 1)
ax.add_patch(Rectangle((ext_b[0], ext_b[2]), k * T.a / 1000, k * abs(T.e) / 1000, fill=False,
                       ec=C['neutral_black'], lw=1.2))
ax.set_aspect('equal')
ax.set_xlabel('Easting, UTM 49N (km)'); ax.set_ylabel('Northing (km)')
panel(ax, 'a')
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(np.ma.masked_equal(cls_b, 0), cmap=cmap, vmin=-0.5, vmax=3.5, extent=ext_b,
           interpolation='nearest', rasterized=True)
rr = roads.cx[ext_b[0] * 1000:ext_b[1] * 1000, ext_b[2] * 1000:ext_b[3] * 1000]
rr[~rr['cut'].astype(bool)].geometry.scale(1e-3, 1e-3, origin=(0, 0)).plot(ax=ax2, color=C['neutral_mid'], lw=0.6, zorder=5)
rr[rr['cut'].astype(bool)].geometry.scale(1e-3, 1e-3, origin=(0, 0)).plot(ax=ax2, color=C['neutral_black'], lw=1.4, zorder=6)
ax2.set_xlim(ext_b[0], ext_b[1]); ax2.set_ylim(ext_b[2], ext_b[3])
ax2.set_aspect('equal')
ax2.set_xlabel('Easting (km)')
ax2.set_title(f'Zoom: $A_{{new}}$ = {zoom_km2:.2f} km$^2$', loc='right', pad=3, fontsize=10)
panel(ax2, 'b')
from matplotlib.lines import Line2D
lax = fig.add_subplot(gs[1, :]); lax.set_axis_off()
lax.legend(handles=[Patch(facecolor=C['red_strong'], label=f"New inundation $F$ ({D['A_new']:.1f} km$^2$)"),
                    Patch(facecolor=C['blue_secondary'], label=f"Pre-flood water $W_0$ ({D['A_W0']:.1f} km$^2$)"),
                    Patch(facecolor=C['gold'], label=f"Receded ({D['A_R']:.1f} km$^2$)"),
                    Line2D([], [], color=C['neutral_black'], lw=1.4, label='Road cut by $F$'),
                    Line2D([], [], color=C['neutral_mid'], lw=0.6, label='Road open')],
           loc='center', ncol=3, frameon=False, handlelength=1.3)
finish(fig, 'fig_q3_newflood_map')
