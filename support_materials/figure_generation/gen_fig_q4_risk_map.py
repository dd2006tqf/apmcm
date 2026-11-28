"""Q4 bivariate choropleth of the 393 place-catchment units of Hengzhou (2026-07-16 F):
hazard = new inundation area H1, exposure = flooded population E1, tertiles over the 259 affected units.
The five highest-ranked units of the TOPSIS index S_j are numbered."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
import geopandas as gpd
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from _utils.plot_utils import _lighten

q4 = load('problem_4_results.json')
u = gpd.read_file(os.path.join(OUT_DIR, 'q4_units.geojson')).to_crs(32649)
u['geometry'] = u.geometry.scale(1e-3, 1e-3, origin=(0, 0))
bnd = gpd.read_file(os.path.join(PUB, 'Boundary', 'Hengzhou_boundary.geojson')).to_crs(32649)
bnd['geometry'] = bnd.geometry.scale(1e-3, 1e-3, origin=(0, 0))
aff = u['affected'].astype(bool).values
assert aff.sum() == q4['n_affected']
assert abs(u['H1'].sum() - q4['A_new_units_sum']) < 1e-6

def tert(v):
    q = np.quantile(v[aff & (v > 0)], [1 / 3, 2 / 3])   # tertiles of the non-zero affected values
    return np.clip(np.searchsorted(q, v, side='right'), 0, 2), q

hc, hq = tert(u['H1'].values)
ec, eq = tert(u['E1'].values)
c00 = np.array(to_rgb(_lighten(C['neutral_mid'], 0.8)))
c20 = np.array(to_rgb(C['red_strong'])); c02 = np.array(to_rgb(C['blue_main']))
c22 = np.array(to_rgb(C['violet'])) * 0.75
grid = [[(1 - i / 2) * (1 - j / 2) * c00 + (i / 2) * (1 - j / 2) * c20 + (1 - i / 2) * (j / 2) * c02
         + (i / 2) * (j / 2) * c22 for j in range(3)] for i in range(3)]   # bilinear between 4 corners
cols = [tuple(grid[hc[k]][ec[k]]) if aff[k] else (1, 1, 1) for k in range(len(u))]

top = u.sort_values('rank').head(5)
fig = plt.figure(figsize=(6.6, 4.2), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.36], height_ratios=[0.9, 1])
ax = fig.add_subplot(gs[:, 0])
u.plot(ax=ax, color=cols, edgecolor=C['neutral_light'], linewidth=0.25)
bnd.boundary.plot(ax=ax, color=C['neutral_black'], lw=0.8)
for n, (_, r) in enumerate(top.iterrows(), 1):
    c = r.geometry.representative_point()
    ax.annotate(str(n), (c.x, c.y), xytext=(c.x + 4, c.y + 4), textcoords='data', ha='center', va='center',
                fontweight='bold', color=C['neutral_black'],
                bbox=dict(boxstyle='circle,pad=0.18', fc='white', ec=C['neutral_black'], lw=0.6),
                arrowprops=dict(arrowstyle='-', color=C['neutral_black'], lw=0.6))
ax.set_aspect('equal')
ax.set_xlabel('Easting, UTM 49N (km)'); ax.set_ylabel('Northing (km)')
panel(ax, 'a')

k = fig.add_subplot(gs[0, 1])
for i in range(3):
    for j in range(3):
        k.add_patch(Rectangle((j, i), 1, 1, facecolor=grid[i][j], edgecolor='white', lw=1))
k.set_xlim(0, 3); k.set_ylim(0, 3); k.set_aspect('equal')
k.set_xticks([0, 1, 2, 3], ['0', f'{eq[0]:.0f}', f'{eq[1]:.0f}', f'{u.E1.max():.0f}'])
k.set_yticks([0, 1, 2, 3], ['0', f'{hq[0]:.2f}', f'{hq[1]:.2f}', f'{u.H1.max():.2f}'])
k.set_xlabel('Flooded population $E_1$')
k.set_ylabel('New area $H_1$ (km$^2$)')
k.xaxis.set_minor_locator(plt.NullLocator()); k.yaxis.set_minor_locator(plt.NullLocator())
panel(k, 'b')
t = fig.add_subplot(gs[1, 1]); t.set_axis_off()
lines = [f"{n}  {py(r['name'])} ({py(r['zone'])})" if r['name'] != r['zone'] else f"{n}  {py(r['name'])}"
         for n, (_, r) in enumerate(top.iterrows(), 1)]
t.text(0.0, 1.0, 'Top-5 units by $S_j$', fontweight='bold', va='top', transform=t.transAxes)
for i, (_, r) in enumerate(top.iterrows()):
    t.text(0.0, 0.80 - i * 0.16, f"{i + 1}  {py(r['name'])}: {r['S']:.3f}", va='top', transform=t.transAxes)
t.text(0.0, 0.80 - 5 * 0.16, 'White: unit not affected', va='top', transform=t.transAxes, color=C['neutral_dark'])
finish(fig, 'fig_q4_risk_map')
