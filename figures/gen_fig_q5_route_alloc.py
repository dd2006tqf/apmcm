"""Q5 recommended plan on the flood-cut road network (decision time 2026-07-16): open depots, delivery
paths (line width ~ tonnage), demand units by served share, roads cut by F and the inadmissible depot."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
import geopandas as gpd
from matplotlib.lines import Line2D

q5 = load('problem_5_results.json')
P = q5['plan_0716']
dep = P['depots']; dem = P['demand']
bnd = gpd.read_file(os.path.join(PUB, 'Boundary', 'Hengzhou_boundary.geojson')).to_crs(32649)
roads = gpd.read_file(os.path.join(OUT_DIR, 'q5_roads_cut_0716.geojson')).to_crs(32649)
km = lambda g: g.scale(1e-3, 1e-3, origin=(0, 0))
cut = roads['cut'].astype(bool)
d = np.array([x['d'] for x in dem]); un = np.clip(np.array([x['u'] for x in dem]), 0, None)
served = 1 - un / d
assert abs((d - un).sum() - P['total_shipped_t']) < 1e-6

fig = plt.figure(figsize=(6.6, 5.0), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.2], width_ratios=[1, 0.3])
ax = fig.add_subplot(gs[0, 0])
km(roads[~cut].geometry).plot(ax=ax, color=C['neutral_light'], lw=0.35, zorder=1)
km(roads[cut].geometry).plot(ax=ax, color=C['red_strong'], lw=1.0, zorder=2)
km(bnd.boundary).plot(ax=ax, color=C['neutral_black'], lw=0.8, zorder=2)
tmax = max(p['tons'] for p in q5['paths_0716'])
for p in q5['paths_0716']:
    xy = np.asarray(p['xy']) / 1000
    ax.plot(xy[:, 0], xy[:, 1], color=C['blue_main'], lw=0.5 + 2.5 * p['tons'] / tmax, alpha=0.75,
            solid_capstyle='round', zorder=3)
cx = np.array([x['cx'] for x in dem]) / 1000; cy = np.array([x['cy'] for x in dem]) / 1000
sz = 8 + 60 * d / d.max()
full = served > 0.999; none = served < 1e-3; part = ~full & ~none
ax.scatter(cx[full], cy[full], s=sz[full], color=C['green_3'], edgecolor=C['teal'], lw=0.6, zorder=4)
ax.scatter(cx[part], cy[part], s=sz[part], color=C['gold'], edgecolor=C['neutral_dark'], lw=0.6, zorder=4)
ax.scatter(cx[none], cy[none], s=sz[none], facecolor='white', edgecolor=C['red_strong'], lw=1.0, zorder=4)
open_set = set(P['open_depots'])
use = P['depot_use_t']
for x in dep:
    X, Y = x['x'] / 1000, x['y'] / 1000
    if x['name'] in open_set:
        continue
    elif not x['admissible']:
        ax.scatter(X, Y, marker='X', s=60, color=C['red_strong'], edgecolor='white', lw=0.6, zorder=6)
    else:
        ax.scatter(X, Y, marker='s', s=26, facecolor='white', edgecolor=C['neutral_mid'], lw=0.9, zorder=5)
order = sorted(open_set, key=lambda n: -use[n])
for n, name in enumerate(order, 1):
    x = next(v for v in dep if v['name'] == name)
    ax.scatter(x['x'] / 1000, x['y'] / 1000, marker='s', s=120, color=C['blue_main'], edgecolor='white', lw=0.8, zorder=6)
    ax.scatter(x['x'] / 1000, x['y'] / 1000, marker=f'${n}$', s=40, color='white', lw=0, zorder=7)
ax.set_aspect('equal')
bb = km(bnd.geometry).total_bounds
ax.set_xlim(bb[0] - 1, bb[2] + 1); ax.set_ylim(bb[1] - 1, bb[3] + 1)
ax.set_xlabel('Easting, UTM 49N (km)'); ax.set_ylabel('Northing (km)')
tax = fig.add_subplot(gs[0, 1]); tax.set_axis_off()
tax.text(0, 0.98, 'Open depots', fontweight='bold', va='top', transform=tax.transAxes)
tax.text(0, 0.91, 'shipped (t)', va='top', color=C['neutral_dark'], transform=tax.transAxes)
for n, name in enumerate(order, 1):
    tax.text(0, 0.82 - (n - 1) * 0.075, f'{n}  {py(name)}: {use[name]:.1f}', va='top', transform=tax.transAxes)
tax.text(0, 0.82 - 6.4 * 0.075, f"Total {P['total_shipped_t']:.1f} of\n{P['total_demand_t']:.1f} t demand", va='top',
         transform=tax.transAxes, color=C['neutral_dark'])
lax = fig.add_subplot(gs[1, :]); lax.set_axis_off()
h = [Line2D([], [], marker='s', ls='', ms=8, color=C['blue_main'], label=f'Open depot (K={len(open_set)})'),
     Line2D([], [], marker='s', ls='', ms=6, mfc='white', mec=C['neutral_mid'], label='Closed candidate'),
     Line2D([], [], marker='X', ls='', ms=7, color=C['red_strong'], label='Depot within 100 m of $F$'),
     Line2D([], [], color=C['blue_main'], lw=2, label='Delivery path (width ~ t)'),
     Line2D([], [], color=C['red_strong'], lw=1.0, label=f"Road cut by $F$ ({q5['cut_road_km']['d0716']:.0f} km)"),
     Line2D([], [], marker='o', ls='', ms=7, color=C['green_3'], mec=C['teal'], label=f'Fully served ({full.sum()})'),
     Line2D([], [], marker='o', ls='', ms=7, color=C['gold'], mec=C['neutral_dark'], label=f'Partly served ({part.sum()})'),
     Line2D([], [], marker='o', ls='', ms=7, mfc='white', mec=C['red_strong'], label=f'Not served ({none.sum()})')]
lax.legend(handles=h, loc='center', ncol=3, frameon=False, handlelength=1.4, columnspacing=1.0)
finish(fig, 'fig_q5_route_alloc')
