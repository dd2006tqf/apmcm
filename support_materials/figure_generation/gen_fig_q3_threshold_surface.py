"""Q3 sensitivity of the Hengzhou new-inundation area (2026-07-16): (a) response surface of A_new over the
threshold shift Delta-tau and the REM guard h* (30 full re-computations); (b) one-at-a-time changes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
from matplotlib.colors import LinearSegmentedColormap
from _utils.plot_utils import _lighten

q3 = load('problem_3_results.json')
grid = q3['sensitivity_grid']
dts = sorted({g['dtau'] for g in grid})
hs = [5, 10, 15, 20, 30, None]
Z = np.array([[next(g['A_new'] for g in grid if g['dtau'] == d and g['hstar'] == h) for h in hs] for d in dts])
base = q3['dates']['20260716']['A_new']
assert abs(Z[dts.index(0.0), hs.index(15)] - base) < 1e-9

fig = plt.figure(figsize=(6.6, 3.4), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
fig.get_layout_engine().set(rect=(0.02, 0, 0.98, 1))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 0.045, 1.1], height_ratios=[1, 0.13])
ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
xi = np.arange(len(hs))
cmap = LinearSegmentedColormap.from_list('a', ['#FFFFFF', _lighten(C['red_strong'], 0.55), C['red_strong']])
levels = np.linspace(0, 80, 17)
cf = ax.contourf(xi, dts, Z, levels=levels, cmap=cmap)
cs = ax.contour(xi, dts, Z, levels=[5, 10, 20, 30, 40, 60], colors=C['neutral_dark'], linewidths=0.6)
ax.clabel(cs, fmt='%g', inline=True, fontsize=plt.rcParams['font.size'] * 0.95)
ax.scatter(xi.repeat(len(dts)).reshape(len(hs), -1).T, np.repeat(np.array(dts)[:, None], len(hs), 1),
           s=6, color=C['neutral_black'], zorder=3, label='Computed case')
ax.scatter([hs.index(15)], [0.0], marker='*', s=110, color=C['blue_main'], edgecolor='white', lw=0.6,
           zorder=4, label=f'Adopted ({base:.1f} km$^2$)')
ax.set_xticks(xi, ['5', '10', '15', '20', '30', '$\\infty$'])
ax.set_xlabel('REM guard $h^*$ (m)')
ax.set_ylabel('Threshold shift $\\Delta\\tau$')
ax.set_yticks(dts)
ax.yaxis.set_minor_locator(plt.NullLocator()); ax.xaxis.set_minor_locator(plt.NullLocator())
panel(ax, 'a')
lax = fig.add_subplot(gs[1, 0:2]); lax.set_axis_off()
lax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=2, frameon=False)
cb = fig.colorbar(cf, cax=cax, ticks=[0, 20, 40, 60, 80])
cb.set_label('New inundation area (km$^2$)')

ax2 = fig.add_subplot(gs[:, 2])
lab = {'delta_db': '$\\delta$ (dB)', 'W0_mode': '$W_0$', 'S1_scale': 'S1 scale', 'MMU_px': 'MMU (px)'}
rows = [r for r in q3['sensitivity_oat'] if abs(r['A_new'] - base) > 1e-9]
rows += [dict(param='hstar', value=v, A_new=Z[dts.index(0.0), hs.index(v)]) for v in (5, 30)]
lab['hstar'] = '$h^*$ (m)'
rows = sorted(rows, key=lambda r: r['A_new'] - base)
y = np.arange(len(rows))
dlt = np.array([r['A_new'] - base for r in rows])
for i, d in enumerate(dlt):
    col = C['red_strong'] if d > 0 else C['blue_main']
    ax2.plot([0, d], [i, i], color=col, lw=1.6)
    ax2.scatter(d, i, color=col, s=26, zorder=3)
ax2.axvline(0, color=C['neutral_black'], lw=0.8)
ax2.set_yticks(y, [f"{lab[r['param']]} = {str(r['value']).replace('sar_only', 'SAR only')}" for r in rows])
ax2.set_xlabel('$\\Delta A_{new}$ (km$^2$)')
ax2.yaxis.set_minor_locator(plt.NullLocator())
dynamic_limits(ax2, x=np.r_[dlt, 0], include_zero=True)
panel(ax2, 'b')
finish(fig, 'fig_q3_threshold_surface')
