"""Q1 error structure: (a) FP / FN area of LightGBM by pre-flood WorldCover stratum, with the
net area error of M0 and of the final M6; (b) TreeSHAP beeswarm of tile-level |area error| drivers."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

q1 = load('problem_1_results.json')
q2 = load('problem_2_results.json')
rows = [r for r in q1['models']['B']['fpfn_worldcover']['rows'] if r['ref_km2'] > 0]
m6 = {r['stratum']: r for r in q2['fpfn_M6']['rows']}
rows = sorted(rows, key=lambda r: r['net_km2'])
names = [r['stratum'] for r in rows]
FP = np.array([r['FP_km2'] for r in rows]); FN = np.array([r['FN_km2'] for r in rows])
NET = np.array([r['net_km2'] for r in rows]); NET6 = np.array([m6[n]['net_km2'] for n in names])
assert np.allclose(FP - FN, NET)

LABEL = {'rho_ref': 'Reference flood share', 'sar_mean': 'SAR mean', 'sar_std': 'SAR std (raw)',
         'ashman_share': 'Bimodality share', 'enl': 'ENL', 'ndvi_mean': 'NDVI mean',
         'share_built': 'Built-up share', 'share_cropland': 'Cropland share', 'share_tree': 'Tree share',
         'share_prewater': 'Pre-flood water share', 'opt_invalid': 'Optical invalid share',
         'boundary_density': 'Label boundary density'}
dr = q1['drivers']
desc = dr['descriptors']
S = np.asarray(dr['abs']['shap_values']); X = np.asarray(dr['X'], float)
order = np.argsort(np.abs(S).mean(0))[::-1][:8]

fig = plt.figure(figsize=(6.4, 5.6), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.32, 0.03], height_ratios=[1, 1.15])

ax = fig.add_subplot(gs[0, 0])
yy = np.arange(len(rows))
ax.barh(yy, FP, color=C['red_2'], edgecolor=C['red_strong'], lw=0.6, height=0.62)
ax.barh(yy, -FN, color=C['blue_secondary'], alpha=0.55, edgecolor=C['blue_main'], lw=0.6, height=0.62)
ax.scatter(NET, yy, marker='D', s=22, color=C['neutral_black'], zorder=3)
ax.scatter(NET6, yy, marker='o', s=24, facecolor='white', edgecolor=C['teal'], lw=1.3, zorder=3)
ax.axvline(0, color=C['neutral_dark'], lw=0.8)
ax.set_yticks(yy, names)
ax.set_xlabel('Area (km$^2$)   FN $\\leftarrow$ | $\\rightarrow$ FP')
dynamic_limits(ax, x=np.r_[FP, -FN, NET6], include_zero=True)
ax.yaxis.set_minor_locator(plt.NullLocator())
panel(ax, 'a')
lax = fig.add_subplot(gs[0, 1:]); lax.set_axis_off()
lax.legend(handles=[Patch(facecolor=C['red_2'], edgecolor=C['red_strong'], label='FP (M0)'),
                    Patch(facecolor=C['blue_secondary'], alpha=0.55, edgecolor=C['blue_main'], label='FN (M0)'),
                    Line2D([], [], marker='D', ls='', color=C['neutral_black'], label='Net M0'),
                    Line2D([], [], marker='o', ls='', mfc='white', mec=C['teal'], mew=1.3, label='Net M6')],
           loc='center', ncol=1, frameon=False, handlelength=1.2)

ax2 = fig.add_subplot(gs[1, :2])
cmap = LinearSegmentedColormap.from_list('feat', [C['blue_main'], C['neutral_light'], C['red_strong']])
rng = np.random.default_rng(2026)   # vertical jitter only (display)
for r, j in enumerate(order):
    s, x = S[:, j], X[:, j]
    lo, hi = np.nanpercentile(x, [5, 95])
    xn = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
    # density-aware jitter: spread more where points are dense
    hist, edges = np.histogram(s, 40)
    dens = hist[np.clip(np.digitize(s, edges) - 1, 0, 39)] / hist.max()
    yj = len(order) - 1 - r + rng.uniform(-1, 1, len(s)) * 0.36 * dens
    sc = ax2.scatter(s, yj, c=xn, cmap=cmap, vmin=0, vmax=1, s=4, alpha=0.85, edgecolors='none', rasterized=True)
ax2.axvline(0, color=C['neutral_mid'], lw=0.7)
ax2.set_yticks(np.arange(len(order))[::-1], [LABEL[desc[j]] for j in order])
ax2.set_xlabel('SHAP value for $|e_t|$ (km$^2$)')
ax2.yaxis.set_minor_locator(plt.NullLocator())
panel(ax2, 'b')
cax = fig.add_subplot(gs[1, 2])
cb = fig.colorbar(sc, cax=cax, ticks=[0, 1])
cb.ax.set_yticklabels(['Low', 'High'])
cb.set_label('Descriptor value')
finish(fig, 'fig_q1_error_drivers')
