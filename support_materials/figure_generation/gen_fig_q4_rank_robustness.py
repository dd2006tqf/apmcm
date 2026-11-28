"""Q4 ranking robustness: (a) baseline rank and 5-95 % rank interval under Dirichlet weight perturbation
(M = 2000) for the 25 highest-ranked units; (b) Spearman agreement and top-10 overlap of alternative
weightings / inputs with the baseline ranking."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
import geopandas as gpd

q4 = load('problem_4_results.json')
u = gpd.read_file(os.path.join(OUT_DIR, 'q4_units.geojson'))
top = u.sort_values('rank').head(25).iloc[::-1]
rob = q4['robustness']
alt = rob['alternative_inputs']

fig = plt.figure(figsize=(6.6, 4.4), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
fig.get_layout_engine().set(rect=(0.03, 0, 0.97, 1))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[0.08, 1])
ax = fig.add_subplot(gs[1, 0])
y = np.arange(len(top))
lvl_col = {1: C['red_strong'], 2: C['gold'], 3: C['teal'], 4: C['blue_secondary']}
ax.hlines(y, top['rank_lo'], top['rank_hi'], color=C['neutral_mid'], lw=2.4, alpha=0.6,
          label='5–95% interval')
for lv in sorted(top['level'].unique()):
    m = top['level'].values == lv
    ax.scatter(top['rank'].values[m], y[m], s=26, color=lvl_col[int(lv)], zorder=3, marker='o',
               label=f'Level {"I II III IV".split()[int(lv) - 1]}')
ax.scatter(top['rank_med'], y, s=14, marker='|', color=C['neutral_black'], zorder=4, label='Median rank')
names = [py(n) for n in top['name']]
ax.set_yticks(y, names)
ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel('Rank of $S_j$ (log scale, 1 = top)')
ax.set_xscale('log')
ax.set_xlim(max(top['rank_hi']) * 1.15, 0.8)
ax.set_xticks([1, 2, 5, 10, 20, 50], ['1', '2', '5', '10', '20', '50'])
ax.xaxis.set_minor_locator(plt.NullLocator())
panel(ax, 'a')
lax = fig.add_subplot(gs[0, :]); lax.set_axis_off()
lax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=6, frameon=False, handlelength=1.0,
           columnspacing=0.9)

ax2 = fig.add_subplot(gs[1, 1])
cases = [('Dirichlet, median', rob['spearman_median'], None),
         ('Dirichlet, 5% quantile', rob['spearman_p05'], None),
         ('Hazard criteria only', rob['hazard_only_spearman'], rob['hazard_only_top10_overlap']),
         ('$W_0$ = SAR only', alt['W0_sar_only']['spearman_all_units'], alt['W0_sar_only']['top10_overlap']),
         ('REM guard $h^*$ = 30 m', alt['hstar_30']['spearman_all_units'], alt['hstar_30']['top10_overlap']),
         ('Flood date 07-04', alt['F_0704']['spearman_all_units'], alt['F_0704']['top10_overlap'])]
cases = cases[::-1]
yy = np.arange(len(cases))
vals = np.array([c[1] for c in cases])
ax2.hlines(yy, 0.5, vals, color=C['neutral_light'], lw=2)
ax2.scatter(vals, yy, s=34, color=C['blue_main'], zorder=3)
for yi, (_, v, o) in zip(yy, cases):
    txt = f'{v:.3f}' + (f'  ({o}/10)' if o is not None else '')
    ax2.annotate(txt, (v, yi), xytext=(0, 7), textcoords='offset points', ha='center', va='bottom')
ax2.set_yticks(yy, [c[0] for c in cases])
ax2.yaxis.set_minor_locator(plt.NullLocator())
ax2.set_xlim(0.5, 1.04)
ax2.set_ylim(-0.6, len(cases) - 0.2)
ax2.set_xlabel('Spearman $\\rho_s$ vs. baseline')
panel(ax2, 'b')
finish(fig, 'fig_q4_rank_robustness')
