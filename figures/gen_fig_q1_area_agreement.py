"""Bland-Altman agreement of tile new-inundation area (1000 held-out tiles):
baseline A (split-Otsu - NDWI) and baseline B (LightGBM)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
from matplotlib.colors import LinearSegmentedColormap

q1 = load('problem_1_results.json')
rho = np.asarray(q1['tile_flood_share'])
cmap = LinearSegmentedColormap.from_list('flood', [C['neutral_light'], C['blue_secondary'], C['blue_main']])

fig = plt.figure(figsize=(6.4, 3.5), layout='constrained')
set_paper_placement(fig, width_fraction=0.95)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 0.26])
axes = []
for k, (key, name) in enumerate([('A', 'A: split-Otsu $-$ NDWI'), ('B', 'B: LightGBM (M0)')]):
    m = q1['models'][key]
    ref = np.asarray(m['A_ref_t']); hat = np.asarray(m['A_hat_t'])
    pa = m['pooled_area']
    mean, diff = 0.5 * (ref + hat), hat - ref
    assert abs(diff.mean() - pa['BA_bias']) < 1e-6
    ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
    axes.append(ax)
    sc = ax.scatter(mean, diff, c=rho, cmap=cmap, vmin=0, vmax=1, s=7, alpha=0.8,
                    edgecolors='none', rasterized=True)
    ax.axhline(0, color=C['neutral_mid'], lw=0.7)
    ax.axhline(pa['BA_bias'], color=C['red_strong'], lw=1.2, label=f"Bias {pa['BA_bias']:+.2f}")
    for v in (pa['BA_loa_low'], pa['BA_loa_high']):
        ax.axhline(v, color=C['red_strong'], lw=1.0, ls='--',
                   label=f"95% LoA [{pa['BA_loa_low']:+.2f}, {pa['BA_loa_high']:+.2f}]" if v == pa['BA_loa_low'] else None)
    ax.set_xlim(0, 5.0)
    ax.set_xlabel('Mean area (km$^2$/tile)')
    if k == 0:
        ax.set_ylabel('Estimate $-$ reference (km$^2$/tile)')
    ax.set_title(name, loc='right', pad=3)
    panel(ax, 'ab'[k])
    lax = fig.add_subplot(gs[1, k]); lax.set_axis_off()
    lax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=1, frameon=False)
axes[0].set_ylim(-5.2, 5.2)
plt.setp(axes[1].get_yticklabels(), visible=False)
cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(sc, cax=cax)
cb.set_label('Reference flood share $\\rho_t$')
finish(fig, 'fig_q1_area_agreement')
