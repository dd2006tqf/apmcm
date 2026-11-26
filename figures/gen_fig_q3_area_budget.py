"""Q3 area budget of Hengzhou for the two flood dates (dumbbell): guard cascade from the raw transferred
model mask to the final new inundation A_new, and the water balance terms W1, W0, receded R and A_net."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *

q3 = load('problem_3_results.json')
d4, d16 = q3['dates']['20260704'], q3['dates']['20260716']
cascade = [('raw_model', 'Raw model'), ('after_W0', 'W₀ guard'), ('after_REM', 'REM guard'),
           ('after_dVV', 'ΔVV guard'), ('after_MMU', 'MMU (final)')]
balance = [('A_W1', 'W₁ water'), ('A_W0', 'W₀ water'), ('A_R', 'Receded'), ('A_new', 'New area'),
           ('A_net', 'Net change')]
for d in (d4, d16):
    assert abs(d['guard_stages_km2']['after_MMU'] - d['A_new']) < 1e-6
    assert d['A_net'] <= d['A_new']

fig = plt.figure(figsize=(6.4, 4.6), layout='constrained')
set_paper_placement(fig, width_fraction=0.95)
fig.get_layout_engine().set(rect=(0.05, 0, 0.95, 1))
gs = fig.add_gridspec(3, 1, height_ratios=[0.1, 1, 1])
c4, c16 = C['neutral_mid'], C['red_strong']


def dumbbell(ax, items, getter, letter):
    y = np.arange(len(items))[::-1]
    a = np.array([getter(d4, k) for k, _ in items]); b = np.array([getter(d16, k) for k, _ in items])
    for yi, u, v in zip(y, a, b):
        ax.plot([u, v], [yi, yi], color=C['neutral_light'], lw=3.2, solid_capstyle='round', zorder=1)
    ax.scatter(a, y, color=c4, s=30, zorder=3, marker='o', label='2026-07-04')
    ax.scatter(b, y, color=c16, s=30, zorder=4, marker='D', label='2026-07-16')
    ax.set_yticks(y, [n for _, n in items])
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel('Area inside Hengzhou (km$^2$)')
    panel(ax, letter)
    tw = ax.twinx()
    tw.set_ylim(ax.get_ylim())
    tw.set_yticks(y, [f'{u:.1f} / {v:.1f}' for u, v in zip(a, b)])
    tw.yaxis.set_minor_locator(plt.NullLocator())
    tw.set_ylabel('07-04 / 07-16 (km$^2$)')
    tw.spines['right'].set_visible(True)
    return np.r_[a, b]


ax = fig.add_subplot(gs[1, 0])
v = dumbbell(ax, cascade, lambda d, k: d['guard_stages_km2'][k], 'a')
ax.set_xlim(0, v.max() * 1.06)
ax2 = fig.add_subplot(gs[2, 0])
v2 = dumbbell(ax2, balance, lambda d, k: d[k], 'b')
ax2.axvline(0, color=C['neutral_black'], lw=0.8)
ax2.set_xlim(v2.min() - 15, v2.max() * 1.06)
lax = fig.add_subplot(gs[0, :]); lax.set_axis_off()
lax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=2, frameon=False)
finish(fig, 'fig_q3_area_budget')
