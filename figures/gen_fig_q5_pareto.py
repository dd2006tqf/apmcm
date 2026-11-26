"""Q5 bi-objective trade-off and parameter sensitivity.
(a) epsilon-constraint Pareto front of priority-weighted unmet share f1 vs weighted delivery time f2
(15 exact HiGHS solves), served population as colour, recommended plan, knee and the 07-04 plan re-optimized
under 07-16 inputs; (b) f1* of 30 one-at-a-time exact re-solves."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
from matplotlib.colors import LinearSegmentedColormap
from _utils.plot_utils import _lighten

q5 = load('problem_5_results.json')
P = q5['pareto']
f1 = np.array([p['f1'] for p in P]); f2 = np.array([p['f2'] for p in P]); sp = np.array([p['served_pop'] for p in P])
assert all(p['optimal'] for p in P)
kn = q5['knee_index']; rec = q5['plan_0716']; rc = q5['recourse_0704_to_0716']

fig = plt.figure(figsize=(6.6, 3.6), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[0.22, 1])
ax = fig.add_subplot(gs[1, 0])
cmap = LinearSegmentedColormap.from_list('pop', [_lighten(C['blue_secondary'], 0.75), C['blue_main']])
ax.plot(f1, f2, color=C['neutral_mid'], lw=1.0, zorder=1)
sc = ax.scatter(f1, f2, color=C['blue_secondary'], s=26, edgecolor=C['neutral_dark'], lw=0.5, zorder=3,
                label=r'Pareto point ($\varepsilon$-constraint)')
ax.scatter(rec['f1'], rec['f2'], marker='*', s=150, color=C['red_strong'], edgecolor='white', lw=0.6, zorder=5,
           label=f"Recommended, {rec['served_pop']:.0f} served")
ax.scatter(f1[kn], f2[kn], marker='D', s=46, facecolor='none', edgecolor=C['neutral_black'], lw=1.3, zorder=5,
           label=f'Knee, {sp[kn]:.0f} served')
ax.scatter(rc['f1'], rc['f2'], marker='^', s=46, color=C['gold'], edgecolor=C['neutral_dark'], lw=0.6, zorder=5,
           label=f"07-04 depots re-used, {rc['served_pop']:.0f} served")
ax.set_xlabel('Weighted unmet share $f_1$')
ax.set_ylabel('Weighted delivery time $f_2$ (min)')
dynamic_limits(ax, x=np.r_[f1, rc['f1']], y=np.r_[f2, rc['f2']], include_zero=True)
panel(ax, 'a')
lax = fig.add_subplot(gs[0, :]); lax.set_axis_off()
lax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=2, frameon=False, handlelength=1.0,
           columnspacing=0.8)

ax2 = fig.add_subplot(gs[1, 1])
groups = [('K', 'Depots K'), ('cap_scale', 'Capacity factor'), ('t_max', 'Time limit (min)'),
          ('q', 'Ration (kg/day)'), ('speed', 'Speed factor'), ('n_veh', 'Fleet size'),
          ('buffer', 'Road buffer (m)'), ('facility', 'Candidate set')]
base = rec['f1']
for i, (g, name) in enumerate(groups[::-1]):
    rows = [r for r in q5['sensitivity'] if r['param'] == g]
    assert all(r['optimal'] for r in rows)
    v = np.array([r['f1'] for r in rows])
    ax2.hlines(i, v.min(), v.max(), color=C['neutral_light'], lw=4)
    ax2.scatter(v, np.full(len(v), i), s=16, color=C['blue_main'], zorder=3)
    j = int(np.argmax(np.abs(v - base)))
    if abs(v[j] - base) > 0.005:
        val = rows[j]['value']
        val = {'schools_only': 'schools'}.get(val, val)
        ax2.annotate(f'{val}', (v[j], i), xytext=(5, 0), textcoords='offset points', va='center',
                     color=C['red_strong'])
ax2.axvline(base, color=C['red_strong'], lw=0.9, ls='--')
ax2.set_yticks(np.arange(len(groups)), [n for _, n in groups[::-1]])
ax2.yaxis.set_minor_locator(plt.NullLocator())
ax2.set_xlabel('Optimal $f_1^*$ (exact re-solve)')
ax2.set_xlim(0.555, 0.67)
panel(ax2, 'b')
finish(fig, 'fig_q5_pareto')
