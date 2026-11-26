"""Q2 ablation as waterfalls M1 -> M6 for IoU, MAE_A and |RAE| on the same 1000 held-out tiles.
Each step shows the change caused by one added component; dashed line = LightGBM M0."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

T = load('problem_2_results.json')['table']
seq = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
steps = ['M1\nSAR', '+optical\nfusion', '+area\nloss', '+modality\ndropout', '+MC\ndropout', '+isotonic\n$\\tau_A$', 'M6']
metrics = [('IoU', 'IoU', 1, 1.0), ('MAE_A', 'MAE$_A$ (km$^2$/tile)', -1, 1.0), ('RAE', '|RAE| (%)', -1, 100.0)]

fig = plt.figure(figsize=(6.6, 5.4), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(4, 1, height_ratios=[0.12, 1, 1, 1])
axes = []
for r, (key, ylabel, better, scale) in enumerate(metrics):
    ax = fig.add_subplot(gs[r + 1, 0], sharex=axes[0] if axes else None)
    axes.append(ax)
    v = np.array([abs(T[m][key]) if key == 'RAE' else T[m][key] for m in seq]) * scale
    ref0 = (abs(T['M0'][key]) if key == 'RAE' else T['M0'][key]) * scale
    x = np.arange(len(steps))
    ax.bar(0, v[0], color=C['neutral_mid'], width=0.62)
    ax.text(0, v[0], f'{v[0]:.3g}', ha='center', va='bottom')
    for i in range(1, len(seq)):
        d = v[i] - v[i - 1]
        good = d * better > 0
        col = C['green_3'] if good else C['red_2']
        edge = C['teal'] if good else C['red_strong']
        ax.bar(i, d, bottom=v[i - 1], color=col, edgecolor=edge, lw=0.7, width=0.62)
        ax.plot([i - 1 + 0.31, i - 0.31], [v[i - 1], v[i - 1]], color=C['neutral_dark'], lw=0.6)
        ax.text(i, max(v[i], v[i - 1]), (r'$\pm$<0.001' if abs(d) < 5e-4 else f'{d:+.3f}') if scale == 1 else f'{d:+.1f}', ha='center', va='bottom')
    ax.plot([len(seq) - 1 + 0.31, len(seq) - 0.31], [v[-1], v[-1]], color=C['neutral_dark'], lw=0.6)
    ax.bar(len(seq), v[-1], color=C['blue_main'], width=0.62)
    ax.text(len(seq), v[-1], f'{v[-1]:.3g}', ha='center', va='bottom')
    ax.axhline(ref0, color=C['neutral_black'], lw=0.9, ls='--')
    top = max(v.max(), ref0)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel(ylabel)
    panel(ax, 'abc'[r])
axes[-1].xaxis.set_minor_locator(plt.NullLocator())
axes[-1].set_xticks(np.arange(len(steps)), steps)
for ax in axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
lax = fig.add_subplot(gs[0, 0]); lax.set_axis_off()
lax.legend(handles=[Patch(facecolor=C['neutral_mid'], label='Start (M1)'),
                    Patch(facecolor=C['green_3'], edgecolor=C['teal'], label='Step improves'),
                    Patch(facecolor=C['red_2'], edgecolor=C['red_strong'], label='Step worsens'),
                    Patch(facecolor=C['blue_main'], label='Final (M6)'),
                    Line2D([], [], color=C['neutral_black'], ls='--', label='LightGBM M0')],
           loc='center', ncol=5, frameon=False, handlelength=1.3, columnspacing=1.0)
finish(fig, 'fig_q2_ablation')
