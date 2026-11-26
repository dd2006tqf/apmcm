"""Q2 method comparison heatmap: all models on the same 1000 held-out tiles (blocked CV) plus the
397 Q1-hard tiles. Colour = per-column rank score (1 best), text = actual metric value."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from _utils.plot_utils import _lighten

q2 = load('problem_2_results.json')
T, HS = q2['table'], q2['hard_set']['metrics']
models = ['A', 'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
names = ['A  split-Otsu', 'M0 LightGBM', 'M1 U-Net SAR', 'M2 +optical fusion', 'M3 +area loss',
         'M4 +modality dropout', 'M5 +MC dropout', 'M6 +isotonic, $\\tau_A$']
cols = [('Precision', 'Prec.', T, 1), ('Recall', 'Rec.', T, 1), ('F1', 'F1', T, 1),
        ('IoU', 'IoU', T, 1), ('kappa', '$\\kappa$', T, 1), ('RAE', 'RAE\n(%)', T, -2), ('MAE_A', 'MAE$_A$', T, -1),
        ('median_abs_RAE_t', 'med\n|RAE$_t$|', T, -1), ('IoU', 'IoU\nhard', HS, 1),
        ('MAE_A', 'MAE$_A$\nhard', HS, -1)]
V = np.full((len(models), len(cols)), np.nan)
for i, m in enumerate(models):
    for j, (k, _, src, _) in enumerate(cols):
        if m in src:
            V[i, j] = src[m][k]
score = np.full_like(V, np.nan)
for j, (_, _, _, d) in enumerate(cols):
    v = V[:, j].copy()
    if d == -2:
        v = -np.abs(v)
    elif d == -1:
        v = -v
    ok = np.isfinite(v)
    r = np.argsort(np.argsort(v[ok]))          # rank score: robust to the outlying baseline A
    score[ok, j] = r / (ok.sum() - 1)

cmap = LinearSegmentedColormap.from_list('score', ['#FFFFFF', _lighten(C['blue_secondary'], 0.6), C['blue_secondary'], C['blue_main']])
fig = plt.figure(figsize=(6.6, 3.5), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.025])
ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
draw_vector_heatmap(ax, score, xlabels=[c[1] for c in cols], ylabels=names, mask=~np.isfinite(score),
                    annot=False, cmap=cmap, vmin=0, vmax=1, cax=cax, cbar_label='Rank within column (1 = best)')
for i in range(len(models)):
    for j, (k, _, _, d) in enumerate(cols):
        if not np.isfinite(V[i, j]):
            ax.text(j, i, 'n/a', ha='center', va='center', color=C['neutral_mid'])
            continue
        face = cmap(score[i, j])
        ink = 'white' if score[i, j] > 0.75 else '#222222'
        txt = f'{100 * V[i, j]:+.1f}' if d == -2 else f'{V[i, j]:.2f}'
        best = score[i, j] == 1
        ax.text(j, i, txt, ha='center', va='center', color=ink, fontweight='bold' if best else 'normal')
ax.axvline(7.5, color=C['neutral_black'], lw=1.0)
ax.axhline(6.5, color=C['red_strong'], lw=1.0, ls='--')
ax.tick_params(axis='x', labelrotation=0)
for t in ax.get_xticklabels():
    t.set_ha('center')
ax.xaxis.tick_top()
finish(fig, 'fig_q2_model_compare')
