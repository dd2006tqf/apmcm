"""Q2 four-level grading: (a) filled density of logit(rho) of the 985 non-empty reference tiles with
the 4-component GMM, Bayes thresholds and 90 % block-bootstrap intervals (Jenks marks for comparison);
(b) reference grade vs Monte Carlo grade of M6 (row-normalized, M = 2000 draws)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap
from _utils.plot_utils import _lighten

q1 = load('problem_1_results.json')
q2 = load('problem_2_results.json')
G = q2['grading']
rho = np.asarray(q1['tile_flood_share'])
rho = rho[rho > 0]
assert len(rho) == G['n_nonempty']
x = np.log(np.clip(rho, 1e-4, 1 - 1e-4) / (1 - np.clip(rho, 1e-4, 1 - 1e-4)))
mu, sd, pi = map(np.asarray, (G['components']['mu'], G['components']['sd'], G['components']['pi']))
lg = lambda p: np.log(p / (1 - p))
theta = np.asarray(G['theta']); lo90, hi90 = map(np.asarray, G['boot_ci90'])
grade_names = ['General', 'Rel. strong', 'Strong', 'Extremely strong']
gcols = [C['blue_secondary'], C['teal'], C['gold'], C['red_strong']]

fig = plt.figure(figsize=(6.6, 3.5), layout='constrained')
set_paper_placement(fig, width_fraction=0.98)
gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 0.045], height_ratios=[0.26, 1])
ax = fig.add_subplot(gs[1, 0])
bins = np.linspace(-9.5, 9.5, 58)
ax.hist(x, bins=bins, density=True, color=C['neutral_light'], edgecolor='white', lw=0.4, label='Tiles (n=985)')
xx = np.linspace(-9.5, 9.5, 600)
tot = np.zeros_like(xx)
for k in range(4):
    f = pi[k] * norm.pdf(xx, mu[k], sd[k]); tot += f
    ax.fill_between(xx, 0, f, color=gcols[k], alpha=0.25, lw=0)
    ax.plot(xx, f, color=gcols[k], lw=1.1, label=grade_names[k])
ax.plot(xx, tot, color=C['neutral_black'], lw=1.0, ls='-', label='GMM mixture')
ymax = max(tot.max(), np.histogram(x, bins, density=True)[0].max()) * 1.18
for t, a, b in zip(theta, lo90, hi90):
    ax.axvspan(lg(a), lg(b), color=C['neutral_mid'], alpha=0.12, lw=0)
    ax.axvline(lg(t), color=C['neutral_black'], lw=1.0, ls='--')
jt = np.asarray(G['jenks_theta'])
ax.plot(lg(jt), np.zeros_like(jt), marker='^', ls='', ms=6, color=C['violet'], clip_on=False, zorder=4,
        label='Jenks breaks')
ax.set_ylim(0, ymax)
ax.set_xlim(-9.5, 9.5)
ax.set_xlabel('logit of reference flood share $\\rho_t$')
ax.set_ylabel('Density')
sec = ax.secondary_xaxis('top')
sec.set_xticks([lg(t) for t in theta], [f'{100 * t:.1f}%' for t in theta])
sec.set_xlabel('Thresholds $\\theta_1,\\theta_2,\\theta_3$ (90% bootstrap bands)')
sec.tick_params(length=3)
sec.xaxis.set_minor_locator(plt.NullLocator())
panel(ax, 'a')
lax = fig.add_subplot(gs[0, :]); lax.set_axis_off()
lax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=4, frameon=False,
           handlelength=1.2, columnspacing=0.8)

ax2 = fig.add_subplot(gs[1, 1]); cax = fig.add_subplot(gs[1, 2])
Cm = np.asarray(q2['grade_mc']['confusion_ref_vs_mc'], float)
R = Cm / Cm.sum(1, keepdims=True)
cmap = LinearSegmentedColormap.from_list('cm', ['#FFFFFF', _lighten(C['blue_secondary'], 0.5), C['blue_main']])
draw_vector_heatmap(ax2, R, xlabels=['G', 'RS', 'S', 'ES'], ylabels=['G', 'RS', 'S', 'ES'],
                    annot=True, fmt='.2f', cmap=cmap, vmin=0, vmax=1, cax=cax, cbar_label='Share of row')
ax2.set_xlabel('MC grade of M6')
ax2.set_ylabel('Reference grade')
panel(ax2, 'b')
finish(fig, 'fig_q2_grading')
