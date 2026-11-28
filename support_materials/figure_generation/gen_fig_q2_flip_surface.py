"""Q2 figure: probability that a unit changes grade, as a function of true flood
proportion p and area-error scale sigma.  Colours follow the paper palette."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import norm as snorm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'output')
FIG = os.path.join(ROOT, 'figures')

BLUE = '#4284B6'
RED = '#BD564E'
BLACK = '#343534'
GOLD = '#EFC632'

with open(os.path.join(OUT, 'q2_results.json'), encoding='utf-8') as f:
    q2 = json.load(f)
TH = np.array(q2['grading']['theta'])          # 0.0593, 0.5298, 0.9689
TILE = 4.9217536

P = np.linspace(0.0, 1.0, 241)
S = np.linspace(0.005, 0.30, 90)
PP, SS = np.meshgrid(P, S)
cdf = snorm.cdf
# probability of staying in the reference grade, then one minus it
stay = np.select(
    [PP < TH[0], PP < TH[1], PP < TH[2]],
    [cdf((TH[0] - PP) / SS),
     cdf((TH[1] - PP) / SS) - cdf((TH[0] - PP) / SS),
     cdf((TH[2] - PP) / SS) - cdf((TH[1] - PP) / SS)],
    default=1.0 - cdf((TH[2] - PP) / SS))
Z = 1.0 - stay

# empirical bins of the 1000 test tiles, sigma = std of the signed area error
Aref = np.array(q2['A_ref_t'])
Ahat = np.array(q2['per_tile']['M6']['A_hat'])
p_hat = Ahat / TILE
delta = p_hat - Aref / TILE
bins = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)]
pts = []
for lo, hi in bins:
    m = (p_hat >= lo) & (p_hat < hi)
    pts.append((p_hat[m].mean(), delta[m].std(), m.sum()))

cmap = LinearSegmentedColormap.from_list(
    'paper', ['#F4F7FA', '#CFE0EC', '#9DC3DA', BLUE, '#195C8F', '#0E3A5C'])
norm = Normalize(0, 0.55)
face = cmap(norm(Z))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.linewidth': 0.5})


def render(az, el, path, show_zl=True):
    fig = plt.figure(figsize=(4.3, 3.1))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d')
    ax.plot_surface(PP, SS, Z, facecolors=face, rstride=2, cstride=2,
                    linewidth=0, antialiased=False, shade=False, rasterized=True)
    ax.plot_wireframe(PP, SS, Z, rstride=8, cstride=8, color='white',
                      linewidth=0.2, alpha=0.38)
    for (p0, s0, n) in pts:
        if s0 > S[-1] + 1e-9:
            continue
        if p0 < TH[0]:
            stay0 = snorm.cdf((TH[0] - p0) / s0)
        elif p0 < TH[1]:
            stay0 = snorm.cdf((TH[1] - p0) / s0) - snorm.cdf((TH[0] - p0) / s0)
        elif p0 < TH[2]:
            stay0 = snorm.cdf((TH[2] - p0) / s0) - snorm.cdf((TH[1] - p0) / s0)
        else:
            stay0 = 1 - snorm.cdf((TH[2] - p0) / s0)
        z0 = 1 - stay0
        ax.plot([p0, p0], [s0, s0], [0, z0 + 0.035], color=BLACK, lw=0.7, ls=(0, (2, 1.5)))
        ax.scatter([p0], [s0], [0], color=BLACK, s=14, depthshade=False)
        ax.scatter([p0], [s0], [z0 + 0.035], color=RED, s=55, edgecolor='white',
                   linewidth=0.4, depthshade=False, zorder=10)
    ax.view_init(elev=el, azim=az)
    ax.set_box_aspect((1.0, 0.62, 0.40), zoom=1.30)
    ax.set_xticks([0, 0.3, 0.6, 0.9])
    ax.set_yticks([0.0, 0.15, 0.30])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, 0.305)
    ax.set_zticks([0.0, 0.2, 0.4, 0.6])
    ax.set_zlim(0, 0.60)
    ax.tick_params(pad=-1, labelsize=8.5)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color('#C9C9C9')
        axis.line.set_linewidth(0.4)
    ax.set_xlabel('True proportion $p$', labelpad=-2, fontsize=10)
    ax.set_ylabel('Error scale $\\sigma$', labelpad=-3, fontsize=10)
    if show_zl:
        ax.set_zlabel('Change probability', labelpad=-4, fontsize=10)
    else:
        ax.zaxis.set_ticklabels([])
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def trim(path):
    from PIL import Image, ImageChops
    im = Image.open(path).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    bb = ImageChops.difference(im, bg).getbbox()
    if bb:
        pad = 6
        im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
                 min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))).save(path)


ta = os.path.join(FIG, '_flip_a.png')
tb = os.path.join(FIG, '_flip_b.png')
render(-58, 46, ta, True)
render(124, 42, tb, False)
trim(ta); trim(tb)

fig = plt.figure(figsize=(7.3, 2.9))
for k, p in enumerate([ta, tb]):
    im = plt.imread(p)
    wfrac = 0.425
    hfrac = wfrac * 7.3 * im.shape[0] / im.shape[1] / 2.9
    bottom = 0.96 - hfrac
    ax = fig.add_axes([0.008 + 0.44 * k, bottom, wfrac, hfrac])
    ax.imshow(im); ax.axis('off')
    ax.text(0.02, 0.99, f'({chr(65 + k)})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

cax = fig.add_axes([0.895, 0.24, 0.011, 0.64])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label('Probability of grade change', fontsize=9)
cb.ax.tick_params(labelsize=8.5, length=2, width=0.4, pad=2)
cb.outline.set_linewidth(0.4)

fig.savefig(os.path.join(FIG, 'fig_q2_flip_surface.pdf'), dpi=300)
fig.savefig(os.path.join(FIG, 'fig_q2_flip_surface.png'), dpi=200)
print('theta', TH, 'bins', [(round(a, 3), round(b, 3), n) for a, b, n in pts])
