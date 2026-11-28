"""Label-semantics test (P1-C7) as ridgelines over all 65.5 M valid pixels of the 1000 tiles:
pre-flood NDWI and post-flood SAR brightness for (i) label 1, (ii) label 0 on pre-flood water,
(iii) label 0 on pre-flood land."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figcommon import *
sys.path.insert(0, os.path.join(ROOT, 'code'))
from common import FI, FEATURES

q1 = load('problem_1_results.json')
ls = q1['label_semantics']['ndwi>0']
NT, H, NF = 1000, 256, len(FEATURES)
cache = os.path.join(ROOT, '_tmp', 'fig_cache', 'label_semantics_hist.npz')
e_ndwi = np.linspace(-1, 1, 161)
e_sar = np.linspace(0, 1, 101)
if os.path.isfile(cache):
    z = np.load(cache)
    h_ndwi, h_sar = z['h_ndwi'], z['h_sar']
else:
    feats = np.memmap(os.path.join(CACHE_Q1, 'feats.f16'), dtype=np.float16, mode='r', shape=(NT, NF, H, H))
    valid = np.memmap(os.path.join(CACHE_Q1, 'valid.u8'), dtype=np.uint8, mode='r', shape=(NT, H, H))
    y = np.memmap(os.path.join(CACHE_Q1, 'y.u8'), dtype=np.uint8, mode='r', shape=(NT, H, H))
    h_ndwi = np.zeros((3, len(e_ndwi) - 1))
    h_sar = np.zeros((3, len(e_sar) - 1))
    for t in range(NT):            # every tile, every valid pixel
        v = valid[t].astype(bool)
        nd = feats[t, FI['ndwi']].astype(np.float32)[v]
        zt = feats[t, FI['zt']].astype(np.float32)[v]
        yy = y[t][v]
        groups = [yy == 1, (yy == 0) & (nd > 0), (yy == 0) & (nd <= 0)]
        for g, m in enumerate(groups):
            h_ndwi[g] += np.histogram(nd[m], e_ndwi)[0]
            h_sar[g] += np.histogram(zt[m], e_sar)[0]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, h_ndwi=h_ndwi, h_sar=h_sar)
n_total = h_ndwi.sum()
assert abs(n_total - ls['n_valid_px']) < 1, (n_total, ls['n_valid_px'])

names = ['Label 1 (reference $F$)', 'Label 0, pre-flood water', 'Label 0, pre-flood land']
cols = [C['blue_main'], C['teal'], C['neutral_mid']]
fig = plt.figure(figsize=(6.4, 3.0), layout='constrained')
set_paper_placement(fig, width_fraction=0.95)
gs = fig.add_gridspec(1, 2)
for k, (edges, hist, xlabel, letter) in enumerate([
        (e_ndwi, h_ndwi, 'Pre-flood NDWI', 'a'),
        (e_sar, h_sar, 'Post-flood SAR, rank-normalized $\\tilde z$', 'b')]):
    ax = fig.add_subplot(gs[0, k])
    xc = 0.5 * (edges[1:] + edges[:-1])
    step = 1.0
    for g in range(3):
        dens = hist[g] / hist[g].sum() / np.diff(edges)
        dens = dens / dens.max() * 0.9
        base = (2 - g) * step
        ax.fill_between(xc, base, base + dens, color=cols[g], alpha=0.35, linewidth=0)
        ax.plot(xc, base + dens, color=cols[g], lw=1.2, label=names[g] if k == 0 else None)
        ax.axhline(base, color=C['neutral_light'], lw=0.6, zorder=0)
    if k == 0:
        ax.axvline(0, color=C['neutral_dark'], lw=0.8, ls='--')
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(['Label 1', 'L0 water', 'L0 land'] if k == 0 else [])
    ax.set_ylim(-0.05, 3.3)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(xlabel)
    if k == 0:
        ax.set_ylabel('Pixel group (scaled density)')
    panel(ax, letter)
shared_legend(fig, fig.axes, where='top', ncol=3)
finish(fig, 'fig_q1_label_semantics')
