# -*- coding: utf-8 -*-
"""Small, self-contained plotting toolkit used by the figure-generation scripts.

The original project imported these helpers from a workspace-level ``_utils``
package that was not part of the repository.  This file keeps the same public
names and signatures so every ``gen_fig_*.py`` script can be re-run from a
clean checkout.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


PALETTE = ["#2E5C8A", "#E8833A", "#C0392B", "#3E8E5A",
           "#7D5BA6", "#2A9D8F", "#D4A017", "#4A6FA5"]

COLORS = {
    "grid": "#D5D9E0",
    "ref_line": "#8A8A8A",
    "text": "#333333",
    "axis": "#4D4D4D",
    "muted": "#8A93A0",
    "bg": "#FFFFFF",
}


def _lighten(base, t):
    """Mix ``base`` towards white by fraction ``t`` (0 = base, 1 = white)."""
    t = float(np.clip(t, 0.0, 1.0))
    r, g, b = to_rgb(base)
    return (r + (1.0 - r) * t, g + (1.0 - g) * t, b + (1.0 - b) * t)


def setup_style(palette="nature"):
    """Set a compact, print-oriented Matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save_fig(fig, path, dpi=400, **kwargs):
    """Save a figure, accepting both absolute and repo-relative paths."""
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.02)
    fig.savefig(path, dpi=dpi, **kwargs)
    print("saved", path)


def nature_palette():
    """Return the shared qualitative palette used across the figures."""
    return {
        "blue_main": "#2E5C8A",
        "blue_secondary": "#6D96C4",
        "blue_light": "#BFD2E6",
        "teal": "#2A9D8F",
        "green_3": "#3E8E5A",
        "gold": "#D4A017",
        "red_strong": "#C0392B",
        "red_2": "#E8833A",
        "violet": "#7D5BA6",
        "neutral_black": "#1F1F1F",
        "neutral_dark": "#4D4D4D",
        "neutral_mid": "#8A93A0",
        "neutral_light": "#D9DDE3",
    }


def nature_markers():
    """Return a compact marker vocabulary for categorical series."""
    return {"o": "o", "s": "s", "D": "D", "^": "^", "v": "v", "x": "X"}


def set_paper_placement(fig, width_fraction=1.0):
    """Keep the requested width fraction without changing the figure geometry."""
    # The scripts already choose an explicit figsize in inches.  This helper is
    # deliberately a no-op so re-running them reproduces the same proportions.
    return fig


def auto_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        kwargs.setdefault("frameon", False)
        return ax.legend(handles, labels, **kwargs)
    return None


def shared_legend(fig, axes, where="top", ncol=3, **kwargs):
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    if not handles:
        return None
    if where == "top":
        return fig.legend(handles, labels, loc="upper center",
                          bbox_to_anchor=(0.5, 1.01), ncol=ncol,
                          frameon=False, **kwargs)
    return fig.legend(handles, labels, loc="lower center",
                      bbox_to_anchor=(0.5, -0.02), ncol=ncol,
                      frameon=False, **kwargs)


def consolidate_shared_legends(fig, axes=None, **kwargs):
    """Compatibility shim: legends are already consolidated by shared_legend."""
    return None


def uncertainty_band(ax, x, lo, hi, color="#6D96C4", alpha=0.22, label=None, **kwargs):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, label=label, **kwargs)
    return ax


def dynamic_limits(ax, x=None, y=None, pad=0.06, include_zero=False):
    """Set axis limits from data with a small symmetric pad."""
    if x is not None:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size:
            lo, hi = float(x.min()), float(x.max())
            if include_zero:
                lo, hi = min(lo, 0.0), max(hi, 0.0)
            span = max(hi - lo, 1e-12)
            ax.set_xlim(lo - pad * span, hi + pad * span)
    if y is not None:
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            lo, hi = float(y.min()), float(y.max())
            if include_zero:
                lo, hi = min(lo, 0.0), max(hi, 0.0)
            span = max(hi - lo, 1e-12)
            ax.set_ylim(lo - pad * span, hi + pad * span)
    return ax


def declutter_axes(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.7)
        s.set_color(COLORS["axis"])


def smart_labels(labels, max_len=14, sep="…"):
    out = []
    for s in labels:
        s = str(s)
        out.append(s if len(s) <= max_len else s[:max_len - 1] + sep)
    return out


def draw_vector_heatmap(ax, cm, xlabels=None, ylabels=None, annot="auto", fmt=".2f",
                        mask=None, cmap="Blues", vmin=None, vmax=None,
                        cax=None, cbar_label=None, **kwargs):
    """Draw a matrix as a vector-safe heatmap, optionally with cell labels."""
    cm = np.asarray(cm, dtype=float)
    if mask is not None:
        cm = np.ma.masked_where(mask, cm)
    im = ax.imshow(cm, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", **kwargs)
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels)
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    if annot:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if np.ma.is_masked(cm[i, j]) or not np.isfinite(cm[i, j]):
                    continue
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                        fontsize=5.2, color="0.15")
    if cax is not None:
        cb = ax.figure.colorbar(im, cax=cax)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=6)
    return im
