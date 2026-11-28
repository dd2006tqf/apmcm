"""Nature-style redraw of the 50 px MMU figures (Python/matplotlib, skill palette)."""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures")
SKILL = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Doubao", "User Data", "Default",
                     ".doubao", "agent_mode", "workspace", ".user_skills", "nature-figure", "scripts")
sys.path.insert(0, SKILL)
from audit_panel_alignment import require_matplotlib_panel_alignment

E13 = os.path.join(ROOT, "experiments", "results", "e13_q1_mmu_grid.json")
E10 = os.path.join(ROOT, "experiments", "results", "e10_mmu_fairness.json")

# Nature-figure palette (skill `references/api.md`)
BLUE = "#0F4D92"; BLUE_LIGHT = "#B4C0E4"; BLUE_MID = "#3775BA"
GRAY = "#767676"; GRAY_LIGHT = "#D8D8D8"; DARK = "#272727"; RED = "#B64342"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})


def panel_label(ax, letter):
    ax.text(-0.02, 1.07, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=DARK, clip_on=False)


def q1_fig():
    d = json.load(open(E13, encoding="utf-8"))
    rows = d["rows"]
    th = [0, 10, 25, 50, 100]
    A = {r["mmu"]: r for r in rows if r["src"] == "A"}
    B = {r["mmu"]: r for r in rows if r["src"] == "B"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.56), constrained_layout=True)

    for ax, key, lab in [(axes[0], "iou", "IoU"), (axes[1], "rae", "Signed RAE")]:
        ya = [A[t][key] for t in th]; yb = [B[t][key] for t in th]
        ax.plot(th, ya, color=GRAY, marker="o", ms=4, lw=1.1, ls="--", zorder=2)
        ax.plot(th, yb, color=BLUE, marker="s", ms=4.2, lw=1.5, zorder=3)
        ax.axvline(50, color=RED, lw=0.9, ls=":", zorder=1)
        ax.set_xlabel("Minimum mapping unit (px)")
        ax.set_ylabel(lab)
        ax.set_xticks(th)
        ax.set_xlim(-4, 112)
        ax.grid(alpha=0.22, lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if key == "rae":
            ax.axhline(0, color=DARK, lw=0.5, alpha=0.55, zorder=1)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylim(-0.15, 1.55)
        else:
            ax.set_ylim(0.24, 0.66)
        ax.text(104, ya[-1], "A", color=DARK, fontsize=8, va="center", ha="center")
        ax.text(104, yb[-1], "B", color=DARK, fontsize=7, fontweight="bold", va="center", ha="center")
        panel_label(ax, "a" if key == "iou" else "b")

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=os.path.join(FIG, "fig_q1_mmu.alignment.json"),
        overlay_svg=os.path.join(FIG, "fig_q1_mmu.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        row_groups=[["a", "b"]], strict=True)
    fig.savefig(os.path.join(FIG, "fig_q1_mmu.svg"))
    fig.savefig(os.path.join(FIG, "fig_q1_mmu.pdf"))
    fig.savefig(os.path.join(FIG, "fig_q1_mmu.png"), dpi=300)
    fig.savefig(os.path.join(FIG, "fig_q1_mmu.tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def q2_fig():
    d = json.load(open(E10, encoding="utf-8"))
    R = {r["name"]: r for r in d["rows"]}
    names = ["Q2_M2", "Q2_M2_mmu50", "Q2_M6", "Q2_M6_mmu50"]
    labels = ["M2", "M2+MMU", "M6", "M6+MMU"]
    colors = [BLUE_LIGHT, BLUE, GRAY_LIGHT, GRAY]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.49), constrained_layout=True)
    specs = [("iou", "IoU", "All tiles", 0.0, 1.0, "a", 0),
             ("rae", "|RAE|", "", 0.0, 0.040, "b", 1),
             ("iou_hard", "IoU", "397 hard tiles", 0.0, 1.0, "c", 0),
             ("rae_hard", "|RAE|", "", 0.0, 0.095, "d", 1)]
    for ax, key, ylab, title, lo, hi, letter, col in zip(axes.ravel(),
                                                         [s[0] for s in specs], [s[1] for s in specs],
                                                         [s[2] for s in specs], [s[3] for s in specs],
                                                         [s[4] for s in specs], [s[5] for s in specs],
                                                         [s[6] for s in specs]):
        vals = [abs(R[n][key]) for n in names]
        ax.bar(range(4), vals, color=colors, width=0.64, edgecolor=DARK, lw=0.5, zorder=2)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylab, labelpad=6)
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", alpha=0.18, lw=0.4, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if key.startswith("rae"):
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if title:
            ax.set_title(title, fontsize=8, pad=3, color=DARK)
        if key == "rae" and vals[3] < 0.0005:
            ax.text(3, hi * 0.02, "≈0", ha="center", va="bottom", fontsize=7.5, color=DARK)
        panel_label(ax, letter)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=os.path.join(FIG, "fig_q2_mmu.alignment.json"),
        overlay_svg=os.path.join(FIG, "fig_q2_mmu.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]], strict=True)
    fig.savefig(os.path.join(FIG, "fig_q2_mmu.svg"))
    fig.savefig(os.path.join(FIG, "fig_q2_mmu.pdf"))
    fig.savefig(os.path.join(FIG, "fig_q2_mmu.png"), dpi=300)
    fig.savefig(os.path.join(FIG, "fig_q2_mmu.tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    q1_fig()
    q2_fig()

