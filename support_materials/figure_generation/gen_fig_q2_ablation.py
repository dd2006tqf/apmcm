"""Q2 ablation as a step-wise dumbbell cascade (M1 -> M6) for IoU, MAE_A and |RAE|."""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures")
SKILL = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Doubao", "User Data", "Default",
                     ".doubao", "agent_mode", "workspace", ".user_skills", "nature-figure", "scripts")
sys.path.insert(0, SKILL)
from audit_panel_alignment import require_matplotlib_panel_alignment

DATA = os.path.join(FIG, "problem_2_results.json")

GRAY = "#767676"; DARK = "#272727"
GREEN = "#8BCF8B"; GREEN_EDGE = "#3C7A4E"
RED = "#B64342"; RED_FILL = "#E6B0AF"
ARROW = "#9A9A9A"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 7.8,
})


def fmt(delta, key):
    if abs(delta) < 5e-4:
        return "\u00b1<0.001"
    if key == "RAE":
        return f"{delta:+.1f}"
    return f"{delta:+.3f}"


def draw_panel(ax, key, title, rows, ylabels, letter, formatter, m0):
    better = 1 if key == "IoU" else -1
    n = len(rows)
    for i, row in enumerate(rows):
        y = n - 1 - i
        start, end = row["prev"], row["curr"]
        good = (end - start) * better > 0
        fill = GREEN if good else RED_FILL
        edge = GREEN_EDGE if good else RED
        ax.annotate("", xy=(end, y), xytext=(start, y),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=0.8,
                                    shrinkA=2.4, shrinkB=3.0, mutation_scale=6))
        ax.plot(start, y, marker="o", mfc="white", mec=GRAY, mew=0.8, ms=4.2, zorder=3)
        ax.plot(end, y, marker="o", mfc=fill, mec=edge, mew=0.9, ms=5.0, zorder=4)
        ax.text((start + end) / 2, y + 0.24, fmt(end - start, key),
                ha="center", va="bottom", fontsize=7.5, color=DARK, zorder=5)

    ax.axvline(m0, color=DARK, lw=0.8, ls="--", zorder=0)
    ax.set_ylim(-0.55, n - 0.05)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ylabels if letter == "a" else [])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_title(title, fontsize=8.5, pad=2, color=DARK)
    ax.xaxis.set_major_formatter(formatter)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.text(0.01, 0.99, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="bold", color=DARK, zorder=5)


def main():
    d = json.load(open(DATA, encoding="utf-8"))["table"]
    seq = ["M1", "M2", "M3", "M4", "M5", "M6"]
    ylabels = ["M5\u2192M6\n+isotonic", "M4\u2192M5\n+MC", "M3\u2192M4\n+modality",
               "M2\u2192M3\n+area", "M1\u2192M2\n+optical"]
    specs = [("IoU", "IoU", "IoU", 1.0, 0.50, 0.82),
             ("MAE_A", "MAE (km\u00b2/tile)", "MAE_A", 1.0, 0.12, 0.55),
             ("RAE", "|RAE| (%)", "RAE", 100.0, 0.0, 38.0)]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=False)
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.18, top=0.86, wspace=0.10)
    for ax, (key, title, field, scale, lo, hi), letter in zip(axes, specs, "abc"):
        vals = [d[m][field] * scale for m in seq]
        m0 = d["M0"][field] * scale
        rows = [{"prev": vals[i], "curr": vals[i + 1]} for i in range(len(vals) - 1)]
        if key == "RAE":
            formatter = FormatStrFormatter("%.0f")
        else:
            formatter = FormatStrFormatter("%.2f")
        draw_panel(ax, key, title, rows, ylabels, letter, formatter, m0)
        ax.set_xlim(lo, hi)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=os.path.join(FIG, "fig_q2_ablation.alignment.json"),
        overlay_svg=os.path.join(FIG, "fig_q2_ablation.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        row_groups=[["a", "b", "c"]], strict=True)
    fig.savefig(os.path.join(FIG, "fig_q2_ablation.svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(FIG, "fig_q2_ablation.pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(FIG, "fig_q2_ablation.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(FIG, "fig_q2_ablation.tiff"), dpi=600,
                bbox_inches="tight", pad_inches=0.03,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()


