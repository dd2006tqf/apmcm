"""Q3 area budget, Nature-style asymmetric layout.

Panel a: guard cascade as a compact staircase for both flood dates; panel b: the
16 July water balance as a waterfall. The figure answers one question: how the
raw transferred mask becomes the final new-inundation area, and why the net
change is negative even though new water is positive.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
QA = os.path.join(ROOT, "_tmp", "qa")
os.makedirs(QA, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 8,
    "axes.labelsize": 8,
})

# Nature-family restrained palette: neutral context, one blue signal family,
# red reserved for recession (loss) and teal for new water (gain).
C = {
    "blue_main": "#0F4D92",
    "blue_soft": "#C9D7EA",
    "red_strong": "#B64342",
    "red_soft": "#E9A6A1",
    "teal": "#42949E",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}


def load(name):
    with open(os.path.join(FIG, name), encoding="utf-8") as f:
        return json.load(f)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.7, pad=2.5)
    ax.yaxis.set_minor_locator(plt.NullLocator())


def panel_label(ax, letter):
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, fontweight="bold", color=C["neutral_black"], clip_on=False)


q3 = load("problem_3_results.json")
d4, d16 = q3["dates"]["20260704"], q3["dates"]["20260716"]
cascade = [("raw_model", "Raw\nmodel"), ("after_W0", "W0\nguard"),
           ("after_REM", "REM\nguard"), ("after_dVV", "ΔVV\nguard"),
           ("after_MMU", "MMU\nfinal")]
for d in (d4, d16):
    assert abs(d["guard_stages_km2"]["after_MMU"] - d["A_new"]) < 1e-6
    assert d["A_net"] <= d["A_new"]

fig = plt.figure(figsize=(7.2, 4.05), layout="constrained")
fig.get_layout_engine().set(rect=(0.02, 0.02, 0.97, 0.96))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.30)

# ---------------------------------------------------------------- panel a: guard cascade
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(cascade))
y16 = np.array([d16["guard_stages_km2"][k] for k, _ in cascade])
y4 = np.array([d4["guard_stages_km2"][k] for k, _ in cascade])

ax.fill_between(x, y16, 0, step="mid", color=C["blue_soft"], alpha=0.18, lw=0, zorder=1)
ax.step(x, y16, where="mid", color=C["blue_main"], lw=1.5, zorder=3)
ax.step(x, y4, where="mid", color=C["neutral_mid"], lw=0.9, ls=(0, (3, 2)), zorder=2)
ax.plot(x, y16, "o", ms=4.1, mfc=C["blue_main"], mec="white", mew=0.7, zorder=4)
ax.plot(x, y4, "o", ms=3.2, mfc="white", mec=C["neutral_mid"], mew=0.8, zorder=4)

# annotate the three largest removals on the 16 July path
for i in (0, 1, 3):
    drop = y16[i] - y16[i + 1]
    if i == 3:
        xlab, halign, ylab, fs = 3.95, "right", 31.6, 6.4
    else:
        xlab, halign, ylab, fs = i + 0.60, "left", (y16[i] + y16[i + 1]) / 2 - 1.5, 7.0
    ax.text(xlab, ylab, f"−{drop:.1f}", ha=halign, va="center", fontsize=fs,
            color=C["red_strong"] if i == 0 else C["neutral_dark"], zorder=5)

ax.text(0.97, 0.98, "16 Jul: 82.6% removed", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.2, fontweight="bold",
        color=C["blue_main"])

ax.set_xticks(x, [name for _, name in cascade])
ax.set_xlim(-0.45, 4.95)
ax.set_ylim(0, 166)
ax.set_yticks([0, 50, 100, 150])
ax.set_ylabel("Remaining area (km²)", labelpad=3)
ax.set_xlabel("Guard stage", labelpad=3)
style(ax)
panel_label(ax, "a")

ax.text(4.22, 35.5, "16 Jul\n26.9", ha="left", va="center", fontsize=7.6,
        color=C["blue_main"], linespacing=1.15, clip_on=False)
ax.text(4.22, 16.0, "4 Jul\n24.5", ha="left", va="center", fontsize=7.0,
        color=C["neutral_mid"], linespacing=1.15, clip_on=False)

# ---------------------------------------------------------------- panel b: 16 July water balance
ax2 = fig.add_subplot(gs[0, 1])
w0 = d16["A_W0"]
w1 = d16["A_W1"]
rec = d16["A_R"]
new = d16["A_new"]
bottom_rec = w0 - rec
net = w1 - w0

terms = [0, 1, 2, 3]
ax2.bar(0, w0, width=0.62, color=C["neutral_light"], edgecolor=C["neutral_mid"], lw=0.6, zorder=2)
ax2.bar(1, rec, bottom=bottom_rec, width=0.62, color=C["red_soft"], edgecolor=C["red_strong"], lw=0.6, zorder=3)
ax2.bar(2, new, bottom=bottom_rec, width=0.62, color=C["teal"], edgecolor=C["teal"], lw=0.6, zorder=3)
ax2.bar(3, w1, width=0.62, color=C["blue_main"], edgecolor=C["blue_main"], lw=0.6, zorder=2)

# dotted connectors make the waterfall reading explicit without a legend
ax2.plot([0.31, 0.69], [w0, w0], color=C["neutral_mid"], lw=0.6, ls=":", zorder=4)
ax2.plot([1.31, 1.69], [bottom_rec, bottom_rec], color=C["neutral_mid"], lw=0.6, ls=":", zorder=4)
ax2.plot([2.31, 2.69], [w1, w1], color=C["neutral_mid"], lw=0.6, ls=":", zorder=4)

ax2.text(0, w0 + 6, f"{w0:.1f}", ha="center", va="bottom", fontsize=7.4, color=C["neutral_dark"])
ax2.text(1, bottom_rec + rec * 0.50, f"−{rec:.1f}", ha="center", va="center",
         fontsize=7.4, color=C["red_strong"])
ax2.text(2, bottom_rec + new * 0.50, f"+{new:.1f}", ha="center", va="center",
         fontsize=7.0, color=C["neutral_black"])
ax2.text(3, w1 + 6, f"{w1:.1f}", ha="center", va="bottom", fontsize=7.4, color=C["blue_main"])

# net change bracket, kept outside the bars
ax2.annotate("", xy=(3.55, w0), xytext=(3.55, w1),
             arrowprops=dict(arrowstyle="<->", color=C["neutral_dark"], lw=0.7,
                             shrinkA=0, shrinkB=0), zorder=5)
ax2.text(3.64, (w0 + w1) / 2, f"net\n−{abs(net):.1f}", ha="left", va="center",
         fontsize=7.0, color=C["neutral_dark"], linespacing=1.15, clip_on=False)

ax2.text(0.98, 0.98, "16 Jul", transform=ax2.transAxes, ha="right", va="top",
         fontsize=7.2, color=C["blue_main"])
ax2.set_xticks(terms, ["W0\nwater", "Receded", "New\nwater", "W1\nwater"])
ax2.set_xlim(-0.48, 4.22)
ax2.set_ylim(0, 320)
ax2.set_yticks([0, 100, 200, 300])
ax2.set_ylabel("Water area (km²)", labelpad=3)
ax2.set_xlabel("Water-balance term", labelpad=3)
style(ax2)
panel_label(ax2, "b")

# Optional render-time alignment gate: set NATURE_FIGURE_SCRIPTS to the skill's
# scripts directory before running. The gate is not run without that opt-in.
scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
if scripts and os.path.isdir(scripts):
    sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig,
        json_out=os.path.join(QA, "fig_q3_area_budget.alignment.json"),
        overlay_svg=os.path.join(QA, "fig_q3_area_budget.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        exemptions=[{"panels": ["a"], "checks": ["panel-width"],
                     "reason": "a is the hero guard-cascade panel; b is the compact water-balance panel"}],
    )
    print("alignment gate: PASS")

fig.savefig(os.path.join(FIG, "fig_q3_area_budget.pdf"), dpi=300, bbox_inches="tight",
            pad_inches=0.02)
fig.savefig(os.path.join(QA, "fig_q3_area_budget.tiff"), dpi=600, bbox_inches="tight",
            pad_inches=0.02)
print("wrote", os.path.join(FIG, "fig_q3_area_budget.pdf"))
