"""Q3 area budget of Hengzhou for the two flood dates (dumbbell), left-right layout:
(a) guard cascade from the raw transferred model mask to the final new inundation A_new;
(b) water balance terms W1, W0, receded R, new area and net change."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 7.2,
    "axes.labelsize": 7.4,
    "legend.fontsize": 7,
})

C = {
    "red_strong": "#B64342",
    "neutral_mid": "#9A9A9A",
    "neutral_light": "#D9D9D9",
    "neutral_black": "#1A1A1A",
}


def load(name):
    with open(os.path.join(FIG, name), encoding="utf-8") as f:
        return json.load(f)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=2.5)
    ax.yaxis.set_minor_locator(plt.NullLocator())


q3 = load("problem_3_results.json")
d4, d16 = q3["dates"]["20260704"], q3["dates"]["20260716"]
cascade = [("raw_model", "Raw model"), ("after_W0", r"W$_0$ guard"), ("after_REM", "REM guard"),
           ("after_dVV", r"$\Delta$VV guard"), ("after_MMU", "MMU (final)")]
balance = [("A_W1", r"W$_1$ water"), ("A_W0", r"W$_0$ water"), ("A_R", "Receded"), ("A_new", "New area"),
           ("A_net", "Net change")]
for d in (d4, d16):
    assert abs(d["guard_stages_km2"]["after_MMU"] - d["A_new"]) < 1e-6
    assert d["A_net"] <= d["A_new"]

fig = plt.figure(figsize=(7.2, 3.6), layout="constrained")
fig.get_layout_engine().set(rect=(0.005, 0.005, 0.99, 0.99))
gs = fig.add_gridspec(2, 2, height_ratios=[0.08, 1], wspace=0.36)
c4, c16 = C["neutral_mid"], C["red_strong"]


def dumbbell(ax, items, getter, letter, right_label=False):
    y = np.arange(len(items))[::-1]
    a = np.array([getter(d4, k) for k, _ in items])
    b = np.array([getter(d16, k) for k, _ in items])
    for yi, u, v in zip(y, a, b):
        ax.plot([u, v], [yi, yi], color=C["neutral_light"], lw=3.0,
                solid_capstyle="round", zorder=1)
    ax.scatter(a, y, color=c4, s=24, zorder=3, marker="o", label="2026-07-04")
    ax.scatter(b, y, color=c16, s=24, zorder=4, marker="D", label="2026-07-16")
    ax.set_yticks(y, [n for _, n in items])
    ax.set_ylim(-0.55, len(items) - 0.45)
    ax.set_xlabel("Area inside Hengzhou (km$^2$)")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=9, pad=3)
    tw = ax.twinx()
    tw.set_ylim(ax.get_ylim())
    tw.set_yticks(y, [f"{u:.1f} / {v:.1f}" for u, v in zip(a, b)])
    tw.yaxis.set_minor_locator(plt.NullLocator())
    tw.tick_params(axis="y", labelsize=6.2, length=0, pad=2)
    if right_label:
        tw.set_ylabel("07-04 / 07-16 (km$^2$)", fontsize=6.6, labelpad=3)
    tw.spines["top"].set_visible(False)
    tw.spines["right"].set_linewidth(0.5)
    tw.spines["right"].set_color("#777777")
    return np.r_[a, b]


ax = fig.add_subplot(gs[1, 0])
v = dumbbell(ax, cascade, lambda d, k: d["guard_stages_km2"][k], "a")
ax.set_xlim(0, v.max() * 1.07)
style(ax)

ax2 = fig.add_subplot(gs[1, 1])
v2 = dumbbell(ax2, balance, lambda d, k: d[k], "b", right_label=True)
ax2.axvline(0, color=C["neutral_black"], lw=0.8)
ax2.set_xlim(v2.min() - 12, v2.max() * 1.06)
style(ax2)

lax = fig.add_subplot(gs[0, :])
lax.set_axis_off()
lax.legend(*ax.get_legend_handles_labels(), loc="center", ncol=2, frameon=False,
           handlelength=1.2, columnspacing=1.5)

fig.savefig(os.path.join(FIG, "fig_q3_area_budget.pdf"), dpi=300, bbox_inches="tight",
            pad_inches=0.02)
print("wrote", os.path.join(FIG, "fig_q3_area_budget.pdf"))
