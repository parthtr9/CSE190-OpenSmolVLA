"""Generate line charts comparing RECAP ablation success rates across 8 iterations."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent
OUT_FILE = RESULTS_DIR / "recap_ablation_results.png"

CONFIGS = {
    "baseline":    {"label": "BC Baseline (50%)",    "color": "#555555", "ls": "--",  "lw": 1.8, "marker": None, "zorder": 2},
    "sparse_noKL": {"label": "Sparse, no KL",        "color": "#D62728", "ls": "-",   "lw": 2.2, "marker": "o",  "zorder": 4},
    "dense_noKL":  {"label": "Dense, no KL",         "color": "#2CA02C", "ls": "-",   "lw": 2.2, "marker": "s",  "zorder": 4},
    "sparse_KL":   {"label": "Sparse + KL anchor",   "color": "#1F77B4", "ls": "-",   "lw": 2.2, "marker": "^",  "zorder": 4},
    "dense_KL":    {"label": "Dense + KL anchor",    "color": "#FF7F0E", "ls": "-",   "lw": 2.2, "marker": "D",  "zorder": 4},
}

# Per-panel y-axis limits — pos0 starts at 100% so needs room
YLIMS = {
    "overall":   (0, 72),
    "cube_pos0": (0, 110),
    "cube_pos1": (0, 72),
    "cube_pos2": (0, 62),
}

PANEL_TITLES = {
    "overall":   "Overall Success Rate",
    "cube_pos0": "Cube Position 0  (-0.100, -0.050)  [easiest]",
    "cube_pos1": "Cube Position 1  (-0.130, -0.050)  [medium]",
    "cube_pos2": "Cube Position 2  (-0.100, -0.020)  [hardest]",
}

# BC baseline value per panel (to draw horizontal reference)
BASELINES = {
    "overall": 50.0,
    "cube_pos0": 100.0,
    "cube_pos1": 30.0,
    "cube_pos2": 20.0,
}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"{name}.csv")


def plot_panel(ax: plt.Axes, df: pd.DataFrame, key: str) -> None:
    iters = df["iteration"].values
    ymin, ymax = YLIMS[key]
    bc_val = BASELINES[key]

    # Shaded BC baseline band
    ax.axhline(bc_val, color=CONFIGS["baseline"]["color"],
               ls=CONFIGS["baseline"]["ls"], lw=CONFIGS["baseline"]["lw"],
               label=CONFIGS["baseline"]["label"], zorder=2)
    ax.axhspan(bc_val - 1, bc_val + 1, color="#555555", alpha=0.08, zorder=1)

    for col, cfg in CONFIGS.items():
        if col == "baseline" or col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        y = vals.copy()
        # iteration 0 row has the shared BC starting value
        if np.isnan(y[0]):
            y[0] = bc_val

        ax.plot(iters, y,
                color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"],
                marker=cfg["marker"], markersize=6.5, markeredgewidth=0.8,
                markeredgecolor="white",
                label=cfg["label"], zorder=cfg["zorder"])

    ax.set_title(PANEL_TITLES[key], fontsize=10.5, fontweight="bold", pad=7)
    ax.set_xlabel("RECAP Iteration", fontsize=9, labelpad=4)
    ax.set_ylabel("Success Rate (%)", fontsize=9, labelpad=4)
    ax.set_xticks(range(9))
    ax.set_xlim(-0.4, 8.4)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.grid(axis="y", alpha=0.2, lw=0.8, color="#444444")
    ax.grid(axis="x", alpha=0.1, lw=0.6, color="#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


# ---------------------------------------------------------------------------
# Build figure: 2×2 panels
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.88, bottom=0.15, left=0.07, right=0.97)

fig.suptitle(
    "RECAP Ablation — SmolVLA SO-101 Cube-Bin Task\n"
    "Success Rate across 8 Fine-tuning Iterations  (3-position cube randomization)",
    fontsize=13, fontweight="bold", y=0.95,
)

panel_keys = ["overall", "cube_pos0", "cube_pos1", "cube_pos2"]
for ax, key in zip(axes.flat, panel_keys):
    plot_panel(ax, load(key), key)

# ---------------------------------------------------------------------------
# Shared legend anchored below the figure
# ---------------------------------------------------------------------------
handles = []
for col, cfg in CONFIGS.items():
    if cfg["marker"]:
        h = plt.Line2D([0], [0], color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"],
                       marker=cfg["marker"], markersize=8,
                       markeredgecolor="white", markeredgewidth=0.8,
                       label=cfg["label"])
    else:
        h = plt.Line2D([0], [0], color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"],
                       label=cfg["label"])
    handles.append(h)

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=5,
    fontsize=9.5,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, 0.02),
    title="Reward Config",
    title_fontsize=9.5,
)

fig.savefig(OUT_FILE, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_FILE}")
plt.show()
