"""Figures for the README (matplotlib, static PNG)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# categorical order is fixed: a controller keeps its colour in every figure
COLORS = {"scripted": "#2a78d6", "sac": "#eb6834", "mpc": "#1baf7a", "mpc_unc": "#eda100"}
NAMES = {"scripted": "Scripted (IK)", "sac": "SAC", "mpc": "MPC", "mpc_unc": "MPC + unc. penalty"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # one-hue ramp for ordered levels (noise, horizon)


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "font.family": "sans-serif", "axes.axisbelow": True,
    })


def learning_curves(runs: dict[str, list[pd.DataFrame]], ax=None):
    """Validation success rate during SAC training, mean and min-max over seeds."""
    ax = ax or plt.gca()
    colors = ["#eb6834", "#898781"]
    for (label, frames), color in zip(runs.items(), colors):
        steps = frames[0].timestep.values
        y = np.stack([f.success_rate.values[: len(steps)] for f in frames])
        ax.fill_between(steps / 1e3, y.min(0), y.max(0), color=color, alpha=0.12, linewidth=0)
        ax.plot(steps / 1e3, y.mean(0), color=color, label=f"{label} ({len(frames)} seed{'s' * (len(frames) > 1)})")
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("validation success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    return ax


def success_bars(summary: pd.DataFrame, conditions: list[str], controllers: list[str], ax=None):
    ax = ax or plt.gca()
    n = len(controllers)
    width = 0.8 / n
    x = np.arange(len(conditions))
    for i, ctrl in enumerate(controllers):
        rows = summary[summary.controller == ctrl].set_index("condition").reindex(conditions)
        pos = x - 0.4 + width * (i + 0.5)
        y = rows.success_rate.values
        err = np.vstack([y - rows.success_ci_low.values, rows.success_ci_high.values - y])
        ax.bar(pos, y, width * 0.88, color=COLORS[ctrl], label=NAMES[ctrl], zorder=2)
        ax.errorbar(pos, y, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        for p, v in zip(pos, y):
            if np.isfinite(v):
                ax.text(p, v + 0.02, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [c.replace("_", " ") for c in conditions])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("success rate (95% Wilson CI)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=n, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    return ax


def multistep_error(errors: pd.DataFrame, feature: str, ax=None, title: str | None = None):
    """Median open-loop error vs horizon, free motion vs contact-rich windows."""
    ax = ax or plt.gca()
    d = errors[errors.feature == feature]
    for contact, color, name in [(False, "#2a78d6", "free motion"), (True, "#eb6834", "contact with the object")]:
        g = d[d.contact == contact].sort_values("horizon")
        ax.plot(g.horizon, g.model_median, color=color, marker="o", markersize=5, label=f"model, {name}")
        ax.plot(g.horizon, g.constant_median, color=color, linestyle=(0, (2, 2)), linewidth=1.2,
                label=f"'nothing moves' baseline, {name}")
    ax.set_xlabel("prediction horizon (steps of 50 ms)")
    ax.set_ylabel(feature.replace("_", " "))
    ax.set_xticks(sorted(d.horizon.unique()))
    if title:
        ax.set_title(title, loc="left")
    return ax


def uncertainty_scatter(calib: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    for contact, color, name in [(False, "#2a78d6", "free motion"), (True, "#eb6834", "contact")]:
        g = calib[calib.contact == contact]
        g = g.sample(min(len(g), 1500), random_state=0)
        ax.scatter(g.spread_mm + 1e-2, g.error_mm + 1e-2, s=8, color=color, alpha=0.35, linewidths=0, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ensemble spread on object position (mm)")
    ax.set_ylabel("actual error of the mean prediction (mm)")
    ax.legend(loc="upper left", markerscale=2)
    return ax
