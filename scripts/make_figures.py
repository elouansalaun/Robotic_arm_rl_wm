"""Build the README figures from the CSV files in results/.

    python scripts/make_figures.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rwm.config import RESULTS_DIR
from rwm.evaluation import plots

FIG = RESULTS_DIR / "figures"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("->", FIG / name)


def sac_curves():
    rl = RESULTS_DIR / "rl"
    groups = {}
    for label, prefix in [("SAC + demonstrations", "sac_demo_s"), ("SAC from scratch", "sac_s")]:
        frames = [pd.read_csv(p) for p in sorted(rl.glob(f"{prefix}*_validation.csv"))]
        if frames:
            n = min(len(f) for f in frames)
            groups[label] = [f.iloc[:n] for f in frames]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plots.learning_curves(groups, ax)
    ax.set_title("SAC: success on 40 fixed validation scenarios", loc="left")
    save(fig, "sac_learning_curves.png")


def world_model():
    wm = RESULTS_DIR / "world_model"
    if not (wm / "multistep_errors.csv").exists():
        return
    errors = pd.read_csv(wm / "multistep_errors.csv")
    errors = errors[errors.contact.notna()]
    errors["contact"] = errors.contact.astype(str).str.lower() == "true"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    plots.multistep_error(errors, "obj_pos_mm", axes[0], "Object position")
    plots.multistep_error(errors, "tcp_pos_mm", axes[1], "Gripper (TCP) position")
    axes[0].set_ylabel("median error, mm (log scale)")
    axes[1].set_ylabel("")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("World model: open-loop prediction error on held-out episodes", x=0.06, y=1.04, ha="left",
                 fontsize=11.5, fontweight="bold")
    save(fig, "world_model_error.png")

    calib = pd.read_csv(wm / "uncertainty_vs_error.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4))
    plots.uncertainty_scatter(calib, ax)
    summary = json.loads((wm / "summary.json").read_text())
    ax.set_title(f"Ensemble disagreement vs error, 5 steps ahead\n"
                 f"(log-log correlation {summary['log_spread_error_correlation']:.2f})", loc="left")
    save(fig, "uncertainty_calibration.png")


def benchmark():
    path = RESULTS_DIR / "benchmark" / "ground_truth" / "summary.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path)
    conditions = [c for c in ["nominal", "new_positions", "new_objects", "heavy_slippery"] if c in set(summary.condition)]
    controllers = [c for c in ["scripted", "sac", "mpc", "mpc_unc"] if c in set(summary.controller)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    plots.success_bars(summary, conditions, controllers, ax)
    save(fig, "benchmark_success.png")


def camera_vs_ground_truth():
    gt_path = RESULTS_DIR / "benchmark" / "ground_truth" / "episodes.csv"
    cam_path = RESULTS_DIR / "benchmark" / "camera_realistic" / "episodes.csv"
    if not (gt_path.exists() and cam_path.exists()):
        return
    from rwm.evaluation.benchmark import summarise

    gt = summarise(pd.read_csv(gt_path))
    cam = summarise(pd.read_csv(cam_path))
    gt, cam = gt[gt.condition == "nominal"], cam[cam.condition == "nominal"]
    controllers = [c for c in ["scripted", "sac", "mpc", "mpc_unc"] if c in set(cam.controller)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(controllers))
    for offset, table, label, color in [(-0.2, gt, "ground-truth object state", "#86b6ef"),
                                        (0.2, cam, "camera (realistic noise)", "#1c5cab")]:
        rows = table.set_index("controller").reindex(controllers)
        y = rows.success_rate.values
        err = np.vstack([y - rows.success_ci_low.values, rows.success_ci_high.values - y])
        ax.bar(x + offset, y, 0.36, color=color, label=label, zorder=2)
        ax.errorbar(x + offset, y, yerr=err, fmt="none", ecolor=plots.INK2, elinewidth=1, capsize=2, zorder=3)
        for xi, v, top in zip(x + offset, y, rows.success_ci_high.values):
            ax.text(xi, top + 0.015, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=8, color=plots.INK2)
    ax.set_xticks(x, [plots.NAMES[c] for c in controllers])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("success rate, nominal (95% CI)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "camera_vs_ground_truth.png")


def perception():
    path = RESULTS_DIR / "perception" / "localisation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df.detected]
    cases = [("nominal", "clean"), ("nominal", "realistic"), ("nominal", "lighting"), ("nominal", "heavy"),
             ("new_objects", "realistic"), ("clutter", "realistic")]
    data = [1e3 * df[(df.condition == c) & (df.noise == n)].xy_error.values for c, n in cases]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.45, patch_artist=True,
                    medianprops=dict(color=plots.INK, linewidth=1.5))
    for box in bp["boxes"]:
        box.set(facecolor="#cde2fb", edgecolor="#3987e5", linewidth=1)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color="#3987e5", linewidth=1)
    ax.set_xticks(range(1, len(cases) + 1), [f"{n}\n{c.replace('_', ' ')}" for c, n in cases])
    ax.set_ylim(0, 6)
    clutter_p95 = np.percentile(data[-1], 95)
    ax.annotate(f"95th percentile: {clutter_p95:.0f} mm\n(object partly hidden)", xy=(len(cases), 5.9),
                xytext=(len(cases) - 1.6, 4.6), fontsize=8, color=plots.INK2,
                arrowprops=dict(arrowstyle="->", color=plots.MUTED, lw=1))
    ax.set_ylabel("object position error\nin the table plane (mm)")
    ax.set_title("Camera localisation error (box: quartiles, whiskers: 5-95 %)", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "perception_error.png")


def uncertainty_sweep():
    path = RESULTS_DIR / "benchmark" / "uncertainty_sweep" / "summary.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    conditions = [c for c in ["nominal", "new_objects", "new_positions", "heavy_slippery"] if c in set(s.condition)]
    for cond, color in zip(conditions, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]):
        g = s[s.condition == cond].sort_values("uncertainty_weight")
        x = np.arange(len(g))
        ax.plot(x, g.success_rate, color=color, marker="o", markersize=5, label=cond.replace("_", " "))
        ax.fill_between(x, g.success_ci_low, g.success_ci_high, color=color, alpha=0.1, linewidth=0)
        ax.set_xticks(x, [f"{w:g}" for w in g.uncertainty_weight])
    ax.set_xlabel("weight of the ensemble disagreement in the planning cost")
    ax.set_ylabel("MPC success rate")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.set_title("Effect of the uncertainty penalty on MPC", loc="left")
    save(fig, "uncertainty_sweep.png")


if __name__ == "__main__":
    plots.setup()
    sac_curves()
    world_model()
    benchmark()
    perception()
    camera_vs_ground_truth()
    uncertainty_sweep()
