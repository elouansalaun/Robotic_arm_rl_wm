"""Markdown tables for the README, generated from the CSV files in results/.

    python scripts/make_tables.py        # -> results/tables.md
"""

import json

import pandas as pd

from rwm.config import RESULTS_DIR
from rwm.evaluation.benchmark import summarise

NAMES = {"scripted": "Scripted (IK)", "sac": "SAC", "mpc": "MPC", "mpc_unc": "MPC + uncertainty penalty"}
CONDITIONS = ["nominal", "new_positions", "new_objects", "heavy_slippery", "clutter"]


def pct(x):
    return f"{100 * x:.0f}"


def success_cell(row):
    return f"{pct(row.success_rate)} [{pct(row.success_ci_low)}-{pct(row.success_ci_high)}]"


def benchmark_table(tag: str) -> str:
    path = RESULTS_DIR / "benchmark" / tag / "episodes.csv"
    if not path.exists():
        return ""
    s = summarise(pd.read_csv(path))
    conditions = [c for c in CONDITIONS if c in set(s.condition)]
    controllers = [c for c in NAMES if c in set(s.controller)]
    lines = ["| controller | " + " | ".join(c.replace("_", " ") for c in conditions) + " |",
             "|---|" + "---|" * len(conditions)]
    for ctrl in controllers:
        cells = []
        for cond in conditions:
            r = s[(s.controller == ctrl) & (s.condition == cond)]
            cells.append(success_cell(r.iloc[0]) if len(r) else "-")
        lines.append(f"| {NAMES[ctrl]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def details_table(tag: str, condition: str = "nominal") -> str:
    path = RESULTS_DIR / "benchmark" / tag / "episodes.csv"
    if not path.exists():
        return ""
    s = summarise(pd.read_csv(path))
    s = s[s.condition == condition]
    lines = ["| controller | success | placement error (mm, median) | steps to success | steps with a collision | "
             "decision time (ms) |", "|---|---|---|---|---|---|"]
    for ctrl in NAMES:
        r = s[s.controller == ctrl]
        if not len(r):
            continue
        r = r.iloc[0]
        lines.append(f"| {NAMES[ctrl]} | {pct(r.success_rate)} % | {r.place_error_success_mm:.1f} | "
                     f"{r.time_to_success_steps:.0f} | {r.collision_steps:.2f} | {r.act_time_ms:.1f} |")
    return "\n".join(lines)


def sac_seeds_table() -> str:
    path = RESULTS_DIR / "benchmark" / "ground_truth" / "episodes.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    d = d[d.controller == "sac"]
    t = d.groupby(["condition", "seed"]).success.mean().unstack()
    t = t.reindex([c for c in CONDITIONS if c in t.index])
    lines = ["| condition | " + " | ".join(f"seed {s}" for s in t.columns) + " |", "|---|" + "---|" * len(t.columns)]
    for cond, row in t.iterrows():
        lines.append(f"| {cond.replace('_', ' ')} | " + " | ".join(f"{pct(v)} %" for v in row) + " |")
    return "\n".join(lines)


def world_model_table() -> str:
    e = pd.read_csv(RESULTS_DIR / "world_model" / "multistep_errors.csv")
    e = e[e.contact.notna()]
    e["contact"] = e.contact.astype(str).str.lower() == "true"
    names = {"obj_pos_mm": "object position (mm)", "tcp_pos_mm": "TCP position (mm)", "grip_mm": "gripper opening (mm)",
             "obj_rot_deg": "object orientation (deg)"}
    lines = ["| quantity | phase | 1 step | 5 steps | 10 steps | 'nothing moves', 10 steps |", "|---|---|---|---|---|---|"]
    for feat, label in names.items():
        for contact in (False, True):
            g = e[(e.feature == feat) & (e.contact == contact)].set_index("horizon")
            lines.append(f"| {label} | {'contact' if contact else 'free'} | {g.model_median[1]:.1f} | "
                         f"{g.model_median[5]:.1f} | {g.model_median[10]:.1f} | {g.constant_median[10]:.1f} |")
    summary = json.loads((RESULTS_DIR / "world_model" / "summary.json").read_text())
    lines.append("")
    lines.append(f"{summary['n_transitions']} transitions ({summary['n_train']} train / {summary['n_val']} validation / "
                 f"{summary['n_test']} test, split by episode); correlation between log ensemble spread and log error "
                 f"at 5 steps: {summary['log_spread_error_correlation']:.2f}.")
    return "\n".join(lines)


def perception_table() -> str:
    s = pd.read_csv(RESULTS_DIR / "perception" / "localisation_summary.csv")
    lines = ["| condition | noise | detected | xy error median / p95 (mm) | yaw error median / p95 (deg) | "
             "width error (mm) | target error (mm) |", "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.condition.replace('_', ' ')} | {r.noise} | {pct(r.detection_rate)} % | "
                     f"{r.xy_error_median_mm:.1f} / {r.xy_error_p95_mm:.1f} | "
                     f"{r.yaw_error_median_deg:.1f} / {r.yaw_error_p95_deg:.1f} | {r.width_error_median_mm:.1f} | "
                     f"{r.target_error_median_mm:.1f} |")
    loop = pd.read_csv(RESULTS_DIR / "perception" / "closed_loop_summary.csv")
    lines += ["", "| condition | object pose from | success (scripted controller) | placement error (mm) |",
              "|---|---|---|---|"]
    for _, r in loop.iterrows():
        lines.append(f"| {r.condition} | {r.observation.replace('_', ' ')} | {pct(r.success_rate)} % "
                     f"({r.episodes} episodes) | {r.place_error_mm:.1f} |")
    return "\n".join(lines)


def sweep_table() -> str:
    path = RESULTS_DIR / "benchmark" / "uncertainty_sweep" / "summary.csv"
    if not path.exists():
        return ""
    s = pd.read_csv(path)
    weights = sorted(s.uncertainty_weight.unique())
    lines = ["| condition | " + " | ".join(f"λ = {w:g}" for w in weights) + " |", "|---|" + "---|" * len(weights)]
    for cond, g in s.groupby("condition", sort=False):
        g = g.set_index("uncertainty_weight")
        lines.append(f"| {cond.replace('_', ' ')} | " + " | ".join(
            success_cell(g.loc[w]) if w in g.index else "-" for w in weights) + " |")
    return "\n".join(lines)


def main():
    sections = [
        ("Benchmark, ground-truth object state (success %, 95 % Wilson interval)", benchmark_table("ground_truth")),
        ("Nominal condition, details", details_table("ground_truth")),
        ("SAC per seed", sac_seeds_table()),
        ("Benchmark, object pose from the camera (realistic noise)", benchmark_table("camera_realistic")),
        ("Uncertainty penalty sweep (MPC)", sweep_table()),
        ("World model, median open-loop error on test episodes", world_model_table()),
        ("Perception", perception_table()),
    ]
    text = "\n\n".join(f"### {title}\n\n{table}" for title, table in sections if table)
    (RESULTS_DIR / "tables.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
