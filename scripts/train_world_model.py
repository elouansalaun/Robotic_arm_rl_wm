"""Train the dynamics ensemble and measure its multi-step prediction error.

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rwm.config import RESULTS_DIR, load_config
from rwm.world_model.dataset import TransitionData
from rwm.world_model.evaluation import multistep_errors, uncertainty_vs_error
from rwm.world_model.training import train_ensemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "world_model")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    torch.set_num_threads(2)
    cfg = load_config("world_model")
    train_cfg = {**cfg["model"], **cfg["training"]}
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    args.out.mkdir(parents=True, exist_ok=True)

    data = TransitionData.concatenate([TransitionData.load(p) for p in args.data])
    train, val, test = data.split_by_episode(cfg["data"]["split"], seed=cfg["data"]["split_seed"])
    print(f"{len(data)} transitions | train {len(train)}  val {len(val)}  test {len(test)} | sources {data.source_names}")

    model, history = train_ensemble(train, val, train_cfg)
    model.save(args.out / "ensemble.pt")
    pd.DataFrame(history).drop(columns="val_mse_members").to_csv(args.out / "training_history.csv", index=False)

    errors = multistep_errors(model, test, horizons=(1, 5, 10))
    errors.to_csv(args.out / "multistep_errors.csv", index=False)
    calib = uncertainty_vs_error(model, test, horizon=5)
    calib.to_csv(args.out / "uncertainty_vs_error.csv", index=False)

    table = errors[errors.feature.isin(["tcp_pos_mm", "obj_pos_mm", "grip_mm", "obj_rot_deg"])]
    print(table.pivot_table(index=["feature", "contact"], columns="horizon", values="model_median").round(2))
    corr = np.corrcoef(np.log(calib.spread_mm + 1e-3), np.log(calib.error_mm + 1e-3))[0, 1]
    with open(args.out / "summary.json", "w") as f:
        json.dump(dict(n_transitions=len(data), n_train=len(train), n_val=len(val), n_test=len(test),
                       sources=data.source_names, epochs=len(history), best_val_mse=min(h["val_mse"] for h in history),
                       log_spread_error_correlation=float(corr), data_files=[str(p) for p in args.data]), f, indent=1)
    print(f"log(spread) vs log(error) correlation at 5 steps: {corr:.2f}")


if __name__ == "__main__":
    main()
