"""Side-by-side video of several controllers on the same scenario.

    MUJOCO_GL=egl python scripts/record_video.py --controllers scripted sac mpc \\
        --sac-run runs/sac/sac_demo_s0 --scenarios 0 1 2
"""

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from rwm.config import RESULTS_DIR, load_config
from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.evaluation.benchmark import make_controller
from rwm.evaluation.rollout import run_episode
from rwm.evaluation.scenarios import test_set

LABELS = {"scripted": "Scripted (IK + trajectories)", "sac": "SAC", "mpc": "MPC (world model)",
          "mpc_unc": "MPC + uncertainty penalty"}


def label(frame: np.ndarray, text: str, status: str) -> np.ndarray:
    img = frame.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    if status:
        color = (20, 140, 60) if status == "success" else (190, 40, 40)
        cv2.putText(img, status, (img.shape[1] - 80, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", nargs="+", default=["scripted", "sac", "mpc"])
    parser.add_argument("--sac-run", type=Path)
    parser.add_argument("--world-model", type=Path, default=RESULTS_DIR / "world_model" / "ensemble.pt")
    parser.add_argument("--condition", default="nominal")
    parser.add_argument("--scenarios", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "videos" / "comparison.mp4")
    args = parser.parse_args()

    sim_cfg = load_config("simulation")
    env = PickPlaceEnv(sim_cfg, condition=args.condition)
    renderer = mujoco.Renderer(env.scene.model, 360, 480)

    def grab():
        renderer.update_scene(env.scene.data, camera=args.camera)
        return renderer.render()

    scenarios = test_set(sim_cfg, args.condition)
    video = []
    for idx in args.scenarios:
        columns = []
        for name in args.controllers:
            ctrl = make_controller(name, env, sac_run=args.sac_run, world_model=args.world_model)
            result = run_episode(env, ctrl, scenarios[idx], record=grab)
            status = "success" if result["success"] else "failure"
            frames = [label(f, LABELS.get(name, name), "") for f in result["frames"]]
            frames[-1] = label(result["frames"][-1], LABELS.get(name, name), status)
            columns.append(frames)
            print(f"scenario {idx} {name}: {status}")
        n = max(len(c) for c in columns)
        for t in range(n):
            # episodes of different lengths: hold the last frame
            video.append(np.concatenate([c[min(t, len(c) - 1)] for c in columns], axis=1))
        video += [video[-1]] * 15
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, video, fps=20, quality=7, macro_block_size=8)
    # lighter GIF for the README (GitHub plays GIFs inline): every third frame, half size
    small = [cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for f in video[::3]]
    imageio.mimwrite(args.out.with_suffix(".gif"), small, duration=150, loop=0)
    renderer.close()
    print(f"{len(video)} frames -> {args.out} (+ .gif)")


if __name__ == "__main__":
    main()
