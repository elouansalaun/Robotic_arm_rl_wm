"""Accuracy of the camera pipeline, alone and in the control loop.

1. single-frame localisation error against ground truth, for several sensor
   noise levels and scene conditions
2. scripted controller fed with camera estimates instead of the true object pose

    MUJOCO_GL=egl python scripts/evaluate_perception.py
"""

import argparse

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd

from rwm.config import RESULTS_DIR, load_config
from rwm.environments.pick_place_env import PickPlaceEnv
from rwm.evaluation.perception import localisation_errors, summarise
from rwm.evaluation.rollout import run_episode
from rwm.evaluation.scenarios import test_set
from rwm.perception.camera import RGBDCamera
from rwm.perception.detection import ObjectDetector
from rwm.perception.estimator import CameraStateEstimator
from rwm.robotics.controllers import ScriptedPickPlace
from rwm.simulation.scene import PickPlaceScene

FRAME_CASES = [("nominal", "clean"), ("nominal", "realistic"), ("nominal", "lighting"), ("nominal", "heavy"),
               ("new_objects", "realistic"), ("clutter", "realistic")]
LOOP_CASES = [("nominal", "clean"), ("nominal", "realistic"), ("nominal", "lighting"), ("nominal", "heavy"),
              ("clutter", "realistic")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--frames-only", action="store_true", help="skip the closed-loop episodes")
    args = parser.parse_args()

    sim_cfg, vision_cfg = load_config("simulation"), load_config("vision")
    out = RESULTS_DIR / "perception"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- frames
    scene = PickPlaceScene()
    cam_cfg = vision_cfg["camera"]
    camera = RGBDCamera(scene.model, cam_cfg["name"], cam_cfg["width"], cam_cfg["height"])
    detector = ObjectDetector(vision_cfg)
    tables, examples = [], []
    for k, (condition, noise) in enumerate(FRAME_CASES):
        scenarios = test_set(sim_cfg, condition)[: args.n_frames]
        # a different scene in each panel of the example figure
        df, frames = localisation_errors(scene, camera, detector, vision_cfg, noise, scenarios, example=2 * k + 1)
        df["condition"] = condition
        tables.append(df)
        for f in frames:
            cv2.rectangle(f, (0, 0), (f.shape[1], 26), (250, 250, 248), -1)
            cv2.putText(f, f"{condition.replace('_', ' ')} / {noise} noise", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (20, 20, 20), 1, cv2.LINE_AA)
        examples += frames
        print(f"{condition:12s} {noise:10s} xy median {1e3 * df.xy_error.median():.1f} mm")
    camera.close()
    frames_df = pd.concat(tables, ignore_index=True)
    frames_df.to_csv(out / "localisation.csv", index=False)
    summary = summarise(frames_df)
    summary.to_csv(out / "localisation_summary.csv")
    print(summary.to_string())
    grid = np.concatenate([np.concatenate(examples[i:i + 3], axis=1) for i in range(0, 6, 3)], axis=0)
    imageio.imwrite(out / "detections.png", grid)
    if args.frames_only:
        return

    # ------------------------------------------------------------ closed loop
    rows = []
    for condition, noise in [("nominal", "ground_truth")] + LOOP_CASES:
        env = PickPlaceEnv(sim_cfg, condition=condition)
        estimator = None
        if noise != "ground_truth":
            estimator = CameraStateEstimator(env.scene.model, vision_cfg, noise=noise, dt=env.dt, seed=1)
            env.state_estimator = estimator
        ctrl = ScriptedPickPlace.from_env(env)
        for sc in test_set(sim_cfg, condition)[: args.n_episodes]:
            r = run_episode(env, ctrl, sc)
            rows.append(dict(condition=condition, observation=noise, success=r["success"],
                             place_error=r["place_error"], collision_steps=r["collision_steps"]))
        if estimator is not None:
            estimator.close()
        env.close()
        last = pd.DataFrame(rows[-args.n_episodes:])
        print(f"closed loop {condition:10s} {noise:12s} success {last.success.mean():.2f}")
    loop = pd.DataFrame(rows)
    loop.to_csv(out / "closed_loop.csv", index=False)
    table = loop.groupby(["condition", "observation"], sort=False).agg(
        success_rate=("success", "mean"), place_error_mm=("place_error", lambda x: 1e3 * x.median()),
        episodes=("success", "size")).round(3)
    table.to_csv(out / "closed_loop_summary.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
