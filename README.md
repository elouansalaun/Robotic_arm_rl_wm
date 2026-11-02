# Robotic pick-and-place with a learned world model

A simulated 6-DOF robot arm finds an object on a table with an RGB-D camera, picks it up and places it on a
target platform. Object shape, size, mass, friction and positions change every episode. I built the whole
pipeline from scratch (robot model, kinematics, perception, reinforcement learning, learned dynamics model and
planner) and compared three ways of controlling the arm on the same test episodes.

![three controllers on the same scenario](results/videos/comparison.gif)

| controller | how it decides |
|---|---|
| **Scripted** | hand-written phases, minimum-jerk Cartesian trajectories, closed-form inverse kinematics |
| **SAC** | neural policy trained by reinforcement learning (300k steps + 100 demonstrations) |
| **MPC** | plans through a learned dynamics model (ensemble of neural networks), replans every step |

Everything runs in MuJoCo on two CPU cores; no robot, no GPU.

## Results

Success rate over the same 50 test episodes per condition (SAC pooled over 3 seeds):

| | nominal | new positions | new objects | heavier, slippery | with camera |
|---|---|---|---|---|---|
| Scripted (IK) | 100 % | 100 % | 100 % | 100 % | 96 % |
| SAC | 86 % | 16 % | 69 % | 83 % | 60 % |
| MPC | 86 % | 64 % | 70 % | 84 % | 68 % |
| MPC + uncertainty penalty | 80 % | 58 % | 88 % | 96 % | 64 % |

- **SAC and MPC are equal on the training distribution, but fail differently outside it.** The policy does not
  extrapolate to positions it has never seen (16 %); the learned dynamics carry over (64 %).
- **Using the model's own uncertainty pays off on new dynamics.** Penalising disagreement between ensemble
  members gains 12 to 18 points on unseen objects and physics, at a small cost on familiar ones.
- **The camera localises objects to about 1 mm**, enough for the scripted controller. The learned controllers
  lose 18 to 26 points because they rely on the full state, including quantities the camera measures badly
  (velocities, the orientation of symmetric objects).
- **The scripted controller is the reference** (it never fails here) but it is the slowest and encodes the task
  entirely by hand; SAC is the fastest (34 steps vs 60) and 400x cheaper per decision than MPC.

![benchmark](results/figures/benchmark_success.png)

## How it works

**Simulation** (`assets/`, `src/rwm/simulation/`): MJCF model of an anthropomorphic arm with a two-finger
gripper, position servos, gravity compensation and tuned contacts. Each episode is a serialisable scenario
sampled from `configs/simulation.yaml`; full simulator snapshots replay bit for bit.

**Kinematics and scripted control** (`src/rwm/robotics/`): forward kinematics and Jacobian (matching MuJoCo
to 1e-15), closed-form inverse kinematics returning all 8 branches, damped-least-squares IK for every control
step, and a phase-based controller that retries when a grasp fails.

**Perception** (`src/rwm/perception/`): depth back-projection, colour segmentation and geometric fitting give
the object position, yaw and size and the target position from a single frame. Tested under depth noise,
missing pixels, lighting changes and occluding distractors.

**Reinforcement learning** (`src/rwm/rl/`, `src/rwm/environments/`): Gymnasium environment and SAC
(Stable-Baselines3). Pure SAC never learned to grasp; mixing scripted demonstrations into every batch (RLPD)
brought it to 85-100 % validation success.

![SAC learning curves](results/figures/sac_learning_curves.png)

**World model** (`src/rwm/world_model/`): ensemble of 5 MLPs predicting the next state (33 dims) and the risk of
an unwanted contact, trained on ~120k transitions. Errors stay under 1 cm over a 0.5 s open-loop rollout, and
the disagreement between members tracks the actual error (correlation 0.76), which the planner uses.

**Model predictive control** (`src/rwm/planning/`): cross-entropy method sampling 256 action sequences of
0.5 s, scored with the same reward as SAC through the learned model, about 190 ms per decision.

## Notes

- Fixed test sets and 95 % confidence intervals everywhere: an early 20-episode test showed an improvement that
  vanished on 50 episodes.
- Several failures were only found by debugging the full loop: a planner exploiting loopholes in the reward
  (throwing objects onto the platform), a sampling distribution that could not express "go down, then hold
  still", and a 1 mm camera error turning into a 10-sigma input for the policy because of input normalisation.
- Long jobs are resumable (weight and replay-buffer snapshots, evaluations that skip finished episodes).
- Unit tests cover kinematics against MuJoCo, snapshot replay, the environment API, perception and the world model.


**Stack**: Python, MuJoCo, PyTorch, Stable-Baselines3, Gymnasium, OpenCV, NumPy, pandas.
