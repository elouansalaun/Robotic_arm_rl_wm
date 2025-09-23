### Benchmark, ground-truth object state (success %, 95 % Wilson interval)

| controller | nominal | new positions | new objects | heavy slippery |
|---|---|---|---|---|
| Scripted (IK) | 100 [93-100] | 100 [93-100] | 100 [93-100] | 100 [93-100] |
| SAC | 86 [80-91] | 16 [11-23] | 69 [61-76] | 83 [76-88] |
| MPC | 86 [74-93] | 64 [50-76] | 70 [56-81] | 84 [71-92] |
| MPC + uncertainty penalty | 80 [67-89] | 58 [44-71] | 88 [76-94] | 96 [87-99] |

### Nominal condition, details

| controller | success | placement error (mm, median) | steps to success | steps with a collision | decision time (ms) |
|---|---|---|---|---|---|
| Scripted (IK) | 100 % | 1.9 | 60 | 0.00 | 0.0 |
| SAC | 86 % | 9.3 | 34 | 0.78 | 0.5 |
| MPC | 86 % | 5.2 | 40 | 0.02 | 193.8 |
| MPC + uncertainty penalty | 80 % | 5.4 | 39 | 0.58 | 187.8 |

### SAC per seed

| condition | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| nominal | 94 % | 92 % | 72 % |
| new positions | 8 % | 28 % | 12 % |
| new objects | 58 % | 68 % | 80 % |
| heavy slippery | 90 % | 84 % | 74 % |

### Benchmark, object pose from the camera (realistic noise)

| controller | nominal |
|---|---|
| Scripted (IK) | 96 [87-99] |
| SAC | 60 [52-67] |
| MPC | 68 [54-79] |
| MPC + uncertainty penalty | 64 [50-76] |

### Uncertainty penalty sweep (MPC)

| condition | λ = 0 | λ = 0.02 | λ = 0.05 | λ = 0.1 | λ = 0.2 |
|---|---|---|---|---|---|
| new objects | 70 [56-81] | 80 [67-89] | 88 [76-94] | 90 [79-96] | 68 [54-79] |
| nominal | 86 [74-93] | 88 [76-94] | 80 [67-89] | 78 [65-87] | 66 [52-78] |

### World model, median open-loop error on test episodes

| quantity | phase | 1 step | 5 steps | 10 steps | 'nothing moves', 10 steps |
|---|---|---|---|---|---|
| object position (mm) | free | 0.3 | 1.6 | 3.3 | 0.0 |
| object position (mm) | contact | 0.8 | 3.6 | 7.7 | 94.3 |
| TCP position (mm) | free | 0.6 | 2.5 | 4.7 | 50.2 |
| TCP position (mm) | contact | 0.8 | 3.2 | 5.5 | 99.1 |
| gripper opening (mm) | free | 0.3 | 0.4 | 0.4 | 5.1 |
| gripper opening (mm) | contact | 0.6 | 1.6 | 2.4 | 4.0 |
| object orientation (deg) | free | 0.3 | 1.3 | 2.5 | 0.0 |
| object orientation (deg) | contact | 0.4 | 2.0 | 3.7 | 2.8 |

118504 transitions (94802 train / 11947 validation / 11755 test, split by episode); correlation between log ensemble spread and log error at 5 steps: 0.76.

### Perception

| condition | noise | detected | xy error median / p95 (mm) | yaw error median / p95 (deg) | width error (mm) | target error (mm) |
|---|---|---|---|---|---|---|
| nominal | clean | 100 % | 0.8 / 1.3 | 0.0 / 0.4 | 1.4 | 1.0 |
| nominal | realistic | 100 % | 0.8 / 1.6 | 1.1 / 2.8 | 1.4 | 0.9 |
| nominal | lighting | 100 % | 0.7 / 1.5 | 0.6 / 2.2 | 1.4 | 0.9 |
| nominal | heavy | 100 % | 0.9 / 1.8 | 1.6 / 5.7 | 1.7 | 0.7 |
| new objects | realistic | 100 % | 1.2 / 2.7 | 0.8 / 2.7 | 7.4 | 0.8 |
| clutter | realistic | 99 % | 0.9 / 37.7 | 0.8 / 14.0 | 1.7 | 1.0 |

| condition | object pose from | success (scripted controller) | placement error (mm) |
|---|---|---|---|
| nominal | ground truth | 100 % (30 episodes) | 1.8 |
| nominal | clean | 93 % (30 episodes) | 3.7 |
| nominal | realistic | 97 % (30 episodes) | 3.0 |
| nominal | lighting | 100 % (30 episodes) | 3.4 |
| nominal | heavy | 100 % (30 episodes) | 3.7 |
| clutter | realistic | 83 % (30 episodes) | 4.5 |
