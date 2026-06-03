# NAO Stage3 Wrist RL Training Summary

## Scope

This report summarizes the three local NAO `velocity_stage3_wrist` training runs recorded on 2026-06-03 after `01:37:15`.

All three runs used:

- Task: `Tracking-Flat-Nao-v0`
- Logger: `wandb`
- Max iterations: `10000`
- TensorBoard scalar coverage: step `0` through step `9999`
- Final checkpoint: `model_9999.pt`

## Samples

| Sample | Motion file | FPS | Frames | Duration | Run directory |
|---|---|---:|---:|---:|---|
| Jensen | `Jensen_nao_gmr_velocity_stage3_wrist.npz` | 50 | 350 | 7.00s | `logs/rsl_rl/nao_flat/2026-06-03_01-37-15` |
| Zhihui | `Zhihui_nao_gmr_velocity_stage3_wrist.npz` | 50 | 484 | 9.68s | `logs/rsl_rl/nao_flat/2026-06-03_01-42-50` |
| Musk | `Musk_nao_gmr_velocity_stage3_wrist.npz` | 50 | 329 | 6.58s | `logs/rsl_rl/nao_flat/2026-06-03_01-43-39` |

Motion array shapes for all three files:

| Field | Shape |
|---|---|
| `joint_pos` | `(T, 42)` |
| `joint_vel` | `(T, 42)` |
| `body_pos_w` | `(T, 43, 3)` |
| `body_quat_w` | `(T, 43, 4)` |

## Velocity Constraint And Reward Design

The training does not use a hard velocity constraint. It uses velocity perturbation ranges plus velocity-tracking reward terms.

### Command Velocity Perturbation Range

Defined in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py` as `VELOCITY_RANGE`:

| Axis | Range |
|---|---:|
| `x` | `(-0.5, 0.5)` |
| `y` | `(-0.5, 0.5)` |
| `z` | `(-0.2, 0.2)` |
| `roll` | `(-0.52, 0.52)` |
| `pitch` | `(-0.52, 0.52)` |
| `yaw` | `(-0.78, 0.78)` |

This range is applied to the motion command velocity during command resampling.

### Global Body Velocity Rewards

Defined in `tracking_env_cfg.py`:

| Reward term | Weight | Std | Purpose |
|---|---:|---:|---|
| `motion_body_lin_vel` | `1.0` | `1.0` | Track global body linear velocities. |
| `motion_body_ang_vel` | `1.0` | `3.14` | Track global body angular velocities. |

### NAO Wrist-Specific Velocity Rewards

Defined in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/nao/flat_env_cfg.py`:

| Reward term | Bodies | Weight | Std | Purpose |
|---|---|---:|---:|---|
| `motion_wrist_lin_vel` | `l_wrist`, `r_wrist` | `2.0` | `1.0` | Emphasize wrist linear velocity tracking. |
| `motion_wrist_ang_vel` | `l_wrist`, `r_wrist` | `2.0` | `3.14` | Emphasize wrist angular velocity tracking. |

This is the sample-specific design focus for the `velocity_stage3_wrist` motions: the policy is encouraged to preserve wrist motion quality more strongly than the base global body velocity terms alone would do.

## Termination Design Relevant To Play/Reset

The play script reuses the training environment termination rules unless overridden.

Default tracking terminations:

| Termination | Meaning | Default base threshold |
|---|---|---:|
| `time_out` | Episode time limit | `episode_length_s = 10.0` |
| `anchor_pos` | Anchor body height tracking error | `0.25m` base, tightened for NAO |
| `anchor_ori` | Anchor orientation/gravity mismatch | `0.8` |
| `ee_body_pos` | End-effector/body height tracking error | `0.25m` base, tightened for NAO |

NAO-specific termination changes:

| Termination | NAO threshold |
|---|---:|
| `anchor_pos` | `0.15m` |
| `ee_body_pos` | `0.15m` |
| `ee_body_pos` bodies | `NAO_FOOT_NAMES` |

This means a replay can reset because of timeout or because the torso/feet deviate from the reference, even if the policy is still producing actions.

## Training Result Summary

| Sample | Final reward | Peak reward | Peak reward step | Final episode length | Peak episode length | Peak episode step |
|---|---:|---:|---:|---:|---:|---:|
| Jensen | 15.27 | 19.60 | 9878 | 219.18 | 276.28 | 9878 |
| Zhihui | 16.10 | 23.33 | 5977 | 217.78 | 308.73 | 5977 |
| Musk | 16.85 | 23.62 | 8992 | 219.06 | 296.79 | 8992 |

### Final Tracking Errors

Lower is better.

| Sample | Anchor pos | Anchor rot | Body pos | Body rot | Joint pos | Joint vel |
|---|---:|---:|---:|---:|---:|---:|
| Jensen | 0.145 | 0.546 | 0.080 | 0.720 | 2.230 | 4.928 |
| Zhihui | 0.231 | 0.526 | 0.120 | 0.900 | 2.866 | 4.176 |
| Musk | 0.271 | 0.556 | 0.126 | 0.887 | 2.210 | 3.975 |

### Final Velocity Errors

Lower is better.

| Sample | Body linear velocity error | Body angular velocity error |
|---|---:|---:|
| Jensen | 0.490 | 2.163 |
| Zhihui | 0.757 | 2.976 |
| Musk | 0.958 | 3.576 |

### Final Termination Metrics

These metrics indicate what kinds of reset/failure conditions were being triggered during training.

| Sample | Timeout | Anchor pos | Anchor ori | EE body pos |
|---|---:|---:|---:|---:|
| Jensen | 0.250 | 0.000 | 0.000 | 1.000 |
| Zhihui | 0.125 | 0.042 | 0.000 | 1.000 |
| Musk | 0.083 | 0.875 | 0.000 | 1.417 |

### Final Reward Components

| Sample | Anchor pos | Anchor ori | Body pos | Body ori | Body lin vel | Body ang vel | Wrist lin vel | Wrist ang vel | Action rate | Joint limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jensen | 0.182 | 0.113 | 0.387 | 0.056 | 0.367 | 0.287 | 0.684 | 0.505 | -1.069 | -0.192 |
| Zhihui | 0.187 | 0.116 | 0.383 | 0.015 | 0.379 | 0.327 | 0.717 | 0.604 | -1.023 | -0.198 |
| Musk | 0.151 | 0.117 | 0.328 | 0.046 | 0.313 | 0.268 | 0.597 | 0.498 | -0.876 | -0.170 |

### Final Loss And Policy Scalars

| Sample | Value loss | Surrogate loss | Entropy | Mean noise std | Final FPS |
|---|---:|---:|---:|---:|---:|
| Jensen | 0.418 | -0.019 | 26.902 | 0.743 | 2347 |
| Zhihui | 0.409 | -0.018 | 26.817 | 0.741 | 1344 |
| Musk | 0.406 | -0.016 | 27.427 | 0.760 | 1915 |

## Per-Sample Conclusions

### Jensen

Jensen is the most stable of the three based on final tracking errors. It has the lowest final anchor position error, body position error, body rotation error, and body linear/angular velocity errors.

Its downside is lower reward ceiling: peak reward is `19.60`, below Zhihui and Musk. This suggests the policy converged to a steadier but less expressive tracking solution.

Recommended use: best first candidate when stable replay is more important than maximum reward.

### Zhihui

Zhihui achieved the strongest peak episode length, `308.73`, and a high peak reward, `23.33`, at step `5977`. However, the final checkpoint regressed relative to that peak.

Final errors are worse than Jensen for body position, body rotation, and joint position. The final reward remains higher than Jensen, but the final tracking quality is less clean.

Recommended use: inspect intermediate checkpoints around `model_6000.pt`, not only `model_9999.pt`.

### Musk

Musk has the highest final reward, `16.85`, and the highest peak reward, `23.62`. It also has relatively good final joint position and joint velocity errors.

However, it has the highest final anchor position error and the highest termination metrics for `anchor_pos` and `ee_body_pos`. This is consistent with the observed play-time reset risk: Musk is more likely to trigger torso/foot deviation termination despite strong reward.

Recommended use: good candidate for expressive motion, but play/evaluation should use reset diagnostics or relaxed termination thresholds to inspect the full motion.

## Overall Ranking By Practical Criterion

| Criterion | Best candidate | Reason |
|---|---|---|
| Stable tracking | Jensen | Lowest final body/anchor position and velocity errors. |
| Highest final reward | Musk | Highest final `Train/mean_reward`. |
| Best peak training behavior | Musk / Zhihui | Musk has highest peak reward; Zhihui has longest peak episode length. |
| Lowest play reset risk | Jensen | Lowest final `anchor_pos` termination and lower tracking errors. |
| Best checkpoint to inspect beyond final | Zhihui | Peak occurs around step `5977`, much better than final trend. |

## Data Availability

Available local data:

| Data type | Location |
|---|---|
| TensorBoard event scalars | `logs/rsl_rl/nao_flat/<run>/events.out.tfevents.*` |
| Final checkpoints | `logs/rsl_rl/nao_flat/<run>/model_9999.pt` |
| Intermediate checkpoints | `logs/rsl_rl/nao_flat/<run>/model_*.pt` |
| Training env config | `logs/rsl_rl/nao_flat/<run>/params/env.yaml` |
| Training agent config | `logs/rsl_rl/nao_flat/<run>/params/agent.yaml` |
| W&B local summaries | `wandb/run-20260603_*/files/wandb-summary.json` |
| W&B output logs | `wandb/run-20260603_*/files/output.log` |

The TensorBoard event files contain the complete scalar curves for all 10000 iterations. This report only summarizes key final values, peak values, and conclusion-level comparisons.

## Suggested Next Checks

1. Replay `Jensen model_9999.pt` as the stable baseline.
2. Replay `Musk model_9999.pt` with `--no_play_reset` or relaxed termination scale to inspect whether the high reward corresponds to visually useful motion.
3. Replay `Zhihui model_6000.pt` because its peak reward and episode length occur around step `5977`.
4. Compare final checkpoints against peak checkpoints using the same play settings and camera/debug options.
