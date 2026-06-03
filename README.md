# NAO Whole-Body Tracking

This repository is a NAO-focused extension of the original [BeyondMimic motion tracking code](https://github.com/HybridRobotics/whole_body_tracking). It keeps the Isaac Lab / RSL-RL training structure from BeyondMimic, while adding practical support for training and replaying GMR-retargeted NAO motions.

The motion-retargeting side is designed to connect with [GMR: General Motion Retargeting](https://github.com/YanjieZe/GMR) and the related fork/reference used in this project, [https://github.com/Djctionary/GMR](https://github.com/Djctionary/GMR). GMR provides humanoid motion retargeting tuned for RL tracking policies and supports conversion flows into BeyondMimic-style motion data.

## What Changed From Upstream BeyondMimic

| Area | Original BeyondMimic | This project |
|---|---|---|
| Main robot focus | G1 / humanoid tracking examples | Adds NAO tracking task and robot config |
| Motion source | WandB registry-first motion loading | Supports local trainable NPZ motions with `--motion_file` |
| Retargeting pipeline | General retargeted generalized-coordinate motions | NAO motions retargeted through GMR-compatible joint/body ordering |
| NAO control | Not the primary target | 24 active NAO joints, passive hand/finger joints locked |
| Reward design | Whole-body pose/velocity mimic rewards | Adds NAO wrist linear/angular velocity rewards |
| Evaluation UX | Basic play script | Adds camera presets, debug-visual hiding, and play reset controls |
| Documentation | Upstream project docs | Adds NAO run commands and three-sample training summary |

## Key Features

- `Tracking-Flat-Nao-v0` task registration.
- NAO URDF asset integration through Isaac Lab articulation config.
- NAO-specific active joint list aligned with GMR/MJCF qpos order.
- Passive hand/finger joints excluded from the policy and held at default pose.
- NAO tracking body set with torso anchor, feet, arms, and wrists.
- Wrist-specific velocity rewards for `l_wrist` and `r_wrist`.
- Local motion training via `--motion_file` for trainable NPZ files.
- Play-time camera controls: `--camera_view`, `--camera_eye`, `--camera_lookat`.
- Play-time visualization controls: `--no_debug_vis`.
- Play-time reset controls: `--termination_threshold_scale`, `--no_play_reset`.
- TensorBoard/W&B summaries for Jensen, Zhihui, and Musk NAO training runs.

## Installation

Install Isaac Sim / Isaac Lab as in the upstream BeyondMimic setup. This repo was developed against:

- Isaac Sim `4.5.0`
- Isaac Lab `2.1.0`
- Python `3.10`
- Linux

Then install this extension from the repository root:

```bash
python -m pip install -e source/whole_body_tracking
```

## GMR Connection

This project expects NAO motions to be retargeted before RL training. The intended upstream retargeting reference is GMR:

- Canonical repo: https://github.com/YanjieZe/GMR
- Project reference/fork: https://github.com/Djctionary/GMR

The NAO robot config documents the GMR assumption directly: the 24 active NAO joints follow the GMR-retargeted `dof_pos` order, matching the MJCF qpos order after the free joint. The local `pkl_to_npz.py` path maps GMR-style pkl columns into Isaac Lab articulation joints by name.

Expected trainable motion location:

```text
motions/trainable/nao/*.npz
```

Example local motions used in this repo:

```text
Jensen_nao_gmr_velocity_stage3_wrist.npz
Zhihui_nao_gmr_velocity_stage3_wrist.npz
Musk_nao_gmr_velocity_stage3_wrist.npz
```

## Training

Example NAO training command:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-Nao-v0 \
  --motion_file "motions/trainable/nao/Musk_nao_gmr_velocity_stage3_wrist.npz" \
  --max_iterations 10000 \
  --num_envs 128 \
  --headless \
  --logger wandb \
  --video \
  --video_interval 5000 \
  --video_length 600
```

For larger runs, increase `--num_envs` and `--max_iterations` as appropriate.

Optional naming:

```bash
--log_project_name BeyondMimic --run_name nao-musk-stage3-wrist-iter10000-env128
```

## Play / Evaluation

Play a local checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Flat-Nao-v0 \
  --motion_file "motions/trainable/nao/Musk_nao_gmr_velocity_stage3_wrist.npz" \
  --num_envs 1 \
  --load_run 2026-06-03_01-43-39 \
  --checkpoint model_9999.pt \
  --camera_view front \
  --no_debug_vis \
  --video \
  --video_length 600 \
  --headless
```

Useful play options:

| Option | Purpose |
|---|---|
| `--camera_view front` | Front-facing NAO replay camera. |
| `--camera_view back,left,right,iso` | Other camera presets. |
| `--camera_eye X Y Z` | Manual camera eye override. |
| `--camera_lookat X Y Z` | Manual camera target override. |
| `--no_debug_vis` | Hide robot/reference frame markers and contact debug visuals. |
| `--termination_threshold_scale 2.0` | Relax non-timeout reset thresholds during play. |
| `--no_play_reset` | Disable play-time terminations, including timeout and fall/reset checks. |

## TensorBoard

View all NAO runs:

```bash
tensorboard --logdir logs/rsl_rl/nao_flat --host 0.0.0.0 --port 6006
```

Then open:

```text
http://localhost:6006
```

or, on a remote server:

```text
http://<server-ip>:6006
```

## Current NAO Results

Three local `velocity_stage3_wrist` samples were trained for 10000 iterations each. TensorBoard contains scalar data through step `9999` for all three runs.

| Sample | Motion duration | Run | Final reward | Peak reward | Final episode length | Conclusion |
|---|---:|---|---:|---:|---:|---|
| Jensen | 7.00s | `2026-06-03_01-37-15` | 15.27 | 19.60 | 219.18 | Most stable final tracking. |
| Zhihui | 9.68s | `2026-06-03_01-42-50` | 16.10 | 23.33 | 217.78 | Strong peak around `model_6000.pt`, final regresses. |
| Musk | 6.58s | `2026-06-03_01-43-39` | 16.85 | 23.62 | 219.06 | Highest reward, but higher reset risk. |

Summary:

- Jensen is the best stable baseline.
- Musk is the best high-reward / expressive candidate, but should be checked with relaxed play resets.
- Zhihui should be evaluated around `model_6000.pt`, not only `model_9999.pt`.

Detailed metrics are in:

```text
docs/NAO_STAGE3_WRIST_TRAINING_SUMMARY.md
```

## Project Structure

```text
scripts/
  rsl_rl/train.py                  # RSL-RL training entrypoint
  rsl_rl/play.py                   # Policy replay/export with NAO-friendly play controls
  pkl_to_npz.py                    # GMR-style pkl to trainable NPZ conversion path
  replay_npz.py                    # Motion replay utility

source/whole_body_tracking/whole_body_tracking/
  robots/nao.py                    # NAO robot, joint mapping, actuators, tracked bodies
  tasks/tracking/config/nao/       # Tracking-Flat-Nao-v0 task and PPO config
  tasks/tracking/tracking_env_cfg.py
  tasks/tracking/mdp/              # Commands, rewards, observations, events, terminations

docs/
  COMMANDS_MIMIC_TRAINABLE.md
  NAO_STAGE3_WRIST_TRAINING_SUMMARY.md
```

## Notes

This is not a clean upstream BeyondMimic clone. It is a working NAO adaptation that preserves the BeyondMimic training core while adding the robot mapping, local data path, wrist-focused rewards, and evaluation controls needed for GMR-retargeted NAO motions.

Please cite or reference the original projects when using this work:

- BeyondMimic: https://github.com/HybridRobotics/whole_body_tracking
- GMR: https://github.com/YanjieZe/GMR
