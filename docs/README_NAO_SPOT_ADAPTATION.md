# Nao / Spot Adaptation Notes

This document reviews what changed in this branch compared with `upstream/main`, focusing on the two downstream robots: Nao and Spot.

Related runnable commands are in `docs/COMMANDS_MIMIC_TRAINABLE.md`.

## Scope From Git History

Main commits on top of `upstream/main`:

- `218cdf1` `feat(Spot): Initial`
- `010fb44` `Update play.py`
- `b74729e` `Integrate Nao into Beyondmimic, Fix Spot Path`
- `9bb5a58` `Generate Test NPZ file. Waiting for Real RL training`

## What Was Adapted For Spot

- Added a Spot robot model and control setup:
  - robot articulation config in `source/whole_body_tracking/whole_body_tracking/robots/spot.py`
  - Spot assets in `source/whole_body_tracking/whole_body_tracking/assets/spot/`
- Added Spot tracking task registration:
  - `Tracking-Flat-Spot-v0` in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/spot/__init__.py`
- Added Spot env config:
  - anchor body changed to `spot`
  - Spot-specific tracked bodies and end-effector termination bodies
  - Spot undesired-contact mask
  - file: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/spot/flat_env_cfg.py`
- Added Spot PPO config:
  - experiment name `spot_flat`, default `max_iterations = 30000`
  - file: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/spot/agents/rsl_rl_ppo_cfg.py`
- Updated motion conversion/replay/training scripts to support local motion files and Spot pipeline usage:
  - `scripts/csv_to_npz.py`
  - `scripts/replay_npz.py`
  - `scripts/rsl_rl/train.py`

## What Was Adapted For Nao

- Added a Nao robot model and actuator grouping:
  - articulation, active/passive joints, action-scale build logic
  - file: `source/whole_body_tracking/whole_body_tracking/robots/nao.py`
  - assets: `source/whole_body_tracking/whole_body_tracking/assets/nao/`
- Added Nao tracking task registration:
  - `Tracking-Flat-Nao-v0` in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/nao/__init__.py`
- Added Nao env config with key Nao-specific constraints:
  - action joint list restricted to 24 active joints
  - passive hand/finger joints are excluded from policy actions
  - default joint-offset randomization disabled to avoid index mismatch with passive joints
  - smaller termination thresholds (`0.15`) for Nao scale
  - file: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/nao/flat_env_cfg.py`
- Added Nao PPO config:
  - experiment name `nao_flat`, default `max_iterations = 30000`
  - file: `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/nao/agents/rsl_rl_ppo_cfg.py`
- Added dedicated pkl-to-npz converter for Nao retarget data:
  - `scripts/pkl_to_npz.py`
  - includes column mapping logic for Nao retarget order and passive joints

## Cross-Robot Pipeline Updates

- `scripts/rsl_rl/train.py` now supports `--motion_file` (local NPZ) in addition to registry artifacts.
- `scripts/rsl_rl/play.py` now supports local `--motion_file` in non-wandb loading mode and raises explicit error when neither motion source is provided.
- `scripts/replay_npz.py` now supports robot selection with `--robot {g1,spot,nao}` and local `--motion_file`.
- `source/whole_body_tracking/setup.py` includes `wandb` dependency for artifact workflow.
- `.gitattributes` and `.gitignore` were updated so trainable NPZ under `motions/trainable/` can be versioned, and Nao/Spot assets can be stored without relying on LFS.

## Current Trainable Motion Inputs

- Spot: `motions/trainable/spot/AI_PlayBow_spot.npz`
- Nao: `motions/trainable/nao/1_wayne_0_1_1_nao.npz`
