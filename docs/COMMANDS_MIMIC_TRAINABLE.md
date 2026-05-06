# Mimic Trainable Commands (Nao / Spot)

These commands use local trainable motions under `motions/trainable/`.

## 1) Spot Training Command

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-Spot-v0 \
  --motion_file "/home/vergil/MENU/Projects/whole_body_tracking/motions/trainable/spot/AI_PlayBow_spot.npz" \
  --max_iterations 30000 \
  --num_envs 512 \
  --headless \
  --logger wandb
```

Debug video flags (optional, append when needed):

```bash
--video --video_interval 10000 --video_length 600
```

## 2) Nao Training Command

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-Nao-v0 \
  --motion_file "/home/vergil/MENU/Projects/whole_body_tracking/motions/trainable/nao/1_wayne_0_1_1_nao.npz" \
  --max_iterations 30000 \
  --num_envs 512 \
  --headless \
  --logger wandb
```

Debug video flags (optional, append when needed):

```bash
--video --video_interval 10000 --video_length 600
```

## Optional Run Naming (Recommended)

If you want cleaner wandb runs, append:

```bash
--log_project_name <project_name> --run_name <run_name>
```

## Weights & Biases Naming Setup (Team / Project / Run)

Use the following mapping in this project:

- Team (entity): set by environment variable `WANDB_ENTITY`
- Project: set by CLI flag `--log_project_name`
- Run name: set by CLI flag `--run_name`

One-time (or per shell) setup:

```bash
wandb login
export WANDB_ENTITY="<your_team_or_org>"
```

Example naming convention:

- `project`: `BeyondMimic`
- `run_name`: `<robot>-<motion>-iter<iters>-env<envs>`
- Spot example run name: `spot-playbow-iter30000-env512`
- Nao example run name: `nao-wayne-iter30000-env512`

Full Spot example with naming:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Flat-Spot-v0 \
  --motion_file "/home/vergil/MENU/Projects/whole_body_tracking/motions/trainable/spot/AI_PlayBow_spot.npz" \
  --max_iterations 30000 \
  --num_envs 512 \
  --headless \
  --logger wandb \
  --log_project_name "BeyondMimic" \
  --run_name "spot-playbow-iter30000-env512"
```
