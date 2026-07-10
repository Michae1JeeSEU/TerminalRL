# Server training guide

This directory prepares simulation-only high-level PPO training for dynamic
terminal airspace reconfiguration and multi-UAV landing-flow coordination.  It
does not connect to ROS2, Crazyswarm2, Lighthouse, or real Crazyflie hardware.

## What to upload

Generate a package and manifest from the project root:

```bash
python scripts/server/pack_for_server.py
```

Upload the generated `terminal_server_training_package.zip` or the files listed
in `SERVER_UPLOAD_MANIFEST.txt`.  The manifest includes:

- `terminal_marl/`
- `docs/`
- `scripts/server/`
- selected training/smoke tests under `tests/`
- `requirements.txt`, `pyproject.toml`, `README.md`, and `AGENTS.md` when present

The packer excludes `.git/`, `runs/`, caches, bytecode, local model outputs,
large logs, and zip artifacts.

## Environment setup without sudo

Example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your cluster uses modules, uncomment an appropriate line in the scripts, for
example:

```bash
# module load anaconda3-2024.10-1
```

All scripts default to CPU execution and set:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

## Recommended stages

| Stage | Purpose | Suggested parameters |
|---|---|---|
| Stage 0 | Environment check and smoke test | `total_timesteps=2048`, `n_envs=2` or `4` |
| Stage 1 | Single-seed pilot | `total_timesteps=100000`, `n_envs=8`, `cpus=12`, `mem=32G` |
| Stage 2 | 3-seed medium/high/bursty small training | seeds `0,1,2`, medium/high/bursty |
| Stage 3 | 5-seed main experiments | `total_timesteps=500000` or `1000000`, `n_envs=16`, `cpus=20`, `mem=64G` |
| Stage 4 | Reward / learning-rate sweep | array jobs over seed, demand, learning rate |
| Stage 5 | Unified evaluation, aggregation, plotting | export CSV summaries and figures |

Use `medium/high/bursty` as the priority demand levels; keep `low` as a
supplement.  Initial seeds `0,1,2` are enough for pilot evidence; expand to
`0-4` for formal paper tables.

## Commands

Smoke:

```bash
bash scripts/server/00_check_env.sh
bash scripts/server/01_train_smoke_cpu.sh
```

Pilot:

```bash
SEED=0 bash scripts/server/02_train_pilot_medium_cpu.sh
```

Main SLURM job:

```bash
PARTITION=normal sbatch --partition="${PARTITION}" scripts/server/03_train_main_cpu.slurm
```

The default `PARTITION` variable is `6240-36C-192G`; change it to `normal` or
another CPU queue on your server.

Sweep array:

```bash
PARTITION=normal sbatch --partition="${PARTITION}" scripts/server/04_train_sweep_array.slurm
```

Evaluation:

```bash
MODEL_PATH=runs/high_level_ppo/ppo_main_medium/model.zip \
sbatch --partition="${PARTITION:-normal}" scripts/server/05_eval_ablation_cpu.slurm
```

Aggregate and plot:

```bash
bash scripts/server/06_aggregate_and_plot.sh
```

## Interpretation

The trained artifact is a high-level dynamic airspace policy.  It adjusts
airspace/flow intent through `HighLevelHRLWrapper`; it is not a low-level flight
controller and does not output free heading, altitude, or velocity-vector
commands.

