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

The scripts follow the previously used SLURM style:

```text
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
```

They also try `module load anaconda3-2024.10-1` and then
`source activate "${CONDA_ENV:-terminal-marl}"`.  Set `CONDA_ENV` before
`sbatch` if your server environment name differs.

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

Create the SLURM log directory once before submitting jobs:

```bash
mkdir -p logs
```

Smoke:

```bash
sbatch scripts/server/00_check_env.sh
sbatch scripts/server/01_train_smoke_cpu.sh
```

Pilot:

```bash
sbatch --export=ALL,SEED=0 scripts/server/02_train_pilot_medium_cpu.sh
```

Main SLURM job:

```bash
sbatch scripts/server/03_train_main_cpu.slurm
```

The default partition in the scripts is `6240-36C-192G`.  Override it at submit
time if your server uses another queue:

```bash
sbatch --partition=normal scripts/server/03_train_main_cpu.slurm
```

Sweep array:

```bash
sbatch scripts/server/04_train_sweep_array.slurm
```

Evaluation:

```bash
sbatch --export=ALL,MODEL_PATH=runs/high_level_ppo/ppo_main_medium/model.zip \
  scripts/server/05_eval_ablation_cpu.slurm
```

Aggregate and plot:

```bash
sbatch scripts/server/06_aggregate_and_plot.sh
```

## Troubleshooting zero-traffic pilot runs

If a pilot run reports `landed_count=0` together with `active_count=0`, do not
continue a longer PPO run.  First verify that the simulation actually contains
traffic:

```bash
python -m terminal_marl.experiments.debug_arrivals \
  --config dynamic_hrl \
  --demand-level medium \
  --policy rule-only \
  --high-level-steps 8 \
  --output-dir runs/debug_arrivals_local
```

If traffic exists (`total_spawned_count>0` and `active_count_max>0`) but
`inner_occupancy`, `final_occupancy`, and `total_landed_count` remain zero,
debug the phase pipeline before launching PPO:

```bash
python -m terminal_marl.experiments.debug_phase_transitions \
  --config dynamic_hrl \
  --demand-level medium \
  --policy rule-only \
  --initial-uavs 1 \
  --disable-runtime-arrivals \
  --high-level-steps 15 \
  --output-dir runs/debug_phase_transitions_local
```

Expected sanity signals:

- the single-UAV phase debug reaches `total_landed_count=1`;
- `debug_arrivals.csv` eventually shows non-zero `inner_occupancy`,
  `final_occupancy`, and `total_landed_count`;
- `vtac_not_eligible` is a normal HOLD reason and should not inflate
  `safety_override_count`;
- `raw_safety_override_count` is available when you need the raw low-level
  rejection count.

Then inspect reward component magnitudes:

```bash
python -m terminal_marl.experiments.reward_diagnostics \
  --config dynamic_hrl \
  --demand-level medium \
  --policies rule-only dynamic-rule high-level-random \
  --episodes 1 \
  --high-level-steps 8 \
  --output-dir runs/reward_diagnostics_local
```

Before uploading a new server package, run a local tiny PPO path check:

```bash
python -m terminal_marl.training.train_high_level_ppo \
  --training-config terminal_marl/configs/training/ppo_debug_local.yaml \
  --total-timesteps 256 \
  --n-envs 1 \
  --vec-env dummy \
  --fail-if-no-traffic \
  --sanity-check-baseline \
  --output-dir runs/high_level_ppo/ppo_debug_local
```

`final_eval.csv` now records `active_count_start`, `active_count_mean`,
`active_count_max`, `active_count_end`, `spawned_count_delta`,
`landed_count_delta`, `total_spawned_count`, `total_landed_count`, and
`safety_override_delta`.  `active_count` is retained as `active_count_end`.

For server jobs, keep `--fail-if-no-traffic` and `--sanity-check-baseline`
enabled for smoke/pilot configurations so that a broken arrival process fails
fast before expensive training begins.

## Interpretation

The trained artifact is a high-level dynamic airspace policy.  It adjusts
airspace/flow intent through `HighLevelHRLWrapper`; it is not a low-level flight
controller and does not output free heading, altitude, or velocity-vector
commands.
