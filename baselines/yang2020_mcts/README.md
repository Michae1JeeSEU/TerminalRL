# Yang--Wei 2020 MCTS baseline

This directory provides a headless, bounded execution wrapper around the
original Yang--Wei multi-aircraft MCTS source. It reuses the original state,
node, search, and three simulator implementations from `upstream/` and does
not retrain anything. See [UPSTREAM.md](UPSTREAM.md) for the exact source
commit and the only runtime compatibility guard.

## Windows quickstart

From a clean checkout, create an isolated environment in this baseline
directory, acquire the pinned upstream snapshot, and install the pinned
headless requirements:

```powershell
Set-Location D:\7_codespace\terminal\baselines\yang2020_mcts
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

The CLI is then invoked from the project root so package imports and output
paths are unambiguous:

```powershell
Set-Location D:\7_codespace\terminal
baselines\yang2020_mcts\.venv\Scripts\python.exe baselines\yang2020_mcts\run.py --case crossing --seed 2 --max-steps 120 --output C:\temp\yang2020\crossing
baselines\yang2020_mcts\.venv\Scripts\python.exe baselines\yang2020_mcts\run.py --case hex --seed 2 --max-steps 120 --output C:\temp\yang2020\hex
baselines\yang2020_mcts\.venv\Scripts\python.exe baselines\yang2020_mcts\run.py --case hex_two_stage --seed 2 --max-steps 120 --output C:\temp\yang2020\hex_two_stage
baselines\yang2020_mcts\.venv\Scripts\python.exe -m unittest discover -s baselines\yang2020_mcts\tests -v
```

The output directory must be outside the repository. These commands do not
upload source files or modify a server checkout.

## Verified server command

The synchronized server uses `/home/user/envs/terminal_env/bin/python` and
the isolated dependency directories `/home/user/envs/yang2020_deps` and
`/home/user/envs/waltz2024_deps`. Run from the synchronized project root and
write artifacts below `$HOME/training_outputs/terminal`:

```bash
ssh lab-server
cd ~/projects/terminal
env PYTHONPATH=/home/user/envs/yang2020_deps:/home/user/envs/waltz2024_deps MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 /home/user/envs/terminal_env/bin/python baselines/yang2020_mcts/run.py --case crossing --seed 2 --max-steps 120 --output "$HOME/training_outputs/terminal/yang2020_mcts/crossing"
env PYTHONPATH=/home/user/envs/yang2020_deps:/home/user/envs/waltz2024_deps MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 /home/user/envs/terminal_env/bin/python -m unittest discover -s baselines/yang2020_mcts/tests -v
```

The server commands reuse the synchronized source and do not use `scp`,
`rsync`, archives, or manual source uploads.

The source defaults are preserved: planning occurs every five simulator steps;
the crossing loop begins planning with counter 1, and both hex loops begin at
counter 0. The crossing, hex, and two-stage distance gates are respectively
`3×`, `2×`, and `5×` the configured minimum separation. Full/lite budgets and
depths are read from the source configuration (`100/30`, `3/2`) and
`simulate_frame` is `10` unless a named diagnostic override is supplied.

The wrapper writes `actions.csv` and `metadata.json`. The metadata records the
seed, source file hashes, exact overrides, MCTS decision count, simulator
counters, wall time, and whether the run naturally ended or was truncated at
`--max-steps`. `--cohort-limit` stops new hex aircraft after the requested
finite cohort and is an explicitly altered workload. Neither truncation nor a
finite cohort is a paper performance result.

The crossing case reports its initial ten aircraft as `generated`, while
`source_id_tracker` preserves the source field (which remains zero for that
fixed-cohort environment). For sector cases, `generated` is the source
`id_tracker`. Simulator counters are retained in raw form as `raw_conflicts`
and `raw_nmacs`. The upstream agents print `raw_conflicts / 2` and
`raw_nmacs / 2`; the wrapper records these as source-reported values and
convenience fields without claiming they are unique physical pair counts.

Rendering and debug breakpoints are disabled by default. `--debug` is exposed
only for source diagnostics and can invoke the upstream debugger on a conflict;
headless verification should omit it.

## Observed verification

On the configured server (Python 3.12.3, isolated dependencies under
`/home/user/envs/yang2020_deps`), the nine wrapper unit tests passed. With seed
2 and the original MCTS budgets, the crossing case ran with `--max-steps 2000`
and naturally ended after 774 steps: 10 generated aircraft, 10 goals, zero
active aircraft, 1,172 MCTS decisions, and about 13.46 seconds. The hex case
ran for its bounded 120 steps, generated 12 aircraft, recorded 156 MCTS
decisions, and was explicitly truncated after about 0.60 seconds. The
hex-two-stage case likewise ran 120 bounded steps, generated 12 aircraft,
recorded 138 MCTS decisions, and was explicitly truncated after about 0.54
seconds. Shapely 2.0.6 was present in the isolated dependency directory for
the sector constructor.
