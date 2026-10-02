# Saxena vertiport schedule baseline

This directory is a transparent, no-training Python/HiGHS port of the
arrival/departure formulation in
`upstream/test_10_vert_schedule_ArrDep.m` at commit
`87a8e5ba7393d622768a7695770a4fa4e09c26bd`.

The port keeps the source network and parameters: four gates, virtual gate
nodes `c0/c1`, taxi nodes `a/b/c`, TLOF `R2`, overflight node `X`, four fix
directions, class-dependent separation tables, TOT/TAT, and the source
objective weights. It builds sparse linear rows with SciPy `milp` and writes
residuals so a result can be inspected without trusting a summary metric.

## Windows quickstart

Create an isolated environment inside this baseline, acquire the exact
upstream snapshot, and install the pinned runtime:

```powershell
Set-Location D:\7_codespace\terminal\baselines\saxena2024_schedule
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -NoProfile -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

The CLI can run from this package directory. Keep solver artifacts outside the
repository:

```powershell
.\.venv\Scripts\python.exe run.py --fixture arrival --output C:\temp\saxena2024_schedule\arrival
.\.venv\Scripts\python.exe run.py --fixture mixed --output C:\temp\saxena2024_schedule\mixed --time-limit 60
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test command intentionally runs with this package as the current working
directory because the tests import the local `model` and `run` modules.

## Verified server command

The synchronized server uses `/home/user/envs/terminal_env/bin/python` with
the isolated dependency directories `/home/user/envs/waltz2024_deps` and
`/home/user/envs/yang2020_deps`. Run from this package directory and write
artifacts below `$HOME/training_outputs/terminal`:

```bash
ssh lab-server
cd ~/projects/terminal/baselines/saxena2024_schedule
env PYTHONPATH=/home/user/envs/waltz2024_deps:/home/user/envs/yang2020_deps PYTHONDONTWRITEBYTECODE=1 /home/user/envs/terminal_env/bin/python run.py --fixture arrival --output "$HOME/training_outputs/terminal/saxena2024_schedule/arrival"
env PYTHONPATH=/home/user/envs/waltz2024_deps:/home/user/envs/yang2020_deps PYTHONDONTWRITEBYTECODE=1 /home/user/envs/terminal_env/bin/python -m unittest discover -s tests -v
```

The server commands reuse synchronized source and do not use `scp`, `rsync`,
archives, or manual source uploads.

The small explicit fixtures are new deterministic inputs. They are not the
MATLAB `rng(26)` random cohort. Use `--input` for a JSON object containing a
`requests` list when a controlled input is needed:

```powershell
python run.py --fixture arrival --output C:\temp\saxena2024_schedule\arrival
python run.py --fixture mixed --output C:\temp\saxena2024_schedule\mixed --time-limit 60
python run.py --fixture tat --output C:\temp\saxena2024_schedule\tat
python run.py --fixture mixed --formulation upstream-legacy --output C:\temp\saxena2024_schedule\legacy
```

On the synchronized server, use an external output root such as
`$HOME/training_outputs/terminal/saxena2024_schedule/verification_20261002`.

Each run writes `inputs.json`, `result.json`, `constraint_checks.json`, and
`node_times.csv`. The JSON records original/effective request times after the
source-style same-direction and 6.5-second different-direction preprocessing,
the formulation selector, solver status, time limit, and evidence metadata.

`paper` is the default wake formulation and adds both ordered pair inequalities
from the paper equation. `upstream-legacy` retains the MATLAB helper's
one-sided `f1 < f2` loop. The source default `gateCapacity=3` auxiliary `y1`
rows are used by default; `--gate-capacity 1` selects the source option-1
branch.

Install the pinned runtime from `requirements.txt`. `bootstrap.ps1` downloads
the exact upstream archive only when the local `upstream` directory is absent;
it never overwrites an existing snapshot. The upstream repository has no
declared license in its API metadata and no license file was included, so the
new port does not redistribute upstream source in tracked files.

Native MATLAB and its Optimization Toolbox have not been run here. Solver
outputs should therefore be reported as newly run Python evidence until the
same fixtures are checked in MATLAB.

Observed verification artifacts are synchronized under
`D:\7_codespace\training_outputs\terminal\saxena2024_schedule\verification_20261002`
and the corresponding server root
`$HOME/training_outputs/terminal/saxena2024_schedule/verification_20261002`.
Seven unit tests passed across seven selected explicit fixtures, all with
status `optimal`: `arrival_final2`, `departure_final2`, `tat_final2`,
`mixed_final3`, `legacy_final3`, `capacity3_final2`, and
`capacity1_two_final`. Across those artifacts the maximum constraint residual
was `3.84e-12` and the maximum integrality error was `9.46e-13`.

The older `capacity1_final` artifact used a four-request TAT fixture and was
`infeasible`. The selected two-request `capacity1_two_final` fixture was
`optimal` and had maximum gate occupancy `1`. Native MATLAB was absent and was
not run; the 2026 final journal model remains unverified. These checks do not
establish a scientific performance result.

The paper-side wake interpretation follows the accessible
[`arXiv:2408.01152v1`](https://arxiv.org/html/2408.01152v1) preprint, C10
Eq. (12). The related final journal record is Saxena, Kuri, Prabhakar, and
Yadav, *The Aeronautical Journal* 130(1350), 2434–2460,
[DOI 10.1017/aer.2026.10173](https://doi.org/10.1017/aer.2026.10173).
The final journal full model has not been independently checked; this port is
anchored to the fixed MATLAB snapshot documented in `UPSTREAM.md`.
