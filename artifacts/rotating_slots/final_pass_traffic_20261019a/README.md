# Fixed one-pass traffic driver — prepared, not launched

`traffic_pass.py` is an orchestration artifact outside production deployment identity.
It launches the existing `performance.paired.collect(kind='runner')`, which executes
the exact known `scripts/server/run_rotating_slots_performance.py traffic` CLI in
fresh `-I -B` children, native threads 1, explicit Linux CPU affinity and nice 10.
No production/test source, physical policy, deployment extra_paths or semantic
identity is changed by this artifact. The driver's SHA is recorded separately.

The fixed 15 cases are rates 60/120/180 and seeds 11/23/47/71/101. Order starts
`(60,23), (120,11), (180,71)` then the remaining cases in lexicographic order.
Each case runs R0 then OPT, sequentially, once from time zero. There is no retry,
reseed, tuning, checkpoint resume, or automatic noisy second group. This version
has no batch scheduler or concurrent launcher.

The user amendment makes numeric speed/wall/CPU thresholds report-only. The
default 7200 seconds is a per-run resource cap, passed both to the existing hard
kernel deadline and the runner's natural-cut soft budget. A hard timeout retains
the actual guard/exit evidence and incomplete descendant CPU lower bounds, then
continues with the remaining fixed jobs. A soft-budget stop is also incomplete
physics. Neither is represented as a physical failure or a completed episode.
Implementation/source/integrity errors or complete-pair semantic differences stop
the pass immediately; the other jobs remain missing. Actual physical failure is
a valid terminal only when the real result and validated replay support it.

## Modes and explicit inputs

Use the environment's Python in a fresh `-I -B` process. `self-check` is synthetic
only and does not import core, call the collector or validate any real replay:

```powershell
./venv/Scripts/python.exe -I -B artifacts/rotating_slots/final_pass_traffic_20261019a/traffic_pass.py self-check
```

`plan` and `run` accept the same explicit arguments. `plan` only prints all 30
commands; it does not claim source verification or launch physics. Example shape
for a later primary-approved Linux command (values are placeholders, not a launch):

```sh
python -I -B artifacts/rotating_slots/final_pass_traffic_20261019a/traffic_pass.py plan \
  --project /home/user/projects/terminal \
  --reference /absolute/byte-verified-R0-export \
  --deployment-manifest /absolute/shared-deployment.json \
  --implementation-commit EXACT_40_HEX_COMMIT \
  --r0-workers 0 --opt-workers 2 --cpus 2 3 \
  --hard-budget 7200 \
  --output /home/user/training_outputs/terminal/PRIMARY_SELECTED_NEW_DIRECTORY
```

Worker choices are independently explicit for R0 and OPT: 0/1/2/4/6. Core budget
is the explicit CPU-list length; kernel process budget is workers+1. The collector
is an extra process, reported separately, with no new process-cap gate. `run`
requires Linux, available CPUs and a new output outside the synchronized project.
It refuses overwrite/resume. Primary selects real inputs, worker counts, affinity,
budget and output only after reviewing this code.

The outer artifact compares
`provenance._source_files(project, extra_paths=deployment['extra_paths'], modules={})[0]`
with the existing deployment map before/after each job and at pass completion.
It labels this a **file-inventory check**, not runtime-import proof. It also checks
the frozen R0 export. The actual child runner retains its unchanged strict
`bootstrap`/`_verify_finish` import/source validation as authority. The driver does
not add itself to the frozen deployment, strip imported modules, or weaken those
child checks.

For each successful result, a separate fresh `-I -B` post-collector subprocess
uses the unchanged runner bootstrap for that selected kernel, asserts the actual
replay module's path under its selected core root, invokes the real
`ReplayBundle.validate`, then verifies finish provenance. It checks replay time
bounds start at exact zero and end at the result's exact stop. This validation
does not advance physics. Raw stdout/stderr and numeric exit are retained. This
subprocess is additional orchestration cost, separate from collector CPU/RSS.

```sh
python -I -B artifacts/rotating_slots/final_pass_traffic_20261019a/traffic_pass.py summarize \
  --campaign /absolute/existing-pass-output
```

`summarize` reads the saved plan/observed rows, checks persisted result, collector
and replay-member hashes, and does not rerun the simulation or replay validator.
Missing jobs remain missing; no new launch is attempted.

## Outputs and metric scopes

The output has `run.json`, `summary.json` and one `observed.json` per attempted
kernel job. Each job retains the collector directory and runner attempt intact.
Summary reports 15-case/30-job coverage, outcomes, complete pairs, exact differences,
timeouts and missing jobs. Complete pairs compare all six existing replay data
members, exact stop `float.hex`, full exact physical-outcome/failure metadata,
stop reason and semantic fingerprint. Physical failures can match across kernels;
timeouts never count as completed-physics pairs. No numeric PASS threshold or
`PERF_ACCEPTED` claim is produced; human observation remains pending.

Rows retain these distinct scopes:

- Driver end-to-end/job wall includes collector, post-validation and output checks.
  Driver CPU is parent-only; post-validation child CPU/RSS is not measured here.
- Collector full child wall spans spawn/join; collector parent CPU and verified
  joined kernel/worker CPU remain separately observed. Timeout/error CPU is
  incomplete with observed reaped CPU retained as a lower bound.
- Kernel `timing` keeps actual end-to-end and named rollout/record-flush spans.
  Nested spans are not summed; kernel resources keep the real startup/join scope.
- Collector+kernel and kernel+worker RSS samples stay separate estimates. Unmatched
  peaks are not added and neither is declared a strict RSS bound.

Complete-pair ratios use R0/OPT collector-child full wall, kernel end-to-end wall,
the single named rollout/record-flush wall and complete collector+kernel tree CPU.
Unavailable values remain null. The overall driver does not claim complete total
CPU by adding nested or unmeasured costs. `run` exit 1 means stopped integrity/
semantic evidence; timeouts and numeric shortfalls alone do not fail the pass.

Preparation reads used the actual runner CLI/result source, replay API and existing
`artifacts/rotating_slots/runner_traffic_tests_20261019c/.../result.json` tiny outputs.
The synthetic self-check validates only fixed coverage/order, command construction,
physical-failure metadata handling, timeout coverage and exact pair differences.
No formal traffic job has been launched by this preparation.
