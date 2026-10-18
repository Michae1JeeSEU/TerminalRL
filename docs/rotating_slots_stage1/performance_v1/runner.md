# P0/P1 runtime integration

The executable entry is `scripts/server/run_rotating_slots_performance.py`.
Only `preflight`, `capture`, and `micro` are implemented. There is no performance
qualification, full traffic campaign, window comparison, or publication command.

The script imports only standard-library modules before loading the external
`performance` helper package. Bootstrap verifies the supplied helper deployment
manifest and exact supplied implementation-commit label before any
`terminal_marl` import. It byte-verifies the fixed R0 export, checks the contract
and semantic fingerprint, and then selects exactly one core import root. R0 core
modules come exclusively from that export; helper code comes from the current
checkout. No `sys.modules` replacement or shared R0/OPT core instances are used.

Each result records the fixed reference commit, selected kernel commit, supplied
helper implementation commit, every verified kernel/helper source hash, separate
import paths, config/profile/rules/candidate/numerical hashes, deployment manifest
hash, and generated observation source hash. The exact observation patch and
original/generated source are saved in each attempt. Checkpoint implementation
identity combines the selected kernel bytes, executed helper files and shim
source. Resume has no implementation-migration bypass.

Both R0 and OPT worlds/pools load the same physical YAML path and exact bytes
from the verified frozen reference export. OPT's source semantic projection and
source raw-config hash remain separate from execution raw-config hash. The
source projection must match frozen R0 semantics; only its raw-byte provenance
field is rebound to the actual loaded reference bytes. Parsed config/profile and
rule/candidate/numerical identities must still match.

## Deployment and commands

Generate the deployment manifest locally after completing and verifying source
changes. Supply the actual local deployment commit explicitly; no server Git is
queried or initialized. Manifest creation refuses an SSH session. Existing
manifest files are not overwritten. A source edit requires a new matching
manifest; source/commit mismatch is rejected before an attempt directory exists.

```powershell
# Local only. Replace COMMIT with the exact local deployment commit.
venv/Scripts/python.exe scripts/server/run_rotating_slots_performance.py preflight `
  --reference artifacts/rotating_slots/performance_reference_r0 `
  --deployment-manifest artifacts/performance_deployment.json `
  --implementation-commit COMMIT --write-deployment-manifest

# Tiny actual R0 emergency-reference/energy smoke, full records.
venv/Scripts/python.exe scripts/server/run_rotating_slots_performance.py micro `
  --reference artifacts/rotating_slots/performance_reference_r0 `
  --deployment-manifest artifacts/performance_deployment.json `
  --implementation-commit COMMIT --output artifacts/micro_unique_attempt `
  --inner-workers 0 --core-budget 1 --process-budget 1 --wall-budget 90 `
  --fixture inner-hold --until .4

# Tiny capture smoke; it still generates the full frozen 3600-second manifest.
venv/Scripts/python.exe scripts/server/run_rotating_slots_performance.py capture `
  --reference artifacts/rotating_slots/performance_reference_r0 `
  --deployment-manifest artifacts/performance_deployment.json `
  --implementation-commit COMMIT --output artifacts/capture_unique_attempt `
  --inner-workers 0 --core-budget 1 --process-budget 1 --wall-budget 90 `
  --rate 60 --seed 23 --capture-until .4
```

For server execution use the synchronized source/reference/manifest and a new
output under `$HOME/training_outputs/terminal`. These examples are local bounded
smokes; they do not authorize a formal server run. `--kernel OPT` explicitly
selects current core code and still enforces frozen semantic preflight. P1
parallel execution is available only for R0, using `--inner-workers 1..6` with
explicit `--core-budget` and `--process-budget` (at least workers and workers+1).

On Linux the pool starts before the heartbeat thread; its own sampler is created
only after the workers. Pool resources include startup, tasks, IPC, serialization
and join according to the adapter's accounting. The pool is joined before replay
packaging and source verification. Serial measurements explicitly report parent
CPU only; they do not claim complete process-tree accounting.
The full 0.1-second RSS sample series is saved as `rss_samples.jsonl`; the recent
in-memory sample window is not the only retained evidence. Parallel prefixes are
explicitly labelled `prefix_execution=R0_PARALLEL` with references to bounded
equivalence tests. This label does not claim full P4 qualification.

## Capture and resume behavior

Capture always generates `ArrivalManifest.generate(seed, rate, 3600.)` once,
keeps physics_dt=0.2 and scheduler_period=5, and invokes the original outer phases
`advance(3600, driver)` then `advance(5400, driver)`. The observation shim may
suspend only after an existing completed physical/event step. It never invokes
an extra advance at the requested checkpoint/capture time. The checkpoint keeps
the original advance target and current outer phase.

Optional `--milestones 300 900 ...` requests pinned natural cuts.
`--capture-until` must be at least 60 simulated seconds after the latest milestone
so a future 60-second window remains possible. Non-grid times defer to an
existing cut. Automatic requests use the checkpoint helper's 30sim/300wall
cadence. The store retains recent eight corpus cuts, latest two complete
checkpoints, and pinned milestones. Reaching the requested prefix ends with
`SUSPENDED_CAPTURE`; it is never labelled a completed 5400-second episode.

Resume supplies `--checkpoint path/to/checkpoint-N.json`, the same seed/rate and
source identity, a later capture-until, and a unique new output directory. The
runner requires a retained indexed checkpoint plus its matching `.runner.json`
sidecar and source attempt ledger. Missing/incomplete pairs are rejected. Stream
restore validates/copies the immutable prefix and saved latest frame into the new
attempt, retaining same-time replacement behavior. Old state bytes remain intact.
`--cache-mode cold|warm` selects derived-cache handling.

The sidecar records checkpoint hash, workload, original phase/target, source
attempt and cumulative wall time at the cut. The independent heartbeat and final
attempt ledger record all work consumed after that cut, including discarded work,
checkpoint/flush, pool lifecycle and replay packaging. A resumed attempt adds the
source attempt's final consumed total rather than starting its budget at zero.
Checkpoint load, restore/copy and save/flush spans are reported separately.
The soft wall budget compares that same cumulative total. End-to-end wall and
parent CPU include bootstrap/config/core loading; execution-phase timing remains
separate for diagnosing rollout costs.

A source attempt must be terminal/suspended. For a crashed attempt, an explicit
`--crash-evidence` JSON file must identify its absolute `source_attempt`,
`source_pid`, an `observed_utc`, and `confirmed_not_running: true`. Missing consumed
work after the latest ledger update is not invented: accounting stays
`INCOMPLETE` and cannot support performance qualification. A future supervisor
must supply stronger crash/CPU accounting before that case is qualified.

## Records and limits

Every attempt saves source identity, PID, UTC start/finish, argv, workload,
requested target, actual time, original advance phase/target, driver calls,
stop reason, distinct physical outcome, checkpoint index/sidecars, wall/CPU
measurements, heartbeat, and a validated ReplayBundle when recording remains
valid. A suspended capture's replay is the complete observed prefix and is
explicitly labelled `complete_observed_prefix`; it is not a full episode.

`--wall-budget` is a **soft budget checked at natural safe cuts**. It saves a
checkpoint and returns `PERFORMANCE_TIMEOUT` without changing physical failure
status. A kernel operation can exceed that budget before reaching a safe cut.
There is no independent hard process-tree supervisor in P1; result fields report
`hard_timeout_enforced=false` and `INCOMPLETE_SOFT_SAFE_BOUNDARY_ONLY`. These smokes
do not establish the contract's full timeout acceptance requirement.

The passive 30wall heartbeat reports last complete simulation time, current
advance phase/target, checkpoint/cache context and active reference ring/slot/time.
It lists the six queried pads as a group; it does not claim individual worker
stack capture. Observer write failures propagate as instrumentation failures.

Capture is benchmark-only. Micro accepts `--mode profile` for its explicitly
bounded <=0.4-second invocation and writes a separate `micro.prof` using the
process-CPU timer. Profiling never wraps an entire capture episode.

## Observed bounded evidence

The runner integration suite starts fresh child interpreters using the fixed R0
export. It checks source/import identities, real inner-hold serial/pool full
ReplayBundle equality from zero, actual unmodified R0 hook-off/hook-on and
0.2→0.4 cold/warm interruption equality, full-manifest prefix capture and copied
resume parity, source/commit mislabels, budget and milestone rejection, bounded
profile output, OPT/R0 frozen-byte config binding and service-event parity, and
soft timeout classification.

Initial observed local outputs:

- `artifacts/performance_p1_runner_deployment_20261018a.json`: explicit local
  deployment manifest generated during development; later source edits invalidate
  it as intended.
- `artifacts/performance_p1_runner_micro_20261018a`: actual R0 inner-hold micro
  completed at 0.4 simulated seconds with a validated replay and checkpoint.
- `artifacts/performance_p1_runner_tests_20261018a`: first integration suite,
  8 passed in 40.09 seconds. Later evidence is recorded by the final suite run.
- `artifacts/performance_p1_runner_tests_20261018c`: final integration suite,
  **10 passed in 60.99 seconds**, exit 0:

  ```powershell
  venv/Scripts/python.exe -m pytest tests/test_rotating_slots_performance_runner.py -q --basetemp=artifacts/performance_p1_runner_tests_20261018c
  ```

No 600-second capture or server job was started by this implementation task.
