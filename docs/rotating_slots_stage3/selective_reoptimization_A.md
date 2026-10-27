# Stage A: selective reoptimization

This implementation consumes the immutable `FT-RS-SELECTIVE-1.0` parent and A
specification. It runs only `FULL_H` and `RULE_LOCAL_K8`, with the shared v8
objective and `global_preselection_frozen_v1` frame. B/C, training, the old
descriptor ranker, changed physics, changed budgets, and unknown configuration
keys are rejected. The parser retains the inactive B/C/model/training sections
in the binding with an explicit `NOT_AUTHORIZED_IN_A` designation.

The callback keeps the full visible snapshot and physical background. Only the
selected set owns new candidate variables. The global frame includes all
admitted aircraft demands; revealed upstream requests remain explicit UNKNOWN
`NOT_ADMITTED_NO_FLIGHT_STATE` metadata. The inherited v8 admitted-aircraft
objective does not assign them fictitious flight energy or admission paths.
Actual upstream pending and wait remain in the closed-loop outcomes.

The existing candidate, continuous safety, downstream, latency, world restore,
checkpoint, metrics, replay and campaign-ledger implementations are reused.
No source code is transferred manually. Local Git freezes the reviewed new
commit; Syncthing delivers it to `~/projects/terminal`. Server verification reads
the frozen file bytes and never invokes Git. It records actual configuration,
input identities, dependency versions, CPU binding, and source differences.

## Preregistered inputs

The input map is explicit JSON, with `initial_parent` and `snapshots` keys. Its
actual campaign mapping is `artifacts/rs_select_a_20261027/input_map.json`.
Original root registry, manifests, initial states, and t0 checkpoints come from
`~/training_outputs/terminal/stage3_learning_v1_0_20261023/roots`.

| Root | Seed | Profile | Saved development H checkpoints |
|---|---:|---|---|
| s3r23 | 7324 | medium24_rate150_v1 | 15, 30, 45, 60 seconds |
| s3r24 | 7325 | stress36_rate180_v1 | 60, 120, 180, 240 seconds |

The medium snapshots are under that original campaign's
`evaluation/closed_loop/s3r23/heuristic_reduced`. High snapshots are under
`stage3_future_learning_high300_20261027/evaluation/heuristic_reduced`.
Their exact preregistered hashes are checked by the runner. These are previously
seen development trajectories with different historical source identities;
neither the historical source nor the old normal scheduler is reused as the
new method implementation.

Each of the eight checkpoint states is restored once, and its revealed state
is projected 20 seconds with the existing projection. FULL_H and local each run
once with the same immutable frame and fresh normal caches. Before each arm the
existing `_clear_worker_accelerators()` also resets process-global numerical,
plan-certificate, polynomial-bound and energy accelerators. The observed before
and empty-after cache snapshots are saved without changing cache policies.
Reset and callback/history initialization time are reported separately. Common
projection and frame work is charged to both arms and deducted, with this setup,
from the same 19-second readiness / 15-second search limits. Historical delivered-review
history is explicitly unknown; actual progress is recovered from recorded
maneuver starts and the executed energy ledger. Unknown ages are retained.
Physical active/soft commitments remain intact. For every nonempty proposed
group the source world's existing `prepare_latency_submission` runs exactly
once; its world-owned token is then discarded. Preparation is measured and
classified against the same outer readiness deadline, including all preceding
work. An accepted result that completes late is unusable. UNKNOWN, rejection,
exceptions or an accepted result without a token remain explicit. No timestamp
is shifted, no packet is published and no future world rollout is performed.
The source physical snapshot version must remain unchanged.

The snapshot metric `snapshot_observed_compute_s` includes shared projection and
frame, cache reset, initialization, planning and this one preparation. It is not
full online pipeline latency: affected-traffic revalidation, online service
delay and actual delivery are unobserved and explicitly null/unknown. Full
availability evidence comes from the closed-loop runs.

Closed-loop runs restore each original t0 and use the new latency wrapper.
Each root/method runs once for 300 simulated seconds, serially on physical
CPU12. CP and numerical libraries use one worker/thread. The frozen protocol is
20-second lead, 5-second heartbeat, 1-second preparation, 4-second search handoff,
candidate6/CP8/hint3/soft30/watchdog60 seconds and attempts32/1024. Boundaries and
checkpoints are saved every 5 seconds. Exact-endpoint delivery is censored.

## Explicit commands

After primary review, passing applicable tests, and the local commit, freeze
source locally (Git is prohibited on the server):

```powershell
& .\venv\Scripts\python.exe -B scripts/stage3/run_selective_reoptimization_a.py freeze --source-commit <reviewed-new-commit> --source-freeze artifacts/rs_select_a_20261027/source_freeze.json
```

Wait for Syncthing to deliver that source, freeze, input map and checks. On the
server, set the launch environment before starting Python:

```bash
cd ~/projects/terminal
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=/home/user/envs/terminal_stage2_v1/lib/python3.12/site-packages
PY=/home/user/envs/terminal_env/bin/python
OUT=$HOME/training_outputs/terminal/rs_selective_reoptimization_a_20261027
```

Run the targeted core, campaign and replay checks in the real CP environment,
with a measured elapsed time and JUnit output. The campaign test
`test_small_actual_checkpoint_and_replay_output_roundtrip` performs an actual
checkpoint/restore/ReplayBundle input-output smoke roundtrip before formal
experiments. Unit and serialized-output checks do not replace the CP tests.

```bash
PREFLIGHT=$HOME/training_outputs/terminal/rs_selective_a_preflight_20261027
mkdir -p "$PREFLIGHT"
$PY -B -m pytest tests/test_rotating_slots_selective_replanning.py tests/test_rotating_slots_selective_campaign.py tests/test_rotating_slots_selective_replay.py -q -p no:cacheprovider --basetemp="$PREFLIGHT/final_tmp" --junitxml="$PREFLIGHT/final_checks.xml"
$PY -B scripts/stage3/run_selective_reoptimization_a.py prepare --output "$OUT" --source-freeze artifacts/rs_select_a_20261027/source_freeze.json --input-map artifacts/rs_select_a_20261027/input_map.json --checks-xml "$PREFLIGHT/final_checks.xml"
$PY -B scripts/stage3/run_selective_reoptimization_a.py credit --output "$OUT" --credit-id server_targeted_checks --credit-wall <measured-test-wall-seconds>
$PY -B scripts/stage3/run_selective_reoptimization_a.py snapshots --output "$OUT"
$PY -B scripts/stage3/run_selective_reoptimization_a.py run --output "$OUT"
$PY -B scripts/stage3/run_selective_reoptimization_a.py report --output "$OUT"
```

Use the actual core-test filename if primary review renames it. Each command is
separate; there is no `all` or automatic B transition. Preparation requires a
fresh output, passing non-skipped test evidence, exact original inputs and an
available CPU12. An attempted stage cannot be reset or automatically retried.
The existing CampaignLedger enforces a single 7200-second Stage A automated
wall cap, including measured external tests, preparation, failed subprocesses,
snapshots, closed loops and offline report/audit. Human editing and reading are
excluded while paused. Credit every earlier measured automatic invocation once
by a stable ID, including failed test attempts. Reusing the same ID with a
different duration is rejected. A source revision preserves the ledger and
requires primary review; it does not grant fresh compute.

Snapshot-pair subprocess watchdog is 150 seconds. Each closed-loop subprocess
watchdog is 1200 seconds. If the remaining phase allowance cannot support a full
registered arm, the runner records BUDGET_LIMITED instead of silently shortening
that arm's cap. A closed-loop timeout preserves existing completed boundaries
and permits the paired task if the same cap remains available. It is not a
physical infeasibility or energy failure. Offline audits run in separate bounded
subprocesses with at most 600 seconds each and ten seconds reserved for the
saved-artifact report. Missing/partial audits stay incomplete.

## Evidence and interpretation

`resolved_run_binding.json` binds the actual source and inputs. The immutable
package files are never rewritten. Raw `scheduler_records.json`,
`normal_tasks.json`, `online_services.json`, `dispatch_timings.json`, public
schema `selection_records.json`, authoritative `boundary_*.json`, checkpoints
and ReplayBundles are saved per arm. Internal `replan_ids` and
`fixed_background_ids` are explicitly mapped to public `selected_ids` and
`background_ids`. Public envelopes retain ages, source masks, relations,
candidate counts, pair checks, solver scale/calls and frame identity.

`system_counts.csv` uses each root's largest common authoritative boundary,
not unequal independently reached endpoints. It includes initial/new cohorts,
normal/emergency/failure/reroute, upstream/airborne pending, unresolved airborne
SOC below the stated .25 reserve threshold, energy, holding, ages, late rate,
normal work, dispatch and availability delay. Pending tasks use missing actual
`finished_time`, not a guessed callback status. All final results remain
separately available. Late rate divides explicit readiness/expiry/revalidation
deadline rejections by all finished tasks and excludes pending tasks.

Complete normal work adds the newly charged selective-history service exactly
once to the existing normal-work accounting. Dispatch overlaps normal work and
must not be added twice. Availability delay is elapsed simulated time from
snapshot to actual delivery; pending tasks are censored. World advance minus
dispatch contains integration, execution checks and recording; it is not a
separate pure-physics measurement. Audit/report cost is offline and recorded
separately. Whole-job time also includes setup, checkpoints and output I/O.

Mandatory overflow/full fallback is reported separately from local tasks and
cannot be called local acceleration. Structural snapshot costs retain the same
global demand/frame and report decision, auxiliary, constraint, candidate and
pair counts. Lower energy must be read alongside service and pending counts.
No fixed speedup factor or positive result is an acceptance requirement.

The report lists A01–A20/X01/X02 explicitly with evidence and status. Core-test
mapping remains subject to primary review; a passing JUnit file alone does not
certify all 22 conditions. A20 reuses the existing independent replay audit for
continuous physical separation, trajectory/energy/event ledger and conservation;
it separately checks all saved resource calendars through ResourceCalendar,
full saved PVA and cross-plan C2 transitions. Missing predecessors remain
unknown. Rejected/pending proposal nonexecution uses saved task proposal hashes
and absolute start times, excluding a separately legitimate commitment of the
same hash. Missing proposal hashes do not prove nonexecution.

Open `evaluation/<root>/<method>/replay` in the existing platform:

```python
from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer
controller = ReplayController('/absolute/output/evaluation/s3r23/RULE_LOCAL_K8/replay')
RotatingSlotsReplayViewer(controller).run()
```

The read-only overlay shows S/B/M, recorded blockers, status and effective-time
countdown. It does not change the physical archive, trajectories or commitments.
An actual human review is still required; automatic checks and rendered labels
leave human acceptance `HUMAN_PENDING`.
