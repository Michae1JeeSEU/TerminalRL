# Stage2 campaign infrastructure

原始首轮正式运行已 **INVALID_STOP**，最后共同窗口840秒，完整3600秒结果未形成；
失败恢复及CLI修复已通过统一服务器回归（218 passed、1 skipped）。实测质量、正常/完整dispatch耗时、840秒审计及
首60秒历史见 [first_campaign_status.md](first_campaign_status.md)。
下面保留实现接口和运行前验证说明，不能用旧测试通过覆盖本轮暴露的异常。

用户已授权修正版在独立目录从同一初态重新成对验证，完整继承旧轮已用预算；
正常启动已确认、首个共同60秒已完成，仍在运行；接续heartbeat已创建，见
[repaired_campaign_status.md](repaired_campaign_status.md)。

This document describes implemented interfaces, not a completed campaign or a
human acceptance record. The formal input is
`new10.16/rotating_slots_stage2_baseline_v1_0/`. The experiment policy is
`rs_s2_loss_continue_throughput_v1_0`; Stage1 configuration and its strict
checkpoint codec retain their original identities.

`terminal_marl/core/rotating_slots/stage2_campaign.py` builds the prescribed
36 holding-aircraft snapshot, with independent Python Random streams 2071
(inverse-CDF truncated normal SOC) and 2072 (service categories), plus ground
services at pads 0/2/4. `seed_hold` and `seed_service` certify the initial geometry
and resources with the existing continuous kernel. The materialized seed71
Poisson manifest has a right-open 3600-second release window. Both methods read
its same bytes. The normal planners receive the executor's revealed snapshot,
never the manifest. Initial normal references are separately bounded and record
UNKNOWN with absent margins whenever a normal chain was not found; the fixture
is never altered to improve those references.

`scripts/server/run_rotating_slots_stage2.py` exposes `freeze-provenance`,
`prepare`, `run`, `resume`, `status`, and `report`. Freeze provenance **locally**;
the operation records the actual Git revision, actual tracked diff and hashes
of active core sources, the checkpoint primitive, runner and replay adapter.
Prepare and run consume that JSON and verify bytes, with no server Git calls.
The first successful dependency probe locks the exact environment in
`environment.lock.json`; subsequent runs reject a changed environment. The
server requires numpy/scipy/PyYAML/pytest/matplotlib and
`ortools==9.15.6755`. PyVista is only required for local viewing. The separately
captured server environment is preserved in
`requirements-rotating-slots-stage2.lock`; no runtime installation occurs.

Example sequence (paths are chosen by the primary agent):

```text
LOCAL: python scripts/server/run_rotating_slots_stage2.py freeze-provenance --output <local-fingerprint.json>
SERVER: python scripts/server/run_rotating_slots_stage2.py prepare --output <training-output-directory> --provenance <synced-fingerprint.json> --preparation-wall-seconds <measured-cost>
SERVER: python scripts/server/run_rotating_slots_stage2.py run --output <same-directory>
SERVER: python scripts/server/run_rotating_slots_stage2.py resume --output <same-directory>
```

Concrete commands from the repository root, once the primary agent has frozen
the reviewed sources and accumulated preparation cost:

```powershell
.\venv\Scripts\python.exe scripts/server/run_rotating_slots_stage2.py freeze-provenance --output artifacts/stage2_first_campaign_v1_provenance.json
```

The source fingerprint travels through the existing Syncthing project sync.
On the server, use the isolated environment and an output directory outside
the source checkout. `PREPARATION_WALL_SECONDS` must be the primary agent's
measured total, supplied once; the command refuses an unset value.

```bash
cd ~/projects/terminal
PY=/home/user/envs/terminal_stage2_v1/bin/python
OUT="$HOME/training_outputs/terminal/stage2_first_campaign_v1_20261021"
"$PY" scripts/server/run_rotating_slots_stage2.py prepare --output "$OUT" --provenance artifacts/stage2_first_campaign_v1_provenance.json --preparation-wall-seconds "${PREPARATION_WALL_SECONDS:?set the measured preparation total first}"
"$PY" scripts/server/run_rotating_slots_stage2.py run --output "$OUT"
"$PY" scripts/server/run_rotating_slots_stage2.py status --output "$OUT"
```

After an actual interruption, use `resume --output "$OUT"` with the same runner,
environment and directory. Do not rerun `prepare`. The runner automatically
exports and audits at its final common boundary within the remaining budget.
An explicitly requested later `report --output "$OUT"` rebuilds the comparison
and curves from saved summaries, without resimulating aircraft.

Root artifacts include `campaign.json`, `budget.json`, `manifest.json`,
`fixture.json`, `initial_references.json`, `environment.lock.json`,
`provenance.json`, `resource_observation.json`, `comparison.json`,
`comparison.csv` and `comparison_curves.png`. Each `energy_slack/` and
`rh_cp_sat/` directory contains `summary_NNNN.json`, `checkpoint_NNNN.json`,
`checkpoint.json`, `states.jsonl`, `decisions.jsonl` and
`dispatch_timings.jsonl`; completed exports use `replay_NNNN/`,
`audit_NNNN.json` and `archive_status.json`. Missing or budget-incomplete
exports/checks retain their actual status. The five curves show cumulative
touchdowns/losses, airborne/upstream populations, occupied pads, cumulative
energy, and full normal callback wall from `dispatch_timings.normal_wall_s`.
Inner algorithm wall remains available separately in decision JSONL and
`inner_algorithm_timing`.

After outputs sync locally, an actual human may open an exported archive with
the established PyVista adapter. Replace the path with an existing export:

```powershell
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_first_campaign_v1_20261021\energy_slack\replay_0060')).run()"
```

The preparation credit is supplied once and includes measured tests and earlier
environment/precompute work. `budget.json` persists the original campaign epoch
and an elapsed lower bound. Restart cannot reset the 10-hour allowance. The
9.5-hour compute deadline leaves at most 30 minutes for export, audit and plots;
the supervisor terminates a child at its deadline and uses its last completely
published checkpoint. Incomplete `.partial` files do not establish progress.

Two resident episode processes bind to different physical cores determined by
Linux package/core topology within the allowed affinity mask. Numerical library
threads and CP workers are one. If two physical cores cannot be identified, the
same two resident worlds execute alternating 60-second blocks. Each process
retains its own numerical caches, reference state and prediction suffix across
blocks. The parent requests the next block after both methods publish the
current boundary, forming the first comparison at60 and continuing those same
worlds. No t=0 restart follows the first comparison.

Settle-only boundaries integrate a remaining positive interval even when it
is at most1e−9 seconds, then settle exactly. That tiny physical interval runs
the existing energy-failure and continuous-safety checks without a fresh normal
dispatch. The literal3600/3600.1-second final tests cover this late-time
roundoff case with real arrival, inbound, transfer and final trajectories.

The narrow Stage2 checkpoint adapter uses the existing whitelisted StateCodec,
hexadecimal float representation, complete executor field inventory, exact
calendars/trajectories/ledger, pending event phase, warm caches, scheduler state,
and validated stream prefix. It does not change Stage1's codec or serialize
callables/pickle. Checkpoint files are hashed and atomically replaced. A
published `summary_NNNN.json` follows complete checkpoint/log writes. Replay
export copies the selected checkpoint's stream prefix instead of truncating
the continuing live stream. Only the latest boundary complete for both methods
enters `comparison.json`, `comparison.csv` and the five curves. Faster results
remain available separately. Counts are never extrapolated to3600.

The comparison includes failures, actual touchdowns, cohorts, unresolved
aircraft/requests, costs, energy, cumulative time and lexicographic rank. Normal
timing uses the full executor callback boundary; the inner algorithm timing is
retained separately. JSONL records preserve all normal and dispatch calls;
quantiles retain raw values, sample counts, overruns and small-sample notices.
Block wall/CPU, startup/restore, checkpoint, executor counters, non-dispatch
executor wall, export and audit costs have separate fields. Nested planner
timers must not be summed as if they were disjoint stages.

The shared candidate/reference allocation uses the first3 seconds of the
existing5-second soft budget, reserving the remaining2 seconds for selection.
This is a common runtime budget partition derived from the frozen overall and
solver limits; it does not change the64/2048/8 candidate caps or give one method
extra candidates. Candidate and selection time remain measured in full.

Both methods use the same lazy candidate generator. It interleaves aircraft
expansion in slack order, keeps already materialized partial chains in a
frontier ordered by estimated normal arrival, energy and signature, and defers
unconstructed successor cursors until the materialized frontier is exhausted.
The ordering is over the available frontier; it does not assert that every
unconstructed continuous alternative was globally sorted or searched. Existing
normal commitments and KEEP/WAIT remain available at budget truncation.
Energy-slack selects a compatible multistep column sequentially without joint
backtracking. RH-CP-SAT selects among those finite columns jointly, constructs
its own timed Energy-slack hint, and uses at most3 solve/validate/refine rounds
under its shared2-second solver allowance. Only certified next prefixes enter
the executor; unsubmitted suffixes retain no resource or execution authority.

Decision JSONL preserves each CP phase's raw solver status, objective, bound,
branches/conflicts and validation/prefix statuses. `UNKNOWN` is not a physical
infeasibility claim. `LEX_LIMITED` retains a certified incumbent when available;
`CATALOG_LEX_OPTIMAL` applies only to the checked finite, quantized catalog when
all required phases meet the implementation's proof checks. Neither label
asserts continuous-space optimality, safety certification or general capacity.

Stage2 adds content-hashed members to the existing ReplayBundle:
`decisions.jsonl`, `dispatch_timings.jsonl`, `failure_archive.json`,
`tentative_predictions.json`, and `stage2_provenance.json`. Tentative plans are
separate from authoritative trajectories/resources and have an `as_of` time.
The existing PyVista adapter displays a separate tentative layer, black isolated
loss markers and the archived ledger, plus normal/full dispatch wall in its
HUD. Isolated markers do not enter the live-traffic separation display. Human
acceptance begins `HUMAN_PENDING`; automated tests do not change it.

Local campaign tests exercise the frozen fixture, shared right-open manifest,
FIFO gate blocking, lexicographic failure/throughput order, pending counts,
persisted budget/common checkpoint boundaries, exact checkpoint restoration,
resident60→120 continuation on an empty integration fixture, and replay/HUD
data. These are bounded clause checks for T01/T22/T23/T24/T26/T27/T28; they do
not independently constitute all28 formal tests. The empty continuation test
does not measure busy36 performance, the PyVista adapter test does not certify
human visual acceptance. The pre-run server validation and the formal run's
last common840-second results are recorded in [first_campaign_status.md](first_campaign_status.md).
This run ended with `INVALID_STOP`, not a completed3600-second or budget-limited
result. Its failure-recovery path exposed a coverage gap; the finite fixes are
verified by the final scoped server regression (218 passed,1 skipped).
Heartbeat was deleted and no run process remains.
The primary agent verified the checkpoint-specific historical slot-overlap
diagnosis, the local CLI fix (23 passed, including9 added CLI cases), and the
slot-origin/local-isolation fix (27 world/audit tests passed).
The original invalid run's exit0 record is retained, not rewritten.
Final recovery validation collected219 tests:218 passed,1 skipped,0 failures/errors;
pytest48.49s,process wall48.767s over UTC2026-10-20T19:49:20Z–19:50:09Z.
The single skip remains the installed-OR-Tools missing-dependency negative case;
both real-CP checks passed. The first recovery regression (217 passed,1 skipped,
1 failed) is retained separately: its outdated upstream assertion used admission
as the occupancy start instead of the actual target capture. The corrected test
still checks original release/admission identities and preserves capture.
The final evidence is `regression_final.log/xml/timing.txt` in the independent
`stage2_recovery_validation_20261021` output directory; XML SHA256
`73dd80fc156858d39cb94e82cc2950bec14a5d9f44cefa7cd88e75f41fed1d66`.
This validates the finite fixes; it does not turn the original invalid840-second
campaign into a completed3600-second run, and no formal campaign was rerun.

Observed local validation during this implementation: campaign12 tests passed
in3.36s; the existing replay/reservation35 tests passed in8.43s; their combined
47-test invocation passed in10.61s. The subsequent required-members regression
passed in1.06s and the amended-admission HUD regression passed in2.66s. Earlier
development invocations, including two corrected assertion failures, consumed
additional preparation time; all observed pytest durations for this task sum
to44.52s, with process startup overhead additional. Credit at least60s for this
task's initial test work when preparing the formal campaign. The later literal
cutoff development added two failed test invocations(4.27s and6.85s), a3.28s
observation probe, and the corrected two cutoff cases plus executor module
(14 passed in11.40s), with process startup overhead additional. Use a
conservative cumulative **100s preparation allowance for this worker's test
and probe work**, including the original60s; do not add100s on top of60s.
This is separate from the primary agent's environment setup and other workers'
measured costs. The complete T01–T28 assertion map and remaining coverage
boundaries are recorded in `acceptance.md`; the final scoped server regression
observation is recorded below.

The primary agent has reviewed preparation-stage server evidence in the isolated
environment: real-CP targeted checks recorded19 passed and1 skipped. The skip
is the missing-dependency negative test, conditional on OR-Tools being absent.
On the busy36 fixture, one Energy-slack callback took4.39349 wall seconds and
one RH-CP-SAT callback took4.64497 wall seconds. Each built95 columns with1286
new attempts and catalog truncation, and each certified/submitted2 actual
normal final prefixes. The world was not advanced. CP recorded6 phase solves
over3 lazy conflict-refinement rounds, rejected unsafe proposals, and retained
`LEX_LIMITED`. This is a preparation integration observation, not an executed
throughput comparison, a complete catalog proof or5-second certification.
The primary agent subsequently verified the final server log and XML:
208 collected,207 passed,1 skipped,0 failures/errors. Pytest reported46.87s,
with47.135s process wall over UTC2026-10-20T17:28:26Z–17:29:13Z. This covers the
four Stage2 modules and existing audit/replay/world/upstream/targeted-world/
practical-world/performance-checkpoint tests. Real-CP T09 ran and passed in
1.696s; T15 ran and passed in0.367s. The only skip was the conditional
missing-dependency negative test. Both literal3600/3600.1-second cutoff cases
were confirmed. Automated validation passed for this tested scope; the28-item
mapping remains bounded rather than an exhaustive proof of all clauses.
Evidence is `preparation/final_validation.log`, `final_validation.xml` and
`final_validation.timing.txt` under the formal output root. XML SHA256:
`7c8742a85742563c37c186fb899b238f5ea586ae03156dad4303e6a5d4c05402`.
Formal run artifacts use
`/home/user/training_outputs/terminal/stage2_first_campaign_v1_20261021`, synced
to `D:\7_codespace\training_outputs\terminal\stage2_first_campaign_v1_20261021`.
Human acceptance remains `HUMAN_PENDING`; the final source revision must come
from the actual committed provenance rather than a prefilled hash.
