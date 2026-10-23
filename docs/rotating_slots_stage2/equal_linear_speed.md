# Equal linear speed profile

The linear-speed wave has completed all three fresh 0-to-600 s scenarios for both
methods. All six exported audits passed with zero issues and CERTIFIED continuous
separation. All three fresh angular control campaigns have launched and their six
method streams passed startup verification at 22:21:23 UTC+08:00 on 2026-10-23.
Their completion, audits and matched rotation-profile comparison remain pending.
The completed result table below describes the linear wave only.

`rotation_profile` is independent of `service_profile`. The default
`equal_angular` preserves the two existing frozen YAML payloads and their hashes.
`equal_linear_21_v1` selects a full frozen baseline-service or half-service YAML
with common tangential speed 21 m/s. Clockwise angular speeds are outer 0.035,
middle 21/450, and inner 0.07 rad/s. The old common-angular field remains as the
unchanged legacy reference; runtime ring motion uses `ring_angular_speed`.
The chosen 21 m/s preserves the existing outer-ring speed, 600*0.035=21 m/s,
while changing the middle and inner ring clocks. The user explicitly confirmed
this choice of 21 m/s. The rotation comparison jointly changes relative-phase
dynamics, middle/inner tangential speeds (15.75/10.5 to 21 m/s), and centripetal
accelerations; it is not an isolated phase ablation.

Only rotation fields differ from the corresponding old service payload. Ring
geometry/capacity, vehicle and safety limits, SOC and arrival streams, service
durations, candidate grids, rescue_need_v7, and quality_v5 budgets remain fixed.
The selected quality_v5 candidate/hint/solver/soft/watchdog limits are
12/3/8/30/60 seconds, max attempts 2048, per-aircraft attempts 64 and refinement
rounds 64. This profile does not change them.

Frozen semantic SHA256 values:

| Service | Rotation | Config SHA256 |
|---|---|---|
| baseline | equal_angular | 0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53 |
| service_half_v1 | equal_angular | 5582b671279c0162dd3f6f72da12368a5b2ecffd12d97f5d59eaf945a47ecf87 |
| baseline | equal_linear_21_v1 | 5709a991fb61af9ce2ed572e29cb10b7152c42bfe2e3659bd657cf74a38f9664 |
| service_half_v1 | equal_linear_21_v1 | 96a28b68491a4705af9943e0f4dd8b1dfde94956745a4635c8e144e147121c38 |

A transfer beginning at t with duration T matches the target slot at t+T.
Writing source/target angular speeds as ws/wt, the phase correction relative to
continuing the source clock is `(target_phase(0)-source_phase(0) +
(ws-wt)*(t+T)) mod 2pi`. The two existing deterministic unwrap branches are
retained. Distinct endpoint angular velocities enter the C2 quintic Hermite
polynomial; the shared-clock case retains its original coefficient arithmetic.
The vector rejection screen uses the same endpoint conditions. Transfer screen
keys include absolute epoch for unequal clocks. An aligned start at another
epoch cannot reuse a rejection from the base epoch. Numeric feasibility and
energy reuse keys include actual angular polynomial coefficients.

Stable slots on different clocks do not have constant distance. Their continuous
separation proof uses the concentric radial/vertical lower bound, at least
sqrt(150^2+40^2) m for adjacent rings. Same-ring slots retain the common-clock
proof. Tail horizons use the longest ring period (still 2pi/0.035 seconds in both
profiles); safety does not infer cross-ring periodicity from that horizon.
Canonical final-energy phase representatives use the source ring angular speed.
Stable energy uses the selected ring/segment clock. Plan continuation before and
after a maneuver resolves its existing config_hash to a known frozen payload,
including after JSON/checkpoint/replay roundtrip. No plan serialization field was
added. Campaign/provenance/checkpoint and replay index metadata persist and
validate the rotation selector plus frozen configuration hash.

## Validation commands

From the new checkout locally:

```powershell
& D:\7_codespace\terminal\venv\Scripts\python.exe -m pytest tests/test_rotating_slots_equal_linear.py tests/test_rotating_slots_physics.py tests/test_rotating_slots_practical_numeric.py tests/test_rotating_slots_targeted_safety.py tests/test_rotating_slots_targeted_energy.py tests/test_rotating_slots_stage2_planning.py tests/test_rotating_slots_stage2_campaign.py tests/test_rotating_slots_stage2_branch_v7.py tests/test_rotating_slots_stage2_cli.py -q
```

The local venv lacks OR-Tools. The prepare serialization test explicitly stubs
`environment_lock` and initial reference search. It verifies frozen fixture,
profile identity, checkpoint, replay and corruption rejection; it does not verify
a real CP solve. Tests that require OR-Tools must execute using the existing
server `/home/user/envs/terminal_stage2_v1/bin/python`, with ortools==9.15.6755.
No dependency installation is required by this change.

## Observed validation evidence

Server validation ran on 2026-10-23 from 18:01:05 to 18:01:23 Asia/Shanghai
using the existing pinned environment: **103 passed, 1 skipped, 1 deselected**.
Pytest reported 17.44 s; the process elapsed time was 17.61 s and exit code was 0.
The skipped case exercises the missing-dependency negative path; the deselected
case is the local Git provenance-prepare test. Evidence files are
`/home/user/training_outputs/terminal/equal_linear21_validation_20261023/validation.log`,
`results.xml`, and `exit_code.txt` in that same directory.

Both actual tiny-method runs reached t=0.4 s with one submitted prefix, one normal
decision and one dispatch each, conservation true, and no failed/invalid world.
RHCP recorded five actual CP phases, all OPTIMAL. Checkpoint and replay roundtrips
validated. These observations cover the tiny fixture; they are not formal
0-to-600 results or evidence of comparative scheduling quality.

The local consolidated run reported **169 passed, 4 skipped, 8 setup errors** in
62.02 s. All eight setup errors are from
`tests/test_rotating_slots_performance_energy_queries.py` accessing the unavailable
historical fixture
`artifacts/rotating_slots/performance_reference_r0/.reference_manifest.json`.
That fixture was not copied or regenerated. The final focused profile run reported
**21 passed, 2 skipped** in 3.07 s; its two actual-CP cases were skipped locally
because the local venv lacks OR-Tools, and then ran in the server validation above.
Local logs are `artifacts/rotating_slots/equal_linear21/local_targeted_tests.log`
and `artifacts/rotating_slots/equal_linear21/new_profile_final_tests.log` in the new
checkout. The local prepare serialization test uses the explicitly documented
stubs; actual CP evidence comes from the server tiny-method runs.

## Formal comparison protocol and current status

The three `equal_linear_21_v1` formal scenarios below completed the requested
t=600 boundary, each freshly initialized at t=0 for both EnergySlack and RHCP.
Matched fresh `equal_angular` controls have launched using the same scenario
inputs, CPU pairs and budgets after fresh resource preflight. Startup is verified;
their t=600 results and audits remain pending. All conditions retain
`service_half_v1`, `rescue_need_v7`, `quality_v5`, the frozen SOC/arrival streams,
and the existing geometry, safety/vehicle limits, service durations and solver
budgets.

| Scenario profile | Initial airborne count | Arrival rate/h | Linear-wave method CPU pair |
|---|---:|---:|---|
| diagnostic18_rate120_v1 | 18 | 120 | 4, 6 |
| medium24_rate150_v1 | 24 | 150 | 8, 10 |
| stress36_rate180_v1 | 36 | 180 | 0, 2 |

Launch preflight confirmed these CPU assignments for the completed linear wave,
with no overlapping prior computation. The angular control wave subsequently
repeated preflight and reused the same pairs. Each rotation/scenario combination has a new
output root and matching locally frozen provenance. No historical checkpoint is
an initialization source. The historical runs with an inherited v6 prefix and a
v7 suffix, recorded in [the v7 status document](rescue_need_v7_status.md), do not
supply a causal rotation-profile control. A matched rotation-profile comparison
remains pending the fresh angular results.

## Completed linear wave: 600 s results

Every campaign has `common_completed_time_s=600` and wrapper exit code 0. The
campaign schema retains `target_s=3600` and therefore records
`status=PARTIAL_REQUESTED_WINDOW`; this does not indicate an incomplete requested
600-second run. At the 600-second cut, pending aircraft remain in the population
accounting and are not counted as failures.

| Scenario | Method | Normal landings | Emergency landings | Airborne pending | Offered | Energy (kWh) |
|---|---|---:|---:|---:|---:|---:|
| diagnostic 18 / 120 per hour | EnergySlack | 22 | 0 | 24 | 46 | 281.0471 |
| diagnostic 18 / 120 per hour | RHCP | 24 | 0 | 22 | 46 | 265.9599 |
| medium 24 / 150 per hour | EnergySlack | 16 | 0 | 41 | 57 | 386.2639 |
| medium 24 / 150 per hour | RHCP | 14 | 0 | 43 | 57 | 389.8847 |
| stress 36 / 180 per hour | EnergySlack | 6 | 0 | 69 | 75 | 588.7542 |
| stress 36 / 180 per hour | RHCP | 6 | 0 | 69 | 75 | 588.4732 |

All six summaries report `F_energy_total=0`, `N_upstream_pending=0`,
`terminated_invalid=false` and `conservation=true`. No `invalid.json` or
`error.json` marker is present in their method directories. Zero emergency
landings does not mean no emergency declarations: the EnergySlack/RHCP declaration
counts are 3/4, 10/11 and 19/22 for diagnostic, medium and stress respectively.

Normal-scheduling callback wall times from `summary_0600.json.normal_timing`
are below. These measure the normal scheduling callback, not every executor
callback and not pure solver time.

| Scenario | EnergySlack mean / P95 (s) | RHCP mean / P95 (s) | RHCP solver-entered callbacks |
|---|---|---|---|
| diagnostic | 15.0506 / 18.1960 | 17.6144 / 23.2966 | 320 / 320 |
| medium | 18.9143 / 24.2285 | 21.7036 / 28.7400 | 308 / 308 |
| stress | 21.0746 / 30.1384 | 24.2182 / 30.9865 | 261 / 261 |

In stress, all 259 EnergySlack and all 261 RHCP normal callbacks recorded
`CANDIDATE_WORK_CUTOFF`. These comparisons concern the specified planners under
finite planning budgets; they do not establish an unconstrained physical landing
capacity bound. Both stress methods landed six initial-cohort aircraft and none
of the 39 new requests by 600 s, while each recorded 44 completed cross-ring
transfers. Diagnostic RHCP recorded two actual `FINAL_EMERGENCY` starts but zero
emergency touchdowns by 600 s. Starts, completed transfers and landings therefore
refer to distinct events.

The primary descriptive comparison is scenario-dependent: RHCP produced two more
normal landings with 15.0872 kWh less energy in diagnostic, two fewer normal
landings with 3.6208 kWh more energy in medium, and the same six landings with
approximately 0.2810 kWh less energy in stress. These are one-seed outcomes within
the linear profile. They establish neither general method dominance nor a causal
rotation-profile effect; the matched angular comparison is still pending.

| Scenario | End time, 2026-10-23 UTC+08:00 | Campaign wall time (s, including preparation credit) |
|---|---|---:|
| diagnostic | 20:20:46 | 8294.1698 |
| medium | 21:12:37 | 11399.3593 |
| stress | 21:55:12 | 13948.6680 |

All six `audit_0600.json` files record `status=PASS`, `issue_count=0` and
`continuous_separation=CERTIFIED`. Their count and derived-cost values agree
with the corresponding direct `summary_0600.json` metrics. The audit scope is an
independent state/event ledger using the shared frozen trajectory, energy and
continuous-separation kernels; it is not an empirical energy oracle. It retains
the existing additive peak-error envelope and does not reconstruct precision-switch
events. The audit physical outcome remains IN_PROGRESS because aircraft are
still airborne at the time cut; human visual review remains pending.

The completed output roots are
`/home/user/training_outputs/terminal/equal_linear21_diagnostic_600s_20261023`,
`equal_linear21_medium_600s_20261023`, and `equal_linear21_stress_600s_20261023`
under that same server parent. Their local synchronized counterparts are under
`D:\7_codespace\training_outputs\terminal` with the same basenames. In each
root, both `energy_slack` and `rh_cp_sat` contain `summary_0600.json`,
`audit_0600.json`, `checkpoint_0600.json`, `replay_0600/`, and
`archive_status.json`; root-level `campaign.json`, `exit_code.txt` and
`end_timestamp.txt` record completion. No output was rerun or modified for this
result update.

## Historical reference comparison at 600 s

The following table compares the completed linear wave with the historical
angular-speed results in [the rescue_need_v7 status document](rescue_need_v7_status.md).
It is a historical reference comparison, not the pending same-version matched
control comparison. Historical diagnostic and stress combine a v6 0-240 s prefix
with a v7 240-600 s suffix; historical medium is v7 from t=0 but uses the older
code. Landing entries show normal + emergency = total; total differences are
linear minus historical angular.

| Scenario | Method | Historical angular landings | Linear landings | Historical angular kWh | Linear kWh | Total landing difference |
|---|---|---|---|---:|---:|---:|
| diagnostic | EnergySlack | 27 + 0 = 27 | 22 + 0 = 22 | 262.148928 | 281.047077 | -5 |
| diagnostic | RHCP | 27 + 2 = 29 | 24 + 0 = 24 | 263.016853 | 265.959891 | -5 |
| medium | EnergySlack | 27 + 2 = 29 | 16 + 0 = 16 | 394.344077 | 386.263858 | -13 |
| medium | RHCP | 22 + 4 = 26 | 14 + 0 = 14 | 406.652004 | 389.884678 | -12 |
| stress | EnergySlack | 24 + 4 = 28 | 6 + 0 = 6 | 579.993689 | 588.754220 | -22 |
| stress | RHCP | 25 + 5 = 30 | 6 + 0 = 6 | 572.083134 | 588.473194 | -24 |

Against this historical reference, normal and total landings are lower in all
six linear rows, with the largest total reductions in stress. Energy differences
have mixed directions. These are single-seed observations under differing code
and, for diagnostic/stress, mixed policy protocols; formal profile comparison
awaits the fresh same-version angular controls. The rotation change also changes
middle/inner tangential speeds and centripetal accelerations, so this table
cannot attribute the differences solely to phase dynamics or establish a physical
capacity optimum.

The original historical audits remain 3 PASS / 3 FAIL, with seven
`EVENT_FRAME_MISSING` issues for `rescue_warning`; their continuous-separation
status is CERTIFIED in all six cases. The new linear audits are 6 PASS with zero
issues. The historical FAIL results are preserved and are not reclassified by
this comparison.

## Angular controls: verified startup record

All three fresh controls started on 2026-10-23 at 22:20:19 UTC+08:00. Startup was
verified at 22:21:23: six state streams and checkpoints had been created, every
stream had positive simulation time, and no error or terminal marker was present.
Observed method-process CPU use was approximately 99.5%-99.9%. The PIDs and frame
times below are historical startup observations, not claims about current process
liveness or completion.

All output roots below are under `/home/user/training_outputs/terminal/`.
State-time pairs are EnergySlack / RHCP respectively. Method PIDs are omitted
because their mapping to method order was not confirmed.

| Control / output-root basename | Wrapper PID | Runner PID | CPUs | Startup state times (s) |
|---|---:|---:|---|---|
| equal_angular_control_diagnostic_600s_20261023 | 183635 | 183638 | 4 / 6 | 11 / 9 |
| equal_angular_control_medium_600s_20261023 | 183636 | 183639 | 8 / 10 | 9 / 4 |
| equal_angular_control_stress_600s_20261023 | 183634 | 183637 | 0 / 2 | 7 / 4 |

The controls use the same pinned Python, numerical threads=1,
`--preparation-wall-seconds 300 --campaign-wall-budget-seconds 36000
--archive-reserve-seconds 1800`, and `run --target 600` with their paired CPUs.
Locally frozen files are
`artifacts/rotating_slots/equal_linear21/angular_diagnostic18_provenance.json`,
`angular_medium24_provenance.json`, and `angular_stress36_provenance.json` in that
same directory. They record revision bce1f90 and a clean freeze. Their source-file
fingerprints and aggregate source hash
`b89f59e4795ca7f1779a3ad43da16a00cc7ed911cb6ad31ebd7048d654812517` are identical
to the linear wave's b7d0d24 source; the selected physical payload change is the
rotation profile.

Matched fixture comparisons passed for all three pairs: manifest bytes are
identical; the 21/27/39 air-plus-ground aircraft records match on IDs, status,
ring, slot, SOC, service class, pad and service-end time; and the pad/slot records
are equal. Kinematics follow the selected profile, as specified above.

Initial angular runtime estimates T0 are 9000 s (diagnostic), 12600 s (medium),
and 18000 s (stress), based on prior observed angular throughput. Early startup
fractions are not linearly extrapolated. Heartbeat `600` was successfully
updated ACTIVE at 2026-10-23 22:23:53 UTC+08:00 with a 297-minute interval; both
the tool result and saved TOML were verified. Its initial next-wake estimate of
2026-10-24 03:20 UTC+08:00 was subsequently superseded.

At the 2026-10-23 23:36:13 UTC+08:00 snapshot, all three runners and six method
processes were alive with no error or terminal markers; EnergySlack/RHCP frame
times were diagnostic 473/460.08 s, medium 360/353.5 s and stress 288/287 s, with
common completed summaries at 420/300/240 s respectively. The latest complete
stress RHCP 60-second block took 933.6015 wall seconds (previous block 901.2268);
its throughput projected approximately 4870 wall seconds for the remaining 313
simulation seconds. Around 23:37, the tool confirmed heartbeat `600` ACTIVE with
an updated 80-minute interval, giving the current next-wake estimate near
2026-10-24 00:58 UTC+08:00 after allowing for scheduling delay. This is an estimate,
not a completion guarantee. Completion checks, angular audits and the matched
comparison remain pending. No general profile dominance is established.

## Historical linear-wave launch record

Implementation commit `b7d0d24b179317031b891f9da4814bfe85b7e8f5` was accepted and
pushed before launch. Locally frozen provenance files are
`artifacts/rotating_slots/equal_linear21/diagnostic18_provenance.json`,
`medium24_provenance.json`, and `stress36_provenance.json` in that same directory.
Each records revision b7d0d24 and an empty tracked diff; source/config byte
verification succeeded on the server. The server uses the existing
`/home/user/envs/terminal_stage2_v1/bin/python` and one numerical-library thread
per process.

All output roots below are under `/home/user/training_outputs/terminal/`.
Times are on 2026-10-23, Asia/Shanghai (UTC+08:00).

| Condition / output-root basename | Wrapper PID | Runner PID | CPUs | Start time |
|---|---:|---:|---|---|
| equal_linear21_diagnostic_600s_20261023 | 174136 | 174137 | 4, 6 | 18:08:07 |
| equal_linear21_medium_600s_20261023 | 174298 | 174300 | 8, 10 | 18:08:19 |
| equal_linear21_stress_600s_20261023 | 174299 | 174301 | 0, 2 | 18:08:19 |

Each preparation used `--preparation-wall-seconds 300
--campaign-wall-budget-seconds 36000 --archive-reserve-seconds 1800`. The measured
campaign preparation values, including the 300-second credited validation cost,
were 301.1787 s (diagnostic), 301.5931 s (medium), and 302.3543 s (stress); actual
preparation elapsed times were approximately 1.18, 1.59, and 2.35 s. Each run used
`--target 600 --cpus` with its tabled pair and the frozen service/rotation/scenario/
rescue selectors above.

The following startup observations are historical. Checks around 45 seconds
after launch observed six growing state streams
with positive simulation times and no error markers. The initial observations
below are EnergySlack / RHCP respectively; byte counts refer to their
`states.jsonl` files, not completed episodes.

| Condition | Latest observed simulation times (s) | State bytes |
|---|---|---|
| diagnostic | 17 / 14.6875 | 735489 / 671525 |
| medium | 9.3887 / 9 | 517450 / 447273 |
| stress | 7 / 7 | 518509 / 500237 |

Initial historical runtime estimates T0 were 9000 s, 12600 s, and 21600 s for
diagnostic, medium, and stress. Early frame times do not support reliable linear
runtime extrapolation. A current-chat heartbeat, ID `600`, was successfully
created and verified ACTIVE around 18:12 with a 15-minute interval for the first
complete 60-second block calibration. The initial creation attempt lacking a
destination created no schedule; the successful retry and subsequent view
confirmed this heartbeat. The launch-time plan was to update the same heartbeat
to the estimated remaining duration R after a usable block observation; the
initial calibration interval was not a repeated 15-minute polling policy.

The linear wave is now complete and angular startup has been verified. The
current control-wave heartbeat plan is recorded above. Remaining work is to verify
the fresh angular runs, check their completed outputs, and compare matched conditions.
The historical startup observations themselves establish no method/profile
advantage.

## Fresh campaign commands

Freeze provenance locally after the final reviewed source changes, with an
ignored artifact path inside this checkout (Syncthing carries this file):

```powershell
& D:\7_codespace\terminal\venv\Scripts\python.exe scripts/server/run_rotating_slots_stage2.py freeze-provenance --output artifacts/rotating_slots/equal_linear21/diagnostic18_provenance.json --service-profile service_half_v1 --rotation-profile equal_linear_21_v1 --scenario-profile diagnostic18_rate120_v1 --rescue-rule-profile rescue_need_v7
```

Use a new output root, with no inherited checkpoint. Preflight the available
physical CPU representatives, load and memory first, and choose two free distinct
representatives for METHOD1_CPU and METHOD2_CPU. One numerical-library thread and
one CP worker remain the existing protocol. Prepare/run use the existing server
environment and never perform Git or install packages:

```bash
cd ~/projects/terminal/.worktrees/equal-linear-speed
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/user/envs/terminal_stage2_v1/bin/python
ROOT_OUT=$HOME/training_outputs/terminal/equal_linear21_diagnostic18_fresh
$PY -m pytest tests/test_rotating_slots_equal_linear.py tests/test_rotating_slots_stage2_planning.py tests/test_rotating_slots_stage2_campaign.py tests/test_rotating_slots_stage2_cli.py -k "not campaign_prepare_identity_and_checkpoint_replay" --basetemp "$HOME/training_outputs/terminal/equal_linear21_server_tests" -q
$PY scripts/server/run_rotating_slots_stage2.py prepare --output "$ROOT_OUT" --provenance artifacts/rotating_slots/equal_linear21/diagnostic18_provenance.json --service-profile service_half_v1 --rotation-profile equal_linear_21_v1 --scenario-profile diagnostic18_rate120_v1 --rescue-rule-profile rescue_need_v7
$PY scripts/server/run_rotating_slots_stage2.py run --output "$ROOT_OUT" --target 600 --cpus METHOD1_CPU METHOD2_CPU
$PY scripts/server/run_rotating_slots_stage2.py status --output "$ROOT_OUT"
```

The server test command excludes the local provenance-freeze test because that
test invokes Git. The branch_v7 test module also invokes local Git and is omitted
from the server command. Prepare/run themselves only verify frozen source bytes.

A tiny real two-baseline smoke is available without campaign preparation or Git:

```bash
$PY -m pytest tests/test_rotating_slots_equal_linear.py -k real_two_method_tiny_output_closure --basetemp "$HOME/training_outputs/terminal/equal_linear21_tiny_smoke" -q -ra
```

Each method uses one actual inner-ring aircraft, unchanged quality_v5 budgets and
rescue_need_v7, and runs from 0 to 0.4 s. It writes summary.json, checkpoint.json
and replay under its separate pytest directory, validates the roundtrip, requires
a submitted feasible prefix, and requires nonempty actual RHCP solver statuses.
Summary includes every solver record/status and submitted count. The smoke fails
and preserves evidence if the fixture does not exercise this path; it does not
expand itself. These smoke tests use a synthetic identity, without provenance
freezing or Git. For the campaign path the initial formal 60-second block provides
startup/partial-output evidence before the target 600 boundary.

Both methods run through the same campaign and produce separate checkpoints,
summaries, trajectories, replay bundles and audit evidence. Inspect both
`energy_slack` and `rh_cp_sat` outputs, completion boundary, invalid/error markers and
archive_status. This is a validation smoke, not the formal 0-to-600 comparison.
The three frozen scenario profiles are diagnostic18_rate120_v1,
medium24_rate150_v1, and stress36_rate180_v1; each formal condition starts from 0
in a new root and explicitly runs to 600. Freeze a matching provenance file per
service/rotation/scenario combination. Fresh equal_angular controls use the same
commands with their matching selector/provenance and separate output roots.
Never use the branch command or older v6/v7 checkpoints to initialize these runs.
