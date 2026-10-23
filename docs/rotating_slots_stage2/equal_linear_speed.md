# Equal linear speed profile

`rotation_profile` is independent of `service_profile`. The default
`equal_angular` preserves the two existing frozen YAML payloads and their hashes.
`equal_linear_21_v1` selects a full frozen baseline-service or half-service YAML
with common tangential speed 21 m/s. Clockwise angular speeds are outer 0.035,
middle 21/450, and inner 0.07 rad/s. The old common-angular field remains as the
unchanged legacy reference; runtime ring motion uses `ring_angular_speed`.

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

## Planned formal comparison

No formal runs have started. The planned sequence first runs the three
`equal_linear_21_v1` scenarios below, each freshly initialized at t=0 and run to
t=600 with both EnergySlack and RHCP. Matched fresh `equal_angular` controls then
use the same scenario inputs, CPU pairs and budgets. All conditions retain
`service_half_v1`, `rescue_need_v7`, `quality_v5`, the frozen SOC/arrival streams,
and the existing geometry, safety/vehicle limits, service durations and solver
budgets.

| Scenario profile | Initial airborne count | Arrival rate/h | Planned method CPU pair |
|---|---:|---:|---|
| diagnostic18_rate120_v1 | 18 | 120 | 4, 6 |
| medium24_rate150_v1 | 24 | 150 | 8, 10 |
| stress36_rate180_v1 | 36 | 180 | 0, 2 |

These CPU assignments are provisional until launch preflight confirms the cores,
load and memory are available. Each rotation/scenario combination has a new
output root and matching locally frozen provenance. No historical checkpoint is
an initialization source. The historical runs with an inherited v6 prefix and a
v7 suffix, recorded in [the v7 status document](rescue_need_v7_status.md), do not
supply a causal rotation-profile control. Formal comparison outcomes remain
unobserved.

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
