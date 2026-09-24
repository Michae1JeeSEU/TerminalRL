# Late merge-slot reconciliation exploratory protocol (2026-09-24)

This document preregisters a bounded paired diagnostic for the opt-in
reconcile_late_merge_slots_enabled path. It measures the operational effect
of reallocating a ring-transfer proposal to the generated plan's actual
canonical endpoint. It does not authorize training, PPO updates, policy
tuning, SafetyFilter threshold changes, or changes to trajectory sampling.

## Predetermined cells

The four cells are the Cartesian product of fresh seeds 841101, 841102 and
the reconciliation flag:

| cell | seed | late-slot reconciliation | status |
|---|---:|---:|---|
| A | 841101 | off | planned |
| B | 841101 | on | planned |
| C | 841102 | off | planned |
| D | 841102 | on | planned |

The seed search completed before this protocol was written and found no
existing 841101 or 841102 references in the repository. The one-step
841101 on pilot is a preflight artifact and is not one of the four cells.

Fixed conditions for every cell:

- rule action K1: [1, 0, 0] for every active UAV;
- dense_warm_start_uavs = 30;
- initial_soc_min = 0.55, initial_soc_max = 0.90;
- route_guard_enabled = true;
- final_approach_geometry_mode = continuous_inward_v1;
- release_arbitration_mode = downstream_priority;
- max_steps = 1800, progress_every_steps = 100;
- no learned policy, teacher, or training call;
- unchanged 80/40/10 m safety thresholds and safety sampling settings.

The only paired factor is the late-slot reconciliation flag. When on, a
generated ring_transfer carrying dynamic_unreachable_merge_time is
reconciled against its actual endpoint before energy and SafetyFilter checks.
When off, the existing fail-closed path remains active.

## Required recorded fields and gates

Each cell records its complete JSON output and progress JSON. Record terminal
status, terminal step, wall time, requested/active/resolved/unresolved counts,
normal/emergency/failed-energy outcomes, release approvals, primary and
secondary rejection reasons, dynamic-unreachable counts, route checks and
negative route checks, minimum actual and swept separation, and the raw
telemetry fields. Report operational_buffer_entries and
operational_buffer_takeovers separately as runtime buffer observations.

A complete cell must resolve all 30 requests, have failed_energy == 0, and
have these hard safety gates equal to zero: hard_separation_violations,
nmac, collisions, shield_fail_closed, joint_safety_fail_closed, and
safety_error. safety_interventions and other normal candidate-rejection
counters are reported but are not hard safety failures. Any incomplete cell
or failed hard gate is recorded as exploratory failure. The paired results do
not authorize PPO.

## Launch template

Run each command through ssh lab-server; write all outputs under
$HOME/training_outputs/terminal/late_slot_reconciliation_20260924/, with a
unique output path per cell:

    python3 -m terminal_marl.training.diagnose_energy_reachability_guard_smoke \
      --seed <SEED> --guard-on --initial-soc-min 0.55 --initial-soc-max 0.90 \
      --max-steps 1800 --progress-every-steps 100 \
      --dense-warm-start-uavs 30 \
      --final-approach-geometry-mode continuous_inward_v1 \
      --release-arbitration-mode downstream_priority \
      [--reconcile-late-merge-slots] \
      --output $HOME/training_outputs/terminal/late_slot_reconciliation_20260924/seed<SEED>_<off|on>_dense30_full.json

The preflight pilot already completed with seed 841101, reconciliation on,
one step, and output
$HOME/training_outputs/terminal/late_slot_reconciliation_pilot_841101_on.json.
It serialized the flag as true, reached step 1 with 30 energy checks,
failed_energy == 0, no safety error, and zero release approvals. No
full-cell command is included in this protocol's execution record.

## Recorded paired results

All four full cells reached terminal completion and resolved all 30 requests.
The hard safety gates were zero in every cell; the failed-energy gate failed
in every cell.

| cell | flag | terminal step | wall seconds | normal | emergency | failed energy | approvals | min actual / swept m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seed841101 off | false | 1067 | 56.2024 | 17 | 9 | 4 | 46 | 80.01876 / 80.01876 |
| seed841101 on | true | 979 | 53.1389 | 19 | 10 | 1 | 51 | 80.01914 / 80.01914 |
| seed841102 off | false | 880 | 42.2530 | 19 | 10 | 1 | 52 | 80.09413 / 80.09413 |
| seed841102 on | true | 1054 | 60.4470 | 13 | 14 | 3 | 43 | 80.02859 / 80.02859 |

For every cell, hard_separation_violations, nmac, collisions,
shield_fail_closed, joint_safety_fail_closed, and safety_error were zero or
null. Operational buffer entries and takeovers were both zero in all four
cells. safety_interventions remained a reported candidate-rejection counter:
84, 123, 104, and 179 for the rows above.

The raw artifacts are:

- /home/user/training_outputs/terminal/late_slot_reconciliation_20260924/seed841101_off_dense30_full.json
- /home/user/training_outputs/terminal/late_slot_reconciliation_20260924/seed841101_on_dense30_full.json
- /home/user/training_outputs/terminal/late_slot_reconciliation_20260924/seed841102_off_dense30_full.json
- /home/user/training_outputs/terminal/late_slot_reconciliation_20260924/seed841102_on_dense30_full.json

The raw release secondary reason counts were:

- seed841101 off: dynamic_unreachable_merge_time 501, emergency_energy_guard 504, merge_slot_retime_conflict 43, predicted_holder_reachable_tube 174, predicted_holder_separation 7, predicted_post_plan_holder_reachable_tube 5, predicted_separation 115.
- seed841101 on: emergency_energy_guard 159, merge_slot_retime_conflict 84, predicted_holder_reachable_tube 397, predicted_holder_separation 17, predicted_post_plan_holder_reachable_tube 51, predicted_post_plan_holder_separation 3, predicted_separation 255.
- seed841102 off: dynamics_unreachable_merge_time 393, emergency_energy_guard 139, merge_slot_retime_conflict 41, predicted_holder_reachable_tube 174, predicted_holder_separation 7, predicted_post_plan_holder_reachable_tube 13, predicted_post_plan_holder_separation 3, predicted_separation 112.
- seed841102 on: emergency_energy_guard 594, merge_slot_retime_conflict 79, predicted_holder_reachable_tube 479, predicted_holder_separation 18, predicted_post_plan_holder_reachable_tube 59, predicted_post_plan_holder_separation 6, predicted_post_plan_holder_swept_separation 1, predicted_separation 244.

The opt-in variant remains diagnostic evidence only. The default configuration
stays off, and no further run was launched under this protocol.
