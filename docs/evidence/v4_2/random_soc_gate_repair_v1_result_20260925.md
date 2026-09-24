# Random-SOC Development Collection and Gate Repair Result

Protocol: [RANDOM_SOC_GATE_REPAIR_V1_20260925](random_soc_gate_repair_v1_20260925.md)

This page records development collection and gate-adapter fit evidence only.
It is not a P3 qualification or PPO admission result. `P3_QUALIFIED=false`,
`PPO_ALLOWED=false`, and `P4_ALLOWED=false` remain unchanged.

## Frozen source and outputs

Source checkpoint:
`/home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt`

Source SHA256:
`714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e`

Server root:
`/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/`

Local synchronized root:
`D:/7_codespace/training_outputs/terminal/random_soc_gate_repair_v1_20260925/`

## Smoke

`smoke846000` used dense24, two steps, label stride 10, and exited `0`.
It produced 18 label rows and 18 actor-label rows; all observations were
finite 88D, event IDs were unique, Teacher execution was disabled,
`external_state_only_labeler=true`, `labeler_called=true`, weights were
unchanged, and all recorded safety counters were zero.

## Seven collection episodes

The following values are direct `Counter(summary["outcomes"].values())`
results from both the server artifacts and the synchronized local copies.
The third outcome column is the explicit `failed_energy` outcome count.

| episode | steps | outcomes (normal / emergency / failed_energy) | labels / actor labels | exit |
|---|---:|---|---:|---:|
| train_flow6_846101 | 664 | 6 / 0 / 0 | 165 / 21 | 0 |
| train_flow12_846102 | 940 | 12 / 0 / 0 | 354 / 66 | 0 |
| train_dense24_846103 | 848 | 14 / 10 / 0 | 1145 / 695 | 0 |
| train_dense24_846104 | 842 | 16 / 8 / 0 | 1191 / 747 | 0 |
| dev_flow6_846201 | 765 | 6 / 0 / 0 | 223 / 78 | 0 |
| dev_flow12_846202 | 966 | 12 / 0 / 0 | 353 / 69 | 0 |
| dev_dense24_846203 | 744 | 19 / 5 / 0 | 1124 / 665 | 0 |

Train actor labels total `1529`, with `teacher_gate=0/1` counts `1170/359`.
Dev actor labels total `812`, with `teacher_gate=0/1` counts `569/243`.
The seven seeds are disjoint; all `2341` actor event IDs are unique. All
label and actor-label observations are finite 88D. All seven summaries have
`terminal=true`, exit `0`, `weights_unchanged=true`, and `safety_error=null`.
`SafetyInvariantError`, collisions, NMAC, hard-separation violations, and
uncertified execution are zero in every episode.

## Gate-adapter fit

The authorized gate-adapter-only fit completed `35/35` epochs. Artifact:
`/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt`

Artifact SHA256:
`e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`

It used Adam `lr=3e-4`, seed `846401`, batch `64`, clip `1`, and
`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`; train/dev rows were `1529/812`.

| metric | before | final |
|---|---:|---:|
| dev RELEASE recall | 0.6131687243 | 0.7860082305 |
| dev WAIT recall | 0.8119507909 | 0.7768014060 |
| dev macro-F1 | 0.7092638611 | 0.7562677855 |
| dev NLL | 1.265022278 | 1.031340837 |
| train NLL | 1.259755254 | 0.755264521 |

Only the adapter changed; non-adapter tensors and proposal logits were
bitwise unchanged. Strict load passed with no optimizer state; no dev
gradients, early stopping, or PPO were used. The wrapper's final CRLF-related
shell exit reported an error after Python completed, while control recorded
Python exit `0`; the fit was not rerun.

## Three paired development rollouts

The authorized source/candidate pairs `flow6/846301`, `flow12/846302`, and
`dense24/846303` completed with Python exit `0`. The table reports direct
`Counter(summary["outcomes"].values())` values and the lifecycle values
recomputed by `p3._lifecycle_aggregate` for each arm.

| scenario / seed | source normal / emergency / failed-energy | candidate normal / emergency / failed-energy | source priority wait / low-SOC wait (count) | candidate priority wait / low-SOC wait (count) |
|---|---:|---:|---:|---:|
| flow6 / 846301 | 6 / 0 / 0 | 6 / 0 / 0 | 19.6343588920 / 7.4313211002 (3) | 18.8161770738 / 7.4313211002 (3) |
| flow12 / 846302 | 12 / 0 / 0 | 12 / 0 / 0 | 34.0008513481 / 14.7380788628 (4) | 21.1418487070 / 6.3875713799 (4) |
| dense24 / 846303 | 18 / 6 / 0 | 23 / 1 / 0 | 236.3354038969 / 112.6689460403 (5) | 122.1058086711 / 42.0025942597 (5) |

Across the three scenarios there were `42` requests: source `36 normal + 6
emergency + 0 failed_energy`, candidate `41 normal + 1 emergency + 0
failed_energy`. The automatically generated aggregate is
`/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/development_pair_aggregate.json`.
Its overall lifecycle values are source priority-weighted wait
`150.6971849500`, low-SOC mean wait `53.7159174128` (count `12`), and
candidate priority-weighted wait `80.1562018040`, low-SOC mean wait
`21.4881016766` (count `12`). The recomputed selection gates are all `true`.

All six arms have replay, safety, no-Teacher, no-update, and
`weights_unchanged` true; all safety counters are zero. The only false arm
gate is dense24 `no_failed_outcomes=false`; all other listed arm gates are
true. This is development evidence only and does not pass P3 qualification.

## Candidate dense24 emergency exact replay

The local `replay_candidate_emergency846303.json` has `exact=true` and
`steps_captured=795`. It targets `arrival_id=20`, UID `148`, with initial SOC
`0.5538904204`. The first observed states were outer at time `202` with SOC
`0.5112237538`, middle at `310` with `0.4446173890`, transfer m2i at `371`
with `0.4039507223`, and inner at `437` with `0.3639134929`. The first
SOC `<=0.40` record was time `377` in transfer m2i at `0.3999507223`.

The eligible-decision records show `WAIT` at each step `437` through `442`.
The exact event trace records `emergency_departure` at time `442` and
touchdown at `503`; touchdown SOC was `0.3179567218`. The earlier
`analysis_candidate_emergency846303.json` last-20 error slice included other
UIDs and did not cover the early state sequence. The first diagnostic replay
returned, but a post-processing `KeyError` prevented it from being written;
the corrected diagnostic replay produced this exact artifact. It fills that
diagnostic gap without changing the historical raw pair evidence and is not a
new fresh evaluation or qualification. The completed bounded shadow check has
`32` independent single-step checks with exact `795`-step replay binding,
unchanged state, no safety error, and zero approved witnesses: `safety_wait=16`,
`final_capacity=10`, and `safety_reject=6`. These observations only record
that the sampled checks did not obtain approval; they do not establish that
all WAIT decisions were incorrect or give the final cause of the emergency
outcome.

## Launch record and next status

The first wrapper attempt for `846101` pre-created the collector output and
exited with `FileExistsError` before checkpoint loading/environment reset. It
produced no summary, progress, or labels and was archived as
`launch_failure_train_flow6_846101`; the subsequent real `846101` run above
is the only counted train episode for that seed.

The gate-adapter fit, three paired rollouts, exact replay, and shadow
observations are development evidence. The next registered step is the
[random-SOC arbitration repair protocol](random_soc_arbitration_repair_v1_20260925.md):
both dense24 arms use the frozen repair checkpoint and differ only in
`release_arbitration_mode`; implementation and preflight are pending before
any `847xxx` reset. No arm is auto-adopted, and no new P3 qualification or PPO
run is authorized here; `P3_QUALIFIED=false`, `PPO_ALLOWED=false`, and
`P4_ALLOWED=false` remain unchanged, and the old held-out evidence remains
untouched.
