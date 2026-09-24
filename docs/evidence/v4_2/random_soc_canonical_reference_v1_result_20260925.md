# Random-SOC Canonical Reference V1 Results

Status: completed development comparison on 2026-09-25. This record reports the
registered smoke `848000` and full pair seeds `848101` and `848102` under
`RANDOM_SOC_CANONICAL_REFERENCE_DEVELOPMENT_V1`. It is a development result and
does not authorize qualification, PPO, or formal P3.

## Fixed inputs and execution status

The canonical initial checkpoint was
`4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c` and the
repair candidate was
`e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`.
All three registered runs completed with exit code `0`. The full runs used
`smoke=false`, horizon `1800`, `development_only=true`,
`selection_report_only=true`, and `config_diff={}`.

| run | status | canonical steps | repair steps | requests per arm |
|---|---|---:|---:|---:|
| smoke `848000` | `SMOKE_COMPLETED` | 2 | 2 | partial, 2-step bound |
| full `848101` | `COMPLETED` | 919 | 788 | 24 |
| full `848102` | `COMPLETED` | 900 | 760 | 24 |

The smoke tapes were parsed with schema `A5V2-L1-S4-SafetyTrace-v4` and had two
`action_tape` entries per arm. In each full payload, `action_tape` contains one
entry per replayed step; `records` is the `SafetyTraceRecorder` rolling
diagnostic deque and is capped at 120. The exact paths and parsed lengths were:

| run | arm | exact ActionTape path | steps | `action_tape` entries | rolling `records` entries |
|---|---|---|---:|---:|---:|
| 848000 | canonical_initial | `/home/user/training_outputs/terminal/random_soc_canonical_reference_v1_20260925/smoke848000/canonical_initial/action_tapes/safety_trace_seed_848000_step_1.json` | 2 | 2 | 2 |
| 848000 | repair_candidate | `/home/user/training_outputs/terminal/random_soc_canonical_reference_v1_20260925/smoke848000/repair_candidate/action_tapes/safety_trace_seed_848000_step_1.json` | 2 | 2 | 2 |
| 848101 | canonical_initial | `/home/user/training_outputs/terminal/random_soc_canonical_reference_v1_20260925/pair848101/canonical_initial/action_tapes/safety_trace_seed_848101_step_918.json` | 919 | 919 | 120 |
| 848101 | repair_candidate | `/home/user/training_outputs/terminal/random_soc_canonical_reference_v1_20260925/pair848101/repair_candidate/action_tapes/safety_trace_seed_848101_step_787.json` | 788 | 788 | 120 |
| 848102 | canonical_initial | `/home/user/training_outputs/terminal/random_soc_canonical_reference_v1_20260925/pair848102/canonical_initial/action_tapes/safety_trace_seed_848102_step_899.json` | 900 | 900 | 120 |
| 848102 | repair_candidate | `/home/user/training_outputs/terminal/random_soc_canonical_reference_v1_20260925/pair848102/repair_candidate/action_tapes/safety_trace_seed_848102_step_759.json` | 760 | 760 | 120 |

## Safety and outcome observations

All full arms had `completed=24`, `terminal=true`, `replay_exact=true`,
`weights_unchanged=true`, `label_rows=0`, `teacher_disabled=true`, and
`PPO_ALLOWED=false`. `SafetyInvariantError`, collisions, hard-separation
violations, NMAC, uncertified execution, and `safety_error` were zero or null
for every arm.

| run | arm | completed | emergency_landed | low-SOC count | low-SOC mean wait (s) | priority-weighted wait (s) |
|---|---|---:|---:|---:|---:|---:|
| 848101 | canonical_initial | 16 | 8 | 4 | 167.8045859084 | 252.5112285519 |
| 848101 | repair_candidate | 21 | 3 | 4 | 158.2963334948 | 122.2999225800 |
| 848102 | canonical_initial | 15 | 9 | 6 | 193.5543016290 | 231.8505665005 |
| 848102 | repair_candidate | 19 | 5 | 6 | 162.0828837852 | 142.4283506272 |

The historical all-complete/`no_failed_outcomes` gate field was `false` for
each full arm. This is retained as an observed old-gate result; the registered
development comparison records `emergency_landed` separately and does not
rewrite that field.

## Paired request-attribute check

For each full seed, the canonical and repair lifecycle rows were compared as
multisets of `(base_priority, round(initial_soc, 10))`. Each arm had 24 unique
tuples and the two multisets were equal within the pair, while the UID sets were
different.

| seed | attribute multiset equal | UID set equal | priority counts in each arm |
|---:|---|---|---|
| 848101 | `true` | `false` | priority 1: 11, priority 3: 9, priority 5: 4 |
| 848102 | `true` | `false` | priority 1: 8, priority 3: 7, priority 5: 9 |

## Development selection boundary

The pair summaries report `selection_gates` with candidate/initial completion,
defined metrics, low-SOC wait non-worsening, and strictly better
priority-weighted wait. The top-level fields remained
`P3_QUALIFIED=false`, `PPO_ALLOWED=false`, `P4_ALLOWED=false`, and
`promotable=false` for both full runs.

The primary selection recorded for the next V5 admission preparation is the
repair candidate `e04e5b4a...`. That selection is for a new V5 admission step
and is not a qualification result.
