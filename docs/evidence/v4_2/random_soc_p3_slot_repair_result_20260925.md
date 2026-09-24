# Random-SOC P3 Slot Repair Results

Date: 2026-09-25
Protocol: [slot repair protocol registered 2026-09-24](random_soc_p3_slot_repair_20260924.md)

Status: bounded development evidence only. This is not formal I1, P3
qualification, training admission, PPO admission, or a gate relaxation.
`P3_QUALIFIED=false`, `PPO_ALLOWED=false`, and `P4_ALLOWED=false` remain.

## Raw output and fixed comparison

The raw server and local roots remain the 2026-09-24 protocol root:

```text
/home/user/training_outputs/terminal/random_soc_p3_slot_repair_20260924/
D:/7_codespace/training_outputs/terminal/random_soc_p3_slot_repair_20260924/
```

All three arms used dense24, seed `845101`, the frozen checkpoint, and the
same safety/energy/K3/Teacher-off contract. The two forced arms share
`trained_release_only` and differ only in the existing late-slot flag;
`trained_lateON` restores the neural gate with that flag enabled.

| arm | sim s | wall s | normal / emergency / failed-energy | unresolved | terminal | min actual / swept m |
|---|---:|---:|---:|---:|---|---:|
| forced_lateOFF | 676 | 26.817433 | 23 / 1 / 0 | 0 | true | 80.888073 / 80.881843 |
| forced_lateON | 671 | 25.868502 | 19 / 5 / 0 | 0 | true | 81.015229 / 80.952364 |
| trained_lateON | 772 | 22.764587 | 18 / 6 / 0 | 0 | true | 82.697692 / 82.596646 |

All three arms had exit status `0`, no `SafetyInvariantError`, zero
collision/NMAC/hard-separation/fail-closed/hash-mismatch counters, unchanged
weights, no Teacher use, and PPO off. The `845000` late-ON two-step smoke was
valid. Server validation recorded `29` tests passing; local diagnostic and
slot checks recorded `17` and `12` passing, while the separate Teacher checks
recorded `12` passing. These test groups are reported separately.

## Decision and next authorized work

The late-ON variant is not adopted; the default remains late-slot OFF. The
forced arms are gate interventions and are not trained-policy results.

The next development preparation is a fresh state-only label collection after
the labeler alignment, followed by a gate-adapter-only supervised-learning
development comparison. The proposal head and proposal encoder remain frozen.
No new data collection or training has started. A new data protocol must first
freeze fresh seeds and limits and pass a small-sample validation.

This result does not consume formal or heldout evidence. The old protocol,
its 2026-09-24 date, and its raw output root remain unchanged. No conclusion
about P3 qualification or PPO authorization is recorded here.
