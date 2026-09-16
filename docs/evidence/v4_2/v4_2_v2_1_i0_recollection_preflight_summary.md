# V4.2 / Observation V2.1 I0 Recollection Preflight

## Final result

```text
DYNAMIC_SOURCE_REPLAY_CLOSURE = PASS
COLLECTOR_RELEASE_FAMILY_COVERAGE = FAIL
COLLECTOR_PROPOSAL_DIVERSITY = FAIL

preflight_unique_release_lifecycles = 109
preflight_release_joint_classes = [11, 12, 13, 14]
preflight_speed_values = [-0.5, 0.5]
preflight_merge_values = [-1.0, 0.5, 1.0]

PRELIGHT_ACTIONTAPE_EXACT_REPLAY = PASS
FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED = false
I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```

## Newly-run protocol

The preflight used fresh seeds 73001–75000 and produced 601 retained,
state-hash-unique semantic decision events. It is an internal collector
qualification, not formal I0 and not a training dataset. Its split contains
no old formal-I0, I0-R, DAgger, closure-bank, or powered-confirmation rows.

| Check | Result |
| --- | --- |
| retry provenance | PASS |
| Teacher determinism | PASS |
| exact opposite-gate observation hash conflicts | 0 / PASS |
| safety: collision / NMAC / hard / invariant / uncertified | 0 / PASS |
| dynamic source replay closure | PASS, 20/20 exact tapes |
| sampled collector ActionTape replay | PASS |
| RELEASE family coverage | FAIL |
| proposal diversity | FAIL |

Preflight artifact hashes:

```text
rows_sha256     = 2b921a86a42fdeba1ec8e4623abcfcc851a0268eb0ef1c20c9733ffc00a3eb83
manifest_sha256 = 1ccb142c9e30d1ba24e5410710cee1b66759efd97cc01759206c3739733d1b12
```

Artifacts are stored at
`~/training_outputs/terminal/v4_2_v2_1_i0_collector_preflight_20260912`.
