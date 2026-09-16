# V4.1 I0 Structured Expert Dataset — PASS

Final artifact: `~/training_outputs/terminal/v4_1_i0_structured_expert_20260910_lineage_r2`.

The initial current-V4.1 collection generated 1,800 fresh Teacher-labelled
states but failed solely because bounded temporal samples shared a lifecycle
event ID.  With explicit user authorization, the final artifact is a
lineage-only repair: it keeps every Observation V2, full-state snapshot,
Teacher target, decoded action, source, seed, state hash and sampling streak
byte-for-byte in meaning.  It only adds `lifecycle_event_id` and appends the
already-recorded streak as `:sample-N` to form a unique retained-sample ID.
There was no relabelling, state mutation, class balancing, duplication, or
use of V3/V4-30/old-failed-I0 labels.

| Gate / evidence | Result |
|---|---|
| Rows / retained IDs | 1,800 / 1,800 unique |
| Dataset SHA-256 | `db308d52af16864b1d4f89c5d62eeb99c77d1f9fe2fbd79a52fd3b69f6579f9d` |
| Frozen cohort SHA-256 | `ff09c69e2578413d39b8c17838c570df19cfc9ae09355256876c9f8756b21df8` |
| Raw / retained WAIT:RELEASE | 5,642:69 / 1,751:49 |
| RELEASE diversity | 8 joint classes; 4 speed levels; 5 merge levels |
| Teacher determinism | PASS, 9 independent source/seed probes |
| Required source families | PASS, including predecessor/successor, interface/headway and resource |
| Frozen cohort | PASS: 169 WAIT, 31 RELEASE; multiple speed/merge targets |
| Safety | collision/NMAC/hard/SafetyInvariantError/uncertified = `0/0/0/0/0` |

Sampling used `event_start_plus_bounded_temporal_subsample`; temporal
duplication factor is 3.173 and retained same-UAV/same-target streak
P50/P95/max is 17.5/61/90. `lifecycle_event_id` groups persistent state;
`decision_event_id` identifies each retained temporal sample.

```text
I0_STRUCTURED_EXPERT_DATASET = PASS
I1_JOINT_BC_ALLOWED = true
PPO_ALLOWED = false
```
