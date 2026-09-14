# V4.2 / Observation V2.1 Final Freeze Decision v2

```text
STATE_ONLY_CLOSURE = PASS
TEACHER_SUPERVISION_CLOSURE = PASS
TEACHER_PRIVILEGED_INFORMATION_RISK = RESOLVED_FOR_SUPERVISION

OBSERVATION_V2_1_FINAL_FREEZE = PASS
FORMAL_V2_1_I0_RECOLLECTION_ALLOWED = true

I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```

The old 600-state confirmation and final-freeze FAIL remain immutable historic
artifacts. This is a new powered confirmation under the revised, frozen closure
protocol: retry is direct online semantics; proposal verdicts are evaluated
only proposal-conditionally. The new bank reached its independent support rule
and every required gate passed.

| Requirement | Result |
|---|---|
| Retry predicate, normalization, determinism, no leakage, exact replay | PASS |
| Other constructible state-only closure | PASS |
| Unresolved state-only missing features | `[]` |
| Gate Probe A | PASS |
| Proposal Probe B | PASS |
| Proposal Probe C | PASS |
| Execution boundary and actor certificate isolation | PASS |
| Observation/action-factorization revision required | false / false |

The bounded Teacher catalogue scan is therefore resolved for supervision, not
exposed to the actor. This does not make certificate outcomes actor inputs.

The sole next authorization is a fresh formal V2.1 structured I0 recollection,
with full state including `env.now` and `safety_retry_after`, the frozen V4.2
hierarchical action schema, and unchanged safety authority. It must not silently
convert old 87D data. This decision does not authorize I1, DAgger, PPO, DAgger
Round-1 rerun, reward/Teacher/Safety changes, class weighting, oversampling, or
action-factorization changes.
