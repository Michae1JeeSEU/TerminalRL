# V4.2 S0 Hierarchical Qualification — PASS

The S0 run performed architecture and semantic checks only. It did not run
I1, DAgger, PPO, an I0 recollection, or alter Teacher, Observation, reward, or
Safety authority.

| Gate | Result |
|---|---|
| V4.2 schema, gate, and proposal semantics | PASS |
| All 25 RELEASE mapping round-trips | PASS |
| Teacher mapping and deterministic encoding | PASS |
| Unweighted BC proposal mask | PASS |
| Requested joint log-probability | PASS |
| Conditional entropy | PASS |
| ActionTape exact replay | PASS |
| One-request ExecutionScheduler boundary | PASS |
| V4.1 checkpoint rejection / V4.2 fresh save-load | PASS |
| Derived I0 lineage | PASS |
| Safety integration | PASS |

Safety counters were all zero: collision, NMAC, hard violation,
SafetyInvariantError, and uncertified were each `0`. Exact replay completed
one of one expected step with reset, configuration, observation, requested
category, post-state, and executed-action checks all true.

```text
V4_2_HIERARCHICAL_SCHEMA = PASS
V4_2_I1_ALLOWED = true
PPO_ALLOWED = false
```

The PASS permits a separately requested fresh V4.2 I1 qualification only; it
does not authorize PPO or DAgger.
