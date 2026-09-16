# Observation V2.1: minimal retry-state design

## Frozen schema

```text
old_observation = phaseA_low_obs_v2_topology_87d
new_observation = phaseA_low_obs_v2_1_retry_state

old_dim = 87
new_dim = 88
append_index = 87
append_feature = safety_retry_remaining_norm
```

The 87 V2 fields retain their existing order, values, and semantics. V2.1
appends, rather than changes, the one explicit current-state value:

```python
retry_remaining = max(0.0, uav.safety_retry_after - env.now)
safety_retry_remaining_norm = clip(retry_remaining / 30.0, 0.0, 1.0)
```

`30.0 s` is frozen in the V4.2 V2.1 qualification configuration. It is not a
tuned constant: every retry setter uses `max(5 s, interface_headway)`, and
`DynamicAirspace` bounds every interface headway at
`SimConfig.headway_max = 30 s`. The configuration rejects a V2.1 horizon that
does not cover its configured maximum retry duration.

Semantics are exactly:

```text
0.0  = no current retry cooldown
>0.0 = a RELEASE is currently retry-masked
```

This feature reads only `env.now` and `uav.safety_retry_after`. It does not
alter `release_eligible`, reservation/topology fields, Teacher behavior,
Safety, or action semantics; it contains no forecast, future action, or
candidate-search result.

## Compatibility

V4.2 Observation-V2 checkpoints are read-only historical artifacts and are
not resumable for V2.1. Checkpoint metadata now distinguishes the 87D V2 and
88D V2.1 named schemas and feature hashes. Formal V2.1 learning, if later
authorized, requires a fresh 88D actor.

## Provenance requirement

A legacy row may be transformed only when its immutable state snapshot has
both the exact capture time (`state_snapshot.time`) and exactly one target UAV
with numeric `safety_retry_after`. Its new feature is then calculable from the
captured state; the stored V2 prefix remains unchanged. No absent retry field
may be treated as zero, inferred from a label, or reconstructed from an
approximate rollout.

The current formal I0/I0-R snapshot format does not meet this requirement:
the target-UAV trace omits `safety_retry_after`. This design document does not
authorize a data repair, recollection, relabeling, or any V2.1 training.
