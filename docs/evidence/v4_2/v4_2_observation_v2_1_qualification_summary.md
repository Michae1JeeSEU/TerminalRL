# V4.2 Observation V2.1 retry-state qualification

## Final decision

```text
old_observation = phaseA_low_obs_v2_topology_87d
new_observation = phaseA_low_obs_v2_1_retry_state

old_dim = 87
new_dim = 88

retry_feature_provenance = FAIL
matched_pair_separability = FAIL (not run: dataset derivation blocked)
OBSERVATION_V2_1 = FAIL

V4_2_V2_1_I1_ALLOWED = false
ROUND1_RERUN_ALLOWED = false
ROUND2_ALLOWED = false
PPO_ALLOWED = false
```

The sole blocker is formal data provenance, not feature semantics: every
formal source row lacks an exact `safety_retry_after` value. Therefore no
derived V2.1 dataset was written, no matched-pair classifier was fitted, and
no I1, DAgger, or PPO operation was called.

## Experiment evidence

- Status / evidence type: `newly_run`.
- Command: `~/envs/terminal_env/bin/python -m terminal_marl.training.run_v4_2_observation_v2_1_qualification ...`.
- Workdir: `~/projects/terminal`.
- Environment: Python 3.12.3, PyTorch 2.13.0+cu130.
- Input: the read-only V4.2 derived-I0 parent manifest
  `ad7dc49ab52660e34165903e3c992d942d99c8e2bab49606ac0d70736810883f`.
- Output: `~/training_outputs/terminal/v4_2_observation_v2_1_qualification_20260911/v4_2_observation_v2_1_qualification.manifest.json`, SHA-256 `ccccf13b25c745a8b4b9b8896ed3cf8cdfae6adf30f9fdb405957c653ffabe76`.

## Runtime-schema checks

The one new field is at fixed index 87 with a frozen 30-second horizon.

| Check | Result |
|---|---|
| Teacher retry-mask WAIT has positive feature | PASS |
| Expired cooldown gives zero | PASS |
| finite and within `[0, 1]` | PASS |
| deterministic | PASS |
| V2 87D prefix byte-identical | PASS |
| no future leakage | PASS |
| V2.1 ActionTape exact replay | PASS |
| V2 87D checkpoint rejected by V2.1 88D loader | PASS |

## Strict derivation audit

| Split | Rows | Exact retry state | Missing `safety_retry_after` |
|---|---:|---:|---:|
| Formal training I0 | 1,500 | 0 | 1,500 |
| Frozen cohort | 243 | 0 | 243 |

`state_snapshot.time` exists, but `SafetyTraceRecorder._state()` serializes a
target UAV without `safety_retry_after`. This is true for every formal row.
Filling the field with zero, inferring it from a Teacher target, or using a
new rollout would violate the frozen-state lineage requirement, so the
qualification stopped before V2.1 data derivation.

## Protocol risks and remaining blocker

- This is an internal schema/provenance qualification, not an external
  generalization result.
- The matched-pair separability gate was not evaluated; no threshold was
  changed and no failed metric was reinterpreted as a pass.
- A future, separately authorized lineage remediation must preserve the exact
  current retry timestamp at state capture. It must then restart V2.1
  derivation and the pre-registered V2-versus-V2.1 matched-pair audit from the
  frozen parent artifacts; it may not silently retrofit current rows.
