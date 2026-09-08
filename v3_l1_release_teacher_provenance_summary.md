# V3 L1 Release Teacher Label Provenance Diagnostic + Repair

Evidence type: newly run repair calibration plus pre-existing frozen L0/release-cohort artifacts. No PPO update was run.

## First divergence and repair

The first divergence was in `TerminalParallelEnv.step`: the L0 builder generated a read-only, SafetyFilter-certified release counterfactual label, while the L1 FULL runtime populated `teacher_corrected_action` only for delayed speed/merge repairs. Consequently a release WAIT/RELEASE preference never reached `teacher_action`, `teacher_used`, the per-head correction comparison, or the BC target.

The repair adds an explicit `teacher_release_label_enabled` L1 configuration gate. In that gate, the runtime computes the same read-only counterfactual before execution and records the resulting teacher-preferred action after the release decision. `teacher_corrected_action`/`teacher_used` now carry the teacher label; `execution_applied_action` and `shield_applied_action` remain independent execution/safety fields. The PPO requested action remains unchanged.

## Calibration rerun

| Targeted ratio | L0-consistent R→W | L0-consistent W→R | Native runtime release corrections | Safety |
|---|---:|---:|---:|---|
| 10% | 54 | 5 | 59 | PASS |
| 20% | 56 | 5 | 61 | PASS |
| 30% | 60 | 6 | 66 | PASS |

For every calibration ratio, native release corrections equal the sum of the two L0-consistent directional labels. The smallest qualifying ratio is **10%**. All runs had collision/NMAC/hard/SafetyInvariantError/uncertified-component = 0 and shield burden = 0.

## Regression evidence

- `tests/test_low_v2_architecture.py` and `tests/test_low_level_action_v3.py`: **37 passed** after the repair.
- Existing V3 requested-action PPO provenance regression remains in that passing suite.
- No PPO was run; therefore no new PPO health or learning claim is made.

## Decision

- Same-state release-label correction-direction consistency: PASS for the newly-run calibration witness states.
- Teacher action / teacher-used / BC-target release provenance: PASS for the repaired L1 runtime path.
- PPO requested-action provenance: PASS.
- Safety: PASS.
- `V3_L1_TEACHER_PROVENANCE = PASS`
- `scenario coverage calibration rerun = PASS`
- `fresh_PPO_allowed = true`

The next permitted operation is a fresh coverage-bridge pilot; it has not been started by this diagnostic.
