# V3 Observation V2.1 Release Separability Summary

## Verdict

`OBSERVATION_V2_1_RELEASE_SEPARABILITY = FAIL`
`fresh_PPO_rerun_allowed = false`

No Observation V2.1 feature was added and no PPO was run.  The required
minimal-repair path is blocked before an observation change: the stored
``release`` target is not a function of the observation (or even of the
captured deployment state) alone.

## Exact-hash attribution

The replayed update-1/update-4 data contained 5 V2 exact observation hashes
with both Teacher labels (12 representative pairs retained).  The complete
state audit captured simulator time, all ten active UAV states, lifecycle and
retry state, all committed plans, all scheduler reservations, interface state,
and future reservations.

Several tick-0 opposite-label pairs have identical complete captured state:
the same 10 UAV positions/ring progress, no committed plans, no reservations,
no retry backoff, and no tentative/previous release.  Yet their labels differ,
for example `release_to_wait:safety_wait` versus
`counterfactual_safe_release`, and `final_capacity` versus `teacher_identity`.
Later pairs also expose real V2 omissions—retry remaining, future reservation
ownership/not-before time, and tentative interface release time—but adding
those cannot resolve the identical tick-0 pairs.

The decisive non-state fork is in the release-label pipeline.  It evaluates
the counterfactual with the *requested* speed and merge-time residuals:

```text
intent = RELEASE + requested.speed_residual + requested.merge_time_residual
Teacher certification -> release label
```

Thus two identical state observations with different sampled speed/merge
heads have opposite release labels.  This is action-conditioned supervision,
not missing physical state.  It cannot be repaired by appending state features
to the release actor without either leaking its own sampled action into its
input or changing the Teacher-label/objective semantics; both are outside the
authorized scope.

## Old V2 vs proposed V2.1

| Check | Result |
|---|---|
| Old V2 exact opposite-label hashes | 5 |
| Full-state-identical opposite-label pairs | present |
| State-only V2.1 can make opposite labels deterministic | no |
| Feature additions | none (minimal safe change) |
| New observation dimension | not applicable |
| Label leakage | none introduced |
| V2.1 supervised probe | not run: its target remains non-functional |

The earlier V2 probes remain evidence for the failure: deduplicated events
had a conditional probability gap of only `0.0434` (balanced `0.0367`), not
the required conditional discrimination.

## Required next diagnostic (not executed)

Before any V2.1 observation work, define and audit one of these label
semantics: (1) release supervision conditional on the jointly selected
speed/merge categories, or (2) a state-only canonical release counterfactual
with frozen residuals.  This is a release-objective / factorized-action
diagnostic, not an observation repair.  Reward, BC/PPO weights and parameters,
Safety, lambda, action catalog, Teacher label, and PPO training remain
unchanged.

## Evidence and checks

- Complete replay artifact:
  `training_outputs/terminal/v3_l1_release_objective_full_state_audit_20260909/v3_l1_release_objective_diagnostic.json`
- Update-1 replay matched the stored 10,620 opportunities and 7,885 R->W
  corrections.
- Safety replay passed; no PPO update was invoked.
- `tests/test_v3_l1_release_objective_diagnostic.py`: 3 passed.
