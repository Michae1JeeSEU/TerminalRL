# Low-Level Joint Action V4 Refactor Summary

## Decision

`LOW_LEVEL_ACTION_V4 = PASS`
`V4_L0_regeneration_allowed = true`

Selected base: `e77831e` (the preceding release-state audit serialization
fix). V3 factorized actor/independent Teacher targets are **SUPERSEDED**;
historical outputs are retained and no V3 checkpoint is reusable for V4.

## Joint catalogue and execution semantics

`low_action_schema = low_action_schema_v4_joint_discrete`; `N_joint = 30`.

| Action family | Count | Meaning |
|---|---:|---|
| WAIT | 5 | speed residual retained for holder/shield execution; merge residual canonicalized to 0 |
| RELEASE | 25 | each speed × merge residual pair is one complete proposal |

The actor samples one requested joint index, decodes it to the unchanged
`[release_request, speed_residual, merge_time_residual]` execution intent,
and ExecutionScheduler executes/checks that one proposal. It does not search
the catalogue. TeacherScheduler has the bounded training-only catalogue search.

## Teacher, PPO and BC provenance

- Teacher returns one deterministic V4 joint target using canonical proposal
  and SafetyFilter verification. V3 release-only counterfactual labels are not
  enabled by V4 qualification/configuration.
- PPO behaviour action, old log-probability and ratio use only
  `a_requested_joint`.
- BC is one joint categorical CE against `a_teacher_joint`; per-component
  disagreement is diagnostic only.
- ActionTape records joint index/decode, joint log-probability, teacher,
  execution/shield application, observation hash and post-state hash. Exact
  tape replay validates the decoded requested joint action.

## Qualification evidence

### Offline joint Teacher/BC probe — newly_run

Server output `v4_low_action_qualification_20260909`, 8 deterministic
warm-start states / 64 agent rows, no PPO update:

| Metric | Result |
|---|---:|
| same full state → same Teacher joint label | true |
| labels observed | 2, 16, 17 |
| initial joint CE | 3.297591 |
| final CE | 0.862961 |
| probe accuracy | 0.625 |
| offline joint supervised probe | PASS |

### Safety integration — newly_run

40 V4 integration steps exercised all 30 actions (5 WAIT, 25 RELEASE):

| Gate | Result |
|---|---:|
| collision / NMAC / hard | 0 / 0 / 0 |
| SafetyInvariantError / uncertified | 0 / 0 |
| holder-holder / holder-committed | covered / covered |
| transfer admission / partial-tick handoff / backup policy | covered / covered / covered |

Protocol limits: this is an integration qualification, not PPO training or a
generalization claim. V4 L0 must be regenerated before any fresh V4 L1 run.

## Modified areas

- V4 action catalogue and joint PPO implementation.
- Training-only joint Teacher enumeration; runtime scheduler boundary retained.
- V4 action spec, checkpoint metadata, ActionTape/replay fields and configs.
- V4 offline qualification and safety integration runners/tests.
- V3 migration marked SUPERSEDED; V4 specification and technical spec added.

## Gate table

| Required gate | Status |
|---|---|
| joint catalog semantics | PASS |
| Teacher joint-label determinism | PASS |
| PPO requested-action provenance | PASS |
| BC joint-target provenance | PASS |
| ActionTape exact replay | PASS |
| safety integration | PASS |
| docs aligned | PASS |
| offline joint supervised probe | PASS |

No PPO, reward/BC-weight/PPO-hyperparameter/lambda/Observation-V2/Safety
authority changes, or L1b/L2 runs occurred in this refactor.
