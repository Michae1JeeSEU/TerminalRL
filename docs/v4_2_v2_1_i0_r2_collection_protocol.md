# Fresh V4.2 / Observation V2.1 I0-R2 Collector Protocol — Preregistered V1

## Freeze and authorization

This protocol is frozen on 2026-09-15 before any I0-R2 reset or collection.
It authorizes one fresh standalone expert collection and nothing else.

```text
I0_R2_PROTOCOL_PREREGISTERED = true
I0_R2_COLLECTION_ALLOWED = true

V4_2_V2_1_I1 = FAIL
CANONICAL_I1_CHECKPOINT = none
V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```

No I0-R2 rows, actor checkpoints, or Q2 rows are produced in this stage.

## Evidence basis

| Evidence | Type | Role |
|---|---|---|
| Formal I0-R, SHA `6bcbc178...` | `preexisting_artifact` | Previous training support |
| Consumed Q, SHA `8d1e80bb...` | `preexisting_artifact` | Diagnostic support reference only |
| Formal I1 checkpoints 4202–4204 | `preexisting_artifact` | Failed qualification evidence |
| Failure attribution, commit `6e29890` | `preexisting_artifact` | Identified RELEASE context gaps |
| This Markdown and YAML protocol | `newly_run` | Prospective collector freeze |

The failure attribution found strong training fit but a 29.38% three-seed
shared RELEASE false-negative rate. Shared false-WAIT states had median nearest
training-RELEASE distance 1.3935 versus 0.5246 for states correct in all three
seeds. The repair scope is therefore limited to the identified I0-to-Q RELEASE
manifold support gap.

## Frozen schemas and semantics

```text
Observation = phaseA_low_obs_v2_1_retry_state
observation_dim = 88
retry_feature_index = 87
retry_norm_horizon = 30 s

Action = low_action_schema_v4_2_hierarchical
Teacher / ranking = unchanged
ExecutionScheduler = unchanged
SafetyFilter / PredictiveShield = unchanged
```

No architecture, loss, threshold, reward, Observation, Teacher, or Safety
revision is authorized.

## Fresh standalone lineage

I0-R2 is not `old I0-R + patch rows`. Every formal row must originate from a
new complete expert episode. The collector must build and hash a seed registry
before its first reset, scan all historical/diagnostic registries, and reject a
collision before reset.

```text
training seeds       = 310001..319999
internal-audit seeds = 320001..329999
future Q2 reserved   = 330001..339999
```

Training and internal-audit partitions are episode-disjoint. They must also be
independent from original I0-R, old cohorts, consumed Q, closure/reachability/
support banks, failed checkpoints, and failure-attribution artifacts in seed,
episode, UAV/lifecycle, decision-event and full-state identity.

## Fixed episode strata

The collector uses a fixed round-robin schedule of complete episodes:

1. `post_wait_release`
2. `retry_expired_transition_release`
3. `conflict_recovery_release`
4. `high_traffic_proposal11`
5. `middle_interface_proposal14`
6. `post_wait_proposal14`
7. `post_conflict_proposal14`
8. `nominal_mixed`
9. `clear_outer`
10. `resource_pressure`

A stratum name expresses an initial/scenario recipe, not a label. All retained
states and targets come from the real simulator and unchanged Teacher. An
episode or row may not be rejected because the resulting Teacher target differs
from the stratum name.

Semantic-event retention remains event start, target/eligibility/lifecycle/
topology/interface/resource change, meaningful normalized-observation change,
and bounded sparse persistence. Raw-tick WAIT flooding is forbidden.

## Frozen context minima

All counts are independent lifecycles, with the separate minimum episode
support specified in the YAML.

| Identified repair context | Minimum lifecycles |
|---|---:|
| post-WAIT RELEASE | 60 |
| retry-expired or transition-cleared RELEASE | 40 |
| post-conflict RELEASE | 40 |
| high-traffic N=12 proposal 11 | 30 |
| middle/interface proposal 14 | 20 |
| post-WAIT proposal 14 | 20 |
| post-conflict proposal 14 | 15 |

The transition families require a same-UAV, same-lifecycle ledger connecting
the preceding WAIT to the current RELEASE. It records both state hashes and
times, prior WAIT rationale, elapsed time, and retry/interface/headway/conflict
transitions. The kind is one of retry, transition, geometry/conflict, or
ordinary WAIT-to-RELEASE.

## Nominal anchors

R2 must retain the nominal manifold rather than become boundary-only:

| Anchor | Minimum independent lifecycles |
|---|---:|
| ordinary WAIT | 250 |
| retry-active WAIT | 100 |
| committed WAIT | 100 |
| interface-frozen WAIT | 80 |
| resource WAIT | 30 |
| clear-outer RELEASE | 100 |
| normal middle/interface RELEASE | 100 |
| proposal 12 RELEASE | 80 |
| proposal 13 RELEASE | 160 |

Teacher-selected joint support `{11,12,13,14}` remains required. No other
catalogue class is forced to appear.

## Size and fail-closed stopping rule

The prospective training range is 2,500–3,500 retained semantic rows. The
internal frozen dataset-audit partition targets 600–900 rows and has a hard
ceiling of 1,000. Collection can stop only after a complete stratum cycle and
all context, nominal, provenance, replay, aliasing and Safety gates pass.

If the training partition reaches 3,500 rows or the audit partition reaches
1,000 without every required gate, I0-R2 fails. Thresholds cannot be lowered
after observing collection outcomes.

## Consumed-Q support closure diagnostic

The old Q is permanently:

```text
Q_SHA256 = 8d1e80bbf7efbf8bdada1814efc3c435daa80dd00564935c2e9dd3ba466dffe5
status = CONSUMED_DIAGNOSTIC_REFERENCE
```

It is never training input, model-selection input, qualification evidence or
threshold-tuning input. After collection only, it may measure support distance
using frozen 88D Euclidean distance, k=10 and radii 0.05/0.10/0.25.

Support closure requires both the context minima above and:

- proposal 11/14 maximum context-distribution total variation no greater than
  0.20 relative to the consumed diagnostic reference;
- post-WAIT and transition/retry reference nearest-RELEASE median no greater
  than 1.00 and median kNN RELEASE fraction at least 0.60;
- post-conflict nearest same-family RELEASE median no greater than 1.00 and
  median kNN RELEASE fraction at least 0.60.

These are prospective collector diagnostics, not revised I1 qualification
thresholds.

## Provenance, replay and Safety gates

Every retained row stores the raw sources needed to reconstruct the exact 88D
Observation, `env.now`, `safety_retry_after`, reset recipe, RNG hashes, full UAV
and environment state, dynamic airspace, scheduler/interface/headway/resource
state, transition ledger, Teacher gate/proposal/rationale, full 25-proposal
audit sidecar and config/state/observation/post-state hashes.

Formal passage requires 100% retry/schema provenance, deterministic Teacher,
100% ActionTape exact replay, zero exact observation-hash opposite-gate
conflicts, and:

```text
collision = 0
NMAC = 0
hard violation = 0
SafetyInvariantError = 0
uncertified execution = 0
```

Only if every frozen gate passes may the future collector declare:

```text
V4_2_V2_1_I0_R2 = PASS
V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED = true
```

I2 and PPO remain forbidden regardless of collector outcome.

## Future qualification

After an I0-R2 PASS, the next order is: preregister I1 Qualification Protocol
V2, collect and freeze a new untouched Q2, train fresh multi-seed actors, then
evaluate Q2 exactly once. The consumed Q can never serve as Q2.

## Protocol risks

- Scenario targeting changes occupancy distributions by design; the nominal
  anchor gates are required to prevent a boundary-only training set.
- Consumed-Q comparison can diagnose coverage but cannot provide untouched
  qualification evidence.
- Euclidean distance weights each normalized coordinate equally and is not a
  causal measure; context lifecycle gates remain conjunctive.
- If a targeted proposal is naturally rarer than expected, the protocol fails
  closed rather than admitting label-conditioned rows or lowering thresholds.
