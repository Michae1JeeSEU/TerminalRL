# V4.2 transition_WAIT vs clear_RELEASE Observability / Decision-Boundary Audit

## Decision

```text
transition_wait_release_observation_separable = false
decisive_teacher_variables = certified release candidate; candidate dynamics feasibility; safety retry/cooldown; committed plan; predicted holder reachable tube
missing_from_observation = [safety_retry_after_or_retry_remaining]
weakly_encoded = [reservation_lifecycle_exact_can_reserve, candidate_trajectory_safety_certificate]
ROOT_CAUSE = MISSING_DECISION_STATE_IN_OBSERVATION
observation_revision_required = true
dagger_protocol_revision_required = false
ROUND1_RERUN_ALLOWED = false
ROUND2_ALLOWED = false
PPO_ALLOWED = false
```

No Observation, policy network, dataset, Teacher, Safety component, threshold,
or training protocol was changed in this audit.

## Experiment evidence

- Evidence type: `newly_run`.
- Final manifest: `~/training_outputs/terminal/v4_2_transition_wait_observability_audit_20260911_rerun/v4_2_transition_wait_observability_audit.manifest.json`, SHA-256 `9d3d1c32000643e6224737315ba868f813f4176e55228f311c09f05ba12ae297`.
- Inputs were read-only formal I1/R1 checkpoints, the 2,002-row D1, 243-row frozen cohort, and formal Round-1 ActionTapes.
- The binary classifier is an independent linear probe; it does not modify or initialize the V4.2 actor.
- A first bookkeeping-only run was superseded so only variables actually used as current Teacher gate predicates are called decisive. This final rerun is the sole artifact used here.

## Matched pairs and full-state audit

The audit creates 90 seed/lifecycle-disjoint WAIT/RELEASE matched pairs: 30 clear outer RELEASE, 30 middle/interface RELEASE, and 30 frozen RELEASE anchors. There are 61 train and 29 development pairs; 12 transition-WAIT centers were prioritized for proximity to the six frozen RELEASE flips. The pair file SHA-256 is `5e580e0783fb9bf5b78cb1ebab7888faf81f99cbe0c340b1ee2b39e819cb24b7`.

The pair file records immutable identifiers and Observation V2. The manifest cross-links each D1 event to an exact replayed Teacher catalogue audit and full simulator fields. Teacher rationales across pair sides are:

| Rationale | Count | Meaning |
|---|---:|---|
| certified release candidate | 60 | at least one RELEASE proposal certified |
| frozen RELEASE anchor | 30 | stored label; candidate scan was not retained in snapshot |
| dynamics unreachable merge time | 50 | all relevant RELEASE candidates fail trajectory feasibility |
| retry cooldown mask | 20 | `env.now < uav.safety_retry_after` |
| committed plan mask | 15 | target UAV already has a committed plan |
| predicted holder reachable tube | 5 | predictive trajectory/safety infeasible |

## Full state versus Observation V2

| Teacher-accessed decision state | V2 status | Audit conclusion |
|---|---|---|
| phase / ring role | Present | explicit one-hot fields |
| release eligibility | Present but incomplete | does not include retry cooldown |
| affected interface / geometry freeze | Present | `interface_frozen` |
| final capacity and LZ availability | Present | final pressure and LZ features |
| target predecessor/successor | Present | topology slots |
| predicted conflict | Present | nearest-conflict slot |
| committed/reserved neighbors | Present / partial | neighbor flag; self-commit only indirect |
| headway since last executed release | Present | `interface_headway_slack` |
| exact reservation lifecycle / `can_reserve` | Indirect | no exact candidate reservation decision |
| candidate trajectory Safety certificate | Indirect | topology is not the complete candidate plan/certificate |
| retry/cooldown remaining | **Absent** | decisive in 20 WAIT cases |
| recent rejection reason | Absent | only affects current Teacher through cooldown |
| previous policy gate/action | Absent | not directly read by current Teacher |
| time since lifecycle transition | Absent | not directly read by current Teacher |
| post-WAIT accumulated eligibility | Indirect | no exact preceding action/cooldown history |

The clean Case-A evidence is that Teacher directly reads `uav.safety_retry_after`, whereas V2's `release_eligible` does not. A state can therefore look release-eligible in V2 yet be a mandatory WAIT until cooldown expiry.

## Pairwise separability using existing Observation V2

| Metric | Development result |
|---|---:|
| Accuracy | .6437 |
| Macro-F1 | .6379 |
| WAIT recall | .7701 |
| RELEASE recall | .5172 |
| ROC-AUC | .7083 |

The predefined clear-separability rule requires accuracy, macro-F1, both recalls `>= .80`, and ROC-AUC `>= .85`; it fails. Thus this is not evidence for DAgger aggregation interference or for a family-aware minibatch remedy now.

Single-group probe AUCs show partial but insufficient information: ring role `.6813`, phase `.6805`, target-ring successor `.6556`, release eligibility `.6552`, assigned-LZ-present `.6060`, nearest predicted conflict `.5981`, speed `.5636`, and target-ring predecessor `.5517`.

## Minimal next-step recommendation

An explicitly authorized Observation V2 → V2.1 design review should begin with one minimal normalized feature for remaining safety retry/cooldown time. Its semantics must match the Teacher predicate and preserve provenance/safety boundaries. Reservation/certificate fields are weakly encoded and require separate justification; they must not be expanded speculatively. No observation revision, DAgger rerun, Round 2, or PPO is authorized by this report.

## Protocol risks

- This is a local 90-pair internal development diagnostic, not an external generalization result.
- Thirty frozen anchors support explanatory pairing only; they were not used to fit or select the probe.
- Frozen-anchor snapshots retain state and Teacher label but not a complete candidate-scanner audit; only D1 pair members have exact replayed catalogue rationales.
