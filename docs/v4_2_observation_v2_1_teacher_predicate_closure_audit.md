# V4.2 Observation V2.1 Teacher-Predicate Closure Audit

## Scope and evidence

This is a read-only internal audit. It creates no formal I0 rows and calls no
I1, DAgger, PPO, class reweighting, Teacher modification, reward modification,
or Safety modification. The audit bank records full simulator state, the 88D
V2.1 observation, Teacher gate/proposal, all 26 catalogue outcomes, source,
seed, lifecycle and state/observation hashes.

Evidence type is `newly_run`: the completed server artifact is
`~/training_outputs/terminal/v4_2_observation_v2_1_teacher_predicate_closure_audit_20260911/v4_2_observation_v2_1_teacher_predicate_closure_audit.manifest.json`
(SHA-256 `fd75ee6e78deb4c307feee521b20d829a2a41aa8980563c5abd4a0f1aa4519f2`).

The audit does not treat a bounded Teacher catalogue scan as student-visible
state. The deployment boundary remains:

```text
actor proposes one decoded action
→ ExecutionScheduler
→ TrajectoryGenerator
→ SafetyFilter
```

## Predicate inventory from the executed Teacher path

| Class | Predicate | Effect on gate | V2.1 closure status |
|---|---|---|---|
| A | retry remaining | masks RELEASE | direct: appended index 87 |
| A | self committed plan | masks RELEASE | sufficiently identified by transfer phase and `release_eligible` |
| A | phase/ring context | masks outside holder rings | direct |
| A | affected interface | masks RELEASE | direct `interface_frozen` |
| A | final capacity | masks inner→final RELEASE | direct resource fields |
| A | landing-zone availability | masks inner→final RELEASE | direct resource fields |
| B | proposal reservation feasibility | candidate result | ACTION_CONDITIONED |
| B | candidate trajectory dynamics feasibility | candidate result | ACTION_CONDITIONED |
| B | SafetyFilter / holder-tube certificate | candidate result | ACTION_CONDITIONED |
| B | catalogue preference order | selects among certified proposals | ACTION_CONDITIONED |
| C | SOC | not read by current Teacher gate | diagnostic only |
| C | last action, rejection history, lifecycle age | not read by current Teacher gate | diagnostic only |

The inventory is based on `_release_context`,
`teacher_joint_expert_v4_1`, `MergeScheduler.propose/can_reserve`,
`TerminalParallelEnv._candidate_plan`, and `SafetyFilter.filter_plan`; it is
not inferred from field names.

## State-only closure and schema sanity

`safety_retry_remaining_norm` is index 87. Indices 0–86 retain V2 ordering
and values. The 30-second normalization horizon follows the maximum design
headway, not empirical tuning. Runtime checks verify finite `[0,1]`,
determinism, retry-active positivity, expired-cooldown zero, no future
leakage, and exact ActionTape replay.

`free_lz_fraction` and `global_free_lz_fraction` are equal in the current
implementation. `KNOWN_REDUNDANT_FEATURE = true`; this audit deliberately
does not alter the frozen 88D schema.

## Matched-pair protocol

Pairs are seed- and lifecycle-disjoint. Each binary probe uses the full 88D
observation and fixed pre-registered gates: accuracy/macro-F1/WAIT
recall/RELEASE recall ≥ .80 and ROC-AUC ≥ .85. The audit records the actual
result for retry, committed, reservation, interface/headway, resource and
conflict/recovery rationales. A rationale with insufficient actual independent
pairs remains `NOT_AVAILABLE`, not a passing proxy.

The completed 300-state bank uses 38 independent seeds: 236 semantic rollout
states and 64 actual interface-transition states. Every row stores the full
26-action Teacher catalogue scan. The actual results are:

| Rationale family | Train/dev pairs | Result |
|---|---:|---|
| retry active vs expired RELEASE | 17 / 6 | PASS: all five metrics 1.0000 |
| committed vs RELEASE | 17 / 4 | PASS: all five metrics 1.0000 |
| resource blocked vs resource-open RELEASE | 10 / 4 | PASS: all five metrics 1.0000 |
| reservation blocked vs reservable | 0 / 0 | NOT_AVAILABLE |
| interface/headway | 17 / 0 | NOT_AVAILABLE |
| conflict/recovery | 6 / 1 | NOT_AVAILABLE |

The zero reservation pairs are consistent with the current executed
`propose(candidate) → can_reserve(proposal)` path: the proposal construction
already enforces the hard earliest headway. This does not turn it into an
observed actor feature or a successful coverage gate.

## Privileged-information finding

The current Teacher takes the RELEASE branch only after scanning the finite
25-proposal catalogue and testing proposal-specific scheduler, trajectory and
SafetyFilter outcomes. Those outcomes are not available to the actor before
it proposes one action. Therefore:

```text
TEACHER_PRIVILEGED_INFORMATION_RISK = true
```

This is a supervision-semantics/deployable-information-set issue, not an
authorization to add candidate certificates, `can_reserve`, or trajectory
results to Observation. The final freeze remains blocked even if all
state-only predicates are closed.
