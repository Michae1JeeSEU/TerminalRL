# V4 WAIT-Speed Semantic Dominance Audit

Audit date: 2026-09-10. Evidence is a read-only audit of the failed formal I0
dataset at `runs/v4_i0_structured_expert_dataset_20260910_r3`; no simulator
state, Teacher label or dataset row was modified.

## Frozen state cohort

105 unique `WAIT`-expert full-state snapshots were selected from the failed I0
artifact. The strata were expert rollout=45, safe warm-start=10, from-empty=10,
fixed anchor=10, targeted conflict=10, interface/headway conflict=10 and
resource constraint=10. Each record in the JSON audit includes full snapshot,
Observation V2, source, seed, state hash, selected WAIT target and all five
candidate records.

Artifact: `runs/v4_wait_speed_semantic_audit_20260910/v4_wait_speed_semantic_dominance_audit.json`
(`SHA256=0d985380fd1baaad546e90ddcc95de0612499470634ab4485bcc3b0655ad3b7d`).

## Five-candidate result

For every frozen state the Teacher enumerates the same WAIT order:

```text
WAIT, 0      rank 0
WAIT, -0.5   rank 1
WAIT, +0.5   rank 2
WAIT, -1     rank 3
WAIT, +1     rank 4
```

The V4 Teacher treats every WAIT candidate as unconditional `wait_legal`. It
does not construct a release proposal, trajectory or reservation, invoke the
SafetyFilter, consume headway/interface margin, or evaluate a future
time-to-next-release opportunity for those candidates. Its only distinguishing
score is `abs(speed_residual)`, followed by joint index. Therefore the five
candidate records have `SafetyFilter=not_invoked_for_WAIT_without_release_plan`
and `predicted_min_separation=not_scored_by_Teacher_for_WAIT`; this is an
observed ownership boundary, not missing measurement.

| Classification | States | Interpretation |
|---|---:|---|
| A — nonzero strictly better | 0 | No evidence. |
| B — equivalent; zero wins tie-break | 105 | Every audited state. |
| C — zero uniquely feasible | 0 | No evidence. |
| D — useful but unexpressed by Teacher | 0 | No evidence of strategic value. |

## Execution ownership

`WAIT` itself is the scheduling decision: it prevents a release/reservation.
The residual can still pass to the ring-holder executor, then be projected by
the PredictiveSafetyShield. It is not a Teacher-scored headway or release-slot
decision; SafetyFilter has no plan to assess for a pure wait.

Thus WAIT speed is primarily an execution/safety adjustment, not a genuine
strategic scheduling action in the present V4 Teacher semantics.

## Decision

```text
WAIT_SPEED_POLICY_SEMANTICS = DOMINATED
recommended_N_joint = 26
I0_failure_root_cause = ACTION_CATALOG_OVERPARAMETERIZATION
I0_recollection_allowed = false
PPO_ALLOWED = false
```

The recommended future catalog is one canonical WAIT plus 25 RELEASE
speed×merge actions. This audit does **not** change the catalog automatically.
Before a new I0 collection, make and validate a versioned 30→26 action-schema
migration and update its checkpoint/replay/config contracts.

I0 coverage philosophy should be: **all decision-relevant expert target
families require sufficient state coverage**. It must not require every
syntactically legal catalog member to appear as a Teacher target.
