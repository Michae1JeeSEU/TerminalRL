# V4.2 / V2.1 RELEASE-manifold audit

## Result

This is a read-only diagnostic.  It did not collect formal I0 rows, train an
actor, alter the Teacher/Observation/SafetyFilter, or change a threshold.

`ROOT_CAUSE = COLLECTOR_SEMANTIC_CLASSIFICATION_ERROR`.

The previous preflight's source-name detector is not a valid semantic-family
detector.  It classified RELEASE rows by the source that happened to emit a
row, while the requested families are properties of the state and transition
history.  Consequently it cannot support a reachability or coverage-gate
decision.

## Evidence inventory

| Evidence | Type | Role |
| --- | --- | --- |
| 601-row collector preflight, SHA256 `2b921a86...a3eb83` | preexisting artifact | read-only input; 109 unique RELEASE lifecycles |
| 15-seed bounded nominal-state search | newly run | recipe adequacy probe only; zero RELEASE witnesses |
| reconciled manifest, SHA256 `82b325d3...7d45e5` | newly run | corrects support extraction and fails closed |

The full machine-readable audit of all 109 RELEASE lifecycles is retained in
`~/training_outputs/terminal/v4_2_v2_1_release_manifold_audit_reconciled_20260912/v4_2_v2_1_release_manifold_audit_manifest.json`.
For every lifecycle it records source, seed, UAV/lifecycle ID, phase/ring,
previous semantic decision, elapsed time from prior WAIT, retry history,
interface, conflict/recovery flags, full state hash, the selected proposal,
the fixed 25-way Teacher preference ordering, and the complete certificate
sidecar.  `headway_age` is explicitly `null`: the historical snapshot retains
headway values but not its age, so the audit did not infer or fabricate it.

## Reclassification of the 109 RELEASE lifecycles

The original source-derived counts were `clear=0`, `middle=8`,
`post-conflict=6`, `post-WAIT=0`, and `other=83`.  The 83 “other” rows are
exactly 32 `expert_rollout`, 27 `safe_warm_start`, and 24
`committed_execution` rows; “other” therefore means *not emitted by one of
four specially named sources*, not “outside the desired decision manifold.”

Applying physical phase/history predicates to the retained snapshots gives:

| Audit finding | Count |
| --- | ---: |
| Source detector vs physical-family mismatch | 96 / 109 |
| Multi-family overlap | 56 / 109 |
| Physical `clear_outer_RELEASE` | 49 |
| Physical `middle_interface_RELEASE` | 47 |
| Physical `post_WAIT_RELEASE` | 56 |
| Physical `post_conflict_RELEASE` | 0 |

These counts overlap by design.  A RELEASE after WAIT can also be outer or
middle-ring.  The sole physically absent requested family is post-conflict;
the old source label `post_conflict_recovery` is not proof that a retained
state followed a conflict WAIT.  Conversely, the old
`post_wait_accumulated_release` source supplied six RELEASE rows, but none
had a retained preceding WAIT under the original detector.  This is a
transition-retention/lifecycle-definition defect, not evidence that post-WAIT
release is absent: 56 physically identified post-WAIT events already exist.

## Bounded targeted search

The new legal-state probe used fresh seeds 82001–82015.  It was constrained to
the declared nominal controls (warm-start traffic count, legal ring phase,
dynamic geometry transition, timing, and normal simulator execution), saved
reset recipes, and did not reuse preflight rows.  It found zero target-family
witnesses and hence wrote no witness tape.  Safety counters remained zero.

This is **not** a global non-reachability finding.  The current source recipes
only changed warm-start count/dynamic flag; they did not yet instantiate the
claimed clear-gap, predecessor-cleared, headway-age, conflict-recovery, or
same-UAV WAIT-to-RELEASE transition predicates.  A zero-witness result under
that recipe is evidence that the recipe is inadequate, not that a proposal or
family is unreachable.

## Post-WAIT result

The physical-history audit distinguishes prior retained WAITs, but historical
sidecars only preserve generic rationale and no direct headway-age timestamp.
It establishes that the current source detector is insufficient to distinguish
retry/headway/conflict/resource/ordinary WAIT causes reliably.  The required
next diagnostic collector must retain a per-UAV transition ledger with:

- prior gate, exact WAIT rationale and state hash;
- same-UAV lifecycle continuity (no reset/UID substitution);
- interface/release eligibility and retry state at both ends; and
- reason-specific `retry`, `headway`, `geometry/conflict`, `resource`, or
  `ordinary` classification.

That is a scenario/collector protocol repair, not relabeling or sampling
weighting.

## Decision

No action-space reduction or coverage-gate relaxation is justified.  The
existing formal thresholds remain frozen because the audit has not produced a
valid targeted reachability measurement.  A valid future search must use
predeclared **state predicates** and reproduce them in reset provenance, then
seek at least ten independent legal/replay-exact/safe/deterministic witnesses
per family where they exist.

```text
CATALOGUE_SUPPORT = 25
CERTIFIED_SUPPORT = 25  (observed at least once across the 109 RELEASE states)
TEACHER_SELECTED_SUPPORT = {11, 12, 13, 14}

ROOT_CAUSE = COLLECTOR_SEMANTIC_CLASSIFICATION_ERROR
coverage_gate_revision_required = false
scenario_recipe_revision_required = true
FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED = false
I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```

Protocol risk: certification support is an aggregate, state-conditional fact;
it does not say all 25 proposals are simultaneously certified.  The selected
support is currently one speed residual (0) and four merge residuals, but it
must not be treated as the reachable set until the repaired, predicate-defined
search completes.
