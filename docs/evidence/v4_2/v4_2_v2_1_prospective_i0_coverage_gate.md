# Prospective V4.2 / V2.1 I0 coverage gate

## Approval

```text
PROSPECTIVE_I0_COVERAGE_GATE = APPROVED
coverage_gate_revision_required = true
FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED = true
```

This gate is prospective only. It does not alter the historical results:

```text
V4_2_V2_1_I0 = FAIL
COLLECTOR_PROPOSAL_DIVERSITY = FAIL
```

## Required training support

```text
required_selected_support = {11,12,13,14}
observed_training_support must be a superset of required_selected_support
```

Each stable class must have at least 10 independent training lifecycles. A dataset containing four different proposal classes does not satisfy the gate. Speed and merge marginals are diagnostics only; there is no independent `speed >= 4` or `merge == 5` requirement.

Naturally observed rare proposals must be retained and reported but are not mandatory hard-count classes. The closure found no rare Teacher-selected proposal.

## RELEASE decision-family requirements

The proposal-support gate does not replace decision-boundary coverage. Formal training I0 must retain at least:

| Predicate-defined family | Independent training lifecycles |
| --- | ---: |
| `clear_outer_RELEASE` | 15 |
| `middle_interface_RELEASE` | 15 |
| `post_WAIT_RELEASE` | 10 |
| `post_conflict_RELEASE` | 10 |

Family flags may overlap and must be computed from current state plus same-UAV transition ledger, never `collection_source`.

## Frozen cohort requirements

The frozen cohort must be separately collected and remain seed-, episode-, lifecycle-, and state-hash-disjoint from training. It must contain:

- at least 80 independent Teacher-RELEASE states;
- all stable selected proposal classes `{11,12,13,14}`;
- at least 10 independent lifecycles for each stable class; and
- sufficient predicate-defined coverage of all four RELEASE families.

No random split, row duplication, class weighting, oversampling, relabeling, or historical-row reuse is allowed.

## Remaining locks

Fresh formal V2.1 I0 recollection is now authorized under this prospective gate. I1 remains blocked until that new formal I0 passes complete provenance, replay, determinism, safety, family, selected-support, and frozen-cohort qualification.

```text
I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```
