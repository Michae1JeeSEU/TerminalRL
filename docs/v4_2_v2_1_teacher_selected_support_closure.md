# V4.2 / V2.1 Teacher-selected support closure

## Decision

```text
STABLE_TEACHER_SELECTED_SUPPORT = [11, 12, 13, 14]
RARE_TEACHER_SELECTED_SUPPORT = []
NOT_OBSERVED_TEACHER_SELECTED = [1,2,3,4,5,6,7,8,9,10,15,16,17,18,19,20,21,22,23,24,25]

stable_speed_support = [0.0]
stable_merge_support = [-1.0, -0.5, 0.0, 0.5]

proposal_12_status = STABLY_TEACHER_SELECTED
proposal_14_status = STABLY_TEACHER_SELECTED

TEACHER_SELECTED_SUPPORT_CLOSURE = PASS
```

## Evidence

- Evidence type: `newly_run` plus read-only historical artifacts.
- New support bank: 460 independent RELEASE lifecycles from 70 new, globally disjoint seeds across seven preregistered strata.
- Historical comparison: the 109-lifecycle audit selected `{11,12,13,14}`; the prior 40 predicate witnesses selected `{11,13}`.
- Proposal 12 targeted validation: 30 predicate-valid independent witnesses, `STABLY_TEACHER_SELECTED`.
- Proposal 14 targeted validation: 10 predicate-valid independent witnesses, `STABLY_TEACHER_SELECTED`.
- ActionTape: 70/70 episode tapes replayed exactly.
- Teacher determinism: PASS.
- Safety: collision, NMAC, hard violation, `SafetyInvariantError`, and uncertified execution all zero in every stratum.
- Formal I0 rows created: 0.

The new RELEASE lifecycle distribution was:

| Stratum | Independent RELEASE lifecycles |
| --- | ---: |
| proposal-12 context | 80 |
| proposal-14 context | 50 |
| nominal mixed traffic | 90 |
| release-ready low traffic | 30 |
| resource pressure | 70 |
| interface transition | 60 |
| ordinary expert rollout | 80 |

Selection counts were proposal 11: 70, proposal 12: 110, proposal 13: 240, and proposal 14: 40. Independent seed counts were 50, 50, 70, and 40 respectively. No RELEASE row was duplicated or selected into the audit based on its final proposal label.

## Support discovery curve

At 50 lifecycles the cumulative support was `{11,12,13}`. Proposal 14 appeared before 100 lifecycles, after which support remained exactly `{11,12,13,14}` at 150, 200, 250, 300, 350, 400, 450, and 460 lifecycles. This supplies the requested saturation evidence.

`CATALOGUE_SUPPORT = 25` and all 25 proposals were certified at least once in the new state search. Certification is not Teacher selection: only 11–14 entered qualified selected support.

## Reproducibility and protocol risk

Command:

```text
python -m terminal_marl.training.run_v4_2_v2_1_teacher_selected_support_closure \
  --output-dir ~/training_outputs/terminal/v4_2_v2_1_teacher_selected_support_closure_20260913
```

Manifest SHA256: `68a80abf287bc8260f34e3c73cc2fdba84b6358bfb8d67f3cce14eefe8a52ad1`.

This is bounded internal simulator evidence, not a global claim over arbitrary geometry or traffic distributions. The closure applies to the frozen nominal V4.2/V2.1 Teacher, catalogue, ranking, SafetyFilter, and declared strata.
