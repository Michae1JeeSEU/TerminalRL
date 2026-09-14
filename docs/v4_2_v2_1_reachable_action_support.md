# V4.2 / V2.1 reachable-action support

All values below use proposal IDs as canonical V4.1 joint indices 1–25.  This
avoids the historical bug of interpreting an item’s position in the Teacher
preference-sorted sidecar as its proposal ID.

| Support notion | Result | Meaning |
| --- | --- | --- |
| `CATALOGUE_SUPPORT` | 25 | every V4.2 execution proposal exists in the fixed catalogue |
| `CERTIFIED_SUPPORT` | 25 | every proposal was final-certified at least once among the 109 historical RELEASE states |
| `TEACHER_SELECTED_SUPPORT` | 11, 12, 13, 14 | Teacher-selected support in those states |

The certified support covers all five speed and all five merge residual values.
Teacher-selected support covers speed `0.0` only and merge residuals
`{-1.0, -0.5, 0.0, +0.5}`.  Selection counts are proposal 11: 13, 12: 27,
13: 60, and 14: 9.

This distinction is material: the first line is a catalogue property, the
second is state-conditional feasibility evidence, and the third is the actual
supervision distribution.  It does not license changing the 25-proposal
catalogue, modifying Teacher ranking, or relaxing the I0 diversity gate.

The first targeted legal-recipe run supplied no new RELEASE witness.  Since
the recipe did not instantiate its advertised semantic predicates, its zero is
not evidence of global non-reachability.  The next permitted diagnostic must
validate predicate-defined recipes before it can classify any proposal as:

- `STABLY_REACHABLE`: at least 10 independent seed/lifecycle witnesses;
- `RARE_BOUNDARY_REACHABLE`: one to nine; or
- `NOT_REACHABLE_IN_AUDITED_SPACE`: zero under a validated, bounded search.

Until then, there is no evidence to reduce the catalogue or alter the frozen
coverage threshold.
