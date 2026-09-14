# Teacher-selected proposal support qualification

## Support distinctions

| Support | Result |
| --- | --- |
| `CATALOGUE_SUPPORT` | 25 |
| `CERTIFIED_SUPPORT` | 22 proposals: 1-3, 6-8, 10-25 |
| `TEACHER_SELECTED_SUPPORT` | joint indices `{11, 13}` |
| Selected speed residuals | `{0.0}` |
| Selected merge residuals | `{-1.0, 0.0}` |

The 40 validated witnesses selected proposal 11 twenty times and proposal 13 twenty times. The other proposals are not silently called unreachable: their sidecars separate “never certified in this search” from “certified but dominated by the fixed Teacher preference order.” Certification is state-conditional, so the 22-proposal union does not imply simultaneous feasibility in every state.

## Gate decision

All four family recipes are stably reachable, yet the validated Teacher-selected support is below the frozen `joint >= 8`, `speed >= 4`, `merge == 5` coverage rule. Therefore:

```text
ROOT_CAUSE = PRE_REGISTERED_COVERAGE_GATE_EXCEEDS_REACHABLE_EXPERT_SUPPORT
coverage_gate_revision_required = true
prospective_coverage_gate_revision_approval_required = true
scenario_recipe_revision_required = false
FRESH_V2_1_FORMAL_I0_RECOLLECTION_ALLOWED = false
```

No gate has been relaxed. A separate prospective approval must first state a reachable-expert-support justification and replace the old coverage criterion before any fresh formal I0 recollection can be authorized.
