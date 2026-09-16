# Observation V2.1 final freeze decision

```text
OBSERVATION_V2_1_FINAL_FREEZE = FAIL
FORMAL_V2_1_I0_RECOLLECTION_ALLOWED = false

I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```

The freeze is denied for two independent reasons:

1. The Teacher's WAIT/RELEASE gate depends on a bounded but exhaustive,
   proposal-conditioned trajectory/SafetyFilter certification scan that the
   deployed actor cannot observe before requesting one action.
2. The audit requires independently matched reservation/headway/conflict
   rationale pairs. Where the current executed path provides insufficient
   actual pairs, the corresponding gate remains `NOT_AVAILABLE`, never PASS.

The completed server audit contains 300 unique semantic states from 38
independent seeds. Its retry, commit and resource matched-pair probes pass,
but reservation (0/0 train/dev), interface/headway (17/0), and
conflict/recovery (6/1) lack the required independent pair coverage. This
freeze decision is therefore based on newly run evidence, not a projected
metric.

No additional observation feature is authorized by this decision. In
particular, proposal `can_reserve`, trajectory feasibility, SafetyFilter
certificate and holder-tube certificate remain action-conditioned runtime
computations. Any next step must separately resolve Teacher supervision
semantics against the student deployment information set, then repeat this
audit before formal I0 recollection.
