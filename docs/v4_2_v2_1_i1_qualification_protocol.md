# V4.2 / Observation V2.1 I1 Qualification Protocol — Preregistered V1

## Freeze and scope

This protocol is frozen on 2026-09-14 before collecting or inspecting the new
`V2_1_I1_Q` cohort and before training seeds 4202–4204. It applies only to a
future fresh I1 qualification run.

```text
I1_PROTOCOL_PREREGISTERED = true
diagnostic seed-4201 checkpoint eligible = false
qualification_evidence_consumed for old 643 rows = true
```

The old 643-row cohort remains `DIAGNOSTIC_REFERENCE`. Neither its metrics nor
the post-conflict attribution can qualify the existing checkpoint.

## Fixed model and training protocol

```text
Observation = phaseA_low_obs_v2_1_retry_state (88D)
Action = low_action_schema_v4_2_hierarchical
Architecture = unchanged shared encoder + gate(2) + conditional proposal(25)
Training data = immutable formal I0-R, 2020 rows
Training SHA256 = 6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05
Fresh seeds = [4202, 4203, 4204]
Canonical checkpoint seed = 4202
Adam learning rate = 3e-4
Batch size = 64
Epochs = 100
Checkpoint = final epoch
Loss = mean CE(gate) on all rows + mean CE(proposal) on Teacher-RELEASE rows
```

All three actors start from fresh initialization and use identical data and
configuration. There is no checkpoint resume, hyperparameter search, class
weighting, focal loss, over/undersampling, balanced sampler, label smoothing,
or threshold tuning. The unique checkpoint for a later I2 is seed 4202,
selected here independently of any qualification result.

## Hard gates fixed before Q-cohort collection

Every seed must satisfy every hard gate on the untouched Q cohort:

| Metric | Required value |
|---|---:|
| WAIT recall | >= 0.90 |
| RELEASE recall | >= 0.70 |
| Gate macro-F1 | >= 0.80 |
| Proposal top-1, Teacher-RELEASE only | >= 0.85 |
| Proposal top-3, Teacher-RELEASE only | >= 0.95 |
| Decoded joint accuracy | >= 0.85 |
| Finite gradients throughout training | true |
| Predictions contain proposals 11, 12, 13 and 14 | true |
| WAIT prediction share and RELEASE prediction share | each >= 0.05 |
| Dominant proposal prediction share | < 0.95 |

Stable-proposal representation is evaluated on Q Teacher-RELEASE states and
uses canonical joint IDs 11–14. A proposal counts as represented only if it is
predicted at least once; Q-cohort target support requirements are defined in
the separate cohort protocol.

The multi-seed aggregation rule is conjunctive:

```text
PASS iff all 3 seeds pass all per-seed hard gates.
```

Mean, standard deviation, minimum, and maximum are reported for every numeric
metric, but aggregates cannot rescue a failing seed. No best-seed selection is
permitted.

## Threshold rationale

The thresholds are based on I1's role as a non-collapsed BC bootstrap and the
previously qualified V4.2/Observation-V2 engineering baseline, not on whether
the consumed seed-4201 V2.1 checkpoint passes:

- Historical qualified V2 values were WAIT recall `0.9415`, RELEASE recall
  `0.7361`, macro-F1 `0.8513`, proposal top-1 `0.9444`, proposal top-3
  `0.9861`, and decoded accuracy `0.8807`.
- The registered floors retain a conservative margin below that prior
  qualified baseline while still requiring both gate classes, a useful
  executable joint policy, and a learned conditional proposal head.
- I1 is a bootstrap for Safe DAgger, not a requirement to solve every rare
  decision boundary before student-visited acquisition begins.

These values are immutable for this protocol version. Changing any value
requires a new protocol version and a new untouched Q cohort; it cannot alter
the result of an already-run qualification.

## Boundary diagnostics

The following are reported but are not V1 hard gates:

```text
clear_outer RELEASE recall
middle_interface RELEASE recall
post_WAIT RELEASE recall
post_conflict RELEASE recall
retry-active WAIT false-RELEASE rate
retry-expired RELEASE recall
per-proposal 11/12/13/14 recall
```

This decision is scientific rather than permissive: family strata can overlap,
and the earlier post-conflict statistic had only 12 lifecycles from two
episodes. The new Q cohort increases family support, but the purpose of I2
Safe DAgger remains acquisition at difficult student/decision boundaries.
Any exact opposite-label observation hash conflict, provenance failure, or
Safety/schema regression is nevertheless a stop condition outside the metric
table.

## One-time evaluation and decision

The Q cohort stays inaccessible to model code until all three final-epoch
checkpoints and training logs are complete and hashed. It is then evaluated
once for all three actors. It is never used for backpropagation, early stopping,
checkpoint selection, seed selection, threshold selection, or architecture
choice.

```text
if conjunctive rule passes:
  V4_2_V2_1_I1 = PASS
  V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = true
else:
  V4_2_V2_1_I1 = FAIL
  V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false

PPO_ALLOWED = false
```

Until that future evaluation, I2 remains blocked.
