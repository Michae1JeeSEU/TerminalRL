# Fresh V4.2 Hierarchical I1 BC on Formal Observation V2.1 I0-R

## Decision

The authorized fresh 100-epoch offline BC run completed successfully, but no
formal V2.1 I1 numerical qualification thresholds were preregistered before the
run. Per the frozen protocol, the observed metrics cannot be used to invent a
post-hoc PASS threshold. Promotion therefore fails closed:

```text
V4_2_V2_1_I1 = QUALIFICATION_THRESHOLD_NOT_PREREGISTERED
V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```

This is not an optimization crash: both losses decreased, WAIT and RELEASE were
learned, all four stable proposals were predicted, gradients remained finite,
and neither head collapsed. The main diagnostic weakness is post-conflict
RELEASE recall (`0.3333`).

## Experiment evidence

- Status / evidence type: `newly_run`, 2026-09-14.
- Environment: server `terminal_env`, PyTorch CPU execution.
- Input: formal I0-R training 2020 rows; frozen cohort 643 rows.
- Training protocol: fresh random actor, Adam, learning rate `3e-4`, batch size
  `64`, `100` epochs, seed `4201`, final-epoch checkpoint.
- Loss: unweighted gate CE on all training rows plus unweighted proposal CE on
  Teacher-RELEASE rows only.
- Frozen use: exactly one evaluation after final-epoch training; never used for
  gradients, model selection, early stopping, or threshold tuning.
- No checkpoint resume, rollout, DAgger, PPO, class weighting, focal loss,
  resampling, row duplication, label smoothing, or proposal reweighting.

Input lineage hashes:

```text
training dataset = 6bcbc1782a171f909282126c4f6d27dd93355b09513e2827093228af4fe05f05
frozen dataset   = 229e9d0c087eac1e238c90d857e74dc71525d4eac1b343ed88f33fa7e0695aa2
I0-R manifest    = 51868f7892395d7d5a369d956f6c374ba536d3c8023ec5d0f42a2d9f208f8a31
```

## Optimization and training metrics

| Metric | Random initialization | Final training |
|---|---:|---:|
| Total BC loss | 3.885685 | 0.216984 |
| Gate CE | 0.712015 | 0.149665 |
| Proposal CE | 3.173669 | 0.067319 |
| Gate accuracy | 0.259406 | 0.938614 |
| WAIT recall | 0.060606 | 0.958092 |
| RELEASE recall | 0.916844 | 0.874200 |
| Gate macro-F1 | 0.238332 | 0.914296 |
| Proposal top-1 | 0.044776 | 0.991471 |
| Proposal top-3 | 0.108742 | 0.997868 |
| Decoded joint accuracy | 0.054950 | 0.938119 |

Gradient norm mean was `3.014560`, maximum `18.223431`, and every recorded
gradient was finite. The optimizer clipped gradients at norm 1.0 after logging
the pre-clip norm.

## One-shot frozen qualification

```text
frozen_gate_ce = 0.156364
frozen_proposal_ce = 0.477458

WAIT_recall = 0.962477
RELEASE_recall = 0.790909
RELEASE_precision = 0.813084
gate_macro_F1 = 0.880809

proposal_top1 = 0.927273
proposal_top3 = 0.981818
speed_agreement = 1.000000
merge_agreement = 0.927273
decoded_joint_accuracy = 0.930016
conditional_entropy = 0.193330
```

Gate confusion matrix:

| Teacher target | Predicted WAIT | Predicted RELEASE |
|---|---:|---:|
| WAIT | 513 | 20 |
| RELEASE | 23 | 87 |

## Stable proposal evaluation

Proposal identifiers below are canonical joint IDs. The head mapping assertion
passed: `11→10`, `12→11`, `13→12`, `14→13`.

| Proposal | Frozen target rows | Recall | Prediction count |
|---:|---:|---:|---:|
| 11 | 13 | 0.846154 | 12 |
| 12 | 23 | 1.000000 | 26 |
| 13 | 64 | 0.953125 | 63 |
| 14 | 10 | 0.700000 | 9 |

Proposal 13 was the dominant prediction, but its share was `0.572727`, far
below the preregistered diagnostic collapse share `0.95`. All four stable
classes appeared in predictions; proposal collapse is `false`.

## Family-stratified gate evaluation

Predicate families are RELEASE-defined and may overlap, so all listed targets
are RELEASE.

| Family | Samples | Lifecycles | Gate accuracy / RELEASE recall |
|---|---:|---:|---:|
| clear_outer | 45 | 45 | 0.711111 |
| middle_interface | 43 | 43 | 0.790698 |
| post_WAIT | 58 | 56 | 0.775862 |
| post_conflict | 12 | 12 | 0.333333 |

The low post-conflict result is a boundary-specific diagnostic failure signal.
It does not authorize reweighting, resampling, architecture changes, or a
post-hoc threshold.

## Retry-stratified evaluation

```text
retry_active_states = 82
retry_inactive_states = 561
retry_active_false_RELEASE_rate = 0.0
retry_expired_RELEASE_states = 110
retry_expired_RELEASE_recall = 0.790909
```

The new retry feature therefore prevented false RELEASE on every retry-active
frozen state in this cohort.

## Collapse checks

```text
gate RELEASE prediction share = 0.166407
gate_collapse = false
proposal_13 prediction share = 0.572727
proposal_collapse = false
all stable classes represented = true
```

## Checkpoint provenance

```text
checkpoint = ~/training_outputs/terminal/v4_2_v2_1_i1_bc_20260914/v4_2_v2_1_i1_final_epoch.pt
checkpoint SHA256 = 3b4ddd305dd873e9f892774d4d86fa321230fd7d6c3103bb0583ce5c6c1fb725
checkpoint manifest SHA256 = f649ec2a336c3ff0756f5492915817bbd69230b217847ae09700e4e3b3f0f984
training code SHA256 = 8630cb5cddef5d4ad9eab7f12c6c63cc7aaf2a07f629dafb375696790d441a58
```

The checkpoint records the 88D observation schema, hierarchical action schema,
architecture fingerprint, all three input hashes, optimizer configuration,
seed, epochs, batch size, and final-epoch selection rule. The strict formal
loader rejects incompatible observation/action metadata and mismatched dataset
lineage.

## Protocol risks and next gate

- This is one fixed training seed; no variance estimate was authorized.
- The frozen cohort is an internal qualification cohort, not an external
  generalization test.
- V2.1 numerical I1 thresholds were absent before the run. This prevents formal
  promotion even though aggregate learnability is strong.
- Post-conflict RELEASE has only 12 frozen samples/lifecycles and recall
  `0.3333`; it requires explicit error attribution before any future protocol
  decision.

The next authorized action is to preregister an I1 qualification protocol or
perform a non-mutating post-conflict error attribution diagnostic. I2 DAgger
preflight and PPO remain blocked.
