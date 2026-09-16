# V4.2 / Observation V2.1 I1 Formal Qualification

## Protocol

Evidence type: `newly_run`, 2026-09-14. All three preregistered final-epoch
actors were trained and hashed before the immutable 1,124-row Q cohort was
opened. The three checkpoints were then evaluated together in one invocation.
No Q sample contributed gradients, model/epoch/seed selection, threshold
tuning, or architecture changes.

```text
Q_DATASET_ACCESS_DURING_TRAINING = false
Q_EVALUATION_COUNT = 1
Q JSONL SHA256 = 8d1e80bbf7efbf8bdada1814efc3c435daa80dd00564935c2e9dd3ba466dffe5
Q manifest SHA256 = a24ef05fb78093a81945f28d91bb16907ebf443c127f2158651613ce401c40f2
```

## Per-seed hard-gate results

| Seed | WAIT recall | RELEASE recall | Gate macro-F1 | Proposal top-1 | Proposal top-3 | Decoded accuracy | Predicted proposals | WAIT/RELEASE share | Dominant proposal share | Result |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| 4202 | 0.972545 | 0.525424 | 0.786013 | 0.807910 | 0.943503 | 0.896797 | 11,12,13,14 | 0.894128 / 0.105872 | 0.531073 | FAIL |
| 4203 | 0.947202 | 0.689266 | 0.821924 | 0.819209 | 0.943503 | 0.897687 | 11,12,13,14 | 0.846975 / 0.153025 | 0.548023 | FAIL |
| 4204 | 0.970433 | 0.548023 | 0.793443 | 0.802260 | 0.949153 | 0.899466 | 11,12,13,14 | 0.888790 / 0.111210 | 0.536723 | FAIL |

Failed gates:

- Seed 4202: RELEASE recall, gate macro-F1, proposal top-1, proposal top-3.
- Seed 4203: RELEASE recall, proposal top-1, proposal top-3.
- Seed 4204: RELEASE recall, gate macro-F1, proposal top-1, proposal top-3.

All seeds passed the WAIT-recall, decoded-accuracy, required-proposal presence,
finite-gradient, prediction-share, and anti-single-proposal-collapse gates.
Because the protocol is conjunctive, neither the aggregate mean nor seed 4203's
near-threshold result can rescue qualification.

## Multi-seed aggregate diagnostics

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| WAIT recall | 0.963393 | 0.011482 | 0.947202 | 0.972545 |
| RELEASE recall | 0.587571 | 0.072499 | 0.525424 | 0.689266 |
| Gate macro-F1 | 0.800460 | 0.015478 | 0.786013 | 0.821924 |
| Proposal top-1 | 0.809793 | 0.007046 | 0.802260 | 0.819209 |
| Proposal top-3 | 0.945386 | 0.002663 | 0.943503 | 0.949153 |
| Decoded accuracy | 0.897983 | 0.001110 | 0.896797 | 0.899466 |
| WAIT prediction share | 0.876631 | 0.021083 | 0.846975 | 0.894128 |
| RELEASE prediction share | 0.123369 | 0.021083 | 0.105872 | 0.153025 |
| Dominant proposal share | 0.538606 | 0.007046 | 0.531073 | 0.548023 |

## Boundary diagnostics (not hard gates)

| Seed | Clear-outer RELEASE | Middle/interface RELEASE | Post-WAIT RELEASE | Post-conflict RELEASE | Retry-active false RELEASE | Retry-expired RELEASE |
|---:|---:|---:|---:|---:|---:|---:|
| 4202 | 0.500000 | 0.492537 | 0.326531 | 0.066667 | 0.000000 | 0.525424 |
| 4203 | 0.693548 | 0.597015 | 0.581633 | 0.433333 | 0.000000 | 0.689266 |
| 4204 | 0.467742 | 0.447761 | 0.438776 | 0.200000 | 0.000000 | 0.548023 |

Per-proposal recall for joint classes 11/12/13/14 was respectively:

- Seed 4202: `0.615385 / 1.000000 / 0.878788 / 0.454545`.
- Seed 4203: `0.615385 / 0.933333 / 0.898990 / 0.545455`.
- Seed 4204: `0.615385 / 0.933333 / 0.878788 / 0.500000`.

Post-conflict remains the weakest family, consistent with the preregistered
expected bootstrap limitation, but the formal failure already follows from
the overall hard gates and is not a post-hoc family threshold.

## Decision

```text
V4_2_V2_1_I1 = FAIL
CANONICAL_I1_CHECKPOINT = none
V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```

The seed-4202 file remains a frozen failed-qualification artifact; it is not a
canonical deployable/I2 checkpoint. Per protocol, no immediate Q rerun,
hyperparameter change, reweighting, oversampling, threshold change, or new Q
collection is authorized. The next valid activity is failure attribution.

## Protocol risks

- This Q cohort is an internal engineering qualification cohort, not evidence
  of external generalization.
- Family flags overlap and family recalls must not be aggregated as disjoint
  strata.
- Q evidence has now been consumed once; any subsequent diagnostic must not be
  represented as another untouched formal qualification.
