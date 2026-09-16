# V4.2 Fresh I1 Hierarchical BC Qualification — PASS

## Experiment evidence

- Status / evidence type: `newly_run`.
- Workdir: `~/projects/terminal` on `lab-server`.
- Input: the lineage-PASS V4.2 derived target view, containing exactly 1,500
  train rows and a separate 243-row frozen cohort. No I0 state was recollected,
  copied, reweighted, or relabelled.
- Fresh actor: shared Observation V2 encoder, `Categorical(2)` gate and
  conditional `Categorical(25)` proposal head; no V4/V4.1 checkpoint was
  resumed.
- Configuration: seed 4201, 100 epochs, batch size 64, Adam learning rate
  0.0003. The objective was unweighted `CE(gate)` on all rows plus
  `CE(proposal)` only on Teacher-RELEASE rows; each term uses its own valid
  sample mean.
- Output: `~/training_outputs/terminal/v4_2_i1_hierarchical_bc_20260910/`.

## Results

| Metric | Before | After train | Frozen cohort |
|---|---:|---:|---:|
| Gate CE | 0.7276 | 0.0989 | 0.2863 |
| Proposal CE | 3.3085 | 0.2477 | 0.1937 |
| Gate WAIT recall | 0.0787 | 0.9828 | 0.9415 |
| Gate RELEASE recall | 0.9223 | 0.6990 | 0.7361 |
| Gate release precision | 0.0687 | 0.7500 | 0.8413 |
| Gate macro-F1 | 0.1366 | 0.8520 | 0.8513 |
| Proposal top-1 (Teacher RELEASE only) | 0.0000 | 0.9126 | 0.9444 |
| Proposal top-3 (Teacher RELEASE only) | 0.0097 | 0.9903 | 0.9861 |
| Proposal speed agreement | 0.0388 | 0.9612 | 0.9722 |
| Proposal merge agreement | 0.0874 | 0.9320 | 0.9583 |
| Decoded executable-action accuracy | 0.0733 | 0.9633 | 0.8807 |
| Conditional entropy | 2.3603 | 0.1128 | 0.3006 |

Frozen-cohort predictions comprise 180 WAIT and 63 RELEASE actions. Predicted
RELEASE proposals use six proposal values (`0, 11, 12, 13, 14, 20`), so the
qualification shows neither all-WAIT/all-RELEASE nor single-proposal collapse.
Gradient norms were finite (mean 3.5545, maximum 41.0636).

No PPO call, DAgger rollout, reward/Observation/Teacher/safety modification,
or provenance/safety regression occurred.

```text
V4_2_I1_HIERARCHICAL_BC = PASS
I2_DAGGER_PREFLIGHT_ALLOWED = true
PPO_ALLOWED = false
```

## Protocol risks

This is one deterministic training seed and uses the frozen cohort as internal
validation, not an independent external test. The result qualifies the next
engineering gate; it is not a claim of external generalization. I2 remains
not run and PPO remains prohibited.
