# V4.2 DAgger Round-1 RELEASE-Retention Regression Diagnostic

## Decision

```text
ROOT_CAUSE = D1_VS_ANCHOR_DISTRIBUTION_CONFLICT
flipped_release_state_count = 6
boundary_state_fraction = 1.0
round1_protocol_revision_required = true
round1_rerun_allowed = false

V4_2_DAGGER_R1 = FAIL              # unchanged; this diagnostic cannot promote it
I2_DAGGER_ROUND2_ALLOWED = false
PPO_ALLOWED = false
```

The result is not explained by a single retraining seed.  All five controlled
retrainings from the same I1 checkpoint, with the identical immutable D1 and
the formal Round-1 CE-only hyperparameters, missed the frozen RELEASE
retention floor of `0.6861`.  No nonzero trajectory checkpoint simultaneously
improved D1 agreement and retained the required frozen RELEASE recall.

## Experiment evidence

- Status / evidence type: `newly_run`.
- Server workdir: `~/projects/terminal`; Python 3.12.3, PyTorch 2.8.0+cu126.
- Inputs (read-only): I1 checkpoint
  `b40bdf0568ef46a9ec259207da5ca89e2fc484a63a13e70fe82c1bbfd03389e1`,
  formal R1 checkpoint
  `59049b400d696e79e58b0d33d4aa32cfaa1ee18300346bdc35c9de1a93bf5249`,
  D1 (2,002 rows)
  `d94c13f1699dfddda2624537e3ac83edcd5b4878518bfb286efd5a7bad680213`,
  and the 243-row frozen cohort.
- Output: `~/training_outputs/terminal/v4_2_dagger_r1_release_retention_diagnostic_20260911_rerun/`.
  The manifest SHA-256 is
  `88b4b6a67f3feb618d6876f105d0fec7533588c44d5e407dd4cab4820569b16a`.
- Diagnostic training: I1 initialization, D1 unchanged, 35 epochs, batch 64,
  learning rate `3e-4`, unweighted `CE(gate)` plus `CE(proposal)` only on
  Teacher-RELEASE rows.  Seeds were 4301–4305.
- Intermediate snapshots: epochs 0, 2, 9, 18, 26, and 35.  Diagnostic
  evaluation restores Torch and NumPy RNG states afterwards, so measuring an
  intermediate checkpoint cannot alter the following minibatch trajectory.
- No D1 row, label, Observation, loss, safety component, reward, or
  qualification threshold was changed.  Diagnostic checkpoints are not
  promotable training artifacts.

An earlier diagnostic attempt was discarded because its intermediate
closed-loop evaluator consumed RNG state and perturbed the remaining training
trajectory.  The rerun above corrects that observer effect; only its manifest
and values are used below.

## Frozen RELEASE transition audit

There are 72 Teacher-RELEASE cohort states.  The I1-to-R1 transition table is
conditional on those states:

| I1 prediction → R1 prediction | Count |
|---|---:|
| correct RELEASE → incorrect WAIT | 6 |
| correct RELEASE → correct RELEASE | 47 |
| incorrect WAIT → correct RELEASE | 2 |
| incorrect WAIT → incorrect WAIT | 17 |

The six harmful flips are all within the diagnostic boundary definition that
either checkpoint has `P(RELEASE)` in `[0.40, 0.60]`.  This does **not** mean
they were all weak I1 decisions: four began with I1 `P(RELEASE) >= 0.709` and
became near-boundary or WAIT after D1 fine-tuning.

| Frozen event / source | Teacher proposal | I1 P(R) → R1 P(R) | I1 margin → R1 margin | State context |
|---|---:|---:|---:|---|
| `expert_rollout`, seed 2101, UAV 2 | 12 | .584 → .220 | .338 → -1.267 | outer ring, 10 UAVs, no reservation |
| `targeted_release_opportunity`, seed 2105, UAV 0 | 13 | .709 → .466 | .891 → -.135 | outer ring, 5 UAVs, clear release context |
| `post_conflict_recovery`, seed 2113, UAV 4 | 12 | .585 → .291 | .344 → -.888 | middle ring; existing approved o2m/i2f reservations plus one rejected o2m attempt |
| `post_conflict_recovery`, seed 2113, UAV 5 | 12 | .781 → .475 | 1.274 → -.099 | same interface/reservation state as prior row |
| `post_wait_accumulated_release`, seed 2114, UAV 0 | 13 | .709 → .466 | .891 → -.135 | outer ring, 5 UAVs, clear release context |
| `post_wait_accumulated_release`, seed 2114, UAV 3 | 12 | .763 → .419 | 1.167 → -.326 | middle ring; approved o2m/i2f reservations plus rejected o2m attempt |

Every row's immutable state hash, full state snapshot, topology, release state,
and nearest five D1 events are retained in the diagnostic manifest.  The
nearest-event evidence is mixed but informative: clear outer release states
are extremely close to new student-visited WAIT labels (for example distance
0.0057 for the targeted/post-WAIT clear-release cases), while their closest
existing release exemplars are also close (0.0046).  The conflict-recovery
states have nearby D1 persistent middle-ring WAIT events and far fewer nearby
RELEASE events.  Thus the evidence is a D1 decision-boundary conflict, not a
Teacher relabeling or fabricated state issue.

## Fine-tuning trajectory

`D1 decoded disagreement` is static action disagreement against the immutable
Teacher targets in D1; the student-state columns are independent label-only
closed-loop diagnostics.  Lower is better for disagreement.

| Epoch | Frozen WAIT recall | Frozen RELEASE recall | Macro-F1 | Proposal top-1/top-3 | D1 decoded disagreement | Student gate / proposal disagreement |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .9415 | .7361 | .8513 | .9444 / .9861 | .1089 | .2240 / .8158 |
| 2 | .9415 | .6667 | .8215 | .8889 / .9861 | .1184 | .2292 / .5455 |
| 9 | .9766 | .6806 | .8546 | .9444 / 1.0000 | .0974 | .1667 / .3125 |
| 18 | 1.0000 | .5972 | .8348 | .9722 / 1.0000 | .0914 | .2865 / .7561 |
| 26 | 1.0000 | .6389 | .8545 | .9306 / 1.0000 | .0859 | .2604 / .7200 |
| 35 | .9708 | .6806 | .8500 | .9306 / 1.0000 | .0874 | .2135 / .4667 |

The frozen retention floor is `.6861`.  RELEASE recall drops below it by epoch
2 and remains below it at all later snapshots.  Although D1 decoded
disagreement improves from `.1089` to `.0874` by epoch 35 and the final
student-state gate/proposal disagreement is below epoch 0, there is no
retention-safe early checkpoint.  Therefore this is not evidence for an
early-stopping remedy selected from the frozen qualification cohort.
All six trajectory label-only validations retained zero collision, NMAC, hard
violation, SafetyInvariantError, and uncertified events; this is a diagnostic
observation, not a modification to Safety authority.

## Same-data retraining-seed audit

| Seed | WAIT recall | RELEASE recall | Macro-F1 | Decoded accuracy | D1 decoded disagreement | Student gate disagreement |
|---:|---:|---:|---:|---:|---:|---:|
| 4301 | .9708 | .6806 | .8500 | .8807 | .0874 | .2135 |
| 4302 | .9766 | .6667 | .8484 | .8807 | .0824 | .2031 |
| 4303 | .9883 | .6389 | .8451 | .8807 | .0854 | .3490 |
| 4304 | .9883 | .6806 | .8639 | .8930 | .0804 | .2917 |
| 4305 | .9942 | .6250 | .8433 | .8807 | .0789 | .3125 |

RELEASE recall mean/std/min/max is `.6583 / .0226 / .6250 / .6806`; all five
runs fail the `.6861` floor.  This excludes the `RETRAINING_VARIANCE`
classification under the requested rule.  It also excludes
`OVER_FINETUNING`: no early snapshot both improves D1 agreement and passes
frozen RELEASE retention.  Since all nonzero snapshots and all seeds lose
some frozen RELEASE recall while fitting D1 more closely, the supported
classification is `D1_VS_ANCHOR_DISTRIBUTION_CONFLICT`.

## Required follow-up boundary

The formal result remains FAIL.  Before any new Round-1 attempt, freeze a
separate DAgger development split and a multi-seed qualification protocol that
does not choose epochs on the frozen cohort.  The next diagnosis should
inspect the conflicting D1 state families and their event sources, not repair
the result with class weighting, oversampling, threshold changes, or Teacher
label changes.

## Protocol risks

- This is an internal 243-state frozen cohort and 5-seed engineering
  qualification, not an independent generalization study.
- The frozen cohort is deliberately inspected for regression diagnosis; it
  must not become an epoch-selection set for a future formal run.
- The nearest-D1 analysis is observation-distance based.  It identifies
  plausible local decision-boundary conflicts but does not by itself prove a
  causal feature attribution.
