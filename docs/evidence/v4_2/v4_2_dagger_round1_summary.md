# V4.2 Safe DAgger Round 1 — FAIL

## Experiment evidence

- Status / evidence type: `newly_run`.
- Workdir: `~/projects/terminal` on `lab-server`.
- Initial student: the fresh V4.2 I1 PASS checkpoint, SHA-256
  `b40bdf0568ef46a9ec259207da5ca89e2fc484a63a13e70fe82c1bbfd03389e1`.
- Parent dataset: the read-only V4.2 derived I0 view, SHA-256
  `681119824a739c6ec08b7900d821e4969eadfa85a9a9a307d0e5f618dfcf61fe`.
- Collection: 14 independent seeds (9801–9814), each with a short 10-step
  student-authoritative trajectory. Teacher was label-only and was queried on
  the student-visited pre-action state; it did not replace the execution
  request.
- Retraining: continued only from the I1 checkpoint with unweighted
  `CE(gate)` on all rows plus `CE(proposal)` on Teacher-RELEASE rows. No PPO,
  reward/Observation/Teacher/Safety change, focal loss, weighting, or
  oversampling was used.

## Dataset and provenance

The semantic sampler retained only event starts, Teacher-target changes,
meaningful normalized-observation changes, or bounded sparse persistence. It
processed 1,120 raw decision ticks and retained 502 unique real
student-visited events (temporal duplication factor 2.231), rather than using
raw ticks as rows.

`D1_round1` contains the 1,500-row parent dataset plus these 502 events:
2,002 rows total. Parent hash, I1 checkpoint hash, round number, Teacher
version, V4.2 action schema, event uniqueness, source tag, and aggregate hash
are recorded in `D1_round1.manifest.json`. D1 SHA-256 is
`d94c13f1699dfddda2624537e3ac83edcd5b4878518bfb286efd5a7bad680213`.

Student execution, Teacher label-only provenance, sampled gate/proposal,
requested joint log-probability, applied/shield action, source/seed/episode/
UAV/lifecycle/decision IDs, and observation/state hashes were recorded for
every retained event. A complete 10-step ActionTape was replayed exactly,
including a direct recomputation of each requested joint log-probability.

## Fixed cohort comparison

| Frozen-cohort metric | I1 | DAgger R1 | Change |
|---|---:|---:|---:|
| WAIT recall | 0.9415 | 0.9708 | +0.0292 |
| RELEASE recall | 0.7361 | 0.6806 | -0.0556 |
| Gate macro-F1 | 0.8513 | 0.8500 | -0.0013 |
| Proposal top-1 | 0.9444 | 0.9306 | -0.0139 |
| Proposal top-3 | 0.9861 | 1.0000 | +0.0139 |
| Speed agreement | 0.9722 | 0.9861 | +0.0139 |
| Merge agreement | 0.9583 | 0.9306 | -0.0278 |
| Decoded executable-action accuracy | 0.8807 | 0.8807 | 0.0000 |

The R1 retention gate allows at most a 0.05 reduction from I1 per required
fixed-state component. RELEASE recall declined by 0.0556, slightly exceeding
that bound (required minimum 0.6861; observed 0.6806). This is the sole
failing Round-1 gate. RELEASE learning itself is not collapsed, and proposal
learning is retained.

## Closed-loop diagnostics and safety

Collection predicted 418 WAIT / 84 RELEASE; Teacher labels were 350 WAIT /
152 RELEASE. On independent label-only validation seeds, the fine-tuned
student predicted 138 WAIT / 54 RELEASE versus Teacher 173 WAIT / 19 RELEASE.
Gate disagreement decreased from 0.3466 to 0.2552; proposal disagreement on
Teacher-RELEASE states decreased from 0.8158 to 0.4737. Multiple proposal
values remained active, so no all-WAIT, all-RELEASE, or single-proposal
collapse occurred.

This short diagnostic had zero completions, throughput, queue growth, and
shield interventions; those are not Round-1 performance gates. Safety was
fully PASS: collision, NMAC, hard violation, SafetyInvariantError, and
uncertified were all zero.

## Decision and protocol limits

```text
V4_2_DAGGER_R1 = FAIL
I2_DAGGER_ROUND2_ALLOWED = false
PPO_ALLOWED = false
```

No Round 2, PPO, or promotion is authorized from this result. The evidence is
an internal 14-seed engineering qualification with one frozen cohort, not an
external generalization or throughput claim. The appropriate next step is a
separate, explicitly authorized diagnosis of the small fixed-cohort RELEASE
recall regression; it must not be repaired through weighting, oversampling, or
label changes.
