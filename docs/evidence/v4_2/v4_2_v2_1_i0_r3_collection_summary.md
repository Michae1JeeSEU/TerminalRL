# Fresh V4.2 / Observation V2.1 I0-R3 Collection and Qualification

## Formal decision

```text
formal_training_rows = 4541
formal_internal_audit_rows = 1511

training_seed_count = 120
audit_seed_count = 40

C01 = 48 lifecycles / 48 episodes (PASS)
C02 = 18 lifecycles / 18 episodes (PASS)
C03 = 24 lifecycles / 12 episodes (PASS)
C04 = 156 lifecycles / 48 episodes (PASS)
C05 = 12 lifecycles / 12 episodes (FAIL; minimum 15/5)
C06 = 18 lifecycles / 18 episodes (PASS)
C07 = 48 lifecycles / 12 episodes (PASS)
C08 = 36 lifecycles / 12 episodes (PASS)
C09 = 78 lifecycles / 48 episodes (PASS)

proposal14_N5_lifecycles/episodes = 12/12 (FAIL)
proposal14_N6_lifecycles/episodes = 18/18 (PASS)
proposal14_N12_lifecycles/episodes = 48/48 (PASS)

shared_FN_coverage_0_25 = 0.2884615385
shared_FN_coverage_1_00 = 0.6730769231

shared_proposal_error_coverage_0_25 = 0.3333333333
shared_proposal_error_coverage_1_00 = 0.5333333333

post_WAIT_coverage_0_25 = 0.3469387755
post_WAIT_coverage_1_00 = 0.5918367347

transition_retry_coverage_0_25 = 0.3928571429
transition_retry_coverage_1_00 = 0.6428571429

post_conflict_coverage_0_25 = 0.3500000000
post_conflict_coverage_1_00 = 0.6166666667

proposal11_coverage_0_25 = 0.3846153846
proposal11_coverage_1_00 = 0.6923076923

proposal14_coverage_0_25 = 0.5000000000
proposal14_coverage_1_00 = 0.6818181818

retry_provenance = PASS
teacher_determinism = PASS
replay = PASS (40/40, exact_replay_rate=1.0)
safety = PASS
independence = PASS
aliasing = PASS (opposite-gate exact conflicts=0)

V4_2_V2_1_I0_R3 = FAIL
V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED = false
V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```

## Gate attribution

The fresh lineage, nominal anchors, stable Teacher support, 88D/retry
provenance, Teacher determinism, ActionTape replay, Safety, aliasing, and
partition/historical independence all passed. The formal failure is
conjunctive and cannot be overridden:

- The complete-cycle stopping rule produced 4,541 training and 1,511 audit
  rows, above the frozen ceilings of 4,200 and 1,400.
- C05 / proposal-14 N=5 reached 12 independent lifecycles, below 15.
- Both shared error-manifold radius gates and all five family/proposal radius
  gates missed the frozen 0.75 at radius 0.25 and 0.90 at radius 1.00 minima.

All nominal anchors passed: ordinary WAIT 672, retry-active WAIT 420,
committed WAIT 402, interface-frozen WAIT 444, resource WAIT 54,
clear-outer RELEASE 300, middle/interface RELEASE 306, and proposal
11/12/13/14 RELEASE lifecycles 114/162/432/90. Teacher-selected support was
`[7, 11, 12, 13, 14, 15]`, which contains the required stable support.

## Diagnostic-only evidence

The k=10 R3-to-reference RELEASE-fraction medians were 0.60 overall, 0.60
post-WAIT, 0.55 transition/retry, 0.50 post-conflict, 0.55 proposal 11, and
0.50 proposal 14. These are `DIAGNOSTIC_ONLY`, not qualification gates.

The R3 global RELEASE prior was 834/4,541 = 0.18366; consumed Q was
177/1,124 = 0.15747. Proposal-14 traffic TV was 0.23939, while the largest
reported phase/ring/interface TV was 0.25455. The frozen bootstrap diagnostic
for that dominant formal dimension had median 0.25455 and 95% interval
[0.02727, 0.48182]. TV is not a hard gate.

## Evidence inventory and locks

- Collection, replay, provenance and diagnostics are `newly_run`.
- Consumed Q and the frozen failure consensus are `preexisting_artifact` and
  contributed zero training or audit rows.
- Internal audit remains `NEVER_TRAIN / NEVER_MODEL_SELECT`.
- No I1 actor training, Q2 collection, DAgger or PPO was run.
