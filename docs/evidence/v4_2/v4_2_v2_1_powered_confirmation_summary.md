# V4.2 / Observation V2.1 Powered Independent Confirmation

## Final result

```text
RETRY_STATE_ONLY_CLOSURE = PASS
OTHER_STATE_ONLY_CLOSURE = PASS
GATE_PROBE_A = PASS
PROPOSAL_PROBE_B = PASS
PROPOSAL_PROBE_C = PASS

TEACHER_PRIVILEGED_INFORMATION_RISK = RESOLVED_FOR_SUPERVISION
OBSERVATION_V2_1_FINAL_FREEZE = PASS
FORMAL_V2_1_I0_RECOLLECTION_ALLOWED = true

I1_ALLOWED = false
DAgger_ALLOWED = false
PPO_ALLOWED = false
```

This is a newly run, audit-only confirmation. It creates no formal I0 rows
and did not run I1, DAgger, PPO, or modify the Teacher, Observation, Safety,
reward, or action factorization.

## Frozen independent bank

- Bank: `~/training_outputs/terminal/v4_2_v2_1_powered_confirmation_20260911/powered_independent_state_bank.jsonl`
- SHA-256: `07db1dddc8e99539f7c5f510644759a0de4577fd9c5dfef13ecb66a6a23dc1c7`
- 3,935 unique full states from 330 new seeds/episodes; neither previous 300-
  nor 600-state bank was reused.
- Train/dev are seed-, lifecycle-, and state-hash-disjoint: 3,150 / 785 states
  and 264 / 66 seeds. All 25 proposal outcomes remain in their state split.
- Stop rule passed without duplication or label selection: train RELEASE 182
  (minimum 100), dev RELEASE 40 (minimum 40), dev retry-WAIT 440 (minimum 40).

Six natural strata were rotated before Teacher evaluation: release-ready low
traffic (636), clear outer release (659), middle/interface release (660),
transition/retry (660), mixed traffic (660), and resource pressure (660).
Target-UAV selection used only fixed phase preference, seed, and step, never a
Teacher label, certificate, or catalogue outcome.

## Revised retry closure

Retry semantic closure is direct runtime equality, not the historic
limited-sample learned retry-pair diagnostic. The appended index 87 feature is
recomputed from `max(0, safety_retry_after - env.now) / 30`. It is finite,
bounded, deterministic, preserves the 87D prefix byte-for-byte, has no future
leakage, and passes exact ActionTape replay. Historic FAIL artifacts remain
unaltered.

## Fixed-protocol probe results

| Probe | Input → target | Result | Metrics |
|---|---|---|---|
| A | 88D observation → any certified RELEASE | PASS | accuracy 0.9962; macro-F1 0.9798; WAIT recall 0.9991; RELEASE recall 0.9417; AUC 0.9998 |
| B | 88D + proposal identity → proposal certified | PASS | macro-F1 0.8736; AUC 0.9963; rejected recall 0.9973; certified recall 0.6595 |
| C | 88D observation → Teacher proposal on RELEASE states | PASS | top-1/top-3 0.95/0.95; speed agreement 0.95; merge agreement 1.00 |

All probes are linear, use no class weighting, and retain the pre-registered
0.5 decision threshold. A satisfies its `.80/.80/.80/.80/.85` gate; B satisfies
macro-F1 `.80` and AUC `.85`; C satisfies top-1 `.60` and top-3 `.80`.

Probe A has two false-negative RELEASE states and zero false-positive WAIT
states. Both are clear outer RELEASE states with two certified proposals. They
show no common missing state-only variable, and the powered dev RELEASE support
is 40; they therefore do not justify another Observation revision.

For B, reservation, dynamics, and SafetyFilter rejection modes were naturally
`NOT_OBSERVED_IN_DOMAIN`; none was fabricated. Holder-tube rejection was
observed (1,515 train / 371 dev), with fixed-threshold rejected recall 1.0.

## Closure boundary and evidence

Interface-frozen and final-capacity pairs each pass with 182 train and 40 dev
pairs; all five pair metrics equal 1.0. Committed and LZ-negative pairs were
not naturally observed and are recorded as `NOT_AVAILABLE`, not manufactured.
No unresolved state-only missing feature remains.

Reservation, dynamics, SafetyFilter, and holder-tube results remain
action-conditioned and are tested only in Probe B. The actor receives none of
them and still sends one decoded action through the unchanged execution,
trajectory, SafetyFilter, shield, and environment boundary.

Qualification manifest:
`~/training_outputs/terminal/v4_2_v2_1_powered_confirmation_qualification_20260912/v4_2_v2_1_powered_confirmation.manifest.json`.
This only authorizes a fresh V2.1 I0 recollection. Historical 87D I0 cannot be
derived, because its snapshots omit `safety_retry_after`; I1, DAgger, and PPO
remain forbidden.
