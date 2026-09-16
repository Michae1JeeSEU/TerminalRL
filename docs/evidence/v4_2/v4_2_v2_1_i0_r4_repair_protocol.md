# I0-R4 prospective repair and curriculum assessment

Registered 2026-09-16, before formal collection. Historical R3 and I1 remain FAIL.
The executable protocol is `terminal_marl/configs/low_v4_2/i0_r4_repair_protocol.yaml`.

## Diagnosis from existing artifacts

`diagnose_v4_2_i0_r3.py` reproduced R3 cycle prefixes: training rows
760/1515/2278/3032/3786/4541, with C05 independent lifecycles
2/4/6/8/10/12. Meeting C05 >=15 at the observed rate requires eight
cycles (about 6,050 rows), incompatible with the 4,200-row ceiling.
The audit needs two cycles to exceed its 800-row minimum, but two cycles
yield 1,511 rows, exceeding its 1,400-row ceiling. More collection under
the same stopping design cannot repair these contradictions.

Several nominally distinct R3 recipes have identical effective configs.
`SourceSpec.phases` selects the target in the older single-UAV collector;
it does not change reset layout, and R3 collects all active UAVs. The new
schedule uses explicit N/static-or-transition/final-capacity/horizon names.
It does not claim that a scenario name forces a particular phase or label.

R3-to-consumed-Q shared-FN coverage at radii .25/1.0 was .288/.673.
Removing mission and entry-gate coordinates for a diagnostic only gives
.481/.827. Shared-proposal-error coverage changes from .333/.533 to
.667/.933. A mismatched one-hot identity contributes sqrt(2) distance
even if the remaining state is identical. Identity is not the whole FN
problem: topology and resource terms also contribute. This analysis does
not establish that R4 generalizes or that any actor qualifies.

## Curriculum decision

Retain fresh structured BC -> Safe DAgger -> student-only closed-loop
qualification -> PPO -> traffic/geometry curriculum. Preserve execution
and safety authority and requested-action PPO semantics. Warm-start BC
is a bootstrap, not evidence of natural-arrival landing performance.

For prospective R4 only, 88D distance becomes diagnostic rather than a
hard surrogate for learnability. Keep R3's actual context/lifecycle minima,
nominal anchors, schema/retry provenance, determinism, exact replay,
independence, aliasing, and zero safety violations as hard gates. R4 PASS
permits registration of fresh I1 protocol V2; it cannot authorize PPO.
Fresh multi-seed prediction and closed-loop qualification must determine
learnability and operational readiness. Do not retrospectively reclassify R3.

## Fixed collection

26 slots: the 20 existing effective R3 configurations, explicitly named,
plus two N=5 static 12-step slots, N=12 transition 42 steps,
N=12 static final-capacity-zero 12 steps, N=8 static 12 steps,
and N=6 static 12 steps. Repeated slots use independent seeds; no rows
are duplicated or accepted/discarded based on labels. The unchanged
semantic-event sampler retains events from every active UAV.

Training: 8 complete cycles, 208 episodes, seeds 510001–510208,
5,000–12,000 retained rows. Internal audit: 2 complete cycles, 52 episodes,
seeds 520001–520052, 1,200–3,500 rows. The primary budget is episode count;
row ceilings are fail-closed resource limits. Check ceilings after each
complete episode; preserve any overrun and fail rather than truncating rows.
The entire schedule, seeds and protocol hash are written before first reset.
Historical JSON and JSONL seeds/identities are checked first.

Pilot: two configurations per partition, one step each, seeds 590001/590002
and 590101/590102. Pilot outputs cannot qualify or enter formal training.
Audit cannot train or select models. Old Q and R3 contribute no new rows.
Q2 remains uncollected and reserved at 430001–439999.

## Remaining implementation gaps

Existing I2 preflight expects a historical 87D derived dataset/checkpoint.
`JointPPOV42` currently has no PPO update method. These require explicit
88D integration and requested-action rollout/update tests before PPO
admission; data qualification alone cannot make the current code train PPO.
