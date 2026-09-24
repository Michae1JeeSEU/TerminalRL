# Landing-flow repair diagnostic and fresh validation protocol (2026-09-24)

This record separates the observed dense-30 baseline behavior from the
opt-in repair protocol. The prior V7 fresh paired validation has run and its
gate remains **FAILED**; the later bounded deadline-fallback check is recorded
separately below. This document does not claim universal qualification or
overall throughput improvement.

## Baseline evidence

The development replay used seed `841301` with the fixed-rule controller,
which supplied a RELEASE action on every control step for every active UAV.
The configuration was the existing 30-UAV warm start, initial SOC range
`0.55--0.90`, route guard enabled, `continuous_inward_v1`,
`downstream_priority`, and the late-slot reconciliation flag set to false.
The run continued through service completion and resolved all 30 requests at
`1364 s`.

The terminal outcome was:

| normal completion | emergency landing | failed energy | safety events |
|---:|---:|---:|---|
| 18 | 8 | 4 | 0 |

The four energy failures were UID 3 at `883 s`, UID 4 at `991 s`, UID 11 at
`992 s`, and UID 12 at `1363 s`. Hard separation violations, NMAC entries,
and collision events were all zero in this replay. The raw headless replay is
`results/flow_baseline_841301.json`.

For UID 3 and UID 4, the early candidate evaluation records while SOC was
above `0.40` were dominated by `dynamics_unreachable_merge_time`: 25 of 33
candidate evaluations for UID 3 and 33 of 43 for UID 4. The same interval
also contains `NO_MERGE_GAP`, `PREDICTED_SEPARATION`, and `HEADWAY` records.
The earliest recorded UID 3 rejection was at `t=0 s`, with
`NO_MERGE_GAP` and `predicted_holder_reachable_tube`; its first recorded
`PREDICTED_SEPARATION` with `dynamics_unreachable_merge_time` was at `t=27 s`.
For UID 4, the earliest recorded rejection was at `t=0 s`, with
`PREDICTED_SEPARATION` and `dynamics_unreachable_merge_time`; the first
`NO_MERGE_GAP` with `predicted_holder_reachable_tube` was at `t=9 s`.

At `552 s`, the recorded free landing zones were `0` and `3`. UID 3 was in
the O ring at SOC `0.221190`, already below the configured 25% landing floor.
UID 7 and UID 23 were both in the I ring, at SOC `0.444473` and `0.523895`.
Their actual release decision for that step was `emergency_lz_priority`, with
`runtime_shortlist_candidates_checked=0`. The `final_lz_pairs_checked` values
present in the telemetry are retained as prior records; they are not treated
as same-step safety rejections. The same free-pad state and decision pattern
was present at 551 s and 553 s.

The existing emergency trigger is `SOC <= 0.40` and emergency processing is
performed before the normal release pass. In addition, the forced energy-guard
path can enter the emergency procedure before that threshold: UID 1 was
committed at SOC `0.451690` and UID 2 at SOC `0.474427`. The 25% landing
reserve, 15 s emergency margin, and 80/40/10 m safety distances remain
unchanged. The repair candidate does change the emergency decision criterion:
normal release is evaluated first, and emergency takeover is gated by route
slack with an explicit normal energy-guard override.

## Opt-in development replay (V1)

The coordinated development replay is recorded in
`results/flow_dev_841301_v1.json`. It used seed `841301`, the same 30-UAV
warm-start and `0.55--0.90` initial SOC range, with
`rule_flow_coordination_enabled=true`. It completed the full service cycle in
`904 s` (`298.8513830997981` wall seconds):

| normal completion | emergency landing | failed energy | safety events |
|---:|---:|---:|---:|
| 25 | 3 | 2 | 0 |

The failed-energy records are UID 8 in the middle ring at `839 s` and UID 3
in the inner ring at `903 s`, both with `energy_remaining=0`. The minimum
actual and swept separations were `80.74365038131837 m` and
`80.74197733504654 m`. The observed normal commits for UID 3 were
`O->M=126 s` and `M->I=298 s`; UID 4 committed `O->M=106 s`, `M->I=340 s`,
and `I->final=494 s`.

## Opt-in development replay (V2)

The next development replay is recorded in
`results/flow_dev_841301_v2.json`. It retained the 30-UAV coordinated setup
and resolved all 30 requests in `984 s` (`358.9175406000577` wall seconds):

| normal completion | emergency landing | failed energy | safety events |
|---:|---:|---:|---:|
| 25 | 2 | 3 | 0 |

The failed-energy records are UID 8 in the outer ring at `829 s`, UID 9 in
the middle ring at `952 s`, and UID 11 in the middle ring at `983 s`. The
reported minimum actual and swept separations were both `80.0 m`; collision,
NMAC, hard-separation, shield-fail-closed, joint-safety-fail-closed, and
safety-error counts were zero.

## Opt-in development failure probe (V3)

`results/flow_dev_841301_v3_failure_probe.json` stopped at simulated time
`182 s` (`183` steps) on the `current_horizon` recursive gate. The reported
unsafe pair was UID `19/21`, with predicted minimum separation
`79.92809338226441 m`; runtime `joint_safety_fail_closed=1` and
`shield_fail_closed=1`. The same record reports minimum actual separation
`80.55763956221097 m` and minimum swept separation
`80.49320204291782 m`, with no actual collision, NMAC, or hard-separation
incident. This is a failed development probe, not a passed validation cell.
An earlier V3 telemetry-only record reporting zero was excluded because it
did not contain the merge exception.

## V3 original-pair witness

The follow-up record
`results/flow_dev_841301_v3_original_pair_probe.json` identifies the actual
state witness at `t=182 s`: UID 19 newly committed a normal final plan with
duration `48 s`, pad `2`, and end time `244.6961303992387 s`; UID 21 was an
uncommitted inner-ring holder. The old filter's point-tube check passed for
this pair, while the runtime pair check reported swept separation
`79.92809338226441 m`. The record's actual minimum and swept separations remain
`80.55763956221097 m` and `80.49320204291782 m`.

The new SafetyFilter nominal swept check uses the runtime `1 s` grid. Its
patched `190`-prefix verification and the V4 test are still ongoing; no pass
is claimed here. This actual-pair witness is separate from the earlier
nominal-ring numerical illustration below.

## T0 same-plan holder-envelope probe

`results/holder_envelope_probe_841301.json` contains 16 T0 candidates from the
same reset geometry: four each for UID 8, UID 3, UID 4 on `O->M`, and UID 15
on its actual `M->I` role. The baseline SafetyFilter used
`max_speed_residual_mps=2.0` and the nominal comparison used `0.0`; both
filters explicitly had `rule_flow_coordination_enabled=false`. The original
`±2 m/s` filter approved `0/16`, while the nominal zero-residual filter
approved `4/16`: UID 8 at `60/72/84 s` and UID 3 at `48 s`. Every candidate's
energy guard result was `null`.

The probe's `min_track_audit` records have `checked_pairs=0`,
`minimum_separation=null`, and `reason=not_run`; they are incomplete audits
and must not be used as distance measurements. Within this T0 same-plan
comparison, the evidence shows that a future action domain inconsistent with
the fixed-rule executed holding speed can reject nominal safe opportunities.

`results/nominal_holder_swept_probe.json` is the earlier separate analytic
calculation, not an executed V3 trajectory: for target chord `80.02 m`, the
point distance is `80.02 m` and the coarse `1 s` swept chord is
`79.9559925350114 m` (the `0.2 s` swept value is `80.01743937365667 m`). This
calculation is not proof of the V3 probe's root cause; the actual-pair witness
above is the separate runtime evidence.

The bounded V4 state probe at `t=250 s` recorded 10 inner-ring holders, no
active final UAV, and five free pads. All 250 tested pad-by-duration final
candidates were rejected: 153 at dynamics generation, 77 by holder-tube
checks, and 20 by committed-plan separation checks. The measured I-ring
neighbor chord gaps ranged from `88.86 m` to `185.92 m`. One concrete record
was UID3, P0, duration `48 s`, rejected on holder UID23 with swept minimum
`11.72012823866065 m`.

The separate departure-clearance probe tested 12 angles and 360 candidates;
142 were dynamics-valid. The best tested following gaps were chord `150--170
m` versus the static `80 m` threshold. These used a `10 m` chord grid and are
approximate tested values rather than a continuous minimum or a safety proof.
The planned V5 design derives an I-ingress gap from the existing `i2f`
headway: angle `0.96 rad`, chord `184.71 m`, effective capacity `6`; it is
intended to protect queue/service clearance before full SafetyFilter
certification and does not guarantee a future final certificate.

The bounded V4 diagnostic was later aborted at progress `900` after the
primary stopped it for the identified I-egress queue jam. The retained record
`results/flow_dev_841301_v4_aborted.json` reports `17` normal, `1` emergency,
`1` failed-energy, and `11` active states at that point. The
`no_failed_energy` gate is recorded as failed; `fullcycle_completion` and
`all_uavs_resolved` remain incomplete. Final-safety and minimum-SOC metrics
are unavailable for this aborted run.

## Opt-in development replay (V5)

The coordinated V5 replay is recorded in `results/flow_dev_841301_v5.json`.
It used the 30-UAV coordinated configuration and reached full service-cycle
resolution in `987 s` (`269.8829413000494` wall seconds):

| normal completion | emergency landing | failed energy | resolved |
|---:|---:|---:|---:|
| 23 | 4 | 3 | 30 |

The failed-energy records were UID 3 at `907 s`, UID 4 at `986 s`, and UID
11 at `986 s`. The safety summary reports minimum actual and swept
separations of `80.13392998381686 m`, zero collision, NMAC,
hard-separation, shield-fail-closed, and joint-safety-fail-closed events, and
no safety error; unresolved requests were `0`.

The separate read-only V5 missed-emergency probe is recorded in
`results/flow_v5_missed_emergency_probe.json`. At `t=330 s`, UID 3 was an
uncommitted middle-ring holder with SOC `0.37754578386941073`. For pad `1`
and a `60 s` emergency route, the candidate had remaining energy
`566.3186758041161`, required energy `506.0399406770562`, and slack
`60.27873512705986`; the full energy check and separate SafetyFilter passed.
The current slack limit was `29.0`, so this candidate was recorded as denied
by that gate. No equivalent witness was recorded for UID 4 or UID 11 within
the 450-step probe.

## Opt-in development replay (V6)

The coordinated V6 development replay is recorded in
`results/flow_dev_841301_v6.json`. It resolved all 30 requests in `765 s`
(`151.29799770005047` wall seconds):

| normal completion | emergency landing | failed energy | resolved |
|---:|---:|---:|---:|
| 25 | 5 | 0 | 30 |

The minimum landing SOC was `26.875%`. The safety summary reports minimum
actual separation `80.53418615721621 m` and minimum swept separation
`80.51995092119691 m`, with zero collision, NMAC, hard-separation,
shield-fail-closed, and joint-safety-fail-closed events and no safety error.

UID 3 committed O-to-M at `10 s`, M-to-I at `164 s` (I-ring arrival at
`224 s`), and a normal final approach at `226 s`, with touchdown at about
`289 s`. UID 4 committed O-to-M at `20 s` and later entered emergency landing
from the middle ring at `387 s`. The replay contained 15 non-zero-bias normal
commits: 9 at `-4 m/s` and 6 at `-8 m/s`.

The current regression check recorded `311` passed, `2` frozen-config SHA
failures, and `1` skipped check. The reference SHA was retained unchanged.
This remains a development result; the current fresh paired gate is recorded
as **FAILED** below and diagnosis continues.

## Opt-in development replay (V7)

V7 added the narrowly scoped `90 s` normal final duration because V6
emergency routes for UID 14 and UID 1 used `90 s`, while the V6 normal final
set did not include that duration. The development replay is recorded in
`results/flow_dev_841301_v7.json` and resolved all 30 requests in `763 s`
(`156.30827339971438` wall seconds):

| normal completion | emergency landing | failed energy | resolved |
|---:|---:|---:|---:|
| 26 | 4 | 0 | 30 |

The safety summary reports minimum actual separation
`80.53418615721621 m` and minimum swept separation
`80.51995092119691 m`, with zero collision, NMAC, hard-separation,
shield-fail-closed, and joint-safety-fail-closed events and no safety error.
This remains a development-only result. Fresh paired validation is complete
for the current two seeds, but the paired gate is **FAILED**. Seed `841501`
coordinated failed the zero-energy-failure requirement for UID 7 at `935 s`
in the middle ring; complete termination and zero safety-event counts do not
override that failed-energy result. The V7 development result has not been
used as fresh validation evidence.

The paired local runs overlap across seeds, so their recorded wall times are
run metadata rather than a controlled CPU or GUI performance benchmark. The
shorter full-cycle values reported here are simulated seconds; they do not by
themselves establish a visualization or wall-clock speedup. The new
candidate-generation cheap-guard/prescreen chain is recorded as an implemented
development path, while performance claims remain outside this evidence.

## Completed single-factor spacing ablations and UID7 probe

The completed spacing ablations are recorded in
`results/flow_dev_841301_spacing_ablation.json` and
`results/flow_dev_841501_spacing_ablation.json`. Each used the process-local
monkeypatch `disable_inner_departure_headway_guard` on
`LandingFlowCoordinator._inner_departure_headway_reason`; the records report
`production_source_modified=false` and derived inner-departure capacity `6`.

| seed | normal / emergency / failed-energy | sim s | wall s | min actual / swept m | requested / resolved |
|---:|---:|---:|---:|---:|---:|
| 841301 | 27 / 3 / 0 | 790 | 65.22123249992728 | 80 / 80 | 30 / 30 |
| 841501 | 26 / 2 / 2 | 869 | 166.5098309996538 | 80 / 80 | 30 / 30 |

The two failed-energy outcomes in the `841501` record are UID 4 and UID 2.
The primary decision is not to adopt unconditional removal of the I-ring
angular-spacing guard: the `841501` single-factor record contains those two
failures, while the `841301` `27 / 3 / 0` result alone is insufficient for
acceptance.

The read-only UID7 records are
`results/flow_v7_uid7_841501_probe.json` and
`results/flow_v7_uid7_841501_inner_spacing.json`. At `t=248 s`, `300 s`, and
`350 s`, the inspected M-to-I candidates include respectively
`bias=0,duration=96 s`, `bias=0,duration=84 s`, and `bias=0,duration=72 s`.
For each row, the probe records no energy-guard reason,
`safety_approved=true`, and `scheduler_can_reserve_at=true`, while the
additional inner-departure queue guard rejects the current release. The
associated blocker counts are `2`, with UIDs `(11,13)`, `(12,6)`, and
`(8,12)` respectively; the sources include inner holders and committed
incoming transfers. The current release rows at those times carry the
`inner_departure_headway` reason.

At `t=300 s`, UID7 has SOC `0.4236893557205689`. Its `90 s`, pad `2`
emergency candidate records `reason=null`, required energy
`542.4532740103896`, remaining slack `93.08075957046378`, and slack limit
`125.66370614359172`; the SOC emergency threshold is `0.40`, so this row is
threshold-rejected at that observation. At `t=400 s`, an inspected
`bias=-4,duration=60 s` candidate is scheduler-reservable but has
`safety_approved=false` with `safety_reason=predicted_separation`; its two
blockers are committed incoming trajectories from UIDs `0` and `1`.

## Additional completed 841501 single-factor ablations

The following completed diagnostic records were read from the corresponding
`results/flow_dev_841501_*_ablation.json` files. Every record reports
`30/30` requested/resolved, minimum actual and swept separation `80/80 m`,
zero collision, NMAC, hard-separation, shield-fail-closed, and
joint-safety-fail-closed counts, and no safety error.

| record | factor | normal / emergency / failed-energy | failed-energy UIDs | sim s | validity |
|---|---|---:|---|---:|---|
| `soc_gate_ablation` v1 | process-local internal emergency SOC threshold `1.0`; outer step gate remained `0.4` | 26 / 3 / 1 | 7 | 936 | invalid isolation |
| `soc_gate_ablation_v2` | process-local emergency-order/step wrapper, temporary threshold `1.0` | 26 / 3 / 1 | 4 | 862 | valid process-local ablation |
| `integrated_urgency_ablation` | normal-first per-UAV fallback, temporary threshold `1.0` | 26 / 3 / 1 | 4 | 862 | valid process-local ablation |
| `hold_prefix_ablation` | M-to-I hold-prefix shape, durations `60/72/84/96 s` | 25 / 4 / 1 | 4 | 862 | valid process-local ablation |
| `positive_bias_ablation` | inner tangential bias set `[0,-4,-8,2] m/s` | 27 / 1 / 2 | 4, 16 | 875 | process-local record |
| `final_capacity_ablation` | process-local final-capacity factor | 23 / 3 / 4 | 2, 4, 16, 3 | 955 | process-local record |

The v1 SOC record is explicitly invalid for isolating the outer gate because
the outer step-level threshold remained active. The v2, integrated-urgency,
hold-prefix, positive-bias, and final-capacity records retain the failed-energy
outcomes shown above. The primary decision is that these factors are not
adopted in production; diagnosis continues and delivery remains incomplete.

## Further 841501 trials and performance evidence

Four additional completed `841501` records are retained below. Each reports
`30/30` requested/resolved, minimum actual and swept separation `80/80 m`,
zero collision, NMAC, hard-separation, shield-fail-closed, and
joint-safety-fail-closed counts, and no safety error.

| trial | factor | normal / emergency / failed-energy | failed-energy UIDs | sim s | status |
|---|---|---:|---|---:|---|
| `current_height` | emergency O/M current-ring-height corridor | 27 / 2 / 1 | 7 | 936 | failed, not adopted |
| `dual_height_urgency` | temporary emergency threshold `1.0` plus O/M current-height-first fallback | 26 / 3 / 1 | 4 | 862 | failed, not adopted |
| `spiral_fallback` | Cartesian final candidate first, then inner-final spiral fallback | 25 / 3 / 2 | 11, 7 | 936 | failed, not adopted |
| `downstream_order` | downstream-order trial | 29 / 0 / 1 | 16 | 875 | failed, not adopted |

The raw records are `results/flow_dev_841501_current_height_ablation.json`,
`results/flow_dev_841501_dual_height_urgency_ablation.json`,
`results/flow_dev_841501_spiral_fallback_ablation.json`, and
`results/flow_dev_841501_downstream_order_ablation.json`.

The downstream-order JSON records UID4's normal O-to-M, M-to-I, and final
commits at `10 s -> 94 s`, `154 s -> 250 s`, and `276 s -> 350.69613039923854 s`
to pad `5`; touchdown was `351 s` and completion `471 s`. UID7's corresponding
commits were `20 s -> 104 s`, `128 s -> 200 s`, and `230 s -> 292.69613039923854 s`
to pad `1`; touchdown was `293 s` and completion `413 s`. The only
failed-energy outcome in this record was UID16 at `874 s`.

The implemented `validate_plan` batch-equivalent fast pass was checked by
`110` equivalence tests. The function microbenchmark in
`results/flow_plan_validation_microbenchmark.json` measured the new validator
at `6.7--7.1%` of the scalar oracle's median time across the three real plans;
this is a function microbenchmark and does not establish whole-simulation or
GUI speed. The M2I preview pre-screen is also recorded as implemented with
`76` tests; full authoritative validation remains in the path.

The current validation combining downstream order with cancellation of the
SOC hard gate has not produced a result. No completion claim is made for that
combination.

## Downstream urgency records and UID16 duration probe

The downstream urgency records below retain the same safety summary as the
other diagnostic cells: `30/30` requested/resolved, minimum actual and swept
separation `80/80 m`, zero collision, NMAC, hard-separation,
shield-fail-closed, and joint-safety-fail-closed counts, and no safety error.

| record | seed | normal / emergency / failed-energy | failed-energy UID | sim s | status |
|---|---:|---:|---:|---:|---|
| `downstream_urgency` | 841301 | 27 / 3 / 0 | — | 737 | not adopted |
| `downstream_urgency` | 841501 | 29 / 0 / 1 | 16 | 875 | not adopted |
| `downstream_dual_height` | 841501 | 29 / 0 / 1 | 16 | 875 | not adopted |

The source records are
`results/flow_dev_841301_downstream_urgency_ablation.json`,
`results/flow_dev_841501_downstream_urgency_ablation.json`, and
`results/flow_dev_841501_downstream_dual_height_ablation.json`. These variants
remain diagnostic records and do not replace the explicitly **FAILED** fresh
gate or establish a zero-energy-failure result.

The bounded read-only probe
`results/flow_downstream_uid16_duration_probe.json` checked `478` candidate
rows through `t=400 s` and recorded `witness=null`, so it has no witness at
`400 s`. Its probe label `m2i_72_84` is a naming label only: the probe code
generated emergency-final candidates for durations `72/84 s`, not normal M2I
ring-transfer candidates. The recorded gate counts were `m2i_72_84`:
`energy=3`, `no_free_pad=10`, `safety=34`, `slack=31`; and `retry_60_90`:
`dynamics=20`, `energy=15`, `no_free_pad=92`, `safety=209`, `slack=64`.
The UID16 state at the probe stop was `ring_middle`, SOC
`0.3166155746916731`, energy remaining `474.9233620375097`, uncommitted, with
`emergency_retry_after=400 s`.

## Latest validator equivalence and UID16 blocker evidence

The bounded record `results/flow_perf_equivalent_100.json` compares the
`validate_plan` fast path with the scalar oracle on the coordinated seed
`841301` for a `100`-step prefix. Its scope is
`validate_plan_fastpath_only`: scalar wall time was `45.31996440002695 s` and
vectorized wall time was `36.03737800009549 s`; the prefix had `24` normal and
`0` emergency commits. The exact state delta was `0.0`, `mismatches=[]`, and
the state and safety trace hashes were identical. Minimum actual and swept
separation were `82.36346079968393 m` and `82.180892081812 m`. This is a
validator-function equivalence measurement, not a whole-simulation or GUI
throughput measurement.

The read-only `results/flow_uid16_blockers_v2.json` record inspected `12`
normal M-to-I candidates for UID16 at each snapshot. The counts below keep
safety approval, scheduler proposal, and their conjunction separate:

| time s | SOC | candidates | safety approved | scheduler proposal | energy+safety+scheduler | queue headway pass |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.4499489080250065 | 12 | 3 | 6 | 1 | 0 |
| 250 | 0.41661557469167315 | 12 | 1 | 12 | 1 | 0 |
| 300 | 0.3832822413583398 | 12 | 1 | 9 | 0 | 0 |

At `t=250 s`, the UID16 `72 s`, zero-bias candidate passed energy
(`required_energy=517.8995719167109`, slack
`107.02379012079882`) and SafetyFilter, and had a scheduler proposal. Its
additional `inner_departure_headway` check failed; the recorded minimum
swept separation was `97.22229834485604 m` against holder UID6. The slack is
below the recorded one-ring period of `125.66370614359172 s`. The older
`m2i_72_84` label in the separate 478-row probe remains an emergency-final
candidate label, not a normal M-to-I ring-transfer label. This latest record
is retained as a new repair-queue-guard observation distinct from the
original legacy baseline. A deadline-aware queue-spacing fallback is pending
trial; no passed gate is claimed and these variants remain unadopted.

The current fresh paired cells are retained with their raw values:

| seed | treatment | sim s | wall s | normal / emergency / failed-energy | min landing SOC | min actual / swept m |
|---:|---|---:|---:|---:|---:|---:|
| 841501 | baseline | 1160 | 106.41245539998636 | 12 / 13 / 5 | 0.2770484728245769 | 80.04813987043184 / 80.04813987043184 |
| 841501 | coordinated | 936 | 346.33047969965264 | 26 / 3 / 1 | 0.27889638519947 | 80.0 / 80.0 |
| 841502 | baseline | 1223 | 114.8394170999527 | 15 / 9 / 6 | 0.3038196676138032 | 80.67629252217722 / 80.51421096342995 |
| 841502 | coordinated | 791 | 242.68585520004854 | 28 / 2 / 0 | 0.2848593050455483 | 80.0 / 80.0 |

The historical V7 paired gate remains **FAILED** because its `841501`
coordinated cell contains the UID 7 failed-energy outcome at `935 s`
(`ring_middle`, SOC `0.0`). The `841502` coordinated cell recorded zero
failed-energy outcomes, but that historical paired gate remains failed. These
`841501/841502` records were subsequently reclassified as development
regression inputs; the independent `841701/841702` bounded fresh check is
reported below.

## Integrated fixed-rule development regression

The frozen opt-in flow was exercised through the
`fixed_rule_dense30_coordinated` CLI using the same 30-warm-start,
`0.55--0.90` SOC, route-guard-on, `continuous_inward_v1`,
`downstream_priority`, late-slot-off configuration. The three development
records below each resolved all 30 requests with zero failed-energy outcomes;
their safety records report minimum actual and swept separation at or above
`80 m` and zero named physical/safety counters.
The raw records are `results/flow_final_dev_841301.json`,
`results/flow_final_dev_841501.json`, and
`results/flow_final_dev_841502.json`; the regression output is
`results/flow_repair_regression_delivery.txt`.

| seed | simulated s | wall s | normal / emergency / failed-energy | min touchdown SOC | min actual / swept m |
|---:|---:|---:|---:|---:|---:|
| 841301 | 757 | 264.76303469995037 | 29 / 1 / 0 | 0.3018781902911374 | 80.1532184002751 / 80.1532184002751 |
| 841501 | 783 | 217.04460870008916 | 29 / 1 / 0 | 0.2857804017484627 | 80.0 / 80.0 |
| 841502 | 840 | 255.41621090006083 | 28 / 2 / 0 | 0.2787034160630745 | 80.0 / 80.0 |

These are development records for the opt-in fixed-rule flow; the default
legacy route remains unchanged. Relative to the original seed `841301`
baseline (`18 normal / 8 emergency / 4 failed-energy`), the coordinated
record is `29 / 1 / 0`. In that record UID3's M-to-I commit was at `140 s`,
inner-ring arrival at `236 s`, and normal touchdown at `299 s`; UID4's
corresponding times were `168 s`, `252 s`, and `327 s`. Commit time and
inner-ring arrival are separate events.

The earlier baseline observations retained in this evidence identify three
separate records: early O-ring candidates were dominated by
`dynamics_unreachable_merge_time`; at `552 s`, UID7 and UID23 received
`emergency_lz_priority` with `runtime_shortlist_candidates_checked=0` while
pads `0` and `3` were free; and the old emergency-before-normal ordering plus
forced energy guard could enter emergency at SOC `0.451690` and `0.474427`.
The newer inner-queue guard is a separate later observation; the
deadline-aware fallback addresses that guard record rather than rewriting the
original baseline facts.

The delivery regression output reports `241 passed` and `2` unchanged
frozen-config source-binding failures (`terminal_marl/core/config.py`). The
focused repair subset has `82` checks; that count is not added to the
`241/2` total. The `validate_plan` equivalence remains a limited 100-step
function/prefix measurement (`45.31996440002695 s` scalar to
`36.03737800009549 s`, identical state and safety traces), not a full GUI or
overall throughput benchmark. The new development wall times above are
absolute run records only.

## Preregistered deadline-fallback check

Seeds `841701` and `841702` are reserved for the next bounded check. A search
of `docs/`, `training/`, and `tests/` found no existing references to either
seed at preregistration time. The development regressions have passed their
run-level gates, and both coordinated cases have now completed. The check
status is now **PASS** for this bounded protocol. The final outputs are
`results/flow_fresh_deadline_841701.json` and
`results/flow_fresh_deadline_841702.json`; their progress snapshots are
retained as auxiliary records. Both final records pass this bounded check:
`841701` has `30 normal /
0 emergency / 0 failed-energy`, `30/30` resolved at `863 s`, minimum touchdown
SOC `0.3451866507009634`, minimum actual/swept separation `80.0/80.0 m`, and
zero named physical/safety counters; `841702` has `27 normal / 3 emergency /
0 failed-energy`, `30/30` resolved at `789 s`, minimum touchdown SOC
`0.2919970856931216`, minimum actual/swept separation `80.0/80.0 m`, and zero
named physical/safety counters. The two independent fresh records total `60`
UAVs with `57 normal / 3 emergency / 0 failed-energy`; the three development
records plus these two fresh records total `5` scenes and `150` UAVs with
`143 normal / 7 emergency / 0 failed-energy`, overall minimum touchdown SOC
`0.2787034160630745`. The bounded fresh deadline-fallback check is **PASS**.
This does not replace the
older V7 fresh gate, which remains **FAILED**, and does not constitute universal
or formal I1/PPO qualification.

The exact PowerShell viewer command for the coordinated 30-UAV scene is:

```powershell
.\venv\Scripts\python.exe -m terminal_marl.visualization.rollout_viewer --mode fixed_rule --scenario fixed_rule_dense30_coordinated --seed 841301 --gui --labels --trajectories --playback-speed 1 --max-simulation-seconds 1800
```

Only coordinated runs will be made, with the same `30` warm-start UAVs,
initial SOC `0.55--0.90`, route guard enabled, maximum `1800` simulated
seconds, and full-cycle termination including ground service. Baseline reruns
are excluded because this check tests zero-energy and safety gates rather than
fresh throughput comparison. No manual state edits or skipped failed UAVs are
allowed. Candidate code is frozen after development, with no tuning between
the two fresh cases; any failure marks the whole check **FAILED** and returns
both seeds to development before any repair.

The hard acceptance records must show `30/30` resolved, zero failed-energy,
zero safety error, collision, NMAC, hard-separation, shield-fail-closed,
joint-safety-fail-closed, and executed-plan hash-mismatch counts; minimum
actual and swept separation must be at least `80 m` within `1e-6`, and every
successful touchdown SOC must be at least `0.25`. Emergency count and times
are descriptive fields with no independent arbitrary threshold gate. This
check remains separate from the older explicitly **FAILED** fresh gate and is
not a formal I1 or PPO qualification; `PPO_ALLOWED=false` remains active.

## Opt-in repair flow

The repair candidate is an opt-in flow for paired diagnosis. Its intended
sequence is:

1. Generate a continuous annular transfer with a finite duration and use the
   actual arrival slot selected by the scheduler.
2. Let the fixed-rule RELEASE path generate a bounded set of canonical
   candidates and order eligible candidates by remaining energy; this path is
   not a learned policy.
3. Reserve a landing pad against the generated executable route, so a pad is
   occupied by the route that actually claims it.
4. Evaluate the normal landing route first.
5. Consider a route-slack emergency route only when the normal route is not
   available under the current timing and energy constraints.
6. Run the complete SafetyFilter and execution-hash checks for every accepted
   route. The 80/40/10 m safety thresholds and the existing fail-closed
   behavior remain unchanged.

The V2 candidate set was selected to provide more dynamics-valid candidates in
the finite search: `O->M` uses `48/60/72/84 s`, `M->I` uses
`60/72/84/96 s`, and final candidates retain `20/28/36/48/60 s`. Each listed
nominal candidate passed the dynamics check in the development inspection;
this is a candidate-generation observation and is not a fresh validation
result.

The V3 implementation contract makes the action domain explicit: with
`rule_flow_coordination_enabled=true`, fixed-rule execution derives
`max_speed_residual_mps=0` and rejects non-zero speed actions; RL and
`rule_flow_coordination_enabled=false` retain `2 m/s`. Runtime certification
continues to use current-state, next-step, and handoff recursive joint+swept
checks. If the zero-residual domain has no certified action, the result is
fail-closed at the existing `80 m` desired separation; it does not inject a
temporary speed change or lower the `40 m` protected floor. SafetyFilter uses
the zero-residual prediction as supplied.

The repair flag is therefore a protocol factor, not a replacement for the
current default. The historical V7 paired cells did not pass their gate; the
default path remains unchanged and `PPO_ALLOWED=false` remains in force.

## Fresh paired validation preregistration

This section records the historical V7 paired protocol. Seed `841301` was the
development anchor, and `841501/841502` were subsequently reclassified as
development regression inputs. Each was run under both values of the single
repair factor:

| seed group | repair opt-in | role |
|---|---|---|
| 841301 | false / true | development smoke and implementation check |
| 841501 | false / true | historical V7 paired validation / development evidence |
| 841502 | false / true | historical V7 paired validation / development evidence |

All cells use the same fixed conditions: 30 warm-start UAVs, initial SOC
`0.55--0.90`, route guard enabled, `continuous_inward_v1`,
`downstream_priority`, a maximum of 1800 steps, full-cycle termination only
after service completion, and the current safety configuration. The only
paired treatment is the repair opt-in: `flagfalse` retains the `2 m/s` action
domain, while `flagtrue` derives `max_speed_residual_mps=0` and rejects
non-zero runtime speed actions; all other listed conditions are fixed. Results
from a fresh cell must not
be used to retune the repair and then reused as that cell's evidence.

Each cell must retain its raw JSON and progress record and report:

- terminal step, simulated duration, wall time, requested/resolved counts;
- normal completions, emergency count, failed-energy count;
- landing SOC for every touchdown and the minimum landing SOC;
- emergency-route duration and total run duration;
- collision, NMAC, hard-separation, shield-fail-closed, and
  joint-safety-fail-closed counts;
- safety error, release and candidate-rejection telemetry, and any incomplete
  or unresolved request.

The hard acceptance requirements are 30 resolved requests, zero
`failed_energy`, zero collision, zero NMAC, zero hard-separation violation,
zero `shield_fail_closed`, zero `joint_safety_fail_closed`, and no safety
error. A cell that misses any requirement is retained and reported as a
failed validation cell; it is not silently repaired by changing parameters.
The historical V7 validation records are complete, and their paired gate is
**FAILED** because the `841501` coordinated cell has one `failed_energy`
outcome (UID 7 at `935 s`, `ring_middle`). The `841502` coordinated cell has
`28` normal, `2` emergency, and `0` failed-energy outcomes. These two records
were subsequently reclassified as development evidence; the independent
`841701/841702` bounded fresh check completed **PASS** under frozen code, with
no tuning between the two cases. The latest `docs/CURRENT_STATUS.md` records
the bounded PASS, preserves the historical V7 failure, and keeps the existing
I0/I1 and PPO admission restrictions active.
