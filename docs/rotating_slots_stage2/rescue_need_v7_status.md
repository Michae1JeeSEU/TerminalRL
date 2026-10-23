# rescue_need_v7: 600 s results, validation and continuation protocol

All six method runs reached the requested 600 s boundary and all three campaign wrappers exited 0. At 600 s, CP versus EnergySlack achieved 30 versus 28 total landings in stress, 29 versus 27 in diagnostic, and 26 versus 29 in medium. All six runs recorded zero energy failures. The original replay audits are **3 PASS / 3 FAIL**, with seven exact `rescue_warning` frame omissions; this is not an unconditional acceptance of the full experiment. Human visual acceptance remains **HUMAN_PENDING**.

The numerical source is [final 600 s facts](../../artifacts/rotating_slots_stage2_20261023/v7_final_0600_facts.json), produced by the [read-only aggregation script](../../artifacts/rotating_slots_stage2_20261023/aggregate_v7_final.py). The script checked summary/audit counts, energy units and derived costs, actual touchdown events, frame counts and the 0–600 s frame ordering without rerunning simulation or audit.

| Scenario | Method | Normal | Emergency | Total | Energy failures | Airborne / upstream pending | Offered (initial + new) | Total kWh |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stress | EnergySlack | 24 | 4 | 28 | 0 | 47 / 0 | 75 (36 + 39) | 579.993689 |
| stress | CP | 25 | 5 | 30 | 0 | 45 / 0 | 75 (36 + 39) | 572.083134 |
| diagnostic | EnergySlack | 27 | 0 | 27 | 0 | 19 / 0 | 46 (18 + 28) | 262.148928 |
| diagnostic | CP | 27 | 2 | 29 | 0 | 17 / 0 | 46 (18 + 28) | 263.016853 |
| medium | EnergySlack | 27 | 2 | 29 | 0 | 28 / 0 | 57 (24 + 33) | 394.344077 |
| medium | CP | 22 | 4 | 26 | 0 | 31 / 0 | 57 (24 + 33) | 406.652004 |

Offered counts exclude the three initially grounded service fixtures. The six 600 s energy totals already contain all realised energy since t=0. For the stress and diagnostic continuations, the 0–240 s inheritance and new 240–600 s increments are:

| Scenario | Method | Inherited 0–240 s kWh | New 240–600 s kWh | Total 0–600 s kWh |
|---|---|---:|---:|---:|
| stress | EnergySlack | 213.144478 | 366.849211 | 579.993689 |
| stress | CP | 211.715379 | 360.367754 | 572.083134 |
| diagnostic | EnergySlack | 104.022094 | 158.126834 | 262.148928 |
| diagnostic | CP | 104.020530 | 158.996322 | 263.016853 |

The inherited energy must not be added to the reported total again, and cumulative summaries must not be summed over time. Medium has no inherited prefix: its full 394.344077 / 406.652004 kWh belongs to the new 0–600 s run. These quantities are modelled aircraft propulsion electricity, excluding unspent SOC energy and ground-service charging electricity. Every summary-minus-audit derived-cost difference, including joules and kWh, was zero. Stress and diagnostic remain **mixed v6-prefix/v7-suffix continuations**; only medium is v7 from zero.

At the same 600 s window, medium CP has five fewer normal and three fewer total landings, three more airborne pending aircraft, and 12.31 kWh more energy than EnergySlack. Stress CP has two more total landings and 7.91 kWh less energy. Diagnostic CP has two more emergency landings, equal normal landings and about 0.87 kWh more energy. These are scenario-specific observed outcomes, not a general method dominance claim.

The common 480 s comparison below uses each new continuation at 480 s against its own old v6 method at 480 s. It does not compare new 600 s outcomes with old 480 s outcomes. Failures are zero in all listed rows.

| Scenario | Method | Old v6 normal / emergency / total | New continuation normal / emergency / total | Old v6 kWh | New continuation kWh |
|---|---|---:|---:|---:|---:|
| stress | EnergySlack | 24 / 0 / 24 | 24 / 1 / 25 | 452.792327 | 451.604751 |
| stress | CP | 24 / 0 / 24 | 24 / 0 / 24 | 448.187139 | 447.969154 |
| diagnostic | EnergySlack | 25 / 0 / 25 | 25 / 0 / 25 | 210.192351 | 207.433088 |
| diagnostic | CP | 23 / 0 / 23 | 22 / 1 / 23 | 206.595101 | 208.150575 |

CP entered the solver in every v7-suffix call: diagnostic 227/227, medium 381/381 and stress 228/228. Strict improvement over its internal hint was recorded in 67, 296 and 141 calls respectively. This improvement concerns the finite candidate prediction objective and a bounded internal EnergySlack hint; it is not proof that CP beats a complete independent EnergySlack decision or the realised full episode.

The following timing table uses **only v7-suffix calls**: t≥240 s for stress/diagnostic and all t≥0 calls for medium. Inherited v6 callback timings are excluded. Values are wall seconds. Candidate and safety-validation columns use the same named stage groups as the recorded `rescue_observations`; stage hierarchies are not additive, and the solver column is the recorded solver time rather than total CP callback time.

| Scenario | Method | Suffix calls | Normal callback mean / P95 | Candidate mean | Safety-validation mean | Solver mean |
|---|---|---:|---:|---:|---:|---:|
| stress | EnergySlack | 235 | 17.582 / 20.389 | 10.978 | 6.291 | 0.000000 |
| stress | CP | 228 | 19.761 / 22.726 | 11.127 | 8.190 | 0.026106 |
| diagnostic | EnergySlack | 217 | 13.387 / 13.991 | 9.668 | 3.629 | 0.000000 |
| diagnostic | CP | 227 | 14.651 / 16.468 | 9.859 | 4.652 | 0.015275 |
| medium | EnergySlack | 380 | 15.133 / 16.120 | 10.287 | 4.690 | 0.000000 |
| medium | CP | 381 | 17.412 / 19.864 | 10.114 | 7.050 | 0.019429 |

Actual `FINAL_NORMAL` maneuver-start events strictly after the first emergency declaration numbered stress 3 / 8, diagnostic 7 / 11 and medium 7 / 4 for EnergySlack / CP. They are actual starts, not future plan reservations. The records and inspected coexistence frames did not reveal an unrelated emergency imposing a global stop on all ordinary finals. This evidence does not prove that every idle pad interval was justified. Pad-reason and protection statistics are samples at dispatches, not time-weighted utilization fractions; future reservations, current ground service and current approaches must remain distinct.

Rescue waiting below covers the recorded 0–600 s window. A completed wait ends at the **actual rescue flight start**, not touchdown. An unstarted aircraft's wait is right-censored at 600 s. `Unresolved` also includes rescues already airborne but not yet landed.

| Scenario | Method | Declared | Actual rescue starts | Started wait mean / P95 (s) | Not started at 600 | Censored age mean / P95 (s) | Unresolved |
|---|---|---:|---:|---:|---:|---:|---:|
| stress | EnergySlack | 22 | 7 | 71.933 / 91.379 | 15 | 73.650 / 131.756 | 18 |
| stress | CP | 22 | 8 | 81.405 / 95.005 | 14 | 71.803 / 138.725 | 17 |
| diagnostic | EnergySlack | 3 | 1 | 99.439 / 99.439 | 2 | 80.000 / 84.500 | 3 |
| diagnostic | CP | 6 | 4 | 73.364 / 96.168 | 2 | 29.411 / 54.530 | 4 |
| medium | EnergySlack | 11 | 5 | 73.314 / 98.478 | 6 | 61.650 / 96.464 | 9 |
| medium | CP | 16 | 6 | 87.358 / 108.426 | 10 | 62.464 / 140.556 | 12 |

In stress, UID11 (`initial_inner_08`) still held in the inner ring at t=600 s with SOC≈0.07414076 under both methods, with no actual rescue start. Its declarations were at 454.831082 s for EnergySlack and approximately 455 s for CP; its unstarted waits were about 145.169 / 145.000 s. The rescue-need classification correction therefore has not solved rescue efficiency or eliminated the remaining need. UID39 (`req_000000`) in stress EnergySlack was declared and actually started at 413.056242 s, then landed at 461.556242 s with SOC≈0.184819. In the CP continuation it had already landed normally at 161.323948 s in the inherited prefix. SOC labels show recorded values; they are not a numeric sorting of the display or a guarantee of landing order.

The preserved audit results are:

| Scenario | EnergySlack audit | CP audit | Original issue count |
|---|---|---|---:|
| stress | FAIL | PASS | 3 / 0 |
| diagnostic | PASS | PASS | 0 / 0 |
| medium | FAIL | FAIL | 2 / 2 |

All seven issues are `EVENT_FRAME_MISSING` for `rescue_warning`. Continuous-separation audit status is `CERTIFIED` in all six; counts, energy and derived costs agree with summaries. The raw physical-outcome field remains `IN_PROGRESS`, and human acceptance remains `HUMAN_PENDING`. Reaching the requested 600 s boundary does not mean all offered aircraft have landed, or that every audit has passed.

Intermediate boundary summaries and same-time replay frames use different executor phases. The campaign advances with `settle_only_at_end=True`, publishes the settled boundary summary/checkpoint, and only the subsequent block executes that timestamp's monitor/dispatch. The recorder then replaces the same-time replay frame. This explains the recorded +1 decision count and occasional additional monitor counters in intermediate replay frames relative to their published summaries. At the final 600 s boundary all compared fields agree. The raw differences remain explicitly listed in the final facts; the quality/energy curves use the published boundary summaries.

The [post-run event-frame fix facts](../../artifacts/rotating_slots_stage2_20261023/v7_event_frame_fix_facts.json) describe a separate local recorder patch applied **after** the formal experiment: five added lines record newly emitted monitor events when that iteration does not dispatch, before physical time advances. Targeted tests passed **5/5 in 3.88 s**, and related regressions passed **91/91 in 19.61 s**, both exit 0. The tested integration snapshots, SOC, energy, pose, active/soft plans, resources, dependencies, costs, events, callback inputs, dispatch times and final metrics were equal; only the missing frame was added. There was no formal rerun, original-output rewrite, audit-rule change, or reclassification of the original three FAIL audits. This patch remains distinct from the frozen experiment source. The primary reviewed the actual diff and tests and accepted this future recording repair; this does not constitute acceptance of the full formal experiment.

Replay checks are indexed by [v7 replay summary](../../artifacts/rotating_slots_stage2_20261023/v7_replay_checks/summary.json). The primary inspected stress EnergySlack at 333.5 s (one RESERVED rescue alongside four active normal finals), medium CP at 305 s (one WAIT rescue alongside four active normal finals), and stress EnergySlack shortly after UID39 touchdown at 461.566 s (P4 SERVICE UID39). The primary also inspected the medium `comparison_curves.png`: its lower CP total landings, higher airborne-pending count, higher energy curve and slower planning matched the recorded facts. HUD overlap limited these images to qualitative checks; these automated inspections do not replace human acceptance.

The compact visual references are the [contact sheet](../../artifacts/rotating_slots_stage2_20261023/v7_replay_checks/contactsheet.png), [stress coexistence frame](../../artifacts/rotating_slots_stage2_20261023/v7_replay_checks/stress_energy_slack_coexist_wait_reserved_normal.png), and [medium coexistence frame](../../artifacts/rotating_slots_stage2_20261023/v7_replay_checks/medium_rh_cp_sat_coexist_wait_reserved_normal.png). The other captured PNGs remain local. The six parameter combinations for [launch_replays.ps1](../../artifacts/rotating_slots_stage2_20261023/v7_replay_checks/launch_replays.ps1) are:

```powershell
Set-Location 'D:\7_codespace\terminal'
& .\artifacts\rotating_slots_stage2_20261023\v7_replay_checks\launch_replays.ps1 -Scene stress -Method energy_slack
& .\artifacts\rotating_slots_stage2_20261023\v7_replay_checks\launch_replays.ps1 -Scene stress -Method rh_cp_sat
& .\artifacts\rotating_slots_stage2_20261023\v7_replay_checks\launch_replays.ps1 -Scene diagnostic -Method energy_slack
& .\artifacts\rotating_slots_stage2_20261023\v7_replay_checks\launch_replays.ps1 -Scene diagnostic -Method rh_cp_sat
& .\artifacts\rotating_slots_stage2_20261023\v7_replay_checks\launch_replays.ps1 -Scene medium -Method energy_slack
& .\artifacts\rotating_slots_stage2_20261023\v7_replay_checks\launch_replays.ps1 -Scene medium -Method rh_cp_sat
```

Useful inspection times are 0, 240, 480 and 600 s for every bundle, plus coexistence times stress EnergySlack 333.5 s, stress CP 261.331082 s, diagnostic EnergySlack 369.060601 s, diagnostic CP 315 s, medium EnergySlack 338.394132 s and medium CP 305 s. For stress EnergySlack, inspect UID39 just after 413.056242 s and 461.556242 s, and UID11 at 600 s. The launcher accepts `-Scene` and `-Method`; the times are viewer inspection targets, not extra launcher parameters.

The terminal-state heartbeat was deleted after completion. No sleeping monitor worker remains. Numerical extraction and the stated limited visual checks are complete; original audit failures and human acceptance limits remain part of the delivery record.

## Frozen protocol, validation and historical startup

This revision registers `rescue_need_v7` and `medium24_rate150_v1` for the bounded Stage2 campaign. The medium fixture has 24 initial airborne aircraft, eight per ring, and seed-71 arrivals at 150/h. The same SOC distribution, service-task draws, geometry and `service_half_v1` configuration are shared by the two methods. The demand generator gains only the supported rate 150; its algorithm, seeds and existing 60/120/180 streams are unchanged.

The requested comparison is three scenarios × two methods, each through simulated t=600 s. Stress and diagnostic use an explicit branch from the complete t=240 s checkpoint of each v6 method; medium starts at t=0. Each pair receives a new 21600 s wall allocation with an 1800 s archive reserve. Parent wall-budget records remain historical evidence and are copied into branch metadata, rather than reset or charged to the new allocation.

| Scenario profile | Initial airborne (outer / middle / inner) | Arrival rate | v7 start | Final simulated time |
|---|---:|---:|---:|---:|
| `diagnostic18_rate120_v1` | 18 (6 / 6 / 6) | 120/h | 240 s, inherited v6 prefix | 600 s |
| `medium24_rate150_v1` | 24 (8 / 8 / 8) | 150/h | 0 s | 600 s |
| `stress36_rate180_v1` | 36 (14 / 12 / 10) | 180/h | 240 s, inherited v6 prefix | 600 s |

Every row uses both `energy_slack` and `rh_cp_sat`, seed 71, the shared frozen SOC/task inputs and `service_half_v1`. The four existing stress/diagnostic method runs retain their respective realised 0–240 s prefixes. The following protocol and startup details describe the frozen setup; terminal outcomes are reported above.

`branch --parent-output OLD --from-time 240 --output NEW --provenance NEW_FROZEN_JSON` is a separate operation from ordinary resume. It requires disjoint roots, new output, valid parent checkpoint schema and payload hash, matching complete prefix summaries, and locally frozen new provenance. Only the v6→v7 rescue-profile transition is supported. Configuration, manifest, service, numerical backend, candidate-policy identifiers, normal objective and planning budgets must match. Changed source paths are limited to the shared rescue implementation and its callers, campaign/CLI/viewer integration, and the rate-allowlist extension in `demand.py`. Source-path permission alone does not establish that a normal-planning algorithm change is allowed; the primary review verifies the actual diff before launch.

The original identity is passed to the unchanged strict checkpoint loader. Its `stream_copy` mechanism creates the new method state stream without opening the parent stream for writing. Frozen config, fixture, manifest and environment bytes are copied unchanged. The migrated checkpoint is saved under the new identity and v7 world profile. The physical clock, aircraft/SOC/energy, active and soft commitments, dependencies, admission/manifest cursor, realised states/events, scheduler records and existing compatible normal-reference caches are inherited. Only derived rescue monitor/search caches are discarded, so the next dispatch assesses need under v7. No extra integration step or synthetic boundary dispatch is performed by migration.

Copied summaries through the branch boundary are labelled `INHERITED_V6_PREFIX`. Subsequent summaries are labelled `V7_SUFFIX_WITH_INHERITED_V6_PREFIX`; replay provenance contains the branch record. These runs are **mixed v6-prefix/v7-suffix continuations**, not v7-from-zero results or proof of equivalence to such a run. Their t=240 existing commitments remain valid commitments.

`metrics.energy_j` and `metrics.energy_kwh` retain cumulative modelled aircraft propulsion electricity from t=0. `inherited_energy_j` records the branch baseline, while `suffix_energy_j` and `suffix_energy_kwh` are current total minus that baseline. Cumulative summaries must never be summed across time. This quantity does not count unspent energy represented by carried SOC or pad ground-service charging electricity.

The observed v6 cases motivating the rescue classification change were: stress `initial_inner_08` (UID11), declared at 380 s with a certified future rescue wait of roughly 100.008 s, and at SOC≈0.285886 at 424 s after 44 s had actually elapsed since declaration; and `req_000000` (UID39), whose necessary normal-reserve lower bound was classified INFEASIBLE at 415 s, while declaration was delayed until 476.25 s when a rescue opportunity was found. This reserve classification is distinct from a finite candidate search proving global normal-path infeasibility. A finite planning horizon supplies bounded candidate evidence. CP caps construction of its internal EnergySlack hint at three seconds; that hint is not an independent, complete EnergySlack baseline decision. Strict improvement over this internal hint therefore does not establish realised full-episode dominance over the independent EnergySlack method. Normal CP objective, solver and candidate ordering remain unchanged by this protocol.

The separately verified [v6 CP diagnostic facts](../../artifacts/rotating_slots_stage2_20261023/v7_cp_diagnostic_facts.json) cover the diagnostic run through 480 s. `candidate_and_commit_counts.decision_aggregates` records 287 normal calls, all 287 with `solver_entered`, with 116 calls reporting strict improvement over the internal hint and 171 reporting no strict improvement. The finite prediction horizon is H=240 s. Its lexicographically minimised objective is `(F, -L, U, tau, J)`: predicted energy/peak failure, negative within-horizon landing count, urgency/unknown-tail rank, predicted completion-time proxy, and predicted energy. These are finite candidate prediction quantities rather than final episode outcomes.

At t=0, `observed_calls.first_strict_improvement_call` records `hint_objective = [0, -6, 36, 11880, 291068276.0708492]` and matching `selected_objective` / `final_objective = [0, -6, 36, 11700, 287459120.6988814]`, with `commit_status = ACCEPTED`. The same evidence file's `final_normal_landing_set_comparison` records 25 normal landings for the independent EnergySlack run and 23 for CP at 480 s. The primary review of the recording/execution chain found no evidence that the solver result was ignored or that the recorded objective was inconsistent with the executed selection. These observations establish improvement of the finite candidate prediction objective in the recorded calls; they do not establish realised full-episode dominance. In particular, the three-second internal hint budget does not reproduce a complete independent EnergySlack decision.

This round therefore leaves the CP objective, solving procedure and candidate ordering unchanged. The shared rescue-need semantics are corrected for both methods together. The stress and diagnostic continuations switch this shared semantics at the preserved 240 s boundary; their subsequent comparison remains a mixed v6-prefix/v7-suffix comparison, while the medium scenario is the new v7-from-zero case.

Server validation is recorded separately in [v7 validation facts](../../artifacts/rotating_slots_stage2_20261023/v7_validation_facts.json), summarised from the synced `stage2_rescue_need_v7_validation_20261023/runtime_clones/facts.json` and validation logs. Server preflight reported matching hashes for all 20 validation-manifest entries and the UI/helper files; the 20 local entries and runtime helper hash were also checked against the reported fingerprints. The six specified server test files passed: **106 passed in 7.66 s**, process exit code 0. The runtime clone checks exited 0 after 144 s (04:04:57–04:07:21 UTC on 2026-10-23). The before/after hashes of nine parent checkpoint/state files matched.

All four t=240 s profile-migration roundtrips retained physical state, cumulative energy, active/soft commitments, emergency metadata and scheduler state. In the separate stress t=420 s clone check, UID39 (`req_000000`) was immediately declared `EMERGENCY_WAITING` at 420 s with reason `NORMAL_RESERVE_IMPOSSIBLE`. It was not charged and had no emergency start time; resources and energy were unchanged by the declaration. UID11's existing reserved rescue remained unchanged. These are clone checks of the policy transition, not formal campaign results.

The fresh medium checks each executed one real callback at t=0:

| Method | Callback wall time | Status | Commit status | Submitted actions |
|---|---:|---|---|---:|
| `energy_slack` | 20.222935665980913 s | `FEASIBLE` | `ACCEPTED` | 21 |
| `rh_cp_sat` | 33.4349310650141 s | `LEX_LIMITED` | `ACCEPTED` | 19 |

The medium CP validation callback entered the solver and retained a certified hint; its final objective equalled the hint objective, so this callback showed no strict improvement. Both checks kept the frozen 30 s soft budget, 8 s solver budget and 64 maximum refinement rounds. The soft budget is cooperative rather than a hard preemptive cutoff: the observed CP callback took 33.4349310650141 s, exceeding 30 s without a budget increase. These were pre-run one-callback measurements, distinct from the completed 600 s results above.

The three formal pairs were subsequently launched and observed advancing. The single startup record is preserved in [v7 launch facts](../../artifacts/rotating_slots_stage2_20261023/v7_launch_facts.json), based on the synced validation root's `launch_facts.json` and `startup_snapshot.json`. Frozen code commit `4da4f3852290f73b6369e67589d3f8d54ad0b80e` was pushed; its source hash is `848437c22063d408e55267d03fcb444febf0e15191715b2c11c21342de1da0c7`. Subsequent documentation-only delivery does not change that frozen source. Stress and diagnostic retain their v6 0–240 s prefixes; medium starts at t=0 under v7.

The historical startup snapshot had local anchor `2026-10-23T04:22:57.0794840Z` and server timestamp `2026-10-23T04:23:37Z`. Worker PIDs below were listed in CPU order; its latest state times were recorded simulation times rather than completed comparison boundaries.

| Pair | Launch UTC, 2026-10-23 | Wrapper / runner PID | Method worker PID @ CPU | Latest EnergySlack / CP state time |
|---|---|---|---|---|
| stress | 04:17:14Z | 163894 / 163897 | 163906 @ 0; 163907 @ 2 | 273 / 255.11254882812494 s |
| diagnostic | 04:17:21Z | 163955 / 163958 | 163967 @ 4; 163968 @ 6 | 281 / 273.49999999999994 s |
| medium | 04:17:28Z | 164016 / 164019 | 164028 @ 8; 164029 @ 10 | 29 / 20.166015625 s |

All six method workers had one thread. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` were each set to `1`. The actual runner commands and output roots were:

```bash
/home/user/envs/terminal_stage2_v1/bin/python -u /home/user/projects/terminal/scripts/server/run_rotating_slots_stage2.py run --output /home/user/training_outputs/terminal/stage2_rescue_need_v7_stress_600s_20261023 --target 600 --cpus 0 2
/home/user/envs/terminal_stage2_v1/bin/python -u /home/user/projects/terminal/scripts/server/run_rotating_slots_stage2.py run --output /home/user/training_outputs/terminal/stage2_rescue_need_v7_diagnostic_600s_20261023 --target 600 --cpus 4 6
/home/user/envs/terminal_stage2_v1/bin/python -u /home/user/projects/terminal/scripts/server/run_rotating_slots_stage2.py run --output /home/user/training_outputs/terminal/stage2_rescue_need_v7_medium_600s_20261023 --target 600 --cpus 8 10
```

The startup snapshot contained no method `error.json` or `invalid.json`. At that historical observation, synced campaign metadata was `RUNNING` with human review `HUMAN_PENDING`; neither method in any pair yet had a `checkpoint_0600.json` or `summary_0600.json` marker. These startup observations were later superseded by the complete 600 s artifacts and exit codes reported above; they did not establish visual acceptance.

The actual frozen arrival inputs were checked separately from the validation checksum manifest:

| Pair | Arrival requests | Arrival `manifest.json` SHA256 |
|---|---:|---|
| stress | 172 | `10b97225bde7992af59c79857b0c387bde69ec3b1153e36867a9d42406ad4bc6` |
| diagnostic | 118 | `b827cae3e8d961a3e94102378c6a9c552bf20f1b12fd30d0445b5fce8cb52743` |
| medium | 146 | `d972c45754727f3ac7266e934b04293cdcf62c637dd98311b34a294b7cd6f7b5` |

The counts 172 / 118 / 146 cover each frozen 3600 s arrival window in full, not the requests actually offered by 600 s. The terminal report must count offered arrivals from the realised events.

All three environment-lock files had SHA256 `b15a7e9c1debc9e91cffa8029e48706f7fd917382d7be05b873f64a61348eba2`. The startup snapshot's `provenance.manifest_sha256` instead identifies `validation_manifest.json` (`c7825adf095d26365fbffff86629658f1677b7c163cfb07691150fd90337d0e7`); it is not an arrival-manifest hash. The three formal provenance file hashes are retained in the launch facts.

Each pair retains a 21600 s total allocation with a cumulative compute-stop threshold at 19800 s, including prior credit and preparation elapsed, and an 1800 s archive reserve. Its 600 s credited prior allocation is a conservative allowance, not a measured 600 s preparation duration. Existing parent budgets remain recorded history and are not modified. The primary initial total-time estimates are 12600 s for stress, 9000 s for diagnostic and 12600 s for medium, informed by the v6 480 s pair wall times of 9153 / 4903 s and the new medium callbacks of 20.22 / 33.43 s. Uneven later density and phase costs limit these estimates.

The historical direct runner-elapsed observation at server 04:21:24Z was 250 / 242 / 235 s for stress / diagnostic / medium. Adding the 133 s server-clock difference gave **derived**, not directly measured, startup elapsed estimates of 383 / 375 / 368 s. The largest estimated remainder was 12232 s at the local startup anchor. The primary then created the sole thread heartbeat `stage2-v7-600` with a 201-minute interval after deducting approximately 172.9 s already elapsed at local 04:25:50Z. The resulting approximately 07:46:50Z recheck was an estimate, not a scheduler-returned next-run field. The initial creation request omitted `destination` and was rejected without creating a schedule; the corrected thread request created one schedule. The monitoring worker exited without a sleeping worker, and the heartbeat was deleted after terminal-state verification.

The original authorised follow-up was terminal-state verification, 600 s and common-480 s comparisons, cumulative/suffix energy accounting, rescue/CP interpretation and replay checks. Those findings are now recorded above. Human acceptance remains `HUMAN_PENDING`, and the original audit FAIL records remain unchanged.
