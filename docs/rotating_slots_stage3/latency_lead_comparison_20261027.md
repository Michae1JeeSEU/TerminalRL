# 20/30 s lead comparison: protocol and completed results — 2026-10-27

The user requested a short comparison of normal-planning lead times using the current `affected_traffic_revalidation_v1` interface. The completed 30 s H/L pair was reused read-only, and exactly one new 20 s H/L pair has completed. No new 30 s pair, seed scan, fit or algorithm change was included. H uses heuristic candidate ordering plus CP-SAT; L uses the same backend with the existing frozen learned ordering. The current version failed to deliver any joint group at 20 s in this trial. This does not establish theoretical infeasibility of a 20 s lead; it does not support deploying the current 20 s configuration.

## Preregistered protocol

| Lead | Evidence | Arm order | Output |
| --- | --- | --- | --- |
| 30 s | Preexisting completed H/L pair, read-only | Original H then L | `/home/user/training_outputs/terminal/stage3_latency_revalidation_20261027` |
| 20 s | Exactly two new formal jobs | H then L, serial | `/home/user/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20` |

Both settings use root `s3r23`, seed 7324, original `medium24_rate150_v1` initial state and manifest, 120 simulated seconds, CPU 12, a 5 s heartbeat and 1 s preparation margin. The absolute ready deadline is snapshot + 29 s for lead 30 and snapshot + 19 s for lead 20. The original candidate-active 6 s, cumulative CP 8 s, hint 3 s, soft 30 s, callback watchdog 60 s, 32 attempts per aircraft and 1024 total-attempt limits remain unchanged. The existing minimum remaining-time logic may tighten them against the absolute deadline; it cannot reset or increase them. The endpoint still censors dispatch/delivery exactly at 120 s.

The new 20 s pair has its own fresh ledger: 900 s watchdog per job and 1800 s evaluation cap, without reset or retry. All five numerical-library thread counts are 1; profiler and rendering are off. Original world configuration, physics, energy, admission, emergency and authoritative safety checks are unchanged. Execution changes are limited to four runner/report/test files; the core was checked against `d84f3a3` with no core diff. The new execution source is commit `422ed74037e14927e44938eedfed95344dd0f93d`, source hash `c7ae6f6f48816d0294b1e9d95d05531ec551391508d115c615f40793c6fa7662`, recorded separately from the unchanged training fingerprint.

Model SHA256 is `dba25b3393a5580bdc3971d9adf7d6e3cff6360404fdb08bc8fbb6f0b36e3b48`; normalizer SHA256 is `1e06b066dad2d657acbe16de6706b180dd4f4ed8ba1bc106cefcc2a3d4f4e7be`. Their original binding/fingerprint is retained. The 30 s comparison source is commit `d84f3a372b04221b5e4c7845a403add906fb59b0`, source hash `e481913216dd4b75f8111bd25a38e2f391ba5f988cfcd3b05b744d2831c5c9d8`. Its inputs, results and ledger are not copied over or rewritten; comparison reporting links the existing [30 s report](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/FINAL_REPORT.md).

## Runner interface and local checks

Only `prepare` accepts `--lead-seconds {20,30}`; omission preserves the existing 30 s default. Preparation stores the selected value in `execution_scope.json`. Verification requires the selected allowed value, every other fixed protocol field, the scope digest and the original preparation receipt. `run`, `job` and `summarize` reject a CLI lead override, and the actual wrapper reads lead/margin from the verified scope. Existing 30 s roots retain their original protocol schema. The intended new preparation specifies `--lead-seconds 20` and the fresh `lead_20` output explicitly; no new `lead_30` preparation is scheduled.

Compilation, CLI help and saved-output campaign fixtures passed locally: **13 tests in 1.09 s**, RC 0. Fixtures serialize and read back both leads, observe the wrapper receiving the scope value, reject invalid values and protocol changes, reject non-prepare overrides, and retain existing output/timeout/aggregation checks. They do not start new physics scenes, model inference or solves. Receipts: `artifacts/latency_lead_campaign_tests_20261027_a.log` and `.xml`. Final targeted verification comprised 19 passing tests: 13 campaign and 6 report fixtures.

## Interpretation boundary and status

Lead is the only intended algorithmic setting changed, but the reused 30 s pair and fresh 20 s pair executed at different times and under potentially different server load. Small wall-time differences are therefore not a strict simultaneous single-factor timing result. One development scenario and one pair per setting support no significance or generalization claim. Frozen-model future-view distribution and the normal-only/emergency-publication abstractions remain applicable.

## Actual four-arm comparison

The 20 s jobs both returned RC 0, reached the common 120 s boundary and independent terminal, with `failed=false` and `terminated_invalid=false`. The new ledger is paused at 271.306401 s evaluation usage. The original 30 s files and ledger were not rerun, copied over or rewritten.

| Saved metric | 30 H | 20 H | 30 L | 20 L |
| --- | ---: | ---: | ---: | ---: |
| Joint join times (s) | 30, 65 | none | 30, 115 | none |
| Actual joined groups / plans | 2 / 13 | 0 / 0 | 2 / 17 | 0 / 0 |
| Committed / finished nonempty tasks | 2 / 3 | 0 / 5 | 2 / 3 | 0 / 5 |
| Endpoint pending / no-prefix tasks | 1 / 0 | 1 / 0 | 0 / 1 | 1 / 0 |
| Ready-deadline rejections | 0 | 5 | 0 | 5 |
| Traffic changed after deadline | 1 | 0 | 1 | 0 |
| Revalidation rounds | 1 | 0 | 2 | 0 |
| Normal / emergency landings / energy failures | 6 / 0 / 0 | 0 / 0 / 0 | 6 / 0 / 0 | 0 / 0 / 0 |
| Airborne pending / admitted | 23 / 29 | 29 / 29 | 23 / 29 | 29 / 29 |
| Cumulative energy (J) | 219238588.34146103 | 236856587.51957327 | 220924953.457677 | 236856587.51957327 |
| Holding (aircraft-s) | 2079.7414594252973 | 2919.7414594252973 | 2079.7414594252973 | 2919.7414594252973 |
| All-task observed full planning work (s) | 86.78324015508406 | 116.41714468807913 | 65.33274777326733 | 113.1393728578696 |
| Full dispatch work (s) | 89.74022091785446 | 120.81146473286208 | 68.28851268999279 | 117.54132244817447 |
| Finished complete planning P95 (s; count) | 25.354964612051845; 3 | 20.587531220982783; 5 | 18.36061074270401; 4 | 19.515989011037163; 5 |

At 20 s, energy increased by 8.0360% for H and 7.2113% for L relative to their respective 30 s runs, while holding increased by 40.3896% for both. Observed planning work increased by 34.1470% / 73.1741% (H/L), and dispatch work by 34.6235% / 72.1246%. All-task work includes already executed computation in pending tasks; pending is not a timeout or finished attempt. Complete-planning P95 uses only finished tasks, not callback-only duration. These small sample counts do not provide stable tail estimates or an algorithmic latency bound.

## Observed deadline bottleneck

Each 20 s arm has six tasks: five `READY_DEADLINE_MISSED` rejections and one `COMPUTING` pending at the endpoint. No-prefix, new-traffic/stale, packet-expiry and revalidation counts are all zero. The absence of dependency rejection is not improved dependency robustness: none of the proposals reached final preparation or revalidation (`prepare_work_s` is null).

All twelve scheduler callbacks record `hint_certified=true`, `deferred=true`, `noop=false`, with `ready_budget.remaining_at_return_s=0`. The saved evidence therefore does not indicate a lack of usable candidates. Search-stage certification is not final publication authorization; final preparation and live dependency checks remain required. The primary's reviewed interpretation is that search/incumbent improvement consumed the available outer window; a nonpreemptible validation call can return beyond it, and priority-service/delivery queues further delay availability. For example, L snapshot 84.83361673663154 s used 18.33181651297491 s normal work (less than 19 s), but finished at 105.17389779857609 s, after its original ready deadline 103.83361673663154 s. Normal work duration alone cannot establish timely delivery.

| 20 s measured phase/search evidence | H | L |
| --- | ---: | ---: |
| CP wall (s) | 24.56814436 | 21.77955758 |
| Prefix validation (s) | 27.02879 | 28.98556 |
| Trajectory validation (s) | 26.63062 | 24.47562 |
| Learned ranks / cached ranks | 0 / 0 | 256 / 0 |

CP allocations were tightened by the remaining deadline; no budget was increased. L feature/inference work was 1.86323347 + 8.12753452 = 9.99076799 s; its cache misses were 250 candidate changes and six initial batches. Shared backend work and the same candidate-active clock remain part of both arms' execution.

The first L callback from the same initial world illustrates why the old duration cannot simply be moved to the earlier join. At lead 30 it took 17.103835 s, with prefix/trajectory validation 1.138370/1.652375 s, CP 8.000662 s and 86 columns. At lead 20 it took 18.993948 s, with prefix/trajectory validation 3.997278/4.090639 s, CP already tightened to 4.510004 s and 80 columns. The corresponding 20 s normal work including projection was about 19.00309 s against deadline 19 s, before preparation began. Phase CPU was approximately wall time, so simple extensive preemption does not explain these observations. A different projection time changes candidate/validation work, but these differently timed runs do not isolate one cause of the validation-duration difference.

The old 30 s finished P95 values cannot predict the 20 s rollout: projected join states differ, failed joins change subsequent closed-loop states, task counts grow from four to six, and phase workloads change. Different execution dates/server load additionally limit strict wall-time attribution. Directly selecting 20 s failed for this version and scenario; it neither establishes theoretical infeasibility of 20 s nor that learning has reached a ceiling. The primary identified a possible next engineering change: stop incumbent improvement before the outer ready deadline and leave measured time for final preparation, guards, delivery and necessary priority service while returning a certified incumbent. This is a deferred proposal, not an implemented repair; no budget increase, heartbeat expansion, retraining or corrective rerun was performed here.

## Saved replay and review status

The 20 s replays each passed their 15-file hash/trajectory checks and saved replay validation. PVA checks covered H 3494 / L 3468 samples, all within tolerance. Both have zero actual normal starts and zero unattributed normal starts. With no new joint join, predecessor C2 evidence is `UNAVAILABLE_PREDECESSOR`, with zero comparisons; it must not be described as C2 passed. These are negative execution results, not failed/invalid world terminals.

The primary reviewed both actual timeline/contact PNGs: the timeline shows five rejections and one pending task per arm with zero landings; saved contact frames continue along the old approved trajectories. Visual readability/fidelity was accepted; static frames do not establish complete human replay acceptance, and `HUMAN_PENDING` remains.

The [20 s report](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/FINAL_REPORT.md), [analysis JSON](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/analysis_summary.json), [timeline CSV](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/timeline.csv), [boundary quality CSV](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/boundary_quality.csv), [timeline PNG](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/planning_timeline.png) / [SVG](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/planning_timeline.svg), [saved contact PNG](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/saved_xy_contact.png), and [H replay](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/evaluation/heuristic_reduced/replay/replay.json) / [L replay](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/evaluation/learned_reduced/replay/replay.json) retain the new evidence separately. The linked original 30 s report remains read-only.
