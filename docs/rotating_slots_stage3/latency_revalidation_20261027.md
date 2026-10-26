# Measured affected-traffic revalidation — 2026-10-27

This iteration implements `affected_traffic_revalidation_v1` for the existing normal-planning latency interface. `planning_effective_state_v1`, the 30 s lead, 1 s preparation margin, 5 s heartbeat, approved plan timestamps, model, physics, energy, safety precision, emergency rules and scheduling budgets are unchanged. The preregistered single 120 s H/L comparison is complete; no training was performed.

## Certification boundary

The original immutable proposed group, energy starts and resource delta remain the publication packet. Its aircraft state/approved commitments, emergency conditions, touched slots/pads and relevant release-support closure are strict dependencies. A changed direct dependency rejects the whole group. Unrelated external dependency-graph entries are not direct dependencies.

New or changed external traffic suspends readiness and starts measured separation revalidation. It projects only currently revealed approved commitments; it does not reveal future requests, advance a cloned world, generate trajectories or invoke a solver. Each required pair involving the proposed group uses the same traffic representation, interval construction and continuous `validate_separation` predicate as the original authoritative preparation. Only `SAFE` accepted results certify a pair; unresolved/unknown results reject. A removed actor is omitted only when its observed state proves it non-live.

Pair certificates bind both immutable traffic signatures and their certified time interval. Reuse requires exact signatures/start and an interval contained in the existing certificate. A larger global horizon causes the necessary extended pairs to be checked again and increments `horizon_extension_pairs`; this implementation does not use a full-preparation fallback (`full_fallback=false`). No geometry or energy integration is added to effective-time publication. Original whole-group atomic publication and its cheap dependency guard remain in force.

## Delivery and CPU accounting

Revalidation startup revokes the world readiness receipt and clears the task's active `ready_time`; a prior receipt is retained only as `previous_ready_time` in round diagnostics. Ordinary guard inspection is read-only. A candidate baseline is promoted only after its measured work and completion guard have been delivered and the guard confirms no further change. A change during work yields `SUPERSEDED` and another bounded round, rather than inheriting the previous result.

The original ready deadline bounds actual computation. Before projection, after projection and between pair checks, the code checks an actual `perf_counter` deadline after subtracting queued priority/guard service. An individual continuous geometry call is nonpreemptible: if it overruns, it returns before the loop stops, and its measured work remains occupied until its real virtual completion. Expiry never caches an unknown result as a pass. Ordinary triggers coalesce during that occupied interval; crossing the deadline does not cancel computation for free.

The optional future-port `planning_wall_deadline` uses the actual performance-clock origin before projection, independent of the injected test-duration clock. It tightens an existing callback budget without adding a preparation reserve or increasing any budget. Monitor/emergency/admission service, prepared dependency inspection and carried handoff delay occupy the same virtual CPU. Successful readiness registration covers promotion guard, registration and handoff; CPU availability is at least the issued `ready_at`. Effective-time atomic publication remains event bookkeeping charged to subsequent online service, with the existing emergency timing mode `existing_priority_response_cpu_charged`.

Cost components are separate: `prepare_work_s` includes initial preparation and its initial ready guard; `revalidation_work_s`/`revalidation_work_cpu_s` include pair work and delivered completion-guard work, never duplicated inside preparation. `dependency_guard_work_s`/CPU cover prepared inspection. `ready_registration_bookkeeping_s`/CPU accumulate across all registration attempts, including baseline promotion guard, while publication and evidence retain their existing separate components. Priority service is recorded separately from normal work and is not counted again as revalidation computation.

## Saved lifecycle fields

Each task records `dependency_protocol`, cumulative `revalidation_count`, `revalidation_work_s`, `revalidation_work_cpu_s`, and `revalidations`. Each round records simulation `started_time`/`due`, affected aircraft IDs, checked/reused pair counts, `full_fallback`, horizon extensions, measured wall/CPU, reason and status. `geometry_completed_time` marks geometry delivery; `guard_delivery_due` marks completion-guard delivery. `completed_time` is absent while the round is pending and is recorded only when that delivery is observed. Round states include `COMPUTING`, `GUARDING`, `CERTIFIED`, `SUPERSEDED`, `REJECTED`, and `LATE`.

Exact expiry reasons are `READY_DEADLINE_MISSED` for an actual result/ready completion beyond the deadline, `TRAFFIC_CHANGED_AFTER_DEADLINE` for fresh uncertified traffic first observed after it, and `REVALIDATION_WORK_DEADLINE` for an exhausted computation allowance (status `UNKNOWN`, not a certificate). `TRAFFIC_REVALIDATION_REQUIRED` requests bounded work; `REVALIDATION_NOT_COMPLETE` is a waiting receipt. Existing direct-dependency and validator rejection reasons remain explicit. A timely prepared packet censored at the rollout endpoint is pending, not a completed join or a deadline rejection.

## Local verification

`tests/test_rotating_slots_planning_latency.py` plus `tests/test_rotating_slots_latency_revalidation.py`: **16 passed in 18.15 s**, RC 0. Compilation of the two implementation files and the new test file also returned 0. The test log and JUnit receipt are `artifacts/latency_revalidation_tests_20261027_e.log` and `.xml`.

The real small-world checks cover an unrelated revealed arrival followed by original-time whole-group execution, unknown separation rejection, emergency pad change rejection, late work remaining occupied until its true return, arrivals after the deadline, changes during revalidation, horizon extension and exact certificate reuse. Two successive approved transfers execute at their unchanged absolute starts (30 s and 80 s), preserve p/v/a continuity and decreasing SOC, with no resource reservation for the proposed action before publication. A pending revalidation checkpoint restores and completes with the original approved hash. The bounded zero-work path starts no projection and stores no certificate. The final clock regression checks promotion-guard time at the ready deadline and preservation of carried CPU handoff after successful registration.

These checks are interface evidence, not evidence of learned-ranking quality, formal closed-loop benefit or statistical generalization. The frozen prior experiment and its saved results are unchanged.

## Shared CP and ranking changes

Both use the same implementation; each callback owns an independent `EvaluationContext`. Reuse is confined to that context and exact catalog/resource inputs, objective and authoritative evaluator identity. Certified incumbents, proved cuts and single-column certificates can be reused only under their bound inputs; `UNKNOWN` produces no cut. Solver-internal state is not retained: a previous solution supplies a warm hint only. The cumulative limits remain 8 s CP, 3 s hint, 6 s candidate-active work and 30 s soft work. The true absolute ready deadline can shrink the remaining allowance; it never resets those limits.

L uses one score batch for up to three consumed work items per ranked group as an approximation. Telemetry retains the actual query budget fraction at scoring and its drift at each pop. Reuse binds actual input dependencies and one immutable content signature per order: unrelated retained coverage does not invalidate an order, while a change to a relevant input does. The existing frozen model is not retrained for future-view states. Shared backend or grouping improvements are not exclusive evidence for learned-ranking quality.

## Preregistered one-pair protocol and status

The formal comparison is fixed before outcome inspection: original medium root `s3r23`, seed 7324, H then L serially on CPU 12, each to 120 simulated seconds. It uses the same initial state and model/normalizer hashes as the completed latency-interface pair: model `dba25b3393a5580bdc3971d9adf7d6e3cff6360404fdb08bc8fbb6f0b36e3b48`, normalizer `1e06b066dad2d657acbe16de6706b180dd4f4ed8ba1bc106cefcc2a3d4f4e7be`. Lead is 30 s, preparation margin 1 s and heartbeat 5 s; each job has a 900 s watchdog and the evaluation total is 1800 s. All five numerical-library thread counts are 1; profiler and rendering are off. Events exactly at endpoint 120 remain censored under the existing endpoint rule.

New outputs belong only to `stage3_latency_revalidation_20261027`; prior raw inputs/results are read-only. There is no 20 s sweep, extra seed or additional fit. The research question is whether the saved run shows multiple actual joint joins, and what full computation costs and executed quality accompany them. A single pair is not generalization or isolated causal evidence.

The implementation checks above are 16 local passes. Separately, the primary recorded Phase 2 local verification as 41 passes with 7 OR-Tools checks deselected, plus a prior 96-test regression run; these overlapping suites are not summed. Reporter checks include a 10-test pass and two later fixture updates, also not summed.

Server verification subsequently completed with **126 passed in 12.60 s**, zero skipped, failed or errored cases, including 17 CP cases; the primary checked the actual JUnit receipt. An earlier malformed `taskset -c12` invocation failed before Python started and ran zero tests. The corrected `taskset -c 12` invocation passed once; the shell-wrapper error is not an experimental outcome.

The implementation was committed and pushed as `d84f3a372b04221b5e4c7845a403add906fb59b0`. The final 101-file freeze has source hash `e481913216dd4b75f8111bd25a38e2f391ba5f988cfcd3b05b744d2831c5c9d8` and matches **101/101** recorded file hashes. Earlier raw-byte differences in 95 local frozen files were verified to be CRLF/LF-only and equal after line-ending normalization; the frozen files were then materialized as the tested main-checkout bytes. This did not change algorithm semantics or prior results.

The sole preregistered pair completed at 17:11:52.300867 UTC, both process return codes 0, both independent terminals at 120 s. The ledger is paused with 182.860477 s evaluation usage. No retry, additional pair, sweep or fit was run.

## Actual 120 s results

H is heuristic candidate ordering with CP-SAT; L uses the frozen learned ordering with the same backend. Both ran on CPU 12 from the same initial state, manifest and model binding under the implementation commit/source hash above.

| Observed metric | H | L |
| --- | ---: | ---: |
| Normal / emergency landings / energy failures | 6 / 0 / 0 | 6 / 0 / 0 |
| Airborne pending / admitted | 23 / 29 | 23 / 29 |
| Holding (aircraft-s) | 2079.7414594252973 | 2079.7414594252973 |
| Cumulative energy (J) | 219238588.34146103 | 220924953.457677 |
| All-task observed full planning work (s) | 86.78324015508406 | 65.33274777326733 |
| Finished complete planning P95 (s; task count) | 25.354964612051845; 3 | 18.36061074270401; 4 |
| Full dispatch work (s) | 89.74022091785446 | 68.28851268999279 |
| Actual committed groups / plans | 2 / 13 | 2 / 17 |
| Finished nonempty tasks / committed | 3 / 2 | 3 / 2 |
| Endpoint pending tasks / no-prefix tasks | 1 / 0 | 0 / 1 |

L used 24.7173% less observed planning work and 23.9042% less dispatch work, with unchanged landings/holding and 0.76919% more energy. All-task work includes already executed computation in H's pending task; finished-task P95 uses completed tasks only. These three/four-sample P95 values are neither stable tail estimates nor algorithmic upper bounds. Job/receipt wall time also includes checkpoint, replay and process overhead and is not pure planning time.

H committed 11 plans together at 30 s and 2 at 65 s. L committed 13 at 30 s and 4 at 115 s. Both join fractions are 2/3 of finished nonempty attempts; H's pending task and L's no-prefix task are excluded. Their actual task compositions differ, so equal fractions do not establish a sustained join-rate improvement. All four tasks per arm exceeded the old 5 s observed-work target; the heartbeat remains 5 s, not a guarantee of a new completed plan every 5 s.

## Revalidation, computation and limits

The new-traffic path was exercised successfully: H certified one round (0.015843 s, 2 checked pairs, 45 reused) before the 65 s join. L certified two rounds in the same group (0.020762 and 0.055825 s; 0.076588 s total, 8 checked pairs, 192 reused) before the 115 s join. There were no `NEW_TRAFFIC`, ready-deadline or compute-exhaustion rejections. Each arm had one `TRAFFIC_CHANGED_AFTER_DEADLINE`: H observed the change at 99.22268 s after deadline 99 s; L at 84.83362 s after deadline 84 s. These are dependency changes after the original preparation opportunity, not solver/computation timeouts; no free late revalidation was performed.

| Shared computation / search evidence | H | L |
| --- | ---: | ---: |
| CP wall (s) | 32.0026846 | 32.0046714 |
| CP calls reusing context state | 4 | 4 |
| Reused pair cuts / single results | 122 / 367 | 33 / 277 |
| Prefix validation (s) | 11.6104 | 2.6421 |
| Trajectory validation (s) | 14.8953 | 4.4218 |
| Physical attempts | 315 | 144 |
| Catalog rounds / pruned columns | 12 / 0 | 8 / 0 |
| Learned ranks / cached learned ranks | 0 / 0 | 361 / 10 |

Reuse has execution evidence but did not reduce CP wall: each arm still used four callbacks with about 8 s CP each. L feature construction plus inference cost 1.7925 + 7.9855 = approximately 9.778 s and consumed the same 6 s candidate-active clock per callback. Its group-cache misses were 348 candidate changes and 13 first batches; reuse worked, but observed hits remained low. Fewer attempts, smaller catalogs and reduced validation work do not alone prove better candidate identification, because inference can displace candidate generation; L also produced one no-prefix result. Shared backend/group/cache improvements are not exclusive learned-ranking gains.

The prior interface pair produced one 13-plan joint group per arm; this iteration produced two groups per arm with different executed trajectories. This establishes that the previously global new-traffic rejection path can be traversed, but the comparison with the prior implementation is not a single-factor ablation. In this pair, lower computation did not improve normal landings, and L energy was slightly higher. The unchanged frozen model is being used on future-view states; one development scenario and one pair support no generalization or isolated causal claim. A 20/30 s lead comparison is deferred to a separately specified next engineering trial; no such extension or fit was performed here.

## Saved replay and review evidence

The primary reviewed the saved results, terminals, task records, receipts and derived report. Both replays passed their 15-file SHA/index checks and `ReplayBundle.validate`. Saved PVA checks covered H 3578 and L 3635 samples with maximum position error below 2.87e-13; maximum velocity/acceleration errors were about 5.53e-14 / 6.98e-15. All actual normal starts correspond to approved acknowledgements at the exact approved start times; both groups per arm start atomically, with no early or unattributed normal starts. All 13 H and 17 L unique-predecessor C2 comparisons were available and checked. Different cross-arm actual trajectory-hash sets describe different execution outcomes, not a control-input identity failure. Proposed hashes for rejected/pending proposals were not saved, so independent per-proposal nonexecution remains unavailable; actual-start coverage is a separate completed check.

The primary viewed the actual timeline/contact PNGs and accepted their readability and fidelity. Static saved-frame inspection does not replace a complete human replay review: `HUMAN_PENDING` remains. The normal-only atomic-publication abstraction and existing priority-emergency timing limitation described above remain applicable.

Outputs are separate from the read-only prior results, under [the local output directory](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027): [report](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/FINAL_REPORT.md), [analysis JSON](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/analysis_summary.json), [timeline CSV](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/timeline.csv), [boundary quality CSV](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/boundary_quality.csv), [timeline PNG](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/planning_timeline.png), [editable SVG](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/planning_timeline.svg), [saved XY contact PNG](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/saved_xy_contact.png), and the authoritative [H replay](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/evaluation/heuristic_reduced/replay/replay.json) / [L replay](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/evaluation/learned_reduced/replay/replay.json).
