# Search-stop and final-authorization handoff: fixed protocol

This iteration reserves 4.0 s before the existing normal-planning ready deadline for preparation, guard and delivery. This is one fixed engineering choice, based on the previous preparation maximum of about 1.52 s and observed priority/handoff delay of about 2 s. It is not a reserve or lead sweep. The search certificate remains distinct from executor authorization and actual joining.

For lead 20 s and preparation margin 1 s, the ready deadline remains snapshot + 19 s. Search stops at the original planning wall deadline minus 4 s, at most snapshot + 15 s before existing queued service is deducted. The scheduler retains the certified incumbent and records first certification, subsequent improvements, return and heavy-check starts/skips. The 4 s allocation does not authorize a longer CP, hint, candidate, callback or experiment budget.

Before starting a heavy check, the new mode estimates its required headroom as `max(1 s, 1.25 × the measured high-water duration for the same kind within this callback)`. This is an estimate, not a completion guarantee. Each CP `Solve` also reserves the prefix, single and geometry check estimates from the remaining search window (3 s total when cold), so a new solver result has time left for certification. The same cumulative 8 s CP limit still applies. Measurements for these check kinds can be nested; they are not additive exclusive phases and must not be summed as a total.

## Execution order, subject to primary review

1. Validate the driver, serialized observations and one-task gate with local fixtures.
2. Review the implementation, then authorize the fixed-state interface check. Restore the original s3r23 t0 checkpoint and unchanged manifest/model once per H/L arm. Use CPU12 serially, lead 20 s, margin 1 s, reserve 4 s and the existing 5 s heartbeat. Advance the original world to 20.001 s, keeping monitoring, emergency response, admission, committed motion and pending delivery active. A thin gate prevents a second normal task after the first ends. Existing recorded world states, PVA/SOC, trajectories and start events are saved in `world_record.json`; no extra integration or full checkpoint is introduced. Each arm has a 90 s process cap and the pair has 180 s. Output: `/home/user/training_outputs/terminal/stage3_search_handoff_20261027/fixed_state`.
3. The primary agent reviews the fixed-state records. The relevant observation is whether a task with a nonempty search certificate starts preparation and reaches its actual authorization/join outcome. A missing certificate or failed delivery is retained as observed evidence; it does not trigger an automatic retry or reserve adjustment. This is the specified single-state interface check, not another short-scenario quality experiment.
4. Only after that review, authorize one 120 s H-then-L closed-loop pair at `/home/user/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop`, with lead 20 s and reserve 4 s. CPU12, 900 s per arm and 1800 s for the pair remain fixed. Exact-120 delivery is censored under the existing endpoint policy.

All stages keep seed 7324, root s3r23, medium24_rate150_v1, the original initial state and manifest, physical and emergency policies, model/normalizer bytes and training fingerprint unchanged. Candidate-active 6 s, CP 8 s, hint 3 s, soft 30 s, callback watchdog 60 s and attempt limits 32 per aircraft/1024 total stay unchanged. Reserve defaults to zero for prior driver behavior and is selectable only during preparation; run/job read its frozen scope. The new protocol records `search_handoff_reserve_v1` when the reserve is nonzero.

The old 20 s run `stage3_latency_lead_comparison_20261027/lead_20` and old 30 s run `stage3_latency_revalidation_20261027` are read-only references. They are not rerun. No 25/35/40 s experiment, fitting, feature change or model update is authorized here. First-certificate/return wall-clock evidence must not be described as final authorization or a simulation delivery timestamp; actual preparation, readiness and joining come from executor records. Replay human acceptance remains `HUMAN_PENDING`.

## Fixed-state observation and current authorization

The fixed-state check completed and the primary agent reviewed its saved evidence. Execution commit: `891adc1a9209944a9731da23f65788332f8756fe`; source hash: `c27b103ed9200d3ecb7d475347c54711a5b85ff4e81ba48cfd3f0415cc139c34`, covering 104 frozen file hashes. Primary-reviewed preflight recorded 72 passed, zero skipped/failed in 12.07 s. The earlier local checks recorded 35 passed with eight OR-Tools skips, plus 29 driver/report/probe fixture passes. An earlier fixture failure remains preserved; these counts do not erase that failed check.

Both fixed-state subprocesses returned RC 0 and reached the common observed terminal time 20.001 s with exactly one callback per arm. Each arm recorded 13 actual `maneuver_started` events, all at 20.0 s; every started trajectory hash was present in its returned proposal. These observations are in `fixed_state/evaluation/<arm>/result.json`; existing PVA/SOC, trajectories and events remain in the corresponding `world_record.json`.

| Recorded observation | H | L |
| --- | ---: | ---: |
| First nonempty certificate, elapsed wall s | 7.469896535971202 | 6.7211152269737795 |
| Callback return, elapsed wall s | 12.161827542004175 | 11.420191184966825 |
| Preparation start, simulation s | 12.164470774354413 | 11.42281109187752 |
| Recorded ready time, simulation s | 13.826293440186419 | 13.164398872992024 |
| Strict certified improvements | 2 | 2 |
| Actual starts at simulation 20.0 s | 13 | 13 |

The returned objective vectors, shown to the following precision, are H `[0, -6, 2629.84955556, 481818294.8, 7307]` and L `[0, -5, 2630.84955556, 515971259.4, 7666]`. L's returned objective is worse under the recorded lexicographic ordering in this fixed state. Timely joining therefore does not establish better learned quality, and neither result is a global-optimality claim or a rollout-quality conclusion.

The primary review also confirmed that original motion and energy consumption continued during computation: `initial_outer_00` SOC changed from `0.6567436446071294` to `0.6390858447994668`, and its position moved from `[600, 0, 110]` to approximately `[458.89178, -386.54667, 110]`. This is existing recorded physical evidence, not an added integration or safety simulation. `HUMAN_PENDING` remains unchanged.

Two earlier shell launch attempts created a stage-directory stub before preparation and were rejected by the fresh-output guard. They produced no prepare/run markers, ledger or evaluation. The three stub files and their hashes were preserved under `/home/user/training_outputs/terminal/stage3_search_handoff_20261027/startup_failure_before_prepare_20261027`; these are startup failures, not scientific reruns. After primary review, launcher artifact SHA256 `199d732d6328314f1db732e3447ab7c5738bebbb62adae2c50e57b657e90f0a3` performed the sole successful fixed-state prepare/run. The old 20 s and 30 s archives were not changed.

The fixed-state review established the interface chain: a nonempty certified proposal returned, preparation began, readiness was recorded before the unchanged deadline, and actual starts occurred. The primary agent then authorized the single preregistered 20 s lead / 120 s closed-loop pair. The completed observation below supersedes the earlier pending status.

## Closed-loop 120 s observation

**Primary-reviewed acceptance: the budget handoff passes for this one s3r23 scenario.** All 11 callbacks recorded a first nonempty search certificate and returned in 11.357687–12.882911 s, before their search deadlines at about 14.999 s from planning origin. All five H and all five L final nonempty proposals entered preparation. This addresses the previous 20 s behavior in which every task missed ready delivery before preparation began. It does not establish model-quality improvement, global optimality or performance across other scenarios.

The execution source remains commit `891adc1a9209944a9731da23f65788332f8756fe` / source hash `c27b103ed9200d3ecb7d475347c54711a5b85ff4e81ba48cfd3f0415cc139c34`. Both subprocesses returned RC 0, the common completed boundary is 120 s, and the ledger is paused with 180.761913 s charged to evaluation against the 1800 s pair cap. The model, initial state, manifest and physical policies were unchanged; no fitting, reserve scan or extra seed was run.

| Actual task outcomes | H | L |
| --- | --- | --- |
| Callback / task count | 6 | 5 |
| Joined groups: simulation time (submitted plans) | 20 (13), 80 (7) | 20 (13), 45 (1), 70 (1), 93.5 (1), 115 (9) |
| Other outcomes | 1 no-prefix, 2 rejected, 1 censored prepared | None |
| Finished nonempty join fraction | 2/4 | 5/5 |
| Revalidation rounds: certified / late | 1 / 2 | 1 / 0 |
| Revalidation measured work, s | 0.021760617 | 0.080378803 |
| Revalidation checked / reused pairs | 2 / 49 | 9 / 207 |

H's two `READY_DEADLINE_MISSED` outcomes were safe rejections after previously ready packets encountered new traffic too late for revalidation. For snapshot 37.294921875, preparation started at 49.864503133 s and the previous ready receipt was 50.032271741 s; new traffic led to revalidation at 56.657376971 s, after the original ready deadline 56.294921875 s. For snapshot 83.5, the first ready receipt was 96.989492034 s and one revalidation completed successfully at 99.790035759 s; another new-traffic revalidation started at 102.693460605 s, after deadline 102.5 s. These are not failures to start preparation or evidence that search exhausted the full 19 s window. H's snapshot-105 task was ready at 119.070119026 s for effective time 125 s; it remains pending at the censored 120 s endpoint and is neither a failed nor a successful join.

H's snapshot-25 no-prefix return follows the unchanged objective: its first nonempty certified key was `[0, 0, 18, 449917810, 6729.5]`, while the final empty-prefix key was lexicographically better, `[0, 0, 18, 417649492.9, 6489.5]`. No candidate or objective was forced to favor a nonempty return. Frozen execution 891 records the first receipt and final return accurately, but does not record the exact instant of this empty-prefix improvement. That instant is unavailable and is not retrofilled. After this run, a two-line telemetry-only correction was completed in the working checkout, with three targeted regression tests passing in 2.05 s under primary review. This later source change records the empty-prefix improvement for future executions; it is separate from frozen execution 891 and does not change its source hash, saved results or model. The experiment is not rerun for it.

| Saved quality and work at 120 s | H | L |
| --- | ---: | ---: |
| Normal / emergency / energy-failed / airborne-pending / admitted | 6 / 0 / 0 / 23 / 29 | 6 / 0 / 0 / 23 / 29 |
| Cumulative energy, J | 222309008.74012685 | 222163277.5954829 |
| Holding, aircraft-s | 1759.7414594252973 | 1893.2414594252973 |
| All-task observed planning work, s | 78.88279208808672 | 63.260321425041184 |
| Finished complete-work P95, s (n=5 each) | 13.886163159739226 | 13.23355327907484 |
| Dispatch total, s | 82.38206493819598 | 66.2932254599873 |
| Dispatch P95, s | 12.480093564838171 | 11.52523332403507 |

L used 19.804662% less observed planning work and 19.529541% less dispatch wall time in this pair. H's pending work is included in the all-task total but excluded from finished-task P95. The arms have five versus six callbacks and different subsequent routes/states, so these differences are not a pure ranking effect or evidence of stable long-run performance. L energy is 0.065553% lower, while holding is 7.586342% higher; it is not uniformly better on quality.

| Saved computation / reuse observations | H | L |
| --- | ---: | ---: |
| CP wall, s | 14.652012418 | 13.350663338 |
| Prefix validation, s | 8.814345188 | 6.713109785 |
| Trajectory validation, s | 16.454028529 | 9.278453458 |
| Stage-3 features / inference, s | Unavailable / unavailable | 2.315718473 / 8.815807075 |
| Same-catalog continuation calls reused | 6 | 5 |
| Caller incumbents accepted | 10 | 5 |
| Reused pair cuts / single results | 136 / 467 | 67 / 331 |
| Solver internal state reused | 0 | 0 |
| Learned ranks / comparable-group cache hits | 0 / 0 | 334 / 9 |
| Missing full-chain callback-aircraft coverage observations | 97 | 85 |

L's cache miss counts were 321 `CANDIDATES_CHANGED` and 13 `NO_BATCH`. Feature encoding plus inference cost 11.131526 s, and coverage limitations remain. All recorded heavy-check skipped and overrun counts were zero in this run. The gate behavior is covered by fixtures, but this rollout does not demonstrate benefit from a gate actually skipping a heavy operation. Heavy-kind measurements may nest and must not be added as exclusive work phases.

The old 20 s run had no joins or normal landings; the old 30 s run had two joins and six normal landings per arm. New first joins are at 20 s versus 30 s in the old 30 s run. New cumulative energy is 6.141935% / 6.203463% lower than the old 20 s H/L values, but 1.400493% / 0.560518% higher than the old 30 s values. These are historical descriptions: execution times, server load, later states and trajectories differ, limiting strict causal attribution. The unchanged learned model was not retrained for future projected states, and this result is not a model-capability ceiling.

Saved replay validation and all 15 file hashes pass in each arm. PVA checks cover H 4018 and L 3856 samples within tolerance. All H 20 / L 25 unique-predecessor C2 rows are `CHECKED`; maximum p/v/a errors are H `2.84217e-14 / 8.88178e-16 / 2.77556e-17` and L `2.04952e-13 / 7.53644e-15 / 2.35514e-16`. Actual normal starts are 20/25, all acknowledged, with no unattributed hashes or start-time mismatches. The two/five committed groups each start together at their effective times.

The new saved `search_handoff.callback_return.prefix_hashes` also retain rejected and pending proposals. Supplemental checks compare these returned hashes with each arm's actual-normal-start set: H's snapshot-37.294921875 rejected proposal has one hash, snapshot-83.5 has two, and the snapshot-105 pending proposal has four; none of these seven hashes started by the observed 120 s endpoint. The pending check is censored at 120 s and says nothing about starts afterward. H's snapshot-25 final prefix is empty. Original `UNAVAILABLE_PROPOSED_HASHES` group rows came from the acknowledgement-only reporter, which did not consume the new return trace; those original rows remain unchanged, while `primary_review/supplemental_proposal_start_checks.json` records the saved-trace supplement. The original producer's limitation is not missing raw proposal evidence in this run.

The primary agent actually viewed both existing PNGs. The timeline clearly shows H's two commits, two late revalidation rejections, one no-prefix task and one pending prepared task, L's five commits, and six normal landings in each arm. The saved 0/60/120 s contact states are readable. Visual QA is `PRIMARY_VIEW_IMAGE_PASS`; these static views do not supply full human safety acceptance, and `HUMAN_PENDING` remains.

Evidence: [analysis and scalar checks](D:/7_codespace/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop/analysis_summary.json), [saved task timeline](D:/7_codespace/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop/timeline.csv), [H raw task records](D:/7_codespace/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop/evaluation/heuristic_reduced/normal_tasks.json), [L raw task records](D:/7_codespace/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop/evaluation/learned_reduced/normal_tasks.json), [timeline figure](D:/7_codespace/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop/planning_timeline.png), [saved contact figure](D:/7_codespace/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop/saved_xy_contact.png), [old 20 s analysis](D:/7_codespace/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20/analysis_summary.json), [old 30 s analysis](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/analysis_summary.json). The copied output-base version of this document is `COMPARISON.md`; old archives remain read-only.

## Fixed commands

The commands below record the fixed protocol; they are not instructions to repeat the completed probe. Use the primary-reviewed frozen checkout and source-freeze paths. Preparation requires a fresh output directory, and current authorization is stated above.

```bash
python "$CHECKOUT/scripts/stage3/probe_search_handoff.py" prepare --checkout "$CHECKOUT" --source-freeze "$FREEZE" --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/fixed_state
python "$CHECKOUT/scripts/stage3/probe_search_handoff.py" run --checkout "$CHECKOUT" --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/fixed_state
# Primary review of fixed-state evidence precedes the following closed-loop pair.
python "$CHECKOUT/scripts/stage3/run_latency_interface.py" prepare --checkout "$CHECKOUT" --source-freeze "$FREEZE" --lead-seconds 20 --search-handoff-reserve-seconds 4 --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop
python "$CHECKOUT/scripts/stage3/run_latency_interface.py" run --checkout "$CHECKOUT" --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop
python "$CHECKOUT/scripts/stage3/run_latency_interface.py" summarize --checkout "$CHECKOUT" --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop
```
