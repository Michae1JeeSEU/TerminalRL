# Rescue opportunity v6 bounded two-scenario protocol

Status: **FOUR RUNS COMPLETED TO 480 s**. Both scenario pairs have a common completed boundary of 480 s and status `PARTIAL_REQUESTED_WINDOW`. All four replay audits passed. Numerical and representative-frame evidence is recorded below; human acceptance remains `HUMAN_PENDING`. The startup observations and initial estimates retained later in this document are historical.

**Final 480 s observations and limits.** The rule implementation and the bounded two-scenario experiment are complete. The quality outcome is not fully successful: below-reserve aircraft with repeated `UNKNOWN` rescue searches remain waiting. Zero failures and zero emergency touchdowns do not establish emergency success. The numerical source is `artifacts/rotating_slots_stage2_20261022/v6_final_0480_facts.json`; original per-step records remain in the frozen campaign roots.

| Scenario / method | Offered | Normal | Emergency | Total | F | Airborne / upstream | Declared / started |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stress / Slack | 67 | 24 | 0 | 24 | 0 | 43 / 0 | 6 / 4 |
| Stress / CP | 67 | 24 | 0 | 24 | 0 | 43 / 0 | 5 / 4 |
| Diagnostic / Slack | 41 | 25 | 0 | 25 | 0 | 16 / 0 | 0 / 0 |
| Diagnostic / CP | 41 | 23 | 0 | 23 | 0 | 18 / 0 | 2 / 1 |

Stress produced 24 normal touchdowns for each method, compared with 17 each in v5. Total touchdowns were 24 each versus v5 totals of 23 (Slack) and 25 (CP), so the total-landing comparison is mixed. Stress airborne backlog increased from 36 to 43 for both methods. Diagnostic backlog changed from 18 to 16 / 18; this single 480 s window does not prove sustainability. Initial/new-cohort successes were stress 22/2 (Slack), 19/5 (CP), and diagnostic 17/8 (Slack), 17/6 (CP).

No declared rescue had landed by 480 s. All 13 declared rescue outcomes are censored at that boundary: 9 had actually started, and 4 remained soft reserved. The wait statistics below end at actual rescue **start**, not landing. Protected rescue end times extend beyond the observation window; complete per-aircraft trigger SOC, slack, future wait, protected start/end and final ACTIVE/SOFT_RESERVED state are retained in the final-facts JSON.

| Scenario / method | Started waits N / mean / max s | Not-started waits N / mean s | Minimum SOC, declared and unlanded |
| --- | --- | --- | --- |
| Stress / Slack | 4 / 58.547 / 88.599 | 2 / 85.000 | 0.138 |
| Stress / CP | 4 / 82.534 / 96.532 | 1 / 65.099 | 0.017 |
| Diagnostic / Slack | 0 / — / — | 0 / — | — |
| Diagnostic / CP | 1 / 65.755 / 65.755 | 1 / 5.666 | 0.032 |

An unresolved 2/3/1/1 airborne aircraft (stress Slack / stress CP / diagnostic Slack / diagnostic CP) had SOC below .25 and no declaration. Their recorded normal status was `INFEASIBLE`; every last rescue-search result was `UNKNOWN` with reason `bounded_search_exhausted_no_infeasibility_proof`. These observations do not prove rescue physically impossible and prevent a claim that every last rescue opportunity was handled.

| Scenario / method | Aircraft | Ring | SOC | Exhaustion events | Last search s |
| --- | --- | --- | --- | --- | --- |
| Stress / Slack | initial_middle_00 | inner | 0.211756 | 72 | 478.500 |
| Stress / Slack | initial_inner_03 | inner | 0.204185 | 53 | 478.500 |
| Stress / CP | initial_outer_03 | middle | 0.232597 | 97 | 476.418 |
| Stress / CP | initial_middle_04 | middle | 0.248377 | 86 | 476.418 |
| Stress / CP | initial_inner_08 | inner | 0.218512 | 60 | 476.418 |
| Diagnostic / Slack | initial_outer_05 | inner | 0.202768 | 75 | 479.873 |
| Diagnostic / CP | req_000000 | inner | 0.173918 | 78 | 478.237 |

All timing values below are wall seconds, rounded to three decimals. Final cumulative call distributions are used once; repeated 60 s summaries are not added together. Candidate construction and safety validation dominate observed normal-planning time. Every logged normal call exceeded the 5 s reference (1251/1251); the normal soft budget remains 30 s and CP solver budget 8 s. Dispatch timing includes accumulated monitor/rescue work and is not a strict contiguous response latency.

| Scenario / method | N | Normal mean / P95 / max | Dispatch mean / P95 / max |
| --- | --- | --- | --- |
| Stress / Slack | 335 | 18.544 / 21.964 / 27.018 | 24.605 / 50.844 / 56.890 |
| Stress / CP | 332 | 20.083 / 23.615 / 34.924 | 25.914 / 50.698 / 56.325 |
| Diagnostic / Slack | 297 | 13.363 / 14.059 / 17.339 | 14.053 / 16.246 / 17.339 |
| Diagnostic / CP | 287 | 14.931 / 16.804 / 30.362 | 16.493 / 21.435 / 30.363 |

| Scenario / method | Candidate mean / P95 | Safety validation mean / P95 | Selected-validation overhead mean | Solver mean / P95 |
| --- | --- | --- | --- | --- |
| Stress / Slack | 11.208 / 12.286 | 7.047 / 11.504 | 0.001 | 0.000 / 0.000 |
| Stress / CP | 11.043 / 12.093 | 8.638 / 12.471 | 0.127 | 0.036 / 0.103 |
| Diagnostic / Slack | 9.898 / 10.770 | 3.381 / 5.126 | 0.000 | 0.000 / 0.000 |
| Diagnostic / CP | 9.864 / 10.646 | 4.911 / 8.935 | 0.021 | 0.028 / 0.183 |

CP entered on 332/332 move-eligible stress calls and 287/287 diagnostic calls. Its recorded strict improvement over its own hint was 296/332 (one UNKNOWN) and 116/287, respectively. These are within-call comparisons; they do not establish realized superiority or global optimality.

Pad observations are dispatch samples. `ground_idle` means no aircraft in ground service; many such pads have an active approach. It is neither a usable-idle count nor time utilization. Flags overlap, while the reason code records the existing classification order.

| Scenario / method | Samples | Ground service | Ground idle | Approach reserved | Future calendar | Rescue protected | Idle without active approach, offered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stress / Slack | 2010 | 1188 | 822 | 1344 | 1325 | 269 | 10 / 2010 |
| Stress / CP | 1992 | 1184 | 808 | 1299 | 1349 | 480 | 5 / 1992 |
| Diagnostic / Slack | 1782 | 906 | 876 | 1430 | 1275 | 0 | 42 / 1782 |
| Diagnostic / CP | 1722 | 941 | 781 | 1297 | 1223 | 94 | 36 / 1722 |

| Scenario / method | Ground reason | Approach reason | Future reason | Not submitted | Submitted |
| --- | --- | --- | --- | --- | --- |
| Stress / Slack | 1188 | 812 | 6 | 1 | 3 |
| Stress / CP | 1184 | 803 | 2 | 0 | 3 |
| Diagnostic / Slack | 906 | 834 | 2 | 37 | 3 |
| Diagnostic / CP | 941 | 745 | 4 | 28 | 4 |

The explicit-conflict, candidate-not-found, emergency-protected and UNKNOWN **reason-code** counts were zero in these samples; rescue-protected flags were nevertheless 269/480/0/94. Future/rescue flag overlaps were 269/476/0/94. Absence of a reason code does not prove absence of that constraint. After the first declaration, offered idle-without-active-approach samples were 3/378, 2/846, not applicable (no declaration), and 16/558. `PLANNING_NOT_SUBMITTED` samples with positive `certified_single_count` were 1/1, 0/0, 37/37 and 28/28. These are observed catalog/dispatch combinations, not a causal account or a counterfactual feasibility proof.

The primary agent viewed 9 representative PyVista PNG frames and both comparison curves. Actual normal-final starts after the first declaration were 2 / 7 / not applicable / 8. No large blanket pad idling was apparent in those representative frames; this is not proof that all opportunities were exploited. Visual status is `AGENT_REPRESENTATIVE_FRAME_REVIEWED`, with human acceptance still `HUMAN_PENDING`.

Replay evidence is in `artifacts/rotating_slots_stage2_20261022/v6_replay_checks/check.json`: four load/render passes, 15 seek checks with all recorded checks true, and 9 PNGs. The viewer code diff from the archive commit is empty. Seek and representative-frame times in seconds are:

| Scenario / method | Seek times | Rendered PNG times |
| --- | --- | --- |
| Stress / Slack | 0.000, 380.000, 420.000, 480.000 | 381.000, 419.344, 480.000 |
| Stress / CP | 0.000, 261.331, 420.000, 480.000 | 262.331, 310.888, 480.000 |
| Diagnostic / Slack | 0.000, 420.000, 480.000 | 480.000 |
| Diagnostic / CP | 0.000, 327.745, 420.000, 480.000 | 328.745, 480.000 |

Recorded PowerShell viewer commands from `check.json`:

```powershell
Set-Location 'D:\7_codespace\terminal'
& .\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; v=RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_stress_480s_20261022\energy_slack\replay_0480')); v.run()"
& .\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; v=RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_stress_480s_20261022\rh_cp_sat\replay_0480')); v.run()"
& .\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; v=RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_diagnostic_480s_20261022\energy_slack\replay_0480')); v.run()"
& .\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; v=RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_diagnostic_480s_20261022\rh_cp_sat\replay_0480')); v.run()"
```

Both wrapper logs contain reliable run/wrapper exit codes 0, while raw exit files remain malformed `0n`. Stress ended at 2026-10-22 14:33:59 UTC with wrapper run elapsed 9153 s; diagnostic ended at 13:23:09 UTC with elapsed 4903 s. Terminal campaign wall values are 9815.156 s / 5555.479 s, including preparation credit; stored terminal budget floors are 9815.154 s / 5555.478 s. These terminal values are used instead of subtracting the budget epoch from a post-completion clock. All four checkpoint/summary/state streams reach 480 s, all four audits pass, and all eight method error/invalid paths are absent. Heartbeat `stage2-v6-480` has been deleted; no further run was launched.

The frozen comparison is exactly two named scenarios, each with `energy_slack` and `rh_cp_sat`, run once to the common 480 s boundary. These are synthetic initial backlogs and a diagnostic load comparison; neither is labelled sustainable or normal traffic.

| Scenario | Initial airborne O/M/I | Manifest |
| --- | --- | --- |
| `stress36_rate180_v1` | 14/12/10 (36) | Seed 71, 180 requests/hour, 3600 s release window; byte-identical to the old v5 manifest |
| `diagnostic18_rate120_v1` | 6/6/6 (18), evenly spaced on existing ring slots | Seed 71, 120 requests/hour, 3600 s release window, materialized before either method starts |

Both use truncated-normal initial SOC with mean .75, standard deviation .08, interval [.60,.90], SOC RNG 2071, service RNG 2072, and short/standard/long probabilities .2/.6/.2. `service_half_v1` remains 30/60/120 s. Existing initial ground occupation remains the same fixture (pads 0/2/4 with 15/30/45 s remaining under this service profile). Geometry, physical/energy parameters, safety, the 25% normal-touchdown SOC reserve and positive-SOC emergency touchdown rules remain unchanged. The normal `quality_v5` budget stays 30 s soft, 8 s solver and 64 refinement rounds, with its other frozen settings preserved.

Prepare with explicit `--scenario-profile` and `--rescue-rule-profile rescue_opportunity_v6`. Profiles are persisted in provenance, campaign metadata and checkpoint identity; changing a profile on run/resume is rejected. The default `legacy_busy_v5` / `legacy_v5` preserves prior callers and archived roots. Each pair uses the same materialized config, fixture and manifest bytes for both methods, one numerical thread and one solver worker per method. Optional run/resume `--cpus A B` selects two available distinct physical-core representatives and freezes that selection; the two scenario pairs can use disjoint core pairs. No old config, campaign root or result is overwritten.

The source fingerprint includes `rescue_rules.py` through the existing rotating-slots source glob. Checkpoints persist the rescue profile and additive emergency state; two named derived rescue caches are excluded and recomputed after restore. Existing replay export and audit remain in place.

Frozen code revision: `9cfb78149f9c986025ba8a08707162d56fb7b645` (pushed). Both `artifacts/rotating_slots_stage2_20261022/v6_stress_provenance.json` and `v6_diagnostic_provenance.json` record source hash `a0bfbc06e428a7826700c1dc898234b5122d716b5596a4d44f3252a82bd8b9dc` and effective config hash `5582b671279c0162dd3f6f72da12368a5b2ecffd12d97f5d59eaf945a47ecf87`. The `energy.py`, `safety.py` and `trajectory.py` file hashes match the v5 provenance; their aggregate numeric-backend hash remains `8eb32dcd82da77b54e43176100462ab9cde3a1d8cc4e3ac56df3eb826f22beea`. The complete planning-budget settings also match v5. Compact evidence is recorded in `artifacts/rotating_slots_stage2_20261022/v6_validation_facts.json`.

The v6 rescue rule protects a specific certified opportunity: aircraft, pad, trajectory hash, start time and complete pad window through ground service. Its energy assessment includes the holding interval before that start. Search tries pads in rounds with at most three duration alternatives per pad, the existing 20–300 s duration grid at 5 s spacing, and current time plus at most two calendar-release starts within a 120 s lookahead. Every returned opportunity passes the existing joint replacement validator and is revalidated on submission. Probing alone reserves nothing; protection follows an accepted rescue commitment. Bounded search exhaustion is `UNKNOWN`; it does not establish physical infeasibility or justify excluding normal candidates.

`rescue_rules.py` versions the trigger constants: monitoring interval 10 s, dispatch allowance 5 s, rescue buffer 15 s, lookahead 120 s and three duration alternatives. A warning can reflect SOC at or below .35, a normal-reserve lower-bound violation, or a small remaining normal-path energy margin. A warning preserves normal candidates. Normal impossibility requires the conservative energy lower bound to exceed the SOC available above .25; an unsuccessful finite normal search remains `UNKNOWN`. A certified rescue becomes required when normal reserve is provably impossible, or when its remaining holding slack after flight and future waiting is at most the 15 s buffer. Normal-path estimates are source/version checked and expire after 5 s; they are estimates rather than infeasibility proofs.

The rescue declaration patch labels the first required rescue takeover immediately, including time spent waiting to submit or start that rescue. This distinguishes declaration, actual maneuver start, emergency charging and touchdown. The final targeted server regression passed after this patch and the corrected test expectations.

Planning reuse remains exact to its inputs. Movement-energy keys contain config hash, full planning node and trajectory hash. Constructed-template keys contain config, aircraft, source ring/slot/version, target, start and duration; failed bounded template searches are not retained as infeasibility evidence. Single/pair validation keys include the immutable snapshot identity, complete column signatures and resource intervals; only resolved results are reused, and `UNKNOWN` is not cached as a rejection. Validation caches are bounded at 8192 entries, constructed templates at 4096, and the derived rescue-opportunity cache at 32. Previous chains trim only executed or fixed prefixes and retain their certified times, with source-state, energy and peak checks before reuse.

The candidate catalog retains both full normal chains and useful partial progress, preserving target/start coverage before adding duration variants. Its retention rule permits up to two retained moves per first target within the bounded catalog. Joint CP validation can return up to 16 explicit conflicts in one pass and add the corresponding single or pair exclusions; explicit joint prefix rejection can add a transaction no-good. `UNKNOWN` adds no cut. These changes keep the normal 30 s soft budget, 8 s solver budget and 64 refinement rounds unchanged.

Summaries retain normal/emergency/total touchdowns, failures, pending backlogs and normal/dispatch timings. `rescue_observations` adds distinct warning, declaration, rescue start and low-reserve touchdown counts; per-aircraft SOC/reserve and peak margins; rescue waits from first trigger to actual rescue start, with waits censored at the observation boundary when no start occurs; and per-aircraft candidate coverage. Initial ground fixtures are excluded by the existing Stage2 cohort definition (fixture with touchdown time zero and zero airborne time). Exclusive candidate timing sums candidate generation, safe-progress generation, reference-seed generation and geometry-template screening. Safety-validation timing sums only the exclusive trajectory-validation and prefix-validation phases; the remaining selected-validation phase is separately labelled management overhead. These components are not added again through their enclosing phases. Solver timing remains separate. Every family records its denominator. Rescue wait completion refers to rescue start, not touchdown.

Decision `pad_opportunities` samples are taken immediately before normal planning. `ground_idle` means no aircraft is undergoing ground service on that pad; `approach_reserved` separately marks an active approach. `idle_without_active_approach` combines these observations without describing an approaching aircraft as physically on the ground. Future-calendar and emergency-protected flags remain separate, so a future protected rescue is included in both counts even when its reason is `FUTURE_CALENDAR`. Samples then merge observed per-pad catalog/validation/submission evidence from that call. Reasons distinguish grounded service, active approach reservation, future calendar, an actually protected rescue, observed explicit trajectory conflict, no candidate, and planning not submitted. Missing evidence remains `UNKNOWN`; candidate absence and bounded search exhaustion are never physical infeasibility certificates. Samples are observations at dispatch timestamps, not time-weighted utilization or a causal proof that a pad could have been used.

The read-only `stage2_campaign.rescue_comparison_tables(root)` helper reads the latest common completed boundary and returns final tables. `comparison.json` includes those tables for new reports. Legacy v5 summaries without new fields explicitly produce `UNKNOWN`.

Historical reference: v5 final follow-up revision `078cfad`, frozen source revision `77c3d16c4a4cb65467dc6e0e30686906e9d2ec32` recorded in `artifacts/rotating_slots_stage2_20261022/v5_provenance.json`. Those results remain historical comparison evidence and are not v6 results.

Historical server validation root: `$HOME/training_outputs/terminal/stage2_rescue_v6_validation_20261022`. All 48 checked file hashes matched. The original regression completed in 160.31 s with **263 passed, 2 failed, 1 skipped and 3 deselected**. All four actual t=0 scenario/method callbacks and their checkpoint roundtrips passed. The primary review diagnosed the two failures as test assumptions: a degenerate phase-0 selection assumption, and an old expectation that two allowed prefixes meant only depth-one progress. The revised catalog retains depth-two progress as well as the complete depth-three chain. Those test assumptions were corrected; the actual CP case passed in the final regression.

The `final_targeted` server validation then completed with **84 passed, 1 deselected in 117.14 s**, exit code 0, with **48/48 file hashes matched**. Local targeted checks recorded **38 passed in 9.92 s** for the declaration fix and **29 passed, 1 skipped in 21.77 s** for the corrected tests; an earlier local check requiring Git recorded 3 passed. These are validation results, not formal 480 s results.

The actual t=0 callback/checkpoint measurements from `final_targeted` server validation were:

| Scenario | Method | Wall s | CPU s |
| --- | --- | --- | --- |
| Stress | `energy_slack` | 26.4099 | 26.4050 |
| Stress | `rh_cp_sat` | 31.7942 | 31.6358 |
| Diagnostic | `energy_slack` | 17.7051 | 17.7020 |
| Diagnostic | `rh_cp_sat` | 25.8133 | 25.7330 |

Preflight observed CPUs 0/2/4/6 as distinct matching physical cores, idle with no competing jobs at that observation. This records the preflight state rather than a guarantee of future load.

The formal roots are under `$HOME/training_outputs/terminal`:

| Root | Method CPU representatives | Requested window |
| --- | --- | --- |
| `stage2_rescue_opportunity_v6_stress_480s_20261022` | `energy_slack`: 0; `rh_cp_sat`: 2 | 480 s |
| `stage2_rescue_opportunity_v6_diagnostic_480s_20261022` | `energy_slack`: 4; `rh_cp_sat`: 6 | 480 s |

This is exactly four method runs across the two scenarios. Each pair has a planned total wall budget of 21600 s: 19800 s before the compute stop and 1800 s reserved for archival work, matching the v5 campaign wall allowance. The common preparation credit is **600 s per pair**, a conservative shared preflight allowance rather than a claim of 600 s actually measured. It includes the two measured server validations (160.31 + 117.14 = **277.45 s**) and the local targeted preflight. These wall allowances do not increase any per-decision planning or solver budget.

Preparation exited 0 once for each pair. Wrapper timing reported coarse durations of 3 s (stress) and 1 s (diagnostic); internal preparation values were 602.5956871503731 s and 601.0257686807308 s, respectively, including the 600 s credit. Stored elapsed floors were 602.5942876653862 s and 601.0241105857422 s, with budget epoch starts 1792669824.5768228 and 1792669833.8348737. These persisted floors update at completed blocks and are not current elapsed-time readings.

Startup verification at server **12:07:25 UTC** confirmed the stress wrapper/runner PIDs **145674/145685**, resource tracker **145697**, and method PIDs **145698** (`energy_slack`, CPU 0) / **145700** (`rh_cp_sat`, CPU 2). Diagnostic wrapper/runner PIDs were **145675/145686**, tracker **145695**, and method PIDs **145696** (CPU 4) / **145699** (CPU 6). Each method had one thread and all four numerical-thread environment settings were 1. Both campaigns reported `RUNNING`.

Both commands run from `/home/user/projects/terminal` using `/home/user/envs/terminal_stage2_v1/bin/python scripts/server/run_rotating_slots_stage2.py run`, with their respective `--output` root, authoritative `--target 480`, and `--cpus 0 2` or `--cpus 4 6`. The existing `campaign.target_s` field remains its default 3600; it does not change the explicit 480 s CLI request. Successful completion of this request should report `PARTIAL_REQUESTED_WINDOW` with a common boundary of 480 s, rather than `COMPLETED_3600`.

At server **12:04:02 UTC**, each method had a nonempty initial checkpoint and an advancing state stream. Observed state times were stress 15 s / 15.3043 s and diagnostic 18.3887 s / 17.7509 s (`energy_slack` / `rh_cp_sat`). These are state-stream observations, not completed boundary checkpoints. All eight method-local `error.json` / `invalid.json` paths were absent at the final startup check. No true campaign exit-code or timing result existed yet.

The shared environment-lock hash is `b15a7e9c1debc9e91cffa8029e48706f7fd917382d7be05b873f64a61348eba2`, matching v5. The stress manifest hash is `10b97225bde7992af59c79857b0c387bde69ec3b1153e36867a9d42406ad4bc6` (172 requests); diagnostic is `b827cae3e8d961a3e94102378c6a9c552bf20f1b12fd30d0445b5fce8cb52743` (118 requests). Both retain seed 71 and a 3600 s arrival window, with rates 180/120 respectively. Fixtures were verified as 14/12/10 plus three ground aircraft and 6/6/6 plus three ground aircraft.

A minor wrapper formatting deviation left the start file with literal bytes `2026-10-22T12:01:26Zn` and the raw exit file with `0n`. Both complete wrapper logs independently contain exact `run_exit_code=0` and `wrapper_exit_code=0` lines. The raw `0n` file is not a valid integer exit-code file. Original output files are preserved.

Historical initial runtime estimates were **12600 s (3.5 h)** for the stress pair and **10800 s (3 h)** for the diagnostic pair. The stress estimate used approximately 365 dispatch calls at the observed 26–32 s t=0 planning scale, with allowance for emergency work and archival processing. The diagnostic estimate was conservative because later workloads differ from the initial dispatch. At the historical estimate anchor (local 2026-10-22 12:04:49.3638422 UTC; server 12:05:41 UTC), observed process elapsed time was 255 s, giving then-remaining estimates of 12345 s / 10545 s. Startup state times were not extrapolated. Both runs have since ended; heartbeat `stage2-v6-480` was successfully deleted by the primary agent. No new run was started.
