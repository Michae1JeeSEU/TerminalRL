# Rescue opportunity v6 bounded two-scenario protocol

Status: **NOT_LAUNCHED**. The formal run protocol and wall budgets are planned below; no formal v6 PID, start time or result is recorded. Validation runs are identified separately.

The frozen comparison is exactly two named scenarios, each with `energy_slack` and `rh_cp_sat`, run once to the common 480 s boundary. These are synthetic initial backlogs and a diagnostic load comparison; neither is labelled sustainable or normal traffic.

| Scenario | Initial airborne O/M/I | Manifest |
| --- | --- | --- |
| `stress36_rate180_v1` | 14/12/10 (36) | Seed 71, 180 requests/hour, 3600 s release window; byte-identical to the old v5 manifest |
| `diagnostic18_rate120_v1` | 6/6/6 (18), evenly spaced on existing ring slots | Seed 71, 120 requests/hour, 3600 s release window, materialized before either method starts |

Both use truncated-normal initial SOC with mean .75, standard deviation .08, interval [.60,.90], SOC RNG 2071, service RNG 2072, and short/standard/long probabilities .2/.6/.2. `service_half_v1` remains 30/60/120 s. Existing initial ground occupation remains the same fixture (pads 0/2/4 with 15/30/45 s remaining under this service profile). Geometry, physical/energy parameters, safety, the 25% normal-touchdown SOC reserve and positive-SOC emergency touchdown rules remain unchanged. The normal `quality_v5` budget stays 30 s soft, 8 s solver and 64 refinement rounds, with its other frozen settings preserved.

Prepare with explicit `--scenario-profile` and `--rescue-rule-profile rescue_opportunity_v6`. Profiles are persisted in provenance, campaign metadata and checkpoint identity; changing a profile on run/resume is rejected. The default `legacy_busy_v5` / `legacy_v5` preserves prior callers and archived roots. Each pair uses the same materialized config, fixture and manifest bytes for both methods, one numerical thread and one solver worker per method. Optional run/resume `--cpus A B` selects two available distinct physical-core representatives and freezes that selection; the two scenario pairs can use disjoint core pairs. No old config, campaign root or result is overwritten.

The source fingerprint includes `rescue_rules.py` through the existing rotating-slots source glob. Checkpoints persist the rescue profile and additive emergency state; two named derived rescue caches are excluded and recomputed after restore. Existing replay export and audit remain in place.

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

The actual t=0 callback/checkpoint measurements from server validation were:

| Scenario | Method | Wall s | CPU s |
| --- | --- | --- | --- |
| Stress | `energy_slack` | 26.4099 | 26.4050 |
| Stress | `rh_cp_sat` | 31.7942 | 31.6358 |
| Diagnostic | `energy_slack` | 17.7051 | 17.7020 |
| Diagnostic | `rh_cp_sat` | 25.8133 | 25.7330 |

Preflight observed CPUs 0/2/4/6 as distinct matching physical cores, idle with no competing jobs at that observation. This records the preflight state rather than a guarantee of future load.

The planned formal roots are under `$HOME/training_outputs/terminal`:

| Root | Method CPU representatives | Planned window |
| --- | --- | --- |
| `stage2_rescue_opportunity_v6_stress_480s_20261022` | `energy_slack`: 0; `rh_cp_sat`: 2 | 480 s |
| `stage2_rescue_opportunity_v6_diagnostic_480s_20261022` | `energy_slack`: 4; `rh_cp_sat`: 6 | 480 s |

This is exactly four method runs across the two scenarios. Each pair has a planned total wall budget of 21600 s: 19800 s before the compute stop and 1800 s reserved for archival work, matching the v5 campaign wall allowance. The common preparation credit is **600 s per pair**, a conservative shared preflight allowance rather than a claim of 600 s actually measured. It includes the two measured server validations (160.31 + 117.14 = **277.45 s**) and the local targeted preflight. These wall allowances do not increase any per-decision planning or solver budget.

Initial runtime estimates are **12600 s (3.5 h)** for the stress pair and **10800 s (3 h)** for the diagnostic pair. The stress estimate uses approximately 365 dispatch calls at the observed 26–32 s t=0 planning scale, with allowance for emergency work and archival processing. The diagnostic estimate is conservative because later workloads differ from the initial dispatch. These estimates are not completion guarantees and will be updated from actual progress. Formal launch, runtime, final tables and interpretation remain **PENDING**.
