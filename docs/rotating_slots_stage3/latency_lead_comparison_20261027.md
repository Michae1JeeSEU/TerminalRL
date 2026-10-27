# Preregistered 20/30 s lead comparison — 2026-10-27

The user requested a short comparison of normal-planning lead times using the current `affected_traffic_revalidation_v1` interface. The 30 s H/L pair is reused read-only from the completed run; only one new 20 s H/L pair will be run. No new 30 s pair, seed scan, fit or algorithm change is included. H uses heuristic candidate ordering plus CP-SAT; L uses the same backend with the existing frozen learned ordering. A 20 s lead is an engineering trial setting, not a deployment value.

| Lead | Evidence | Arm order | Output |
| --- | --- | --- | --- |
| 30 s | Preexisting completed H/L pair, read-only | Original H then L | `/home/user/training_outputs/terminal/stage3_latency_revalidation_20261027` |
| 20 s | Exactly two new formal jobs | H then L, serial | `/home/user/training_outputs/terminal/stage3_latency_lead_comparison_20261027/lead_20` |

Both settings use root `s3r23`, seed 7324, original `medium24_rate150_v1` initial state and manifest, 120 simulated seconds, CPU 12, a 5 s heartbeat and 1 s preparation margin. The absolute ready deadline is snapshot + 29 s for lead 30 and snapshot + 19 s for lead 20. The original candidate-active 6 s, cumulative CP 8 s, hint 3 s, soft 30 s, callback watchdog 60 s, 32 attempts per aircraft and 1024 total-attempt limits remain unchanged. The existing minimum remaining-time logic may tighten them against the absolute deadline; it cannot reset or increase them. The endpoint still censors dispatch/delivery exactly at 120 s.

The new 20 s pair has its own fresh ledger: 900 s watchdog per job and 1800 s evaluation cap, without reset or retry. All five numerical-library thread counts are 1; profiler and rendering are off. Original world configuration, physics, energy, admission, emergency and authoritative safety checks are unchanged. The main-checkout core was checked against `d84f3a3` with no core diff when this runner change was prepared. The execution source freeze for the new driver will be recorded separately from the original training fingerprint.

Model SHA256 is `dba25b3393a5580bdc3971d9adf7d6e3cff6360404fdb08bc8fbb6f0b36e3b48`; normalizer SHA256 is `1e06b066dad2d657acbe16de6706b180dd4f4ed8ba1bc106cefcc2a3d4f4e7be`. Their original binding/fingerprint is retained. The 30 s comparison source is commit `d84f3a372b04221b5e4c7845a403add906fb59b0`, source hash `e481913216dd4b75f8111bd25a38e2f391ba5f988cfcd3b05b744d2831c5c9d8`. Its inputs, results and ledger are not copied over or rewritten; comparison reporting links the existing [30 s report](D:/7_codespace/training_outputs/terminal/stage3_latency_revalidation_20261027/FINAL_REPORT.md).

## Runner interface and local checks

Only `prepare` accepts `--lead-seconds {20,30}`; omission preserves the existing 30 s default. Preparation stores the selected value in `execution_scope.json`. Verification requires the selected allowed value, every other fixed protocol field, the scope digest and the original preparation receipt. `run`, `job` and `summarize` reject a CLI lead override, and the actual wrapper reads lead/margin from the verified scope. Existing 30 s roots retain their original protocol schema. The intended new preparation specifies `--lead-seconds 20` and the fresh `lead_20` output explicitly; no new `lead_30` preparation is scheduled.

Compilation, CLI help and saved-output campaign fixtures passed locally: **13 tests in 1.09 s**, RC 0. Fixtures serialize and read back both leads, observe the wrapper receiving the scope value, reject invalid values and protocol changes, reject non-prepare overrides, and retain existing output/timeout/aggregation checks. They do not start new physics scenes, model inference or solves. Receipts: `artifacts/latency_lead_campaign_tests_20261027_a.log` and `.xml`.

## Interpretation boundary and status

The comparison will describe actual joint joins, complete planning/dispatch costs, deadline and dependency outcomes, terminal/common-boundary quality, energy and holding. Lead is the only intended algorithmic setting changed, but the reused 30 s pair and fresh 20 s pair execute at different times and under potentially different server load. Small wall-time differences are therefore not a strict simultaneous single-factor timing result. One development scenario and one pair per setting support no significance or generalization claim. Frozen-model future-view distribution and the normal-only/emergency-publication abstractions remain applicable.

The new 20 s pair is pending primary source review and formal execution. No 20 s outcome is recorded here.
