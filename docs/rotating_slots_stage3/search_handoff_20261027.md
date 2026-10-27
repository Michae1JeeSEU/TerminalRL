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

## Commands prepared for later authorization

Use the primary-reviewed frozen checkout and source-freeze paths in these commands. Preparation is a separate step and requires a fresh output directory.

```bash
python "$CHECKOUT/scripts/stage3/probe_search_handoff.py" prepare --checkout "$CHECKOUT" --source-freeze "$FREEZE" --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/fixed_state
python "$CHECKOUT/scripts/stage3/probe_search_handoff.py" run --checkout "$CHECKOUT" --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/fixed_state
# Primary review of fixed-state evidence precedes the following closed-loop pair.
python "$CHECKOUT/scripts/stage3/run_latency_interface.py" prepare --checkout "$CHECKOUT" --source-freeze "$FREEZE" --lead-seconds 20 --search-handoff-reserve-seconds 4 --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop
python "$CHECKOUT/scripts/stage3/run_latency_interface.py" run --checkout "$CHECKOUT" --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop
python "$CHECKOUT/scripts/stage3/run_latency_interface.py" summarize --checkout "$CHECKOUT" --output /home/user/training_outputs/terminal/stage3_search_handoff_20261027/closed_loop
```
