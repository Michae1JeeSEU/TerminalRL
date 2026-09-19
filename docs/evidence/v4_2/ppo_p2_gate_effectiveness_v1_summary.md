# P2 Gate Effectiveness V1 Terminal Summary

> Date: 2026-09-19  
> Evidence type: `newly_run`  
> Terminal status: `P2_EFFECTIVENESS_EVALUATION=FAIL`  
> Authorization: `P2_PPO_ADMISSION_REVIEW_ALLOWED=false`; `PPO_ALLOWED=false`; `P3_ALLOWED=false`

## Protocol

The once-only paired deterministic evaluation used fresh seeds `788001–788006`, with
`flow6/flow8/flow12` repeated twice. Each seed ran the repaired candidate, the frozen
pre-repair calibrated baseline, and the nominal rule under identical fixed-geometry natural-flow
conditions. Teacher, labels, optimizer updates, and policy updates were disabled. The aggregation
unit is the completed UAV, weighted across 52 requests per controller.

Protocol SHA256:
`5882361adcabaff4c0017cae73a8b95dc8029e204942bf4a83879981bd321d84`.

## Terminal evidence

All 18 controller episodes completed, and each controller served 52/52 requests. Safety counters,
shield interventions, reverse-transfer samples, Teacher calls, labels, and updates were zero;
action-tape replay and requested-action likelihood checks were exact.

| Controller | Completion (s) | Weighted wait (s) | Request-to-touchdown (s) |
|---|---:|---:|---:|
| repaired candidate | 880.3333 | 15.9619 | 309.5577 |
| calibrated baseline | 876.0000 | 17.3097 | 310.1923 |
| nominal rule | 898.1667 | 14.3801 | 319.8654 |

The candidate improved both preregistered metrics relative to the baseline. Request-to-touchdown
gap closure passed at `1.0`; waiting-time gap closure was `0.4600799557`, below the frozen `0.50`
threshold. The overall result is therefore FAIL even though all hard gates and the other three
effectiveness checks passed.

Formal manifest SHA256:
`6aaebeac4dc8dcf425ce250df6c6abcd991921263605b3dd71d3d717f0783e88`.

## Failure localization

Five of six paired seeds reduced candidate waiting relative to baseline. The exception was
`788003/flow12`: candidate wait was `22.8488 s` versus baseline `15.8853 s`. Read-only trace
inspection showed additional rejected or delayed RELEASE attempts in late m2i/i2f traffic,
especially for UAVs 7, 8, and 10. Because repair V3 changed only the six gate-adapter parameters
and left the conditional proposal head unchanged, the current evidence is consistent with a
gate–proposal/execution-feasibility coordination bottleneck under dense late flow12 traffic. This
is an attribution hypothesis, not yet a repair authorization.

## Protocol risks and boundary

- The six formal seeds are consumed and may not be rerun, selected, or used for training/tuning.
- The result is an internal fixed-geometry natural-flow evaluation; it does not establish broader
  traffic, geometry, or hardware generalization.
- The near-threshold aggregate does not justify changing the threshold after observing results.
- Trace localization used failed-evaluation artifacts descriptively; a fresh label-only diagnostic
  is required before choosing gate versus proposal support collection.

The next allowed experiment is a fresh `flow12` feasibility-attribution diagnostic on
`789001–789006`, with the repaired student authoritative and Teacher label-only. It creates no
training, PPO, P3, curriculum, or checkpoint-promotion authority.

