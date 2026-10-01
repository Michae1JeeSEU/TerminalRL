# P3 prefix fallback V2 technical-recovery execution record

## Status and evidence boundary

This file records a root-approved implementation-recovery admission. It is not a scientific resampling protocol, performance result, training run, PPO admission, qualification, or default-runtime promotion. V1 and V2 artifacts remain separate. V1 is retained as ineligible/unknown and is not overwritten or rerun for the selected business failure.

## V1 retained record

- V1 no-env preflight: `PREFLIGHT_OK`, SHA256 `c82a5a749bc4afa92468a9f7c6e2ddbf28748afa6568e3766a80501ca44cab46`.
- Pilot PID `2136806`: terminal status `technical_error`.
- The earlier missing-pilot report referred only to an early startup snapshot. A later retained pilot artifact exists with SHA256 `1cf74aa3ee3369ad0da4604d7a9e8b64e04e31b465f25ef8aca8b1b8ab554b10`; the retained arm/pair/sidecar/failure proof has SHA prefix `39cd0d...`.
- Observed V1 accounting: `31` native forwards, `33` registrations, `generated=14`, `admitted=13`, `observed_unresolved=14`, and `N/E/F/D=0/0/0/0`. Full performance metrics are unknown.
- Model-caller RNG, alias, and distribution checks passed. `technical/hard_safety_clear=false` is an incomplete certificate; it is not evidence of actual unsafe behavior or of zero verified violations.

## Minimal native singleton repair

The helper now preserves native canonical singleton RELEASE requests, including newly admitted requests absent from the policy map (the retained UID `28` record has `calls=0`). A singleton delegates directly to the unchanged original native function and does not request an order provider, sample, or expansion. A registered native K=3 request retains the prefix checks. Other malformed shortlist lengths fail closed. The repair does not change the native call semantics.

The supplied focused validation contains `21` passing tests plus successful compile and diff checks. These checks validate the helper contract; they do not convert V1 unknown metrics into performance evidence.

## V2 technical-recovery admission

- Output root: `p3_followup_20260930/prefix_fallback_20261001_v2`.
- Fixed scope: the original four cells, checkpoint, policy seeds, 1800-step cap, and thresholds.
- Protocol binding: `p3_prefix_fallback_protocol_20261001.md`, protocol SHA256 `29aa458a81e50aba4d0249cb1fe3b856829a1261feac3a3ad507c68fcfbdbc0c`.
- The new preflight must bind the singleton helper repair in addition to the existing source, checkpoint, configuration, and protocol bindings.
- Pilot cell 1 is permitted only after the new preflight passes and root reviews it. Full cells 2–4 require a separate root review of the pilot result.

V2 has no automatic relaunch, seed resampling, new population, budget expansion, PPO, held-out, teacher, optimizer, or default-promotion path. `PPO_ALLOWED=false`. Failures, censored runs, errors, and incomplete certificates remain retained; unknown metrics remain unknown.
