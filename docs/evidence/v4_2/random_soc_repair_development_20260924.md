# Random-SOC Repair Development Comparison

Protocol: `RANDOM_SOC_REPAIR_DEVELOPMENT_COMPARISON_V1_20260924`
Registered: 2026-09-24, before consuming a new reset.
Status: development comparison only. This record is not an I1 qualification, a
training protocol, a curriculum promotion, or a PPO authorization. Keep
`PPO_ALLOWED=false`.

## Purpose and decision boundary

The comparison separates controller requests from execution feasibility in the
current dense30 V2.1 diagnostic. It uses one fresh common-random-number seed
for two controller arms:

1. the frozen trained P2 K3 checkpoint with deterministic inference;
2. an explicit `nominal_rule` request generator that sends the canonical
   zero-speed, zero-timing RELEASE proposal to each eligible actor.

The rule arm is an execution-request reference with an explicit identity. It
does not make a neural-behavior claim, and it does not enable the fixed-rule
flow coordinator. The result may retain observations about request patterns,
same-step execution acceptance, emergency outcomes, failed-energy outcomes,
and unresolved requests. It does not establish that either controller is
qualified or that a future training repair is sufficient.

## Frozen checkpoint and runtime configuration

Both arms use the same current dense30 diagnostic configuration and the same
checkpoint lineage. The trained arm is loaded from:

```text
/home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt
SHA256 714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e
```

The frozen runtime values are:

| Field | Value |
|---|---|
| vehicle profile | `eh216s_research_v2` |
| observation | V2.1, `88D`, `phaseA_low_obs_v2_1_retry_state` |
| action | V4.2 hierarchical, runtime K3 bound |
| energy capacity | `1500.0` normalized units |
| minimum landing SOC | `.10` |
| admission/queue reserve SOC | `.25` |
| initial SOC mode | `uniform` in `[.55, .90]` |
| warm start | `30` UAVs, all present at `t=0`, steady initial state |
| final approach geometry | `continuous_inward_v1` |
| release arbitration | `downstream_priority` |
| route-energy guard | enabled |
| late-slot reconciliation | disabled |
| final queue capacity | `3` |
| landing-zone assignment | `nearest_available` |
| emergency trigger SOC | `.40` |
| emergency energy margin | `15.0` seconds |
| safety thresholds | `80 / 40 / 10 m` for safe/NMAC/collision |
| rule flow coordination | `false` for both arms |
| maximum speed residual authority | `2.0 m/s`, unchanged from the learned diagnostic |
| Teacher, labels, optimizer, updates | disabled / absent / zero / zero |

The checkpoint metadata and SHA-256 must be checked against the frozen
diagnostic contract. The only approved checkpoint metadata migration remains
`vehicle_profile_version=eh216s_research_v1` to
`vehicle_profile_version=eh216s_research_v2`; the checkpoint payload is not
rewritten.

## Controller definitions

### `trained`

Use the checkpoint above with deterministic inference, one policy forward per
environment step, and the existing K3 runtime shortlist contract. The actor's
requested gate and proposal are retained exactly in the run records.

### `nominal_rule`

Use no neural policy and no Teacher. For every active UAV that is eligible as
an uncommitted outer-, middle-, or inner-ring actor, request:

```text
gate = RELEASE
proposal = 12  # zero-based V4.2 RELEASE proposal index
v4.1_joint_index = 13
decoded_normalized_action = [1.0, 0.0, 0.0]
speed_residual = 0.0
merge_time_residual = 0.0
```

All other active UAVs request canonical `WAIT`. The nominal request is passed
through the same K3-bounded runtime interface with a one-item shortlist
containing the same canonical RELEASE request. The nominal arm must not use an
empty or dummy action flag. Its records must identify
`controller=nominal_rule` and `neural_policy_used=false`.

The two arms retain `max_speed_residual_mps=2.0`; proposal 12 has zero
normalized speed and timing residuals. The comparison therefore does not
derive the zero-speed contract by enabling `rule_flow_coordination_enabled`.

## Trained-arm counterfactual probes

Only the trained arm records the scheduled neural-request probes. Probe times
are `0`, `60`, `120`, and `240` simulated seconds. At each probe time, sort
eligible active UAV identities by UID and select, for each ring, the first
eligible UAV whose actor request is `WAIT` and the first whose actor request is
`RELEASE`. The maximum is six targets per probe time and twenty-four targets
for the complete run.

For each selected target independently:

1. take a full deep copy of the environment;
2. replace only that target's requested action with the nominal RELEASE
   proposal 12 and its shortlist with a one-item nominal shortlist;
3. execute exactly one `shadow.step()` on the copied environment;
4. retain the shadow decision, release telemetry, action trace, safety error,
   and state digests before and after the shadow step;
5. verify that the original environment state digest is unchanged.

The target is the smallest numeric UID in the ring/gate slot at that same
probe time. The probe is a single-target, same-step counterfactual. An
`approved` result
with no `SafetyInvariantError` is a bounded executable witness for that
target at that state. It is not a global alternative, a future-slot
certificate, a proof about all 25 proposals, or a label for every WAIT.
Safety errors, rejected requests, missing eligibility, and other outcomes are
retained as observations. No labels, optimizer calls, or policy updates are
written by the probe.

## Seed registry and execution order

A read-only search before registration found no existing project record for
`843000`, `843101`, or `random_soc_repair_development_20260924`. These are new
development identifiers. The old formal seeds and `842102` remain untouched;
no prior output is consumed or overwritten.

The paired development comparison uses fresh seed `843101` exactly once per
controller arm. The common seed is intentional: both arms receive the same
arrival, gate, mission, priority, and initial-SOC random draws.

The execution order is fixed:

1. Run a two-step trained smoke with probe time `0` into `smoke/`. Check source
   synchronization hashes, checkpoint identity, output paths, and strict JSON
   serialization before starting either 1800-step arm.
2. Run the trained arm once with seed `843101` into `trained843101/`, with
   probes at `0,60,120,240`.
3. Run the `nominal_rule` arm once with the same seed `843101` into
   `nominal843101/`, preserving the same K3 runtime bound.

The server output root is:

```text
/home/user/training_outputs/terminal/random_soc_repair_development_20260924/
```

The local synchronized mirror is:

```text
D:/7_codespace/training_outputs/terminal/random_soc_repair_development_20260924/
```

The intended command interface is:

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.training.diagnose_trained_dense30_v2 \
  --checkpoint /home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt \
  --output /home/user/training_outputs/terminal/random_soc_repair_development_20260924/smoke \
  --seed 843000 --max-steps 2 --controller trained --probe-times 0

/home/user/envs/terminal_env/bin/python -m terminal_marl.training.diagnose_trained_dense30_v2 \
  --checkpoint /home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt \
  --output /home/user/training_outputs/terminal/random_soc_repair_development_20260924/trained843101 \
  --seed 843101 --max-steps 1800 --controller trained --probe-times 0,60,120,240

/home/user/envs/terminal_env/bin/python -m terminal_marl.training.diagnose_trained_dense30_v2 \
  --checkpoint /home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt \
  --output /home/user/training_outputs/terminal/random_soc_repair_development_20260924/nominal843101 \
  --seed 843101 --max-steps 1800 --controller nominal_rule
```

The command, resolved checkpoint path, exit status, source-sync hash result,
and output file list are retained for each invocation. Source files are
modified only in the local checkout and reach the server through Syncthing;
manual source packaging or transfer is outside this protocol.

## Strict record contract

The smoke record is checked first. Each arm then retains strict JSON records
with finite numeric values or explicit `null`, including:

- protocol/controller/seed/checkpoint identity and source-sync hashes;
- frozen configuration and K3/88D schema markers;
- command, exit status, step and simulated-time counts, terminal state,
  horizon state, and stop reason;
- WAIT/RELEASE requests, decoded proposals, release decisions, vetoes,
  rejection reasons, and per-UAV outcomes;
- normal, emergency, failed-energy, and unresolved counts and identities;
- minimum actual/swept separation and all available named safety counters;
- trained weight digest before/after and unchanged status;
- `diagnostic_only=true`, `training_called=false`, `teacher_labels_used=false`,
  optimizer/update count `0`, `PPO_ALLOWED=false`;
- trained-arm shadow-probe rows with original-state-before/after digests,
  target identity, same-step decision, and `approved_witness`.

Missing counters remain unavailable information and are not filled with zero.
Raw failure records, including failed-energy outcomes, are preserved. An
energy failure is a diagnostic observation and does not by itself discard an
arm or authorize a repair.

## Stop and scope rules

- A source-sync mismatch, checkpoint identity mismatch, malformed output, or
  non-strict JSON record stops the sequence before the paired arms.
- A `SafetyInvariantError` or geometric safety incident stops the current arm
  immediately. The exact error, time, active identities, state digest, and
  safety counters are retained; the primary agent reviews before any next arm.
- The two arms are executed once each. No rerun, seed replacement, consumption
  of `842102`, formal-seed consumption, optimizer update, Teacher call,
  checkpoint rewrite, PPO admission, or qualification claim is part of this
  comparison.
- This document records a controller/execution-feasibility comparison only.
  Decisions about targeted imitation, Safe DAgger, and subsequent curriculum
  work remain with the primary agent after reviewing the raw records.
