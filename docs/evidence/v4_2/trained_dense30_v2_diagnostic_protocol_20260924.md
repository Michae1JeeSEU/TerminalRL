# Trained Dense30 V2 Diagnostic Protocol

Registered 2026-09-24 before any smoke or bounded diagnostic reset. This is a bounded,
diagnostic-only continuation of the low-level curriculum. It does not qualify
the actor, authorize a curriculum promotion, or authorize PPO. Keep
`PPO_ALLOWED=false`.

## Purpose and decision boundary

Run the frozen P2 K3 actor in the dense30 V2 warm-start scenario to describe
what the learned policy requests and what the executable runtime accepts. The
diagnostic answers these descriptive questions:

- What are the actor's WAIT and RELEASE frequencies, overall and by phase or
  ring/interface?
- When RELEASE is requested, which rejection reasons occur, and what are the
  corresponding execution traces?
- How many requests finish as normal, emergency, failed-energy, or unresolved?
- What are the minimum actual and swept separations, and what named safety
  counters are emitted?

These observations may inform a targeted imitation or Safe DAgger decision.
They do not establish zero-failure qualification, rule-controller equivalence,
PPO/P4 authorization, or A6 authorization.

## Frozen checkpoint and schema

The only model input is the trained P2 K3 checkpoint:

```text
/home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt
SHA256 714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e
```

The resolved server path is passed to `--checkpoint PATH` after the primary
agent verifies it. The loader must verify the SHA-256 and the complete
checkpoint identity before inference. The source payload is immutable
historical PPO output; this diagnostic performs no optimizer step and writes no
checkpoint.

The expected deployment schema is V4.2 hierarchical K3 with deterministic
inference and Observation V2.1:

- observation dimension: `88`;
- observation feature hash:
  `8cde32fb4d7481649aad5d01fae0e97c3bb7671c0379d1c4847915493d7beb98`;
- neighbor range: `None`;
- action: V4.2 hierarchical K3;
- Teacher labels/sidecar: absent;
- optimizer/training calls: absent.

The only approved metadata migration is
`vehicle_profile_version=eh216s_research_v1` to
`vehicle_profile_version=eh216s_research_v2`. Every other expected metadata
field must match, including the 88D feature hash above. The historical source
protocol records a `.25` floor; the target runtime explicitly uses
`min_landing_soc=0.10`. The checkpoint payload does not embed that config
floor, so the `.25 -> .10` value is recorded as historical provenance and
target runtime configuration, not inferred from the payload. The existing
global loader remains strict and unchanged. A diagnostic-only loader validates
ALL fields against the `checkpoint_metadata(config)` schema contract with exact
field comparison, allows only the approved `vehicle_profile_version` v1-to-v2
delta, and strictly loads model state without an optimizer.

## Frozen scenario and runtime configuration

Use the existing dense30 `build_config` semantics with exactly these values:

| Field | Frozen value |
|---|---|
| warm start | `30` UAVs, all present at `t=0`, steady initial state |
| initial SOC | uniform in `[0.55, 0.90]` |
| route guard | enabled |
| final approach geometry | `continuous_inward_v1` |
| release arbitration | `downstream_priority` |
| late-slot reconciliation | disabled |
| final queue capacity | `3` |
| landing-zone selection | nearest available LZ |
| emergency trigger | `0.40` |
| emergency energy margin | `15` |
| safety thresholds | `80 / 40 / 10 m` |
| rule flow coordination | `rule_flow_coordination_enabled=false` |
| target runtime energy/profile | `vehicle_profile_version=eh216s_research_v2`, `min_landing_soc=0.10`, `energy_capacity=1500.0`, `soc_reserve=0.25`; `reference_max_flight_time_s=1500.0` is a separate profile field |

The learned actor's nonzero speed and timing proposals must not be mixed with a
rule coordinator. If the run enables rule-flow coordination or injects a
coordinated-rule proposal into the learned run, stop before collecting bounded diagnostic
data and mark the run invalid. The result must not be described as equivalent
to a coordinated-rule regression.

## Seeds, order, and horizon

The output root on the server is:

```text
/home/user/training_outputs/terminal/trained_dense30_v2_diagnostic_20260924/
```

Keep smoke and bounded diagnostic artifacts in separate directories. Use each
seed once; do not reuse a seed, tune against a diagnostic result, or replace a
failed seed.

| Run | Seed | Limit | Role |
|---|---:|---:|---|
| smoke | `842000` | `5` steps | I/O, checkpoint identity, schema, and JSON serialization only |
| diagnostic-1 | `842101` | at most `1800` simulated seconds | fresh diagnostic seed |
| diagnostic-2 | `842102` | at most `1800` simulated seconds | fresh diagnostic seed |

The complete-cycle condition includes service completion. Reaching the time or
step ceiling with unresolved requests is an incomplete horizon record, never a
PASS. The smoke record cannot qualify the actor or authorize any training.

Before the server run, execute the focused loader/runtime tests selected by the
primary agent, verify source-file hash equality between the local checkout and
the synchronized server checkout, and run the smoke command with the actual
checkpoint. Bounded diagnostic seeds may start only after the smoke output passes the I/O,
identity, schema, and strict-JSON checks.

The exact runner interface is:

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.training.diagnose_trained_dense30_v2 --checkpoint /home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt --output /home/user/training_outputs/terminal/trained_dense30_v2_diagnostic_20260924/OUTPUT --seed N --max-steps N
```

The preregistered invocations are:

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.training.diagnose_trained_dense30_v2 --checkpoint /home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt --output /home/user/training_outputs/terminal/trained_dense30_v2_diagnostic_20260924/smoke --seed 842000 --max-steps 5
/home/user/envs/terminal_env/bin/python -m terminal_marl.training.diagnose_trained_dense30_v2 --checkpoint /home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt --output /home/user/training_outputs/terminal/trained_dense30_v2_diagnostic_20260924/seed842101 --seed 842101 --max-steps 1800
/home/user/envs/terminal_env/bin/python -m terminal_marl.training.diagnose_trained_dense30_v2 --checkpoint /home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt --output /home/user/training_outputs/terminal/trained_dense30_v2_diagnostic_20260924/seed842102 --seed 842102 --max-steps 1800
```

Preserve each resolved command and exit status in the run record.

## Required output contract

Each smoke or bounded diagnostic directory must contain a strict JSON run record with
finite numeric values serialized as JSON numbers or explicit `null`, never
`NaN` or `Infinity`. The record must include:

1. **Identity and provenance:** protocol id/date, run kind, seed, resolved
   checkpoint path, checkpoint SHA-256, approved metadata migration, schema
   version/dimension/feature hash, neighbor range, deterministic K3 marker,
   and the frozen configuration above.
2. **Execution bookkeeping:** command, exit status, requested count, resolved
   count, unresolved count, terminal/service-completion flag, actual simulated
   seconds, step count, stop reason, and whether the complete cycle including
   service finished.
3. **Actor requests:** WAIT/RELEASE counts and frequencies, with phase or
   ring/interface breakdown where the runtime exposes it. Record the decoded
   conditional speed and timing proposal for RELEASE requests only.
4. **Release diagnostics:** every observed rejection reason and count, plus
   execution traces for requested, accepted, rejected, and safety-blocked
   RELEASE events. A trace includes the time, UAV identity, phase/interface,
   requested action, proposal, runtime decision, and applicable reason/counter.
5. **Terminal outcomes:** normal, emergency, failed-energy, and unresolved
   counts, with the per-UAV terminal outcome or unresolved identity retained.
6. **Safety:** minimum actual separation, minimum swept separation, and the
   runtime's named counters, including collisions, NMAC, hard separation
   violations, uncertified execution, `SafetyInvariantError`, safety
   interventions, shield fail-closed, and joint-safety fail-closed when
   present. A missing counter is unavailable information and must not be
   silently written as zero.
7. **Integrity:** `diagnostic_only=true`, `training_called=false`,
   `teacher_labels_used=false`, optimizer-step count `0`,
   `PPO_ALLOWED=false`, and a weight/checkpoint identity check showing that
   the loaded weights were unchanged.

WAIT is a request category and contributes to the gate frequency. Do not label
a WAIT as `false_wait` unless the same record has an independently retained
feasibility witness for an executable RELEASE under the frozen safety/runtime
contract. Otherwise retain it as `wait_unclassified`.

## Stop and failure rules

- A checkpoint hash mismatch, an unapproved metadata/schema mismatch, a source
  sync mismatch, a failed smoke check, or an unexpected rule-flow coordinator
  stops the protocol before bounded diagnostic execution.
- On `SafetyInvariantError` or any safety incident, stop the current episode
  immediately, retain the exact error, time, active identities, diagnostic
  payload, and safety counters, and skip the remaining diagnostic seed(s) in
  the cohort pending primary-agent assessment. Do not resume or rerun the
  failing or skipped seed(s).
- Any safety incident or early stop leaves the corresponding time/horizon and
  completion fields incomplete. It cannot be marked PASS by filling missing
  requests or by changing the horizon.
- A process or serialization failure produces a failed/incomplete record and
  does not authorize a replacement seed.

## Diagnostic usability and interpretation

The primary agent may mark an artifact usable for diagnosis only when its
identity/schema/configuration, no-Teacher/no-training bookkeeping, strict JSON
serialization, seed bookkeeping, and unchanged-weight check are all directly
verified. Zero failed-energy outcomes, zero unresolved requests, zero safety
events, or any other qualification-level result is not an automatic gate for
this diagnostic and does not promote the curriculum.

The bounded diagnostic artifacts are descriptive evidence for the next targeted imitation
or Safe DAgger decision. They do not authorize PPO, P4, or A6.
