# Random-SOC P3 Repair Development Comparison

Protocol: `RANDOM_SOC_P3_REPAIR_DEVELOPMENT_COMPARISON_V1_20260924`
Registered: 2026-09-24, before consuming a new reset.
Status: development diagnosis only. This record is not P3 training, P3
qualification, a P3 PASS, curriculum promotion, PPO admission, or a claim
that a repair is sufficient. Keep `PPO_ALLOWED=false`.

## Purpose and frozen boundary

This comparison uses the actual P3 V4 scenario construction to separate
controller requests from execution feasibility under random initial SOC. It
uses fresh paired seeds and the same frozen runtime configuration for every
controller in a scenario. The comparison does not modify the checkpoint,
proposal weights, proposal logits, optimizer, Teacher, or heldout evidence.

The trained controller is the frozen P2 K3 checkpoint already used by the
diagnostic comparison:

```text
/home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt
SHA256 714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e
```

The `nominal_rule` arm is an explicit execution-request reference. For each
eligible uncommitted ring actor it requests canonical RELEASE proposal `12`
(zero-based V4.2 proposal index; V4.1 joint index `13`; decoded normalized
action `[1, 0, 0]`; zero speed and timing residual). Other active actors
request canonical WAIT. It uses the same K3 runtime interface with a one-item
shortlist, does not enable the fixed-rule flow coordinator, and makes no neural
behavior claim. The request is not an empty or dummy action.

The `trained_release_only` arm is used only in dense24. It retains the frozen
trained checkpoint, neural proposal logits, and neural proposal Top3, while
replacing the eligible actor gate with RELEASE. It does not replace the
proposal or alter policy weights. Gate replacement is recorded as an
ablation, not as PPO sampling or a training operation.

## Frozen P3 V4 runtime

The comparison follows the existing P3 V4 schedule and priority/SOC/reward
values:

| item | frozen value |
|---|---|
| priority | `random_135`, values `1/3/5` |
| initial SOC | uniform `[0.55, 0.90]` |
| mission probabilities | `[0.45, 0.25, 0.10, 0.20]` |
| priority time weight | `1.0` |
| SOC urgency weight/threshold | `2.0` / `0.65` |
| profile / capacity | `eh216s_research_v2` / `1500` |
| landing floor / queue reserve | `.10` / `.25` |
| motion / route | `phase_motion_v2` / route guard on |
| final approach / late slot | `continuous_inward_v1` / reconciliation off |
| emergency guard | landing guard on; trigger `.40`, margin `15 s` |
| observation/action | V2.1 `88D` / V4.2 hierarchical K3 |
| Teacher/high-level/updates | off / off / zero |
| safety thresholds | `80/40/10 m`; minimum separation unchanged |
| LZ/release arbitration | `flow6/flow12`: `lowest_id` / `legacy_insertion`; `dense24`: `nearest_available` / `downstream_priority` |
| warm-start reachability check | `dense24=true`; `flow6/flow12=false` (not applicable; no warm start) |
| rule flow coordination | `rule_flow_coordination_enabled=false` |
| speed residual authority | `2.0 m/s`, unchanged |

Scenario schedules and horizons are the P3 V4 values. `flow6` is
`flow_from_empty`, six UAVs, first arrival `10 s`, interval `60 s`, horizon
`1210 s`, final capacity `1`. `flow12` is `flow_from_empty`, twelve UAVs,
first arrival `10 s`, interval `60 s`, horizon `1570 s`, final capacity `1`.
`dense24` is `warm_start_rings`, twelve warm UAVs with ring allocation
`[6,4,2]`, twelve continuing arrivals at `15 s`, horizon `1800 s`, and final
capacity `3`.

## Seed registry and arm order

A read-only exact-token search found no existing use of `844000`, `844101`,
`844102`, or `844103` in the project docs/code/tests or accessible training
output text. The new output root was absent at registration. Existing
`843101` development evidence, P3 formal/heldout seeds, and the old
`829001–829006` heldout records remain untouched and are not repurposed.

The server output root is:

```text
/home/user/training_outputs/terminal/random_soc_p3_repair_development_20260924/
```

The local synchronized mirror is:

```text
D:/7_codespace/training_outputs/terminal/random_soc_p3_repair_development_20260924/
```

Run the frozen `dense24` trained smoke with seed `844000` for two steps first.
The primary agent reviews its startup, checkpoint identity, strict JSON, and
source synchronization before any full arm. Then execute each paired scenario
once in this order: `flow6` seed `844101` trained and nominal_rule; `flow12`
seed `844102` trained and nominal_rule; `dense24` seed `844103` trained,
nominal_rule, and trained_release_only. The same seed is intentional within
each pair. All horizons and per-arm time limits remain those of P3 V4.

The command interface is:

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.training.diagnose_trained_dense30_v2 \
  --checkpoint /home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt \
  --output <output-dir> --seed <seed> --max-steps <steps> \
  --scenario flow6|flow12|dense24 --controller trained|nominal_rule|trained_release_only
```

## Record and stop rules

Each record retains controller identity, seed, scenario, frozen configuration,
checkpoint/source hashes, requested gate/proposal, K3 shortlist, release and
emergency decisions, energy outcomes, safety counters, minimum actual/swept
separation, terminal state, and strict finite JSON (`null` for unavailable
values). `trained_release_only` retains the neural proposal Top3 and records
the gate override explicitly. No labels, optimizer calls, weight updates,
PPO sampling, checkpoint rewrite, or qualification result is written.

Raw emergency and failed-energy observations are retained. A
`SafetyInvariantError` or geometric safety incident stops the current arm;
the exact error, simulated time, active identities, state digest, and safety
counters are retained, and the primary agent reviews before the next arm.
The primary agent also reviews the smoke before the first full pair and each
completed pair before proceeding. No minimum-separation threshold is relaxed.

This document records a bounded diagnosis to support later targeted repair
review. It does not alter the existing P3 protocol, consume heldout evidence,
or authorize training, qualification, or PPO.
