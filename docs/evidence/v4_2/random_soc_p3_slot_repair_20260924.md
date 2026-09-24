# Random-SOC P3 Slot Repair Development Comparison

Protocol: `RANDOM_SOC_P3_SLOT_REPAIR_DEVELOPMENT_V1_20260924`
Registered: 2026-09-24, before consuming a new reset.
Status: bounded development diagnosis only. This is not training, P3 PASS,
qualification, PPO admission, or a claim of energy completion. Keep
`P3_QUALIFIED=false` and `PPO_ALLOWED=false`.

## Purpose and observed trigger

The current P3-profile dense24 diagnostic observed arrival `18` (`UID32`)
with 28 repeated `safety_reject` events. Each of the three recorded
candidates was `dynamics_unreachable_merge_time` (84 candidate observations).
The nominal runs for arrivals `17/18` also contained emergency landings. These
are the bounded observations motivating a slot-reconciliation comparison; they
do not establish a cause or a qualification result.

The comparison uses the existing `reconcile_late_merge_slots_enabled` config
flag. When enabled, the existing execution path rechecks the actual endpoint
time against scheduler gaps and then applies the complete SafetyFilter. It does
not create a new planner, expand the candidate grid, lower a safety threshold,
or guarantee energy feasibility or qualification.

## Frozen checkpoint and P3 runtime

All arms use the frozen checkpoint:

```text
/home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt
SHA256 714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e
```

The dense24 P3 builder values remain fixed: twelve warm-start UAVs with
`[6,4,2]` ring allocation, twelve continuing arrivals at 15-second
intervals, 1800-second horizon, `final_queue_capacity=3`,
`nearest_available` LZ assignment, and `downstream_priority` release
arbitration. Initial SOC is uniform `[.55,.90]`; priorities are `1/3/5`.

The execution contract remains `eh216s_research_v2`, capacity `1500`, landing
floor `.10`, queue reserve `.25`, `phase_motion_v2`, route guard on,
`continuous_inward_v1`, emergency landing/priority guard on with trigger `.40`
and margin `15 s`, 88D V2.1 observations, K3 shortlist, Teacher off,
high-level off, flow coordination off, and safety thresholds `80/40/10 m`.
The only planned runtime comparison change is the late-slot flag.

## Arms and order

The two-step dense24 smoke uses seed `845000`, controller
`trained_release_only`, with late-slot reconciliation enabled. The primary
agent reviews smoke startup, checkpoint/source identity, and strict outputs
before any full arm.

Then run each arm once with the same seed `845101`, in this order:

1. `forced_lateOFF`: `trained_release_only`, flag disabled.
2. `forced_lateON`: `trained_release_only`, flag enabled.
3. `trained_lateON`: `trained`, flag enabled.

The forced arms replace only eligible actor gates with RELEASE and retain the
frozen neural proposal Top3. The trained arm uses the frozen trained policy.
No Teacher labels, optimizer calls, weight updates, PPO sampling, checkpoint
rewrite, or new training data are part of this comparison.

## Output and command contract

Server output root:

```text
/home/user/training_outputs/terminal/random_soc_p3_slot_repair_20260924/
```

Local synchronized mirror:

```text
D:/7_codespace/training_outputs/terminal/random_soc_p3_slot_repair_20260924/
```

Use the existing diagnostic interface with `--scenario dense24`,
`--controller trained|trained_release_only`, and append
`--reconcile-late-merge-slots` only for the ON arms. The default remains off;
the smoke uses `--max-steps 2`, full arms use `--max-steps 1800`, and all use
the fixed checkpoint above.

## Records and stop rules

Retain controller, flag, seed, config/checkpoint/source hashes, requested and
executed decisions, candidate reasons, normal/emergency/failed-energy/
unresolved outcomes, safety counters, separation minima, terminal state, and
strict JSON. Preserve raw safety rejects and failures. The primary agent
reviews the smoke and each completed arm before the next arm. A
`SafetyInvariantError` or geometric safety incident stops the current arm and
the sequence pending review. Do not consume formal or heldout seeds, alter
the P3 protocol, relax gates or thresholds, train, qualify, or authorize PPO.
