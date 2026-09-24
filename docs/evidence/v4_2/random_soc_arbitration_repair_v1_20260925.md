# Random-SOC Arbitration Repair V1 Development Protocol

Status: registered before any `847xxx` reset. This is a development comparison
only; it is not a P3 qualification, checkpoint promotion, or PPO admission
protocol.

## Purpose and frozen checkpoint

This comparison tests the already fitted gate-adapter checkpoint under the
existing dense24 execution path while changing only the release arbitration
mode. Both arms use the identical candidate checkpoint:

`/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt`

SHA256:
`e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`

The dense24 baseline uses `release_arbitration_mode=downstream_priority`.
The variant uses `release_arbitration_mode=downstream_soc_priority`. This
field is the only configuration difference. `flow6` and `flow12` are not
changed by this protocol.

## Frozen scenario and execution semantics

Both dense24 arms keep late-slot OFF, K3, final capacity `3`, initial SOC
uniform `.55–.90`, priorities `1/3/5`, energy capacity `1500`,
`min_landing_soc=.10`, `soc_reserve=.25`, emergency threshold `.40`, route
guard enabled, and the `80/40/10 m` geometry. Teacher execution, labels,
optimizer updates, and PPO are disabled. The proposal head and encoder remain
frozen. The current `diagnostic.build_scenario_config("dense24", reconcile_late_merge_slots=False)`
V2 execution config is used; this refers to the current builder output rather
than an older historical config hash. No shortened horizon or hidden config
copy is allowed.

Before any reset, the runner must assert that the two arm configs differ only
in `release_arbitration_mode`, and the small-sample output parser and source
sync check must pass. Outputs go below
`/home/user/training_outputs/terminal/random_soc_arbitration_repair_v1_20260925/`;
the project source directory is not an output location.

## Seeds, bounds, and stop rules

The fresh smoke uses seed `847000`, with one baseline arm and one variant arm,
each limited to `2` steps while still binding the complete configuration.
After smoke review, paired dense24 runs use seeds `847101` and `847102`; each
arm runs to terminal or the existing `1800 s` horizon. Every outcome,
including emergency and failed-energy outcomes, is retained. A failed run is
not deleted, rerun, or replaced with another seed. A `SafetyInvariantError`
stops subsequent arms and preserves the artifacts already produced. Ordinary
performance-gate failure does not cancel the preregistered pair.

## Gates and evidence boundary

Use the existing P3 hard gates, exact replay, requested-likelihood, K3
shortlist, geometry, and source/candidate lifecycle selection checks without
changing thresholds. Report the raw arm outcomes and all gate fields. This
protocol does not alter `P3_QUALIFIED`, `PPO_ALLOWED`, or `P4_ALLOWED`, and
held-out seeds `829001–829006` remain untouched.

Both arms are repair candidates. The strict wait comparison is a report of
the arbitration component tradeoff only; it is not a new P3 promotion gate
and does not replace the original source checkpoint
`714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e`. If the
repair is later considered for continuation, a separate fresh protocol must
compare that original source with the final combined candidate under the
unchanged P3 safety, complete-normal-outcome, priority-wait-improvement, and
low-SOC-wait-nonworse requirements. This document does not auto-adopt either
arm or declare qualification.

The preceding exact emergency replay has `795` steps. The current `32`
independent single-step shadow checks report unchanged state, no safety error,
and zero approved witnesses: `safety_wait=16`, `final_capacity=10`, and
`safety_reject=6`. Those observations only record that the sampled checks did
not obtain approval; they do not establish that all WAIT decisions were
correct or incorrect, or that SOC-priority arbitration will be effective.

Execution of the registered smoke and paired runs proceeds after
implementation, preflight, and primary review. No training or qualification
is authorized by this protocol.
