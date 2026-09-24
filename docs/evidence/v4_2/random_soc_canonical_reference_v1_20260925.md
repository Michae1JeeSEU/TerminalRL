# Random-SOC Canonical Reference V1 Development Protocol

Status: registered. The loader implementation is commit `29a4dba` and its `32`
tests passed; the evaluator's `27` tests passed, and the read-only server
checkpoint/hash/profile-load preflight passed. The user requested a pause on
2026-09-25 after the current code iteration. No `848000/848101/848102` reset or
rollout has occurred, and no held-out seed has been consumed. This is a
development reference comparison before further training. It does not change
qualification thresholds, consume held-out seeds, promote a model, or
authorize PPO.

## Purpose and frozen checkpoints

The comparison separates improvement over the historical P3 canonical initial
from improvement over the current repair source. The two frozen checkpoints
are:

Canonical initial:
`/home/user/training_outputs/terminal/p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt`

SHA256:
`4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`

Repair candidate:
`/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt`

SHA256:
`e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`

The verified common tensors are six `actor_encoder` tensors, two
`proposal_head` tensors, and the scalar `gate_temperature`; these are bitwise
identical between the checkpoints. The verified temperature is `2.5`. This
statement does not claim that all non-gate parameters are identical.

## Frozen runtime configuration and metadata checks

Both arms use the current
`diagnostic.build_scenario_config("dense24", reconcile_late_merge_slots=False)`
configuration with `release_arbitration_mode=downstream_priority`. All runtime
configuration fields must match. The retained settings are 80/40/10 m,
final capacity `3`, initial SOC uniform `.55–.90`, priorities `1/3/5`, energy
capacity `1500`, `min_landing_soc=.10`, `soc_reserve=.25`, emergency trigger
`.40`, route guard enabled, 88D observations, and K3 shortlist semantics.
Teacher execution, labels, optimizer updates, and PPO are disabled.

Canonical checkpoint metadata may differ only by the verified
`vehicle_profile_version` migration from `eh216s_research_v1` to
`eh216s_research_v2`. Source files and weights are not modified. Checkpoint
SHA256 values and every other metadata field must be verified strictly. The
evaluator's `27` tests and the read-only checkpoint/hash/profile-load preflight
have passed; no run starts during the requested pause.

## Seeds, outputs, and stop rules

The fresh smoke uses seed `848000`, with one arm per checkpoint and `2` steps
each while binding the complete `1800 s` horizon configuration. After smoke
review, full dense24 pairs use seeds `848101` and `848102`; each arm runs to
terminal or the `1800 s` upper bound. All outcomes are retained, including
emergency and failed-energy outcomes. A safety or software failure stops the
remaining registered arms and preserves produced artifacts. A performance
gate failure does not cancel the registered pair. Failed runs are not deleted,
rerun, or replaced, and no additional seed is added.

Output root:
`/home/user/training_outputs/terminal/random_soc_canonical_reference_v1_20260925/`

## Gates and evidence boundary

Use the existing P3 hard gates together with exact replay, requested
likelihood, geometry, low-SOC wait, and priority-weighted wait checks. Report
raw arm outcomes, all gate fields, checkpoint bindings, and metadata checks.
This comparison is development evidence only: it does not automatically
qualify either checkpoint, replace the current P3 protocol, promote a model,
or authorize PPO. Historical seeds `829001–829006` remain untouched. The
comparison informs a later training starting point without changing thresholds
or selecting a model after consuming held-out evidence.

On resume, execute the registered canonical smoke and full pair after the
primary review; then decide separately whether to repair and independently
qualify. No new algorithm experiment is authorized during the pause.
