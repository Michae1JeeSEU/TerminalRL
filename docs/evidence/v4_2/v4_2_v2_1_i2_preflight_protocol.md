# 88D Safe DAgger preflight V1

Preregistered 2026-09-17 after I1 V2 PASS. This replaces neither historical
87D preflight evidence nor its protocol. The input is the canonical seed6201
checkpoint from I1 V2, SHA256
`f1a5dce05657173d97c402da6d28fa47b59bf016bd61304b3d5fb84f23c7b172`.
Its qualification manifest SHA256 is
`0e3817ad4fc928f20ba8f6cff7cf7b61801507c557e62daa53aa845039cf87cc`.

The protocol YAML is `terminal_marl/configs/low_v4_2/i2_v2_1_preflight.yaml`.
Run two fresh seeds 630001/630002, six steps each, eight warm-start UAVs,
fixed geometry, V4.2/88D observations and unchanged safety. Check historical
seed independence before reset. The student samples requested gate/proposal;
committed/nondecision agents use masked canonical WAIT. Teacher produces
deterministic sidecar labels only and cannot replace actions. No training.

Require qualified input lineage, complete rollouts, finite requested-action
likelihoods, unique decision IDs, exact replay including labels/execution,
direct recomputation of requested-action log probabilities, and zero safety
counters/errors. This engineering gate permits a separately preregistered
bounded DAgger round; it cannot establish throughput or PPO admission.

Before formal execution, verify the full path using seed639001, one UAV,
one step. Smoke cannot authorize a DAgger round. Q2 remains consumed and is
not opened, selected against, or used for training by this stage.
