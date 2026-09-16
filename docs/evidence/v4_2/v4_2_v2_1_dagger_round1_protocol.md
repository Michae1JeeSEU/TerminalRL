# 88D Safe DAgger Round 1

Preregistered 2026-09-17 following 88D preflight PASS, manifest SHA256
`daa321fb09e73108ef8236d43db1fc10a21d33dabddf2e8fcded659902b2486c`.
Preflight completed two seeds × six steps, all seven gates PASS, zero safety
counts, and exact action/label replay and requested likelihood recomputation.

Authority: `terminal_marl/configs/low_v4_2/i2_v2_1_round1.yaml`, hashed by the
new 88D runner. Historical 87D runs and their FAIL/PASS status are unchanged.

Collect 12 complete 30-step stochastic student episodes, seeds640001–640012,
cycling warm-start N=3,6,8,12. Geometry remains fixed and safety unchanged.
Teacher labels are sidecars, never behavior actions. Retain event start,
target change, phase change, RMS observation change >=.20, or persistence
>=3 steps. No label-conditioned stopping, row duplication or truncation.
Require 500–4,000 new semantic events; outside this range fail before training.
Replay every episode and recompute requested-action likelihoods before training.

D1 is all 7,576 immutable R4 training rows once plus every retained new event
once. Initialize from qualified I1 seed6201; reset Adam, lr .0003, batch64,
35 epochs, clipnorm1, unweighted hierarchical CE, training RNG6401. Each
epoch shuffles once and visits each row once. Freeze final checkpoint before
evaluation; no best-checkpoint selection or retraining based on outcomes.

Evaluate the existing R4 internal audit only as an internal retention guard
after training freeze, using unchanged I1 V2 hard gates. It never trains or
selects a checkpoint and is not independent qualification evidence. Q2 is
not opened. Fresh student-visited validation uses seeds650001–650004, N=3,
6,8,12, 12 steps each. Require both WAIT and RELEASE actor decisions in total,
complete episodes, provenance, exact replay and zero safety violations/errors.

All gates permit I3 student-only qualification only; PPO remains disabled.
I3 will separately assess completion, throughput, queue and shield burden on
longer fresh scenarios before admission. Short DAgger validation does not
establish operational performance.

Smoke uses distinct seeds649001/649002, one UAV × one step, one CE epoch.
It exercises collection, serialization, training, checkpoint and validation;
its small sample may fail performance gates and cannot authorize I3/PPO.
