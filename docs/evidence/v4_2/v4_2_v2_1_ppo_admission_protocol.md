# PPO initial-curriculum admission and entry validation

Preregistered 2026-09-17 after I3 PASS. I3 manifest SHA256:
`e4350050629207045d880f4cc9b82347daf08771985397665a7e72a12cc16c25`.
All six independent student-only episodes completed all 24 requested UAVs,
ended with drained queues, zero safety counts and zero actor shield usage.
Completion times: warm3 264/233 s, flow3 515/472 s, flow6 867/807 s
(deterministic/stochastic). All action/state/likelihood replays passed.
The deterministic flow6 run has only 33 s of deadline margin; larger loads
and shorter headways remain outside this admission.

The review validates the exact R4 → I1 V2 → DAgger R1 → I3 artifact chain,
schema, student-only behavior, safety, completion and queue/throughput,
shield burden and immutable canonical checkpoint. Review PASS authorizes
only an entry validation, not a formal PPO block. Source hashes bind the
entry, rollout and hierarchical PPO modules; changes invalidate admission.

Authority: `terminal_marl/configs/low_v4_2/ppo_v2_1_initial_entry.yaml`.
Entry validation uses fresh seeds670001/670002, warm3/flow3, 32 steps each,
stochastic student requested actions, no Teacher and no high-level control.
Before each update require exact replay and recomputed requested logprob
(initial ratio one), finite correct on-policy transitions and zero safety.
Use Adam3e-5, two epochs, minibatch32, gamma.99, GAE.95, clip.1, entropy.001,
value coefficient.5. Require finite updates/parameters, approximate KL<=.05,
checkpoint save/load equality and successful continuation from restored
optimizer. Entry-test checkpoints cannot initialize formal training.

Only successful review plus entry validation writes PPO_ALLOWED=true.
The formal entry always starts from qualified DAgger checkpoint
`a97e0284b07b18404f60cd666c72806cbf5e261bf807c7f9f3a1d897c69b1584`.
The registered first block is at most 12 episodes, fresh seeds starting680001,
cycling warm3/flow3/flow6, at most900 s each. Safety, completion, shield,
finite-update or KL failure stops the block and preserves its evidence.
No verification rollout is reused in PPO optimization outside its entry test.

PPO ratios use only the sampled requested gate and RELEASE-conditional
proposal. Committed/nondecision agents are excluded from the actor loss;
their state remains critic context. Team reward/critic are counted once per
world step, including empty arrival/service intervals. Time limits bootstrap
but never propagate GAE across reset. Neither Teacher nor shield-applied
actions replace the behavior action used for the ratio.

This grants initial fixed-geometry N3–N6 curriculum training only. Full
static-geometry qualification, high-level dynamic airspace and hardware
validation remain separate downstream gates.
