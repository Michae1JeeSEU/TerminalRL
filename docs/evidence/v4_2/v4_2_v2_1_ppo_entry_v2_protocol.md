# PPO entry V2: natural-arrival timing correction

Preregistered 2026-09-17 before new verification resets. V1 remains FAIL:
`v4_2_ppo_entry_verify_20260917/admission.json` SHA256
`559b19c7cc3b607d7143da00762468fedd821c2cfa1ddec13376b5814940415e`.
Seed670001 warm3 passed safety/replay, finite PPO update (KL~1.99e-8),
checkpoint save/load. Seed670002 flow3 had zero actor decisions in its first
32 s, so no second update was performed and PPO_ALLOWED stayed false.

Diagnosis: natural arrivals include an approach interval before entering a
ring where the low policy can decide. The already-completed stochastic I3
flow3 tape first requested RELEASE at step68. A 32 s flow rollout therefore
does not exercise an actor update, despite correctly advancing empty and
approach intervals. Do not unmask approach/committed states to force an update.

New authority: `terminal_marl/configs/low_v4_2/ppo_v2_1_initial_entry_v2.yaml`.
Fresh seeds671001/671002 use warm3 for32steps and flow3 for128steps. Initialize
again from the qualified DAgger checkpoint, never from V1 test output. Keep
all optimizer parameters, qualification evidence, admission scope and hard
gates unchanged. I3 and Q2 are not rerun. The longer window is an engineering
input-duration correction, not a relaxation of actor/provenance requirements.

The old YAML and failure artifact remain immutable. Review and entry results
use fresh `v4_2_ppo_review_v2_20260917` and `v4_2_ppo_entry_verify_v2_20260917`
directories. A new source-bound admission is required after this code change.
