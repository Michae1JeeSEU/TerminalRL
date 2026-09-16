# I1 Qualification Protocol V2 / fresh Q2

Preregistered 2026-09-17 after I0-R4 PASS, before Q2 resets or fresh actor
training. YAML authority: `terminal_marl/configs/low_v4_2/` files
`i1_v2_1_qualification_protocol_v2.yaml` and `i1_v2_1_q2_cohort.yaml`.
Both hashes are frozen in `run_v4_2_v2_1_i1_v2.py`.

## Training and decision

Use exactly 7,576 R4 training rows; no R4 audit, historical I0/Q, or Q2 rows
enter optimization. Fresh V4.2/88D actors use seeds 6201/6202/6203, canonical
seed 6201 specified in advance. Keep the existing architecture, unweighted
gate CE + RELEASE-conditional proposal CE, Adam lr .0003, batch 64,
100 epochs, gradient norm clip 1.0, final checkpoint only. No resumption,
class weights, duplicated rows, resampling, or model/threshold selection.

Per-seed requirements remain unchanged from V1: WAIT recall >=.90,
RELEASE recall >=.70, gate macro-F1 >=.80, proposal top-1 >=.85,
top-3 >=.95, decoded accuracy >=.85; all stable joint classes 11–14
predicted, finite gradients, each gate prediction share >=.05,
dominant proposal share <.95. All three seeds must pass every gate.
Report mean/std/min/max and the preexisting boundary diagnostics.

## Q2 collection and freeze

Retain Q1's nine effective scenario recipes and semantic sampler; use a
fixed round-robin schedule of five complete cycles (45 episodes), seeds
430001–430045. This was chosen before collection to cover the unchanged
Q1 support minima without label-conditioned stopping. Minimum 1,200 and
maximum 2,400 rows; if a complete episode exceeds the ceiling, retain it
and fail. Do not truncate or discard rows/episodes to make gates pass.

Before first reset, scan historical JSON/JSONL seeds and identities.
Any collision fails without replacement. Freeze requires full schedule,
cardinality, all original Q support minima, 88D/retry provenance,
deterministic Teacher, exact replay for all nine strata, zero safety
violations, historical independence and no opposite-gate exact-observation
conflict. Qualification data has no policy/checkpoint input.

Separate smoke seeds 591001/591002 execute the first two strata for one
step each, validate serialization and replay, and cannot authorize training.
Synthetic-array BC unit tests are implementation checks, not formal I1 runs.

## Single evaluation and failure handling

The pipeline trains only after Q2 freezes. Q2 metadata authorizes the run;
an access guard blocks Q2 data/manifest and R4 audit throughout optimization.
Write and hash all three final checkpoints before opening Q2 rows. An
exclusive persistent attempt marker disallows rerunning training against
the same cohort, and a second marker is created before evaluation reads.
Even a crash during evaluation consumes Q2; no automatic retry is allowed.
If Q2 collection fails, no actor is trained. If any I1 seed fails, no I2 or
PPO starts. Qualification logs retain all seeds and failures.

I1 PASS permits 88D Safe DAgger preflight; PPO still requires student-only
closed-loop qualification and admission. Q2 evaluates warm-start decision
generalization, not natural-arrival operational performance.
