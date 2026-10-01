# P3 prefix-feasible-gate replay protocol

**Status: pre-registered replay scope; not launched.** This document records the fixed comparison approved for the next P3 review. It does not authorize training, PPO admission, checkpoint promotion, default runtime changes, or a claim that P3 is solved; `PPO_ALLOWED=false`.

## Existing evidence boundary

The four previously evaluated trained gate-value states all had `C=0`. The fresh six-scene screen failed its recorded screen: `AP=132N, 12E` and `BP=134N, 10E`, with `F=D=U=0` for both labels. The three recorded AP-versus-BP cost differences are `+2.9843324913833236%`, `+0.5281622267627961%`, and `+0.7351975729166904%`. These values are retained as input evidence. Any business difference from the prefix replay is execution-comparison evidence, not evidence of a training repair.

## Fixed inputs and scope

The only new phase replays the fresh V4 `BEFORE-prefix` tapes for fixed cells 1–4. The full input binding is:

`30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195`

No new trajectory, policy population, or seed is created, and no other seed is used. The output root is:

`/home/user/training_outputs/terminal/p3_followup_20260930/prefix_feasible_gate_20261001`

Cell selection is mechanical. `action_tape.runtime_shortlist_result` with `used=true` and `rank > 3`, together with the sidecar registration emitted by the native sampler for an `actor_mask=true` and `requested_gate=1` request, identifies only a potential candidate key; the tape need not expose `actor_mask`. The sidecar is no-environment provenance, while the actual mask is validated during replay. These keys must not be taken from labelled `pre_state.release_state` or historical `decision_history`. For each potential key, save the pre-tick world, execute the original tape action exactly once, and confirm the post-tick `env.last_release_decisions[uid] == "approved"` with replayed rank still `> 3`. The confirmed event is eligible only when the saved pre-state is `phase=ring_inner`, has `0.4 < SOC <= 0.65`, has `actor_mask=true`, has no retry block with `now >= safety_retry_after`, and has `committed_plan=None`. Replay the original order and select the first event satisfying all predicates, ordered by `(step, numeric UID)`; offline keys are not filtered by phase or SOC. Record counts and reasons for events excluded by these physical predicates, and do not compute `p` or `D` for them. Only after confirmation, compute both models' gate probability and all-25 proposal vector on the saved pre-world; do not execute the original action again. Selection cannot use gate change, reward, terminal outcome, or any other performance value. If no actual confirmed event exists, the cell remains missing; it is not replaced by another state or budget expansion. Missing mask or SOC values are read from the replay observation and remain unknown when unavailable.

The original tape is replayed exactly once for each cell over its full cycle, with at most 1,800 recorded ticks per cell and at most 7,200 existing ticks in total. The preflight performs no environment creation, model forward, gradient operation, or optimizer operation.

## Pilot and full phases

After serialized-output and test review, the pilot may run cell 1 only. Root review is required after the pilot. FULL, if separately authorized, reuses the immutable pilot result and replays cells 2–4; it is never started automatically.

At the confirmed event, the frozen `BEFORE=3fe7` and `AFTER=d449` policies are evaluated with the same saved 88D pre-tick world observation under `torch.no_grad`. The phase records the actual replay mask and post-action approval, gate probability and its difference, and all 25 proposal probabilities. The proposal vector `q` must be bitwise identical between the two model evaluations. The original tape action and shortlist have already been executed once for replay; they are not executed again. A small NPZ stores the selected observation, actual mask, post-action approval, and measured outputs.

The phase does not infer a long-horizon return or a local `m` when the replay does not expose it: `D` and `local_m` remain `None`. Local probability differences are not summed and reported as a global gain. Four selected cells do not support a causal or population claim.

## Required records and stop rules

Every error, truncation, business failure, and unknown value is retained. A technical, model, RNG, environment, safety, or exact-replay error writes the available partial result and stops the phase without automatic restart. A technical pass is not equivalent to resolving P3.

Each serialized phase result records the fixed cell/tape binding, selected `(step, UID)`, source model and checkpoint hashes, 88D observation availability, mask/SOC availability, gate probabilities, the full 25-way `q`, bitwise-`q` comparison, original action/shortlist replay checks, RNG/config checks, and unknown fields. The phase does not introduce a new latency gate, business acceptance gate, or comprehensive provenance system.

The historical batch-04 training protocol and metadata agree on `gamma=lambda=1` and team `full_episode_mc` credit. Normal, emergency, and failed signals pass through the scalar team reward. This replay does not treat the absence of an explicit energy reward term or team credit as a root-cause finding.

No optimizer, Teacher, PPO, held-out data, formal default promotion, physics change, retry change, safety change, or public-K3/prefix-25 change is permitted.
