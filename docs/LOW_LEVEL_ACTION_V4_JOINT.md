# Low-Level Action V4: joint discrete scheduling intention

V4 replaces factorized V3 supervision. The policy emits one categorical
joint-action index, the Teacher supervises one joint-action index, and vehicle
execution remains the existing normalized intent / trajectory / SafetyFilter
pipeline.

## Catalogue

`low_action_schema = V4_joint_discrete`, `N_joint = 30`.

| Class | Count | Decode |
|---|---:|---|
| WAIT | 5 | `[WAIT, speed∈{-1,-.5,0,.5,1}, merge=0]` |
| RELEASE | 25 | `[RELEASE, speed×merge]` |

WAIT speed is retained because it affects holder/runtime-shield semantics.
WAIT merge is canonicalized to zero because it has no proposal meaning.

## Provenance

PPO stores only `a_requested_joint`, its `joint_logprob`, and the resulting
PPO ratio. Teacher, execution and shield actions never enter that ratio. BC is
one categorical CE against `a_teacher_joint`; component disagreement is only a
diagnostic.

TeacherScheduler may enumerate the bounded 30-action catalogue during
training, using canonical proposal/SafetyFilter verification. ExecutionScheduler
checks and executes only the one decoded policy request at runtime; it does not
search the catalogue.

ActionTape records joint index, decoded action, teacher/execution/shield action,
joint log-probability, observation hash and post-state hash. V3 traces remain
read-only historical artifacts.

## Expert versus repair semantics

`teacher_joint_label()` is the state-conditioned V4 expert used by joint CE:
for a frozen full state it evaluates the same ordered 30-action catalogue and
does not accept the policy request as an input.  Runtime execution remains a
separate one-request certification path; its applied/repair outcome is an
intervention diagnostic and never a joint-CE target or PPO-ratio action.

The offline `run_v4_teacher_expert_repair_semantics_diagnostic.py` performs a
same-state × 30-request audit, including candidate feasibility and the
release-first deterministic tie-break.  A PASS additionally requires an
explainable natural training/validation expert-label distribution and a
non-collapsed CE-only probe.  A semantics PASS is required before any fresh
V4 L1 PPO rerun.

## V4 L0 and L1 preflight

`run_v4_l0_teacher_dataset.py` creates a fresh ≈3,600-row V4-only dataset
from Full Teacher catalogue queries.  It retains the frozen nominal low and
medium, warm-start and from-empty L0 curriculum, but never converts a V3
per-head label.  Every record contains Observation V2, requested and Teacher
joint indices plus decoded actions, joint/component disagreement diagnostics,
scenario/state metadata, and `low_action_schema = V4_joint_discrete`.

The collector reports all 30 requested/Teacher/correction frequencies and the
requested→Teacher matrix without relabelling or artificial Teacher balancing.
A compact CE-only bootstrap is a diagnostic: it calls no PPO update and saves
no training checkpoint. Any collision, NMAC, hard separation,
`SafetyInvariantError`, or uncertified component writes an exact V4 ActionTape
and blocks qualification.

`run_v4_l1_preflight.py` is a 64--128-row smoke test only. It validates
Categorical(30) sampling, requested joint log-probability, entropy, finite
joint-CE auxiliary calculation, external Teacher intervention comparison,
one-proposal execution, safety, exact ActionTape replay, checkpoint
save/load, and deterministic validation. It is not a PPO run and cannot
authorize L1b/L2 or a long L1 job.

`run_v4_l1_short_pilot.py` is the separately authorized six-update
Full-Teacher PPO diagnostic: a fresh V4 actor/critic, 4 normal-L1 episodes per
update, 600-second horizon, fixed natural/warm/held-out-anchor validation at
baseline/update3/update6, and complete V4 checkpoints. It stops on a safety
event, non-finite PPO health, or the specified sustained conservative-WAIT
collapse. It records the base commit and working-tree diff digest because the
current repository ACL prevents a commit; it does not authorize formal L1,
L1b, L2, reward/BC/hyperparameter changes, or a V3 checkpoint resume.
