# P2 V4 held-out evaluation protocol

Status: preregistered evaluator; no formal evaluation result is claimed here.
This document freezes the evidence boundary for a once-only, paired comparison
of the formal P2 V4 student checkpoint, the frozen baseline, and the nominal
rule controller.

## Frozen inputs and lineage

The evaluation protocol is
`terminal_marl/configs/low_v4_2/ppo_p2_v4_evaluation.yaml`.
Its SHA256 is:

```text
b3666bfee8a6c21002cab9088437f4d061b81128a46bab3df02ab83ba235ac81
```

The candidate is the formal V4 P2 checkpoint `checkpoint_8.pt`:

```text
checkpoint SHA256:          5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f
stage:                      V4_2_P2_FORMAL_TRAIN
formal:                     true
training manifest SHA256:   73502ed0f956e517c47dd14ea6bfcc411cb927eba43c4a409fdee119fa7c82f0
training manifest stage:    train
training manifest records:  8
training protocol SHA256:   01c92c44aecc50035e76b89a4e8d330fac5cfe5124b79d593c92a69260154ebf
```

The candidate manifest is accepted only when it is `stage: train`, `PASS: true`,
formal, has exactly eight records, has all record gates true, has source hashes
equal to `p2_source_hashes()`, and names `checkpoint_8.pt` as its final
checkpoint.

The paired baseline is frozen as:

```text
checkpoint SHA256: 6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f
stage:             V4_2_V2_1_PPO
formal:            true
```

The runner is
`terminal_marl/training/evaluate_v4_2_p2_v4.py`, whose current SHA256 is:

```text
05b9502f3f4ddc83cf7fe718ffefa3efebb3815ec067a8bb7acc6d1e81e89a12
```

The evaluator binds the V4 training source snapshot through
`evaluation_source_hashes()`:

| source | SHA256 |
|---|---|
| `terminal_marl/training/run_v4_2_ppo_p2_resume_v4.py` | `bb9dfcf15d6c74385e8b1d6ea3e47e527f9c83c4d67840f275335e5ac6b337b6` |
| `terminal_marl/configs/low_v4_2/ppo_p2_resume_v4.yaml` | `01c92c44aecc50035e76b89a4e8d330fac5cfe5124b79d593c92a69260154ebf` |
| `terminal_marl/algo/student_ppo_v4_2.py` | `bd590f907960de26d5290121f816e1cd428b20e2981705a2a6a5327b5b179301` |
| `terminal_marl/algo/joint_ppo_v4_2.py` | `abc2f3678b68de7808328ffdeccc5fa7f18d6c21527a38627fdcfc38fd173490` |
| `terminal_marl/core/checkpoints.py` | `2fc41879c256e86b04a2702aacb5a9cd0a02b9ae549bd10faf508dc82acec686` |
| `terminal_marl/core/trajectory_generator.py` | `8cb448b41893cb788cf013cc941d70b04f82102680b87384ce19dadb21bba0aa` |
| `terminal_marl/core/trajectory_time_parameterization.py` | `03517565d617a466d34b50ab9ad4879cc5e512ec2826f7ddac87ba55486a2850` |
| `terminal_marl/envs/parallel_env.py` | `8ea8b6fcf8f111c7c07610440da090d7c208bedccb734a5c6cfbaa647429e6cf` |
| `terminal_marl/training/student_rollout_v4_2.py` | `d7a91e38f2af95fe75e81e5f0686ceb08f109176210a19c24ca7c7fa2a537573` |
| `terminal_marl/training/run_v4_2_v2_1_student_qualification.py` | `a11cc3f3a3f773a00348864d52852ce16d27520bc1aca9abe76c8a768c59b59b` |
| `terminal_marl/training/run_v4_2_v2_1_ppo.py` | `fa6476faaccaeafd683316890a27fbee553dfefe41b3f5e44695e7c087b86d61` |
| `terminal_marl/training/run_v4_2_ppo_block2.py` | `b59cba2228b54a4fc3fa9e21aa2a75679033506563b23838827a90e4eb02c4ed` |
| evaluation runner | `05b9502f3f4ddc83cf7fe718ffefa3efebb3815ec067a8bb7acc6d1e81e89a12` |
| evaluation YAML | `b3666bfee8a6c21002cab9088437f4d061b81128a46bab3df02ab83ba235ac81` |

Each learned controller is restored with `StudentPPOV42` and
`load_checkpoint(..., load_optimizer=False)`. Checkpoint bytes and semantic
payload lineage are checked before any rollout. Evaluation is deterministic,
does not call PPO updates, disables Teacher substitution and label collection,
and keeps the rule controller policy-free.

## Cohorts and scenario order

The formal held-out cohort is exactly six seeds:

```text
755001, 755002, 755003, 755004, 755005, 755006
```

The scenario cycle is:

```text
flow6, flow8, flow12, flow6, flow8, flow12
```

Every seed is evaluated for each controller:
`candidate`, `baseline`, and `rule`, producing 18 records. The formal cohort
is checked against the historical evidence registry before any simulator reset;
historically consumed seeds are rejected without replacement.

The smoke probe is deliberately separate and must run before any formal
evaluation if a preflight is desired:

```text
seed:     770000
scenario: flow6
steps:    160
```

Smoke never checks or reserves held-out seeds and never creates
`p2_v4_heldout_consumed.json`. It is diagnostic only and cannot produce
`PASS: true`.

## Hard gates

Each of the 18 formal records must satisfy the lifecycle and execution gates
for all three controllers:

- all requests touch down and the episode reaches terminal state;
- all safety counters are zero;
- action-tape replay is exact;
- ring-transfer geometry has no reverse samples and no inner excursion above
  0.05 m.

The two learned controllers additionally require exact requested-action
likelihood replay and an actor shield fraction no greater than 0.05. The rule
controller requires zero shield interventions. All controllers are deterministic
and no controller may perform an update.

Formal `PASS` is possible only when there are exactly 18 records, six records
for each controller, the prescribed seed/scenario shape is present, and every
record gate is true. A smoke manifest remains non-PASS even if fabricated or
diagnostic records happen to satisfy their per-record gates.

## Metrics and paired comparison

Each record stores requests, completed count, terminal status, elapsed seconds,
active and queue peaks, safety counters, shield counts, weighted wait seconds,
and request-to-touchdown seconds. Controller aggregates report completed,
completion fraction, mean seconds, completion-weighted wait and
request-to-touchdown means, all-gates status, safety counts, and shield
counts/fraction.

For every seed/scenario pair, the manifest records candidate-minus-baseline and
candidate-minus-rule deltas for:

```text
seconds
weighted_wait_seconds
request_to_touchdown_seconds
```

Progress is written after each controller record. The final manifest is written
only after the complete formal or smoke loop reaches its terminal boundary.

## Once-only reservation and evidence boundary

Immediately before formal rollouts, after historical seed rejection, the
candidate checkpoint parent reserves:

```text
p2_v4_heldout_consumed.json
```

The reservation is created exactly once with exclusive file creation and binds
the protocol, candidate, baseline, training manifest, seed list, and output
path. Smoke never reserves this file. The history root is read-only.

This document is a preregistration and does not claim an evaluation result.
No formal controller-quality PASS, improvement claim, or downstream training
permission may be inferred until an actual terminal manifest and its rollout
artifacts are produced and independently inspected.
