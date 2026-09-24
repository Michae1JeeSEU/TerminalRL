# Random-SOC P3 Gate Repair Development Protocol

Protocol: `RANDOM_SOC_GATE_REPAIR_V1_20260925`
Registered: 2026-09-25, before the first reset.
Status: preregistered development only; this does not mean that training has
started. No collection, training, rollout, qualification, P3 PASS, or PPO
admission has started under this protocol.

## Fixed source and runtime

The source policy is the frozen P2 checkpoint:

```text
/home/user/training_outputs/terminal/p2_runtime_shortlist_formal_v1_808001_808006/checkpoint_6.pt
SHA256 714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e
```

Use the current P3 scenarios and existing horizons: flow6 `1210` steps,
flow12 `1570` steps, and dense24 `1800` steps. Preserve late-slot
reconciliation OFF, `80/40/10` safety thresholds, landing floor `.10`,
admission reserve `.25`, K3 shortlist, and the existing P3 builders. The
proposal network and proposal encoder stay frozen. Existing P3 safety,
energy, route-guard, profile, arbitration, and mission settings remain
unchanged. Keep `P3_QUALIFIED=false` and `PPO_ALLOWED=false`.

## Fresh seed allocation

| purpose | seed | scenario | horizon |
|---|---:|---|---:|
| smoke | 846000 | dense24 | 2 steps |
| collection | 846101 | flow6 | 1210 |
| collection | 846102 | flow12 | 1570 |
| collection | 846103 | dense24 | 1800 |
| collection | 846104 | dense24 | 1800 |
| label-dev | 846201 | flow6 | 1210 |
| label-dev | 846202 | flow12 | 1570 |
| label-dev | 846203 | dense24 | 1800 |
| rollout-dev | 846301 | flow6 | 1210 |
| rollout-dev | 846302 | flow12 | 1570 |
| rollout-dev | 846303 | dense24 | 1800 |
| adapter training | 846401 | — | — |

The fresh allocation is paired by scenario where a source and candidate are
compared. It does not reuse formal or held-out seeds. The smoke must verify
source/checkpoint identity and output schema before collection.

## State-only label collection

Collect labels with Teacher label-only execution: the Teacher proposes the
reference gate but is not executed in the environment. For each UID, query
when an eligible actor first appears, when its mask changes from non-actor to
actor, or when its requested gate changes relative to the previous tick.
Non-actor rows are queried periodically only when `tick % 10 == 0`. A full
cycle cannot terminate because a label-count target was reached. Compute each
query label twice and require agreement. The physical/execution state digest
before and after each query must be unchanged, and every `decision_event_id`
must be unique. Train and dev seeds are disjoint; repeated computations are
not retained as duplicate training rows. Train on rows with
`actor_mask=true`; retain non-actor rows for audit. Preserve the existing
safety and full-cycle stop rules and raw labels.

## Adapter-only development training

Train only the gate adapter with fresh Adam (`lr=3e-4`), seed `846401`,
`35` epochs, batch size `64`, and clip `1`. Use inverse-frequency weighting
over the two gate classes in the training-set cross-entropy. The proposal
network, proposal encoder, and every non-adapter tensor remain frozen. Do
not use an old D1 anchor, dev gradients, early stopping, threshold tuning,
or any extra classifier threshold or gate. Report the final epoch only:
train/dev confusion matrices, both class recalls, macro-F1, NLL,
bitwise identity of non-adapter tensors, and unchanged proposal logits.
These scores are development evidence and do not promote the candidate.

## Paired rollout development

Before collection, the smoke must verify finite 88D observations, the output
schema, the label-to-gate/proposal mapping, unchanged weights, and successful
output parsing. After the adapter artifact is available, run the source and
candidate in paired development rollouts using seeds `846301`, `846302`, and
`846303` for flow6, flow12, and dense24 respectively, at the horizons above.
Set `Teacher enabled=false` and collect no labels in these rollouts. Preserve
late-slot OFF and the current safety/config contract. Each full cycle runs to
terminal or the frozen horizon. A safety exception stops that arm immediately
and its result is retained; failed-energy outcomes are not deleted and their
seed is not rerun. The primary agent reviews the smoke, collection/label-dev
artifacts, and paired rollouts before any qualification decision; this
protocol does not authorize qualification, P3 PASS, or PPO.

Old held-out evidence remains untouched. No new gate, classifier threshold,
proposal change, PPO sampling, or qualification run is part of this protocol.
