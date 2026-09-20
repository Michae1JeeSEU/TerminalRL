# P2R Dense Landing-Flow Resource Semantics V1 Summary

> Date: 2026-09-21  
> Evidence type: newly-run paired diagnostic  
> Scope: dense fixed-geometry landing-flow resource semantics; no policy training or checkpoint promotion  
> Historical boundary: all prior P2 protocols, seeds, checkpoints and PASS/FAIL conclusions remain immutable

## 1. Question and intervention

The `p2_dense24` visualization exposed three operational symptoms: lowest-ID landing-zone assignment could
select a much farther free LZ; `final_queue_capacity=1` serialized a six-LZ system; and same-tick insertion
order could process upstream holders before downstream demand. V1 therefore compared the same frozen
original K=3 policy and identical seeds under:

- `legacy`: `lowest_id`, insertion-order arbitration, final capacity 1;
- `throughput_v1`: `nearest_available`, `INNER → MIDDLE → OUTER` arbitration with
  priority/wait/UID tie-breaks, final capacity 3.

Every release remained subject to the existing reservation, headway, canonical trajectory, SafetyFilter and
shield path. The policy did not output `lz_id`, the action/observation schema did not change, and no Teacher
action was executed.

## 2. Scenario and artifacts

- frozen checkpoint: `gate_repair_v3_qualification_candidate.pt`, SHA256
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`;
- scenario: 12 warm-start UAVs in canonical 6/4/2 O/M/I allocation plus 12 arrivals at 15-second intervals;
- paired horizon: 300 s;
- paired seeds: `809006–809009`;
- server artifacts: `~/training_outputs/terminal/p2r_dense_multiseed_20260921`;
- counterfactual opportunity seed: `809010`;
- counterfactual artifact: `~/training_outputs/terminal/p2r_counterfactual_20260921/seed809010.json`.

Seed `809006` was run locally; seeds `809007–809010` were run on `lab-server`. These seeds are diagnostic
and must not be reused as future P2R training or qualification seeds.

## 3. Paired results

Across seeds `809006–809009`:

| Metric | Legacy | Throughput V1 | Observation |
|---|---:|---:|---|
| touchdowns by 300 s | 23 | 30 | +30.4% |
| service completions by 300 s | 17 | 22 | +29.4% |
| max concurrent final | 1 on every seed | 2 on every seed | serialization removed without forcing capacity use |
| mean assignment distance per seed | 214.5–254.1 m | 151.5–160.2 m | consistently shorter |
| assignment regret | 88.7–123.9 m | 0 on every seed | nearest-free contract met |
| collision / NMAC / hard separation / uncertified execution | all 0 | all 0 | hard safety unchanged |

The lower INNER/MIDDLE occupancy under Throughput V1 is expected: ring capacity is a safety upper bound,
not a utilization target. When certified downstream service is available, draining these rings is the desired
throughput behavior.

## 4. Counterfactual opportunity attribution

The raw `not_requested` count includes agents executing committed plans and therefore cannot measure missed
landing opportunities. On seed `809010`, Throughput V1 had 108 inner-ring actor decision steps with a free LZ
and a nominal final slot. The policy did not request release on 95 of them, but bounded read-only catalogue
certification found an individually safe RELEASE on only 4 steps. Thus confirmed gate misses were
`4/108 = 3.70%` of inner actor steps; the other 91 nominal-resource-idle steps were not executable under the
current trajectory/headway/safety constraints.

This probe attaches a TeacherScheduler only to enumerate read-only counterfactual candidates. Student actions
remain the sole environment execution input; the result is diagnostic and does not authorize imitation or PPO.

## 5. Decision

The observed visualization problems are attributed primarily to resource semantics, not to a globally broken
P2 gate policy or an unsafe desire to fill every ring. V1 resolves the concrete LZ-distance, upstream-order and
single-final bottlenecks and passes the dense multi-seed hard-safety check.

No new PPO run is launched from this diagnostic alone. A P2R training round is warranted only after a fresh,
multi-seed protocol shows a stable individually-certified-but-not-requested gap large enough to optimize, with
separate train/dev/heldout seeds. Future dense qualification must mix warm-start competition with
flow-from-empty regression and must report throughput, latency, final concurrency, LZ regret, counterfactual
opportunity and hard-safety metrics.
