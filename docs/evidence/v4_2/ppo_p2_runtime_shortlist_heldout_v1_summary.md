# K=3 Runtime Shortlist Paired Heldout V1

## Frozen comparison

- Protocol SHA256: `eee2543918bf80610e80ccf11efdac602369d3bb46dd2226ce53ed87d57950e6`
- Seeds: untouched `809001–809006`, with flow6/flow8/flow12 repeated twice.
- Controllers: formal `trained_k3`, original qualified `initial_k3`, and nominal `rule`.
- Formal manifest SHA256: `3c1bc69a518f7e5d99bb606d98b0356397e86f97b493947a0f14b0c8b49b010b`.
- Trained checkpoint SHA256: `714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e`.

## Result

Server output: `/home/user/training_outputs/terminal/p2_runtime_shortlist_heldout_v1_809001_809006`

Manifest SHA256: `f44acf5e5c78925b71202e0bb81ce4c2c56d302a51fd207025b2228714a666ce`

| Controller | Completed | Weighted wait (s) | Request-to-touchdown (s) | Mean completion (s) | Safety | Shield fraction |
|---|---:|---:|---:|---:|---:|---:|
| trained K3 | 52/52 | 4.805102 | 305.403846 | 920.833333 | 0 | 0 |
| initial K3 | 52/52 | 4.838487 | 305.384615 | 920.833333 | 0 | 0 |
| rule | 52/52 | 15.066060 | 316.480769 | 891.666667 | 0 | 0 |

All 18 controller episodes passed completion, safety, geometry, exact replay, requested likelihood,
Teacher/update isolation, and K=3 contract gates. The trained policy used runtime fallback 12 times,
and the proposal-generating subgraph remained exactly frozen.

Relative to the initial K3 candidate, formal PPO changed weighted wait by `-0.033386 s` but changed
request-to-touchdown by `+0.019231 s`. Relative to rule it improved the two metrics by `-10.260958 s`
and `-11.076923 s`, respectively.

## Decision

The preregistered gate required both primary metrics to be non-worse than the initial K3 candidate.
The positive `+0.019231 s` request-to-touchdown delta therefore makes the immutable result
`PASS=false`. The formal checkpoint is not promotable and static-geometry qualification is not
authorized from it. The threshold is not relaxed and the heldout seeds are not reused.

The failure does not invalidate the frozen K=3 runtime boundary: both learned K3 controllers remained
safe, complete, and substantially better than rule on wait and request-to-touchdown. It shows that this
PPO block did not add a consistent heldout benefit over the original qualified K3 candidate.

## Independent selection review

The subsequent read-only selection review preserved the heldout FAIL, rejected the formal checkpoint,
and selected the original K3 candidate for the next curriculum protocol. Review SHA256:
`5617c96255c040797abf2d2f44ed81ce53be955ed94d3a8447a5c943b2f59969`.
All nine review gates passed. This sets `P2_PPO_STOP=true` and only
`P3_PROTOCOL_PREREGISTRATION_ALLOWED=true`; it does not directly authorize P3 execution or checkpoint
promotion.
