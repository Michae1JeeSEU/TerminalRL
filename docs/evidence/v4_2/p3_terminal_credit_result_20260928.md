# P3 terminal reward-order and full-cycle credit result（2026-09-28）

本结果对应冻结的 [terminal reward-order/full-cycle credit protocol](p3_terminal_credit_protocol_20260928.md)。配对权威产物为本地同步的 [corrected summary sidecar](../../../../training_outputs/terminal/p3_terminal_credit_20260928/reward_order/summary_corrected.json)；旧 `summary.json` 的错误 pairing section 保留，不作为配对报告来源。后续 gate-only probe 见 [terminal gate repair protocol](p3_terminal_gate_repair_protocol_20260928.md)。

## Existing development tapes 的 reward 重建

本轮只读重放 source `03c0` 与 V7 run0@80、run1@240、run2@240 的既有六个 development seeds，共 `24` tapes。`24/24` 的 replay、reward components、terminal ledger 和全 request 均通过；`unresolved=0`。raw return 与既有 summary 的 `reward_total` 逐条一致（浮点误差范围内）。

| arm | raw return sum | episode-start discounted return sum（γ=.99） | normal | emergency | failed_energy |
|---|---:|---:|---:|---:|---:|
| source `03c0` | 1606.131863 | 74.787596 | 76 | 8 | 0 |
| run0@80 | 1543.186357 | 74.421692 | 76 | 7 | 1 |
| run1@240 | 1601.928370 | 71.265154 | 76 | 8 | 0 |
| run2@240 | 1563.341301 | 67.047586 | 72 | 12 | 0 |

三个 candidate 的聚合 raw 与 episode-start discounted return 都低于 source。这一结果不支持把完整退化全面归因于 reward 总量偏爱坏结果。`dense24/873106` 的 run0 个案相对 source 为 raw `-67.136088`、episode-start discounted `+0.206033`、normal `-1`、failed-energy `+1`，说明 episode 起点折扣 return 与终态排序存在个案不一致；该折扣 return 不等于 PPO actor loss，也不能证明唯一因果关系。

## Counterfactual terminal credit 事实

CF 使用 `12` states × `2` repeats，形成 `24` reports；`72` 个逻辑 branches 中 `52` 为真实执行、`20` 为复用。全部 full-terminal、exogenous signatures、ledger、state、RNG 和 weights checks 通过；真实 safety counters 为 `0`，`uncertified_execution` 为 `N/A`。重复 branch 汇总的 `939 normal` 不作为独立性能证据。

稳定标签共 `5` 个：state `1,2` 为 `WAIT`，state `3,6,11` 为 `RELEASE`；其余 `7` 个为 `uncertain`。稳定标签的 normal 数量持平，观察到的变化只在 wait 指标。

- state `6`（flow12）两次 repeat 的 priority wait 分别为 `31.51→16.20`、`30.89→17.32`；最终值取完整 replay 的 raw 指标。
- state `10`（dense24）的一步 proxy 为 `RELEASE-WAIT=+0.133583`；两次 suffix 折扣回报差为 `-1.574011`、`-1.508527`。这只是该状态的局部方向不一致事实，不是全局因果证明。

## Gate-only probe 当前状态

V1 `1`-step smoke 已完成：`exit0`、`COMPLETED`、`1 attempt / 0 accepted`；source-anchor mean KL 为 `0.00031149349524639547`，max KL 为 `0.014092326164245605 > 0.01`；完整 model 与 optimizer 已回滚，`nonadapter=true`、`proposal=true`、`roundtrip=true`、`adapterchanged=false`。V1 full fit/evaluation 不启动；该机械零更新尚不能判断 terminal-label 效果。原 V1 smoke 输出保持 immutable：`/home/user/training_outputs/terminal/p3_terminal_gate_repair_20260928/smoke`。

V2 bounded-backtracking protocol 待实现、测试和执行，见 [bounded-backtracking protocol](p3_terminal_gate_repair_v2_protocol_20260928.md)。V2 仍沿用 source `03c0`、五标签、`11589` anchors、`weighted_CE + 10 * mean_KL`、Adam base LR `1e-4`、clip `1.0`、最多 `120` logical updates 和 gate-only 冻结范围；尚无 V2 candidate 或评估结果。`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`，不消费 heldout。