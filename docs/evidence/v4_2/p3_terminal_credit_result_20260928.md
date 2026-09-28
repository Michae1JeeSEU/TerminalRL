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

## Gate-only probe V2 终态

V1 `1`-step smoke 的机械零更新作为历史事实保留。V2 smoke 为 `COMPLETED exit0`：`1 accepted / 2 optimizer attempts`，accepted trial scale `.5`，max KL `0.0034920275`。V2 full 为 `5 accepted / 6 logical / 18 optimizer`，accepted scales `[.5,1,.5,.25,.0625]`；第 6 个 logical update 的全部 trial 拒绝并完整回滚，最终 mean KL `0.000055444827`、max KL `0.0096337497`，freeze/roundtrip invariants 全部通过。candidate checkpoint SHA256 为 `27ea5a017f9471250f632e0c39ca64a6fbc2531612b4918dd6efbeacbd98cb19`。

| stage | status | observed facts |
|---|---|---|
| V2 smoke | `COMPLETED exit0` | `1 accepted / 2 optimizer attempts`; scale `.5`; max KL `0.0034920275` |
| V2 full fit | `COMPLETED` | `5 accepted / 6 logical / 18 optimizer`; final mean/max KL `0.000055444827 / 0.0096337497`; full rollback on logical update 6 |
| matched evaluation | `COMPLETED` | `technical_ok=true`; `6/6` cells; `84` requests; all ledger checks pass; `candidate_retained=false`; selected `03c0` |

source 为 `76 normal / 8 emergency`，candidate 为 `74 / 10`；两者 `failed/unresolved/safety=0`。aggregate priority wait 为 `86.66905761611285→87.53247367861496`，low-SOC wait 为 `89.77926924393867→102.624593631769`。四个 flow6/flow12 cells 不变。dense24/873103 的 normal 为 `21→21`，priority wait `99.77872368→104.41590090`，low-SOC wait `123.60082027→125.77824302`；dense24/873106 的 normal 为 `19→17`，priority wait `159.20141913→157.05353879`，low-SOC wait `145.30236533→176.32696493`。主审拒绝 candidate 并保留 `03c0`；有界更新技术有效，但稀疏五标签未转化为完整收益，首处分歧及反向改善已列事实，本文不作完整因果判断，也不声称训练退化已解决。

结果根为 `/home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928`： [smoke report](../../../../training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/smoke/report.json)、[fit report](../../../../training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/fit/report.json)、[candidate checkpoint](../../../../training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/fit/terminal_gate_repair.pt)、[evaluation manifest](../../../../training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/evaluation/manifest.json)。代码提交为 `9346ee3`，定向测试 `15 passed`。`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`，不消费 heldout。

## 主审解释与未执行下一步（2026-09-28）

source weightedCE 为 `0.174300178885`，candidate 为 `0.168798521161`。稀疏目标的三个稳定标签事实如下：

| state | label / weight | source → candidate | source weightedCE / observed full-cycle fact |
|---|---|---|---|
| 2 | WAIT / `0.00905526` | CE `17.531902`；weightedCE `0.158755898476`（`91.1%`） | 完整后续收益仅 priority wait `0.129199155313 s`，normal 不变 |
| 6 | RELEASE / `0.95162936` | `p=0.999998689→0.999998093` | source weightedCE `0.0000012479`；原策略已近乎必选 |
| 11 | RELEASE / — | `p=0.148311973→0.161266223` | 最小收益 `0.044117647059 s`，normal 不变 |

主审解释为：稀疏硬 CE 目标受微小收益与高置信反标签主导。该解释用于记录本轮审查结论，不支持继续放大五标签拟合或重启 PPO。

首个 requested-action 分歧的事实为：dense873103 在 step13、uid4/arrival4，dense873106 在 step35、uid3/arrival3；两处均有相同 observation hash 和 active UID，source 为 RELEASE、candidate 为 WAIT，但两边 applied/executed 均为 WAIT，因此不能作为物理执行差异的完整因果证明。873106 中 normal→emergency 的 arrival4/16/18 等待分别增加 `26.280`、`180.995`、`14.737 s`；这三项单向变化与总体 normal `-2` 同时存在，另有反向改善，不能只报告三项而省略反向变化。

下一步（未执行）是优先从 source 拥堵场景提取能改变 normal/emergency 结局、且 source 存在实际动作改进空间的完整周期动作优势样本，再设计优势目标。该方向尚未验证；本轮不扩大五标签 fit，不重启 PPO。代码 `9346ee3`、定向测试 `15 passed`，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false` 保持不变。
