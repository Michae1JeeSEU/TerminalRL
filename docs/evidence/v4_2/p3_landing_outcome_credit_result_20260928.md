# P3 landing outcome credit result（2026-09-28）

## 范围与终态

本次是 source-only 的终态动作机会数据诊断，使用 source `03c0`、train dense24、seed `873103` 的 arrival 12/13/21 目标；不使用 validation 或 heldout，未训练。`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

Full `COMPLETED exit0`：`120` runtime rows、`6120` 单步 branches；3 个 selected states × 3 repeats，共 `27 logical / 24 actual / 3 reuse`。source replay/outcomes/model/RNG checks 均为 true，policy update=false，training=false。heartbeat `landing-outcome-credit-diagnosis` 已终态删除。

## Selected states

| state | arrival / uid / step | source pRelease | qK3 | C |
|---|---|---:|---:|---:|
| state0 | 12 / 42 / 102 | 0.000710682827 | 1 | 24 |
| state1 | 12 / 42 / 139 | 0.998769342899 | 9.7754e-15 | 9 |
| state2 | 21 / 309 / 366 | 0.759813606739 | 1 | 13 |

`Release-vs-WAIT` 的 normal gains（按 3 repeats）依次为 `[2,1,1]`、`[1,1,1]`、`[1,0,0]`；`target_normal` gains 依次为 `[1,1,1]`、`[0,0,0]`、`[1,0,0]`。三个 state 的 failed deltas 均为 `0`，policy gaps 依次为 `1.3323857562`、`0.0012306571`、`0.0800621311`。

proposal deltas 依次为 `[0,0,0]`、`[0,-1,1]`、`[0,0,0]`，均不满足 proposal opportunity 条件。state0 的 priority-weighted wait 变化为 `[+16.54458,-11.51764,-16.29416] s`，low-SOC wait 变化均为下降：`[-12.63103,-24.98760,-19.92814] s`。

## 主审记录的结论边界

- state0 是当前优先研究的局部 gate 机会：source 仅约 `0.0711%` 放行，目标 UAV 三次均由 emergency 转为 normal。
- state1 source 已以 `99.8769%` 放行，gate 改进空间小；normal 增量不作为高价值 gate 标签。
- state2 的证据较弱。
- 三个 state 属于同一 development 场景，收益不相加。MC3 与 global-RNG 后续 actor 路径可能分叉；单一 development 场景、未训练结果不能声称稳定收益、泛化收益或训练退化已解决。

下一步限定为围绕 state0 验证局部动作优势并设计受 KL 约束的 gate 目标，保留 proposal/source 与安全边界，并匹配完整场景评估；不直接重启 PPO 或扩大旧 hard-CE fit。本 heartbeat 不启动新训练；旧 V2 结果保持不变。

## 技术核验与证据

arrival12/13/21 各有 `40` runtime rows，C 非空数分别为 `2/0/2`。非空 steps 为 arrival12 `[102,139]`、arrival21 `[366,369]`；step369 距366仅3步，按 `<10` 排除，因此最终为3 states。

全部9个 state/repeat 的 acceptance、arrival signature、base/model/parent RNG 均为 true；27 branches 的 terminal、full-requests24、ledger、weights 均为 true；collision、hard-interval、NMAC、execution-plan-hash mismatch 增量均为 `0`。没有独立 exogenous 命名字段，未声称检查未提供字段；本结果记录 arrival signature 一致。

source step139 的 requested action 为 RELEASE proposal11，但 executed action 为 WAIT；runtime proposal11 为 safety_reject，文案为 `HEADWAY/merge_slot_retime_conflict`。该点不写作 target 已解决；target 三次结果仍为 emergency。

| state/repeat | WAIT normal | source normal | certified normal |
|---|---:|---:|---:|
| 0/0 | 21 | 23 | 23 |
| 0/1 | 22 | 23 | 23 |
| 0/2 | 21 | 22 | 22 |
| 1/0 | 20 | 21 | 21 |
| 1/1 | 22 | 23 | 22 |
| 1/2 | 21 | 22 | 23 |
| 2/0 | 21 | 22 | 22 |
| 2/1 | 21 | 21 | 21 |
| 2/2 | 21 | 21 | 21 |

每行均为 `24` requests，余者为 emergency，failed 为 `0`。

- [冻结协议](p3_landing_outcome_credit_protocol_20260928.md)
- [full report](../../../../training_outputs/terminal/p3_landing_outcome_credit_20260928/full/report.json)
- [full summary](../../../../training_outputs/terminal/p3_landing_outcome_credit_20260928/full/summary.json)
- [full selection](../../../../training_outputs/terminal/p3_landing_outcome_credit_20260928/full/selection.json)
