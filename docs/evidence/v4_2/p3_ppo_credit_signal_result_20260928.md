# P3 PPO credit signal diagnostic result（2026-09-28）

## 结果状态

Evidence type: `newly_run`。本次按 [PPO credit signal protocol](p3_ppo_credit_signal_protocol_20260928.md) 完成一次 source `03c0` 的只读 replay；未训练、未更新、未采样、未启用 Teacher、未消费 heldout。

代码提交为 `77a549bbc12c2dfdc7abb12f9361e325eb72b2e4`，定向测试为 `32 passed`（`9` diagnostic、`11` lifecycle、`12` evaluation）。首次启动预检因远端 cwd 遗漏产生 `ModuleNotFoundError`，发生在启动前且未执行 reset/step；修正 cwd 后唯一 replay 为 `COMPLETED exit0`，不计预检为第二次实验。

输出根：

```text
/home/user/training_outputs/terminal/p3_ppo_credit_signal_20260928/trace
```

直接观察到：`702/702` exact steps、`24/24` ledger/all-requests、`512` actor steps、terminal step `701`；source weights unchanged、RNG restored；collision、hard-separation、NMAC 均为 `0`。`report.scope` 无 training/optimizer/sampling/Teacher/heldout。诊断在 worker 监控窗口内完成，无 active run、无 heartbeat，且未创建 heartbeat。

## Credit signal 记录

source fixed tape raw reward 为 `448.98876382803303`，reward components 吻合。`dt=1` 时，`gamma=.99` 半衰期为 `68.96756 s`，`lambda=.95` 固定。关键 step 的 source requested gate、critic value、单轨迹 MC return（`gamma=.99`）、GAE raw、actor-step normalized GAE，以及 fixed critic 仅切换 `gamma=1.0` 的 normalized 对照为：

| step | UID / arrival | source requested | critic V | MC `gamma=.99` | GAE raw `gamma=.99` | normalized `gamma=.99` | normalized fixed-critic `gamma=1.0` |
|---:|---|---:|---:|---:|---:|---:|---:|
| 102 | 42 / 12 | 0 | 14.9018993 | 69.9018934 | 2.24120283 | -0.590941548 | -0.516637743 |
| 103 | 42 / 12 | 1 | 14.9006491 | 70.7249064 | 2.66584110 | -0.548706889 | -0.467637658 |
| 104 | 42 / 12 | 0 | 14.9006081 | 71.7584548 | 3.32892299 | -0.482756525 | -0.396039546 |

负的 normalized 局部代理信号不能宣称 step102 必然鼓励 WAIT；它表示该样本局部代理项倾向降低已采样 WAIT 概率，完整 batch 的其他梯度与 KL 仍会影响最终更新。actor-step 标准化不据此写成 bug；fixed critic 下仅切换 gamma 未改变三步信号符号。V 与单轨迹 MC 的差异不等于已证明所有 V7 退化根因或 critic 统计失准。

## Mission pairing

mission pairing 只覆盖 dense24、seed `873103` 的 `24` requests；source 新 trace 记录 mission，candidate 仅在内存分析副本中按 source exact replay 的 immutable request 属性、`seed+arrival_id` 做 join，结果列于本文，不写候选 JSON sidecar，也不改历史 outputs。`request_time` 与 `base_priority` 完全一致，SOC 重建容差为 `1e-12`，实际 `maxdiff=3.885780586188048e-15`；不能把全 `84` rows 写成已完成 mission 核对。

| mission | source（count / normal / emergency_landed / failed / mean wait） | candidate（count / normal / emergency_landed / failed / mean wait） |
|---|---|---|
| cargo | `6 / 5 / 1 / 0 / 116.960214809` | `6 / 6 / 0 / 0 / 112.718649405` |
| emergency | `1 / 1 / 0 / 0 / 168.329161346` | `1 / 1 / 0 / 0 / 179.505236768` |
| standard | `13 / 11 / 2 / 0 / 128.954875674` | `13 / 12 / 1 / 0 / 100.700289030` |
| surveillance | `4 / 4 / 0 / 0 / 76.481889822` | `4 / 4 / 0 / 0 / 119.457418021` |

变化为：arrival5 `STANDARD` `normal→emergency_landed`；arrival12 `CARGO`、arrival13/21 `STANDARD` `emergency_landed→normal`。唯一 emergency mission 为 arrival20，两边均为 `normal`，wait `+11.176075422`，touchdown `545→567`；没有完成同时争用同一资源的冲突层检查，因此不据此宣称发生普通任务抢占紧急任务，也不据此宣称已保证紧急任务优先。

## Candidate reward 补充

six-cell 总体 source reward `1606.1318631737224` → candidate `1621.4070350399447`（`+15.2751718662`），normal `76→78`、emergency `8→6`、priority-weighted wait `+0.95309 s`、low-SOC wait `−2.67283 s`；dense24/873103 单 cell source `448.9887638280331` → candidate `464.2639356942554`，normal `21→23`、emergency `3→1`。按当前 soft preference 语义，这是待进一步验证的 normal 收益信号；旧 `candidate_retained=false` 历史保持，不事后改旧 manifest，不自动晋级或切换 source。source fixed-tape MC 与此前 counterfactual 三-repeat 结果来自不同 future trajectories，不能混作同一数据。

## 结论边界

本次只读诊断未改变实际 `gamma=.99`、reward、scheduler 或 source，未启动新的 PPO；`PPO_ALLOWED=false`。后续回到完整周期 reward/discount objective 与 critic/team-credit 一致性验证，不把本次固定 replay 写成对所有 V7 退化的根因证明。

## Evidence links

- [new trace report](../../../../training_outputs/terminal/p3_ppo_credit_signal_20260928/trace/report.json)
- [prior normal-advantage evaluation record](../../../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/evaluation/records/dense24_873103.json)
- [prior reward-order summary](../../../../training_outputs/terminal/p3_terminal_credit_20260928/reward_order/summary_corrected.json)
