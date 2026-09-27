# P3 terminal gate repair protocol（2026-09-28）

## 冻结范围

本文件冻结一次有界的 gate-only development probe。输入为已完成的 terminal-credit summary 与 selection（来自本地同步输出目录），source checkpoint 为 `03c0`。本 probe 只验证终态标签是否能够改善最终等待，同时检查正常降落是否保持；不预先声称正常降落数量会增加。

本轮只使用两个 repeat 方向一致的五个 state：`state_index=[1,2,3,6,11]`。其中 `state_index=[1,2]` 的标签为 `WAIT`，`state_index=[3,6,11]` 的标签为 `RELEASE`。其余七个 uncertain state 不进入 fit。

proposal、actor encoder、base gate、gate temperature 和 critic 全部冻结；只有 `gate_adapter` 可以训练。source 模型加载时不得复制 optimizer state。

## 训练输入与目标

复用已有 source train 三个 seed 的 gate anchors：

```text
/home/user/training_outputs/terminal/p3_runtime_rank_repair_20260926/anchors/anchors.npz
```

该 anchor 输入包含 `11589` 个 observations。它只用于 source gate 分布 KL 锚定，不扩展训练数据范围。

从 `03c0` 重置 Adam，固定 full-batch、学习率 `1e-4`、最多 `120` 次更新和 gradient clip `1.0`。监督 CE 的样本标签来自上述五个 state；每个 state 两个 repeat 中较小的 priority-weighted wait 改善秒数作为正权重，并在全体五个 state 上归一化。五个 state 的 normal/failure 结果相同；low-SOC 已按 Pareto 规则筛选，不为不可比较的 N/A 合成零值。

目标函数固定为：

```text
weighted_CE + 10 * mean_KL(source_gate || candidate_gate)
```

每个 optimizer step 都检查 anchor gate KL：mean 必须 `<= 0.001`，max 必须 `<= 0.01`。任一约束超过上限时，恢复该步之前的完整 model 与 optimizer state，并终止；不搜索其他参数、不调整超参数、不重启训练。最终只接受最后一个满足约束的 checkpoint。

## 不变量与执行顺序

fit 前先运行本轮测试，再运行 `1` step 输出 smoke，确认输出字段、checkpoint 和 rollback 状态可序列化；通过后才运行一次正式 bounded fit。正式输出根目录必须为 fresh：

```text
/home/user/training_outputs/terminal/p3_terminal_gate_repair_20260928
```

最多生成一个正式 fit candidate 和一个 smoke 结果，不扩展候选矩阵。fit 技术通过后，只在已经消费的六个 development seeds `873101`--`873106` 上运行 source/candidate matched evaluation；复用既有 source 完整 ledger 记录，仅新增 candidate 的 `6` 次 rollout，不重跑 source，并记录完整 request lifecycle。

候选 checkpoint 必须逐张量验证 proposal 与所有非-adapter 权重和 source 完全一致；gate adapter 以外的参数不得变化。source 与 candidate 的环境、协议、seed、request 顺序和生命周期字段必须可配对。

## 保留条件与边界

候选只有同时满足以下条件才保留：每个场景的 normal 数量不减少；failed、unresolved 和真实 safety events 不增加；聚合 priority wait 与 low-SOC wait 均不增加；并且至少有一项等待指标严格下降。否则保留 `03c0` source。

该 probe 不自动开启 PPO、P3 晋级、heldout 或 formal 评估。六个 development seeds 已被消费，本轮结果不具有独立泛化资格。结果区在执行前保持空置；本文件只记录冻结协议，不记录尚未发生的 fit 或评估结果。
