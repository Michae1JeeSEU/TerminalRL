# P3 Task-neutral Joint-team Learning-curve Development Protocol

日期：2026-09-29。状态：已预注册，尚未启动。本协议登记一次有界连续学习曲线 development
run；先完成定向 tests 与 check-only，再由主审准入唯一 full。不自动放行 formal、heldout、
P4 或全局 `PPO_ALLOWED`。

## 1. 固定起点、连续 update 与训练预算

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_learning_curve_20260929
```

恢复 `3fe7` actor 与 `b136` critic，使用 fresh Adam，并在同一次 run 的六次 update 之间持久
optimizer 状态；不沿用被拒的 `ac7` 权重，不回到 full-episode `gamma=1`/`lambda=1`。六个 batch
各收集六条完整 native on-policy rollouts，共 `36` rollouts、`864` requests；每批采样参数不变，
只在该批 steps 收集完成后执行一次 PPO update。每批 env seed 与 policy seed 依序为：

| batch | env seeds | policy seeds | permutation seed |
|---:|---|---|---:|
| 1 | `877101..877106` | `878201..878206` | `878001` |
| 2 | `877107..877112` | `878207..878212` | `878002` |
| 3 | `877113..877118` | `878213..878218` | `878003` |
| 4 | `877119..877124` | `878219..878224` | `878004` |
| 5 | `877125..877130` | `878225..878230` | `878005` |
| 6 | `877131..877136` | `878231..878236` | `878006` |

训练总上限为 `64800` steps（每条 rollout horizon `1800`）。`q` source 固定为 `3fe7`；q anchors
累计所有已采样 actor worlds，保持现有 world 等权/内 actor 等权；`C` 只按当前 batch 的
`totalactor / steps` 计算，不跨 batch 累积。每次 checkpoint 保存完整学习指标、accepted/rejected、
KL、q/freeze、reward、ledger 与安全字段。

参数保持 joint-team 配置：

```text
gamma=.99, lambda=.95, learning_rate=3e-5, ppo_epochs=2, batch_size=32
clip=.1, value_coef=.5, entropy_coef=.001, target_rollout_kl=.01
anchor_coeff=.1, max_kl=.05, soft_target=.02, gain=400
```

只训练 `gate_adapter` 与 critic；`proposal`、`actor_encoder`、`gate_head`、`temperature`
逐位冻结。reward、task priority、SOC/安全约束和 proposal 语义保持不变。

## 2. 连续学习曲线 evaluation

在第 `2`、`4`、`6` 次 update 后各评估四个 dense24 cells：env seeds `875101..875104`、
policy seeds `875201..875204`。before 四条 records 复用绝对路径：

```text
/home/user/training_outputs/terminal/p3_task_neutral_joint_team_actor_20260928/full/evaluation/records/before
```

after 为三次 evaluation slices 共 `12` 条新 rollouts，最多 `21600` steps；每个 slice 为
`4` 条 rollouts、`96` requests。上述 records 均为已消费 development 证据，不是 fresh 或
heldout。每次 evaluation 报告 normal/emergency/failed/unresolved、安全、mean/p95/max wait、
low-SOC wait 和 reward，并保留完整 energy failure 记录。

## 3. 停止与准入边界

无 Teacher、无 heldout，`PPO_ALLOWED=false`；后续 formal 不自动放行。本轮是扩大到连续学习的
有界 development run，不是单因果消融，也不沿用 `ac7` 权重或 full-episode MC。不得因表现差
早停、挑选 seed 或扩预算；只有技术异常、安全违规或 accepted update 为 `0` 时停止，输出保留、
不自动重跑。主审最终根据完整学习曲线决定候选取舍或进一步修复，不自动选择 best checkpoint。

先完成定向 tests 与 server check-only；主审准入后才启动唯一 full。当前尚未启动，不写 PID、
exit 或性能结果；task-neutral、SafetyFilter、reward、proposal 与所有 formal/heldout 边界不变。
