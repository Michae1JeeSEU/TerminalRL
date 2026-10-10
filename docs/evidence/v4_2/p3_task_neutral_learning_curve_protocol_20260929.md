# P3 Task-neutral Joint-team Learning-curve Development Protocol

日期：2026-09-29。状态：运行中。本协议登记一次有界连续学习曲线 development
run；定向 tests/check-only 已通过，主审已批准唯一 full，运行 PID 与首条 rollout 已确认。
不自动放行 formal、heldout、
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

新 runner 实现与定向 suite 共 `67 passed`；终态修正后的 learning-curve checks 为 `9 passed`，
pycompile 与 diffcheck 通过，四个源码/测试 SHA 已完成本地-服务器同步核对。server check-only
report 为 `CHECK_ONLY_OK`、`exit=0`；full report 最终为 `COMPLETED`、`technical_ok=true`、
`exit=0`。六批 training 共 `36` rollouts、`864` requests、`28030` steps，执行 `6` 次 PPO calls，
`1758` accepted、`0` rejected；训练统计为 `714 normal / 149 emergency / 1 failed_energy`，
q/freeze/optimizer 均为 `true`。full checkpoint SHA256 为
`b1144a698b193d30a67b887e68bfc169aca183573de1ff0dcf954bf4177ca0bc`。配对 evaluation 为 `12` 条
new 与 `4` 条 reuse，三次 before 均为 `80 normal / 16 emergency / 0 failed / 0 unresolved`，
安全计数均为 `0`；baseline mean/p95/max wait `124.5896/281.3078/361.1838`，low-SOC
`119.3941`，reward `1810.5198`。after 依次为：

| update | normal/emergency/failed/unresolved | mean/p95/max wait | low-SOC | reward |
|---:|---|---:|---:|---:|
| 2 | `77/19/0/0` | `127.737/314.625/408.166` | `116.411` | `1778.264` |
| 4 | `75/21/0/0` | `138.778/323.653/399.081` | `140.425` | `1714.662` |
| 6 | `77/19/0/0` | `143.843711/352.115802/447.42` | `123.97` | `1743.085531` |

heartbeat `task-neutral-ppo-learning-curve` 已删除。主审拒绝本轮性能 candidate，但保留
joint-team 数学修正，不宣称退化已解决。下一步只读核对完整周期目标与每秒折扣、critic 训练目标
匹配；未批准下一次环境训练。task-neutral、SafetyFilter、reward、proposal 与所有 formal/heldout
边界不变。
