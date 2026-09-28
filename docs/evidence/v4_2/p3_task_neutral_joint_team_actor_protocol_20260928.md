# P3 Task-neutral Joint-team Actor Aggregation Protocol

日期：2026-09-28。状态：已设计，尚未启动。本协议只登记一次有界 development actor
aggregation validation；定向 tests 与 check-only 通过后，仍须主审准入唯一 full。它不自动
放行 formal、heldout、P4 或全局 `PPO_ALLOWED`。

## 1. 目的与固定范围

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_joint_team_actor_20260928
```

本轮只验证 actor loss 的 team aggregation 权重定义。固定 `3fe7` actor 与 `b136` critic，
复用 joint-batch 的六个 task-neutral development training cells：
`(873103,873103)`、`(873106,873106)`、`(874601,874701)`、`(874602,874702)`、
`(874603,874703)`、`(874604,874704)`。训练预算为六个 cells、每 cell 上限 `24` requests、
合计 `144` requests、最多 `10800` steps，收集后只执行 `1` 次 joint PPO update；这些 seeds 均为已消费的
development 数据，不构成独立或 untouched cohort。

本轮使用原 joint-batch 的 `gamma=.99`、`lambda=.95`，不沿用被拒 full-episode credit repair
的 `gamma=1`、`lambda=1`。其余参数保持原值：

```text
learning_rate=3e-5, ppo_epochs=2, batch_size=32, clip=.1
value_coef=.5, entropy_coef=.001, target_rollout_kl=.01
permutation_seed=875001
```

只训练 `gate_adapter` 与 critic；`proposal`、`actor_encoder`、`gate_head`、`temperature`
逐位冻结。`q` 与 KL reference 沿用 joint-batch 原参数，不改 reward、安全、proposal 或
task-neutral 语义。所有 seed、终态和不利结果均记录，固定预算一次，不因结果不利而重跑。

## 2. Actor aggregation 定义

将六条完整 episode 合并为本次 update 的一个 batch。令 `actor_count_t` 为合并 batch 中时刻
`t` 的 actor rows 数，`T` 为六条 episode 的全部 steps 总数，包含没有 actor rows 的时间步，
定义整个 update 唯一共享常数：

```text
C = totalactor / T
```

所有 minibatch 共用同一个 `C`，禁止按 episode 分别重归一。对每个 minibatch，新的 actor loss
定义为：

```text
actor_loss = sum(all actor clipped losses) / (minibatch world_count * C)
```

当 `C=0` 时 actor loss 为 `0`。该定义使用一个 update-level 常数，不按每个 world 的
UAV 数量再除以 `N`；历史 world-mean 行为作为兼容默认保留。critic、advantage normalization、
entropy 和 KL 的定义不变，仍使用 per-agent clipped ratios，不构造 joint ratio 的乘积。

## 3. Toy 验证证据与限制

当前可枚举 toy 验证记录了 objective weighting 不一致的数学证据：等概率选择
`A: N=1, R=-a` 与 `B: N=4, R=.5*sum(a)`，共享 `p=.5` 时，`E=.25`、
`dE/dtheta=.125`；finite difference 为 `0.124999999995`。对 `32` 个等权枚举样本，
production default update 从 `theta=0` 到 `-1.25e-6`，方向相反；world-sum 与 flat normalization
的数学方向为正。该 toy 只说明 aggregation objective 的权重证据，不证明实际退化的全部根因。

一般 policy-gradient 原理可参见 [Spinning Up policy gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)；
上述多-agent aggregation 推导是本项目自己的定义与验证记录。

## 4. 配对 evaluation 与准入边界

复用旧 joint-batch/full evaluation records、profile/cell 验证，以及绝对路径
`/home/user/training_outputs/terminal/p3_task_neutral_joint_batch_20260928/full/evaluation/records/before`
下的精确六条 before records；不重跑 before。新 after 沿用同六个 evaluation cells，每臂 `114` requests，
最多 `9980` steps：

| 场景 | env seed | policy seed |
|---|---:|---:|
| dense24 | `875101` | `875201` |
| dense24 | `875102` | `875202` |
| dense24 | `875103` | `875203` |
| dense24 | `875104` | `875204` |
| flow6 | `875105` | `875205` |
| flow12 | `875106` | `875206` |

这些 before/after cells 都属于已消费的 development 证据，不称独立泛化或 heldout。报告整体与
分 cell 的 `normal/emergency/failed/unresolved`、安全、mean/p95/max wait、low-SOC wait 和
reward，保留全部失败；不增加每 cell 或子组无退化硬门槛。只有在结果显示改善后，才由主审
决定是否独立验证，本协议不授权扩预算或继续训练。

先完成定向 tests 与 server check-only；check-only 通过且主审准入后才启动唯一 full。当前
尚未启动，不写 full PID、exit 或性能结果。任务 priority、safety、proposal、`829xxx`
heldout、`PPO_ALLOWED=false` 和 formal/P4 边界均不变。
