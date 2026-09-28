# P3 Task-neutral Joint-team Sampling Repeats Protocol

日期：2026-09-29。状态：已预注册，尚未启动。本协议只登记一次固定模型的 sampling
stability validation；先完成代码定向 tests 与 check-only，再由主审准入唯一 full。不自动
放行 formal、heldout、P4 或全局 `PPO_ALLOWED`。

## 1. 固定模型、范围与目的

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_joint_team_repeats_20260929
```

before 固定 `3fe7` actor，after 固定 joint-team checkpoint
`ac7fbb5b3b506e693c6a3073c99b85cc3c0d395644755a7496cff8d48c5fd05a`；两臂均不训练、不执行
optimizer、不使用 Teacher。验证四个 dense24 场景 `env=875101..875104`，每个场景三条
policy sampling：每臂共 `12` records、`288` requests，合计两臂 `24` records；其中每臂每个
场景的一条原 policy record 复用既有结果，另两条为新 sampling。四个场景全部纳入，不只选择
不利 cell。

本轮目的为区分策略采样轨迹波动与稳定退化，不构成独立新物理场景泛化证据。原 joint-team
单 seed 结果保持为配对背景，不把它单独当作新的独立证据。

## 2. Seed 映射与复用边界

每个环境的原 policy seed 复用 joint-team full 的对应 before/after record；绝对复用根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_joint_team_actor_20260928/full/evaluation/records/{before,after}/dense24_{env}.json
```

| env seed | 复用 policy seed | 新 policy seeds |
|---:|---:|---:|
| `875101` | `875201` | `876201`, `876202` |
| `875102` | `875202` | `876203`, `876204` |
| `875103` | `875203` | `876205`, `876206` |
| `875104` | `875204` | `876207`, `876208` |

因此新 sampling 为两臂共 `16` 条 rollout（每臂 `8` 条），总步数上限 `28800`；复用的八条
原 policy records 不重跑。所有 env seed、policy seed、完整 request ledger、终态和不利结果
均保留，不表现差早停、不换 seed、不扩预算、不重跑。

## 3. 报告与技术边界

分别报告全部 `12` 对配对结果与新 sampling-only 的 `8` 对结果，至少包含 aggregate 与按
场景的 `normal/emergency/failed`、安全计数、unresolved、mean/p95/max wait、low-SOC wait
和 reward。before/after 的物理 arrival、SOC、service 与安全约束保持原配置；不消费 heldout，
不进入 formal qualification，task-neutral、task priority、proposal、SafetyFilter 和
`PPO_ALLOWED=false` 边界不变。

先完成定向 tests 与 server check-only；主审通过后才运行唯一 full。当前尚未启动，不写 PID、
exit 或性能结果。只有主审基于重复结果决定独立验证或继续修复；本协议不证明新物理泛化，也
不预断稳定退化的根因。
