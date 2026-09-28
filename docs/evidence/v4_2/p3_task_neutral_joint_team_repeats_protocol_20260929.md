# P3 Task-neutral Joint-team Sampling Repeats Protocol

日期：2026-09-29。状态：唯一 full 已完成。本协议只登记一次固定模型的 sampling
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

代码提交为 `8d67f578a32a858735712c926c98bf6b38fc0a0f`，定向 tests 为 `17`。server check-only
为 `CHECK_ONLY_OK`、`exit=0`；验证 `24` unique records、`8` reuse records、`16` new rollouts，
`runtime=0`，真实 horizon `1800`、new-rollout 总上限 `28800`，before=`3fe7`、after=`ac7`。
唯一 full launcher PID 为 `2065459`，full child PID 为 `2065466`，cwd 为 `/home/user/projects/terminal`，
cmdline 已核对包含 `--joint-team-cohort`。最终 `exit=0`、report 为 `COMPLETED`、`technical_ok=true`；
两臂各 `288` requests：before `236 normal / 52 emergency / 0 failed`，after `230 / 58 / 0`，
安全与 unresolved 均为 `0`；mean wait `131.8504766→135.6220302`，p95
`315.1479841→327.7887054`，max `692.8147076→692.8147076`，low-SOC wait
`115.4893036→125.5779790`，reward `5359.1727272→5276.9198028`。new-only 两臂各 `192`
requests：before `156/36/0`，after `151/41/0`；mean `135.4809029→141.2362062`，p95
`331.1184447→333.4887312`，low-SOC `113.5369283→127.2227828`，reward
`3548.6529080→3475.6094504`。所有 freeze 与 safety/unresolved 检查通过，heartbeat `automation`
已删除。主审拒绝 `ac7` 性能候选，但保留 joint-team 数学修正，不宣称退化已解决。
只有主审基于重复结果决定独立验证或继续修复；本协议不证明新物理泛化，也不预断稳定退化的根因。
