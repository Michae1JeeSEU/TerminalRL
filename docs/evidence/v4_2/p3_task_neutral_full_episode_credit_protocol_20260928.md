# P3 Task-neutral Full-episode Credit Repair Protocol

日期：2026-09-28。状态：唯一 full 已完成。本协议只登记一次有界 development credit
repair；不自动放行 formal、heldout、P4 或全局 `PPO_ALLOWED`。

## 1. 固定起点与训练数据

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_full_episode_credit_20260928
```

恢复 `3fe7` actor 与 `b136` critic，作为同一 task-neutral baseline control；不从已拒绝的
`c961` checkpoint 继续。复用 joint-batch 的六个原生 on-policy training cells：
`(873103,873103)`、`(873106,873106)`、`(874601,874701)`、`(874602,874702)`、
`(874603,874703)`、`(874604,874704)`。六个 cell 的 development 数据均已消费，
不视为独立或 untouched cohort。

六个 cell 各收集完整 terminal episode，收集期间不更新参数；训练上限为 `144` requests、
最多 `10800` steps。所有 seed、终态和不利结果都纳入记录，不因结果不利而重跑。

## 2. Full-episode return 与唯一 update

使用 `gamma=1`、`lambda=1`，return 等于各 episode 实际后缀 reward 和；episode 边界不互相
bootstrap。该设置检验有限周期目标对齐，不声称它是唯一 root cause；已有 `b136` value
可能不适配长回报并增加方差，但 `lambda=1` 不依赖 bootstrap credit。

其余参数保持 joint-batch 原值：

```text
learning_rate=3e-5, ppo_epochs=2, batch_size=32, clip=.1
value_coef=.5, entropy_coef=.001, target_rollout_kl=.01
```

只训练 `gate_adapter` 与 critic；`proposal`、`actor_encoder`、`gate_head`、`temperature`
逐位冻结。执行一次 joint `PPO.update`，permutation seed 为 `875001`；`q` 与 KL reference
均沿用原 joint-batch 参数，不作修改。reward、安全、proposal 和 task-neutral 语义保持不变。

## 3. 配对 evaluation 与复用边界

复用绝对 root
`/home/user/training_outputs/terminal/p3_task_neutral_joint_batch_20260928/full` 下的 evaluation
records、before 六条 records、checkpoint/profile/cell 验证；不重跑 before。固定六个 evaluation
env/policy seed 对为：

| 场景 | env seed | policy seed |
|---|---:|---:|
| dense24 | `875101` | `875201` |
| dense24 | `875102` | `875202` |
| dense24 | `875103` | `875203` |
| dense24 | `875104` | `875204` |
| flow6 | `875105` | `875205` |
| flow12 | `875106` | `875206` |

新 after 只收集这六个 evaluation cells 的六条 rollout，最大 `9980` steps，仍为每臂 `114`
requests 的同一负荷映射。配对合计 `12` rollouts，其中 before 复用六条、after 新收集六条；
新环境 evaluation 只有六条。全部 seed 与结果均为已消费 development 证据，不称独立或 untouched。

报告整体与分 cell 的 `normal/emergency/failed/unresolved`、安全、mean/p95/max wait、
low-SOC wait 和 reward；不引入每 cell 或子组无退化硬门槛。若结果显示改善，再由主审选择
独立验证；本协议本身不授权扩预算或继续训练。

## 4. 先决检查与状态边界

代码提交为 `f8a8b53`，定向 tests 为 `45`（其中 joint 为 `16`）。check-only 为
`CHECK_ONLY_OK`、`exit=0`，并验证 `gamma=1`/`lambda=1`、复用六条 records、六条新 evaluation
rollouts 上限 `9980`、training 上限 `10800`，且 `env=0`、`update=0`、`runtime=0`。
首次外层直接执行不可执行 launcher 脚本，因权限不足未启动且无产物；随后改用绝对 Python 调用
launcher 成功，该次不是实验重跑。唯一 full child PID 为 `1530563`，cwd 为
`/home/user/projects/terminal`，cmdline 已核对包含 `--full-episode-credit`。最终 `exit=0`、report
为 `COMPLETED`，六个 training cells 合计 `4635` steps，执行 `1` 次 update，`290` accepted、
`0` rejected；KL mean/final/max 为 `.0004373/.0005048/.0014622`，MC match 为 `true`，最大误差
`3.0517578125e-05`，q、freeze、roundtrip 检查通过。full checkpoint SHA256 为
`13a98ce2b6289126a5831f6cda5e26e857e1a49d33ab3e606edb5051542f191e`。heartbeat 准确 ID 为
`automation`、间隔 `5 min`，已删除。保留 `829xxx` heldout，不改变 global `PPO_ALLOWED=false`，
也不自动改变 formal/P4/static qualification。

配对 evaluation 为六条复用 before 与六条新 after，单臂各 `114` requests，安全与 unresolved
计数均为 `0`：before `98 normal / 16 emergency / 0 failed`，after `93 / 21 / 0`；mean/p95/max
wait 为 `107.778/273.736/361.184→118.833/307.999/377.930 s`，low-SOC wait
`98.613→111.458 s`，reward `2202.562→2121.598`，normal deltas 为 `[-1,-1,-1,-2,0,0]`。
主审不保留该 checkpoint，也不把 `gamma=1`、`lambda=1` 视为既定修复；继续以 `3fe7` actor
作为开发起点。时间尺度仍是协议中的合理假设，但本次实验不支持问题已解决。

补充时间跨度依据只用于信用归因边界：`dt=1 s` 时，`gamma=.99` 在 `200 s` 的衰减为
`.134`、`400 s` 为 `.018`；已有 request→touchdown 统计的 normal median 为 `286 s`、
`p90=506.5 s`，emergency median 为 `426 s`。这些是请求到 touchdown 的物理统计，不是
actor 到落地的延迟，不将其当作实测 credit interval；实际奖励配置为正常 touchdown `+25`、
应急 touchdown 净 `+17`、progress 权重 `0`。旧 joint 六个 action tapes 各只保留尾部 `120 s`；仅一例可确证最后普通 actor
到 touchdown：`episode_02`、`uid352`、`arrival21`，last actor `710`→touchdown `770`，共
`60 s`，`(.99*.95)^60=.0252074`。其余 `115 normal + 28 emergency` 缺少完整 actor 历史，
不能写成总体 actor→touchdown 延迟已确认或主因已证明。当前结论仍为退化未解决；不得据此
宣称问题已解决。下一步仅做本地可枚举 toy 验证，检查 actor 是否按每 world 人数取均值会改变
team reward 梯度方向；未授权下一次环境训练。

## 5. 后续验证边界

本终态的 `gamma=1`、`lambda=1` checkpoint 不保留，full-episode credit repair 未解决退化；
完整失败结果与配对指标保持在本页。当前只登记 actor aggregation objective 的 toy 证据与
有界验证协议，见 [joint-team actor aggregation protocol](p3_task_neutral_joint_team_actor_protocol_20260928.md)。
该协议尚未完成定向 tests/check-only，也未授权启动环境 full。
