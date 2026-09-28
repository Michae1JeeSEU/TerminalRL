# P3 Task-neutral Joint-batch Development Protocol

日期：2026-09-28。状态：check-only 已完成，唯一 full 已终态；六个 training cells 已完成，
已执行一次 joint update，配对 evaluation 已完成。本协议只记录一次有界
development fit 与配对 evaluation，不自动放行 formal、heldout、P4 或全局 `PPO_ALLOWED`。

## 1. 固定起点与数据

恢复 `3fe7` actor；允许使用 `b136` critic warmup 作为价值初始化，但 actor 必须保持 bitwise
`3fe7`，不得从已拒绝的 `fa0e` 继续。代码核查显示 `diagnostic_step_from_capture` 硬编码
`stochastic_behavior=False`，因此不把诊断 capture 数据重标为训练数据，也不把旧 replay tapes
直接作为本次 update 输入。

本次收集六条原生、完整、随机的 on-policy training rollouts，固定 env-policy seed 对为：
`(873103,873103)`、`(873106,873106)`、`(874601,874701)`、`(874602,874702)`、
`(874603,874703)`、`(874604,874704)`。初始为 `b136` critic 与 `3fe7` actor，六次收集期间
参数不变，之后合成为一个 batch 并执行一次 joint `PPO.update`。旧六条 tapes 仅作参照；其已知
步数 `777+830+741+775+754+758=4635` 不作为本次实际预算。episode 边界不得互相 bootstrap，
运行上限由现有 config horizon 在 check-only 中明确，不另行增加样本。

## 2. 唯一 fit 边界

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_joint_batch_20260928
```

代码提交 `c38071f` 对应的服务器 check-only 已完成，`status=CHECK_ONLY_OK`、`exit=0`；
相关 tests 为 `28`、joint-batch tests 为 `6`；`training_rollouts=0`、`ppo_updates=0`、
`evaluation_rollouts=0`。
check-only 确认 training 的六个 dense cells 各 `1800` horizon、合计 `10800`；固定
evaluation 的每臂预算为四个 dense cells 各 `1800`、一个 flow6 为 `1210`、一个 flow12 为
`1570`，合计每臂 `9980`、两臂 `19960`。主审已核对该 report，core `exit=0`，已批准唯一
full。full PID 为 `1132703`；cmdline 模块 `run_p3_task_neutral_joint_batch` 与 cwd 已核对。
终态结果见第4节；heartbeat `ppo` 已删除。

只进行一次 production `PPO.update`，参数固定为：

```text
ppo_epochs=2, batch_size=32, learning_rate=3e-5
gamma=.99, gae_lambda=.95, clip=.1
value_coef=.5, entropy_coef=.001, target_rollout_kl=.01
```

固定 `q=3fe7`，沿用原 anchor、KL 和 freeze 约束；只训练 `gate_adapter` 与 critic，
`proposal`、`actor_encoder`、`gate_head`、`temperature` 冻结。目的为检查跨场景 joint batch
是否改变开发行为；数据量和多样性同时变化，因此不是单变量严格因果消融，不预断 root cause。

训练 batch permutation seed 固定为 `875001`。完整终态的 `energyfailure` 用于学习并保留，
不因单条训练样本出现该结果而剔除；技术不完整或安全失败则停止。warmup 不是已保留候选，
仅提供未改动 actor 的价值初始化；`fa0e` 不续训。

代码、定向测试和 check-only 已完成，core `exit=0`，主审已批准唯一 full；终态结果见第4节。
当前只读核对奖励/折扣时间跨度与学习信号，不新训练；后续由主审决定，边界以本协议为准。
不得增加样本、继续拟合或修改
reward、gamma、lambda、安全和 task-neutral 语义。保留既有完整失败记录，不增加每场景或子组
无退化硬 gate。critic 当前 mean/max pool 缺少全局 count/time/service context；本协议不同时
修改该架构。

## 3. 固定 evaluation

只在 fit 完成并通过既有技术核对后评估以下新物理 cells；before 为 `3fe7`，after 为一次
joint-batch 新 checkpoint：

| 场景 | env seed | policy seed |
|---|---:|---:|
| dense24 | `875101` | `875201` |
| dense24 | `875102` | `875202` |
| dense24 | `875103` | `875203` |
| dense24 | `875104` | `875204` |
| flow6 | `875105` | `875205` |
| flow12 | `875106` | `875206` |

共 `12` rollouts，两臂各 `114` requests。报告整体与分场景的 `normal/emergency/failed/unresolved`、
安全、mean/p95/max wait、low-SOC wait 和 reward；所有 seed 与失败结果纳入。验收由主审综合
normal、failed、安全和 wait 分布，不引入逐场景或子组无退化门槛。

启动前先做简短项目文本冲突核对；若发现冲突，返回主审，不擅自替换 seeds、起点或边界。
本协议不消费 `829xxx` heldout，不构成 formal qualification，安全约束保持不变。

## 4. Full 终态记录（2026-09-28）

full report 为 `COMPLETED`、`technical_ok=true`、`full.exit=0`。六条 training rollouts 实际合计
`4635` steps，执行 `1` 次 PPO update，`290` accepted、`0` rejected；KL mean/final/max 为
`.00053613/.00039126/.00407790`，q、freeze、roundtrip 检查正常。`joint_batch.pt` 的 SHA256
为 `C961512CD5EDD5A4AE191A8F29D9D799AB971B66C8F44E53C333889F10D6EA0E`。

配对 evaluation 共 `12` rollouts、两臂各 `114` requests，安全计数为 `0`。before 为
`98 normal / 16 emergency / 0 failed`，mean/p95/max wait 为
`107.7779/273.7358/361.1838`，low-SOC wait `98.6126`，reward `2202.5621`；after 为
`97 normal / 17 emergency / 0 failed`，mean/p95/max wait 为
`113.1974/306.1524/414.6886`，low-SOC wait `91.7405`，reward `2178.6727`。六个 cell 的
normal deltas 为 `[1,0,0,-2,0,0]`。

主审不保留该 `c961` checkpoint 候选，继续以 `3fe7` actor 为起点；训练退化未解决，保留全部
失败证据。heartbeat `ppo` 已删除。当前只读核对奖励/折扣时间跨度与学习信号，不新训练，后续
由主审决定。
