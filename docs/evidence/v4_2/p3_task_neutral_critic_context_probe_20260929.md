# P3 Task-neutral Critic Context Probe Protocol

日期：2026-09-29。状态：`COMPLETED exit0`。本协议承接 larger-batch development 终态，
用于检查 critic 是否能利用当前状态中已有的队列、时间和服务上下文；预测误差改善只说明 critic
拟合信息变化，不等同降落性能改善，也不证明唯一根因。

## 1. 输出、起点与 replay 范围

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_critic_context_probe_20260929
```

固定 source `3fe7` actor 与 `a82` critic 初始化，沿 episodic protocol 的 checkpoint 路径加载。
复用 larger-batch 首批 `24` 条已消费 development tapes：env seeds `879101..879124`、policy
seeds `879201..879224`，合计 `18560` steps、`576` requests；前 `18` 条 fitting 为 `14072` steps，
后 `6` 条 validation 为 `4488` steps。所有 tapes 的 checkpoint 均为 `a82`，technical/lifecycle/
weights 检查为 `true`。按完整 episode 划分，validation 不参与拟合、不 early-stop、不调参。这些
tapes 已消费，属于 development，不是 heldout。probe 单 episode 上限为 `1800` steps，总 replay cap
为 `43200` steps；其中 episode07 为 `1017` steps，不沿用历史 `1000` step 限制。当前输出 ROOT
已用于本次 full；full `COMPLETED exit0`，`24` replay episodes、`18560` steps、`576` requests，
前 `18` fitting (`14072` steps)、后 `6` validation (`4488` steps)。replay/terminal/ledger/safety/
actor-bitwise/validation-excluded 均为 `true`，`0` actor/PPO/Teacher；`full.log`/`full.exit` 位于该 ROOT
级别，不在 `ROOT/full/` 内。heartbeat 已删除。

## 2. 两个 critic 与 context10

两臂均从 `a82` 初始化。legacy 臂保持现有 mean/max critic；context 臂只在 `critic_head` 第一层
追加 context10 列，新增列零初始化，以保持初始输出一致。context10 只由当前状态计算：
pre-step time/max-sim、`active/24`、`admission_count/24`、`service_count/24`、
`ungenerated_requests/24`、队列 `SOC mean/min`（空队列为 `0`）、
`queue_wait_sum/(24*300)`、`LZ service remaining sum/(nLZ*120)`、
`LZ service remaining max/120`。

禁止使用未来 SOC、action、reward 或 labels；这些是 critic 特权输入，不进入 `88D` actor。
本 probe 只检验当前状态与有限 cohort 的 critic 预测，不改变 task-neutral、reward、安全或 actor
语义。

## 3. 固定拟合配置与记录

每个 critic 使用同一 train-row 排列、`20` epochs、`lr=3e-4`、minibatch `64`、gradient clip `1`、
seed `879501`，目标为 `gamma=1` 的完整 episode MC scaled return（`reward_scale=.01`）。最终
保存两臂各自的 `20` epoch 权重，不做 best checkpoint 选择。

报告 train/validation 的整体、actor-world 聚合及逐 episode raw/scaled MAE、MSE、EV，并记录
replay exact、terminal、MC、actor unchanged 等既有检查。validation before 的 legacy all/actor
MAE/MSE/EV 为 `50.58621886/52.92280352`、`4255.2521/4407.1505`、`.818094/.713071`；
legacy after 为 `23.262293/27.636879`、`987.4543/1201.8624`、`.958943/.923804`；context after
为 `22.700983/26.337413`、`859.766/1026.6398`、`.966977/.940369`。context 逐 episode MAE 仅
`3/6` 改善，结果不一致。主审结论为 critic 重新拟合可改善跨 episode 预测，新增 `10D` 增益有限，
不证明根因，不改 actor 维度或接入新架构，退化未解决。该 probe 不进行 policy sampling、
actor update、PPO、Teacher、新 evaluation 或 heldout 消耗。

## 4. 执行边界

执行顺序为：实现与定向 tests → server check-only（`0` runtime）→ 主审审阅 → 唯一 full replay。
当前 full 正在 `REPLAYING`，已处理 `1/24` episodes，尚未 fit；heartbeat
`task-neutral-critic-context-probe` 已创建、间隔 `5 min`，exit 尚无。若 validation 预测改善，仅作为
是否考虑 PPO 的信息，不能表述为降落收益；未授权后续训练。`PPO_ALLOWED=false`、formal/heldout/P4
边界保持不变。

动机仅为当前 critic 使用 active `88D` 的 mean/max 聚合，未显式包含 count、time、service remaining，
且 queue/service 不在 world 表示中的输入充分性假设；这不是已证的唯一根因。
