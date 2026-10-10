# P3 Task-neutral Episodic Learning Protocol

日期：2026-09-29。状态：V1 critic-only warmup、V2 `warmup_scaled` 与唯一 full 均已完成；`learning_check`
为 `CHECK_ONLY_OK exit0`。本协议登记一次有限周期 return 目标一致性的 development 实验；不自动放行
formal、heldout、P4 或全局 `PPO_ALLOWED`。

## 1. 动机与历史边界

每步 `dt=1 s`。原 joint-team 使用 `gamma=.99`、`lambda=.95`，而评估 raw 累计 reward 不折扣。
本轮将有限周期未折扣 return 统一用于 critic target 与 PPO，PPO 固定 `gamma=1`、`lambda=1`；critic-only
warmup 使用完整 episode 零 bootstrap MC target，不使用 GAE lambda。历史 `13a9` full-episode MC 使用旧 world-mean 聚合（早于
`fb1a9a6`），并加载 `gamma=.99` 的 `b136`；它不是“正确 actor 加权且 critic 目标一致”的
试验，因此不把 gamma 视为唯一退化根因。

一个仅用于说明 objective 的例子是：正常 touchdown terminal reward `+25` 若在 `.99` 下延后
`60 s`，折扣约为 `25*.99^60≈13.68`，低于应急 touchdown 净 `+17`。这不是实际轨迹的因果
证明，完整 reward 还包含等待与 SOC 项。有限与折扣 return 的一般原理参见
[Spinning Up RL introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)。

## 2. Critic-only warmup

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_episodic_learning_20260929
```

先用 source `3fe7` 的六条既有 baseline tapes，合计 `4775` steps、`84` requests，执行一次
`gamma=1`、完整 episode 零 bootstrap MC target 的 critic-only warmup：`20` epochs、learning rate
`3e-4`、minibatch `64`、seed `874501`，actor 完全冻结，禁止 PPO。V1 已 `COMPLETED exit0`；raw
MSE `34524.459→22900.692`、MAE `149.512→115.898`、EV `.01450→.05077`，dense24 after
MAE 为 `196.358/137.696`。actor bitwise unchanged，无 PPO；V1 warmup 输出原封保留。主审认为拟合不足，
未准入 PPO；这些是 V1 warmup 的数据内拟合事实。

### 2.1 V2 scaled warmup registration

V2 只修复数值单位：固定 `reward_scale=.01`、`gamma=1`、`lambda=1`。所有 reward 统一乘以正常数
`.01`，不改变相对权重或最优期望累计目标；critic 输出使用同一 scaled unit，初始 critic output head
按 `.01` 缩放以维持 raw-value 解释。warmup targets、PPO step rewards/value/return 与 checkpoint
metadata 使用一致的 scaled unit；evaluation 仍报告 raw reward。

V2 的 check-only 与 warmup 输出分别为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_episodic_learning_20260929/warmup_scaled_check
/home/user/training_outputs/terminal/p3_task_neutral_episodic_learning_20260929/warmup_scaled
```

两者仍使用同六条 tapes、`4775` steps/`84` requests、`20` epochs、`lr=3e-4`、minibatch `64`、
seed `874501`，先完成 tests/check-only，经主审 review 后只运行一次 `warmup_scaled`；不循环拟合调参。
V2 `warmup_scaled` 已 `COMPLETED exit0`，使用 `gamma=1`、`reward_scale=.01`；actor/noncritic
bitwise unchanged，无 PPO、无 Teacher。normalized MSE/MAE 为 `3.4524459/1.4951182→.05620785/.11677936`，
raw MSE/MAE 为 `34524.459/149.511819→562.078547/11.677936`，EV 为 `.01449966→.95482847`。
这只表示同数据上的价值拟合变化，不表示泛化或降落改善。warmup+pilot 定向 tests 为 `24`、learning
定向 tests 为 `19`，联合 `5` 文件共 `83 passed`，compile 通过。`learning_check` 已
`CHECK_ONLY_OK exit0`，配置为 `gamma=1`、`lambda=1`、joint-team、`reward_scale=.01`，source 为
`3fe7`，warmup checkpoint SHA256 为
`a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710`，`runtime=0`。主审已批准唯一
full：launcher PID `3752129`、child PID `3752139`，cmdline/cwd 已核对；最终 `COMPLETED exit0`、
`technical=true`，共 `36` rollouts/`864` requests/`27973` steps，`6` calls、`1754` accepted/`0`
rejected。六批 MC 均 match，最大误差 `4.768e-7`，units/q/freeze/optimizer 全部通过；最终 checkpoint
SHA256 为 `de3d169bb6f69b1072a901123fda28a7f545b7dfaab9b1dcb687289dca881848`。首个 rollout 为 env
`877101`/policy `878201`、`725` steps、`24` completed、`terminal=true`、reward `490.6230053838741`；
technical/lifecycle/weights 正常，safety `0`。

每个 evaluation snapshot 均为 `96` requests、failed/unresolved/safety 均为 `0`。before 为
`80 normal / 16 emergency`，mean/p95/max wait `124.589624/281.307806/361.183776 s`，low-SOC
`119.394054`、raw reward `1810.519819`。batch2 after 为 `83/13`（normal/emergency），mean/p95/max
`139.976895/327.301206/399.940214 s`，low-SOC `110.448643`、raw reward `1800.730385`，normal
deltas `[2,1,1,-1]`；batch4 after 为 `80/16`，mean/p95/max `142.946290/326.148072/692.814708 s`，
low-SOC `118.926817`、raw reward `1776.482964`，normal deltas `[1,-1,0,0]`；batch6 after 为 `74/22`，
mean/p95/max `140.979419/340.844414/692.814708 s`，low-SOC `116.6394`、raw reward `1731.358232`，
normal deltas `[-1,-2,-1,-2]`。

heartbeat `episodic-ppo-repair` 已删除。主审不保留 `de3d` 最终 candidate，不另挑 batch2 作为稳定收益；
joint-team、完整 return 与单位修正代码保留，退化未解决。下一步仅对首批六条既有轨迹做梯度离散度诊断：
`6` replays/`4497` steps、`0` sampling/optimizer/PPO/evaluation，不构成新训练准入；原失败结果全部保留。

## 3. 后续连续 PPO development

V2 `warmup_scaled` 后以同一 `3fe7` actor 初始化，fresh Adam 持续六批；不从 `b114` learning-curve checkpoint
继续。完全复用已消费的 training seeds `877101..877136`、policy seeds `878201..878236`、
permutation seeds `878001..878006`，以及第 `2/4/6` 次 evaluation 的 env seeds `875101..875104`
与 policy seeds `875201..875204`。六批各六条完整 native on-policy rollouts，共 `36` rollouts、
`864` requests、training 上限 `64800` steps；evaluation 为 `12` 条新 rollouts、上限 `21600`
steps，baseline 四条 records 复用原 before 路径，均为已消费 development 证据，不是 fresh cohort。

PPO 除 `gamma=1`、`lambda=1` 外沿用 learning-curve 的 joint-team 参数、q source 与 anchors、
freeze 范围、reward、安全、task-neutral 和 proposal 语义；`C` 仍只按当前 batch 的
`totalactor/steps` 计算。只训练 `gate_adapter` 与 critic，proposal/actor encoder/gate head/
temperature 逐位冻结。

## 4. 准入、停止与结果边界

执行顺序为：本地 tests → V2 `warmup_scaled` check-only → 主审 review → 一次 `warmup_scaled` →
`learning_check` → 主审授权唯一 full。V1 warmup 已完成但未准入 PPO；V2 `warmup_scaled` 已完成，
`learning_check` 为 `CHECK_ONLY_OK exit0`，唯一 full 已 `COMPLETED exit0`；无 Teacher、
无 heldout，`PPO_ALLOWED=false`，
后续 formal 不自动放行。技术异常、安全违规或 accepted update 为 `0` 时停止；不因性能差重跑、
换 seed 或扩预算，保留全部结果。该实验是控制条件改变的 development run，不是 fresh 验证、
单因果消融或已解决退化的证据；不声称问题已解决。

## 5. Gradient-dispersion diagnostic terminal result

gradient diagnostic full 已 `COMPLETED exit0`，输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_episode_gradients_20260929/full
```

该 run 对首批六条既有 episodic tapes 做了 `6` exact replays、`4497` steps、`144` requests，记录
`17825` aggregate actor decisions 与 `3656` shared actor worlds；per-episode `actor_steps`
`[558,642,626,556,605,669]` 是 actor-world steps（合计 `3656`），不是 individual actor counts。MC、aggregate-sum、model-unchanged 均为
`true`，optimizer 为 `0`。coherence 为 `.50597656`，mean vector norm 为 `.01370283`，vector SE 为
`.01139388`；episode-total cosine 为 `[.48855,-.45577,.84693,.54055,.82063,.27637]`，leave-one-out
最低 cosine 为 `.84912`。release `p<1e-4` 占 `.5601122`，`p>1-1e-4` 占 `.0120617`；critic MAE
为 scaled values `[.564079,.476808,.404775,.431144,.733960,.555642]`，raw 值需除以 `.01`。
主审解释为存在轨迹间分歧，但不是单 episode 决定方向；六条样本不证明根因或统计显著。原失败结果
保留，task-neutral、安全、heldout 与 `PPO_ALLOWED=false` 边界不变。

## 6. Next larger-batch development protocol

下一步已登记 [larger-batch protocol](p3_task_neutral_larger_batch_protocol_20260929.md)，用于更大独立
批次测试；当前尚未启动，不把该选择表述为根因已证或问题已解决。
