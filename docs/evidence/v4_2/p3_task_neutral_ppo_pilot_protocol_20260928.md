# P3 Task-neutral PPO Pilot Protocol

状态：已由主审授权，尚未启动。本协议只授权一次 scoped task-neutral development pilot，
不等同于 formal/global PPO 放行；`development_pilot_authorized=true` 仅覆盖本协议明确的
3 个 production PPO updates，`PPO_ALLOWED=false` 保持。

## 1. 固定起点、配置与输出

candidate 固定为 `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt`，
SHA256 为 `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`；source `03c0`
仍是 baseline 对照。使用 baseline 同一 task-neutral profile：`task_priority_enabled=false`、
`low_level_priority_time_weight=0`、`low_level_soc_urgency_time_weight=2`，并保持其余奖励、安全和
接口配置。新输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_ppo_pilot_20260928
```

check-only 只执行一次，不执行 env step 或 update。full 只包含三个完整 on-policy training
episodes，顺序与 seed 固定为：`flow6/873101` → `flow12/873102` → `dense24/873103`。
每个 episode 只进行一次 production PPO update，参数为：

```text
ppo_epochs=2, batch_size=32, learning_rate=3e-5
gamma=.99, gae_lambda=.95, clip=.1
value_coef=.5, entropy_coef=.001, target_rollout_kl=.01
```

上述 PPO 参数沿用 V7 已用值；reward 与 discount 不改。

这里的 `3 updates` 指 `3` 次 `update()` 调用；每次按 `2` epochs、以 `32` 个 team steps
分 minibatch。实际 optimizer accepted/attempted 步数另行记录，不能写成总共 `3` 个 Adam steps。

## 2. 更新边界与 reference

只训练 `gate_adapter`、`critic_encoder`、`critic_head`；`proposal`、`actor_encoder`、
`gate_head`、`temperature` 冻结，使用 fresh Adam。固定初始 candidate 作为 `q`；
GateReference anchors 只能来自这三个 training rollout 的 task-neutral actor observations，
world 等权、world 内 actor 等权。reference coefficient `.1`、max KL `.05`、soft target `.02`、
gain `400` 固定；不得使用旧 priority observations 或 validation 数据生成训练 anchors。

禁止手工 normal 标签拟合、Teacher、probability floor，且不同时修改 reward 与 discount。
每个 episode 必须先形成完整 terminal/ledger/safety 记录且 `failed_energy=0`，才允许执行该次
update。finite、冻结不变量或有效 accepted update 为 `0` 时停止并保存失败事实，不自动 retry、
调参或扩预算。记录 critic 与单轨迹 MC/GAE、gate probability 前后、accepted updates 和 KL；
三次 update 只用于 development learnability/退化早期检查，不作为收敛或稳定泛化证明。

三个 training episodes 完成后保存 `pilot.pt`；checkpoint 还必须写入 `curriculum_id`、
`task_priority_enabled=false` 和完整实际 neutral 配置。roundtrip 与 final evaluation 必须显式
检查并使用同一 profile；通用 checkpoint loader 不会自动恢复这个新开关。随后只做一次 final evaluation：
复用同一 task-neutral baseline 的 candidate 记录，不重跑 source 或 baseline；新增 final 使用
6 个 development cells、seeds `873101..873106`，共 `84` requests，并执行既有 ActionTape exact
lifecycle replay。报告 `normal/emergency/failed/unresolved`、全请求 mean/p95/max wait、
low-SOC wait、reward 和安全字段。性能失败按事实保留，不筛选、不扩预算，candidate 取舍由主审决定。

## 3. 入口与授权纪律

```bash
python -m terminal_marl.training.run_p3_task_neutral_pilot \
  --output-dir ROOT/check_only --check-only

python -m terminal_marl.training.run_p3_task_neutral_pilot \
  --output-dir ROOT/full
```

先完成本地定向单测与输出路径验证，再通过 `ssh lab-server`（require_escalated）、既有 cwd/python 和本地 Syncthing
源码执行 check-only；主审确认 check-only 后才允许 full。不得使用 `scp` 或服务器 git。
失败产物保留，不自动重跑。formal、heldout、P4、static qualification 和全局 `PPO_ALLOWED`
均不因本协议自动放行。
