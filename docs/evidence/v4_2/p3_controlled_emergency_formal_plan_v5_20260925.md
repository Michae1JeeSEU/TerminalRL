# P3 Controlled-Emergency Formal Training Plan V5

状态：`FORMAL_ADMITTED / READY_FOR_FORMAL_TRAINING`。bounded pilot 已完成并通过技术
准入检查，正式 campaign **尚未启动**。本文冻结正式大规模 P3 PPO 的训练规格、证据边界
和启动前 check-only preflight；runner 已实现并通过定向测试，实际正式 job 尚未运行。
`FORMAL_P3_ALLOWED=true` 表示 admission readiness，不表示正式训练已经开始，也不表示
`P3_QUALIFIED` 或 `P4_ALLOWED`。

`PPO_ENTRY_ALLOWED` 已在完整 stochastic no-update entry 通过后设置为 `true`，且 entry
不等同于课程结业；`FORMAL_P3_ALLOWED` 已在 bounded pilot technical PASS 和主代理 review
后设置为 `true`。绑定 admission JSON 的 check-only preflight 已执行并通过；正式 job 尚未启动。

## 1. 冻结起点与范围

正式训练沿用 V5 admission contract 的环境、奖励、安全元数据和 trainable scope，不引入
新算法。固定起点是 repair candidate E04：

```text
SHA256: e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353
path: /home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt
```

三次正式 run 彼此独立，每次都从上述相同 checkpoint 开始，并使用 fresh Adam optimizer。
optimizer state 不跨 run 携带。低层策略定义保持 V4.2：Observation V2.1 为 `88D`，
`K=3` runtime shortlist，`late-slot=OFF`，`Teacher=false`；actor 请求 `WAIT` 或
`RELEASE` 和条件性的 speed×timing proposal，轨迹执行仍由既有安全层完成。

环境和安全边界继续使用：`energy_capacity=1500`、`min_landing_soc=.10`、
`soc_reserve=.25`、`emergency_trigger=.40`，初始 SOC 为 `Uniform(.55,.90)`，priority
为 `{1,3,5}`，以及既有 `80/40/10 m` separation/floor 边界。actor intervention fraction
上限为既有 `.05`；普通 shield correction 可以报告为非零，但 shield 或 joint fail-closed、
collision/NMAC 和其他 unsafe execution 仍是硬失败。

各场景使用既有完整回合 horizon：`flow6=1210 s`、`flow12=1570 s`、`dense24=1800 s`；
late-merge reconciliation 保持关闭（`reconcile_late_merge_slots_enabled=false`）。

奖励仍使用既有 priority time weight `1.0`、SOC urgency time weight `2.0`、threshold
`.65`。当前 summary 已保存 `reward_component_totals` 与 `reward_total`，并校验组件和的
一致性；`urgency` 是 priority 与 SOC 的合并项，不虚构独立的 priority/SOC 分解。

可训练参数只有 `gate_adapter` 与 `critic`（配置中对应
`gate_adapter`、`critic_encoder`、`critic_head`）；`actor_encoder`、`gate_head`、
`proposal_head`、`gate_temperature` 冻结。固定 PPO 超参数为 Adam `3e-5`、`epochs=2`、
`batch_size=32`、`gamma=.99`、GAE `=.95`、clip `=.1`、entropy coefficient `.001`、
target KL `.01`、maximum approximate KL `.05`。

## 2. Admission evidence and current state

V5 entry 和 bounded pilot 已按注册 seed 完成。entry manifest 为
`/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/entry/manifest.json`
（SHA256 `7c3536ac2d3d977866993c2a072a251c6eac480c38955f9cd7b9da6c770ceec5c`）。pilot
manifest 为
`/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/pilot/manifest.json`
（SHA256 `4fc0dba78e19a8d34dd78db7ce641ec11f496d4ab87ecbfc059646467b5a651b`）。pilot wrapper
exit code 为 `0`，manifest 标记 `PILOT_TECHNICAL_PASS=true`、`performance_review_required=true`。

六个 registered PPO episode 均通过 rollout/update/restore gate，记录为 `296` accepted
minibatches、`0` rejected；所有六个 arm 的 `failed_energy=0`、`unresolved=0`、完整回合和
安全计数均通过。各 arm 的 approximate KL 为
`0.000801/0.000250/0.003189/0.000246/0.001775/0.000153`，最大 accepted rollout KL
为 `0.001954/0.000968/0.006811/0.000791/0.003793/0.000395`，均在既有 target/max
纪律内；六个 checkpoint restore 均为 exact。

同初始状态 paired development 的 before/after 直接观测为：

| 场景 | before normal/emergency | after normal/emergency | priority-weighted wait (s) | low-SOC wait (s) |
|---|---:|---:|---:|---:|
| flow6 | `6/0` | `6/0` | `16.737 → 16.737` | `12.675 → 12.675` |
| flow12 | `12/0` | `12/0` | `30.972 → 31.151` | `68.006 → 68.506` |
| dense24 | `20/4` | `21/3` | `123.016 → 125.778` | `177.697 → 133.661` |

dense 配对应急数为 `4→3`，low-SOC wait 变化为 `-44.036s`，priority-weighted wait
变化为 `+2.763s`（约 `+2.25%`）。这些是继续学习所需的性能权衡报告，不构成严格改善
证明、课程结业或模型晋级；`emergency_landed` 仍是允许学习的带代价结果。

当前 admission 状态为 `PPO_ENTRY_ALLOWED=true`、`FORMAL_P3_ALLOWED=true`，但正式 campaign
仍为 **NOT STARTED**，`P3_QUALIFIED=false`、`P4_ALLOWED=false`。启动前 admission JSON 必须
使用 `P3_FORMAL_ADMISSION_DECISION_V5`，并绑定 pilot manifest/protocol SHA、E04 SHA、
实际 config snapshots 和 exact optimizer boundary；admission protocol YAML SHA256 为
`85d490dd8737b03fa59e954731b0f3446292288bec6395f4179da5e49048fa21`，本 formal YAML SHA256
为 `e3f88e52a327c64621226631c1575cd48ffec4d93b98d505c561fe55b33b22b1`。服务器实际
check-only preflight 已通过，使用 `/home/user/envs/terminal_env/bin/python`、cwd
`/home/user/projects/terminal`，exit `0`、status `CHECK_ONLY_OK`；结果写入
`/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_check_only_run0.json`，
run0 正式目录未创建。

## 3. 正式训练矩阵

正式规模固定为 **3 个独立 run × 每 run 240 个 episode = 720 个 episode**。每个 run 的
episode 调度重复下列四格模式 60 次：

```text
[flow6, dense24, flow12, dense24] × 60
```

其中 `flow6` 和 `flow12` 是旧自然流 anchor，占 50%；`dense24` 占另外 50%。这部分比例
是训练调度规格，不是通过 heldout 结果回调得到的阈值。

| run index | registered train seeds |
|---:|---|
| 0 | `850001..850240` |
| 1 | `851001..851240` |
| 2 | `852001..852240` |

每个 episode 都要求 full-cycle/full-horizon 记录；每个被接受的 episode 保存 checkpoint。
标准 checkpoint 状态应包含 policy、optimizer、RNG、timestep、episode index、protocol
digest 和 source SHA。正式 runner 已实现并通过定向测试，但尚未提供可自动续训的 resume
路径；恢复只能在人工核验“最后一个完整 episode”后进行，不能因中断自动重跑或替换 seed。
正式 job 启动前已执行 admission-bound check-only preflight，确认当前配置快照、optimizer
边界、协议 SHA 和 E04 source SHA 一致；正式 job 尚未启动。

## 4. Development 与 heldout 边界

Development 使用六个已登记 seed 和场景顺序：

| seed | 场景 |
|---:|---|
| `853101` | `flow6` |
| `853102` | `flow12` |
| `853103` | `dense24` |
| `853104` | `flow6` |
| `853105` | `flow12` |
| `853106` | `dense24` |

正式训练前先记录 paired baseline，之后在第 `40/80/120/160/200/240` 次 update 后按当前
策略再次记录。每个 paired record 使用同一组明确保存的初始状态；这种 before/after
development seed reuse 是有意设计，不能当作独立 heldout 证据，也不重复消费新训练 seed。
Development 必须保留完整回合和全部请求终态核算；若 development 出现 full-resolution
failure，暂停正式 run，交由主代理审查。

`854001..854012` 共 12 个 seed 只为后续独立 qualification protocol 保留。本次准入准备
不消费正式训练或 heldout seeds；正式启动时按注册顺序首次消费，失败后禁止自动重跑或替换
seed，也不能把 heldout 用作调参数据。

## 5. 终止、结果和报告

以下情况停止当前正式训练或阻止该 update：安全失败、能量耗尽/`failed_energy`、请求
likelihood 失败、非有限数值、update integrity 失败、冻结参数改变、proposal/temperature
改变，以及 unsafe 或 fail-closed 执行。正式 training episode 的自然 horizon 截断和相应
`unresolved` 只作为训练报告字段保留，不自动改写成普遍安全失败；development/evaluation
中的 `unresolved` 或 full-resolution failure 是硬失败，并按前述规则暂停正式 run。

账本必须使用实际 enum：`completed`、`emergency_landed`、`failed_energy`、`unresolved`。
经过安全层认证、没有能量耗尽并完成应急着陆的 `emergency_landed` 是可计价的学习结果，
本身不自动停止 PPO。每类负荷都要报告正常完成率、controlled-emergency 数量与比例、
priority 分层等待、low-SOC 等待、failed-energy/unresolved 计数，以及完整请求分母。

dense `<=10%`、sparse `<=5%` 是后续 qualification 的 provisional research targets，
不是正式训练的自动 stop 条件，也不为历史或本轮结果追溯 PASS。正式训练不自动晋级模型，
不自动设置 `P3_QUALIFIED` 或 `P4_ALLOWED`。

## 6. 授权与启动边界

本计划已获得明确的 admission decision：`FORMAL_P3_ALLOWED=true`，并且该决定绑定已通过的
bounded-pilot report、当前 runtime config 和上面列出的 source checkpoint SHA。正式 runner
只能接受阶段为 `P3_FORMAL_ADMISSION_DECISION_V5` 的 admission JSON，其中必须同时提供
`initial_checkpoint_sha256`、pilot manifest 路径及 SHA、admission protocol SHA、formal
protocol SHA、`actual_config_snapshots` 和 exact `optimizer_boundary`。当前状态为：

```text
PPO_ENTRY_ALLOWED=true
FORMAL_P3_ALLOWED=true
P3_QUALIFIED=false
P4_ALLOWED=false
```

runner 命令已实现并通过定向测试，且 admission-bound `--check-only` 已验证通过；本轮 formal
campaign 仍为 **NOT STARTED**。以下是 registered run 0 的正式启动命令（服务器 cwd
`/home/user/projects/terminal`，Python `/home/user/envs/terminal_env/bin/python`，线程环境与
pilot wrapper 一致）：

```bash
cd /home/user/projects/terminal
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/user/envs/terminal_env/bin/python -m terminal_marl.training.run_v4_2_p3_controlled_emergency_formal_v5 \
  --run-index 0 \
  --admission-report /home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_admission_decision.json \
  --checkpoint /home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt \
  --output /home/user/training_outputs/terminal/p3_controlled_emergency_formal_v5_20260925/run0
```

run 1/2 使用相同的 admission/checkpoint 参数，将 `--run-index` 和末尾输出目录分别改为
`1/run1` 或 `2/run2`；本轮尚未启动任何正式 run。

已执行并通过的 `--check-only` 命令为（`N=0`；`--output` 保持正式 run 路径，检查结果由
标准输出重定向到 admission 输出文件）：

```bash
cd /home/user/projects/terminal
/home/user/envs/terminal_env/bin/python -m terminal_marl.training.run_v4_2_p3_controlled_emergency_formal_v5 \
  --run-index 0 \
  --admission-report /home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_admission_decision.json \
  --checkpoint /home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt \
  --output /home/user/training_outputs/terminal/p3_controlled_emergency_formal_v5_20260925/run0 \
  --check-only \
  > /home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_check_only_run0.json
```

输出根目录固定为：

```text
/home/user/training_outputs/terminal/p3_controlled_emergency_formal_v5_20260925/runN
```

正式训练规格文件为
`terminal_marl/configs/low_v4_2/p3_controlled_emergency_formal_v5.yaml`。

## 7. Pilot observed resource reference

bounded pilot 的服务器控制记录从 `2026-09-25 04:05:49.320` 开始，到状态写入
`COMPLETED` 的 `04:18:13.368`，两次控制时间戳相差 `744.048s`。该 pilot 包含
`6` 个 PPO episode、`3` 个 development seed 的 before/after（共 `6` 个 development 回合）
和全部 action tape 落盘；
这只是同一 workload mix 的观测参考，不是正式 720-episode campaign 的测量保证。

pilot 输出总量直接观测为约 `288,014,792` bytes，其中 `12` 个 action tape 合计
`281,128,950` bytes，`6` 个 checkpoint 合计 `6,181,738` bytes（约每个 `1.03 MB`）。
按正式规格的 `720` 个 training episode 加 `126` 个 development episode 做粗略规划，
串行 wall-time 约 `15–25h`；若三个独立 run 并行且 CPU/磁盘资源允许，约 `6–10h`。这些
是基于 pilot 的粗略估计，不是实测保证；硬失败、暂停复核、资源竞争和实际 tape 大小都会
改变终态时间。服务器与同步本地镜像各预留约 `25–35 GB` 磁盘空间。

正式 YAML 的 runtime/disk estimate 字段只引用本节，后续不因更新估算而改变冻结 protocol
内容；formal admission-bound check-only 已通过，正式 job 尚未启动。

## 8. Seed collision check

在创建本计划和配置前，已对现有 `terminal_marl/configs`、`docs/evidence` 和
`docs/CURRENT_STATUS.md` 搜索全部新登记的 `850001..850240`、`851001..851240`、
`852001..852240`、`853101..853106`、`854001..854012`。在这些已检查路径中未观察到碰撞；
这不是对仓库外部或未检查路径的全局碰撞声明。本次准入准备没有消费正式训练或 heldout
seeds；正式启动时按注册顺序首次消费，失败后禁止自动重跑或替换 seed，也不能追加未登记
seed。
