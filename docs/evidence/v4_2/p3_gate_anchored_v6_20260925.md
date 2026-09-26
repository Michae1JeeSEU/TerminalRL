# P3 gate anchored V6 protocol draft（2026-09-25）

状态：`READY_FOR_SMOKE_AND_PILOT`。V6 runner 已实现，外部 V6/PPO/GateReference 测试共
`34 passed`；冻结 YAML SHA256 为
`043ae7296d838e0970aadf78667621c16596c100e92992ce4e62fc2f1e7c0cf8`。本文与
`terminal_marl/configs/low_v4_2/p3_gate_anchored_v6.yaml` 冻结下一阶段协议边界。服务器 smoke 已完成，
首次 pilot 尝试已失败并保留现场，修复后的服务器定向测试已通过，pilot restart1 已完成 technical pass；主审已批准并启动
formal `3×240`，从 `03c0` 与 fresh Adam 重新开始，不使用 pilot checkpoint。三组 check-only 已通过并完成启动验证；本轮不消费新的
heldout，不创建 heartbeat，也不修改 V5 冻结文件。

## 1. 固定来源与目的

V6 直接从 coverage gate repair checkpoint `03c0` 开始：

```text
SHA256: 03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba
path: /home/user/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/fit853401/p3_gate_coverage_repair_v2.pt
```

低层策略定义沿用 V5：Observation V2.1 为 `88D`，`K=3`，`Teacher=false`，late-slot
保持关闭，actor 输出 `WAIT` 或带条件 speed×timing proposal 的 `RELEASE`。环境、几何、reward、
安全边界和 ledger enum 继续沿用 V5：`80/40/10 m`、`energy_capacity=1500`、
`min_landing_soc=.10`、`soc_reserve=.25`、`emergency_trigger=.40`、初始
`Uniform(.55,.90)`、priority `{1,3,5}`；`completed`、`emergency_landed`、`failed_energy`、
`unresolved` 的语义不变。受安全认证、完成着陆且未耗尽能量的 `emergency_landed` 仍为可报告的
学习结果，不自动终止 PPO；安全、能量、likelihood、非有限、冻结参数和 restore 失败仍是硬停止。

V6 的变化只在 PPO update 中增加固定来源 gate 分布约束。模型架构保持原 calibrated
gate-adapter；`gate_adapter` 与 `critic` 可训练，actor encoder、base gate/gate head、proposal
head 和 temperature 冻结。新项不设 probability floor，不改 proposal，不在线调用 Teacher，
也不按 UID 或 SOC 覆盖动作。

对当前 gate 分布 `p` 与固定来源 gate 分布 `q_source`，新增正则项为：

```text
reference_gate_kl = KL(q_source || p_current)
PPO update objective adds 0.1 * reference_gate_kl
```

`q_source` 始终来自上述 `03c0` checkpoint。累计 reference weighted KL 的硬上限为
`.05`；超过上限时沿用现有 minibatch model+optimizer rollback。既有 rollout/requested-likelihood
检查、approximate-KL `target=.01 / maximum=.05`、finite 检查和 update integrity 仍全部生效。
该约束是 PPO 训练中的固定参考分布正则，不是概率 floor 或安全层替代品。

## 2. Anchor 数据边界

anchor 只由两组组成，各组权重 `.5`，每组内部对 `WAIT`/`RELEASE` 做平衡：

| group | 输入 | 合计 |
|---|---|---:|
| old train anchor | `train_flow6_846101`、`train_flow12_846102`、`train_dense24_846103`、`train_dense24_846104` 的四个 `actor_labels.jsonl` | 1529 |
| coverage original/bounded run0/run1 | `p3_gate_coverage_repair_v2_20260925/collection/{original_run0,bounded_run0,original_run1,bounded_run1}/labels.jsonl` | 1062 |

旧 anchor 的四个输入分别为 `21/66/695/747` 行；coverage 四个输入的合计为 `1062` 行。dev、V5 heldout `854001–854012` 和历史
heldout `829001–829006` 均不进入 anchor、不用于调参。offline state-only Teacher 产生的
labels 只作为数据来源；正式 rollout 仍为 `Teacher=false`。

## 3. Bounded pilot

pilot 由 `12` 个 sequential PPO episodes 组成，场景调度为：

```text
[flow6, dense24, flow12, dense24] × 3
```

注册 training seeds 为 `859201–859212`，smoke seed 为 `859000`、`max_steps=2`。pilot
development 使用 stochastic paired before/after：`flow6=859301`、`flow12=859302`、
`dense24=859303`，before 与 after 使用同一初始状态集合，要求完整回合和全部请求终态核算。

pilot 只用于检查真实 PPO 的 update/likelihood/restore/frozen/reference-KL 行为，以及
normal、controlled-emergency、priority wait、low-SOC wait 和 anchor drift。它不是 qualification，
也不把 pilot 最后 checkpoint 作为 formal 起点；formal 必须重新从 `03c0` 使用 fresh Adam。

## 4. Formal 训练注册表

formal 规格为 `3` 个独立 run、每 run `240` episodes，共 `720` episodes。每个 run 重新从
`03c0` 和 fresh Adam 开始，重复：

```text
[flow6, dense24, flow12, dense24] × 60
```

| run | registered training seeds | development seeds / 场景 |
|---:|---|---|
| 0 | `860001–860240` | 下方六个 development seed 全部执行 |
| 1 | `861001–861240` | 下方六个 development seed 全部执行 |
| 2 | `862001–862240` | 下方六个 development seed 全部执行 |

每个 formal run 都使用同一组 development seeds：`863101 flow6`、`863102 flow12`、
`863103 dense24`、`863104 flow6`、`863105 flow12`、`863106 dense24`；不为不同 run
追加 development seed。

formal development 的六个回合按 `flow6, flow12, dense24, flow6, flow12, dense24` 配置，
先做 paired baseline，之后在第 `40/80/120/160/200/240` 次 update 后重复。before/after 使用
相同初始状态集合；development full-resolution failure 或 `unresolved` 交由主审暂停审查。
training episode 的自然 horizon 截断单独报告，不把它自动改写为所有 training episode 的硬失败。

V5 heldout `854001–854012` 与历史 heldout `829001–829006` 在本协议中均保持 untouched。新
候选 seed 在启动前只做只读碰撞检查；若发现已使用，返回主审，不自行替换 seed。

## 5. 安全、保存与结果报告

硬停止保持：安全失败、能量耗尽、requested likelihood 失败、非有限值、update integrity 失败、
冻结参数改变、proposal/temperature 改变、reference source 改变、reference KL cap 超限和
checkpoint restore 失败。`emergency_landed` 只报告，不自动停止；training horizon 的 unresolved
只报告，development unresolved 是失败并暂停相应 formal run。每个接受的 episode 保存 policy、
optimizer、RNG、timestep、episode index、protocol digest 和 source SHA，并要求 restore exact。
程序照常执行自身 hard-stop、checkpoint 和 development 检查，不自动 restart。

每个 load class 报告 full-request denominator、normal rate、`emergency_landed` count/rate、
failed-energy、unresolved、priority 分层等待、low-SOC 等待、anchor gate drift、reference KL
及冻结/restore 结果。dense `<=10%`、sparse `<=5%` 只作为后续 qualification 的 provisional
research targets，不是本协议的自动训练停止或历史结果追溯 PASS 条件。

用户已明确大规模启动验证后不进行 heartbeat 或持续跟进；本协议因此不注册监控 automation，
也不授予失败后的自动重跑权限。训练完成后，另行独立评估 dense30 以及同负荷的随机环数量/安全
非均匀角度；这些评估的 seeds/config 在训练结果后确定，本轮不启动额外 stress、不消费 heldout。

## 6. 启动前只读准备事实

已观察到 source `03c0` checkpoint 文件存在；`f8203d4` 关键文件的 local/server SHA 对照一致。
在相关 P3/SOC39 输出根的 `740` 个 manifest/report/progress 文件中，V6 pilot、formal 和 smoke
seed 未发现 exact hit；local docs/source 也未发现 exact hit。这是限定路径的只读搜索事实，不构成
对仓库外或未检查路径的全球 seed 未消费保证，正式启动前仍须重复核对。

服务器准备性快照为 `28` CPU、load `0`、可用内存约 `25 GiB`、磁盘可用约 `797 GB`，未观察到
相关项目运行进程。旧 formal restart1 的实际参考为：run0 `217 accepted / 218 records`、总时长
`18,088 s`，checkpoint 间隔均值 `81.679 s`、P95 `160.114 s`；run1 checkpoint 间隔均值
`74.334 s`，run2 为 `85.380 s`。据此先给出三个 run 并行、每 run 240 episode 的粗略 wall-time
参考 `5.5–8 h`；待 pilot 和 formal 首批计时后修订，不是运行保证。

## 7. 当前状态与下一步

当前 V6 YAML 为 `READY_FOR_SMOKE_AND_PILOT`：runner 实现已完成，外部 V6/PPO/GateReference
测试 `34 passed`。修复后的真实 calibrated adapter + GateReference PPO transactional test 记录
`accepted=2`、`rejected=0`、`rollout_KL=2.0283e-9`、`reference_KL=5.6516e-9`，Adam state 非空且冻结
gates 全为 true。服务器 smoke manifest 为：

```text
/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/smoke/manifest.json
```

`859000` smoke 使用两步、无 PPO，状态为 `SMOKE_COMPLETED PASS`；checkpoint restore exact、
weights unchanged、安全计数为零。该局部 smoke 的 `unresolved=24` 未进行 full-scenario 终态核算，
不作为 formal 结果。

首次 pilot 尝试 PID `2588122`，退出码 `1`，现场根为：

```text
/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/pilot/
```

predev 的 flow6/flow12/dense24 分别为 `6/0/0`、`12/0/0`、`20/4/0`
（normal/emergency_landed/failed_energy），均 resolved 且安全计数为零；首个训练 episode
`859201/flow6` 为 `6 normal`、无 failed energy、完整 resolved。PPO 调用后完整性检查引用不存在的
`formal_v5._frozen_state_equal`，因此该次 pilot 为 `0 accepted`、无 checkpoint。原 pilot 根、日志与
现场保留。修复后的 runner 使用 `p3_legacy._frozen_state_equal`；上述服务器 `34` 项测试通过后，主审批准一次
`pilot_restart1`，沿用 `03c0`、fresh Adam 和同一组 seeds，重复不构成新的独立样本。该次 pilot 已完成
`PILOT_TECHNICAL_PASS=true`：manifest 为
`/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/pilot_restart1/manifest.json`，SHA256
为 `2545c9cb88dd86828018bc36eeee3401d894250e41278a5f40b7a86bab63634f`；12 个 episode、198 requests（181 normal、
17 emergency_landed、0 failed_energy、0 unresolved）、622 accepted minibatches、0 rejected、12 次 restore exact，
final reference KL `.0166467186`，第 11 个 episode 的最大值 `.0170518`，mean `Release source` `.35982→.34847`。
paired development 的 flow6 无变化；flow12 priority wait `+1.08689s`、low-SOC wait `+1.5s`；dense24 的 emergency
`4→7`、normal `20→17`，priority wait `114.998896→103.393995s`（`-11.6049s`），low-SOC wait
`133.919561→135.778840s`（`+1.8593s`）。全部 development 回合 full-resolved 且 `failed_energy=0`；technical pass
不等于性能指标全部改善或 P3 qualification。旧辅助 `priority_outcomes` 表对流入 UID 缺少完整映射，不能作为完整 priority 分母；
本轮 priority wait 使用 lifecycle 全 `24` rows 且 replay exact。主审已批准 formal `3×240` 沿用 `03c0` 与 fresh Adam，
不使用 pilot checkpoint。三组 check-only 通过后，于 `23:21:36/36/37 +08:00` 启动 formal run0/run1/run2；decision 为
`/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/formal_admission_decision.json`，SHA256
`a57dcba2c98ebbd573d905feec36518728fa72459f3b65a48921ee97c17d489e`。child PID 为
`2592623/2592625/2592627`，supervisor PID 为 `2592622/2592624/2592626`，输出根为
`/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/formal/runN`；三组均已生成首个
`checkpoint_episode_0001.pt`，大小分别为 `1,039,047/1,038,983/1,038,983` bytes，并进入 episode 2。
启动核验时 CPU 约 `99.9%`、stderr 为空；完成启动核验后按用户要求停止观察，不创建 heartbeat 或持续监控。按 pilot 每 episode
平均 `71.353s` 估计，240 episodes 约 `4.76h`，另加约 `42` 个 development episodes `0.49h`，含余量的参考窗口为
`5.5–8h`，预计 `2026-09-26 04:50–07:20`；该时间不代表完成保证。formal 启动记录之后的终态见下文；P3/P4 仍未通过，
不把实现测试、smoke 或 pilot technical pass 写成 PPO/P3 qualification。V5 admission/formal YAML 与历史输出保持不变；程序自身
hard-stop、checkpoint 和 development 检查照常执行，失败后不自动 restart。

formal campaign 的终态复核如下（服务器进程与退出码已复核）：run0 为 `FAILED`，`79 accepted / 80 records`，exit code `1`；
run1 为 `COMPLETED`，`240/240`，exit code `0`；run2 为 `FAILED`，`48 accepted / 49 records`，exit code `1`。run0 失败 seed `860080`、run2 失败
seed `862049` 均在首个 minibatch 遇到 `reference_kl_exceeds_max` 并 rollback，且两次仅
`accepted_updates` 为 false。三 run 的 last reference KL（run0/run1/run2）为
`.048804771/.008465138/.049026713`。全部训练 records 的 full-request 分母合计 `6078`：`5374 normal`、
`704 emergency_landed`、`0 failed_energy`、`0 unresolved`；accepted episodes 合计 `367/720`，records 合计 `369`，
其中包含两条 rollout 已完成但 update 未接受的 records；`367/367` checkpoint restore exact。11 个 development 阶段均
`hard_failure=false` 且 `full_resolution=true`。run0/run2 的 development 仅为 `0/40`，不把 episode 40 的指标称为
最终 `79/48` checkpoint 性能。run1 的 final development 两个 dense 回合分别为 normal `19→21`、`19→19`，emergency
`5→3`、`5→5`；priority wait `114.606→118.235s`、`127.983→141.995s`，low-SOC wait
`152.235→101.874s`、`105.807→109.148s`。修复后本批未出现能量耗尽；三 run 计划未完成，不能宣称跨 seed 稳定、
普遍优于 source 或 P3 通过。保留 run1 候选，不自动晋级 P4、不恢复训练，未启动 heldout 或 dense30 新实验。主审下一步记录为：
保持安全与 reference cap，先小规模验证回退步长及允许完整回滚的零更新回合恢复语义；不直接放宽 cap，也不从 I0 重训。
服务器完整复核还确认三 run `otherfailures=0`、stderr 均为 `0 bytes`、development 安全非零记录均为 `0`；run1 的 `11` 次拒绝均为
rollout KL，未发生 reference 拒绝，accepted reference KL 最大值为 `.0444951<.05`，因此下一步聚焦两次提前停止的更新恢复语义，
不直接判断 reference cap 普遍过紧。
