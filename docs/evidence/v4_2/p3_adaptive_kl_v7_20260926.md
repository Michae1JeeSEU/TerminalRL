# P3 adaptive-KL V7 准备记录（2026-09-26）

本页记录 V7 formal runner、协议、adaptive probe 终态和 formal 启动边界。它不替代 P3 qualification，也不把 probe checkpoint 自动晋级为正式模型。

## 当前状态

- 用户目标是完成有效的三组独立 `240` episode formal PPO，并在训练结束、模型冻结后对同一场景集合执行评估。
- 当前 adaptive probe 为 run0/run2 各 `8` 个回合，输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_probe_v1_20260926/`，两组均 `8 UPDATED / 0 SKIPPED_KL / 0 HARD_FAILED`、`8/8` restore exact；accepted minibatches 为 run0/run2 `298/332`，最终 reference KL 为 `.0077864/.0133243`。
- probe 训练诊断各为 `132` requests，合计 `264`：run0 `110 normal + 22 emergency_landed`，run2 `117 normal + 15 emergency_landed`，合计 `227 normal + 37 emergency_landed`，`0 failed_energy`、`0 unresolved`；before/after development 均 full-resolution 且 hard failure 为 false。dense24 paired 为 run0 emergency `1→5`、priority wait `+15.6374s`、low-SOC wait `+40.9181s`，run2 emergency `3→1`、priority wait `+5.8564s`、low-SOC wait `+2.5737s`；稀疏 flow wait 有改善。该结果支持 primary 的有效更新审查，但不表示性能全面改善，也不晋级 probe checkpoint。
- formal campaign 已终态：run0 `FAILED exit1`，`80 processed / 80 UPDATED / 0 skipped`、`3954 accepted minibatches`；run1 `COMPLETED exit0`，`240/240 UPDATED`、`11737 accepted minibatches`；run2 `COMPLETED exit0`，`240/240 UPDATED`、`11808 accepted minibatches`。三 run 训练请求合计 `9240`（`8211 normal + 1029 emergency_landed`），训练 `failed_energy/other/unresolved=0`；development 的失败请求单独记录。全部 `560` updates 的 gate delta 均大于零，训练 safety 计数为零，restore/RNG false 为零；reference KL 最大值（run0/run1/run2）为 `.0243685003/.0255418718/.0265666898`，rollout KL 最大值为 `.0099292190/.0099943196/.0099836820`。`P3_QUALIFIED=false`、`P4_ALLOWED=false`，不自动恢复 run0。
- formal 三 run 已终态；全请求 lifecycle 回放已完成：manifest `/home/user/training_outputs/terminal/p3_adaptive_kl_terminal_dev_metrics_20260926/manifest.json`，SHA256 `7188f7160511446a91cb873ed5cd607378eede3bc00d567467cf25f26496f3a1`；`24/24 COMPLETED`、exit `0`、error `0`、validation error `0`，四组均为 `84/84` terminal rows、missing `0`、exact/ledger 一致。`smoke_v2` 为 `1/1 COMPLETED`，`6` requests/`6` terminal rows、missing `0`、ledger/replay/outcome 全通过、exit `0`；首次 smoke 输出不作为本次指标证据。本轮仅重放既有 tape，不是新 policy rollout，不自动恢复。
- 冻结协议文件为 [`p3_adaptive_kl_v7.yaml`](../../../terminal_marl/configs/low_v4_2/p3_adaptive_kl_v7.yaml)，本地 SHA256 为 `d2bbd3a3361abb0589099fd2096b8f404f61c34bf94db10a958de99fc3f14a53`。

## V7 formal campaign 终态（2026-09-26，全请求 lifecycle 回放已完成）

formal 输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/`。三 run 均已退出：

| run | status | processed / UPDATED / SKIPPED | accepted minibatches | training requests（normal / emergency_landed / failed_energy / other / unresolved） |
|---|---|---:|---:|---:|
| run0 | `FAILED exit1` | `80 / 80 / 0` | `3954` | `1320 / 1185 / 135 / 0 / 0` |
| run1 | `COMPLETED exit0` | `240 / 240 / 0` | `11737` | `3960 / 3518 / 442 / 0 / 0` |
| run2 | `COMPLETED exit0` | `240 / 240 / 0` | `11808` | `3960 / 3508 / 452 / 0 / 0` |
| 合计 | — | `560 / 560 / 0` | — | `9240 / 8211 / 1029 / 0 / 0` |

上表合计 accepted minibatches 不做求和展示；各 run 数值以对应 manifest 为准。所有 `560` 个 update 的 gate delta 均大于零，训练 safety 计数为 `0`，restore/RNG false 为 `0`。reference KL 最大值（run0/run1/run2）为 `.0243685003/.0255418718/.0265666898`，rollout KL 最大值为 `.0099292190/.0099943196/.0099836820`。

final development 各为 `84` requests：run0 在 episode `80` 为 `76 normal + 7 emergency_landed + 1 failed_energy`，run1 在 episode `240` 为 `76 normal + 8 emergency_landed + 0 failed_energy`，run2 在 episode `240` 为 `72 normal + 12 emergency_landed + 0 failed_energy`；baseline 为 `76 normal + 8 emergency_landed`。run0 的 development failed-energy 单独计入 development，不混入上表训练请求的零 failed-energy 统计。

对应 checkpoint SHA256 为：run0 episode `0080` `ad42e9a72cac4943834041482e93adc34f123c851c4297f7462381fe0d52ff4e`；run1 episode `0240` `db74245759653bb61f8d17b0071cf9a04c2e3334e019a63cd99dff988cc23b3a`；run2 episode `0240` `4797ae2eea5da0e60a4b24e2fd0a60a6860c37b8efe1c77c0366a33a2d8fb5f9`。checkpoint 均位于上述 formal root 的对应 run 目录。

主审结论为可学习性成立，但三 seed 全通过不成立；目前没有性能全面提升证据，`P3_QUALIFIED=false`、`P4_ALLOWED=false`，不自动恢复 run0。原 `64` episode final evaluation 的前置条件未满足，未消费 `880xxx` 或既有 heldout。全请求 lifecycle 回放已完成：manifest `/home/user/training_outputs/terminal/p3_adaptive_kl_terminal_dev_metrics_20260926/manifest.json`，SHA256 `7188f7160511446a91cb873ed5cd607378eede3bc00d567467cf25f26496f3a1`；`24/24 COMPLETED`、exit `0`、error `0`、validation error `0`，四组均为 `84/84` terminal rows、missing `0`、exact/ledger 一致。`smoke_v2` 为 `1/1 COMPLETED`，`6` requests/`6` terminal rows、missing `0`、ledger/replay/outcome 全通过、exit `0`；首次 smoke 输出不作为本次指标证据。本轮仅重放既有 tape，不是新 policy rollout。source baseline 与三个 final development 共 `24` 个既有 tape 的全请求指标已完成。


四组全请求 lifecycle 指标如下（等待单位为秒；low-SOC 为 25 个样本均值）：

| policy / checkpoint | normal | emergency_landed | failed_energy | priority-weighted wait | low-SOC wait |
|---|---:|---:|---:|---:|---:|
| baseline | 76 | 8 | 0 | 86.6690576161 | 89.7792692439 |
| run0@80 | 76 | 7 | 1 | 110.881727846 | 115.732325433 |
| run1@240 | 76 | 8 | 0 | 90.1429896281 | 93.538322835 |
| run2@240 | 72 | 12 | 0 | 95.8325596837 | 92.876852833 |

主审结论为：run1 的 normal 数量持平但等待退化；run2 的 normal 数量减少 `4` 且等待退化；run0 保留 `1` 个 development `failed_energy`。可学习性成立，但性能全面提升不成立，P3 目标未完成。下一步先冻结与实际 K3 执行一致的可执行标签覆盖诊断/小 probe，不直接启动大规模训练，不开启 late flag 晋级；原 heldout 不消费。

## 固定边界与候选正则

reference gate 的候选软惩罚为

```text
L_ref = 0.1 * K + 400 * relu(K - 0.02)^2
```

其中 `K` 为固定 03c0 reference gate distribution 到当前 gate distribution 的 weighted KL。reference hard cap `.05`、rollout-KL target `.01`、fresh Adam、proposal 子图冻结、SafetyFilter 和能量 hard gates 均保持原 V6 边界。此候选不增加 probability floor，不强制 RELEASE，不按 UID/SOC 覆盖动作，也不改变 proposal 或温度定义。

V7 adaptive probe 使用 `run_p3_adaptive_kl_probe_v1`：run0 seeds `860080–860087`，run2 seeds `862049–862056`；前四个回合用于既有停止点诊断重现，后四个为原计划中尚未消费的诊断回合。probe 是 admission evidence，不能作为独立 formal 训练或 qualification 样本。

本次 probe 已完成上述计划。它保留了原 03c0 reference、proposal 冻结和安全/能量边界；primary 依据有效更新机制允许新的 formal，但保留 dense24 priority/low-SOC 等待退化作为性能审查事实，不作普遍性能改善或 P3 结业结论。

formal 计划从同一 03c0 source checkpoint、每 run fresh Adam 开始，沿用 `[flow6, dense24, flow12, dense24]×60`：run0 `870001–870240`、run1 `871001–871240`、run2 `872001–872240`。development seeds 为 `873101–873106`，按 flow6/flow12/dense24 重复两次，在 processed episode `40/80/120/160/200/240` 做 paired full-resolution evaluation；heldout `854001–854012` 与历史 `829001–829006` 不消费。

训练后 evaluation 仅在 formal 完成并冻结模型后执行：同一场景集合使用 seeds `880101–880112` 对比 03c0 与各最终 checkpoint；另登记 dense30 no-inflow 和 same-load random dense24 stress。random layout opt-in 默认 `false`，只用于后续 evaluation，不反向改变训练协议。

## 实现与检查

V7 runner 已实现 checkpoint 的 model/Adam/reference/RNG 精确恢复，保存 processed/updated/skipped counters、per-update gate delta、最终 checkpoint SHA、配置快照和实际 wall time；development cadence 以 processed episode 计数，避免 skip 后重复同一评估路径。形式上仍区分 `UPDATED`、安全 `SKIPPED_KL` 与 hard failure；连续安全 skip 的 stalled 规则不等同于 P3 qualification。

本地定向集成测试覆盖 adaptive penalty、adaptive probe、recovery、V7 formal、random layout、GateReference、Student PPO、旧 KL guard 和 runtime shortlist，共 `69 passed`。V7 专项测试为 `7 passed`。evaluation runner 提交为 `5292945`；服务器 evaluation 定向测试为 `7 passed`，source-only 两步 smoke 为 `SMOKE_COMPLETED`。formal server preflight/test 为 `7 passed`（约 `1.38s`），三 run check-only 均为 `CHECK_ONLY_OK`，且每 run 已生成 predev summary/tapes、source/admission/cmdline 校验通过。另运行旧 SharedPPO 路径得到 `8 passed / 1 failed`；失败原始错误为：

```text
TypeError: SharedPPO.__init__() got an unexpected keyword argument 'policy_distribution'
```

该旧路径错误仅作原始事实记录，未据此归因 V7 或本轮 adaptive-KL 实现。原 V6 formal、bounded recovery、受限 emergency 与既有开发证据保持历史含义，不被本页替换；当前仍不能声称 formal 已完成、策略已晋级或 P3/P4 已通过。

## V7 run0 dev80 6-state witness（2026-09-26，离线只读）

诊断文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_state_witness_20260926/state_witness.json`。原 tape `897/897` exact 且 `outcomes_match`；模型、checkpoint、live environment 与 RNG 未改变。以下同一既有 tape 的 6 个状态均为 `certified RELEASE=0`：

| time | source P(RELEASE) | episode80 P(RELEASE) | reason_counts |
|---:|---:|---:|---|
| 136 | 0.0008189225 | 0.0005967535 | 25 `dynamics_unreachable_merge_time` |
| 152 | 0.9314221 | 0.8835936 | 25 `dynamics_unreachable_merge_time` |
| 200 | 0.0873282 | 0.1432162 | 18 `dynamics_unreachable_merge_time` + 7 `predicted_separation` |
| 250 | 0.000151486 | 0.000217653 | 25 `dynamics_unreachable_merge_time` |
| 297 | 1.622394e-10 | 1.049770e-10 | 25 `dynamics_unreachable_merge_time` |
| 500 | 0.001674179 | 0.001753280 | 18 `dynamics_unreachable_merge_time` + 7 `emergency_energy_guard` |

`filter_events=0` 是诊断插桩在 reset 重建 SafetyFilter 前仍挂接旧实例造成的，不能解释为没有 safety reject；后续仅修正临时 helper 的真实 emergency reason 读取。主审结论限定为：这些状态不能只靠提高 release 概率解决，也不能由 6 个状态证明整条轨迹不存在可行窗口或 PPO 没有责任。state witness 记录时 run1/run2 为 `109 UPDATED / 0 skipped / 0 hard`；formal 已终态为 run1 `COMPLETED 240/240`（`11737 accepted minibatches`）、run2 `COMPLETED 240/240`（`11808 accepted minibatches`），CPU `99.9%`、restore/RNG 无 false；dev120 两组均完整，各为 `84` requests（`75 normal + 9 emergency_landed + 0 failed_energy + 0 other + 0 unresolved`），`hard_failure=false`；dev200 两组均完整，各为 `84` requests：run1 `76 normal + 8 emergency_landed`、run2 `74 normal + 10 emergency_landed`，能量/other/unresolved 均为零，安全检查全通过。

## V7 run0 dev80 filter witness（2026-09-26，离线只读）

filter witness 文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_filter_witness_20260926/filter_witness.json`。原 tape `897/897` exact 且 `outcomes_match`；13 条 target filter 记录（其中 10 条为 emergency）均为拒绝。`t=152` 的 3 条均为 `dynamics_unreachable_merge_time`；`t=297`、`t=302` 各为 `predicted_separation + predicted_holder_reachable_tube`；`t=497` 的 3 条及 `t=502` 的 2 条均为 `predicted_holder_reachable_tube`。主审限定结论为：已确认正常合流预约时间不可执行与应急轨迹受交通安全约束阻挡，随后连接到能量不足；正常 safety reject 不作为 SafetyFilter bug，也不据此声称 6 个状态覆盖整条轨迹。随后已完成原失败轨迹状态副本的有界 `slot-reconciliation` off/on 候选可行性对照（同 6 点、25 网格），结果见下节；保持能量与 SafetyFilter 不变。不改冻结 formal、不重启、不把 state-only 结果当 closed-loop 效果。历史 late-slot 已有 mixed 结果，不能直接开启正式训练。formal 目标未完成，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、final evaluation 尚未启动。

## V7 run0 dev80 slot-state compare（2026-09-26，离线只读）

对照文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_state_compare_20260926/slot_state_compare.json`。prefix `2` exact；full `897` exact 且 `outcomes_match`（prefix 未全终态）；`liveconfig=false`，live environment 与 RNG 未改变。

| time | off certified RELEASE | on certified RELEASE |
|---:|---:|---:|
| 136 | 0 | 0 |
| 152 | 0 | 0 |
| 200 | 0 | 0 |
| 250 | 0 | 12 |
| 297 | 0 | 6 |
| 500 | 0 | 0 |

`on` 的 `t=250` slack 为 `195.644..236.227`，`t=297` slack 为 `165.852..192.081`；其余拒绝仍来自同一 SafetyFilter/energy 检查。主审限定结论为：旧 slot 截止时间挡住了部分经实际 endpoint 重新预约后可安全执行的候选；现有 gate 在这些状态仍保持低 `P_RELEASE`，不能只打开 flag 就宣称修复或晋级。该对照未放宽安全阈值，也不是闭环效果。随后已完成已消费 `873106` dev seed、source `03c0` 与 run0 episode80 上的有界 inference-only late-slot off/on 闭环配对，结果见下节；该对照不使用新 seed/heldout、不做 PPO、不覆盖旧 outputs，也不改变 formal。`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

## V7 run0 dev80 late-slot closed-loop pair（2026-09-26，技术完成）

输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_closed_loop_20260926`。对照根状态为 `FULL_COMPLETED`，四个 cell 技术检查通过；各 `24` requests；weights unchanged、`Teacher=false`、`PPO=false`、安全 counter 为 `0`、各 tape exact。source off 为 `19 normal + 5 emergency_landed + 0 failed_energy`，source on 为 `15 normal + 9 emergency_landed + 0 failed_energy`；episode80 off 为 `18 normal + 5 emergency_landed + 1 failed_energy`，episode80 on 为 `14 normal + 10 emergency_landed + 0 failed_energy`，其余 `other/unresolved` 均为 `0`。

| cell | steps | outcomes (normal / emergency_landed / failed_energy) |
|---|---:|---|
| source off | 761 | 19 / 5 / 0 |
| source on | 881 | 15 / 9 / 0 |
| episode80 off | 897 | 18 / 5 / 1 |
| episode80 on | 825 | 14 / 10 / 0 |

四个 cell 的 full tape 均 exact；同算法 model digest 一致，之前不同算法摘要不作直接比较。该对照消除了已知单例的 energy failure，但 source 与 episode80 两组的 normal 均下降 `4`，emergency 分别增加 `4` 与 `5`，不构成晋级依据，也不改变 formal 或触发重启。旧 lifecycle 只统计 `completed_uavs`；episode80 off 漏计 `1` 个 failed（`23 rows / lowSOC=9`，on 为 `24 / 10`），旧等待指标仅保留为成功样本条件统计；同分母结论使用下文 v3 全请求数据。v3 lifecycle observer 已完成四个 tape 的 full-request terminal wait 核对，结果见文档末尾。下一步先将已验证的 failure-inclusive 逻辑固化到独立离线评估指标路径并做针对性测试，不改正在运行的 formal 依赖；随后由 primary 准备并冻结与 proposal 随机 K3 一致的外部 gate 标签/小 probe 协议，保留旧 flow/priority 表现。当前不宣称已适配或开启新训练。formal 目标未完成，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、final evaluation 尚未启动。

## V7 shortlist 与冻结 K3 语义核对（2026-09-26）

`terminal_marl/training/train_p3_gate_coverage_repair_v2.py` 的报告明确：`state_only_gate_feasibility_over_all_26_candidates` 不是 neural K3 success guarantee；collector 的 shortlist 只用于 replay，不作为 label 判据。既有 witness/slot compare 中，`t=250` 的 frozen proposal top3 为 `[11, 10, 13]`，late-on certified 集合与其相交于 `11/10`；`t=297` 的 top3 为 `[10, 14, 13]`，late-on certified 为 `[15, 22, 21, 23, 20, 24]`，无交集。runtime 使用 sampled proposal 排第一，再加最高的其余两个 proposal（`student_rollout_v4_2.py` lines `94–143`）；因此 deterministic top3 仅是诊断参考，不等同所有 stochastic shortlist。

formal run1/run2 继续使用原协议。若后续研究外部 gate 适配，标签需对齐冻结 proposal 的实际 K3 执行能力，并先进行有界采样/输出预检和小 probe；当前未启动该适配，不解冻 proposal、不增加强制 RELEASE、probability floor 或在线 Teacher，也不把 catalogue 可行性等同于闭环成功。

## V7 run0 dev80 slot lifecycle v3（2026-09-26，full-request 核验完成）

结果文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_lifecycle_v3_20260926/manifest.json`。四个 tape 均 exact；每个有 `24` 个 unique terminal rows，完整 wait、priority、initial SOC、ledger、outcome 均匹配，独立算术复核吻合。v1 漏记 completed 路径、v2 使用错误 key；两份失败产物保留为历史记录，不作为模型失败证据。

| cell | priority-weighted wait（24 rows） | low-SOC wait（10 rows） |
|---|---:|---:|
| source off | 159.201419 | 145.302365 |
| source on | 158.035714 | 158.798314 |
| episode80 off | 213.842109 | 210.452604 |
| episode80 on | 174.378916 | 137.549010 |

原 episode80 off 的成功样本条件统计为 priority `165.793583`、low-SOC `147.079037`（low-SOC 9 rows），不再与完整 24-row/10-row 结果作同分母比较。arrival15 初始 SOC 为 `.5566427132`：off wait `780.8147076`，`t=896` `failed_energy`、touchdown null；on wait `208.8147076`，`t=561` terminal `emergency_landed`。late-on 对该已知单例消除了 energy failure 并降低全请求等待，但 source 与 episode80 的 normal 均减少 `4`，emergency 比例分别为 `5/24→9/24` 与 `5/24→10/24`；不晋级、不修改正式配置、不重启 run0。formal 终态为 run1 `COMPLETED 240/240`、`11737 accepted minibatches`，run2 `COMPLETED 240/240`、`11808 accepted minibatches`，restore/RNG 无 false；dev200 两组均完整，各为 `84` requests：run1 `76 normal + 8 emergency_landed`、run2 `74 normal + 10 emergency_landed`，能量/other/unresolved 均为零、安全检查全通过；dev120 两组均为 `75 normal + 9 emergency_landed`、共 `84` requests、无 energy/other/unresolved、`hard_failure=false`。

下一步先将已验证的 failure-inclusive 逻辑固化到独立离线评估指标路径并做针对性测试，不改正在运行的 formal 依赖；随后才登记并由 primary 审查、冻结与 proposal 随机 K3 一致的外部 gate 标签/小 probe 协议，先做小样本输出检查。保留旧 flow/priority 表现，不解冻 proposal、不增加强制规则、不启动新 formal 或恢复 run0，final heldout 仍未消费。以上只记录计划，不表示公共旧 lifecycle 统计已修复或外部 gate 已适配。
## V7 lifecycle metrics 与 K3 feasibility helper 状态（2026-09-26，最终核验）

独立 `terminal_lifecycle_metrics` 模块已接入全请求 terminal wait 指标路径；final evaluator 尚未启动，legacy 成功样本统计保留，formal 依赖未修改。run0 episode80 off 的真实 tape 为 `897/897` exact，`24` requests：`18 normal + 5 emergency_landed + 1 failed_energy`；全请求 priority-weighted wait 为 `213.8421085607388`，low-SOC `10` 个样本均值为 `210.45260434162256`。该 `897` exact 核验仅由 stdout 记录，未生成新的持久化文件。

新的 `k3_gate_feasibility` 纯 helper 按实际 sampled proposal `j` 加其余最高的两个 proposal 工作；三组 metrics/eval/K3 定向测试最终 `28 PASS`，py_compile 通过。六个既有 witness 的质量值为 `0, 0, 0, 1, 8.6933744925e-10, 0`，该值基于保存概率的近似 rank，仅用于诊断，不是训练结果，也不是闭环成功概率。

聚合规则已在独立模块与测试中固化：所有 `record.requests`（含 legacy、missing、zero-row）均计入分母并保留 missing；`landingSOC` 只统计成功且 touchdown 可用的记录，failed SOC 单列为 `terminalSOC`。这不表示公共旧 lifecycle 函数已修改。

正式 run 终态为 run0 `FAILED`（`80 UPDATED`、`3954 accepted minibatches`）、run1 `COMPLETED 240/240`（`11737 accepted minibatches`）、run2 `COMPLETED 240/240`（`11808 accepted minibatches`）；restore/RNG 无 false。dev200 已完成，各为 `84` requests：run1 `76 normal + 8 emergency_landed`，run2 `74 normal + 10 emergency_landed`，能量/other/unresolved 均为零，安全检查全通过。final evaluation 未启动，不宣称 P3 结业，不重启 formal。下一步由 primary 冻结与实际 K3 随机执行一致的外部标签/小 probe 协议；fit 尚未启动。

## V7 reward 与 actor credit accounting（2026-09-26，代码事实）

`student_rollout_v4_2.py:349` 将每个 UAV 的环境 reward 汇总为 `sum(rewards.values())` 后写入 team scalar；`StudentStepV42` 与 GAE 使用 `gamma=.99`、`lambda=.95`（`student_ppo_v4_2.py:20–33,67–75`）。每个 world step 的 advantage 复制给该步的各 actor；没有 actor step 的 UAV 不进入 actor loss，但仍保留在 GAE/critic（`student_ppo_v4_2.py:197–221`），没有额外按 actor 数加权。

当前 reward 组件事实为：urgency `=-.01*dt*(clip((priority-1)/4)+2*clip((.65-currentSOC)/(.65-.25)))`，time `=-.01*dt`，approved release `+.1`、reject `-.05`，touchdown `+25`，emergency 额外 `-8`（净 `+17`），energy `-20`。后处理的 priority-weighted wait 为 `(external+internalwait)*basepriority` 的加权指标，low-SOC 按 `initialSOC<=.65` 分组；两者与 reward 不是同一目标。本段只记录实现事实，reward 本轮保持冻结，不据此单独归因性能变化。

## V7 K3 proposal coverage diagnostic（2026-09-26，完成）

完整诊断根为 `/home/user/training_outputs/terminal/p3_k3_coverage_diagnostic_v1_20260926`，manifest SHA256 为 `0a1d329a929bdaf0131cf052fddd918ad0695d0db7245fedf3eb61b798006690`。任务 `COMPLETED exit0`，包含 `6` 个 source report 与 `1` 个 witness report：`36` 个 source states、`6` 个 witness states，missing `0`；7 份报告均 exact、outcomes/RNG/model 一致，`policy_update=false`、`teacher_online=false`。source 的 off/on 各有 `22` 个 catalogue 可行状态；`q` 分布为 `q=0:14`、`0<q<.01:3`、`.01<=q<.5:0`、`q>=.5:19`。source 中没有同时满足 `q>=.5` 且 `P_RELEASE<.1` 的样本，因此不能据此表述为广泛 gate 抑制。

三个低 q 但 catalogue 可行的 source 状态为：flow6/seed873101/step187/arrival2/outer/SOC `.719587`、priority 1、`q=1.09908e-25`；dense24/seed873103/step0/arrival0/outer/SOC `.821394`、priority 1、`q=6.3502e-19`；dense24/seed873106/step0/arrival0/outer/SOC `.832284`、priority 3、`q=5.32975e-19`。witness 单列为 off `0/6`、on `2/6` 可行；`t=250` 的 on 状态 `q=1`，四个 checkpoint 的 `P_RELEASE` 为 `.000151486/.000217653/.000386301/.000515108`；`t=297` 的 on 状态 `q=8.69337449e-10`，即使 gate 为 1 也几乎选不到 certified proposal。`42` 个状态在 `4` 个 checkpoint 上共 `168` 组 proposal logits 与 source 逐组一致。`q` 是单状态 certified 尝试概率，不是最优 RELEASE 标签或闭环成功概率。

smoke_v3 位于 `/home/user/training_outputs/terminal/p3_k3_coverage_diagnostic_v1_20260926_smoke_v3`，`SMOKE_COMPLETED`、exact `69/69`，首步观测 `88D`，off/on certified 候选为 `24/25`，proposal map `0..24`，RNG/live/weights unchanged；新模块两项定向测试和编译通过。更早 v1 的 `StopAfterFirstSample` RuntimeError 被 SafetyTrace 捕获而未 exact，v2 因 proposal mapping incomplete，均保留为软件预检失败，不作为模型失败。

primary 对该诊断的限定解释是：slot 机制、少数 gate 压制和冻结 proposal 覆盖共同出现在这些观测中，不能由单独训练 gate 或 probability floor 整体解决；观测状态不代表总体泛化。reward、安全/能量边界保持不变，不重启 formal、不消费 heldout，未启动 fit；受限 proposal 适配范围待用户确认，`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

## V7 run0 dev_after_0080 outcome totals erratum（2026-09-26）

服务器 manifest：`/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/run0/dev_after_0080/manifest.json`。逐 record 复核的六个 dev seed 请求终态按 manifest 顺序为：

| dev record | normal | emergency_landed | failed_energy |
|---:|---:|---:|---:|
| 1 | 6 | 0 | 0 |
| 2 | 12 | 0 | 0 |
| 3 | 22 | 2 | 0 |
| 4 | 6 | 0 | 0 |
| 5 | 12 | 0 | 0 |
| 6 | 18 | 5 | 1 |
| 合计 | 76 | 7 | 1 |

因此该组 `84 = 76 + 7 + 1`，`other/unresolved=0`。文档此前的 `74 normal + 9 emergency_landed + 1 failed_energy` 汇总为勘误后的错误数字，现以 manifest 六条明细为准；不据此推断来源，run0 的退出/失败定性保持为开发能量 hard gate 触发。

## V7 proposal-head probe V1（2026-09-26，fit 终态；后续评估见下节）

用户已授权受限 proposal-only 适配。注册协议、collector、fit runner 的提交分别为
`350dc1e`、`170c184`、`299d3d1`；注册协议文件未修改。实际 collection manifest 为
`COMPLETED`，六个 source records 均 exact、outcomes/RNG/model invariant 通过且无 errors，
共保存 `164` 行（train `82`、validation `82`；相对最多 `192` 行缺失 `28`），rows SHA256 为
`595f644e5d7376a44fe1be98ee8bfe997d9ee4856c102050df4b8323f1004ba8`。

唯一一次 fit 使用固定 03c0 source（SHA256
`03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`），只更新
`proposal_head.weight/bias` 共 `2425` 个参数；gate、encoder、critic、temperature、SafetyFilter、
energy/reward、late-slot OFF 与 K=3 语义保持冻结。fit 在第 `12/40` 个 epoch 因 train retention
KL cap 停止：`35` accepted、`1` rejected，proposal parameter delta norm 为 `0.65643138`，
train KL 为 `0.04770959`，validation KL 为 `0.10888803`。状态为
`STOPPED_KL_CAP`，没有重试或自动调参。

proposal-head checkpoint 为
`/home/user/training_outputs/terminal/p3_proposal_head_probe_v1_20260926/fit/p3_proposal_head_probe_v1.pt`，
SHA256 为 `d7a0908b30395ddaa90cb1ffe6f2f5c0d49176e946927ea9fb5a6b44bdb0bade`。fit 的 train qK3
均值由 `0.5121951158` 变为 `0.5121951176`，validation 由 `0.4999999951` 变为
`0.4999999959`，未见有意义的整体 coverage 改善；只读复核中 source `qK3<.01` 且 C 非空的
4 条状态仍全部 top3 miss，164 条中 top3 改变 `30` 条，source-covered→candidate-uncovered 为
`0` 条。该段记录 fit 完成时的状态；六个 development seed 的 corrected frozen evaluation 见下节。

本结果只记录 proposal-only 开发试验，不能写成 formal training 完成、P3/P4 通过或 heldout 结果。
旧 p3-v7 automation 已经工具核实不可用/删除，当前不声称存在 active automation；正式训练、heldout
与后续晋级仍保持未启动/未消费状态。

## V7 proposal-head probe V1 corrected development evaluation（2026-09-26，终态）

校正后的离线评估产物为
`/home/user/training_outputs/terminal/p3_proposal_head_probe_v1_20260926/eval_summary_corrected/manifest.json`，
SHA256 为
`f1db01557c112817b1e84501236242dfec077324702d9b4b335e366fd7e78354`，状态为
`REPORTING_ERROR_CORRECTED`，对应六个 development records。原始评估 manifest 的 SHA256 为
`4c0b48d1503ea70c621a9595049c73d79e1e41b6152f7ab90f1e5e57f6f2ffaa`，source 03c0 SHA256
仍为 `03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`；原始记录已保留。
校正使用 evaluator 修复提交 `46c27db` 的离线汇总，未重新 rollout、fit 或调用 training/update
（`policy_forward_executed=false`、`training_or_update_called=false`、`environment_replayed=false`）。

| split | baseline normal / emergency / failed_energy / other / unresolved | proposal-head normal / emergency / failed_energy / other / unresolved | priority-weighted wait baseline → candidate | low-SOC wait baseline → candidate |
|---|---:|---:|---:|---:|
| overall (84) | 76 / 8 / 0 / 0 / 0 | 75 / 9 / 0 / 0 / 0 | 86.6690576161 → 92.0527228621 | 89.7792692439 → 89.6788480393 |
| train (42) | 39 / 3 / 0 / 0 / 0 | 40 / 2 / 0 / 0 / 0 | 70.6967099479 → 81.1172039290 | 62.4444731092 → 47.3267282907 |
| validation (42) | 37 / 5 / 0 / 0 / 0 | 35 / 7 / 0 / 0 / 0 | 102.9545885720 → 103.2026637351 | 115.0113887529 → 128.7731124226 |

六个 record 均 exact/aligned，weights unchanged；SafetyInvariantError、collision、
hard-separation、NMAC、uncertified-execution 等安全计数均为 `0`。校正报告明确
`admission_gates_reconstructed=false`、`checkpoint_promotable=false`、`development_screen=false`、
`P3_QUALIFIED=false`、`P4_ALLOWED=false`，且不包含 heldout。该结果将本次 proposal-only 用户授权试验记为已完成但未通过 development screen；它不表示 formal training 完成或 P3 晋级。

原评估中的 `KeyError: 'hard_gates'` 属于报告阶段错误；校正报告复用了已保存的六个原始 tape/record，未重跑环境。下一步由 primary 在再次训练前安排有界的现有状态 counterfactual 检查，分别核对 WAIT/RELEASE utility/credit 与 proposal coverage；在该检查前不自动重复 fit、改变 KL 或启动新的 PPO，且不新增 protocol/seed。
