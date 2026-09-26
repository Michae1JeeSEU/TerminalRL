# P3 adaptive-KL V7 准备记录（2026-09-26）

本页记录 V7 formal runner、协议、adaptive probe 终态和 formal 启动边界。它不替代 P3 qualification，也不把 probe checkpoint 自动晋级为正式模型。

## 当前状态

- 用户目标是完成有效的三组独立 `240` episode formal PPO，并在训练结束、模型冻结后对同一场景集合执行评估。
- 当前 adaptive probe 为 run0/run2 各 `8` 个回合，输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_probe_v1_20260926/`，两组均 `8 UPDATED / 0 SKIPPED_KL / 0 HARD_FAILED`、`8/8` restore exact；accepted minibatches 为 run0/run2 `298/332`，最终 reference KL 为 `.0077864/.0133243`。
- probe 训练诊断各为 `132` requests，合计 `264`：run0 `110 normal + 22 emergency_landed`，run2 `117 normal + 15 emergency_landed`，合计 `227 normal + 37 emergency_landed`，`0 failed_energy`、`0 unresolved`；before/after development 均 full-resolution 且 hard failure 为 false。dense24 paired 为 run0 emergency `1→5`、priority wait `+15.6374s`、low-SOC wait `+40.9181s`，run2 emergency `3→1`、priority wait `+5.8564s`、low-SOC wait `+2.5737s`；稀疏 flow wait 有改善。该结果支持 primary 的有效更新审查，但不表示性能全面改善，也不晋级 probe checkpoint。
- primary 已允许从同一 03c0 source checkpoint、fresh Adam 启动 `3×240` formal。formal 输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/`；control 路径为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/_control/runN/{pid,status.json,exitcode,...}`。run0 child PID `2641745` 已 `FAILED exit1`，在 `80 processed / 80 UPDATED / 0 skipped / 0 training hard`、`3954 accepted minibatches` 后停止，`80/80` restore 与 RNG exact、zero-gate-delta 计数为 `0`；其训练请求为 `1320`（`1185 normal + 135 emergency_landed`、`0 failed_energy`、`0 unresolved`）。停止发生在 `dev_after_0080` 的第六个 seed `dense24/873106`，不是 training episode 80；退出由开发能量 hard gate 触发，日志未见软件异常：该 dev 记录为 `18 normal + 5 emergency_landed + 1 failed_energy + 0 unresolved`（24 requests），失败 arrival 为 `arrival_id=15`，dev80 六个 seed 合计 `84` requests（`74 normal + 9 emergency_landed + 1 failed_energy + 0 unresolved`），六个 dev seed 虽 `complete=true` 但该组 `full_resolution=false`、`hard_failure=true`；安全计数为零、weights unchanged、replay exact。最新 checkpoint 为 run0 episode 80，原输出根保留，不重启、不替换 seed 或 cap。主审随后核验原 tape `897/897` exact 且 `outcomes_match`；失败 arrival 为 runtime UID `72`（对应 `arrival_id=15`），`t=135` 进入 outer，至 `t=896` energy_depleted。outer 请求计数为 `WAIT=679`、`RELEASE=82`、`approved=0`（WAIT 不计作 rejected）；首次 outer RELEASE 为 `t=152`、SOC `.496309`，返回 `safety_reject`。在 `878` 次 emergency 函数调用中有 `120` 次真实 attempt；首次为 `t=297`、SOC `.399643` 的 safety reject，另有安全候选拒绝累计 `134`、energy 候选拒绝 `247`、LZ/dynamics `0`。`t=557`、SOC `.226309` 首次出现 energy reject；`t=573` 的 normal RELEASE 首次触发 `emergency_energy_guard`，slack `-93.7824`。上述计数使用不同分母，不能互相相加或直接作同一类拒绝率。诊断文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_dev80_diagnostic_20260926/diagnostic_v2.json`。主审当前结论仅限于：存在早期正常/应急路径被阻挡、后期能量不可行的链条；尚不能区分错过可行窗口、安全过滤过严或规划缺口，也不能直接归因于只缺训练。六-state witness 已完成；随后已完成 slot-state off/on 对照，结果见文档末尾，不反馈原轨迹或修改运行中依赖。run1/run2 child PID `2641747/2641746` 仍在运行；UTC 05:42:22 快照分别为 run1 `153 UPDATED / 0 skipped / 0 hard`（`7537 accepted minibatches`）、run2 `155 UPDATED / 0 skipped / 0 hard`（`7598 accepted minibatches`），PID/CPU 正常。此前 `97 UPDATED` 快照的请求合计作为历史记录保留：run1 `1590`（`1408 normal + 182 emergency_landed`）、run2 `1590`（`1412 normal + 178 emergency_landed`）；两者均为 `0 failed_energy`、`0 unresolved`。首个 dev40 性能回归已记录：baseline `76/8`、run0 `74/10`、run1 `73/11`、run2 `76/8`（normal/emergency_landed），dense24 priority wait 增加约 `20–49s`；配置未改变。主审已完成当前原始 tape 与闭环对照取证，尚无根因或修复结论；三 run 全部完成条件未满足，`64` episode final evaluation 尚未启动。
- heartbeat `p3-v7` 每 `20` 分钟持续跟进剩余 run 与 run0 诊断；用户已授权该持续跟进，异常或终态通知，不自动重启。
- 冻结协议文件为 [`p3_adaptive_kl_v7.yaml`](../../../terminal_marl/configs/low_v4_2/p3_adaptive_kl_v7.yaml)，本地 SHA256 为 `d2bbd3a3361abb0589099fd2096b8f404f61c34bf94db10a958de99fc3f14a53`。

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

`filter_events=0` 是诊断插桩在 reset 重建 SafetyFilter 前仍挂接旧实例造成的，不能解释为没有 safety reject；后续仅修正临时 helper 的真实 emergency reason 读取。主审结论限定为：这些状态不能只靠提高 release 概率解决，也不能由 6 个状态证明整条轨迹不存在可行窗口或 PPO 没有责任。state witness 记录时 run1/run2 为 `109 UPDATED / 0 skipped / 0 hard`；随后 UTC 05:42:22 快照为 run1 `153 UPDATED / 0 skipped / 0 hard`（`7537 accepted minibatches`）、run2 `155 UPDATED / 0 skipped / 0 hard`（`7598 accepted minibatches`），PID/CPU 正常；dev120 两组均完整，各为 `84` requests（`75 normal + 9 emergency_landed + 0 failed_energy + 0 other + 0 unresolved`），`hard_failure=false`。

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

原 episode80 off 的成功样本条件统计为 priority `165.793583`、low-SOC `147.079037`（low-SOC 9 rows），不再与完整 24-row/10-row 结果作同分母比较。arrival15 初始 SOC 为 `.5566427132`：off wait `780.8147076`，`t=896` `failed_energy`、touchdown null；on wait `208.8147076`，`t=561` terminal `emergency_landed`。late-on 对该已知单例消除了 energy failure 并降低全请求等待，但 source 与 episode80 的 normal 均减少 `4`，emergency 比例分别为 `5/24→9/24` 与 `5/24→10/24`；不晋级、不修改正式配置、不重启 run0。当前 formal 快照为 UTC `05:42:22`：run1 `153 UPDATED / 0 skipped / 0 hard`、`7537 accepted minibatches`，run2 `155 UPDATED / 0 skipped / 0 hard`、`7598 accepted minibatches`，CPU `99.9%`；`dev160` 尚未生成。dev120 两组均为 `75 normal + 9 emergency_landed`、共 `84` requests、无 energy/other/unresolved、`hard_failure=false`。

下一步先将已验证的 failure-inclusive 逻辑固化到独立离线评估指标路径并做针对性测试，不改正在运行的 formal 依赖；随后才登记并由 primary 审查、冻结与 proposal 随机 K3 一致的外部 gate 标签/小 probe 协议，先做小样本输出检查。保留旧 flow/priority 表现，不解冻 proposal、不增加强制规则、不启动新 formal 或恢复 run0，final heldout 仍未消费。以上只记录计划，不表示公共旧 lifecycle 统计已修复或外部 gate 已适配。