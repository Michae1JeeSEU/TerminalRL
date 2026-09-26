# P3 adaptive-KL V7 准备记录（2026-09-26）

本页记录 V7 formal runner、协议、adaptive probe 终态和 formal 启动边界。它不替代 P3 qualification，也不把 probe checkpoint 自动晋级为正式模型。

## 当前状态

- 用户目标是完成有效的三组独立 `240` episode formal PPO，并在训练结束、模型冻结后对同一场景集合执行评估。
- 当前 adaptive probe 为 run0/run2 各 `8` 个回合，输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_probe_v1_20260926/`，两组均 `8 UPDATED / 0 SKIPPED_KL / 0 HARD_FAILED`、`8/8` restore exact；accepted minibatches 为 run0/run2 `298/332`，最终 reference KL 为 `.0077864/.0133243`。
- probe 训练诊断各为 `132` requests，合计 `264`：run0 `110 normal + 22 emergency_landed`，run2 `117 normal + 15 emergency_landed`，合计 `227 normal + 37 emergency_landed`，`0 failed_energy`、`0 unresolved`；before/after development 均 full-resolution 且 hard failure 为 false。dense24 paired 为 run0 emergency `1→5`、priority wait `+15.6374s`、low-SOC wait `+40.9181s`，run2 emergency `3→1`、priority wait `+5.8564s`、low-SOC wait `+2.5737s`；稀疏 flow wait 有改善。该结果支持 primary 的有效更新审查，但不表示性能全面改善，也不晋级 probe checkpoint。
- primary 已允许从同一 03c0 source checkpoint、fresh Adam 启动 `3×240` formal。formal 输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/`；control 路径为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/_control/runN/{pid,status.json,exitcode,...}`。run0 child PID `2641745` 已 `FAILED exit1`，在 `80 processed / 80 UPDATED / 0 skipped / 0 training hard`、`3954 accepted minibatches` 后停止，`80/80` restore 与 RNG exact、zero-gate-delta 计数为 `0`；其训练请求为 `1320`（`1185 normal + 135 emergency_landed`、`0 failed_energy`、`0 unresolved`）。停止发生在 `dev_after_0080` 的第六个 seed `dense24/873106`，不是 training episode 80；退出由开发能量 hard gate 触发，日志未见软件异常：该 dev 记录为 `18 normal + 5 emergency_landed + 1 failed_energy + 0 unresolved`（24 requests），失败 arrival 为 `arrival_id=15`，dev80 六个 seed 合计 `84` requests（`74 normal + 9 emergency_landed + 1 failed_energy + 0 unresolved`），六个 dev seed 虽 `complete=true` 但该组 `full_resolution=false`、`hard_failure=true`；安全计数为零、weights unchanged、replay exact。最新 checkpoint 为 run0 episode 80，原输出根保留，不重启、不替换 seed 或 cap。主审随后核验原 tape `897/897` exact 且 `outcomes_match`；失败 arrival 为 runtime UID `72`（对应 `arrival_id=15`），`t=135` 进入 outer，至 `t=896` energy_depleted。outer 请求计数为 `WAIT=679`、`RELEASE=82`、`approved=0`（WAIT 不计作 rejected）；首次 outer RELEASE 为 `t=152`、SOC `.496309`，返回 `safety_reject`。在 `878` 次 emergency 函数调用中有 `120` 次真实 attempt；首次为 `t=297`、SOC `.399643` 的 safety reject，另有安全候选拒绝累计 `134`、energy 候选拒绝 `247`、LZ/dynamics `0`。`t=557`、SOC `.226309` 首次出现 energy reject；`t=573` 的 normal RELEASE 首次触发 `emergency_energy_guard`，slack `-93.7824`。上述计数使用不同分母，不能互相相加或直接作同一类拒绝率。诊断文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_dev80_diagnostic_20260926/diagnostic_v2.json`。主审当前结论仅限于：存在早期正常/应急路径被阻挡、后期能量不可行的链条；尚不能区分错过可行窗口、安全过滤过严或规划缺口，也不能直接归因于只缺训练。下一步对同一既有 tape 的 `6` 个状态做离线只读可行性与 source/episode80 概率对照，不反馈原轨迹或修改运行中依赖。run1/run2 child PID `2641747/2641746` 仍在运行，最新采样均为 `117 UPDATED / 0 skipped / 0 hard`；请求分别为 run1 `1590`（`1408 normal + 182 emergency_landed`）与 run2 `1590`（`1412 normal + 178 emergency_landed`），两者均为 `0 failed_energy`、`0 unresolved`。首个 dev40 性能回归已记录：baseline `76/8`、run0 `74/10`、run1 `73/11`、run2 `76/8`（normal/emergency_landed），dense24 priority wait 增加约 `20–49s`；配置未改变。主审正在只读回放原始 tapes 取证，尚无根因或修复结论；三 run 全部完成条件未满足，`64` episode final evaluation 尚未启动。
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

`filter_events=0` 是诊断插桩在 reset 重建 SafetyFilter 前仍挂接旧实例造成的，不能解释为没有 safety reject；后续仅修正临时 helper 的真实 emergency reason 读取。主审结论限定为：这些状态不能只靠提高 release 概率解决，也不能由 6 个状态证明整条轨迹不存在可行窗口或 PPO 没有责任。state witness 记录时 run1/run2 为 `109 UPDATED / 0 skipped / 0 hard`；当前最新采样为 `117 UPDATED / 0 skipped / 0 hard` 且仍在运行；`dev120` 尚未发生，未补充请求聚合。

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

`on` 的 `t=250` slack 为 `195.644..236.227`，`t=297` slack 为 `165.852..192.081`；其余拒绝仍来自同一 SafetyFilter/energy 检查。主审限定结论为：旧 slot 截止时间挡住了部分经实际 endpoint 重新预约后可安全执行的候选；现有 gate 在这些状态仍保持低 `P_RELEASE`，不能只打开 flag 就宣称修复或晋级。该对照未放宽安全阈值，也不是闭环效果。下一步在已消费的 `873106` dev seed、source `03c0` 与 run0 episode80 上做有界 inference-only late-slot off/on 闭环配对；先核验原 off 评估的 RNG/配置可复现，等价确认后复用原 off 结果，最多 `4` cells，不使用新 seed/heldout、不做 PPO、不覆盖旧 outputs。若仍失败保留结果，再决定是否需要 gate 外部标签适配，当前不预断该结论。run1/run2 当前最新为 `117 UPDATED / 0 skipped / 0 hard` 且仍运行，`dev120` 尚未发生；formal 目标未完成，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、final evaluation 尚未启动。