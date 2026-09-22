# P3 当前 MDP 基线 low-SOC stress 正式结果（2026-09-22）

## 运行身份与冻结输入

本结果对应冻结协议 [当前 MDP P3 baseline 协议](p3_current_mdp_baseline_protocol_20260922.md)
与 [`p3_current_mdp_baseline_v1.yaml`](../../../terminal_marl/configs/low_v4_2/p3_current_mdp_baseline_v1.yaml)。

- cohort：`low_soc_stress`，SOC 为 `Uniform(.55,.90)`，fresh seeds
  `835101–835103`，rule 与 original K3 配对，共 `6` 条记录；
- evaluator：`terminal_marl/training/evaluate_v4_2_p3_current_mdp_baseline_v1.py`，
  服务器 SHA256 `4adefab947cd21ed3b15b53698c2450d995fa9ebc1cf59944e558600606ad08f`；
- protocol SHA256：`5b6bec77c7fb77d3385cc0f983971dbc5db96c1746a2513089a47be365674432`；
- original K3 checkpoint SHA256：
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。

服务器正式输出为
`/home/user/training_outputs/terminal/p3_current_mdp_baseline_low_soc_stress_20260922`。
服务器 `manifest.json` SHA256 为
`9063c5f13a08d1b6c3c636dc8c02f3db532563500fc801ecbd8c21b1ebc3d4c0`，
`records.json` SHA256 为
`6e773d6f09beea4de38039d6534ab2f60f6cbba06afa9e4c88dc0404aeb782a1`。
运行日志为
`/home/user/training_outputs/terminal/p3_current_mdp_baseline_low_soc_stress_20260922.log`，
SHA256 为
`87102a6e3d5060aa9946800abab8ea381bb8aaa3c10e50f201a20d3d24ab0e51`。
manifest 状态为 `COMPLETE`，6/6 records 已落盘，formal seeds 各消费一次。

## 终态与 gate

manifest 的 `PASS=true` 只表示本 low-SOC stress cohort 的完整性与实现 gate 通过：
完整请求记账、exact action tape replay、安全计数全零、Teacher-free、labels/updates
为零，以及 runtime shortlist contract 均通过。`completed/24` 与
`diverted_low_soc` 是本 cohort 按协议要求报告的结果，不是 all-complete gate。
所有记录均 `terminal=true`、`unresolved=0`；`P3_QUALIFIED=false`、
`PPO_ALLOWED=false`、`P4_ALLOWED=false`。

六条记录如下。等待和 request-to-touchdown 均按已完成 UAV 条件统计；低 SOC 分流保留在
完整的 24 请求分母中。

| seed | controller | 完成/请求 | 低 SOC 分流 | episode(s) | 完成者平均等待(s) | 完成者平均 request-to-touchdown(s) | integrity/safety gates |
|---:|---|---:|---:|---:|---:|---:|---|
| 835101 | rule | 24/24 | 0 | 727 | 85.11773780807086 | 271.75 | pass |
| 835101 | original K3 | 18/24 | 6 | 771 | 205.3876103024359 | 371.0 | pass |
| 835102 | rule | 23/24 | 1 | 887 | 61.70735752988912 | 248.65217391304347 | pass |
| 835102 | original K3 | 23/24 | 1 | 761 | 126.02739836574393 | 312.7391304347826 | pass |
| 835103 | rule | 24/24 | 0 | 724 | 82.29342469919052 | 267.7083333333333 | pass |
| 835103 | original K3 | 21/24 | 3 | 691 | 88.29399990002196 | 264.7142857142857 | pass |

六条记录的 `SafetyInvariantError`、`collisions`、`hard_separation_violations`、
`nmac` 和 `uncertified_execution` 均为 `0`；action tape replay exact，且没有
Teacher、labels 或 policy updates。

## 按 controller 汇总

| controller | 完成/请求 | 完成率 | 低 SOC 分流 | terminal records | 完成者平均等待(s) | 完成者平均 request-to-touchdown(s) |
|---|---:|---:|---:|---:|---:|---:|
| rule | 71/72 | 0.9861111111111112 | 1 | 3/3 | 76.57939667368624 | 262.90140845070425 |
| original K3 | 62/72 | 0.8611111111111112 | 10 | 3/3 | 136.2867926734906 | 313.38709677419354 |

三条 paired seed 的 `rule_minus_original_k3` 均为 `null`，原因是两侧没有同时满足
完整且有效的 paired record。协议要求的 paired delta 因而不计算；不能从不同 seed
或只含完成者的子集拼接出 paired effect。

## 后续边界

本 low-SOC stress 结果完成了预注册的完整性、安全与 replay 检查，但不授权 P3 PPO、
checkpoint promotion、P4 或 A6。下一步应先补充逐 UAV 的 SOC、分流 reason、分流时刻，
以及 queue/transition bottleneck 的诊断 instrumentation；随后另行预注册 fresh repair
protocol、seeds 与 gates，再考虑任何新的 P3 训练。`835101–835103` 已消费，不能正式
重用、reset 或用于调参；安全阈值保持不变。
