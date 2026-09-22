# P3 当前 MDP 基线 high-SOC 正式结果（2026-09-22）

## 运行身份与冻结输入

本结果对应冻结协议 [当前 MDP P3 baseline 协议](p3_current_mdp_baseline_protocol_20260922.md)
与 [`p3_current_mdp_baseline_v1.yaml`](../../../terminal_marl/configs/low_v4_2/p3_current_mdp_baseline_v1.yaml)。

- cohort：`high_soc_qualification`，SOC 为 `Uniform(.90,1.00)`，fresh seeds
  `835001–835003`，rule 与 original K3 配对，共 `6` 条记录；
- evaluator：`terminal_marl/training/evaluate_v4_2_p3_current_mdp_baseline_v1.py`，
  服务器 SHA256 `4adefab947cd21ed3b15b53698c2450d995fa9ebc1cf59944e558600606ad08f`；
- protocol SHA256：`5b6bec77c7fb77d3385cc0f983971dbc5db96c1746a2513089a47be365674432`；
- original K3 checkpoint SHA256：
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。

服务器正式输出为
`/home/user/training_outputs/terminal/p3_current_mdp_baseline_high_soc_20260922`。
服务器 `manifest.json` SHA256 为
`36baca0fe1bffa80d93b406a57747397b9aede4a39c7073b8fad31255821c232`，
`records.json` SHA256 为
`9da0ee170d10ffd198c9160c25df0f244e2ae4a5b4237fb7e1d8912195b4a9b0`。
manifest 状态为 `COMPLETE`，`training_called=false`，正式 seeds 各消费一次。

## 终态与 gate

六条记录均有 `status=completed`、`terminal=true`、`unresolved=0`、exact request
accounting、exact action tape replay、Teacher-free、labels/updates 为零和 runtime
shortlist contract 通过。六条记录的安全计数均为零：
`SafetyInvariantError=0`、`collisions=0`、`hard_separation_violations=0`、
`nmac=0`、`uncertified_execution=0`；replay safety 计数也均为零。

全 cohort 的 all-complete gate 未通过，因此 manifest 中
`PASS=false`、`P3_QUALIFIED=false`、`PPO_ALLOWED=false`、`P4_ALLOWED=false`。
失败字段只有完成/无分流相关 hard gate：`all_requested_complete=false`、
`no_diverted_requests=false`、`all=false`。这是一轮 baseline evidence，不授权
P3 PPO 训练、checkpoint promotion 或 P4/A6。

| seed | controller | 完成/请求 | 低 SOC 分流 | episode(s) | 完成者平均等待(s) | 完成者平均 request-to-touchdown(s) | hard gates |
|---:|---|---:|---:|---:|---:|---:|---|
| 835001 | rule | 22/24 | 2 | 1104 | 159.33221586971067 | 336.27272727272725 | fail |
| 835001 | original K3 | 24/24 | 0 | 835 | 147.8915538540162 | 325.9166666666667 | pass |
| 835002 | rule | 24/24 | 0 | 665 | 92.58623649415517 | 276.875 | pass |
| 835002 | original K3 | 24/24 | 0 | 846 | 164.7955559474993 | 353.1666666666667 | pass |
| 835003 | rule | 24/24 | 0 | 648 | 77.76958449354542 | 262.75 | pass |
| 835003 | original K3 | 22/24 | 2 | 1122 | 116.8585614798804 | 298.6363636363636 | fail |

按 controller 汇总：rule 与 original K3 均为 `70/72` 完成、`2` 条低 SOC 分流，
且安全计数全零。未完成请求均被明确记为 `diverted_low_soc`，没有 unresolved 或
missing outcome。

## 配对指标

协议要求完成且有效的 paired record 才进入配对差值。只有 seed `835002` 两个
controller 都满足条件，其 rule − original K3 差值为：episode 时间
`-181.0 s`，完成者平均 request-to-touchdown `-76.29166666666669 s`，完成者平均
等待 `-72.20931945334414 s`。seed `835001` 与 `835003` 的配对指标为 null，
原因是对应一侧未完成全部 24 个请求；不以完成者均值替代全体绩效。

## 分流个体抽查边界

依据冻结 seed/config 对分流个体重建的初始 SOC 为：seed `835001` rule 的 diverted
`UID0/12` 分别为 `0.994652208223/0.919132046959`；seed `835003` original K3
的 diverted arrival `14/16` 分别为 `0.943275096637/0.950628307586`。这些数值是由
冻结 seed/config 重建，并非逐 UAV 记录字段。当前 `records.json` 与对应 raw
summary 不包含逐 UAV 的初始 SOC、分流 reason 或分流时刻，因而不能据此确定
具体分支归因。

## 后续边界

本 high-SOC cohort 的 all-complete gate 失败，不授权 P3 PPO 或既有失败协议继续训练；
后续诊断/修复必须独立预注册 fresh protocol、seeds 和 gates。冻结协议中的 low-SOC
stress cohort `835101–835103` 已在独立报告 [P3 low-SOC stress 结果](p3_current_mdp_low_soc_stress_result_20260922.md)
中完成；该 cohort 的 `PASS` 仅表示 integrity gates 通过，不授权 PPO。high-SOC 与
low-SOC 结果保持为两个独立报告。
