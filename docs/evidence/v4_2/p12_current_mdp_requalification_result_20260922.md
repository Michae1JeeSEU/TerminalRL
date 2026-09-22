# 当前 MDP P1/P2 checkpoint 复评结果（2026-09-22）

## 运行身份与冻结输入

本结果对应冻结协议 [当前 MDP P1/P2 复评协议](p12_current_mdp_requalification_protocol_20260922.md) 与
[`p12_current_mdp_requalification_v1.yaml`](../../../terminal_marl/configs/low_v4_2/p12_current_mdp_requalification_v1.yaml)，协议 SHA256 为
`44724cf715be202dbbccc83e802ea27cba16ae78fb6f1052a7a0a7c758a2d56f`。

- evaluator：`terminal_marl/training/evaluate_v4_2_p12_current_mdp_v1.py`，服务器 SHA256
  `8473f8f9a02a1b8a325b30edca24314fdf4a00d22c062dd77cf67b1cb309e21b`；
- P1 K1：`ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt`，SHA256
  `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`；
- P2 原始 K3：`p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt`，SHA256
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。

服务器正式输出为
`/home/user/training_outputs/terminal/p12_current_mdp_requalification_20260922`。manifest
`manifest.json` SHA256 为
`57c50b926fc03a907a21c83738626a41b0c65077a7b4547d7f9ac43aaada80af`。

## 终态与记录矩阵

manifest 状态为 `FORMAL_COMPLETE`，记录数为 `20/20`，`failure=null`。正式 seeds 为一次性消费：

- P1：`834101–834104`，每个 seed 各有 candidate/rule 一条记录；
- P2：`834201–834206`，每个 seed 各有 candidate/rule 一条记录。

每条记录均满足 `completed=requests`、`terminal=true`、`all_gates=true`、exact replay 为 true，且无
false hard gate：

| 阶段 / controller | seed: 完成数/请求数 |
|---|---|
| P1 / candidate | `834101: 3/3`，`834102: 6/6`，`834103: 3/3`，`834104: 6/6` |
| P1 / rule | `834101: 3/3`，`834102: 6/6`，`834103: 3/3`，`834104: 6/6` |
| P2 / candidate | `834201: 6/6`，`834202: 8/8`，`834203: 12/12`，`834204: 6/6`，`834205: 8/8`，`834206: 12/12` |
| P2 / rule | `834201: 6/6`，`834202: 8/8`，`834203: 12/12`，`834204: 6/6`，`834205: 8/8`，`834206: 12/12` |

安全计数总和均为零：`SafetyInvariantError=0`、`collisions=0`、`hard_separation_violations=0`、
`nmac=0`、`uncertified_execution=0`。`safety_counts_valid` 全部为 true，`safety_error` 全部为 null。
`shield_actor_fraction` 和 `shield` 的最小值、最大值均为 `0`。

## 描述性 paired 指标

这些指标只描述已完成记录，协议没有预设 improvement threshold。

| 阶段 | candidate mean wait (s) | rule mean wait (s) | candidate-rule wait delta (s) | candidate RTT (s) | rule RTT (s) | candidate-rule RTT delta (s) |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 59.21658969142434 | 8.351137293207188 | +50.86545239821715 | 287.1666666666667 | 238.54166666666666 | +48.62500000000003 |
| P2 | 5.734572452355265 | 7.794466418817502 | -2.059893966462237 | 234.49305555555554 | 239.64583333333337 | -5.152777777777828 |

## 授权字段与启动记录

manifest 中 `PPO_ALLOWED=false`、`P3_ALLOWED=false`、`training_called=false`、
`checkpoint_saved=false`。硬门槛通过不等于 PPO 授权。

首轮启动 wrapper 因提前在 canonical output 目录内写入 PID 文件，在第一次 reset 前触发
`output already exists`；首轮 stub 与 log 保留。第二次启动将 PID 文件写在输出目录外，使用
`p12_current_mdp_requalification_20260922_run2.log` 和
`p12_current_mdp_requalification_20260922_run2.pid`，完成全部 formal records，未重复 formal seed。

截至本证据提取时，服务器 run2 PID 已退出，log 末行记录 `FORMAL_COMPLETE`。本地镜像目录
`D:/7_codespace/training_outputs/terminal/p12_current_mdp_requalification_20260922` 已出现，
但 manifest 尚未同步；当时目录仅观察到 `registry_progress.json`，因此未能在本地完成 manifest 同哈希核对。

P3 high-SOC baseline 当前另行运行中；本结果不将其状态记为通过，也不改变 `PPO_ALLOWED=false`。
