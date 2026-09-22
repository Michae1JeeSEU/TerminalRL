# 当前 MDP 下 P1/P2 checkpoint 复评协议（2026-09-22）

## 目的与冻结边界

本协议用于在当前速度、安全和能量语义下，对既有 P1/P2 checkpoint 做一次全新的、只读的资格与效果复评。它不训练、不更新策略、不保存新 checkpoint，也不改变历史 P1/P2 结果。协议文件为
[`p12_current_mdp_requalification_v1.yaml`](../../../terminal_marl/configs/low_v4_2/p12_current_mdp_requalification_v1.yaml)，SHA256 为
`44724cf715be202dbbccc83e802ea27cba16ae78fb6f1052a7a0a7c758a2d56f`。

冻结的 checkpoint 为：

- P1 K1：`ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt`，SHA256
  `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`；
- P2 原始 K3：`p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt`，SHA256
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。

正式种子只使用一次，并与历史 registry 做碰撞检查：P1 为 `834101–834104`，请求数依次为
`3,6,3,6`；P2 为 `834201–834206`，请求数依次为 `6,8,12,6,8,12`。每个 seed 都运行 candidate/rule 配对回合。诊断 smoke 使用 `832001`、请求数 `6`、最多 `20` 步，不消费正式种子。

## 固定仿真语义

- Observation V2.1 `88D`，V4.2 hierarchical action，deterministic rollout；无 Teacher、无 labels、无 optimizer update。
- 自然流从空场开始，arrival time 为 `10 + 60*i` 秒，horizon 为最后一次 arrival 加 `900` 秒。
- 固定 `SOC=.8`、energy capacity `1500`；安全参数为 `80/40/10 m`。
- 沿用历史 `lowest_id` LZ assignment、`legacy_insertion` release arbitration 和 final capacity `1`。
- P1 使用 K1；P2 candidate 通过 `student_rollout_v4_2.rollout` 使用 K3 runtime shortlist，rule 沿用现有 pilot episode 路径。

## 执行与停止规则

checkpoint hash、协议 hash 和输入存在性必须在第一次 simulator reset 或输出写入前通过。每个 episode 开始前写入 progress，结束后立即写入该 record 与 progress；中途失败时保留已有 `records.json`、`progress.json` 和单回合 action tape。

每个 record 检查 exact replay、safety counters、完整请求、terminal、shield actor fraction `<=.05`、actor decisions、geometry observation、无 reverse transfer/inner excursion 以及 Teacher-free 约束。P2 RELEASE 行还必须保留三个 proposal shortlist。

formal 运行中，只要 safety counter 非零、safety 字段缺失/无效、replay 异常或任意 gate FAIL，立即停止，不启动后续 episode。输出 terminal `FORMAL_FAIL` manifest，记录已尝试 seeds、已消费 seeds、已持久化 records、错误和失败 gate。只有完整矩阵全部通过后才聚合 paired mean wait 与 request-to-touchdown；这些指标只作描述性效果报告，不设置额外 improvement threshold。

## 授权与当前运行边界

本协议始终写入 `PPO_ALLOWED=false`、`P3_ALLOWED=false`、`training_called=false` 和
`checkpoint_saved=false`。当前只完成代码、单元测试和 20 步本地 smoke 验证；正式 seed 尚未消费。

服务器 SSH preflight 当前仍关闭连接，因此尚未启动 formal server evaluation。服务器运行必须遵守本地源码经 Syncthing 同步、输出写入 `~/training_outputs/terminal` 的项目规则；本地 smoke 结果不构成 formal qualification。

SSH 与 Syncthing 状态均验证通过后，才可在服务器执行以下 formal 命令；本轮未执行该命令：

```bash
python -m terminal_marl.training.evaluate_v4_2_p12_current_mdp_v1 \
  --output "$HOME/training_outputs/terminal/p12_current_mdp_requalification_20260922" \
  --history-root "$HOME/training_outputs/terminal" \
  --p1-checkpoint "$HOME/training_outputs/terminal/ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt" \
  --p2-checkpoint "$HOME/training_outputs/terminal/p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt"
```
