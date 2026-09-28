# P3 terminal gate repair V2 bounded-backtracking protocol（2026-09-28）

## 状态与冻结范围

本文件登记一次 V2 gate-only development probe。V1 已完成 `1` step smoke：`exit0`、`COMPLETED`、`1 attempt / 0 accepted`；source-anchor mean KL 为 `0.00031149349524639547`，max KL 为 `0.014092326164245605 > 0.01`；完整 model 与 optimizer 已回滚，`nonadapter=true`、`proposal=true`、`roundtrip=true`、`adapterchanged=false`。V1 不启动 full fit 或 evaluation；该机械零更新不能判断 terminal-label 效果。V1 smoke 输出保持 immutable：

```text
/home/user/training_outputs/terminal/p3_terminal_gate_repair_20260928/smoke
```

V2 仅在实现、定向测试和 smoke 完成后执行一次 full fit；本文不记录尚未取得的 V2 结果。source 为 `03c0`，沿用 V1 的其他冻结配置；不启动 heldout、PPO、P3 或 P4 晋级。

## 输入、目标与冻结参数

复用五个稳定标签 state `1,2,3,6,11` 及其已登记权重，source train anchors 为 `11589` observations：

```text
/home/user/training_outputs/terminal/p3_runtime_rank_repair_20260926/anchors/anchors.npz
```

目标函数沿用 V1：

```text
weighted_CE + 10 * mean_KL(source_gate || candidate_gate)
```

Adam base learning rate 为 `1e-4`，gradient clip 为 `1.0`，最多 `120` 个 logical updates。proposal、actor encoder、base gate、gate temperature、critic 及其余非 adapter 参数全部冻结；只有 `gate_adapter` 可更新。

## 固定 bounded backtracking

每个 logical update 从该步开始前保存的完整 model 与 Adam state 开始，依次尝试固定 scales：

```text
[1.0, 0.5, 0.25, 0.125, 0.0625]
```

每个 trial 恢复同一个 step-before 状态，以 `base learning rate * scale` 更新，并累计检查 source-anchor KL。首个同时满足 mean KL `<= 0.001` 且 max KL `<= 0.01` 的 trial 接受；所有 trial 均拒绝时恢复完整 model 与 Adam state 并停止。下一 logical update 仍从 base learning rate `1e-4` 开始，不跨步沿用 scale，不调参或重试候选矩阵。训练 CLI 新增 `--bounded-backtracking`。

## 执行顺序与输出

输出根必须为 fresh：

```text
/home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928
```

先运行本轮 `1` logical-step smoke，再由 primary 查看真实 smoke 输出；通过后只运行一次 full fit。fit 技术通过后，复用已经消费的六个 development source ledger，仅新增 candidate 的六次 rollout，不重跑 source；evaluation 仍先 `--check-only`，再 full。

```bash
cd /home/user/projects/terminal && /home/user/envs/terminal_env/bin/python -m terminal_marl.training.train_p3_terminal_gate_repair --output-dir /home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/smoke --smoke --bounded-backtracking
cd /home/user/projects/terminal && /home/user/envs/terminal_env/bin/python -m terminal_marl.training.train_p3_terminal_gate_repair --output-dir /home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/fit --bounded-backtracking
cd /home/user/projects/terminal && /home/user/envs/terminal_env/bin/python -m terminal_marl.training.evaluate_p3_terminal_gate_repair --checkpoint /home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/fit/terminal_gate_repair.pt --fit-report /home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/fit/report.json --output-dir /home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/evaluation --check-only
cd /home/user/projects/terminal && /home/user/envs/terminal_env/bin/python -m terminal_marl.training.evaluate_p3_terminal_gate_repair --checkpoint /home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/fit/terminal_gate_repair.pt --fit-report /home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/fit/report.json --output-dir /home/user/training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/evaluation
```

V1 保留标准继续适用，见 [V1 protocol](p3_terminal_gate_repair_protocol_20260928.md)：每个 cell 的 normal 不下降，failed/unresolved/真实 safety events 不增加，aggregate priority wait 与 low-SOC wait 均不增加且至少一项严格下降；否则保留 `03c0`。

V2 结果取得前，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`；本协议不宣称 terminal-label 修复或训练退化已解决。