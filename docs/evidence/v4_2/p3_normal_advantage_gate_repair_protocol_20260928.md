# P3 normal-advantage gate repair protocol（2026-09-28）

## 冻结范围

本协议固定 source `03c0`，只在 train dense24、seed `873103`、arrival12、uid42 的固定 local window `steps 100..104` 上做终态动作优势诊断；五点含已消费的 step102 anchor，不称为独立新样本，不换步、不按结果选最佳点。source、proposal、安全边界和旧输出保持不变；不使用 validation/heldout。

诊断 CLI：

```text
python -m terminal_marl.training.diagnose_p3_landing_outcome_credit --local-window --output-dir ROOT/credit [--smoke]
```

smoke 只取 step102 单点、1 repeat；full 固定五点，每点 3 repeats × 3 branches，最多 `45` logical branches 与 `255` runtime 单步 branches。允许 C 为空或 source executed action 非 WAIT 的对照；不筛掉、不换点、不选择最佳。

每次诊断沿用完整请求 ledger、arrival signature、真实 safety、model digest 和 RNG 检查。任一技术检查失败即停止，不自动重跑。输出根为 `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928`，使用全新的 `credit_smoke/`、`credit/`、`fit_smoke/`、`fit/`、`evaluation/` 子目录；旧目录不覆盖。

## 固定训练条件

仅当 full 诊断通过且至少有一个稳定正 normal opportunity 时进入训练。对每个固定点，令 `A` 为三次 `RELEASE-WAIT` normal 差值：三次均 `>=0` 或均 `<=0` 时取均值；混合符号或全零时令 `A=0`。两分支 failures 均为 `0`、技术检查与字段均完整；不使用微小 wait CE。

训练 loss 固定为：

```text
-mean[A * (p_new - p_source)] + 10 * meanKL
```

五点包含零/负对照；旧 `11589` anchors 仅作为 KL 锚点，优势目标只使用固定五点。只更新 gate adapter，其他参数与 proposal 固定。`meanKL<=.001`、`maxKL<=.01`；Adam `lr=1e-4`、gradient clip `1`，backtracking scales 为 `[1,.5,.25,.125,.0625]`。smoke 最多 `1` logical update，full 最多 `120` logical updates；若某一步全部 scales 拒绝，则完整回滚 model 与 optimizer 并停止。禁止扫 lr，禁止按 evaluation 选择 checkpoint。

训练 CLI：

```text
python -m terminal_marl.training.train_p3_terminal_gate_repair --normal-advantage --bounded-backtracking --terminal-credit-root ROOT/credit --output-dir ROOT/fit[_smoke] [--smoke]
```

## 评估与准入边界

评估复用既有脚本：

```text
python -m terminal_marl.training.evaluate_p3_terminal_gate_repair --checkpoint ROOT/fit/terminal_gate_repair.pt --fit-report ROOT/fit/report.json --output-dir ROOT/evaluation
```

先按现有代码的 `--check-only` 语义在独立目录或不创建产物地检查，再决定 full evaluation；不把 check-only 当作结果。复用原六个已消费 development cells；train selection 仅为 `873103`，其他 cells 不称 heldout。候选保留条件不放宽，主审同时检查 normal、priority/low-SOC wait 与 failed/safety；loss 下降不等于训练退化已解决。`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`；运行前不写成功结论。

执行顺序固定为：本地 tests → source-window smoke → full 诊断 → 主审检查 → fit smoke → 一次 full fit → matched six-cell evaluation → 主审检查。长任务先核实 PID 与预期产物，再建立 heartbeat；终态删除 heartbeat。

GLOBAL RNG、五点邻域相关性和单场景 MC 样本量限制需在报告中保留；不包装为稳定收益或泛化证明。本文件记录已预注册的执行范围，不构成额外用户许可要求。
