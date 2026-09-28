# P3 normal-advantage gate repair protocol（2026-09-28）

## 原始冻结范围（历史输入假设）

本协议固定 source `03c0`，只在 train dense24、seed `873103`、arrival12、uid42 的固定 local window `steps 100..104` 上做终态动作优势诊断；五点含已消费的 step102 anchor，不称为独立新样本，不换步、不按结果选最佳点。source、proposal、安全边界和旧输出保持不变；不使用 validation/heldout。

诊断 CLI：

```text
python -m terminal_marl.training.diagnose_p3_landing_outcome_credit --local-window --output-dir ROOT/credit [--smoke]
```

smoke 只取 step102 单点、1 repeat；full 固定五点，每点 3 repeats × 3 branches，最多 `45` logical branches 与 `255` runtime 单步 branches。允许 C 为空或 source executed action 非 WAIT 的对照；不筛掉、不换点、不选择最佳。

每次诊断沿用完整请求 ledger、arrival signature、真实 safety、model digest 和 RNG 检查。任一技术检查失败即停止，不自动重跑。输出根为 `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928`，使用全新的 `credit_smoke/`、`credit/`、`fit_smoke/`、`fit/`、`evaluation/` 子目录；旧目录不覆盖。

## 输入修订 V2（当前活动条款）

exact source replay `702/702` true 且 UID42/arrival12 存在；steps `99..101` 为 inbound、`ring_role=null` 且已有入场 TrajectoryPlan，steps `102..105` 为 `ring_outer/outer/无 committed_plan`。原假设 `100/101` 为环上决策的 phase 假设错误，不是数据缺失；实际有效 steps 为 `102,103,104`，原 credit 输出保留为 FAILED。该修订是针对输入窗口检查的明确例外，不是性能筛选或超参搜索；不补窗口外样本。当前活动条款固定这 3 点，使用 `153` runtime branches（3×51）与 `3×3×3=27` logical branches；均匀使用 3 行 A，objective、KL、lr、clip、backtracking、updates 和 evaluation 条款保持不变。

活动诊断输出为 `ROOT/credit_v2`；smoke 固定 step102 单点并复用已完成的 `credit_smoke` serialization/branch 验证，full 只用 steps `102,103,104`。本轮修订只改变输入步集合，不改变 serialization 或分支逻辑；先重跑 `11+28` 定向测试，再启动 v2 credit。`fit_smoke/`、`fit/`、`evaluation/` 尚未创建。活动训练使用旧 `11589` anchors 加 3 个固定点，共 `11592` 个 KL anchors；优势目标只用这 3 点，按三行均匀加权。

活动命令为：

```text
python -m terminal_marl.training.diagnose_p3_landing_outcome_credit --local-window --output-dir ROOT/credit_v2 [--smoke]
python -m terminal_marl.training.train_p3_terminal_gate_repair --normal-advantage --bounded-backtracking --terminal-credit-root ROOT/credit_v2 --output-dir ROOT/fit[_smoke] [--smoke]
```

原 heartbeat 已删除；恢复运行时重新建立并记录新的 heartbeat ID。v2 full 在代码测试完成前不启动，结果取得前不写五点收益或训练成功结论。

## 固定训练条件

仅当 full 诊断通过且至少有一个稳定正 normal opportunity 时进入训练。对每个固定点，令 `A` 为三次 `RELEASE-WAIT` normal 差值：三次均 `>=0` 或均 `<=0` 时取均值；混合符号或全零时令 `A=0`。两分支 failures 均为 `0`、技术检查与字段均完整；不使用微小 wait CE。

训练 loss 固定为：

```text
-mean[A * (p_new - p_source)] + 10 * meanKL
```

三点包含零/负对照；旧 `11589` anchors 仅作为 KL 锚点，加当前三点共 `11592` 个 KL anchors；优势目标只使用固定三点。只更新 gate adapter，其他参数与 proposal 固定。`meanKL<=.001`、`maxKL<=.01`；Adam `lr=1e-4`、gradient clip `1`，backtracking scales 为 `[1,.5,.25,.125,.0625]`。smoke 最多 `1` logical update，full 最多 `120` logical updates；若某一步全部 scales 拒绝，则完整回滚 model 与 optimizer 并停止。禁止扫 lr，禁止按 evaluation 选择 checkpoint。

训练 CLI：

```text
python -m terminal_marl.training.train_p3_terminal_gate_repair --normal-advantage --bounded-backtracking --terminal-credit-root ROOT/credit_v2 --output-dir ROOT/fit[_smoke] [--smoke]
```

## 评估与准入边界

评估复用既有脚本：

```text
python -m terminal_marl.training.evaluate_p3_terminal_gate_repair --checkpoint ROOT/fit/terminal_gate_repair.pt --fit-report ROOT/fit/report.json --output-dir ROOT/evaluation
```

先按现有代码的 `--check-only` 语义在独立目录或不创建产物地检查，再决定 full evaluation；不把 check-only 当作结果。复用原六个已消费 development cells；train selection 仅为 `873103`，其他 cells 不称 heldout。候选保留条件不放宽，主审同时检查 normal、priority/low-SOC wait 与 failed/safety；loss 下降不等于训练退化已解决。`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`；运行前不写成功结论。

执行顺序固定为：本地 tests → source-window smoke → full 诊断 → 主审检查 → fit smoke → 一次 full fit → matched six-cell evaluation → 主审检查。长任务先核实 PID 与预期产物，再建立 heartbeat；终态删除 heartbeat。

GLOBAL RNG、三点邻域相关性和单场景 MC 样本量限制需在报告中保留；不包装为稳定收益或泛化证明。本文件记录已预注册的执行范围，不构成额外用户许可要求。
