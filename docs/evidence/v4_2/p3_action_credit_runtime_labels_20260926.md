# P3 runtime action-credit / K3 branch labels（2026-09-26，partial diagnostic）

## 状态与输入

旧的 `16` 状态诊断输入为：

```text
/home/user/training_outputs/terminal/p3_action_credit_20260926/manifest.json
```

该诊断为 `COMPLETED`，`16/16` exact/outcomes true；它只保存 WAIT 与 `25` 个 K3 proposal 结果，属于 partial state diagnostic，不是 singleton branch label 数据，也不是完整训练集。

本轮已验证的 full runtime 输出根为：

```text
/home/user/training_outputs/terminal/p3_action_credit_runtime_labels_20260926
```

该输出包含 `16/16 COMPLETED`，每个状态 `51` 个 branches，共 `816` 条 branch 记录；exact、outcomes、state、model、RNG 均为 `true`，命名 safety counters 无非零值。新输出中的 `51` branches 才是 branch-level singleton labels；旧 `16` 状态不作此称呼。

原 report 的 FP/FN 两个字段名称相反，但原始 `C` 与 branch 内容正确。代码修复后的离线更正 summary 状态为 `REPORTING_LABEL_NAMES_CORRECTED`，路径为：

```text
/home/user/training_outputs/terminal/p3_action_credit_runtime_labels_20260926/reporting_correction/summary.json
```

原始报告保留，不 rerollout。

## Catalogue false positives 与更正语义

以下两条原始 catalogue `[2]` 是 false positive：

| seed / step | 原始 `C` | 更正后 `C` | 直接执行事实 |
|---|---|---|---|
| `873103 / 0` | `[2]` | `[]` | `25` 个 K3 proposal 全部 `safety_reject` |
| `873106 / 0` | `[2]` | `[]` | `25` 个 K3 proposal 全部 `safety_reject` |

另外两个低 q 状态的 `C` 保持不变：`873101 / 188` 为 `[20,21,22]`，`873102 / 311` 为 `[2,3,4,9]`；其余 `12` 个 `C` 也保持不变。由 runtime 单候选 `C` 推导的 `q` 与真实 `25`-K3 weighted approved 结果之间的最大误差为 `5.029141902923584e-08`。

## 旧 partial 诊断中的直接事实

旧 `16` 状态中的分组记录为：

| 分组 | 状态数 | 直接观测 |
|---|---:|---|
| `C` 为空 | 6 | `approved=0`，`delta_reward=-.05` |
| high-q | 6 | `approved≈1`，`delta_reward=+.1` |
| 低 q | 4 | 见上面的 full runtime 更正语义 |

在更正后，`873101 / 188` 的 `j=20/21/22` 可批准、`deltaQ≈+.11144`；`873102 / 311` 的 `j=2/3/4/9` 可批准、`deltaQ≈+.100087`。两条 `C=[2]` 的 step0 记录应归入 catalogue false positive，不作为 proposal 漏覆盖训练样本。

## 解释边界与后续范围

本轮修复诊断与执行上下文标签路径已经由 runtime 输出验证；同组代码检查为 `34 tests PASS`。这只确认标签路径事实，不表示训练已完成或性能已改善。静态 catalogue 可行性不等于实际 step 执行上下文可行性。rank loss 是独立候选，尚未 fit 或接入 formal；WAIT/RELEASE 分支比较的 utility 只是 one-step critic proxy，不是长期 advantage，也不定义 rank surrogate 的训练目标。训练退化与 P3 qualification 尚未解决，旧 `16` 状态是 partial，不能当作完整训练集。

已有 [source logit counterfactual JSON](../../../results/p3_k3_logit_counterfactual_20260926.json) 的四条记录只描述 logit-space 反事实，不等于实际 runtime 改善。reward、安全边界与 heldout 不变。

下一步是在现有 source tapes 上补齐完整 runtime labels，再进行一次 bounded rank-aware proposal 修复与 `6` 个 development seed evaluation；不直接启动 formal 或消费 heldout。reporting correction 已完成，但不代表训练或 P3 qualification PASS，也不据此作 PPO credit 根因或同时机预约抢占结论。

## 本轮交付文件

- `terminal_marl/training/diagnose_p3_action_credit.py`
- `terminal_marl/training/p3_action_credit.py`
- `terminal_marl/training/k3_coverage_objective.py`
- `tests/test_p3_action_credit.py`
- `tests/test_k3_coverage_objective.py`
- `docs/CURRENT_STATUS.md`
- `docs/evidence/v4_2/p3_action_credit_runtime_labels_20260926.md`
- `results/p3_k3_logit_counterfactual_20260926.json`
