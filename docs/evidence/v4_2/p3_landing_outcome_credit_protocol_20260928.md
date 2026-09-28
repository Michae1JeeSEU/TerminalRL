# P3 landing outcome credit protocol（2026-09-28）

## 目的与边界

本轮只做 source-only 的终态动作机会诊断，寻找能够改变 `normal/emergency` 结果的真实动作机会。已有 82 个 train runtime rows 中缺 arrival 12；arrival 13 仅有 empty C；arrival 21 多为 empty C 或已必选 RELEASE。本轮只使用 source `03c0` 的 train dense24、seed `873103` 中 arrival 12/13/21 的三个 emergency 目标，不从 validation 选样、不使用 heldout、不训练。

保留 source `03c0`。本轮是数据诊断，`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`；在结果取得前不预写成功或候选效果。

## 候选与 runtime C

候选必须来自 exact source replay，目标位于 `ring_outer/middle/inner`，且 source 实际执行 WAIT。每个 arrival 按 step 均匀取最多 40 个点（含端点），三者合计不超过 120；runtime C 探测前先持久化 probe list。

真实 C 只来自 same-prestate、fixed-others-tape-actions 的 25 个 singleton 实际 `env.step`。不能使用旧几何 catalogue 标签；现有 runtime 工具实际产生的 51 branches 也照实记录。

每个 arrival 的非空 C 依次选择：

1. 最高 `(1-pRelease) * qK3`；
2. 最高 `pRelease * (1-qK3)`；
3. 最低 SOC。

按顺序去重，并排除与已选 step 距离小于 10 的点；tie 按 `(step, uid)` 升序。候选不足时少选，最终 selection 总数不超过 9。完整 branch 开始前冻结 final selection。

## 完整分支

每个选中 state 使用 3 个 continuation RNG repeats 和 3 个 branch：

- WAIT；
- source-distribution RELEASE；
- runtime 最高 logit 的 certified RELEASE。

logical branches 总数不超过 81；分别报告 runtime branch 与完整 branch 的实际复用数。首步其他 actor 固定 tape actions，之后使用 frozen source 直到完整终态，单分支上限为原 1800 steps。收集完整 request ledger、exogenous 记录和真实 safety counters。

## 判定与报告

局部 gate normal opportunity 要求三次 repeat 的方向一致（每个差值均 `>= 0` 且至少一个 `> 0`），failed 不增加，且所有技术检查通过。wait-only 不生成训练标签。

gate source-policy gap 定义为：所选方向的可改善概率（`p` 或 `1-p`）乘以 mean normal gain。它只作为局部 MC 估计，不是性能或泛化保证。proposal opportunity 单列，不能当作 gate 标签。原 global-RNG actor 路径可能分叉；报告该限制，不包装统计证明。

不修改 reward、安全规则、几何或 proposal。报告每个目标、C、三分支和 repeat 的实际记录，并保留 source/candidate action、terminal、ledger、exogenous、safety 字段。

## 执行顺序与输出

先运行本地定向测试。随后只做一次固定 smoke：seed `873103`、step `526`、uid `309`、arrival `21`，单 row runtime 加 1 个 continuation repeat。固定 smoke 的 step526/uid309 在 source 实际执行 RELEASE；这是技术 fixture 例外，标记 `fixture_not_full_selection=true`，不进入 full selection 或机会统计。full 仍严格要求 source 实际执行 WAIT。主审通过后才运行一次 full，不重复、不扩预算。输出根目录为 `/home/user/training_outputs/terminal/p3_landing_outcome_credit_20260928`，使用 `smoke/` 和 `full/` 子目录：

```text
python -m terminal_marl.training.diagnose_p3_landing_outcome_credit --output-dir /home/user/training_outputs/terminal/p3_landing_outcome_credit_20260928/smoke --smoke
python -m terminal_marl.training.diagnose_p3_landing_outcome_credit --output-dir /home/user/training_outputs/terminal/p3_landing_outcome_credit_20260928/full
```

本协议执行范围限于数据诊断；PPO/P3/P4 不启动，结果取得前不写结果性结论。