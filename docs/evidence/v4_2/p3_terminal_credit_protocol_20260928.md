# P3 terminal reward-order and full-cycle credit protocol（2026-09-28）

## 冻结范围

本文件冻结一轮只读诊断协议，用于检查训练 reward 排序与完整终态收益之间的关系，并构造有限的 terminal counterfactual credit 证据。协议不修改奖励、不训练、不启动 PPO formal 或 heldout 评估，也不预设诊断结果。

输入使用 source checkpoint `03c0` 以及既有 source/V7 run0@80、run1@240、run2@240 的同六个 development seed tapes，共 `24` tapes。proposal、geometry、safety、lateoff 和 K3=`3` 全部冻结。所有数据均为已使用的 development 数据，不具有泛化资格。

## A. reward 排序诊断

对上述既有 tapes 只重放，不改变 reward 实现、不训练。对每条 tape 同时保存并配对：

- 原 reward 的 episode 总和；
- `gamma=0.99`、从 episode 起点折扣的 return；
- 各 reward component 的贡献；
- 原始 `normal`、`emergency`、`fail` 数量以及 priority wait、low-SOC wait。

输出必须能按 tape、seed 和策略版本配对比较。该部分只提供直接观测，不预先认定 reward 需要修改。

## B. counterfactual 状态选择

counterfactual 只使用 `873101`--`873103` 的 train tapes。输入为现有 `164` 个 runtime rows。每个 seed 按 `(step, uid, arrival)` 排序，依次选择：

1. 最早的 empty-`C` row；
2. 最早的 nonempty row，且 `qK3 < .999`；如果没有该 row，则选最早的 nonempty row；
3. nonempty rows 的 median row；
4. nonempty rows 的 latest row。

选择结果去重；不足四个时从剩余排序 rows 均匀补足到四个。完整的选定列表必须在任何 branch run 启动前落盘。每个选定 state 使用两个 continuation RNG repeats，每个 repeat 执行三个 branch，因此总上限为 `3 × 4 × 2 × 3 = 72` branch runs。

## C. branch 与完整终态执行

每个 state/repeat 执行以下三个 branch：

1. `WAIT`；
2. 按 source action distribution sampled 的 `RELEASE`，使用原 K3；
3. 最高 logit 的 runtime-certified `RELEASE`，使用 K3。

若没有 runtime-certified alternative，第三 branch 标记为 `no alternative`，并复用同一 source branch；不得虚构可行 proposal。完全相同的 branch 可以使用缓存，但不得跨 state 缓存。

branch 的第一步固定其他 actor 的 tape actions；从第一步之后，由冻结 source 在新 state 上实时 stochastic 决策，继续运行到完整 scene 终态。各 repeat 分别重置同一 policy RNG，并共享外生场景。由于 actor 数量可能分叉，随机流不保证完全对齐；两个 repeats 只作为诊断证据，不作为统计证明。

每个 branch 必须记录完整 request 生命周期（含 prefix），并报告全 request 的 safety、failed 和 unresolved 结果，同时报告目标机与其他 actor 的终态指标。不能只记录目标机的一步 reward 或局部结果。

## D. gate 标签边界

gate 标签只由 `WAIT` 与 source-distribution `RELEASE` pair 的完整终态比较产生。runtime-certified proposal branch 只作 proposal 诊断，不替代 gate pair，也不直接产生 gate 标签。

pair 比较顺序固定为：

1. 失败数量优先；
2. 失败数量相同时比较 normal 数量；
3. normal 数量相同时，检查 priority wait 与 low-SOC wait 是否有一致改善。

只有两个 repeats 的方向一致时才形成明确标签；方向不一致记为 `uncertain`，不自动进入 fit。

## E. 执行门槛与报告边界

先完成小样本输出与序列化检查，再完成首个 flow6 state 的全终态 smoke；通过后才运行完整协议。任一阶段出现 error 或 safety failure，立即停止并将原始证据返回 primary，不自动重跑。

本协议不消费 untouched heldout，不启动 PPO formal，不自动开启 gate fit，不改变奖励，也不修改 `CURRENT_STATUS.md`。结果区在执行前保持空置；本文件只冻结协议，不记录尚未发生的结果。
