# 同状态 60 秒实用性能对照（2026-10-20）

本固定窗口的 wall 耗时由 1930.277277 秒降至 971.431502 秒，提速 **1.987044 倍**、耗时下降 **49.673992%**；推进 60 秒仍需 **16.19 分钟**，尚非实时。结论仅适用于本次窗口。

基线为 `498229033c66e4decea36e6e651814ba7e9036f3`，优化版为 `e343c77897169ff390c5026ed77c27aa9ea4ba8a`（含已推送的 `c59aa6b`）。两版各在新进程、冷缓存下执行一次；同用 CPU 2、0 worker、数值库单线程，无 profiler 或渲染。从 W71-B 检查点恢复相同物理状态与原 driver/cursor，时间从 `720.9445824868908` 推进到 `780.9445824868908`，恰为 60 秒。计时含惰性预计算及记录刷新，恢复与回放导出在计时外。

物理、配置、资源及调度约束保持原规则。优化采用保守粗筛加原连续检查回退、稳定轨迹包装、转移成功/失败复用、能量与峰值积分累计量复用，以及 `1e-11` 精度的诱导速度求根。物理推进仍为 0.2 秒及精确事件时刻。

运行时近似包括普通应急监测按 1 秒及事件触发；35% SOC 门控也检查已承诺轨迹的预测 SOC；低 SOC、缺少参考或已知临近风险时恢复高频检查，不确定或临界时回退精确计算。相位分片表采用经验采样残差余量 `2e-5 SOC`，非严格全局界；插值 q 不用于碰撞或规划储备认证。本窗口仅生成 12 cells / 36 samples，`table_hits=0`，不能将本次提速归因于表命中；可见变化主要是调用减少与复用。旧峰值递归 2556570 次与新 `_peak_table` 19275 次语义不同，不作同类直接比较。遥测区间由 314 减至 85，`_record` 调用次数不等于写盘帧数。

两版初始 metrics 完全一致、最终状态分类一致；调度时刻仅 2 处相差不超过 `9.1e-13 s`。窗口能耗相差 `0.000152349472 J`。两份冻结基线检查器复核均为 PASS、`failures=[]`，各检查 56 个计划、1372 个连续配对区间，细化 38 次最小距离；所得相同距离区间高于 80 m。优化版最大单区间能量、SOC、峰值时间误差分别为 `2.143788e-7 J`、`2.975918e-15`、`1.453359e-6 s`。

代码范围：[energy](../../terminal_marl/core/rotating_slots/energy.py)、[safety](../../terminal_marl/core/rotating_slots/safety.py)、[trajectory](../../terminal_marl/core/rotating_slots/trajectory.py)、[world](../../terminal_marl/core/rotating_slots/world.py)，以及 [replay](../../terminal_marl/core/rotating_slots/replay.py)、[audit](../../terminal_marl/core/rotating_slots/audit.py)、[checkpoint](../../terminal_marl/core/rotating_slots/performance/checkpoint.py)、[运行脚本](../../scripts/server/compare_rotating_slots_practical.py)和配套 tests。`c59aa6b` 组合测试 165 passed（80.04 秒）；已推送的 `e343c77` 最终解析检查 34 passed（1.96 秒）。

输入：[W71-B 检查点](D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/corpus_frozen_retention_20261018a/inputs/W71-B/checkpoint-00000041.json)；[原实现冻结](../../artifacts/rotating_slots/practical_perf_20261020/before_source)。两版回放含完整历史前缀，数值复核仅覆盖本 60 秒窗口。既有平台原 CLI 在 `750.9445824868908 s` 导出三视图，主对话已查看俯视和三维图；截图不构成连续安全证明。

| 指标 | 基线 | 优化版 |
|---|---|---|
| wall / process 秒 | 1930.277277 / 1925.460347 | 971.431502 / 968.870752 |
| 能量表、峰值表、参考模板预计算并集秒 | 97.469217 | 32.004557 |
| 相位表预计算秒（与前项重叠，不相加） | 不适用 | 1.250272 |
| `evaluate_power` 调用 | 1923140 | 417614 |
| `power_bounds` 调用 | 3805096 | 387046 |
| `integrate_energy` 调用 | 45038 | 17499 |
| `generate_final` 调用 | 29071 | 3047 |
| `world._reference` 调用 | 7100 | 2 |
| `validate_separation` 调用 | 414740 | 420559（略增） |
| 正常着陆 / 应急声明 / 应急着陆 / 失败增量 | 1 / 2 / 0 / 0 | 1 / 2 / 0 / 0 |
| 窗口能耗 J | 167537676.46593451 | 167537676.46608686 |
| 冻结原检查器 | PASS，`failures=[]` | PASS，`failures=[]` |
| 最小距离区间 m | [84.58846797780782, 84.59512368042725] | [84.58846797780782, 84.59512368042725] |
| JSON / 完整回放索引 / 离线复核 | [结果](D:/7_codespace/training_outputs/terminal/practical_perf_20261020/before/result.json) / [回放](D:/7_codespace/training_outputs/terminal/practical_perf_20261020/before/replay/replay.json) / [复核](D:/7_codespace/training_outputs/terminal/practical_perf_20261020/before_original_audit.json) | [结果](D:/7_codespace/training_outputs/terminal/practical_perf_20261020/after/result.json) / [回放](D:/7_codespace/training_outputs/terminal/practical_perf_20261020/after/replay/replay.json) / [复核](D:/7_codespace/training_outputs/terminal/practical_perf_20261020/after_original_audit.json) |
| 原平台截图 | [俯视](D:/7_codespace/terminal/artifacts/rotating_slots/practical_perf_20261020/before_views/top_down.png) / [侧视](D:/7_codespace/terminal/artifacts/rotating_slots/practical_perf_20261020/before_views/side.png) / [三维](D:/7_codespace/terminal/artifacts/rotating_slots/practical_perf_20261020/before_views/3d.png) | [俯视](D:/7_codespace/terminal/artifacts/rotating_slots/practical_perf_20261020/after_views/top_down.png) / [侧视](D:/7_codespace/terminal/artifacts/rotating_slots/practical_perf_20261020/after_views/side.png) / [三维](D:/7_codespace/terminal/artifacts/rotating_slots/practical_perf_20261020/after_views/3d.png) |
