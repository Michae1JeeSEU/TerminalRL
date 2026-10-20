# 定向数值优化：同状态 60 秒对照（2026-10-20）

本窗口 wall 耗时由 970.495714 秒降至 618.864966 秒，提速 **1.568186546 倍**、减少 **36.232076315%**；推进 60 秒仍需 **10.31 分钟**，尚非实时。正常落地、应急和失败增量一致；结论仅限本窗口。

BEFORE 为 `fcb11c495ab5be61d0454f6c5cf00b5b5d4ebcdf`，AFTER 实现为已提交推送的 `7ccfa2251f46c8a2c1961b583a285cf9efe5c3e1`。唯一 W71-B 检查点窗口为 `720.9445824868908→780.9445824868908 s`；每版新进程、冷缓存运行一次，同用 CPU 2、0 worker、数值库单线程，无 profiler/渲染。保持物理步长 0.2 秒、调度 5 秒加事件，以及现有应急监测、阈值与遥测配置不变。

本轮仅作四项定向调整：

1. `validate_plan` 先查结构、端点及便宜的运动违规见证；仅明确超限才早拒绝，采样未违规不代表通过，不确定包络仍做完整检查。
2. 每个执行短步先作保守距离/速度碰撞粗筛，未排除的配对仍用原连续检查，稳定配对保留解析路径；不用于完整未来提交。
3. 能量累计表初始网格上限 1 秒，整计划数值误差额度为 `max(1 J, 1e-6×能量)`；求根按残差停止，临近功率/SOC 边界再细化，按 plan 锁定精度并复核已执行记录。
4. 峰值未决宽度额度与保护时间二分宽度均为 0.01 秒，临近峰值预算时细化。source clear 取偏晚端、capture 取偏早端；实际机动、接地与 pad 事件时刻不舍入，冷恢复保留数值记录。

这些是数值目标，未构成全局严格积分误差证明。12 文件组合测试 **203 passed（62.04 秒）**。旧 R0 逐字节候选对比随允许的数值与拒绝诊断变化不再一致，本轮未将其作为验收或修改断言，未宣称全仓测试通过。BEFORE wrapper 未可靠捕获退出码，但结果 `completed`、完整 60 秒、完成日志与回放校验通过，未重跑；AFTER 及两次审计真实退出码均为 0。

两版均由 [4982290 冻结原检查器](../../artifacts/rotating_slots/practical_perf_20261020/before_source)复核：各 56 个计划、1372 个连续配对区间、85 个遥测区间及 12 个确定性执行能源样本，均 PASS、`failures=[]`。能源结果为抽查，连续安全与运动条件全部检查；最小距离精度 0.1 m，不改变 80 m 安全要求。审计耗时单列，不计入运行提速。

两版 273 条轨迹的 hash 集合、序列及 `trajectories.json` SHA 相同；末态 49 机状态分类一致，能耗相差 0.234609127 J。11 个 `source_slot_cleared` 中 6 个按保守端点延后 0.00347–0.00779 秒，引起 7 处 driver 索引时刻差，其中一次跨过 750 秒调度点而重排；计划提交、机动开始、请求、接地、pad 释放及应急触发时刻相同。拒绝诊断随早拒绝顺序和候选评估时刻变化，SOC/q_e 有数值微差，不代表执行轨迹变化，也不作普遍等价声明。

| 指标 | BEFORE | AFTER |
|---|---|---|
| wall / process 秒 | 970.495714 / 968.380386 | 618.864966 / 617.256858 |
| 预计算并集秒（计时内，不另加） | 32.082262 | 16.408605 |
| 相位表预计算秒（与前项重叠，不相加） | 1.257226 | 0.869183 |
| 原检查器审计耗时秒 | 5.994682 | 5.939842 |
| `evaluate_power` 调用 | 417614 | 130966 |
| `power_bounds` 调用 | 387046 | 165799 |
| `integrate_energy` 调用 | 17499 | 21715（增加） |
| `generate_final` / `world._reference` 调用 | 3047 / 2 | 3047 / 2 |
| `validate_separation` 调用 | 420559 | 210858 |
| 标量 `root_iterations` | 21641649 | 7190700 |
| 早拒绝 / 短步粗筛接受 | 未启用 | 157790 / 167043 |
| 正常着陆 / 应急声明 / 应急着陆 / 失败增量 | 1 / 2 / 0 / 0 | 1 / 2 / 0 / 0 |
| 窗口能耗 J | 167537676.46608686 | 167537676.700696 |
| 末态 49 机最大跨版 SOC 差 | 参照 | 4.838296916e-10 |
| 能源抽查最大能量误差 J | 1.205038279e-5 | 0.037321213 |
| 能源抽查最大 SOC / 峰值秒误差 | 1.680600104e-13 / 5.347093293e-7 | 4.839976198e-10 / 0.002787709 |
| 原检查器状态 | PASS，`failures=[]` | PASS，`failures=[]` |
| 最小距离上下界 m | [84.54047967919458, 84.59512368042725] | [84.54047967919458, 84.59512368042725] |

产物：[BEFORE JSON](D:/7_codespace/training_outputs/terminal/targeted_perf_20261020/before/result.json) / [AFTER JSON](D:/7_codespace/training_outputs/terminal/targeted_perf_20261020/after/result.json)，[BEFORE 回放](D:/7_codespace/training_outputs/terminal/targeted_perf_20261020/before/replay/replay.json) / [AFTER 回放](D:/7_codespace/training_outputs/terminal/targeted_perf_20261020/after/replay/replay.json)，[BEFORE 原审计](D:/7_codespace/training_outputs/terminal/targeted_perf_20261020/before_original_audit.json) / [AFTER 原审计](D:/7_codespace/training_outputs/terminal/targeted_perf_20261020/after_original_audit.json)，[差异摘要](../../artifacts/rotating_slots/targeted_perf_20261020/comparison_summary.json)、[BEFORE 冻结源](../../artifacts/rotating_slots/targeted_perf_20261020/before_source)与 [32 文件清单](../../artifacts/rotating_slots/targeted_perf_20261020/freeze_files.json)。两版原平台[俯视前图](../../artifacts/rotating_slots/targeted_perf_20261020/before_views/top_down.png) / [后图](../../artifacts/rotating_slots/targeted_perf_20261020/after_views/top_down.png)、[三维前图](../../artifacts/rotating_slots/targeted_perf_20261020/before_views/3d.png) / [后图](../../artifacts/rotating_slots/targeted_perf_20261020/after_views/3d.png)已查看，截图仅辅助理解。

代码范围：[energy](../../terminal_marl/core/rotating_slots/energy.py)、[safety](../../terminal_marl/core/rotating_slots/safety.py)、[world](../../terminal_marl/core/rotating_slots/world.py)、[checkpoint](../../terminal_marl/core/rotating_slots/performance/checkpoint.py)、[运行与审计脚本](../../scripts/server/compare_rotating_slots_practical.py)；配套 [targeted energy](../../tests/test_rotating_slots_targeted_energy.py)、[safety](../../tests/test_rotating_slots_targeted_safety.py)、[world](../../tests/test_rotating_slots_targeted_world.py)、[runner](../../tests/test_rotating_slots_practical_runner.py) 测试及既有 physics/world 边界 fixture。
