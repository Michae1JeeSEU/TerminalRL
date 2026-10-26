# 完整模型验证状态

更新日期：2026-10-26；最新原筛查快照为 UTC 09:00:08，后续批次于 UTC 09:21:50 启动并通过启动验证，heartbeat 于 UTC 09:24:23 更新。证据类型为 **newly_run**；seed 11 几何修复 8/8 完成，72-job 非几何研究与 36-job 同版本几何研究的完整结果仍为 **pending**。方法、需求配对和统计口径见[验证协议](full_capacity_validation_protocol.md)，必要界见[推导补充](full_capacity_bounds_derivation.md)。资格检查与 pilot 的逐 job 摘录见[资格检查 CSV](tables/full_capacity_qualification.csv)和[pilot CSV](tables/full_capacity_pilot.csv)。

## 2026-10-26 09:24 UTC 当前状态

原筛查 A（`screen`）在 [UTC 09:00:08 快照](../../../training_outputs/terminal/full_service_capacity_20261025/screen/heartbeat_check_20261026T085946Z.json)中已有 95/96 个终态：75 COMPLETED、20 个已知 INVALID、1 个运行中，无新增失败。排除旧实现的全部 24 个几何 jobs（20 INVALID、4 COMPLETED）后，非几何研究为 71/72 完成；唯一未完成项为 `high_variance66_rate180_seed71_rh_cp_sat`，PID 234402、CPU 18，仿真时间 1814.036 s、墙钟 19409.9 s。该快照无 `run.exit`，不能宣称 A 已全部完成。

seed 11 修复批次 B（`geometry_repair_seed11_v2`）已完成 8/8，`run.exit=0`，全部运行至 3000 s，实际结果与图已由主对话复核。90/180 h⁻¹ 的测量 cohort 联合完成数分别为 40/40、75/75，应急、失败及未决均为 0。详见[结果笔记](full_capacity_geometry_seed11.md)、[8 行 CSV](tables/full_capacity_geometry_seed11.csv)及[原始报告](../../artifacts/full_capacity_20261025/geometry_repair_seed11_v2_report/report.json)。单 seed 的初步图尚不定位容量阈值，也未作为最终论文图使用；B 不重跑。

当前按 source 版本分别组织两个研究：v1 非几何研究保留 72 jobs 及其自身 half6 对照；v2 几何研究为 36 jobs，即 24 个半径变体 jobs 加 12 个同 v2 half6 对照。两个研究封套 [full_nongeometry_study_v1.json](../../experiments/service_capacity/full_nongeometry_study_v1.json) 与 [full_geometry_study_v2.json](../../experiments/service_capacity/full_geometry_study_v2.json) 均为 **analysis-only**，不能作为仿真运行规格。已有[非几何部分报告](../../artifacts/full_capacity_20261025/nongeometry_study_partial_20261026/report.json)为 71/72、排除 24 项；[几何部分报告](../../artifacts/full_capacity_20261025/geometry_study_partial_20261026/report.json)为 8/36、缺少 28 项，反映后续批次启动前的覆盖。工具要求共同 source/runtime/policy/protocol 完全一致、所选检查字段严格为 true，不放宽 source-hash 配对检查，不跨版本合并。

后续 C、D 均于 UTC 2026-10-26 09:21:50 启动，状态为 **RUNNING**，已确认进程、输入、progress 及 events/timeseries，尚无 `run.exit`；启动验证不等于完整结果通过。

| 批次及输出目录名 | 输入规格 | jobs / workers | wrapper / Python PID | CPU |
|---|---|---:|---|---|
| C：geometry_remaining_with_controls_v2 | [seeds 23/71：half6+radius250+radius350](../../experiments/service_capacity/full_geometry_remaining_with_controls_v2.json) | 24 / 12 | 261833 / 261837 | 0,2,4,6,8,10,14,16,17,19,20,21 |
| D：geometry_control_seed11_v2 | [seed 11：half6 对照](../../experiments/service_capacity/full_geometry_control_seed11_v2.json) | 4 / 4 | 261834 / 261838 | 24,25,26,27 |

C/D 启动验证分别确认 24/4 份输入、12/4 份 progress，并已有实际 events/timeseries；C 初检时一个 job 对应文件仍为空。仿真时间分别约 65–170/90.94–181.87 s，callback 18–49/23–45 次。主对话另读到 C/D 的非空 60 s timeseries 及约 115/100.35 s progress。A 其他 worker 在 5 s CPU 采样内增量为 0，复用其闲置核，未停止 A；A 的活动 CPU 18 及保留的 CPU 12/13 不占用。

[UTC 09:19:52 预检](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_followup_v2_preflight/preflight_20261026T091921Z.json)确认 14 组 manifest 与原筛查一致、CP solver 为 OPTIMAL、可用内存约 24.6 GB、磁盘约 561 GB，所选 CPU 为不同物理核且无繁重任务重叠。C/D 和 B 均执行同一冻结目录 `/home/user/projects/terminal/artifacts/full_capacity_20261025/frozen_source_v2`，source hash 为 `afa8b9fb780b2f63f1bfe1f08564798dc43f72e379137a7b0f5361d561271db9`；解释器为 `/home/user/envs/terminal_stage2_v1/bin/python`，`PYTHONDONTWRITEBYTECODE=1`，数值库线程数均为 1。C/D 的规格从主仓库绝对路径 `/home/user/projects/terminal/experiments/service_capacity/` 读取，未复制进冻结源码；两规格 SHA-256 分别为 `3edda7840e0d491cda320138fe6cb679997ecbeea35d164c4c5bb81bf5741b82`、`5361c624e57328564e47fef715d0e51839cb91c06f6ea4a685699955b86d9de4`。

C/D 输出分别为服务器 `/home/user/training_outputs/terminal/full_service_capacity_20261025/geometry_remaining_with_controls_v2` 与 `/home/user/training_outputs/terminal/full_service_capacity_20261025/geometry_control_seed11_v2`，同步本地同名目录，根路径为 `D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025`。精确命令见 [C command.txt](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_remaining_with_controls_v2/command.txt) 和 [D command.txt](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_control_seed11_v2/command.txt)。

初始预计总耗时 C 为 9 h、D 为 6 h：B 的低率 job 为 2120–3672 s、高率为 11499–16217 s；C 的 1974 请求相对 B 的 644 请求，按 12 核平均 CPU 成本约 4.7 h，再考虑旧 half6 seed 71 单 job 23577 s 的尾部及非线性搜索开销，预留至 9 h；D 用 4 核并行同类基线任务，预留至 6 h。这些时间估计不用于容量拟合，也不是完成承诺。同一 heartbeat `automation-2` 已于 UTC 09:24:23 成功更新为 **ACTIVE**，间隔 538 min，扣除启动后已过时间，目标约 UTC 18:21:50（北京时间 10 月 27 日约 02:22）一起复查 A/C/D。等待期间无驻留监控 worker；唤醒后核对终态和完整性，再继续已授权的结果分析、图表及文档交付，未完成则按实际进度更新同一 heartbeat。

结果笔记、CSV、4 个规格及研究分析工具等 9 个新增文件已随提交 `945c5c976a061c6f378fecd21bfe4e6b7930e77b` 推送至 `origin/codex/arrival-restructure`。完整研究报告尚未就绪。

## 2026-10-26 00:45–01:00 UTC 历史观测与启动记录

[原始状态快照](../../../training_outputs/terminal/full_service_capacity_20261025/screen/capacity_check_20261026T0045Z.json)记录 96 jobs 中 53 个 **COMPLETED**、20 个 **INVALID**、12 个运行中、11 个尚未启动；合计 96，尚无 `run.exit`。原始输入、需求及输出保持不变，其余不涉及半径变体的 72 jobs 继续原筛查。当前不能据此给出完整模型容量结论。

20 个 INVALID 的原始异常分布如下。异常是运行有效性记录，不计为正常着陆损失或容量观测。

| profile | 范围 | jobs | 原始观测 |
|---|---|---:|---|
| radius350 | 两方法、90/180 h⁻¹、seeds 11/23/71 | 12 | `invalid slot` |
| radius250 | 两方法、180 h⁻¹、seeds 11/23/71 | 6 | physical world safety-stop，`failed=True, terminated_invalid=True` |
| radius250 | energy_slack、90 h⁻¹、seeds 11/23 | 2 | `AUTHORITATIVE_TRAJECTORY_MISMATCH` |

radius250 另有 4 个 jobs 标记 COMPLETED。不过，旧实现下 radius250/radius350 的全部 24 jobs 均暂时排除在几何因素推断之外，包括这 4 个已完成 jobs。主对话已观察到 `TrajectoryPlan.state` 及 `intervals` 在轨迹前后延续时使用默认几何，未沿用半径变体；修复后的 diff 已通过主对话复核，前后圆周延续改为从已序列化的边界极坐标系数推导，不再读取默认配置。payload/hash schema 和物理安全阈值均未改变。该历史观测时尚未通过修复后的正式重跑验证，不能断言 6 个 safety-stop 已全部解决，也不能把旧几何组的异常转化为策略或物理容量结论。

修复的本地新增测试为 13 passed；最终范围回归为 121 passed、2 skipped，耗时 25.30 s。本地缺少 OR-Tools，不能用本地结果替代服务器 CP 证据。服务器[新增几何测试日志](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_repair_seed11_v2_preflight/pytest.log)为 13 passed、0.92 s，退出码 0；[API/需求预检](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_repair_seed11_v2_preflight/api.log)确认 OR-Tools 9.15.6755 的 solver preflight 为 OPTIMAL，且 4 组 profile/rate 的 seed 11 manifest 与原筛查完全一致，每个 profile 的 90/180 h⁻¹ 完整到达时域分别为 56/105 请求。该 solver 预检与启动证据不表示 8 个正式 jobs 已通过。将修复前 HEAD 中的原始方法仅在内存中恢复后的红色回归为 4 failed、2 passed、7 deselected，复现 radius250 几何错误及 radius350 的 slot 20 异常。另有未纳入修复改动的 `performance_execution_reuse` 测试在第 260 行因 `SimpleNamespace` 缺少 `_stage2_enabled` 失败，使用原始及修复方法均出现相同失败；该项不计入上述通过数。这些检查验证实现边界，不提供新的容量或理论结果。

[部分报告](../../artifacts/full_capacity_20261025/screen_partial_20261026_0045_report/report.json)的覆盖为 53 complete / 20 failed / 23 incomplete（后者包含 12 个运行中和 11 个尚未启动），无遗漏、重复或未预期 jobs。53 个完整结果的配置、manifest、cohort 和请求守恒检查通过，遗漏飞机数为 0；能量恒等式最大绝对残差为 0.0004319995641708374 J，等待时间重建最大绝对残差为 4.547473508864641e-12 s。这些输出一致性检查不解除上述 4 个旧几何完整结果的排除状态。

UTC 01:00 的启动验证时，seed 11 几何修复状态为 **RUNNING**，完整结果当时为 **pending**。[修复规格](../../experiments/service_capacity/full_geometry_repair_seed11_v2.json)仅从正式筛查规格中选取 radius250/radius350 和 seed 11，保留两方法、90/180 h⁻¹、warmup/measurement/drain=600/1800/600 s、SLA=600 s、progress=60 s 及 Poisson 需求，共 2×2×2×1=8 jobs；未更改随机流协议、几何定义或物理与安全参数。UTC 2026-10-26 00:59:16 启动，wrapper PID 252556，Python parent PID 252558；6 workers 绑定 CPU 0、2、24、25、26、27，与原筛查 12 核及保留的 CPU 12 不重叠，数值库线程数均为 1。[启动快照](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_repair_seed11_v2/startup_snapshot_20261026T010015Z.json)的内部观测时间为 UTC 01:00:30.375185：8 份输入目录，6 份有效 progress 均超过 100 s 仿真时间，radius250 的 90 h⁻¹ 为 232.81–232.91 s、180 h⁻¹ 为 103.96–116.31 s，radius350 的 90 h⁻¹ 为 190.78 s；6 个活动 jobs 均已有非空 events/timeseries。该快照尚无 `run.exit`，不表示完整运行通过。

修复验证初始预计总耗时 T0=8 h：8 jobs 中两档到达率各 4 个，已有有效 half6 jobs 的平均耗时在 90/180 h⁻¹ 分别约 2143/15884 s，按 6 核平均分摊得到 `(4×2143+4×15884)/6≈12018 s`，约 3.34 h 的 CPU 平均分摊基线；已有单 job 最长约 23577 s（6.55 h），再为未知几何尾部开销预留至 8 h。原筛查剩余 23 jobs 按同负载下未失败 jobs 的成本及当前阶段粗估约 6 h，并保留尾部裕量；未使用 73/96 的 job 数比例作线性外推。这些估计仅用于复查安排，不是完成承诺或容量结论。

历史调度：heartbeat `automation-2` 于 UTC 2026-10-26 01:00:57 更新为 ACTIVE，间隔 479 min，当时计划约 UTC 08:59:16（北京时间 10 月 26 日约 17:00）复查 A/B。该调度已由上文 UTC 09:24:23 的更新接替；当时计划的其余几何 jobs 现已随 C 启动。历史输入和输出均保留。

修复及初始状态文档已随提交 `108c826d07cb2c76bb06bbae8ee12edc243f840c` 推送至 `origin/codex/arrival-restructure`。

## 2026-10-25 资格检查、pilot 与启动记录

单请求资格检查于 UTC 06:21:15–06:21:50 执行，wrapper PID 232454，8 workers，CPU 4、6、8、10、16、18、20、22，退出码 0。[分析报告](../../artifacts/full_capacity_20261025/qualification_report/report.json)核对了 8 profile×2 方法共 16/16 个完整结果，无缺失、重复或异常。每 job 只有一个请求，均正常接地，300 s 内正常接地且 SOC≥0.25 的联合比例为 1，请求守恒均成立；`rate120` 是单请求规格占位，不能解释为 120/h 运行表现。这些比例为单例观测，不构成统计置信结论。

| inner 半径（m） | 设计槽位 K | 模型计算稳定等待功率（kW） | 仿真接地延迟（s） | 运动学必要下界（s） |
|---:|---:|---:|---:|---:|
| 250 | 15 | 100.52 | 178.5 | 73.0 |
| 300 | 18 | 92.77 | 178.5 | 73.5 |
| 350 | 21 | 85.65 | 183.5 | 75.5 |

表中两方法的接地延迟相同，所有资格检查的环上等待均为 0。更大 K 没有在该自由流单例中产生更短航程时间；固定角速度下的稳定等待功率同时降低，但本例没有实际等待，因此不能据此给出容量排序或宣称 SOC 效应。报告中的能量恒等式最大绝对残差约为 3.54×10⁻⁸ J，等待时间重建残差为 0 s，遗漏飞机为 0；单请求的有限 cohort 数量上界 1 只是平凡一致性检查，不验证上界紧性。

Poisson pilot 于 UTC 06:22:04–06:37:06 完成，wrapper PID 232860，2 workers，CPU 24、26，退出码 0。[pilot 报告](../../artifacts/full_capacity_20261025/pilot_report/report.json)核对 2/2 个完整结果，half6、seed 11、120/h，warmup/measurement/drain/SLA 为 0/300/600/600 s。两方法均为 12 offered、12 正常接地、12 满足 600 s 与 SOC≥0.25 联合条件，无未决结局；这是一个配对 seed 的端到端检查与计算成本标定，不是容量证明。

| pilot 方法 | 墙钟耗时（s） | 延迟 p95（s） | middle 等待（aircraft-s） | inner 等待（aircraft-s） | 总等待（aircraft-s） |
|---|---:|---:|---:|---:|---:|
| energy_slack | 689.378 | 215.204 | 152.089 | 约 0 | 152.089 |
| rh_cp_sat | 900.622 | 215.899 | 91.371 | 22.318 | 113.689 |

上述等待分配与延迟差异仅描述本次配对样本，不据此判定策略优劣。pilot 能量恒等式最大绝对残差约 2.22×10⁻⁷ J，等待时间重建残差约 2.84×10⁻¹³ s，遗漏飞机为 0。有限 cohort 上界及观测联合完成数均为 12；小规模、低拥堵 pilot 的相等计数不验证一般上界紧性。

96-job 正式筛查于 UTC 2026-10-25 06:41:42 启动，启动时状态为 **RUNNING**，结果为 **pending**。wrapper PID 234381，Python parent PID 234385；12 workers 绑定 CPU 4、6、8、10、16、17、18、19、20、21、22、23，均为不同物理核。UTC 06:42:22 的启动验证观测到 96 份输入、存活进程、12 份 progress，仿真时间范围 64.19–200.69 s，累计 391 次 callback，并已有有效 events/timeseries 产物；当时可用内存 30,276,020 kB、剩余磁盘 608,221,257,728 bytes，643 文件 source hash 与资格检查/pilot 相同。这些证据确认正常启动，不表示已有完整正式结果。

初始总耗时估计 T0=18 h：pilot 总 CPU 秒除以两方法共 24 个请求为 `(688.61+898.62)/24≈66.13` CPU-s/请求；首轮平均率 135/h、到达时域 2400 s 对应每 job 期望 90 请求，96 job 在 12 核下线性基线约 13.23 h，再为场景和负载异质性预留到 18 h。载荷及候选搜索成本可能不线性增长，18 h 只是初始运行时间估计，不是容量拟合或完成承诺。目前没有完整模型容量结论。

历史调度记录：heartbeat `automation-2` 于 UTC 2026-10-25 06:43:42 成功更新为 ACTIVE，间隔 1078 min，当时计划在 UTC 2026-10-26 00:41:42（北京时间 10 月 26 日 08:41）复查；该时间已过去，不代表当前下一次唤醒时间。当时等待期间没有驻留监控 worker；原定唤醒后核对完成状态与产物，继续结果分析、论文/文档更新及提交推送。当前观测与已更新调度见上文。

## 测试范围

以下计数分别对应不同测试调用，存在重叠，不相加为独立覆盖数。

| newly_run 检查 | 结果与限制 |
|---|---|
| 服务器预检：[精确命令](../../../training_outputs/terminal/full_service_capacity_20261025/preflight/command.txt)、[日志](../../../training_outputs/terminal/full_service_capacity_20261025/preflight/pytest.log) | 128 passed、1 skipped、2 failed；服务器同步源码目录不含 `.git`，两项失败均在 `git rev-parse HEAD` 返回 128：`test_stage2_service_profile.py::test_prepare_checkpoint_and_export_roundtrip_exact_config`、`::test_prepare_rejects_profile_provenance_mismatch` |
| 本地六文件回归：`test_capacity_config.py`、`test_stage2_service_profile.py`、`test_rotating_slots_practical_world.py`、`test_rotating_slots_stage2_planning.py`、`test_rotating_slots_rescue_declaration_v7.py`、`test_rotating_slots_upstream.py`（均在 `tests/`） | 91 passed、2 skipped，包含上述两项 Git 依赖测试；本地未安装 OR-Tools，不能用本地结果替代服务器 CP 执行证据 |
| [test_capacity_bounds.py](../../tests/test_capacity_bounds.py)、[test_service_capacity.py](../../tests/test_service_capacity.py) | 合计 30 passed |
| [test_full_capacity_report.py](../../tests/test_full_capacity_report.py)、[test_capacity_bounds.py](../../tests/test_capacity_bounds.py)、[test_service_capacity.py](../../tests/test_service_capacity.py) | 合计 41 passed |
| [test_full_service_capacity.py](../../tests/test_full_service_capacity.py) | 8 passed |
| 研究分析工具及原报告测试：`test_full_capacity_study.py`、`test_full_capacity_report.py` | 35 passed、12.4 s；实数据部分报告分别确认非几何 71/72、几何 8/36 的选择与覆盖，不表示完整研究已完成 |

## 输入与复现

原筛查运行源码冻结提交为 `6c64016f6a2d554f5f5fa4c719039ddd6ae58e31`，[643 文件清单](../../artifacts/full_capacity_20261025/source_freeze_v1.json)的 source hash 为 `d119f7744405a568f6862274d0b31608f524aae8536f8a9c2f8b7e75f816f606`；报告实现提交为 `6d1a3f`。源码经 Syncthing 同步，原筛查服务器执行冻结目录 `/home/user/projects/terminal/artifacts/full_capacity_20261025/frozen_source_v1`，解释器为 `/home/user/envs/terminal_stage2_v1/bin/python`。原始[资格检查目录](../../../training_outputs/terminal/full_service_capacity_20261025/qualification)、[命令](../../../training_outputs/terminal/full_service_capacity_20261025/qualification/command.txt)及[run.exit](../../../training_outputs/terminal/full_service_capacity_20261025/qualification/run.exit)，以及[pilot 目录](../../../training_outputs/terminal/full_service_capacity_20261025/pilot)和[run.exit](../../../training_outputs/terminal/full_service_capacity_20261025/pilot/run.exit)位于同步输出目录。

几何修复 v2 冻结源码以 v1 的 `6c64016f6a2d554f5f5fa4c719039ddd6ae58e31` 为基底，运行代码仅叠加修复提交 `108c826d07cb2c76bb06bbae8ee12edc243f840c` 的 `trajectory.py` 改动，另新增两个测试文件与修复规格，共 646 文件；[v2 清单](../../artifacts/full_capacity_20261025/source_freeze_v2.json)的 source hash 为 `afa8b9fb780b2f63f1bfe1f08564798dc43f72e379137a7b0f5361d561271db9`。服务器执行目录为 `/home/user/projects/terminal/artifacts/full_capacity_20261025/frozen_source_v2`。修复输出单独写入服务器 `/home/user/training_outputs/terminal/full_service_capacity_20261025/geometry_repair_seed11_v2`，同步至本地 `D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\geometry_repair_seed11_v2`；[实际启动命令](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_repair_seed11_v2/command.txt)如下，预检证据保存在相邻 [geometry_repair_seed11_v2_preflight](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_repair_seed11_v2_preflight) 目录。

```bash
taskset -c 0,2,24,25,26,27 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/user/envs/terminal_stage2_v1/bin/python scripts/analysis/run_full_service_capacity.py --spec experiments/service_capacity/full_geometry_repair_seed11_v2.json --output /home/user/training_outputs/terminal/full_service_capacity_20261025/geometry_repair_seed11_v2 --workers 6 --cpus 0,2,24,25,26,27
```

正式筛查输出为服务器 `/home/user/training_outputs/terminal/full_service_capacity_20261025/screen`，同步本地 `D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\screen`；[实际启动命令](../../../training_outputs/terminal/full_service_capacity_20261025/screen/command.txt)使用冻结源码目录及以下参数：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/user/envs/terminal_stage2_v1/bin/python scripts/analysis/run_full_service_capacity.py --spec experiments/service_capacity/full_factorial_screen_v1.json --output /home/user/training_outputs/terminal/full_service_capacity_20261025/screen --workers 12 --cpus 4,6,8,10,16,17,18,19,20,21,22,23
```

本地可从已完成的原始输出重建报告（不会重跑仿真）：

```powershell
.\venv\Scripts\python.exe scripts/analysis/analyze_full_service_capacity.py --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\qualification --output artifacts/full_capacity_20261025/qualification_report
.\venv\Scripts\python.exe scripts/analysis/analyze_full_service_capacity.py --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\pilot --output artifacts/full_capacity_20261025/pilot_report
```

以下为 A/C/D 完成并通过主对话复核后，在 `D:\7_codespace\terminal` 下重建两个同版本研究报告的命令；当前未执行这两条完整报告命令。`--figures` 生成结果图，图仍需按实际 seed 数解释和查看。

```powershell
.\venv\Scripts\python.exe scripts/analysis/analyze_full_capacity_study.py --spec experiments/service_capacity/full_nongeometry_study_v1.json --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\screen --output artifacts/full_capacity_20261025/nongeometry_study_report --figures
.\venv\Scripts\python.exe scripts/analysis/analyze_full_capacity_study.py --spec experiments/service_capacity/full_geometry_study_v2.json --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\geometry_repair_seed11_v2 --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\geometry_remaining_with_controls_v2 --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\geometry_control_seed11_v2 --output artifacts/full_capacity_20261025/geometry_study_report --figures
```

各阶段运行入口、规格和资源设置规则见[协议中的复现模板](full_capacity_validation_protocol.md#复现入口与运行记录)。B 已完成且结果经主对话复核；A/C/D 的完整结果和两项完整研究仍待复核。
