# 完整模型验证状态

更新日期：2026-10-26；最新筛查快照为 UTC 2026-10-26 00:45:42.970345。证据类型：资格检查、Poisson pilot、测试及本次筛查快照为 **newly_run**；完整正式筛查及几何修复验证为 **pending**。方法、需求配对和统计口径见[验证协议](full_capacity_validation_protocol.md)，必要界见[推导补充](full_capacity_bounds_derivation.md)。可纳入版本管理的逐 job 摘录见[资格检查 CSV](tables/full_capacity_qualification.csv)和[pilot CSV](tables/full_capacity_pilot.csv)，数值直接选自对应报告。

## 2026-10-26 筛查观测与修复计划

[原始状态快照](../../../training_outputs/terminal/full_service_capacity_20261025/screen/capacity_check_20261026T0045Z.json)记录 96 jobs 中 53 个 **COMPLETED**、20 个 **INVALID**、12 个运行中、11 个尚未启动；合计 96，尚无 `run.exit`。原始输入、需求及输出保持不变，其余不涉及半径变体的 72 jobs 继续原筛查。当前不能据此给出完整模型容量结论。

20 个 INVALID 的原始异常分布如下。异常是运行有效性记录，不计为正常着陆损失或容量观测。

| profile | 范围 | jobs | 原始观测 |
|---|---|---:|---|
| radius350 | 两方法、90/180 h⁻¹、seeds 11/23/71 | 12 | `invalid slot` |
| radius250 | 两方法、180 h⁻¹、seeds 11/23/71 | 6 | physical world safety-stop，`failed=True, terminated_invalid=True` |
| radius250 | energy_slack、90 h⁻¹、seeds 11/23 | 2 | `AUTHORITATIVE_TRAJECTORY_MISMATCH` |

radius250 另有 4 个 jobs 标记 COMPLETED。不过，旧实现下 radius250/radius350 的全部 24 jobs 均暂时排除在几何因素推断之外，包括这 4 个已完成 jobs。主对话已观察到 `TrajectoryPlan.state` 及 `intervals` 在轨迹前后延续时使用默认几何，未沿用半径变体；修复后的 diff 已通过主对话复核，前后圆周延续改为从已序列化的边界极坐标系数推导，不再读取默认配置。payload/hash schema 和物理安全阈值均未改变。尚未通过修复后的正式重跑验证，不能断言 6 个 safety-stop 已全部解决，也不能把旧几何组的异常转化为策略或物理容量结论。

修复的新增测试为 13 passed；最终范围回归为 121 passed、2 skipped，耗时 25.30 s。本地缺少 OR-Tools，CP 服务器验证仍为 pending。将 HEAD 中的原始方法仅在内存中恢复后的红色回归为 4 failed、2 passed、7 deselected，复现 radius250 几何错误及 radius350 的 slot 20 异常。另有未纳入修复改动的 `performance_execution_reuse` 测试在第 260 行因 `SimpleNamespace` 缺少 `_stage2_enabled` 失败，使用原始及修复方法均出现相同失败；该项不计入上述通过数。这些检查验证实现边界，不提供新的容量或理论结果。

[部分报告](../../artifacts/full_capacity_20261025/screen_partial_20261026_0045_report/report.json)的覆盖为 53 complete / 20 failed / 23 incomplete（后者包含 12 个运行中和 11 个尚未启动），无遗漏、重复或未预期 jobs。53 个完整结果的配置、manifest、cohort 和请求守恒检查通过，遗漏飞机数为 0；能量恒等式最大绝对残差为 0.0004319995641708374 J，等待时间重建最大绝对残差为 4.547473508864641e-12 s。这些输出一致性检查不解除上述 4 个旧几何完整结果的排除状态。

几何修复验证为 **PLANNED**，尚未启动。[seed 11 修复规格](../../experiments/service_capacity/full_geometry_repair_seed11_v2.json)仅从正式筛查规格中选取 radius250/radius350 和 seed 11，保留两方法、90/180 h⁻¹、warmup/measurement/drain=600/1800/600 s、SLA=600 s、progress=60 s 及 Poisson 需求，共 2×2×2×1=8 jobs；未更改随机流协议、几何定义或物理与安全参数。主对话复核修复实现及这 8 个 jobs 的有效输出后，再决定执行其余 seeds 23/71 的 16 个几何 jobs；新输出应与原筛查分开保存，原始输入、需求及输出不覆盖。本段不构成已启动或已验收的记录。

下一次复查时间及同一 heartbeat 的更新确认 **pending**；本次未记录新的已生效调度时间。下文 2026-10-25 的 heartbeat 时间为历史启动记录，不能视为当前下一次唤醒时间。

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

历史调度记录：heartbeat `automation-2` 于 UTC 2026-10-25 06:43:42 成功更新为 ACTIVE，间隔 1078 min，当时计划在 UTC 2026-10-26 00:41:42（北京时间 10 月 26 日 08:41）复查；该时间已过去，不代表当前下一次唤醒时间。当时等待期间没有驻留监控 worker；原定唤醒后核对完成状态与产物，继续结果分析、论文/文档更新及提交推送。当前观测与待更新调度见上文。

## 测试范围

以下计数分别对应不同测试调用，存在重叠，不相加为独立覆盖数。

| newly_run 检查 | 结果与限制 |
|---|---|
| 服务器预检：[精确命令](../../../training_outputs/terminal/full_service_capacity_20261025/preflight/command.txt)、[日志](../../../training_outputs/terminal/full_service_capacity_20261025/preflight/pytest.log) | 128 passed、1 skipped、2 failed；服务器同步源码目录不含 `.git`，两项失败均在 `git rev-parse HEAD` 返回 128：`test_stage2_service_profile.py::test_prepare_checkpoint_and_export_roundtrip_exact_config`、`::test_prepare_rejects_profile_provenance_mismatch` |
| 本地六文件回归：`test_capacity_config.py`、`test_stage2_service_profile.py`、`test_rotating_slots_practical_world.py`、`test_rotating_slots_stage2_planning.py`、`test_rotating_slots_rescue_declaration_v7.py`、`test_rotating_slots_upstream.py`（均在 `tests/`） | 91 passed、2 skipped，包含上述两项 Git 依赖测试；本地未安装 OR-Tools，不能用本地结果替代服务器 CP 执行证据 |
| [test_capacity_bounds.py](../../tests/test_capacity_bounds.py)、[test_service_capacity.py](../../tests/test_service_capacity.py) | 合计 30 passed |
| [test_full_capacity_report.py](../../tests/test_full_capacity_report.py)、[test_capacity_bounds.py](../../tests/test_capacity_bounds.py)、[test_service_capacity.py](../../tests/test_service_capacity.py) | 合计 41 passed |
| [test_full_service_capacity.py](../../tests/test_full_service_capacity.py) | 8 passed |

## 输入与复现

运行源码冻结提交为 `6c64016f6a2d554f5f5fa4c719039ddd6ae58e31`，[643 文件清单](../../artifacts/full_capacity_20261025/source_freeze_v1.json)的 source hash 为 `d119f7744405a568f6862274d0b31608f524aae8536f8a9c2f8b7e75f816f606`；报告实现提交为 `6d1a3f`。源码经 Syncthing 同步，服务器执行冻结目录 `/home/user/projects/terminal/artifacts/full_capacity_20261025/frozen_source_v1`，解释器为 `/home/user/envs/terminal_stage2_v1/bin/python`。原始[资格检查目录](../../../training_outputs/terminal/full_service_capacity_20261025/qualification)、[命令](../../../training_outputs/terminal/full_service_capacity_20261025/qualification/command.txt)及[run.exit](../../../training_outputs/terminal/full_service_capacity_20261025/qualification/run.exit)，以及[pilot 目录](../../../training_outputs/terminal/full_service_capacity_20261025/pilot)和[run.exit](../../../training_outputs/terminal/full_service_capacity_20261025/pilot/run.exit)位于同步输出目录。

正式筛查输出为服务器 `/home/user/training_outputs/terminal/full_service_capacity_20261025/screen`，同步本地 `D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\screen`；[实际启动命令](../../../training_outputs/terminal/full_service_capacity_20261025/screen/command.txt)使用冻结源码目录及以下参数：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/user/envs/terminal_stage2_v1/bin/python scripts/analysis/run_full_service_capacity.py --spec experiments/service_capacity/full_factorial_screen_v1.json --output /home/user/training_outputs/terminal/full_service_capacity_20261025/screen --workers 12 --cpus 4,6,8,10,16,17,18,19,20,21,22,23
```

本地可从已完成的原始输出重建报告（不会重跑仿真）：

```powershell
.\venv\Scripts\python.exe scripts/analysis/analyze_full_service_capacity.py --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\qualification --output artifacts/full_capacity_20261025/qualification_report
.\venv\Scripts\python.exe scripts/analysis/analyze_full_service_capacity.py --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\pilot --output artifacts/full_capacity_20261025/pilot_report
```

各阶段运行入口、规格和资源设置规则见[协议中的复现模板](full_capacity_validation_protocol.md#复现入口与运行记录)。正式筛查已通过历史启动验证，最新部分结果及几何排除范围见上文；完整完成状态与修复后的结果仍待复核。
