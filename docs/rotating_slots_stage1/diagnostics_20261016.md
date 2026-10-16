# 2026-10-16 正式诊断部分结果：benchmark、dt 与部分 traffic

截至 **2026-10-16 20:27:08 UTC（北京时间 10 月 17 日 04:27:08）**，正式 61 项矩阵已完成全部 40 项 benchmark、3 项 dt 及 10 项 traffic，共 53 项；另有 5 项 traffic 与 3 项 scheduler 运行中，8 个 worker 忙碌，未观察到软件错误。服务器 20:37:01 UTC 的审计复查确认新增七项均已通过，累计 53 项回放检查 PASS、零问题、连续间隔 `CERTIFIED`。正式退出标记与 `diagnostics.json` 尚未生成，94 项目录仍为 **92 PASS、G09 `NOT_RUN`、V12 `HUMAN_PENDING`**。

## 数据与测量条件

结果来自服务器输出的本地同步副本：`D:\7_codespace\training_outputs\terminal\rotating_slots_formal_f67b5c8_20261016\diagnostics\<job>\result.json`。53 项均记录源码提交 `f67b5c8462433ba701191c493a37e4a8e472334e`、源码集合 SHA-256 `110be32211d37e82772d806d86ae599d0a58a32e01af3465be836a58de902750`、配置语义 hash `0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53`。

服务器为 Intel Core i7-14700，28 个逻辑 CPU、20 个物理核；正式矩阵使用 20 workers。Python 3.12.3、NumPy 2.5.1、SciPy 1.18.1、PyYAML 6.0.3；psutil 未安装，RSS 来自 `resource.ru_maxrss`。OMP、OpenBLAS、MKL、NumExpr、VECLIB 线程均为 1，已记录的两个 OpenBLAS 线程池也均为 1。viewer 在计时仿真中关闭；cProfile 全程启用，使用 `process_time` 时钟。

以下是已执行命令记录，工作目录为 `/home/user/projects/terminal`：

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.scripts.run_stage1_diagnostics --output /home/user/training_outputs/terminal/rotating_slots_formal_f67b5c8_20261016/diagnostics --workers 20 --code-commit f67b5c8462433ba701191c493a37e4a8e472334e
```

最新状态见 [04:27 服务器快照](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/summary.md)，此前 benchmark/dt 快照见 [21:46 核验记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1346/summary.md)。benchmark/dt 算术汇总保存在 [partial_report_aggregates.json](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1346/partial_report_aggregates.json)，仅派生汇总，不替代或创建正式 `diagnostics.json`。

## 1 秒 benchmark

初始稳定占用数为 10/30/60/83，两种模式各重复 5 次（`r0`–`r4`），共 40 个唯一 job。`stable` 保持稳定占用；`inflight_attempt` 包含额外在途准入构造，20/20 次记录 `inflight_admitted=true`，不能把它理解为纯稳定状态计算。每项仿真时长均为 1 s。83 槽位的五次 `inflight_attempt` 在 1 s 时均记录 `actual_airborne_count=84`，展示了本构造中槽位数不等于总空中容量上限。

表中 P95 使用现有 `benchmark_summary` 的 nearest-rank 定义：排序后第 `ceil(0.95×5)=5` 项，即五次中的最大值。吞吐列为每次“仿真秒 / 墙钟秒”的中位数；RSS 列为五次记录的最大值，按 1 MiB = 2²⁰ bytes 换算。

| 初始稳定占用数 | 模式 | 重复 | 墙钟中位数 / s | 墙钟 P95 / s | CPU 中位数 / s | 仿真秒/墙钟秒中位数 | 最大 RSS / MiB |
|---|---|---|---|---|---|---|---|
| 10 | inflight_attempt | 5 | 83.394 | 161.043 | 83.393 | 0.01199126 | 74.363 |
| 10 | stable | 5 | 182.719 | 186.850 | 182.691 | 0.00547287 | 73.723 |
| 30 | inflight_attempt | 5 | 282.765 | 545.233 | 282.764 | 0.00353650 | 76.129 |
| 30 | stable | 5 | 505.556 | 513.896 | 505.342 | 0.00197802 | 75.895 |
| 60 | inflight_attempt | 5 | 549.017 | 1050.870 | 549.012 | 0.00182144 | 78.508 |
| 60 | stable | 5 | 1033.240 | 1046.116 | 1032.870 | 0.00096783 | 77.441 |
| 83 | inflight_attempt | 5 | 1025.844 | 1980.378 | 1025.836 | 0.00097481 | 79.648 |
| 83 | stable | 5 | 1375.521 | 1488.345 | 1375.205 | 0.00072700 | 78.867 |

这些数值包含初始配置、seed 场景、参考候选构造及连续约束检查、cProfile 和回放归档，表示 **1 s fixture 的端到端成本**，不能当作预热后长运行的每仿真秒成本。cProfile 开销尚未单独量化。进程池会复用进程，`ru_maxrss` 是进程生命周期峰值，可能带入此前 job 的内存高水位，并非每项隔离测得的内存峰值。本协议没有绝对吞吐通过阈值。

这些计时来自 20 worker 同时运行的矩阵，并非隔离单进程基准；各模式中位数不能单独用于断言在途场景固有计算量更低。进程池跨 job 复用，未单独控制每项冷/热缓存条件，表格仅说明该协议下的观测。

### CPU 分类分布

下表对 40 项的 `measured_components.exclusive_cpu_seconds` 按类别求和，百分比以互斥分类合计 **25,532.733033 CPU s** 为分母。没有累加嵌套的 `world_cpu_seconds` 计时。

| 类别 | 含义 | exclusive CPU 合计 / s | 占比 |
|---|---|---|---|
| motion_power | 轨迹求值、形状/界限计算与能源函数 | 16170.903191 | 63.3340% |
| other | 其余已采样函数 | 8844.404061 | 34.6395% |
| separation | safety 间隔函数自身 | 375.407491 | 1.4703% |
| serialization | 回放与 JSON 归档函数自身 | 96.510470 | 0.3780% |
| generation | 轨迹生成函数自身 | 33.533340 | 0.1313% |
| state_update | 其他 world 状态更新函数自身 | 10.097483 | 0.0395% |
| emergency | 指定应急入口/参考路由函数自身 | 1.603405 | 0.0063% |
| resources | 资源日历函数自身 | 0.273593 | 0.0011% |

分类规则以函数归属计 exclusive 时间：应急路径调用的能源、轨迹、间隔函数各归自己的类别。因此参考候选形状检查与能源计算的大部分时间进入 `motion_power`，不能将较大的嵌套应急计时解释为应急路由函数自身耗时。以上是观测到的类别分布，不是进一步的原因归属。

## dt 对照

三个 job 使用同一已承诺正常 final 脚本、初始 SOC 0.8、short 服务，执行步长分别为 0.05/0.1/0.2 s，推进至 203.5 s。表中能源、SOC、接地与最终位置均取 `result.json` 顶层字段，连续最小距离取顶层 `minimum`，没有用粗粒度的 `metrics.minimum_separation` 替代证书。

| dt / s | 能源 / J | 最终 SOC | 接地 / s | 最终 ENU / m | 连续最小距离区间 / m |
|---|---|---|---|---|---|
| 0.05 | 13807771.506749442 | 0.6209387448548933 | 143.5 | [100, 0, 1.7763568394002505e-15] | [126.3420680482248, 126.34283293216514] |
| 0.1 | 13807771.506749535 | 0.6209387448548933 | 143.5 | [100, 0, 1.7763568394002505e-15] | [126.3420680482248, 126.34283293216514] |
| 0.2 | 13807771.506749481 | 0.6209387448548933 | 143.5 | [100, 0, 1.7763568394002505e-15] | [126.3420680482248, 126.34283293216514] |

最大能源差为 `9.313225746154785e-8 J`，按 `(max−min)/min` 计算的相对差为 `6.744915891461807e-15`。相对 dt=0.2 s，三个最终位置的欧氏差均为 0 m；五个事件的类型、飞机 ID 与顺序一致，最大事件时差为 `3.410605131648481e-12 s`（dt=0.1 的服务结束事件），接地时差为 0 s。三份最小距离结果均为 `CERTIFIED`，区间宽度 `0.0007648839403344709 m`，小于 0.001 m，见证时刻均为 `58.366943359375 s`。这些对照只对应此固定脚本，尚不覆盖运行中的流量或 scheduler 敏感性结果。

## 十个已完成 traffic 结果

下表覆盖截至快照已完成的十个 traffic job，ID 格式为 `traffic_rate<rate>_seed<seed>`，直接取各 `result.json` 的实际 metrics；`failed` 为 `failed_before_landing`，`simtime` 为实际仿真停止时刻。[traffic_metrics.json](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/traffic_metrics.json)保留原值、来源与守恒检查。job 的 `COMPLETED` 表示该次脚本已结束，包含物理失败导致提前终止的运行。

| rate | seed | planned | offered | unreleased | admitted | rerouted | normal | emergency | failed | censored | simtime / s | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 11 | 73 | 33 | 40 | 27 | 6 | 18 | 1 | 1 | 7 | 1391.8134693898965 | true |
| 60 | 23 | 52 | 52 | 0 | 50 | 2 | 47 | 3 | 0 | 0 | 5400 | false |
| 60 | 47 | 66 | 66 | 0 | 58 | 8 | 54 | 4 | 0 | 0 | 5400 | false |
| 60 | 71 | 72 | 28 | 44 | 24 | 4 | 12 | 3 | 1 | 8 | 1207.924968881649 | true |
| 60 | 101 | 74 | 74 | 0 | 64 | 10 | 61 | 3 | 0 | 0 | 5400 | false |
| 120 | 23 | 117 | 45 | 72 | 40 | 5 | 26 | 2 | 1 | 11 | 1462.8111138523973 | true |
| 120 | 71 | 118 | 36 | 82 | 25 | 11 | 12 | 2 | 1 | 10 | 791.6924626874761 | true |
| 120 | 101 | 138 | 33 | 105 | 26 | 7 | 9 | 4 | 1 | 12 | 784.8752089210166 | true |
| 180 | 23 | 193 | 38 | 155 | 30 | 8 | 11 | 1 | 1 | 17 | 809.4926152758579 | true |
| 180 | 101 | 188 | 45 | 143 | 29 | 16 | 13 | 0 | 1 | 15 | 695.9349873539879 | true |

rate 60 的五个 seed 已齐：23、47、101 完成 3600 s 到达窗口及 1800 s drain，11、71 在能源失败时提前终止；rate 120、180 尚未齐。当前十项中七项截尾均记录 `cost_is_lower_bound=true`，其累计代价是截尾下界，不能跨截尾结果与完整窗口结果直接求均值比较。十个结果均记录 `conservation=true`，未释放和截尾数量保留如表。

七个失败事件的实际时间线如下，依据对应飞机自身的 events 与序列化轨迹；原两项见 [failure_timelines_v2.json](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1608/failure_timelines_v2.json)，新增五项见 [failure_timelines.json](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/failure_timelines.json)。应急声明时间单列，不假定其总与转移完成同刻。

| rate / seed / 飞机 | middle→inner 转移完成 / s | 应急声明 / s | `ENERGY_FAILURE` / s |
|---|---|---|---|
| 60 / 11 / req_000019 | 1187.2060507520548 | 1187.2060507520548 | 1391.8134693898965 |
| 60 / 71 / req_000015 | 963.485330669024 | 963.485330669024 | 1207.924968881649 |
| 120 / 23 / req_000026 | 1128.092429556767 | 1172.0000000000005 | 1462.8111138523973 |
| 120 / 71 / req_000005 | 465.2359218932885 | 500.83592189328846 | 791.6924626874761 |
| 120 / 101 / req_000007 | 365.21627249850053 | 493.9999999999998 | 784.8752089210166 |
| 180 / 23 / req_000002 | 500.2694426498897 | 518.6000000000008 | 809.4926152758579 |
| 180 / 101 / req_000006 | 409.2134536844115 | 409.2134536844115 | 695.9349873539879 |

七架飞机均在完成 middle→inner 转移后、inner 应急等待且尚未启动应急 final 时 SOC 耗尽。新增五项中，四项在转移完成后才触发并声明应急，rate 180 / seed 101 则在转移中先触发、完成后声明。rate 180 / seed 23 与 rate 120 / seed 101 另分别在 765.7694426498897 s、768.6882216022708 s 记录 `PAD_DEADLINE_MISSED`；这是当时记录的固定 pad 释放条件检查，不据此宣布全局不可行。回放检查与事件记录支持这些应急等待中的物理能源失败，未见已批准转移违规的证据；共享数值内核的边界仍适用，不等同于全部实现正确。

本次使用的 `DeterministicDriver` 是验收驱动器，不是选定的研究 baseline 或优化器；当前十个结果不代表全部 15 个 traffic 场景的完整统计结论。其余五个 traffic 和三个 scheduler 结果仍待完成。

## 已完成回放检查与剩余边界

[43 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_43_report.json)记录了每项实际命令、退出码、审计文件和日志；43 项退出码均为 0、PASS、零问题、连续间隔 `CERTIFIED`。检查命令形式为本地 `venv\Scripts\python.exe -m terminal_marl.scripts.run_stage1_tests --audit <job的replay目录> --output <对应审计目录>`，完整实际参数以索引为准。

43 项的物理 episode 终态均为 `IN_PROGRESS`：benchmark 在 1 s 结束时仍有 holding/在途飞机；dt 的目标飞机已于 143.5 s 接地、203.5 s 完成服务，但 seed 的 `other_ground` 服务需 300 s，尚未结束。这与 job 已完成、回放检查 PASS 属于不同含义。

最先完成的三个 rate 60 回放的独立检查于 2026-10-16 16:12:50 UTC 在服务器并行启动，每项设置 1800 s 超时，输出位于正式运行父目录的 `independent_audits/<job>`。命令、PID 和日志位置见 [启动核验](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1608/audit_batch/startup.txt)，[16:18:43 UTC 查询](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1608/audit_batch/query2.txt)确认三个审计均已结束：

| rate 60 seed | 审计退出时刻 / UTC（2026-10-16） | 退出码 | 审计 | 物理结局 |
|---|---|---|---|---|
| 11 | 16:14:38 | 0 | PASS、0 issues、CERTIFIED | ENERGY_FAILURE |
| 23 | 16:16:35 | 0 | PASS、0 issues、CERTIFIED | COMPLETE |
| 71 | 16:15:34 | 0 | PASS、0 issues、CERTIFIED | ENERGY_FAILURE |

三份审计的守恒等式均为 true，`documented_hard_failure_runs` 均为 0。合并后的 [46 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_46_report.json)保留原 43 项记录并加入三个 traffic 的实际产物，不重跑已有检查。审计 PASS 表示记录与约束检查通过，物理能源失败仍保留为能源失败。

本轮新增七项回放审计于 2026-10-16 20:29:31 UTC 在服务器启动，输出沿用 `independent_audits/<job>`；既有 46 项不重复执行。[query2 的五个失败结局审计](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/audit_batch/query2.txt)与[服务器 20:37:01 UTC query3 的两个完成结局审计](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/audit_batch/query3.txt)共同确认七项均退出码 0、PASS、零问题、连续间隔 `CERTIFIED`，守恒等式均为 true，`documented_hard_failure_runs` 均为 0。rate 60 / seed 47、101 的物理结局为 `COMPLETE`，其余新增五项仍为 `ENERGY_FAILURE`。完整的 [53 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_53_report.json)包含 53 个唯一 job、53 PASS、零问题，无重复或意外项，距全部 61 项尚余八项。

这里的独立检查是分离于 World 的记录重算与约束核对，仍共享冻结的轨迹、能源和连续间隔数值内核，不是独立实测 oracle。点质量能源替代模型及重建参考程序的边界见 [sources](sources.md) 和 [qualification](qualification.md)。余下八项运行中结果及正式报告完成后，才能消费已有 40 项 benchmark 的正式 G09 证据并更新合并；本报告不改变 G09 或人工 V12 状态。
