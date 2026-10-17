# 2026-10-16 正式诊断部分结果：benchmark、dt 与部分 traffic/scheduler

**2026-10-17 10:25 UTC 的服务器核验确认原正式 attempt 已中断。** 用户确认服务器曾断电并已恢复；SSH 已恢复，服务器重启后原诊断进程不存在。已有 60 个 `COMPLETED` 结果和 60 项通过的回放审计保留，最后的 `traffic_rate180_seed71` 没有结果或完整 replay，原退出标记与正式 `diagnostics.json` 也未生成。本次中断不记作该场景的物理 `ENERGY_FAILURE` 或实现故障结果。单项恢复现已启动并生成 progress/states，完整报告仍待完成。以下 benchmark、dt、traffic 和 scheduler 表格及结论仍基于既有产物；94 项目录保持 **92 PASS、G09 `NOT_RUN`、V12 `HUMAN_PENDING`**。

## 断电中断、同步恢复与单项重跑

[重启后直接核验](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_snapshot.json)记录服务器于北京时间 18:16:28 启动；10:25 UTC 查询时，旧 PID `3703967`、`3703969`、`3703989` 均不存在，也没有匹配的诊断进程。61 个目录中 60 项结果为 `COMPLETED`，仅 `traffic_rate180_seed71` 缺少 `result.json` 与 `replay/replay.json`，父目录 `diagnostics.exit` 和 `diagnostics/diagnostics.json` 均不存在。这次直接观测取代此前 SSH 不可达时的未知状态；此前默认沙箱的名称解析失败与提权后 SSH 握手关闭是不同阶段的历史观测，不能用其判定仿真的物理结局。

原 job 最后 progress 自身时间为 06:33:37.137329 UTC，记录仿真 1265 s、wall `71462.03031056374 s`。最终保留的 `states.jsonl` 为 **345685317 bytes**、mtime 06:46:58.951994 UTC，最后可见完整状态时间为 `1274.8430578523012 s`，文件尾不是完整 JSON 行；这些不是可恢复的完整 world checkpoint，也不能用 mtime 推断精确断电时刻。[本地重新哈希](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/local_hash_parity.json)与[服务器只读哈希](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_partial_sha.json)一致，states SHA-256 为 `5dea16f8cd568ad9353060e1ac6825534133b41630dbcc19300c248fe4bed025`；当前本地 27 个冻结源码/配置文件实算 SHA 与服务器及 f67 provenance 全部一致。

用户授权下的同步修复已完成：[服务器更名记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/syncthing_repair/server_rename_repair.json)保留 11 个历史路径的内容、mtime 与 SHA，仅将非法名称改为 Windows 兼容名称；[本地忽略配置记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/syncthing_repair/local_ignore_repair.json)在本地 `.stignore` 原 448-byte 前缀之后追加 28 个精确临时测试/cache 排除项。该配置不提交 Git，未删除文件或更改 ACL，原有 receive-only 本地额外文件保留。服务器 10:32 UTC 和本地 10:36:37 UTC 的快照均见 code/results 为 `idle`、errors/pullErrors/needTotalItems 为 0；这不要求两端全目录文件数量相同。

冻结内核没有完整 world checkpoint 恢复功能，原 CLI 不支持 resume，原路径重跑还会与既有 job 目录冲突。恢复编排使用新增的 [wrapper](../../scripts/server/recover_rotating_slots_diagnostics.py)，独立编排提交为 `3068e6ad8a5ffe12c8ac8ba212c1f2f72f6782f0`，脚本 SHA-256 为 `ae62823d50b32a4b540b2fb5c8c9e0d016e8155e26e1954e96e40eb980ba5836`；它在冻结 27 文件 glob 之外，物理数值代码仍为 f67。对应[定向验证记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/wrapper_tests.txt)为 33 PASS，包含真实 0.2 s 独立 pilot 的输出路径 smoke；该 pilot 不是正式矩阵结果或 G09 证据。

恢复协议先验证精确 61 项目录、60 项来源的 provenance/SHA 与 replay，再只将 `traffic_rate180_seed71` 按原冻结参数从仿真 0 重跑，输出到新的 `/home/user/training_outputs/terminal/rotating_slots_recovery_f67b5c8_20261017/diagnostics`。原输出所有字节、60 个结果对象及既有审计保留；单个独立 job 使用 1 worker，数值库线程均为 1。只有新项真实 `COMPLETED` 且 replay 验证通过，才将原 60 项与新 1 项按规范顺序形成完整 `diagnostics.json`，其中显式记录恢复来源、实际命令和两次 attempt 的计时，不将中断前 wall 拼成连续运行耗时。

[服务器只读预检](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_preflight/preflight_observation.json)已于 10:39:59.942–10:40:42.901 UTC 完成：`PREFLIGHT_PASSED`、退出码 0，精确目录 61 项、已验证来源 60 项，唯一重跑项为 `traffic_rate180_seed71`；stderr 为空，预检未创建目标 `diagnostics` 目录。

[实际启动记录](../../../training_outputs/terminal/rotating_slots_recovery_f67b5c8_20261017/parent/launch.json)确认单项恢复于服务器时钟 2026-10-17 10:47:24.664872 UTC 启动，wrapper PID `7697`、Python PID `7698`；工作目录仍为 `/home/user/projects/terminal`。实际命令如下：

```bash
/home/user/envs/terminal_env/bin/python scripts/server/recover_rotating_slots_diagnostics.py --source /home/user/training_outputs/terminal/rotating_slots_formal_f67b5c8_20261016/diagnostics --output /home/user/training_outputs/terminal/rotating_slots_recovery_f67b5c8_20261017/diagnostics --code-commit f67b5c8462433ba701191c493a37e4a8e472334e --recovery-commit 3068e6ad8a5ffe12c8ac8ba212c1f2f72f6782f0 --workers 1
```

新 root 的 `parent/recovery.log`、`parent/recovery.exit` 分别保存日志及退出标记，`parent/launch.json` 保存启动命令与进程记录。[原始启动后核验](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_launch/verification2.stdout)及[本地时间包围记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_launch/verification2.meta.json)确认：服务器快照时钟 10:50:13.097697–10:50:13.118350 UTC，7697/7698 仍运行、cwd/命令正确；`recovery_manifest.json` 为 `RUNNING`，仅重跑 seed 71、复用 60 项、1 worker。progress 自身时间 10:49:16.553638 UTC 记录仿真 `64.55795671792151/5400 s`、wall `70.699175142 s`，failed 0、`truncated=false`、`conservation=true`；states 为 1844750 bytes，最后完整状态时间 `77.00000000000003 s`、`failed=false`，退出标记未出现且日志为空。相同 SSH 调用的本地时间为 10:49:28.677594–10:49:29.051304 UTC，exit 0、stderr 为空；该次服务器时钟约比本地快 44.24 s，不沿用断电前的时钟差。

同一 heartbeat 已更新并读回 `ACTIVE`，约北京时间 **10 月 17 日 22:50** 首次复查。观察间隔为 4 小时，依据是该 seed 不同阶段成本不均且此前多小时仍持续推进；这是首次观察安排，不是任务完成 ETA，不能用当前低占用的初始化进度外推。实际唤醒可有数分钟误差；本地电脑及 Codex 应用需保持运行。

完成后只对新的最后一项执行独立回放审计并合并原 60 项索引，避免重复已有审计；G09 消费新恢复 root 的正式 `diagnostics.json`，其中原 40 项 benchmark 仍来自旧 root，不重跑。严格合并和 V12 人工验收仍待后续真实证据。

## 数据与测量条件

结果来自服务器输出的本地同步副本：`D:\7_codespace\training_outputs\terminal\rotating_slots_formal_f67b5c8_20261016\diagnostics\<job>\result.json`。60 项均记录源码提交 `f67b5c8462433ba701191c493a37e4a8e472334e`、源码集合 SHA-256 `110be32211d37e82772d806d86ae599d0a58a32e01af3465be836a58de902750`、配置语义 hash `0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53`，对应 replay 索引均存在。

服务器为 Intel Core i7-14700，28 个逻辑 CPU、20 个物理核；正式矩阵使用 20 workers。Python 3.12.3、NumPy 2.5.1、SciPy 1.18.1、PyYAML 6.0.3；psutil 未安装，RSS 来自 `resource.ru_maxrss`。OMP、OpenBLAS、MKL、NumExpr、VECLIB 线程均为 1，已记录的两个 OpenBLAS 线程池也均为 1。viewer 在计时仿真中关闭；cProfile 全程启用，使用 `process_time` 时钟。

以下是已执行命令记录，工作目录为 `/home/user/projects/terminal`：

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.scripts.run_stage1_diagnostics --output /home/user/training_outputs/terminal/rotating_slots_formal_f67b5c8_20261016/diagnostics --workers 20 --code-commit f67b5c8462433ba701191c493a37e4a8e472334e
```

断电前的有效直接状态见 [此前更正后的服务器快照](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/summary.md)，此前 benchmark/dt 快照见 [21:46 核验记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1346/summary.md)。benchmark/dt 算术汇总保存在 [partial_report_aggregates.json](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1346/partial_report_aggregates.json)，仅派生汇总，不替代或创建正式 `diagnostics.json`。

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

最大能源差为 `9.313225746154785e-8 J`，按 `(max−min)/min` 计算的相对差为 `6.744915891461807e-15`。相对 dt=0.2 s，三个最终位置的欧氏差均为 0 m；五个事件的类型、飞机 ID 与顺序一致，最大事件时差为 `3.410605131648481e-12 s`（dt=0.1 的服务结束事件），接地时差为 0 s。三份最小距离结果均为 `CERTIFIED`，区间宽度 `0.0007648839403344709 m`，小于 0.001 m，见证时刻均为 `58.366943359375 s`。这些 dt 对照只对应此固定脚本，不替代 traffic 与 scheduler 的结果分析。

## 十四个已完成 traffic 结果

下表覆盖截至快照已完成的十四个 traffic job，ID 格式为 `traffic_rate<rate>_seed<seed>`，直接取各 `result.json` 的实际 metrics；`failed` 为 `failed_before_landing`，`simtime` 为实际仿真停止时刻。[本轮 traffic 汇总](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/traffic_metrics.json)和[新增结果](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/new_completed.json)保留来源记录。job 的 `COMPLETED` 表示该次脚本已结束，包含物理失败导致提前终止的运行。

| rate | seed | planned | offered | unreleased | admitted | rerouted | normal | emergency | failed | censored | simtime / s | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 11 | 73 | 33 | 40 | 27 | 6 | 18 | 1 | 1 | 7 | 1391.8134693898965 | true |
| 60 | 23 | 52 | 52 | 0 | 50 | 2 | 47 | 3 | 0 | 0 | 5400 | false |
| 60 | 47 | 66 | 66 | 0 | 58 | 8 | 54 | 4 | 0 | 0 | 5400 | false |
| 60 | 71 | 72 | 28 | 44 | 24 | 4 | 12 | 3 | 1 | 8 | 1207.924968881649 | true |
| 60 | 101 | 74 | 74 | 0 | 64 | 10 | 61 | 3 | 0 | 0 | 5400 | false |
| 120 | 11 | 128 | 52 | 76 | 42 | 10 | 26 | 3 | 1 | 12 | 1242.4315865974008 | true |
| 120 | 23 | 117 | 45 | 72 | 40 | 5 | 26 | 2 | 1 | 11 | 1462.8111138523973 | true |
| 120 | 47 | 117 | 89 | 28 | 60 | 29 | 43 | 9 | 1 | 7 | 2500.3071263697407 | true |
| 120 | 71 | 118 | 36 | 82 | 25 | 11 | 12 | 2 | 1 | 10 | 791.6924626874761 | true |
| 120 | 101 | 138 | 33 | 105 | 26 | 7 | 9 | 4 | 1 | 12 | 784.8752089210166 | true |
| 180 | 11 | 188 | 44 | 144 | 32 | 12 | 10 | 2 | 1 | 19 | 688.1399611573279 | true |
| 180 | 23 | 193 | 38 | 155 | 30 | 8 | 11 | 1 | 1 | 17 | 809.4926152758579 | true |
| 180 | 47 | 184 | 50 | 134 | 34 | 16 | 16 | 3 | 1 | 14 | 1012.3628477209778 | true |
| 180 | 101 | 188 | 45 | 143 | 29 | 16 | 13 | 0 | 1 | 15 | 695.9349873539879 | true |

rate 60 的五个 seed 已齐：23、47、101 完成 3600 s 到达窗口及 1800 s drain，11、71 在能源失败时提前终止；rate 120 的五个 seed 也已齐，全部能源失败截尾，rate 180 仅 seed 71 尚未完成。当前十四项共三项完整、十一项能源失败截尾，截尾项均记录 `cost_is_lower_bound=true`；其累计代价是截尾下界，不能与完整窗口结果直接求均值比较。十四个结果均记录 `conservation=true`，未释放和截尾数量保留如表。

十一个 traffic 失败事件的实际时间线如下，依据对应飞机自身的 events 与序列化轨迹；此前事件见 [首批](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1608/failure_timelines_v2.json)、[第二批](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/failure_timelines.json)和[第三批](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0025/failure_timelines.json)，本轮两项及 scheduler_1.0 事件见 [本轮失败时间线](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/failure_timelines.json)。应急声明时间单列，不假定其总与转移完成同刻。

| rate / seed / 飞机 | middle→inner 转移完成 / s | 应急声明 / s | `ENERGY_FAILURE` / s |
|---|---|---|---|
| 60 / 11 / req_000019 | 1187.2060507520548 | 1187.2060507520548 | 1391.8134693898965 |
| 60 / 71 / req_000015 | 963.485330669024 | 963.485330669024 | 1207.924968881649 |
| 120 / 11 / req_000027 | 831.6699368659772 | 951.5348721919898 | 1242.4315865974008 |
| 120 / 23 / req_000026 | 1128.092429556767 | 1172.0000000000005 | 1462.8111138523973 |
| 120 / 47 / req_000068 | 2145.0 | 2209.403663635252 | 2500.3071263697407 |
| 120 / 71 / req_000005 | 465.2359218932885 | 500.83592189328846 | 791.6924626874761 |
| 120 / 101 / req_000007 | 365.21627249850053 | 493.9999999999998 | 784.8752089210166 |
| 180 / 11 / req_000010 | 510.7753377237965 | 510.7753377237965 | 688.1399611573279 |
| 180 / 23 / req_000002 | 500.2694426498897 | 518.6000000000008 | 809.4926152758579 |
| 180 / 47 / req_000017 | 695.9692601901377 | 721.6000000000004 | 1012.3628477209778 |
| 180 / 101 / req_000006 | 409.2134536844115 | 409.2134536844115 | 695.9349873539879 |

十一架飞机均在完成 middle→inner 转移后、inner 应急等待且尚未启动应急 final 时 SOC 耗尽。应急触发可能发生在转移中，也可能在转移结束后的等待中；声明时间如表。rate 180 / seed 23 与 rate 120 / seed 101 此前分别在 765.7694426498897 s、768.6882216022708 s 记录 `PAD_DEADLINE_MISSED`；本轮 rate 120 / seed 47 与 rate 180 / seed 47 分别在 2474.2993756860465 s、920.9692601901377 s 记录同类事件。这是当时记录的固定 pad 释放条件检查，不据此宣布全局不可行。回放检查及事件记录支持应急等待中的物理能源失败，未见已批准转移中意外失败的证据；共享数值内核边界仍适用。

本次使用的 `DeterministicDriver` 是验收驱动器，不是选定的研究 baseline 或优化器；当前十四个结果不代表全部 15 个 traffic 场景的完整统计结论，最后一个 traffic 仍待完成。

## scheduler 部分结果

`scheduler_1.0`、`scheduler_5.0`、`scheduler_10.0` 的原始结果已齐，均使用 rate 120、seed 11、3600 s 到达窗口与 1800 s drain，周期分别为 1/5/10 s。它们是单一 seed 的周期对照，不是额外的 traffic 随机 seed。三个周期及默认 `traffic_rate120_seed11` 的 128 项请求 manifest 一致。[本轮字段对照](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/scheduler_pair_comparison.json)完整递归比较 36 个 `result.metrics` 字段，无排除项；此前已核对默认 traffic 与同协议独立运行的 5 s job 在这些字段、事件类型/飞机 ID 序列及对应时刻均一致。

| 周期 / s | simtime / s | normal | emergency | failed | censored | 决策次数 | 累计代价下界 |
|---|---|---|---|---|---|---|---|
| 1 | 1242.4315865974004 | 26 | 3 | 1 | 12 | 1573 | 1022582.9083206373 |
| 5 | 1242.4315865974008 | 26 | 3 | 1 | 12 | 583 | 1022579.0339872935 |
| 10 | 1242.4315865974008 | 26 | 3 | 1 | 12 | 459 | 1022596.7258709894 |

三项均为能源失败截尾、`cost_is_lower_bound=true`，均记录 planned 128、offered 52、unreleased 76、admitted 42、rerouted 10 及 `conservation=true`。失败飞机均为 req_000027，831.6699368659772 s 完成 middle→inner 转移；1 s 周期于 951.6000000000001 s 在 inner 触发并声明应急，5/10 s 周期则为 951.5348721919898 s。三者均在 `EMERGENCY_WAITING` 耗尽 SOC、未启动应急 final，失败时刻最大差约 `4.55e-13 s`。本表保留实际停止时刻与代价下界，不将其当作完整窗口代价均值。

该 seed 的 1 s、10 s 分别相对 5 s 的记录差异如下；三项均为截尾结果：

| 记录量 | 1 s − 5 s | 10 s − 5 s |
|---|---|---|
| 累计空中时间 | +3.874333343748731 s | +17.691883695837532 s |
| 累计 holding 时间 | −11.125666656250587 s | +27.691883695837305 s |
| 决策次数 | +990（583→1573） | −124（583→459） |
| 能源 | +566547.5837806463 J（+0.15737432882792746 kWh） | +1845704.2151902914 J（+0.5126956153306423 kWh） |
| 累计代价下界 | +3.8743333438178524 | +17.69188369589392 |

三项各有 604 个事件，但 1 s 与 5 s 的序列在索引 232 开始分歧，5 s 与 10 s 的序列也有分歧，不将全部事件强行逐位配对计算时间差。相近失败时刻不代表整条轨迹或事件过程相同。`metrics.minimum_separation` 是粗粒度记录，不能作为连续距离证书或小于 0.001 m 精度的证明。这些差异只描述本 seed 的截尾对照，不支持选定最优周期、完整窗口均值或性能因果结论。

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

此前新增七项回放审计于 2026-10-16 20:29:31 UTC 在服务器启动，输出沿用 `independent_audits/<job>`；既有 46 项不重复执行。[query2 的五个失败结局审计](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/audit_batch/query2.txt)与[服务器 20:37:01 UTC query3 的两个完成结局审计](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/audit_batch/query3.txt)共同确认七项均退出码 0、PASS、零问题、连续间隔 `CERTIFIED`，守恒等式均为 true，`documented_hard_failure_runs` 均为 0。rate 60 / seed 47、101 的物理结局为 `COMPLETE`，其余新增五项仍为 `ENERGY_FAILURE`。当时合并的 [53 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_53_report.json)包含 53 个唯一 job、53 PASS、零问题，无重复或意外项，当时距全部 61 项尚余八项。

上一轮新增四项回放审计于 2026-10-17 00:26:20 UTC 在服务器启动，[00:28:54 UTC 查询](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0025/audit_batch/query1.txt)确认全部退出码 0、PASS、零问题、连续间隔 `CERTIFIED`，守恒等式均 true，`documented_hard_failure_runs` 均为 0；四项物理结局均为 `ENERGY_FAILURE`。[57 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_57_report.json)保留旧 53 项记录并加入四项，57 个唯一 job 均 PASS、零问题，无重复或意外项，当时尚余四项。本轮三个新完成回放于服务器 04:39:04 UTC 启动独立审计，输出仍为 `independent_audits/<job>`；见 [启动核验](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/audit_batch/startup.txt)。[04:41:51 UTC 查询](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/audit_batch/query1.txt)确认三项均退出码 0、PASS、零问题、连续间隔 `CERTIFIED`，守恒等式均为 true，`documented_hard_failure_runs` 均为 0，物理结局均保留为 `ENERGY_FAILURE`。[60 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_60_report.json)保留旧 57 项记录，累计 60 个唯一 job 均 PASS、零问题、连续间隔 `CERTIFIED`、退出码 0，无重复或意外项；61 项目录中仅剩 `traffic_rate180_seed71` 未纳入。

这里的独立检查是分离于 World 的记录重算与约束核对，仍共享冻结的轨迹、能源和连续间隔数值内核，不是独立实测 oracle。点质量能源替代模型及重建参考程序的边界见 [sources](sources.md) 和 [qualification](qualification.md)。最后一个 traffic 及正式报告、回放检查完成后，才能消费已有 40 项 benchmark 的正式 G09 证据并更新合并；本报告不改变 G09 或人工 V12 状态。
