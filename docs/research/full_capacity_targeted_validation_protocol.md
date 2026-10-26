# half6 载荷与时域的定向补充验证

状态：**RUNNING**。E/F 均于 UTC 2026-10-26 21:36:46 启动并通过启动验证，完整结果 pending。已完成的 72-job 非几何研究与 36-job 同版本几何研究没有结束全部研究工作；v2 的 half6 在 180/h 下三个 seed、两方法均达到 95% 联合成功描述性阈值，尚未定位容量阈值。本轮只增加以下 14 jobs，分别检查更高载荷和较长时域，不扩展整个因素矩阵，也不自动追加其他率、seed 或时域。

| 批次及规格 | profile / 标称率（/h） / seeds / 方法 | warmup / measurement / drain / SLA / progress（s） | jobs |
|---|---|---|---:|
| E：[full_baseline_transition_v2.json](../../experiments/service_capacity/full_baseline_transition_v2.json) | half6；210、240；11、23、71；energy_slack、rh_cp_sat | 600 / 1800 / 600 / 600 / 60 | 12 |
| F：[full_baseline_horizon_v2.json](../../experiments/service_capacity/full_baseline_horizon_v2.json) | half6；180；71；energy_slack、rh_cp_sat | 600 / 5400 / 600 / 600 / 60 | 2 |

E 的 210/240 h⁻¹ 均低于 half6 零损失 pad 必要上界 271.6981/h 和 ε=0.05 选择性松弛上界 296.6014/h。它们用于观察完整动态约束下的服务质量变化；低于资源上界不保证通过。F 预先选择既有运行中计算负担较高的 seed 71，作为时域压力敏感性检查，不解释为无偏重复抽样或稳定性证明。

两批均使用 Poisson 需求、既有 half6 物理配置和同一冻结 v2 源码 `/home/user/projects/terminal/artifacts/full_capacity_20261025/frozen_source_v2`，source hash 为 `afa8b9fb780b2f63f1bfe1f08564798dc43f72e379137a7b0f5361d561271db9`。解释器为 `/home/user/envs/terminal_stage2_v1/bin/python`，沿用现有 runtime/policy 身份，`PYTHONDONTWRITEBYTECODE=1`、数值库线程数为 1；不改引擎、几何、安全、SOC、服务时间或搜索协议。新规格从主仓库绝对目录 `/home/user/projects/terminal/experiments/service_capacity/` 读取，不复制到冻结目录。同版本基线仅取已有 v2 half6，不与 v1 基线直接配对；严格保留 source/runtime/policy 与相应协议字段检查，时域变更单列为 F 的显式对照。

搜索使用固定墙钟预算，分配的核心执行速度和负载可能变化，从而改变预算内候选覆盖；相同 source/runtime 和输入前缀不保证决策搜索完全相同。保留实际 CPU affinity，分析时检查已有的候选目录及截断指标；当前未观察或归因任何硬件效应。F 用于时域压力敏感性检查，不能解释为纯粹的时域因果效应或稳定性证明，口径与[首轮协议](full_capacity_validation_protocol.md)一致。

## 需求配对与输出口径

准备阶段本地仅调用冻结 v2 的 `validate_spec` 与 `generate_manifest`，未在本地运行仿真。E 的 6 组 rate/seed、F 的 1 组 rate/seed 均验证两方法生成完全相同的 manifest。E 每种方法的完整到达数/测量 cohort 数如下；12 jobs 合计 1662 个完整到达请求，标称 Poisson 率不替代实际 cohort 计数。

| 标称率（/h） | seed | 完整到达数 | 测量 cohort 数 |
|---:|---:|---:|---:|
| 210 | 11 | 128 | 93 |
| 210 | 23 | 118 | 88 |
| 210 | 71 | 135 | 95 |
| 240 | 11 | 151 | 111 |
| 240 | 23 | 141 | 109 |
| 240 | 71 | 158 | 116 |

F 的到达时域为 `[0,6000)`、仿真终点 6600 s，共 294 个到达，测量 cohort `[600,6000)` 为 260 个请求。其 `[0,2400)` 的 116 个请求、release、gate、SOC/task latent marks 与 [C 的 half6/rate180/seed71 原始输入](../../../training_outputs/terminal/full_service_capacity_20261025/geometry_remaining_with_controls_v2/manifests/half6_rate180_seed71.json)逐项完全一致。将长 manifest 截取并恢复 `arrival_window=2400` 后，完整 hash 为 `df7f4786bfd024f0da7737181ea4790a5c8dd080c219000b27bc2eadacce4597`，latent-marks hash 为 `7c34d08ce79047d134b2eba6a081a3ddaddd7b1e6e954814ef2ec03168b88a60`，均与原输入一致；F 全时域 manifest hash 为 `aa840013467886a8a57481260497e3bd8df6655a659f600630b36ff8c2ed51b6`，因时域和后续到达增加而不同。

联合成功仍要求正常接地、外部 release 后 600 s 内接地及 SOC≥0.25，以全部测量 offered 请求为分母；应急、失败、转送、上游未决和空中未决分列。E 报告每个载荷的全部逐 seed 结果、非加权均值和最小—最大范围，不预设通过率随载荷单调。F 除全测量 cohort 外，可显式报告 `[600,2400)`、`[2400,4200)`、`[4200,6000)` 三个互不重叠的 1800 s release cohort，输入计数分别为 82、92、86，均跟踪至本次终点；它们共享系统状态和资源，不作为三个独立重复。阈值口径沿用[首轮协议](full_capacity_validation_protocol.md)，经验网格集合的解释见[必要界补充第 11 节](full_capacity_bounds_derivation.md#11-允许损失的分位工作量与接收域外近似)。

## 运行记录与初始估时

E 使用 12 workers、CPU `0,2,4,6,8,10,14,16,17,19,20,21`，wrapper/Python PID 为 276420/276426；F 使用 2 workers、CPU `24,26`，PID 为 276421/276425，保留 CPU 12/13。两批开始时间均为 UTC 2026-10-26 21:36:46。[正式预检](../../../training_outputs/terminal/full_service_capacity_20261025/baseline_targeted_v2_preflight/preflight_20261026T213350Z.json)确认 solver OPTIMAL、无其他实验负载、内存约 30.60 GB 和磁盘约 600.07 GB。实际 E/F campaign 的 source（含文件清单与 hash）、policy、runtime 与 C 精确相等；外部规格 SHA-256 分别为 `b40852896277200d9f6807626124587cfe598cfc9b97f9b6d3cbe612baabca07`、`82427d02ca3479f7ce182e762450de8f1f095e15db34beaa6d16acfe536d2b0e`。

[启动快照](../../../training_outputs/terminal/full_service_capacity_20261025/startup_snapshot_20261026T213822Z.json)在 UTC 21:38:22 确认 E 的 12 份、F 的 2 份输入/identity 均存在，全部已有非空 events/timeseries；E 仿真时间 91.20–145 s、callback 31–56，F 为 125–135 s、callback 43–47，无错误、尚无 `run.exit`。主对话另独立读取 E/F 的实际 120 s timeseries。输出为服务器 `/home/user/training_outputs/terminal/full_service_capacity_20261025/baseline_transition_v2` 和 `baseline_horizon_v2`，同步至本地 `D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025` 下同名目录。两目录各保存 `run.sh`、`run.log`、`run.pid`、`start_utc.txt`、`command.txt`、`launch_metadata.json`；精确命令见 [E command.txt](../../../training_outputs/terminal/full_service_capacity_20261025/baseline_transition_v2/command.txt) 与 [F command.txt](../../../training_outputs/terminal/full_service_capacity_20261025/baseline_horizon_v2/command.txt)，工作目录为冻结 v2，规格使用主仓库绝对路径。

初始总耗时预算 E 约 18 h、F 约 24 h。E 用 `25555.34×158/116≈34808 s`（约 9.67 h）的单 job 请求数缩放基线，再为非线性搜索与拥堵尾部预留至 18 h。F 用 `294/116≈2.5345` 将既有 ES/CP 耗时 17132.60/25555.34 s 粗略缩放至约 12.06/17.99 h，再预留至 24 h；这些不是容量拟合或完成承诺。heartbeat `automation-2` 已于 UTC 21:40:24 更新成功为 ACTIVE，间隔 1436 min，扣除已过时间后计划约 UTC 2026-10-27 21:36（北京时间 10 月 28 日约 05:36）复查两批。等待期间无驻留监控 worker；先核对退出状态、完整产物与协议，再进行下述分析，不自动扩展实验。

## 完成后的分析入口

[full_baseline_transition_study_v2.json](../../experiments/service_capacity/full_baseline_transition_study_v2.json) 是 **analysis-only** 研究封套，`automatic_launch=false`，不能交给仿真运行器。它只选择同一 v2、标准 1800 s measurement 的 half6：C 中 8 jobs、D 中 4 jobs、E 中 12 jobs，共 24 jobs，覆盖 90/180/210/240 h⁻¹、三个 seed 和两方法；排除 C 的 16 个几何 jobs，不需要 B，也不增加运行任务。F 的 measurement=5400 s 协议不同，单独分析，不与该 24-job 研究合并。

以下命令须等对应批次完成并通过主对话复核后，在 `D:\7_codespace\terminal` 下执行；当前未执行，不据此生成或宣称新结果。

```powershell
.\venv\Scripts\python.exe scripts/analysis/analyze_full_capacity_study.py --spec experiments/service_capacity/full_baseline_transition_study_v2.json --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\geometry_remaining_with_controls_v2 --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\geometry_control_seed11_v2 --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\baseline_transition_v2 --output artifacts/full_capacity_20261025/baseline_transition_study_report --figures
.\venv\Scripts\python.exe scripts/analysis/analyze_full_service_capacity.py --campaign D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025\baseline_horizon_v2 --output artifacts/full_capacity_20261025/baseline_horizon_report --figures
```

F 的三个 1800 s release cohort 目前只验证了输入计数；后续须从原始 F 请求、事件及终态，按上文明确边界和同一 600 s SLA 做有范围的 cohort 分析或补充实现。现有单 campaign 命令不自动提供这三窗口结局，不填入未计算的数据，也不把窗口视为独立重复。
