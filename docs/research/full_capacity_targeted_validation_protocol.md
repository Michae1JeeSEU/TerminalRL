# half6 载荷与时域的定向补充验证

状态：**PLANNED**。已完成的 72-job 非几何研究与 36-job 同版本几何研究没有结束全部研究工作；v2 的 half6 在 180/h 下三个 seed、两方法均达到 95% 联合成功描述性阈值，尚未定位容量阈值。本轮只增加以下 14 jobs，分别检查更高载荷和较长时域，不扩展整个因素矩阵，也不自动追加其他率、seed 或时域。

| 批次及规格 | profile / 标称率（/h） / seeds / 方法 | warmup / measurement / drain / SLA / progress（s） | jobs |
|---|---|---|---:|
| E：[full_baseline_transition_v2.json](../../experiments/service_capacity/full_baseline_transition_v2.json) | half6；210、240；11、23、71；energy_slack、rh_cp_sat | 600 / 1800 / 600 / 600 / 60 | 12 |
| F：[full_baseline_horizon_v2.json](../../experiments/service_capacity/full_baseline_horizon_v2.json) | half6；180；71；energy_slack、rh_cp_sat | 600 / 5400 / 600 / 600 / 60 | 2 |

E 的 210/240 h⁻¹ 均低于 half6 零损失 pad 必要上界 271.6981/h 和 ε=0.05 选择性松弛上界 296.6014/h。它们用于观察完整动态约束下的服务质量变化；低于资源上界不保证通过。F 预先选择既有运行中计算负担较高的 seed 71，作为时域压力敏感性检查，不解释为无偏重复抽样或稳定性证明。

两批均使用 Poisson 需求、既有 half6 物理配置和同一冻结 v2 源码 `/home/user/projects/terminal/artifacts/full_capacity_20261025/frozen_source_v2`，source hash 为 `afa8b9fb780b2f63f1bfe1f08564798dc43f72e379137a7b0f5361d561271db9`。解释器为 `/home/user/envs/terminal_stage2_v1/bin/python`，沿用现有 runtime/policy 身份，`PYTHONDONTWRITEBYTECODE=1`、数值库线程数为 1；不改引擎、几何、安全、SOC、服务时间或搜索协议。新规格从主仓库绝对目录 `/home/user/projects/terminal/experiments/service_capacity/` 读取，不复制到冻结目录。同版本基线仅取已有 v2 half6，不与 v1 基线直接配对；严格保留 source/runtime/policy 与相应协议字段检查，时域变更单列为 F 的显式对照。

## 需求配对与输出口径

本地仅调用冻结 v2 的 `validate_spec` 与 `generate_manifest`，未运行仿真。E 的 6 组 rate/seed、F 的 1 组 rate/seed 均验证两方法生成完全相同的 manifest。E 每种方法的完整到达数/测量 cohort 数如下；12 jobs 合计 1662 个完整到达请求，标称 Poisson 率不替代实际 cohort 计数。

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

## 计划资源与启动边界

拟定 E 使用 12 workers、CPU `0,2,4,6,8,10,14,16,17,19,20,21`；F 使用 2 workers、CPU `24,26`，保留 CPU 12/13。正式启动前须重新检查实际负载、物理核映射、内存、磁盘、冻结 source 与规格哈希、CP solver 和需求配对；资源配置以主对话预检后的明确命令为准。输出计划为服务器 `/home/user/training_outputs/terminal/full_service_capacity_20261025/baseline_transition_v2` 和 `baseline_horizon_v2`，同步至本地 `D:\7_codespace\training_outputs\terminal\full_service_capacity_20261025` 下同名目录，分别保存输入、命令、进度和产物。

初始运行预算暂取 E 约 18 h、F 约 24 h，整体按约 24 h 安排后续复查，尚无实际启动或预计完成时刻。F 的完整请求数相对原 seed 71 为 `294/116≈2.5345`；将既有 ES/CP 耗时 17132.60/25555.34 s 仅按请求数粗略缩放得到约 12.06/17.99 h，再为状态积累与非线性搜索尾部预留至 24 h。E 的载荷、请求数及搜索负担增加，暂留 18 h；这些是运行预算，不是容量拟合或完成承诺。主对话复核本地输入检查、提交和服务器预检后另行下达启动命令，确认进程、日志和实际中间产物后才能改为 RUNNING；本协议不宣称已经启动，也不授权自动扩展实验。
