# 完整旋转槽位仿真的服务容量验证协议

协议日期：2026-10-25。本文保留完整 `rotating_slots` 模型的首轮验证范围；当前运行进展见[状态文档](full_capacity_validation_status.md)，已完成的 seed 11 几何修复结果见[结果笔记](full_capacity_geometry_seed11.md)，完整研究仍在进行。几何、资源、SOC 与服务容量的推导见 [terminal_service_capacity.md](terminal_service_capacity.md) 第 2–5 节；其中第 8 节的 inner–pad FCFS 结果属于简化机制实验，不作为本协议的完整终端结果。

## 研究对象与固定条件

本轮比较因果调度策略 `energy_slack` 和 `rh_cp_sat` 在相同外部需求下的正常着陆、应急、损失、延迟与积压表现。两者统一使用 `downstream_service_v8`、`quality_v5`、`budget_scale=1`、`online_policy=None`、`prefix_reuse=False`，应急规则为 `rescue_need_v7`，入场规则为 `shared_fifo_admission`。完整连续轨迹、能源积分、峰值预算、安全分离和资源预约均由现有引擎执行；调度计算耗时单独记录，当前模型不将其注入物理逻辑时钟。

统一的是墙钟预算上限及其他现有搜索限制，并非固定且相同的候选目录。几何和方法会改变计算耗时及预算内实际覆盖的候选，需同时记录 dispatch/solver 耗时、候选目录数量和截断情况。配对方法差异反映这些实现条件下的总体表现，不能只归因于资源分配目标函数。

每个 job 从空系统开始，既无在场飞机也无预置 pad 服务。请求在 warmup 加 measurement 的时段内释放，之后停止到达并执行指定长度的 drain。有限 drain 不保证所有请求已有终局，未完成请求保留在分母并单列。模型研究进场、最后进近、下降及地面服务，没有联合起飞、到离场混合冲突或充电过程。

配置由完整冻结 `service_half_v1` payload 派生，使用显式 `capacity_study_v1` 元数据；规范 YAML loader 仍只接受原有冻结 payload。除表内因素外，车辆、能源代理模型、环高度、角速度、门位置、安全阈值和机动搜索网格保持一致。静态布局检查不替代单次机动及联合执行的动态安全检查。

| profile | 相对 half6 的变化 | 保持与解释 |
|---|---|---|
| `half6` | 6 pad；30/60/120 s，概率 0.2/0.6/0.2 | 平均服务 66 s；inner 半径 300 m、18 槽位 |
| `original6` | 服务时长 60/120/240 s | 类别概率相同，平均 132 s |
| `pads3half` | 3 个等角间隔 pad | 半径仍为 100 m；角度 0、120、240 度，是原 6 pad 位置中的 0、2、4 号物理位置，新配置编号为 0、1、2 |
| `radius250` | inner 半径 250 m、15 槽位 | 100 m 设计弦长规则；固定角速度 |
| `radius350` | inner 半径 350 m、21 槽位 | 同一设计规则；固定角速度 |
| `low_soc` | 截断正态参数由 0.75、0.08、[0.60,0.90] 改为 0.60、0.08、[0.45,0.75] | 对相同 SOC uniform 标记作平移后的分位数映射，其他输入相同 |
| `deterministic66` | 三类时长均为 66 s | 类别概率保持 0.2/0.6/0.2，服务均值为 66 s、方差为 0 |
| `high_variance66` | 30/60/120 s 的概率改为 0.6/0/0.4 | 实际时长为 30 或 120 s，服务均值仍为 66 s |

半径对照同时改变槽位数、稳定圆周速度、等待功率、转移/进近轨迹与冲突几何。由此得到的是这一耦合设计变更的总体效应，不能单独归因于缓冲增大，也不预设容量随半径单调变化。等均值服务对照用于观察分布变化在完整模型中的表现，不把相同资源上界解释为相同运行结果。

## 配对需求与统计出口

需求生成器以 seed 派生相互独立的 arrival、gate、SOC、task 随机流。同一 seed、到达率和时域在不同方法/profile 下共用外部释放时刻、gate 以及逐请求的 SOC/task uniform 标记，调度方法不影响其生成。相同服务分布的对照共用任务类别；服务概率变化时使用同一 task uniform 作各自分布的映射，因此 `high_variance66` 的实际类别可以不同。`low_soc` 只改变同一 SOC uniform 的映射，不改变到达或任务；方法之间则使用完全相同的 manifest。不同到达率分别按相应 Poisson 参数生成，不能把不同率下的请求数视为相同。

单请求资格检查另采用 release=0、standard 类任务，非 low_soc profile 的初始 SOC 为 0.85，low_soc 使用其分布中位数。此时 `rates_per_hour=120` 是 schema 所需的占位值，不生成 120/h 的 Poisson 流，也不用于容量推断。

measurement cohort 由外部 release 落在 `[warmup, warmup+measurement)` 的请求组成。分母包含正常接地、应急、失效、转送以及仍在上游或空中的请求；warmup cohort 另外统计。SLA 延迟从外部 release 到接地计算，正常且按时并满足接地 SOC≥0.25 的比例使用同一全 offered 分母。类别结局也要保留，以识别选择性丢弃长任务产生的表面提升。

首轮筛查预先指定的主要描述性服务质量标准为：全部 offered 请求中，外部 release 后 600 s 内正常接地且接地 SOC≥0.25 的比例达到 95%。90% 和 99% 为同一指标的次要描述性敏感性阈值。各 profile×方法×率先报告逐 seed 比例，再报告三个 seed 的非加权均值与最小—最大范围；同时保留 offered 数和原始结局，避免合并请求掩盖 seed 差异。同一运行中的飞机共享资源、相互影响，不能当作独立 Bernoulli 样本计算二项置信区间。三个 seed 下达到这些描述性阈值不构成经过置信认证或稳定性证明的容量结论；资格检查仍使用 300 s SLA，仅用于单请求执行检查。

当前引擎没有将入场前等待实体化为空中飞行，因此上游等待期间不扣 SOC；上游等待仍计入外部 release SLA。该边界限制了结果对实际场外盘旋或转场的解释，不能宣称场外能耗为零。正常接地与地面服务完成分别处理，pad 占用包括 13.5 s 保护下降及后续地面服务。

同时报告两种不同的速率：measurement 钟表窗口内接地数除以窗口长度，以及该窗口释放 cohort 在 drain 终点前的正常结局数除以 measurement 长度。后者可包含 measurement 结束后的接地，不能称为同一钟表窗口的即时吞吐。观察窗口始末的上游/空中积压及其漂移为有限窗口描述，单窗口 drain 完成或零漂移都不证明长期稳定。

## 三阶段规格与启动边界

| 阶段与规格 | profile / 方法 / seed / 到达率 | warmup / measurement / drain / SLA（s） | job 数与用途 |
|---|---|---|---|
| [单请求资格检查](../../experiments/service_capacity/full_single_qualification_v1.json) | 8 profile；2 方法；11；120 占位；`single` | 0 / 300 / 0 / 300 | 16；检查各变体可运行、动态可行轨迹与实际执行，不估容量 |
| [配对 pilot](../../experiments/service_capacity/full_paired_pilot_v1.json) | half6；2 方法；11；120/h；Poisson | 0 / 300 / 600 / 600 | 2；检查端到端产物及实际计算成本，不估容量 |
| [首轮因素筛查](../../experiments/service_capacity/full_factorial_screen_v1.json) | 8 profile；2 方法；11、23、71；90、180/h；Poisson | 600 / 1800 / 600 / 600 | 96；比较已定因素和策略的有限窗口表现 |

各规格 progress interval 均为 60 s 的仿真进度写入间隔，不是服务器主动查询频率。首轮筛查只有两个载荷点：90/h 低于 `pads3half` 与 `original6` 的全服务 pad 必要上界，180/h 高于这两项上界，用于跨越资源工作量约束的区间。180/h 仍低于 half6 的该项上界，不能据此宣称已经定位 half6 的容量转变。这里的 factorial 仅指已列 profile×方法×seed×率的组合，不增加因素间的组合变体。

主对话先复核资格检查与 pilot 的退出状态、完成结果、配置/需求配对、请求守恒、轨迹与能源输出以及每种方法实际耗时，再决定首轮筛查的 CPU、worker 和初始总耗时估计 T0。**正式首轮筛查只能在主对话完成该复核并给出明确服务器命令后启动。** 本协议没有授权自动扩展载荷、时域、seed 或交互矩阵；后续细化由主对话依据结果另行确定。

## 必要资源界与运行结果的关系

全服务、固定类别混合且长期边界工作/H 消失时，pad 必需工作量给出

\[
C_{\mathrm{all}}^{UB}=\frac{3600m}{13.5+E[S]}.
\]

half6、deterministic66、high_variance66 的该项数值均约为 271.6981/h，pads3half 为 135.8491/h，original6 为 148.4536/h。这些是松弛其他约束的必要上界，不是完整终端可达容量或策略服务保证。推导和 pad 预约区间见 [研究笔记第 3 节](terminal_service_capacity.md#3-必需资源时间给出正常流的必要上界)。

若允许最多 ε 比例请求非正常完成，可选择丢弃较长任务，不能直接把上述全服务界用于 offered rate。固定 offered 类别概率 `p_c` 时，令 `y_c` 为每个 offered 请求中选择正常服务的类别质量，求 `0≤y_c≤p_c`、`Σy_c≥1−ε` 下最小的 `Σy_c(13.5+S_c)`，其倒数乘 `3600m` 才给出该选择性松弛下的必要 offered-rate 上界。对应实现由 [capacity_bounds.py](../../terminal_marl/analysis/capacity_bounds.py) 的 `loss_adjusted_pad_upper` 提供；类别接收域推导见 [研究笔记第 5 节](terminal_service_capacity.md#5-可接受需求率需要服务质量与稳定性定义)，完整界的边界见 [full_capacity_bounds_derivation.md](full_capacity_bounds_derivation.md)。主要 95% 联合成功标准蕴含正常比例至少 95%，因此 ε=0.05 的选择性 pad 界是相应必要资源检验；该松弛没有约束 SLA 或能量，不能替代联合成功标准，也不提供容量验收保证。次要 90%/99% 阈值同理对应 ε=0.10/0.01，不从结果图上事后挑选。

`rh_cp_sat` 只优化有限预算内构造的候选列，求解器对该候选模型报告的 bound/gap 不覆盖未生成轨迹、所有未来决策或所有可行策略。因此它不构成完整终端容量的全局上界。首轮结果只支持所测需求、时域和策略的比较；稳定性和长期服务质量仍需相应证据。

## 复现入口与运行记录

源码只在本地 `D:\7_codespace\terminal` 修改、检查并固定版本，经 Syncthing 同步到服务器 `/home/user/projects/terminal`。服务器仅通过 `ssh lab-server` 运行；先确认同步到对应本地版本，不使用手工上传源码。输出统一放在 `/home/user/training_outputs/terminal/full_service_capacity_20261025/` 的阶段子目录，不能写入源码目录。

以下是待主对话填入并确认的命令模板，未表示已经启动。执行前记录本地源码提交、服务器可用 CPU/负载/内存、worker/CPU 绑定、线程数、开始时间、日志路径和 PID。`WORKERS_*`、`CPUS_*`、每阶段 T0 均待资格检查/pilot 与服务器预检确定；不预填虚构的并行度或耗时。各数值库线程数固定为 1，实际 worker 数仍受可用 CPU 与内存约束。

```bash
cd /home/user/projects/terminal
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

/home/user/envs/terminal_stage2_v1/bin/python scripts/analysis/run_full_service_capacity.py \
  --spec experiments/service_capacity/full_single_qualification_v1.json \
  --output /home/user/training_outputs/terminal/full_service_capacity_20261025/qualification \
  --workers "${WORKERS_QUALIFICATION:?待主对话确定}" --cpus "${CPUS_QUALIFICATION:?待预检确定}"

/home/user/envs/terminal_stage2_v1/bin/python scripts/analysis/run_full_service_capacity.py \
  --spec experiments/service_capacity/full_paired_pilot_v1.json \
  --output /home/user/training_outputs/terminal/full_service_capacity_20261025/pilot \
  --workers "${WORKERS_PILOT:?待主对话确定}" --cpus "${CPUS_PILOT:?待预检确定}"

# 只有资格检查和 pilot 经主对话复核、明确下达正式命令后，才执行本段。
/home/user/envs/terminal_stage2_v1/bin/python scripts/analysis/run_full_service_capacity.py \
  --spec experiments/service_capacity/full_factorial_screen_v1.json \
  --output /home/user/training_outputs/terminal/full_service_capacity_20261025/screen \
  --workers "${WORKERS_SCREEN:?待pilot后确定}" --cpus "${CPUS_SCREEN:?待预检确定}"
```

运行时保留现有 runner 写出的规格、配置、需求、每 job identity、progress、events、decisions、dispatch timings、timeseries、aircraft、trajectories 和结果。完成检查同时核对退出状态、预期 16/2/96 job 的覆盖、缺失/重复/失败及结果完整性；进度到 100% 或进程存在不能单独代表成功。计算时间超过一小时或查询预算用尽时，按项目规则由主对话安排同一对话 heartbeat 接续；本协议文件本身不创建调度。
