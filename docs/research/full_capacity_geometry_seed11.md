# 半径变体修复后的 seed 11 结果

修复后的 radius250/radius350 在两方法、标称 90/180 aircraft/h 下共 8/8 jobs 完成，每个 job 均运行至 3000 s，`run.exit=0`。8 个测量 cohort 的正常接地、600 s 内接地且 SOC≥0.25 的联合完成数均等于 offered；应急、失败、上游未决及空中未决均为 0。本次仅有 seed 11，两个载荷点均未定位服务质量阈值，不能据此给出容量排序或统计显著性结论。

warmup/measurement/drain 为 600/1800/600 s；90/180 是 Poisson 标称率，完整到达时域每个 profile 分别有 56/105 请求，测量 cohort 分别为 40/75 请求。因此测量窗口的实际 offered 到达率为 40/0.5=80 和 75/0.5=150 aircraft/h；这两个归一化到达计数既非窗口接地吞吐，也非容量阈值。以下 p95 属于测量 cohort；inner 等待量统计全部已准入飞机，含 warmup 与 measurement 到达并跟踪至仿真结束，单位为 aircraft-s。

| 标称率（/h） | 方法 | inner 半径（m） | 联合完成/offered | cohort 延迟 p95（s） | 全部到达 inner 等待（aircraft-s） |
|---:|---|---:|---:|---:|---:|
| 90 | energy_slack | 250 | 40/40 | 231.054 | 20.000 |
| 90 | energy_slack | 350 | 40/40 | 251.170 | 86.302 |
| 90 | rh_cp_sat | 250 | 40/40 | 207.079 | 36.498 |
| 90 | rh_cp_sat | 350 | 40/40 | 236.395 | 62.464 |
| 180 | energy_slack | 250 | 75/75 | 301.957 | 1039.209 |
| 180 | energy_slack | 350 | 75/75 | 316.404 | 2352.394 |
| 180 | rh_cp_sat | 250 | 75/75 | 247.513 | 587.022 |
| 180 | rh_cp_sat | 350 | 75/75 | 258.957 | 1071.969 |

在同一 source 版本、同 seed 配对随机输入下，两方法在两个标称率上均观察到 radius350 的 p95 和 inner 等待量高于 radius250。这是单 seed 的描述性差异，呈现了半径变更中相互竞争的机制，未单独识别各机制的因果贡献。设计槽位数从 15 增至 21，稳定等待功率从 100.5215 降至 85.6519 kW，但接地时间必要下界从 73.0 增至 75.5 s。两者均为 6 pad、相同服务时间分布，故零损失 pad 必要上界同为 271.6981/h，允许非正常比例 ε=0.05 的选择性松弛上界同为 296.6014/h；这些资源上界不保证满足 SLA、SOC 或完整动态约束。

[现有理论](terminal_service_capacity.md)中，静态储存量为 \(K(R)=\lfloor\pi/\arcsin[d/(2R)]\rfloor\)；final 必要时间至少为径向位移/速度、垂直位移/速度与 \(|\omega|R/a_{h,\max}\) 三项的最大值，再叠加其他必经阶段。能量可行性还要求 \(Q(s_i-0.25)\ge E_{{\rm move},i}+\sum_rP_rW_{ir}\)，约束的是按功率加权的等待预算。因此不能从更大的 \(K\) 推出完整服务容量随半径单调增加；本次未拟合经验系数。详细推导见[必要界补充](full_capacity_bounds_derivation.md)，统计口径见[验证协议](full_capacity_validation_protocol.md)。

结果来自 [report.json](../../artifacts/full_capacity_20261025/geometry_repair_seed11_v2_report/report.json)；[8 行 CSV](tables/full_capacity_geometry_seed11.csv)直接选取逐 job 字段，保留完整数值精度，其中 pending 分为 upstream/airborne 两列，pad utilization 为报告的测量窗口占用比例。source hash 为 `afa8b9fb780b2f63f1bfe1f08564798dc43f72e379137a7b0f5361d561271db9`，修复提交为 `108c826d07cb2c76bb06bbae8ee12edc243f840c`。配置、manifest、cohort 与请求守恒检查通过，遗漏飞机为 0；能量恒等式最大绝对残差 4.090368747711182×10⁻⁶ J，等待重建残差 1.7053×10⁻¹³ s，有限 cohort 上界违例为 0。报告中的两张单 seed 曲线图仅为初步结果图，平坦的联合完成率不定位阈值，尚未作为最终论文图使用。

旧实现的 24 个几何 jobs（包括 4 个完成项）继续排除，原始输出保留，运行记录见[状态文档](full_capacity_validation_status.md)。后续 16 个几何 jobs（seeds 23/71）及 12 个同 v2 的 half6 对照为 **PLANNED**，尚无此后续批次的启动记录。几何研究将使用同版本的 36 jobs（24 几何+12 对照）；原 v1 的其余 72 jobs 继续其自身 half6 对照分析。保留报告中配对差分的严格 source-hash 检查，不将跨版本结果直接拼成同质研究。
