# Selective repair v3：设计、验证与问题诊断（2026-10-29）

本轮研究在固定多环终端区内，改进局部重规划对象、候选覆盖和跨环补给的组织方式，并修正交付阶段已知交通变化的处理时序。研究问题是：在相同物理、安全、能源、pad 服务与全局目标下，有限的搜索工作能否产生更多可实际交付的正常动作，进而改善正常着陆效率与应急代价。

两种负载的新版本均已完成 300 秒，并通过保存轨迹的自动检查。四项实现及定向机制修复得到支持，但本轮整版开发配对没有显示整体质量改善：中负载正常着陆数持平，高负载下降，两者能耗、holding 与完整正常规划工作均增加。预测目标改善没有转化为已观察的整体收益。本记录不宣布优化成功、不进入 B，也不据此继续调权；旧合同与旧结果文档保持原样。

## 1. 设计范围与固定条件

显式行为版本为 `opportunity_executable_v3`；保留 `opportunity_cooperative_v2` 作为诊断与对照入口。两者采用相同的 lead 20 秒、heartbeat 5 秒、名义 K8 与局部最大 12 架约束。候选生成仍限 6 秒、CP 总额度仍为 8 秒，最多两次 scheduler 调用、一次有界扩展。`FULL_H` 保留完整可决策对象，并使用对应版本的候选构造方式。

本轮不引入学习、动态 K、lead 扫描，也不改变全局五维目标、downstream 尾部估计、几何与运动学、能源约束、pad 服务或应急规则。不以强制 MOVE 代替目标比较；搜索选择、交付通过与实际执行分别记录。

实现版本为 `4c41909`。本轮诊断绑定的源码摘要为 `2a5629bf756be0b48273b1f47523818370d0fa23f8a4f787f35b06e301a0ac76`，冻结源包含 1221 个文件，具体运行绑定见 [R4 v3 运行输入](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r4_boundary65_v3/resolved_run_binding.json)。

### 1.1 优先级一：普通选机与扩展必须能在本轮 effective time 调整

普通机会、公平席及协同队列只向本轮可开始新动作的节点分配搜索位置。判据读取 `_node` 和真实 active/soft 承诺链的末端时间，而非 source slot 的提前释放或 target capture 时间。链末端等于 effective time 时允许调整；仍在执行且末端晚于 effective time 的对象不作为本轮立即释放资源的解法。

这项限制不截断真实 hard urgency 及其 required、definite 依赖闭包。未被普通队列选择的对象仍进入完整固定背景、能源和下游评价；记录 `NOT_ADJUSTABLE_AT_EFFECTIVE`、节点时间、承诺末端与 effective time。普通资源阻塞扩展同时检查阻塞对象能否在本轮调整，以及其可调整节点是否对应需要释放的具体资源。近似关系只用于排序，不提升为 hard 依赖。

### 1.2 优先级二：先尝试不同近期目标，再细化同一目标

v3 从已有几何 screen 和 descriptor 中，为每个即时 target 提取首个可继续检查的描述。空闲或近期有限释放的目标优先；需要其他已选对象联合释放的目标仍保留，严格不可行的 committed 资源重叠继续由原有预筛处理。

首个目录按飞机交织，每架最多推进一个 target attempt。增长阶段先补到两个不同即时 target 的 attempt；已经有效复用的旧 target 计入覆盖。随后 reference、delayed/full-chain 与剩余 target coverage 公平交织，剩余新 target 优先于同一 target 的重复 progress/alternative。两个 target 沿用现有首轮目录的覆盖需求，不增加预算。`target_attempt_coverage` 分别记录 attempted、reused、viable 和 remaining unseen；有限时间内未访问的 target 不被宣称已经完成覆盖，更不被宣称不可行。

### 1.3 优先级三：交付阶段已知脏状态直接进入复验

`PREPARING` 阶段完成 guard 后，如果已经知道存在 `TRAFFIC_REVALIDATION_REQUIRED`，直接开始受影响交通复验，避免把已知脏状态继续作为待就绪包推进。复验仍受原 ready deadline 和公共在线服务计时约束。

这项修改针对已知脏状态的处理时序，并配套记录预算与阶段耗时。公共 admission、guard、复验和发布工作按实际发生计费；没有把公共工作移出预算，也不因修改了时序而推断历史逾期任务必然可交付。

### 1.4 优先级四：从可执行 pad 机会反推内环补给

near-term 小组以可在 effective time 出发的 pad 机会为 anchor，再沿具体被释放的内环、中环资源反向加入可立即动作的 M/O follower。每组给出 pad、anchor 启动可用性、预计到达及释放依赖；同一被释放资源只接一架 follower。排序首先比较预计到达时间，协同链接仅作同等近期机会的次序依据，因此不会仅因组较大而优先选择更晚的 pad 机会。

局部最多保留三个候选组。完整可选组放不下时，仍尝试该 anchor 的完整 definite 依赖闭包，不把真实依赖截掉，也不让组尺寸掩盖单独有价值的 anchor。所有上述到达与释放信息均为排序估计，不能替代单机、联合及交付认证。

## 2. 定向验证与诊断方法

服务器核心回归已通过，包括 effective time 前、等于及之后的承诺末端，未来 hard 节点及 required closure，普通不可释放对象不扩入，不同 pad 的 attempt 次序、旧目标复用、full/delayed 构造交织、I/M/O 具体释放链接，以及未选对象仍保留为背景。诊断还检查 UNKNOWN 原样保留、原 witness/区间不被补造、冲突去重与上限、同目录强制组合不增加正式 CP 调用。

| 检查范围 | 已观察结果 | 证据 |
| --- | --- | --- |
| 核心、选择、候选、安全说明与 latency 相关回归 | 123 通过，0 失败，0 跳过 | [core.xml](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/tests/core.xml) |
| v3 runner | 10 通过；该文件另含 9 项较早的报告测试 | [runner_report.xml](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/tests/runner_report.xml) |
| 最新报告与可读 replay 检查 | 15 通过，0 失败，0 跳过 | [report_final.xml](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/tests/report_final.xml) |
| 遥测 JSON 修复定向检查 | 24 通过，0 失败，0 跳过 | [telemetry_json_fix.xml](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/tests/telemetry_json_fix.xml) |
| 接续 runner 检查 | 13 通过，0 失败，0 跳过 | [continuation_runner.xml](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/tests/continuation_runner.xml) |
| 接续后的最新报告检查 | 19 通过，0 失败，0 跳过 | [continuation_report.xml](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/tests/continuation_report.xml) |

这些检查按范围列出。报告测试存在重复，不能将这些测试批次简单相加作为独立测试总数。

候选诊断在本次真实回调生成的目录中保存 candidate hash、轨迹与资源、五维分值、已选组合和原始冲突说明。强制组合 probe 仅替换指定候选，其余对象保持该次实际 selected；完整固定背景和同一全局目标参与评价。probe 不发布、不执行，不追加正式 CP 调用；额外验证耗时单列。无精确 witness 时保留原区间或 UNKNOWN。

历史 R4/R12 没有保存完整原始 callback 输入和候选目录。因此本次使用邻近边界的已保存物理状态与 selector history，清空 normal/candidate cache 后重新生成目录：R4 邻近边界为 source 65 → effective 85 秒，R12 为 210 → 230 秒。这是同状态的新诊断，**不是历史 R4/R12 的逐位重放**。证据中的 `historical_catalog_reconstruction=false`、cold-cache 标记、保存的 history 以及 physical unchanged 状态用于说明这一区别。参见 [R4 v2 诊断](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r4_boundary65_v2/actual_diagnostic.json) 与 [R12 v2 诊断](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r12_boundary210_v2/actual_diagnostic.json)。

执行中有两处命令路径拼写错误，分别在脚本路径和 source-freeze 路径检查时失败，均未开始物理运行；修正 setup 后继续原定任务，未为取胜而重复运行实验。原错误保留于 [sequence_cpu12.log](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/logs/sequence_cpu12.log) 和 [smoke_s3r23.log](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/logs/smoke_s3r23.log)。中负载 10 秒 smoke 的恢复运行确实为 COMPLETE，但遗漏外层 watchdog；仅将其 `full_job_wall_s=13.779860096983612` 秒单列为 `LOWER_BOUND_UNWRAPPED_SMOKE`，外部 setup wall 未知，不能据此声称整个总预算已完整计量。[smoke 结果](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/smokes/s3r23/evaluation/s3r23/RULE_LOCAL_K8/result.json)

正式两臂 launcher 已经主对话检查，配置内建 1200 秒/臂和本轮新增工作的 5400 秒 aggregate 上限；旧历史 ledger 不改。这些预算设置不补写遗漏的 setup 耗时，也不改变上述 smoke 的下界标记。

正式中负载臂在最近成功保存 60 秒边界后中断，进度记录正在推进至 65 秒，日志显示 `save_logs → canonical` 因 `inf` 无法写入严格 JSON 而失败。[中断日志](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/logs/arm_s3r23.log) 失败对象未落盘；已保存 60 秒 checkpoint 中的 scheduler 记录没有非有限 float，因此不能将某一潜在字段路径写成已证实的唯一根因。新增资源重叠遥测允许无穷区间末端，`cb707605` 的最小修复仅在输出副本中用 `{"$nonfinite_float":"+inf"}` 或 `-inf` 标记保留原意，NaN 仍拒绝；资源对象、返回认证结果、缓存数值与策略不变。

中负载随后从保存的 60 秒 checkpoint 接续至 300 秒，保留历史帧、selector history、pending wrapper/packet、references/templates/previous 与 world energy cache；未序列化的 provider validation cache 和进程全局缓存从空状态重建。因此该臂不是从零重跑，也不能声称重新产生了中断前相同的 wall 服务时序。原正式 launcher 的 FAILED 与原目录只读保留；最终报告显式将中负载结果入口改为 continuation 目录。[接续记录](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/continuations/s3r23_serialization_fix/continuation_finished_provenance.json)

失败 child 的实测 wall 为 65.544925 秒，接续 child 为 418.871810 秒，两者在新预算账本中分别计费。接续程序内部 job 为 418.385304 秒，与保存的前缀边界成本组合后记录 471.537634 秒；这一累计口径排除了未保存的失败尾段及其序列化开销，不能冒充全部消耗，也不能再与两个 child wall 相加。[预算账本](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/repair_budget_ledger.json)

本轮账本最终结算 child wall 为 1374.067592 秒（22.90 分钟），未超过 5400 秒上限，已无 running 或 budget stop。30 项记录包括 25 项 completed、2 项 failed、2 项已计量旧记录 credit 和 1 项上述下界 smoke。两项 failed 分别是中负载正式臂的 65.544925 秒中断，以及 source-freeze 路径拼写错误造成的 1.314528 秒 smoke setup 失败。这个结算含下界 credit，额外 setup 耗时仍未知，不能称为全部历史总计算量；旧历史账本未改。

## 3. 各问题状态的定向诊断

### 3.1 R4 邻近状态：可通过的 partial MOVE 仍可能使目标变差

在 `r4_boundary65_v2` 的新目录内，将 UID15/M02（`initial_middle_02`）单独替换为其 MOVE，batch 为 SAFE、prefix 为 ACCEPTED，但相对于该次实际 selected，全局风险项增加 2756.856333 Wh，预计能耗增加 10.0318424 MJ；F、L 和预计 touchdown 总项不变。该飞机自身的预计 touchdown 仍为 575.5 秒，407 飞机秒的预计 holding 由中环转到内环，其预计总能耗由 30.6029985 MJ 增为 40.6348409 MJ。[单候选 probe](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r4_boundary65_v2/probe_initial_middle_02.json)

UID14/M01（`initial_middle_01`）的单 MOVE 则与保持内环槽 1 的 M00 出现 `RESOURCE_CONFLICT`，因而不能作为独立可交付的替换。[资源冲突 probe](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r4_boundary65_v2/probe_initial_middle_01.json)

### 3.2 R12 邻近状态：等待位置改变与具体槽位竞争

在 `r12_boundary210_v2` 的新目录内，UID14/M01 的单 MOVE 同样为 SAFE/ACCEPTED，但全局风险项增加 3342.798778 Wh、预计能耗增加 12.0340756 MJ。该飞机的预计 touchdown 仍为 827.5 秒，514 飞机秒的预计等待由中环转到内环，预计总能耗由 38.6079895 MJ 增为 50.6420651 MJ。[单候选 probe](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r12_boundary210_v2/probe_initial_middle_01.json)

UID15/M02 的单 MOVE 与该次已选 O02（`initial_outer_02`）竞争内环槽 2，资源重叠从 250.556640625 秒延续至无穷。该 probe 的 batch 为 CONFLICT，prefix 未检查。[资源冲突 probe](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r12_boundary210_v2/probe_initial_middle_02.json)

这些观察仅说明：对本次新目录，保持部分 KEEP 有资源或既定目标上的理由，尚未构成 solver bug 的证据。partial MOVE 后的终端环等待评价可使向内移动代价升高，而不改变预计 touchdown。本轮保持目标和尾部估计，检验更完整候选与联合组合是否能改善这一结果，不通过强制 MOVE 消除 KEEP。

### 3.3 同状态 v3 目录的预测目标比较

五维目标按既有顺序写为 `[F, L, risk(Wh), Ehat(J), tau(s)]`，其中 L 保留实现中的负计数表示。以下均为目录预测值。

| 新诊断状态 | v2 实际选择的预测目标 | v3 实际选择的预测目标 |
| --- | --- | --- |
| R4 邻近：65 → 85 秒 | `[0, -7, 9931.492167, 913831382.3, 14796.5]` | `[0, -7, 7869.869944, 892135431.1, 14396.5]` |
| R12 邻近：210 → 230 秒 | `[0, -5, 15943.875667, 807436687.8, 14317.5]` | `[0, -5, 12983.88, 794017875.4, 14077.5]` |

两处 v3 目录均在 F、L 相同后获得更低的后续目标项。它们没有执行物理动作、没有发布，不能作为正常着陆数、等待或能耗的实际收益。具体 selected、全局背景与 CP 记录见 [R4 v3](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r4_boundary65_v3/actual_diagnostic.json) 和 [R12 v3](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r12_boundary210_v3/actual_diagnostic.json)。

这两处的时间表现存在权衡，并未显示普遍提速。R4 的实际 K 从 10 降为 8，扩展从 1 次降为 0 次，但首个非空认证更晚；R12 的首个非空认证更早，而完整 callback 仍略长。

| 新诊断状态 | 首个非空认证 wall（v2 → v3） | callback wall | candidate wall | geometry wall | prefix wall |
| --- | --- | --- | --- | --- | --- |
| R4：65 → 85 秒 | 4.238854 → 5.546833 s | 8.637765 → 9.027052 s | 1.908259 → 2.744571 s | 5.499670 → 4.579290 s | 0.849019 → 1.302379 s |
| R12：210 → 230 秒 | 8.208835 → 4.574737 s | 8.172575 → 9.047296 s | 2.848965 → 2.636105 s | 4.122700 → 4.432222 s | 0.428823 → 1.197795 s |

首个非空认证的计时从 planning origin 开始，包含 projection；callback 从回调入口计时。二者起点不同，不能简单相减推导认证之后的工作量。表中阶段耗时按原记录范围列出，也不额外求和来替代完整 callback 时间。上述时间与目录证据来自同一组 actual diagnostic 文件。

### 3.4 历史 R10：公共入场工作本身已超过剩余 ready 余量

历史 R10 的 ready deadline 为 186.004765745 秒。在 185 秒发生的公共 admission 工作耗时 1.032922729 秒，已超过当时剩余的 1.004765745 秒，随后记录 `READY_DEADLINE_MISSED`。因此，当前证据不足以将该次逾期归于无谓 heartbeat 等待或重复计费，也不能声称本轮 PREPARING 修改能够挽救历史 R10。具体时间见 [公共服务记录](D:/7_codespace/terminal/artifacts/rs_selective_repair_review_20261029/new_results/arms/s3r24/evaluation/s3r24/RULE_LOCAL_K8/online_services.json) 和 [normal task 记录](D:/7_codespace/terminal/artifacts/rs_selective_repair_review_20261029/new_results/arms/s3r24/evaluation/s3r24/RULE_LOCAL_K8/normal_tasks.json)。

### 3.5 R1：尝试的目标增多，但有效 pad 仍只有 P5

在 `r1_boundary0_v3` 中，UID5（`initial_inner_02`）的有效候选目标仍只有 P5。target coverage 记录尝试 P4、P5；P0 在 screen 后没有可继续的 descriptor，P1、P2、P3 尚未访问。因此当前只能确认探索次序有所变化，不能声称 P0/P1 已补齐，也不能声称所有 pad 已覆盖或均已证明不可行。[R1 v3 目录与覆盖记录](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r1_boundary0_v3/actual_diagnostic.json)

### 3.6 R8 邻近状态：消除非本轮可执行对象占位，但预测目标变差

在 source 130 → effective 150 秒的新诊断中，v2 将 UID25/O00（`initial_outer_00`）扩入并选为 MOVE，但其 first start 为 167.004765 秒，不能在本轮 effective time 启动。v3 读取其承诺末端 167.004765 秒，记录 access=false 并将其保留为固定背景，实际 K 从 10 降为 8，扩展从 1 次降为 0 次。

按本轮 effective time 提取首前缀，v2 只有 O10 的一条计划从 150 秒开始；v3 的 O05（M→I6）与 O10（O→M19）均从 150 秒开始，共两条。这里的首前缀属于未发布、未执行的新诊断结果。v3 修正了未来节点占用普通搜索位置的问题，但其预测目标由 v2 的 `[0, -7, 8853.8779, 845420922.9, 14423.5]` 变为 `[0, -6, 9290.5506, 862590313.4, 14577.5]`，包括预测 L 从 -7 变为 -6。该差异不能隐藏；v2 的预测值包含本轮尚不能执行的未来 final，实际质量仍须由正式 300 秒运行判断。[R8 v2](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r8_boundary130_v2/actual_diagnostic.json)、[R8 v3](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r8_boundary130_v3/actual_diagnostic.json)

### 3.7 R10 新邻近状态：与中环背景飞机的具体几何冲突

source 170 → effective 190 秒的 v3 新检查点给出了具体 witness：UID35/O10（`initial_outer_10`，当时位于中环槽 19）的 MOVE 候选，与 UID21/M08（`initial_middle_08`，中环槽 18、BACKGROUND）在绝对时刻 210.0 秒的距离为 77.78962840681909 米，距离上界 77.78962840681916 米仍小于 80 米。绝对检查区间为 `[190.0, 609.519580205131]`。该冲突对手是中环背景飞机；目标内环槽空闲并不能排除途中几何冲突。[原始 witness 与双方身份](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r10_boundary170_v3/actual_diagnostic.json)

对应 candidate hash 为 `2d58172d6ef2d6891c6c89a27e5ec3efb35efb5d9bd2ddf86dbda61e40acc0f6`，trajectory hash 为 `3775955e9113509b10915c3bd3edb31eed53a6fc8a0133ba4a802cd3209c21af`。这是新邻近检查点中的证据，不能冒充历史 source 167.004765 秒输入的精确 witness 或原始重放。

### 3.8 P2 机会：已识别 anchor，尚未形成 P2 首前缀

在 source 235 → effective 255 秒的新诊断中，v2 选择 O02（UID27，`initial_outer_02`）到 P2 的 final，但 first start 为 273.651658 秒；另一条 O01 的 M→I0 计划从 255 秒开始。v3 仍只产生 O01 这条立即计划。其机会小组已识别 O05/O10 到 P2 的 anchor，但最终没有可交付的 P2 final；两版本轮首前缀均为一条，不能宣称 P2 空闲问题已经解决。

这一状态的预测 L 同样从 v2 的 -6 变为 v3 的 -5。机会被识别、候选被选中与形成可交付 final 是不同环节；需要正式 300 秒的实际交付与着陆结果判断质量。[P2 v2](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/p2_boundary235_v2/actual_diagnostic.json)、[P2 v3](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/p2_boundary235_v3/actual_diagnostic.json)

## 4. 正式 300 秒对照结果

本节比较每种负载各一个开发样例的旧 v2 与新 v3 整版结果，公共端点均为 300 秒。中负载的新结果采用上述 continuation，原失败输出不覆盖。完整逐轮选择、每环能量/时间、超龄记录、回退、准入分母、证据时序和 P2 事件分别保存在 [完整摘要 JSON](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/v3_analysis_summary.json)、[可读报告](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/REPORT.md) 与 [逐轮 rounds.csv](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/rounds.csv)。这些是整版开发配对，未隔离各项修改的因果贡献。

### 4.1 实际着陆、能耗与等待

| 负载 | 正常着陆 v2 → v3 | 总能耗 MJ | holding 飞机秒 | 完整正常规划工作 s |
| --- | --- | --- | --- | --- |
| 中：s3r23 | 14 → 14 | 552.523207 → 560.596198 | 5385.848641 → 5410.136790 | 170.456709 → 186.908410 |
| 高：s3r24 | 15 → 12 | 799.097823 → 816.733533 | 8750.572076 → 8909.641108 | 165.746352 → 183.122021 |

四个单元的应急着陆、着陆前失败及未解决低 SOC 计数均为 0。中负载末态 airborne pending 为 26 → 26，高负载为 34 → 37；upstream pending 均为 0。没有观察到正常着陆和应急代价的整体改善。

| 负载 | 内环 holding 飞机秒 v2 → v3 | 中环 | 外环 | 每轮超龄 review 对象数：峰值/末轮 v2 → v3 |
| --- | --- | --- | --- | --- |
| 中 | 734.3161 → 975.8947 | 1408.2040 → 1329.2227 | 3243.3285 → 3105.0194 | 19/19 → 18/17 |
| 高 | 1034.0541 → 1265.3464 | 2992.3687 → 3070.9984 | 4724.1492 → 4573.2963 | 30/29 → 30/29 |

两种负载的内环 holding 均增加；相应内环 holding 能量，中负载由 68.1248 增至 90.5368 MJ，高负载由 95.9324 增至 117.3901 MJ。每环完整能量及末态环别计数见摘要，不把 inbound 强行归入某一 holding 环。超龄对象数按各轮记录的 review debt 统计，不等同于失败。

### 4.2 选择规模、交付与落地时刻

| 负载 | 任务数 v2 → v3 | committed 非空组 | join（已完成非空组） | NO_PREFIX | deadline 类拒绝 | actual K 分布 |
| --- | --- | --- | --- | --- | --- | --- |
| 中 | 21 → 16 | 12 → 10 | 12/13 → 10/13 | 7 → 2 | 1 → 3 | `{8:11,9:8,10:1,11:1}` → `{8:16}` |
| 高 | 16 → 17 | 12 → 13 | 12/13 → 13/13 | 2 → 3 | 1 → 0 | `{8:5,9:4,10:7}` → `{8:17}` |

四单元均留有 1 个 pending task，未计入已完成非空组的 join 分母。新中负载 join 为 76.92%，新高负载为 100%；任务、committed 和 join 不能替代实际着陆数。actual K 采用最终返回 `scope.actual_k`，不采用顶层初选名义值。中负载扩展轮数由 10 降为 0，高负载由 11 降为 0；两种版本在两根上记录的全量重规划回退（`full_fallback_rounds`）均为 0，overflow scope 也均为 0。

新中负载三个拒绝均为 `TRAFFIC_CHANGED_AFTER_DEADLINE`：snapshot 分别为 80、139.222680 和 243.979572 秒，其最后复验预算已为 0，排队公共服务分别为 0.593630、0.998693 和 0.667964 秒。这些任务的搜索工作为 8.377732、12.125119 和 11.082138 秒，并非搜索本身均耗满 19 秒；后期交通变化已无剩余时间复验，保留拒绝符合原 deadline。本轮没有通过延长 deadline 绕过它们。高负载旧版的一次 `READY_DEADLINE_MISSED` 在新结果中消失，但这一交付变化未带来更多 300 秒前的着陆。

中负载正常 FINAL 实际启动由 15 增至 17，落地数仍为 14；新版本在 283.1977 秒启动的三条 FINAL 都未在 300 秒前落地。高负载 FINAL 启动由 17 降为 16，300 秒前落地由 15 降为 12；新版本在 236.0927、263.8130、283.8130 秒启动的四条 FINAL 均未在端点前落地，223.5 秒之后没有 touchdown。这里观察到的是动作与终点之间的时间关系，不能由整版配对确定单一退化原因。中负载双方记录的 finish→next start 总等待均为 0。

### 4.3 三个主要阶段的总 wall 与每次观测均值

下表各单元写为“总秒 / 均秒”，n 为该版 callback 的观测次数。阶段采用已有的 exclusive timing，扣除嵌套子阶段；不能再加 dispatch 当作另一份工作。

| 负载与版本 | n | candidate generation / single vehicle | trajectory validation | prefix validation |
| --- | --- | --- | --- | --- |
| 中 v2 | 21 | 54.673637 / 2.603507 | 68.417484 / 3.257975 | 28.796149 / 1.371245 |
| 中 v3 | 16 | 58.697443 / 3.668590 | 56.254093 / 3.515881 | 40.998600 / 2.562412 |
| 高 v2 | 16 | 32.520041 / 2.032503 | 76.153970 / 4.759623 | 36.481043 / 2.280065 |
| 高 v3 | 17 | 51.944915 / 3.055583 | 72.426088 / 4.260358 | 37.543995 / 2.208470 |

候选阶段在两种负载下均更贵。中负载 trajectory 总量虽下降，单次均值与 prefix 均值仍上升；高负载的 trajectory 与 prefix 单次均值均没有变慢。因此不能将整体退化全部归于 CPU 或缓存无效。中负载接续重建未保存缓存是一项实际比较限制，也不能解释全部差异。CP、公共服务和其余阶段保留于完整报告，不在这里重复展开。

### 4.4 保存结果检查与研究判断

两根保存结果的自动检查均为 PASS，`full_a20_covered=true`，continuous separation 为 CERTIFIED，C2 为 PASS、PVA 为 CHECKED、resource、proposal nonexecution 和 independent replay 均为 PASS。中负载的检查针对 continuation 完整保存结果，高负载针对原臂。[中负载检查](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/continuations/s3r23_serialization_fix/evaluation/s3r23/RULE_LOCAL_K8/audit_result.json)、[高负载检查](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/arms/s3r24/evaluation/s3r24/RULE_LOCAL_K8/audit_result.json)、[后处理完成记录](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/postprocess_finished.json)

两臂 replay issues 均为空，任务与 callback 记录数一致；actual submitted、committed 事件与对应 maneuver started 的匹配数，中负载均为 31、高负载均为 33。六对新 v2/v3 snapshot 的 checkpoint SHA 与完整 timing 均为 MATCHED，但仍不是历史原始输入。人工播放状态保留 HUMAN_PENDING，自动检查通过不冒充已人工观看。

四项机制的实现与定向修复成立；R8 的普通不可执行对象占位被去除，交付 guard 与复验时序修正也保持原安全约束。但 R1 的有效 target 仍不完整，P2 邻近快照仍没有 P2 首前缀，R4/R12 的 partial MOVE 目标权衡没有因强制动作而被掩盖。两根的实际结果不支持整体优化收益，不能据此宣布优化成功、进入 B 或继续调权。当前结论限定在本轮整版开发配对。
