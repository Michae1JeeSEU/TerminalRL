# Selective repair v3：设计、验证与问题诊断（2026-10-29）

本轮研究在固定多环终端区内，改进局部重规划对象、候选覆盖和跨环补给的组织方式，并修正交付阶段已知交通变化的处理时序。研究问题是：在相同物理、安全、能源、pad 服务与全局目标下，有限的搜索工作能否产生更多可实际交付的正常动作，进而改善正常着陆效率与应急代价。

本文先记录实现与定向诊断。正式 300 秒对照结果尚未纳入，预测目标的改善不等于实际着陆收益。旧合同与旧结果文档保持原样。

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

这些检查按范围列出。报告测试存在重复，不能将三行简单相加作为独立测试总数。

候选诊断在本次真实回调生成的目录中保存 candidate hash、轨迹与资源、五维分值、已选组合和原始冲突说明。强制组合 probe 仅替换指定候选，其余对象保持该次实际 selected；完整固定背景和同一全局目标参与评价。probe 不发布、不执行，不追加正式 CP 调用；额外验证耗时单列。无精确 witness 时保留原区间或 UNKNOWN。

历史 R4/R12 没有保存完整原始 callback 输入和候选目录。因此本次使用邻近边界的已保存物理状态与 selector history，清空 normal/candidate cache 后重新生成目录：R4 邻近边界为 source 65 → effective 85 秒，R12 为 210 → 230 秒。这是同状态的新诊断，**不是历史 R4/R12 的逐位重放**。证据中的 `historical_catalog_reconstruction=false`、cold-cache 标记、保存的 history 以及 physical unchanged 状态用于说明这一区别。参见 [R4 v2 诊断](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r4_boundary65_v2/actual_diagnostic.json) 与 [R12 v2 诊断](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/snapshots/s3r24/r12_boundary210_v2/actual_diagnostic.json)。

执行中有两处命令路径拼写错误，分别在脚本路径和 source-freeze 路径检查时失败，均未开始物理运行；修正 setup 后继续原定任务，未为取胜而重复运行实验。原错误保留于 [sequence_cpu12.log](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/logs/sequence_cpu12.log) 和 [smoke_s3r23.log](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/logs/smoke_s3r23.log)。中负载 10 秒 smoke 的恢复运行确实为 COMPLETE，但遗漏外层 watchdog；仅将其 `full_job_wall_s=13.779860096983612` 秒单列为 `LOWER_BOUND_UNWRAPPED_SMOKE`，外部 setup wall 未知，不能据此声称整个总预算已完整计量。[smoke 结果](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/smokes/s3r23/evaluation/s3r23/RULE_LOCAL_K8/result.json)

正式两臂 launcher 已经主对话检查，配置内建 1200 秒/臂和本轮新增工作的 5400 秒 aggregate 上限；旧历史 ledger 不改。这些预算设置不补写遗漏的 setup 耗时，也不改变上述 smoke 的下界标记。

正式中负载臂在最近成功保存 60 秒边界后中断，进度记录正在推进至 65 秒，日志显示 `save_logs → canonical` 因 `inf` 无法写入严格 JSON 而失败。[中断日志](D:/7_codespace/training_outputs/terminal/rs_selective_repair_v3_20261029/logs/arm_s3r23.log) 失败对象未落盘；已保存 60 秒 checkpoint 中的 scheduler 记录没有非有限 float，因此不能将某一潜在字段路径写成已证实的唯一根因。新增资源重叠遥测允许无穷区间末端，本轮最小修复仅在其输出副本中用 `{"$nonfinite_float":"+inf"}` 或 `-inf` 标记保留原意，NaN 仍拒绝；资源对象、返回认证结果及缓存中的数值不变。该中断不记为完成的 300 秒结果，后续结果与续接记录另行补入。

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

待正式结果与交付、执行记录完成后填写。本节当前不填入运行指标，不将 smoke、邻近状态 probe 或预测目标差异作为正式对照结果。
