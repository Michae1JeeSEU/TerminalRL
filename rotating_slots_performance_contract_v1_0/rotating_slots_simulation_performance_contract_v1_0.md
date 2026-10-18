# 旋转槽位终端区仿真性能优化与验收合同

**合同编号：FT-RS-S1-PERF-1.0**  
**版本：1.0 · 正式实施规范**  
**制定日期：2026-10-18**  
**所属阶段：场景阶段1的性能资格补充，不是阶段2调度算法研究**  
**当前实施状态：NOT_STARTED；本文规定要求，不是实现完成或实验通过报告。**

> 核心原则：冻结研究问题，不冻结低效实现。优化后的代码应当更快地计算同一个问题，而不是通过少看飞机、减少候选、改变应急时机或放宽认证来获得速度。

本合同面向后续代码实施与验收。文中的“必须”“禁止”“通过条件”均为规范性要求；性能门槛是本次制定的工程验收目标，不是既有测量结果、理论加速保证或实机实时认证。

---

## 0. 执行摘要与文档效力

本次必须完成：版本与运行规则隔离；可恢复慢状态；低开销计时；重复轨迹/区间计算消除；等价缓存；剩余能量与峰值时间复用的资格验证；可关闭、可核算的单种子内部并行；固定输入的一致性对照；完整来流重跑；既有PyVista回放平台验收。

本次禁止开展：新beam/CP-SAT/MILP基线、学习训练、奖励调参、机型替换、能源模型重新标定、空域或准入规则再设计、BlueSky迁移，以及以改变物理步长或调度周期作为提速措施。

正式发布的优化实现必须通过“语义等价”和“性能达标”两个独立门槛。只通过其中之一，不得写为PERF_ACCEPTED；原场景未完成的验收，也不会因性能通过自动完成。

本合同与配套文件的关系：

| 文件 | 效力 |
|---|---|
| 本Markdown合同 | 人工可读的主规范 |
| `performance_acceptance_config_v1_0.yaml` | 机器可读的版本、比较规则、实验矩阵、门槛；不得修改物理配置 |
| `performance_acceptance_catalog_v1_0.json` | 70项新增性能/等价/恢复/回放测试要求，不是测试结果 |
| `source_index_v1_0.json` | 本次代码审查来源与固定Git对象 |
| `IMPLEMENTATION_TASK.md` | 给实施者的任务入口 |
| `PACKAGE_VALIDATION.json` | 仅检查交付文件格式、数量和相互一致性，不验证仿真代码 |

若主合同与配套配置不一致，程序必须报`CONTRACT_INCONSISTENT`并停止验收，不能静默选择宽松值。

## 1. 版本基线与规则隔离

### 1.1 三个版本必须分别标识

| 标识 | Git提交 | 用途 |
|---|---|---|
| H0历史实验内核 | `f67b5c8462433ba701191c493a37e4a8e472334e` | 原慢种子与历史60项结果的来源，仅用于历史解释 |
| H1上一轮审查快照 | `b936f6ef7b47156f9c3d13ee941580a5da902ebe` | 已增加独立的冻结内核应急参考并行脚本 |
| R0本次优化前参照 | `1c5afd37c891a0ed4776f64367fc7264db3c1c54` | 本次主要同状态与完整episode对照的语义基线 |

R0的有效策略标识为：

```text
upstream_capacity_only_pad_arrival_v1_20261018
```

R0相对旧合同已有两项单独记录的运行修订：暂时无法安全入场且未证明容量不足的请求留在上游等待；正常和应急进近都按下降开始至服务结束预约pad，可在服务结束前出发。其依据为仓库`docs/rotating_slots_stage1/amendment_20261018.md`，不是本性能合同新创设的规则。[S1][S2]

**本次固定R0，不把分支名当作不可变版本。**实施开始时获取HEAD、工作树改动、实际导入路径和文件hash。分支已前进时，在独立worktree/源码副本中保留R0对照；本合同不授权以新的行为变更替代R0，也不要求回退或覆盖用户当前工作树。

### 1.2 规范优先关系

物理与能源参数继承`new10.16/`冻结配置；运行语义采用冻结合同加R0修订记录。原合同与修订之间已明确的冲突按修订处理。其余行为以R0源码及可执行测试为等价参照，不在本次重新解释。

特别禁止：

- 用旧f67完整耗时与R0优化后耗时直接计算“性能加速倍数”；
- 将新规则减少应急、改变准入或改变停止时间带来的耗时差异归为数值内核加速；
- 用旧60项结果补齐新策略的完整traffic资格；
- 新代码运行时继续把`TERMINAL_CODE_COMMIT`伪装成f67；
- 修改`new10.16/`原始合同、物理YAML或94项目录，以绕过来源校验。

H0结果可进入历史附录，不进入主要配对速度统计。需要研究H0时，必须单独创建`policy_family=legacy_f67`组；它不能代替R0组。

### 1.3 语义指纹与实现指纹

每个结果必须同时保存：

```text
reference_commit / implementation_commit / actual_source_files_sha256
physical_config_bytes_sha256 / config_hash / vehicle_profile_hash / energy_profile_hash
policy_id / effective_rules_hash / candidate_policy_hash / numerical_policy_hash
execution_backend / profiler_mode / native_thread_limits / worker_allocation
manifest_sha256 / fixture_or_checkpoint_sha256 / command / imported_module_paths
```

物理文件hash并不覆盖代码中的运行修订、硬编码阈值、候选顺序及数值预算。因此必须另建语义指纹，覆盖这些字段。只核对YAML一致不构成等价性证明。

服务器仅是执行环境时，沿用项目现有本地修改、同步、服务器只读核hash流程；不得为获取commit而要求服务器擅自初始化Git或修改同步配置。新增优化模块、执行wrapper、测试入口和编译产物来源必须加入hash清单，不能继续只覆盖旧固定glob而漏掉真正执行的文件。[S9]

## 2. 源码事实与本次优化对象

以下为已核对的代码事实，不是尚未运行的性能收益预测。[S3]—[S10]

| 已读代码事实 | 影响 | 对应本合同工作 |
|---|---|---|
| `advance()`每轮物理/事件循环运行`_update_emergency()` | 高频监测内包含昂贵参考计算 | 保留监测时刻，优化所需计算的实现 |
| `_update_emergency()`先对参与监测的飞机求qE，再判断0.35与储备表达式 | 高SOCholding飞机也可能触发完整参考求值 | 不以SOC阈值直接删算；优先精确复用 |
| `_reference()`精确键包含ring、slot、time、version、profile | holding跨步通常不能命中完整参考；在途终点可固定 | 拆分形状/数值产物与状态绑定，不能简单删除time |
| 六pad模板实例化；失败再扫描20至300秒时长 | 有界枚举仍可叠加大量验证 | 保留有限候选策略及fallback，缓存等价产物 |
| `validate_plan()`同区间多次组装`polar_bounds()` | 区间对象和导数表达重复构造 | 共享一次区间jet，不改变舍入界 |
| 焦耳积分已有分段前缀表，而`_peak_measure()`仍递归查询 | “已缓存能耗”不等于整个能源查询便宜 | 同时审查峰值功率时间，不能只重复添加焦耳缓存 |
| `_check_execution()`在每对飞机内部调用`_base()` | 每区间可出现N(N−1)次基础轨迹构造 | 每架一次构造、配对共享 |
| 已有稳定同角速度间隔解析判定、旋转不变数值证书与向量化积分 | 不能把已有优化写成新增成果 | 量化现有命中与失效，再改粒度 |
| `DeterministicDriver`是验收驱动器 | 当前慢运行不是研究beam基线 | 冻结驱动行为，公共内核优化先行 |
| 已有独立单种子pad参考进程池，每任务仍分析与返回统计 | wall可能降低，CPU和通信仍需核算 | 适配R0、区分profile/benchmark、控制总进程资源 |

上一轮提出的“应急pad现在忙就提前跳过”，**不能直接用于R0**。R0允许出发时pad仍忙，只要求实际保护区间无冲突。此处只能针对已知候选的`[descent_start, service_end)`做正确的区间预检，不能重新引入旧`PAD_NOT_CURRENTLY_FREE`规则。[S2][S3]

旧benchmark的motion/power占比只作为热点线索；其冷启动、全程cProfile、多job并发测量不构成本次隔离计时的基线。旧14个traffic的数字不在本合同中重新声称已独立复算。

## 3. 必须保持的不变量

### 3.1 物理、安全与能源

必须保持R0有效规则下的全部原参数及含义，至少包括：三环几何与顺时针共同角速度；37/28/18槽位；四入口需求；六pad；正常相邻向内转移；应急允许各环直接final；15m捕获与13.5s固定垂降；C2连接；速度、加速度、jerk、下降及功率域限制；80m三维中心距离；25%正常接地储备；累计120s峰值功率预算；失败与截尾定义。

不得因性能修改能源模型、80轮诱导速度根算法的数值政策、积分最大分段、递归预算、残差容差、连续验证的UNKNOWN处理或ground/stable合法性检查。

对已开始机动，始终满足：

```text
approved_plan == committed_plan == executed_plan == replay_authoritative_plan
```

严禁用位置采样替代连续认证；保留80m精确边界，数值比较容差不能转化为空间安全余量。

### 3.2 时间与决策

正式参数保持`physics_dt=0.2s`、周期调度`5s`、默认预览`240s`、原精确事件切分和原机动承诺边界。0.1/0.05s只能作为原有收敛测试，不得作为本次新默认。

不得把应急监测降频、合并原有不同事件时刻、重新对齐浮点步网格，或因保存checkpoint额外调用`advance(t_checkpoint)`制造新的截断时间。

计时实验保持R0默认仿真延迟语义：wall-clock计算耗时仅被测量，不能自动写成`simulated_latency`。显式延迟测试保留原输入；进程池耗时变化不得引起不同物理迟到事件。

### 3.3 候选与验证语义

保持原目标slot/pad顺序、时长顺序、相位展开分支、首个可行返回规则、canonical分区与代表点、fallback规则、qE六pad比较、平局选择和无参考标志。

对同一请求，优化前后必须返回相同候选身份、轨迹payload、状态及最终提交结论。缓存和并行不得改变候选集合，也不得让任务先完成者获得优先权。

`NOT_FOUND`仍表示有限候选未找到；`UNKNOWN`仍表示当前数值预算不能证明；二者均不能变成物理不可行证明。

### 3.4 R0修订语义

保留上游FIFO、实际入场时SOC、等待代价、无强制上游超时、容量依据及以下守恒：

```text
offered = admitted + rerouted + upstream_waiting
```

不得把上游请求添加成物理飞机或计入终端区能量与成对间隔，也不得凭空扣除/补充模型外能源。保留请求属性、不重新抽gate/SOC/任务。

正常和应急pad保护区间保持从下降开始到服务结束；保持soft与committed区别，仅允许R0授权的重叠soft撤销。旧“等pad空闲后才飞”的deadline证明不得恢复。[S2]

## 4. 变更分类与发布边界

| 级别 | 内容 | 本次权限 |
|---|---|---|
| E0 纯等价实现 | 相同输入结果复用、循环外提、对象复用、相同运算次序的预计算 | 必做，测试通过后允许发布 |
| E1 数值等价实现 | 公式组织、前缀/区间树、向量化或编译执行改变末位舍入，但不改变离散行为 | 必须满足数值容差、保守证据和边界回退；逐项开关 |
| E2 策略/证明能力改变 | 改候选、qE近似、不同停止准则、旧UNKNOWN转SAFE、降低检查频率 | 不纳入本次正式实现；另立研究合同 |

严格等价不要求CPU计数、缓存命中、耗时或内部递归次数相同；它要求调度可观察行为与物理结果相同。不可用“都是零事故”“总落地数一样”替代逐项对照。

发现R0自身有安全/能源正确性缺陷时，不得为保留历史结果继续承认错误。必须提交独立`CORRECTNESS_BLOCKER`，暂停相关性能归因；正确性修复使用新语义版本和新oracle证据，不混入本次“等价加速”。

## 5. 实施阶段、顺序和退出条件

| 阶段 | 必须交付 | 进入下一阶段的条件 |
|---|---|---|
| P0 来源与行为冻结 | 源码/规则指纹、差异表、R0隔离运行入口、可执行oracle | 不存在未说明规则漂移 |
| P1 测量与恢复基础 | benchmark/profile双入口、被动计数、checkpoint、共同状态语料 | hook开关与中断恢复等价通过 |
| P2 低风险热点重构 | 每区间base复用、共享polar jet、常量与导数预计算 | 数值、候选、固定计划回放通过 |
| P3 能源与证书复用 | 稳定功率、分段证书、剩余能量、峰值时间及有界缓存 | 保守性与阈值对照通过 |
| P4 并行执行资格 | R0兼容进程池、固定归并顺序、完整CPU/内存核算 | 1/2/4/6 worker等价与故障测试通过 |
| P5 性能实验与完整回归 | 固定窗口配对、60s持续窗口、15场traffic、94项有效准则映射 | 自动等价与性能门槛同时通过 |
| P6 回放与发布 | 既有平台观察、审计报告、开关默认值、版本与交接 | 人工观察完成，未完成项不得提前写PASS |

每阶段遵循“代码→对应测试→测量→文档→版本检查点”，不得最后补写一份笼统成功报告。P2/P3修改逐项提交，可回退；P4不能替代P2/P3的单核资格。

## 6. P0：参照实现与输入语料

### 6.1 R0 oracle不得被优化实现替换

保留独立的R0源码环境，不与优化实现共享Python模块全局cache。禁止在同一进程中通过覆盖模块名使reference不知不觉导入新函数。

可增加只做观测、序列化和profiling开关的instrumentation shim，命名R0+OBS；其patch必须保存，且在小场景中证明开关前后状态、事件和候选相同。shim不得重排原函数、改参数或补充调度逻辑。

正式对照变体：

```text
R0_SERIAL_BENCH: R0语义，单执行进程，无cProfile，全量记录
R0_SERIAL_PROFILE: 同输入/语义，仅启用性能分析，用于归因
OPT_SERIAL_BENCH: 优化内核，单执行进程，无cProfile，全量记录
OPT_PARALLEL_BENCH: 同优化内核，父进程串行修改状态，内部参考进程池，无cProfile
```

已有f67并行恢复脚本继续服务其旧来源，不得直接解除来源检查后用于R0。R0并行适配必须有自己的版本、policy校验与来源清单。[S7][S8]

### 6.2 不让oracle成本再次阻塞每轮开发

小型候选/数值语料先运行；完整原种子无需在每次代码修改后从0重复运行。共同状态checkpoint只构造一次并hash冻结。

允许从已通过直接查询和短场景等价检查的R0语义并行适配获取前缀状态，但必须标记`prefix_execution=R0_PARALLEL`及证据；不能把新优化代码产生的状态伪装为旧服务器实际checkpoint。

只有states/replay而无内部状态时，可以重建“同物理状态查询fixture”，但不得标为可恢复checkpoint或连续episode；它只参加数值/提交单点测试，不代替完整状态窗口。

## 7. P1：计时、诊断和慢操作治理

### 7.1 双模式

`benchmark`模式关闭cProfile、逐函数分析和逐pad统计展开，保留全量权威状态/事件/轨迹归档及低成本计数。`profile`模式在固定短窗口或固定任务批启用，不默认分析整场5400s运行。

两种模式必须输入、输出语义相同；profile运行时间不能作为无干扰速度结果。至少对三个共同窗口执行profile on/off对照，单独报告仪器开销，不猜测倍率。

### 7.2 计时口径

必须分别提供：

- end-to-end wall：载入配置、冷cache/表构建、worker初始化、执行、正常checkpoint、记录、flush与ReplayBundle封装的累计wall；
- rollout wall：从共同状态恢复完成到窗口结束及对应记录flush；
- parent CPU：user+system或明确的进程CPU计时；
- process-tree CPU：父与所有子进程，包括初始化、IPC、结果序列化和统计；
- exclusive kernel CPU：只用于解释热点；
- cache warmup、编译、checkpoint加载、独立事后审计、可视化分别单列。

不得累加相互嵌套的`world.cpu`字段；`perf_counter()`字段不得标成CPU。并行completed-task CPU只是下界，不能替代完整进程树CPU。进程退出前必须收集其累计CPU；采样估算必须声明误差与遗漏。

每个计时窗记录：仿真起终时刻、实际推进量、空中/holding/在途/服务/上游数量、原决策次数、qE查询数、fallback数、候选逻辑数、验证逻辑数、真实计算次数、cache命中/淘汰、CPU/内存和记录字节数。

“逻辑验证请求数”与“实际执行数值验证次数”分开；缓存命中仍对应一个逻辑请求，不能在报告中把候选消失伪装成缓存收益。

### 7.3 低开销归因

耗时按调用来源和实际函数两个维度记录：execution、emergency-reference、actual-final-attempt、admission/freeflow、driver、archive。解释应急路径时包含其子调用成本；解释独占CPU时不重复累计父子。

启用低成本统计相比完全关闭统计的额外wall不得超过同输入中位数5%；不满足则减少统计频率/采用累积计数，但不得删权威仿真记录。该5%门槛不适用于专门profile模式。

### 7.4 卡慢、超时与中断

每30秒wall更新一次独立运行心跳，必须同时包含“最后完整仿真时刻”和当前活动任务上下文；不得为心跳推进仿真或制造调度事件。

连续300秒wall没有完成一个物理/事件步时写`SLOW_OPERATION`；连续1800秒无步完成时保存父/子堆栈、活动query/phase/pad/duration、cache统计和最新有效checkpoint索引。诊断观察不改变正在计算的候选结论。

外部任务时间预算到达时写`PERFORMANCE_TIMEOUT`，在安全边界保存或保留最近完整checkpoint；必要时终止整个任务进程树。它不是`ENERGY_FAILURE`、`NOT_FOUND`或`UNKNOWN`，不得纳入物理失败率或正常完成平均值。

不得未经明确运行命令停止用户现有长任务、覆盖原输出或修改Syncthing设置。实验使用新的attempt目录。

## 8. P1：真实checkpoint与恢复

### 8.1 保存内容

checkpoint至少覆盖：时间及原advance目标/外部调用阶段；当前循环执行位置；飞机全部字段；requests及上游状态与FIFO顺序；manifest内容、游标和随机生成器状态（若尚存在在线随机）；active/soft计划与插入顺序；slot/pad区间及committed状态；依赖图；source-clear时刻和已处理标识；下一心跳；初始settled状态；应急资格/声明/待处理/费用；SOC、累积焦耳和peak秒；正常final的权威预测SOC；pad使用累计；失败状态；driver可影响行为的状态；record/event/rejection流的偏移、同刻最后帧及hash。

还必须覆盖影响控制流的“看似日志”状态。例如R0用历史事件判断某个source-clear是否已经处理；不能因裁掉历史而在恢复后重复产生事件。可用等价处理集合替代扫描，但需要与旧扫描结果一致并纳入checkpoint。[S3]

### 8.2 原子、精确、可追溯

使用明确schema、临时文件+flush/fsync+原子rename，保存至少最近两个完整checkpoint。浮点必须可无损round-trip；无穷预约端点采用带类型标签表示，禁止将Infinity转换成null再猜测含义。不能直接沿用会把非有限值转null的报告序列化器作为checkpoint格式。[S9]

按每30个仿真秒或300个wall秒请求一次保存，**在原有循环安全切点完成**。不得额外切分物理步；不得重算位置、SOC或重新排序容器。执行中的worker结果必须归并完毕后保存，或者显式序列化为尚未产生副作用的待执行任务。

保存原advance调用边界，例如3600s到达窗和5400s总窗；恢复不能多调用一次应急监测、重复处理同刻请求，也不能错过尚未处理事件。checkpoint请求可以延后到安全切点，必须记录延后量，不能假装精确保存到请求时刻。

### 8.3 缓存与恢复

checkpoint按“语义必需状态”和“纯加速缓存”分层。canonical表即使可重建，也要保证代表点和候选策略不受历史影响。允许冷恢复或热恢复，但二者必须同结果；不得改变数值策略使cache命中决定SAFE/UNKNOWN。

恢复必须检查策略、物理配置、候选策略、数值政策和轨迹schema。实现hash不同但已登记为本次合格等价版本时才允许显式迁移；版本不明直接拒绝。来源不兼容时不得继续运行后补说明。

续写原流前验证完整前缀hash和偏移。半条JSON仅可在新恢复attempt中按checkpoint边界丢弃未提交尾部，旧证据文件只读保留。恢复不得重复记应急费、入场数、服务结束或失败事件。

### 8.4 性能核算

checkpoint构造/保存/恢复耗时单列；完整episode资格包含正常保存成本。多次恢复的一次逻辑实验，累计所有成功与废弃计算时间，并另报停机等待；不得通过重启把2小时预算重新归零。

## 9. P2：低风险内核重构

### 9.1 基础轨迹每区间每机一次

`_check_execution()`先构建按原飞机顺序的基础轨迹映射，再遍历原配对顺序。不得提前改变失败定位、minimum记录或异常处理。

计数验收：对N个合法检查对象，单次正常完整区间的基础轨迹构造不超过N；配对覆盖仍为N(N−1)/2。失败后二分定位新增查询单列。对象在本区间内不得被修改或跨状态版本错误复用。

同一稳定轨迹若继续采用当前start/end时间元数据，先只做区间内复用；跨物理步规范化必须单独验证不能改变plan hash、校验来源与回放引用。

### 9.2 共享polar jet

将同一个segment、同一[a,b]区间的r/theta/z及0—3阶导数包络作为只读jet。速度、加速度、jerk和功率从它派生；避免重复`polar_bounds()`与Interval构造。

保留外扩舍入与原有代数公式。预计算导数系数和逆时长幂时，优先保持原浮点运算次序；改变求值顺序的Horner/编译版本按E1验收，不得自动启用fastmath或float32。

### 9.3 常量和元数据

机型面积、profile常数、各环固定几何和稳定功率等可在profile作用域缓存。缓存必须包含所有依赖hash。固定字段变化时重新计算；禁止复用旧profile产物。

不可因immutable dataclass就取消入口完整性检查：外部反序列化、重hash伪造STABLE/GROUND、配置不符和非规范垂降仍必须拒绝。只能复用“已被验证的对象/形状证书”，不能把标签当作真实性证书。

## 10. P3：能量、峰值时间和证书复用

### 10.1 已有能力必须复用

当前`_segment_table()`已有焦耳积分前缀表，且去除水平绝对旋转常量；本次应扩展和审查它，而不是建立第二套能耗公式或互不一致的cache。[S5]

分段数值对象建议包含：

```text
shape_key
  ├─ immutable derivative coefficients / interval jet
  ├─ motion and power-domain certificate
  ├─ original-tolerance energy cells / prefix data
  └─ certified peak-threshold interval partition
binding
  ├─ exact aircraft/version/start/pad/slot metadata
  └─ authoritative trajectory payload and hash
```

形状证书不能代替状态与资源绑定。旋转不变性仅适用于无风、各向同性的运动/功率部分；全球坐标间隔与固定pad几何必须使用实际轨迹。

### 10.2 稳定盘旋和固定垂降

稳定功率按ring/profile计算，保留速度与向心加速度贡献；10秒holding参考可直接复用合格值。运行SOC仍按原查询区间累计，不得以周期近似替代。

固定垂降可以共用其三段的数值证书和积分产物；所有pad位置、实际时间和C2连接仍独立验证。相同数值模型允许复用不等于允许跳过整条final的前段检查。

### 10.3 已承诺轨迹的剩余能量

高频`energy(plan, now, end)`应复用同一合格分段表示。区间必须精确裁切，端点不足一个积分cell时局部按原精度积分；不能将now向0.2s或任意格点取整。

禁止用“完整能耗−一个无误差控制的近似已用能耗”作为剩余能量。必要时提供一致的累计表示及误差核算，保证接地储备与应急触发不因消减误差翻转。

### 10.4 峰值功率持续时间

`_peak_measure()`返回的是对超连续功率时长的保守估计，阈值交叉未知小区间的处理属于数值政策。[S5]

可缓存被认证低于/高于阈值的区间以及必须细分的节点。对子区间查询，仅重算边界节点及未解节点。不得直接相减两个“保守上界前缀”并声称差值仍为保守上界。

若新树的细分顺序不能保持R0的峰值保守性或E1容差，则该查询回到R0算法；若差异足以改变120s预算或动作选择，则优化分支不得发布。不能因为它更精确/更少保守就把新增可行动作归为同一问题的无损优化。

### 10.5 验证与安全证书缓存

精确计划证书和数值形状证书分开；受影响的配置、数值政策、轨迹版本、种类和区间必须进入key。纯几何/功率失败可在确切相同输入下缓存；交通冲突失败不得脱离world版本或相关轨迹/区间复用。

范围包含关系只允许在数学上正确时复用：例如已证明整个区间SAFE的证书可以支持其子区间安全，但其最小距离数值不能冒充子区间精确最小值。独立回放审计必须重新认证，不依赖运行时缓存宣告通过。

本次不以“更强证明器让旧UNKNOWN变SAFE”争取吞吐收益。发生此变化时归E2，退出本合同主发布分支。

### 10.6 缓存资源上限

精确缓存允许调整容量，但禁止无界增长。默认可重建缓存预算：单进程512MiB，每个内部参考worker256MiB；必须同时有entry上限、字节估算和真实内存测量。权威轨迹归档不应被误当cache删除。

canonical模板表保留原3×24定义，不因cache压力随机重选代表点；eviction只能导致重算，不能改变结果。profile/配置切换、对象篡改、旧checkpoint恢复和worker重建均要验证失效路径。

## 11. 应急参考：本次允许与不允许的优化

### 11.1 允许的主线

保留每个原监测时刻和qE语义；优先复用稳定holding能量、固定在途终点查询、固定下降段、相同形状证书、原精确query结果与纯pad候选并行。固定终点参考可采用pinning避免被通用FIFO挤出；pinning必须随机动完成/版本变化释放。

`_reference()`仍忽略其他交通/占用，为六pad有限候选能量参考；它不是即时安全救援保证。不能根据pad忙碌状态改其候选集合。

### 11.2 相位缓存边界

holding飞机相对地面pad的相位随时间变化。禁止直接删除time、四舍五入time、使用phase bin值替代精确qE，或复用上一时刻参考计划而不重新验证其当前来源与形状。

纯数值形状已完全相同的产物可以复用；对称/旋转映射必须保留实际metadata、端点和固定pad关系，不把“环共同角速度”误用于“地面pad也旋转”。

### 11.3 惰性触发判定不作为本次完成前置

仅凭高SOC跳过参考计算、学习预测qE、插值参考值或改变刷新周期均不允许进入本次正式后端。带证明的惰性算法可记录为后续设计，不实现为默认分支，也不以其潜在收益满足本次门槛。

原因：即使只想证明“不触发”，也必须用针对当前有限参考策略的有效上界；而精确`q_e`、`reference_missing`与事件仍是可观察状态。不能只保持布尔alert却静默改为过期遥测。[S3]

## 12. 资源预检的特殊边界

R0允许pad尚忙时出发，因此不能在生成轨迹前仅看`pad.free_now`跳过。已知候选的下降开始和服务结束后，可复用与`submit()`完全相同的区间资源判定；未来committed依赖、soft重叠和可撤销范围必须保留。

资源预检若改变了“原本首先报告的动力学/峰值/SOC拒绝原因”，会改变rejection与`final_attempt_rejected`可视化事件。主发布要求保留既有可观察拒绝语义；若无法便宜地保留，则不启用这个预检优化，优先完成其他热点。

允许记录独立的`performance_trace.early_rejection_proof`，但不能伪造未执行数值验证的witness，也不能删去用户用于解释失败的候选记录。逻辑候选序列与最终选择不得改变。

## 13. P4：单种子内部并行

### 13.1 并行范围

只允许对同一不可变快照下相互独立的参考pad候选、已定义的纯数值任务并行。父进程独占时间、飞机、应急状态、资源日历、提交、取消、事件和需求揭示。

任务key携带精确query、pad、source version、各语义hash。结果按原query及pad顺序归并，再按原平局规则选择；不得使用“第一个完成”“第一个可行”取代原逻辑。

现有f67 wrapper只作为实现参考。R0适配必须从当前读取的world构建，不能隐式调用旧冻结policy；不得移除源校验强行兼容。[S7][S8]

### 13.2 数量与资源

正式无GPU、CPU参考环境下，默认选择单核优化后端。P4测量1/2/4/6个内部worker，单独保留无进程池的纯serial对照；`inner_workers=1`不是无IPC的serial。

获得资格后，可自动从已测变体中选择满足全部绝对门槛且配对窗口几何平均wall最小者。wall在5%以内时选择完整CPU更低者，再选择worker更少者。该规则只基于冻结性能语料；不得按运行中SOC或未来交通结果换后端。

单个正式job最多1父+6子，单个数值库线程=1。CPU亲和性按真实物理core/性能等级记录，避免R0放效率核而OPT放性能核。多job并发时必须满足总预算，严禁20个独立job各再开6子而不做资源规划。

### 13.3 异常、生命周期与IPC

worker exception、进程死亡、任务超时、错误hash或序列化失败必须传播为实现/运行故障，不能转成物理不可行或悄悄换较小候选集。取消/退出必须join或由supervisor清理所有子进程。

benchmark模式不对每个pad创建cProfile对象，不返回完整逐函数统计；保留结果、有限计数和必要审计身份。profile模式另行启用。worker传回的大对象大小、序列化与等待时间、重复cache预热成本单列。

主报告必须包含完整进程树CPU与峰值同时RSS；不能把各进程生命周期峰值相加后称为某一时刻真实总内存，也不能只统计父进程。

## 14. 数值与行为等价标准

### 14.1 必须精确相同的离散字段

manifest全部属性；策略与参数；候选遍历及选择身份；批准/拒绝状态与主要reason；事件类型、飞机、slot/pad、依赖及顺序；upstream/FIFO结果；应急声明与一次性收费；着陆、分流、失败及截尾身份；soft/committed状态；真假与缺失标志；权威轨迹payload和hash。

增加性能元数据应在sidecar或明确的performance namespace，不插入原物理事件流。checkpoint事件也不改变原`decision_times`或source-clear去重逻辑。

### 14.2 连续量比较容差

仅对同一环境和同一权威输入的E1数值结果，允许以下最大比较差异：

| 量 | 比较门槛 |
|---|---|
| 位置每分量 | 绝对1e-7 m |
| 速度、加速度、jerk每分量 | 绝对1e-8（各自SI单位） |
| 非失败事件时刻 | 绝对1e-8 s，顺序仍须精确一致 |
| 数值定位的首次物理失败时刻 | 绝对1e-5 s，失败飞机/原因/先后必须一致 |
| 电池与轴功率 | `abs_diff <= max(1e-5 W, 1e-10*abs(reference))` |
| 每次查询及每机累计能量 | `abs_diff <= max(1e-3 J, 1e-10*abs(reference))` |
| 每机每步SOC、接地SOC、qE | 绝对1e-10 |
| 峰值功率累计时间 | 绝对1e-7 s，且不减弱原保守保证 |
| 系统累计浮点指标 | 应由逐机/逐事件容差推导并同时报告绝对差，不以大相对容差遮盖局部错误 |

这些是新制定的回归比较门槛，不是安全放松量、物理模型误差或厂家精度。原配置中更严格的数值要求仍保留。单位、NaN/Infinity和None必须显式处理，禁止通用round后比较。

### 14.3 决策边界优先

在80m、25%接地储备、0.35、qE表达式、SOC=0、功率域/上限、120s peak预算附近，即使数值差落在表中，只要批准、触发、失败或平局选择改变，就不通过。

无法证明新快路径给出同一判定时，当前查询回到冻结参考算法；回退时间计入性能。不可使用“差别很小”允许离散结果翻转。

SAFE/UNSAFE/UNKNOWN三态逐项比较。数值证书内部区间数、方法标签、上下界宽度可不同，但都必须保守且覆盖原查询区间；不能把更宽或仅采样得到的界称为同等证据。独立认证仍要验证实际执行轨迹。

### 14.4 回归层级

L0：同一原函数重复调用与cache on/off。  
L1：函数、候选、区间查询对照。  
L2：固定已批准动作的执行回放。  
L3：同checkpoint完整验收驱动、含事件和上游队列。  
L4：从0开始的15个原来流manifest完整运行或同原因物理失败停止。  
L5：独立审计与既有平台人工观察。

不得只通过L1就宣布L4等价；不得用优化后实现生成全部expected再让它自己通过。

## 15. 冻结测试语料与实验矩阵

### 15.1 数值/候选语料

使用固定语料seed`20261018`，并保存最终显式输入而不仅保存seed。至少包括：三环所有83个稳定槽位；六pad完整垂降；入场、O→M、M→I、正常final和三种来源应急final；phase wrap与24分区边界的两侧；非0.2s网格事件；起终时刻及内部随机查询区间；明显不可行与接近约束的候选。

正常数值语料每种机动至少20条不同形状（不能满足时给出generation evidence，不虚构）；其中成功与拒绝案例分别保存。新增负向例不改变正常来流分布。

峰值时间必须覆盖全低于阈值、全高于阈值、单次穿越、多次穿越、切触/窄区间及跨segment查询。边界输入以原公式构造并保存原结果；不能要求真实机动恰好覆盖所有抽象根形态，必要时用明确标注的函数级fixture。

### 15.2 六个真实共同窗口

主要同状态语料来源固定为：

| 窗口ID | R0来流 | 首选checkpoint请求时刻 |
|---|---|---:|
| W71-A | rate180, seed71 | 300s |
| W71-B | rate180, seed71 | 900s |
| W71-C | rate180, seed71 | 1200s |
| W60 | rate60, seed23 | 600s |
| W120 | rate120, seed11 | 600s |
| W180 | rate180, seed11 | 540s |

取R0实际自然切点，不额外切步。每个checkpoint来源和实际时刻必须保存。首选时刻在终止后或其后不足60s有效执行时，采用**最后一个距真实停止至少60s的已保存自然切点**；同seed重复则向前取最近未用、间隔至少30s的切点。该替代只因可用性，不看哪个窗口更快；记录原目标、停止状态与替代原因。

若整场不足60s或无法取得足够不同checkpoint，窗口记`UNAVAILABLE`，不得以空场或别的容易seed代替。补证前总体性能资格保持未完成。历史H0相同seed的checkpoint不得混入R0语料。

采集语料期间，首选里程碑checkpoint必须pin保存；除正常最近两个恢复点外，额外保留最近八个30s请求所产生的自然切点，以便停止前的确定性替代。语料确定后固定保留六个输入checkpoint，不能被常规淘汰删除。

语料在P2优化调参之前冻结hash。每个窗口比较：R0_SERIAL_BENCH与OPT_SERIAL_BENCH各3次，从同checkpoint推进10s，顺序AB/BA/AB；选定执行后端另从同checkpoint推进60s共3次。

checkpoint加载和cache暖机不混入热窗口统计，但必须单列；主同状态配对采用**相同checkpoint、空派生cache起点**，从恢复完毕开始计时，窗口内部产生的预热不能排除。额外热cache结果仅作次表，明确相同逻辑预热序列，不能预算未来答案。

R0单次10s窗口诊断wall上限7200s。达到上限记性能TIMEOUT，保留统计下界，不把截断时间当完成时间计算加速。主要相对门槛须六个窗口各三个完整有效配对；不足则未通过，不缩短窗口规避慢例。

### 15.3 结构规模测试

分别运行N=10/30/60/83稳定占位场景，保留正常应急监测、连续安全和完整记录，各推进30s、3次。另设含真实在途机动的mixed fixture，记录实际空中数而不是只报槽位数。

“固定动作执行微基准”与“完整world benchmark”分开；前者只用于定位执行成本，不能通过关闭应急或驱动后冒充完整world吞吐。

原1s冷fixture可继续报告，但不能作为唯一资格依据。初始化时间、冷模板、warm steady-state及归档必须分列。

### 15.4 完整traffic与周期回归

正式来流矩阵：rate`60/120/180` × seed`11/23/47/71/101`，共15项；到达窗口3600s、drain1800s，使用R0策略及5s周期，不重抽manifest。

先对`rate60_seed23`、`rate120_seed11`、`rate180_seed71`完成R0与优化版本的从0同语义对照，再运行其余项。优化前完整证据可分段checkpoint续算，但每段必须同源可追溯，不能用独立反事实片段拼成完整episode。

完整来流的语义对照可使用通过等价资格的R0并行执行来降低一次性参照成本；其时间不进入单核加速主表。15项全部必须有R0同policy oracle或逐段等价链，缺少的不能由旧f67结果补齐。

所有实际物理失败必须与参照同原因、同飞机、同事件顺序。失败是有效的运行结局，不要求性能重构降低失败率；调度质量改善属于后续阶段。

保留原dt=0.05/0.1/0.2脚本收敛和周期1/5/10的测试：比较每个设置“优化前后相同”，不是要求三个调度周期产生同样行为。正式默认不改变。

## 16. 性能通过门槛

以下是本合同新设的工程资格门槛，不是对未实现优化的收益承诺。任何门槛未达成，保持`NOT_QUALIFIED`，不得降低标准后继续沿用同一版本。

### 16.1 必须有单核增量

对六个10s共同窗口，设第i个窗口R0与OPT单核3次完整测量的中位wall分别为B_i、O_i：

\[
S_{serial}=\exp\left(\frac1{6}\sum_i\log(B_i/O_i)\right).
\]

通过条件：`S_serial >= 2.0`，且每窗口`O_i <= max(1.10*B_i, B_i+0.05s)`。另报CPU加速，不允许把关闭profiler的收益计入此比值，因为两边都必须关闭。

单核门槛避免“只加机器资源但重复计算原封不动”。它不是理论最优要求，未达成时按热点继续优化，不改变场景。

### 16.2 可用后端的绝对门槛

选定后端必须同时满足：

| 项目 | 通过条件 |
|---|---|
| 六个60s持续窗口 | 各窗中位wall/仿真秒 ≤10；六窗该比值的几何均值 ≤5 |
| 完整15项traffic | 每项从启动到ReplayBundle完整落盘的累计有效wall ≤7200s；进程树最多1父+6子 |
| 进程树峰值RSS | 纯serial≤2GiB；最多6子并行≤8GiB |
| 单核回归劣化 | 符合16.1逐窗条件 |
| 并行完整CPU | 对同完成60s窗口，进程树CPU中位数 ≤OPT_SERIAL同窗的1.5倍 |
| 完整记录 | 输出无缺帧、无事件丢失，主性能不能建立在关闭全量记录上 |
| 错误/超时 | 0个被掩盖或误记为物理结果的实现故障/超时 |

15项原有物理失败截尾不等于性能未完成，但必须比较到相同真实停止点；不能通过提前物理失败降低完整耗时。末端仍有上游/空中未解状态按原规则记录，不能为了“清空”延长或缩短5400s边界。

独立事后安全审计耗时单列，不藏进其他费用或删除；它必须完成才获得资格，但不以图形渲染速度影响模拟内核吞吐。

### 16.3 统计与公平性

记录每次原值、median、nearest-rank P95、最大值、配对比率和完整资源设置。3次窗口的P95就是最大值，不假装精确估计总体尾部。事件级P95/P99仅在样本数足够时报告样本量；不足时报告最大值。

主比较为同机、同依赖、同CPU性能等级和同记录策略的隔离运行；并发矩阵结果只能另表报告。三次重复的同一变体wall最大/最小超过1.25时整组标`NOISY`，检查外部负载后最多重做整组一次；原数据保留。仍NOISY则不能用于通过门槛，不能挑最快的重复。

资源开销从第一次启动算起；提前构表、worker预热、JIT/编译不能免费隐藏。可另报摊销曲线，但冷端到端门槛仍保留。

## 17. 原94项目录与新增70项测试

原目录保留不动，按R0有效准则建立mapping：原ID、原要求hash、是否受Oct18修订影响、有效准则文本/hash、实际测试函数、证据路径、结果。不得因老expected与新规则冲突把正确新行为判为错误，也不得静默修改原expected。

本合同附录A及JSON定义70项新增要求。它们与原94项可能覆盖同一行为，**不能简单相加后宣称“164个独立测试全部通过”**。每项必须明确断言、输入、oracle、日志和范围；目录枚举、测试代码存在都不等于运行通过。

状态统一为`NOT_RUN / PASS / FAIL / BLOCKED / PERFORMANCE_TIMEOUT / HUMAN_PENDING`；某项超时或缺证不能写PASS。对故意注入故障的负向测试，正确报告故障是该测试的PASS，不是场景没有故障。

## 18. 既有visualization平台验收

必须复用现有`RotatingSlotsReplayViewer`/`Terminal3DViewer`及项目真实启动入口，不新建替代viewer、不用几张截图代替交互验收。[S10]

提供reference与optimized成对ReplayBundle及同时间轴定位方式。界面展示真实三维位置、同一权威轨迹、当前/未来slot和pad预约、SOC/功率/peak、应急源环、上游队列与等待、服务倒计时。沿用R0对未来committed pad为RESERVED的显示修订，不把地面暂时无人误显示为无预约。[S2]

最低人工观察集：正常入场与跨环；有上游等待的准入；服务未结束前起飞且下降区间不冲突；外环应急经过内部空域；繁忙资源下应急等待及失败；checkpoint续跑接缝；与原始候选拒绝对应的失败解释；同一对飞机80m附近连续距离证据。

可视化只读归档，不重新生成轨迹、不在seek时推进world，也不平滑掉轨迹或时间差异。服务器benchmark关闭viewer；本地观察耗时不计入模拟吞吐。

人工记录包含observer、观察时间、代码/配置/包hash、查看时刻/事件、结论与异常。未实际观察保留`HUMAN_PENDING`。自动测试与独立审计通过不自动替代此签收。

## 19. 完成标准、发布与阶段2接口

状态依次为：

```text
SPEC_READY → IMPLEMENTING → AUTO_EQUIVALENCE_PASSED
→ PERFORMANCE_QUALIFIED → VISUAL_ACCEPTED → PERF_ACCEPTED
```

总体`PERF_ACCEPTED`必须满足：P0—P6必需产物齐全；70项新增要求通过；原94项有R0有效映射并按适用准则完成；所有数值/离散边界等价；六窗口及15traffic门槛通过；无隐藏失败/缺证；既有平台人工观察完成。

本合同不授权自动启动阶段2。交接报告应写出已经合格的默认后端、公共API、实际速度与CPU/内存、仍然占主要成本的函数、缓存/预热策略和已知建模边界，以便阶段2在共同内核上开发。

若只完成自动等价但性能不足，写`AUTO_EQUIVALENCE_PASSED; PERFORMANCE_NOT_QUALIFIED`，继续优化公共内核；若性能达到但存在语义差异，写`EQUIVALENCE_FAILED`并回退相关开关。原场景本身未获得STAGE1_ACCEPTED时，不因本合同完成而自动补发。

## 20. 交付目录与可执行入口要求

建议在仓库新增文档目录`docs/rotating_slots_stage1/performance_v1/`，实施代码优先原模块重构与独立性能工具目录，不复制第二套业务world。reference由固定worktree或只读原版本包提供。

结果目录按项目已有服务器输出根生成：

```text
<OUTPUT_ROOT>/rotating_slots_perf_v1/<attempt_id>/
  provenance/          # reference、optimized、规则和依赖hash
  inputs/              # manifests、候选语料、共同checkpoint索引
  checkpoints/         # 原子检查点，带schema、偏移和hash
  equivalence/         # 函数、候选、状态、事件逐项差异
  profiles/            # 有界profile原文件和调用归因
  benchmarks/          # 每次配对原值，不只汇总
  traffic/             # 15项各自result与ReplayBundle
  audits/              # 独立连续安全/能源审计
  visual/              # 成对回放索引和真实人工记录
  summary/             # gate结果、缺证、性能报告、下一步交接
```

实施必须提供并测试CLI，至少支持`preflight / capture / micro / paired / parallel / regression / traffic / audit / summarize`；支持输出目录、reference位置、配置、checkpoint、显式worker预算及profile开关。**这些是待实现的接口要求，本规范不声称它们现在已存在。**最终交付必须给真实可复制命令，不给虚构的可运行占位命令。

每个运行结果保存attempt、实际完整命令、退出码、起止时刻、有效计算时间、恢复次数、停止原因和产物hash。旧数据不覆写，不从历史截图恢复数值，不根据输出文件存在就判完成。

## 21. 来源说明

[S1] GitHub提交`1c5afd37c891a0ed4776f64367fc7264db3c1c54`：`fix: align emergency pad arrivals and queue unsafe entries`。  
[S2] 该提交的`docs/rotating_slots_stage1/amendment_20261018.md`：有效策略、上游队列与pad到达口径、旧证据隔离。  
[S3] 该提交的`terminal_marl/core/rotating_slots/world.py`：`advance`、`_check_execution`、`_reference`、`_update_emergency`、`_try_final`、`submit`及状态/资源链。  
[S4] `trajectory.py`：Segment与极坐标包络、固定垂降、候选顺序。  
[S5] `energy.py`：根、功率、分段前缀表、`_peak_measure`。  
[S6] `safety.py`：连续证书、数值cache与稳定结构证明。  
[S7] `scripts/server/parallel_rotating_slots_runtime.py`（H1引入）：纯pad任务、顺序归并、prefetch与统计范围。  
[S8] `scripts/server/recover_rotating_slots_parallel.py`（H1引入）：旧f67恢复入口和parent/child计时范围。  
[S9] R0的`qualification.py`：来源收集与报告序列化；`diagnostics.py`和`driver.py`：当前验收驱动与分析器。  
[S10] R0的`docs/rotating_slots_stage1/README.md`及回放实现；历史`diagnostics_20261016.md`仅用于解释已发生的测量条件。

完整固定链接及已核对Git blob列于`source_index_v1_0.json`。上述来源支撑代码事实和已存在规则；本合同的2倍、5/10秒计算比、2小时、内存及误差回归门槛是新制定的工程规范，不能归于这些来源的实测结论。

---

# 附录A：新增性能验收目录

下表由同包JSON生成；所有条目的初始状态为NOT_RUN，实际执行后另存结果，不修改规范目录。

## A. 来源与语义隔离（P0）

| ID | 测试情形 | 预期行为及最低证据 |
|---|---|---|
| A01 · 参照版本隔离 | 同时可访问R0、优化分支及历史f67输出。 | 主对照实际导入R0，H0不进入配对速度分母；reference不共享优化模块全局cache。 **证据：**实际import路径、commit、文件hash及负向混入测试。 |
| A02 · Oct18规则指纹 | 输入相同物理YAML但分别启用旧准入规则与R0上游规则。 | effective_rules/policy指纹区分两者；禁止仅因config_hash相同认为等价。 **证据：**来源差异表和误配拒绝日志。 |
| A03 · 实际执行文件覆盖 | 新增优化helper、worker、wrapper或编译入口。 | 所有实际执行实现进入来源hash；修改任一文件使实现身份变化。 **证据：**source_files清单、篡改后校验记录。 |
| A04 · 物理与数值政策冻结 | 故意改80m、0.2s、0.35、peak预算、根迭代或候选时长。 | preflight拒绝语义漂移，不以新数值完成性能验收。 **证据：**各字段负向测试。 |
| A05 · 历史产物只读 | 使用旧60项结果与未完成seed目录作为输入资料。 | 旧bytes/hash不变；不补写旧PASS，不覆盖原attempt。 **证据：**前后hash清单。 |
| A06 · 服务器来源核验 | 服务器无Git且显式提供commit标签；标签与文件hash不匹配。 | 不能仅信任环境变量；实际文件不符则拒绝正式运行。 **证据：**部署清单和误标commit测试。 |
| A07 · 观测shim零副作用 | 对同一小型world启用/关闭被动checkpoint与计数hook。 | 状态、事件、候选、decision_times和轨迹hash一致；没有额外advance切点。 **证据：**R0/R0+OBS对照。 |
| A08 · 语义修复不冒充优化 | 注入一个会把旧UNKNOWN变SAFE或改变上游结果的修改。 | 归类E2/CORRECTNESS_BLOCKER并阻断等价发布，不能只依据无事故通过。 **证据：**分类器及gate负向测试。 |

## B. 轨迹、运动与连续安全内核（P2）

| ID | 测试情形 | 预期行为及最低证据 |
|---|---|---|
| B01 · 每区间每机一次base | N=10/30/60/83稳定或混合对象的一次完整无失败检查区间。 | 基础轨迹构造不超过N；复用后结果不变；失败定位新增调用单列。 **证据：**函数计数和原实现输出。 |
| B02 · 成对检查覆盖 | 同上，并增加两架在途、地面服务对象。 | 原应检查的N(N−1)/2配对均被覆盖；顺序和首个失败处理一致。 **证据：**pair审计清单。 |
| B03 · polar jet共享 | 对同一个segment、区间的运动与功率检查。 | 同一检查节点共享一次极坐标jet；导出的0—3阶包络与原有效包络一致且保守。 **证据：**构造计数与包络对照。 |
| B04 · 导数与状态求值 | 任意非网格时刻、端点、长final高阶z及短segment。 | p/v/a/jerk满足第14节；权威payload/hash不变。 **证据：**逐字段最大误差及最坏输入。 |
| B05 · 区间外扩有效 | 解析可解多项式、随机输入、近端点极值。 | 保留外扩舍入及包络覆盖；有限采样只作反例筛查，不代替保守性论证。 **证据：**解析/区间证明与差分测试。 |
| B06 · 篡改与伪造标签 | 修改已缓存plan、重hash伪STABLE/GROUND、非规范DESCENT或错profile。 | 所有入口仍拒绝；cache不能绕过真实性与结构检查。 **证据：**反序列化和cache命中负向日志。 |
| B07 · C2与metadata绑定 | 复用形状但改变aircraft/version/start/slot/pad。 | 独立检查实际源目标、连接和身份；不能借旧证书接受不匹配。 **证据：**metadata边界样例。 |
| B08 · 80m唯一硬界 | 79.999999、80、80.000001m可解析间隔与稳定跨环对象。 | 同R0判定；不加82m，不接受不足80m，不把回归容差当安全余量。 **证据：**解析距离与认证结果。 |
| B09 · 采样间冲突 | 两个执行采样时刻安全但中间进入80m的连续轨迹。 | 连续认证发现；不能通过减少pair或粗采样漏检。 **证据：**连续见证与原认证器对照。 |
| B10 · 候选序列不变 | 覆盖所有机动类型、57个final时长和相位分支。 | 候选逻辑顺序、首个可行、selected hash及NOT_FOUND状态同R0。 **证据：**完整候选trace/hash。 |

## C. 能量、功率与峰值时间（P3）

| ID | 测试情形 | 预期行为及最低证据 |
|---|---|---|
| C01 · 稳定功率仍含转弯 | 三环全部83槽位、多个绝对时刻和查询长度。 | 复用值与原v/a功率符合容差；向心加速度没有归零。 **证据：**每环功率、SOC及源公式对照。 |
| C02 · 固定垂降数值复用 | 六pad、多起始时间的三段13.5s垂降。 | 焦耳、peak、domain与原值一致；位置、时间及连接仍实际绑定。 **证据：**六pad数值/metadata测试。 |
| C03 · 任意区间能量 | 跨cell、跨segment、长度小于0.2s和非网格边界。 | 积分满足原精度和第14节；不把查询端点取整。 **证据：**显式区间与误差表。 |
| C04 · 剩余能量一致 | 同一已承诺轨迹每步查询now→end并累计执行耗能。 | 剩余/已用/完整能量相容；储备、qE和事件不因前缀消减误差翻转。 **证据：**固定计划逐步对照。 |
| C05 · profile与配置失效 | 在独立负向fixture中改变profile或功率参数。 | 旧数值cache不被误复用；正式规则变更被preflight拒绝。 **证据：**cache key/失效日志。 |
| C06 · peak全低与全高 | 明确在连续功率阈值两侧的函数级轨迹。 | peak为0或完整区间长度；保留原有效域与功率上限校验。 **证据：**解析时间长度对照。 |
| C07 · peak阈值复杂交叉 | 单次、多次、切触、窄区间与分段交叉。 | 不低估原保守时间；不能相减两个上界当作区间上界；必要时回退原查询。 **证据：**节点区间证据、fallback和最大差。 |
| C08 · 累计120s边界 | 初始已用peak与候选peak之和位于120s两侧及等号。 | 同R0批准/拒绝；不分航段清零，末位差不能放宽预算。 **证据：**预算边界trace。 |
| C09 · 25%接地储备 | 接地预测SOC位于0.25两侧与等号。 | 正常final结论及实际touchdown SOC同R0；不能改成15m处或起飞前阈值。 **证据：**start/15m/touchdown对照。 |
| C10 · 根与功率失效 | 诱导根分支不可用、域外下降、负功率、NaN或过流。 | 失败状态与原语义相同，不剪裁为正功率、不自动接受、无GPU/float32近似。 **证据：**错误分类与原程序对照。 |

## D. 应急、资源与可观察行为（P3）

| ID | 测试情形 | 预期行为及最低证据 |
|---|---|---|
| D01 · 高SOC仍保留qE语义 | SOC高于0.35、但不同参考能耗的holding与inflight状态。 | 不只靠SOC阈值跳过；q_e、reference_missing及alert同R0。 **证据：**监测输入/输出逐字段对照。 |
| D02 · 相位分区边界 | 三个环、六pad、全部24分区边界±小偏移及2π wrap。 | 精确相位分别验证；禁止time/phase bin近似复用改变qE或候选。 **证据：**reference候选与最优pad对照。 |
| D03 · 模板无效fallback | canonical不存在、exact实例化失败与完整搜索NOT_FOUND。 | fallback集合/顺序与最终None一致；不能把模板失败判作物理不可行。 **证据：**attempts与missing事件。 |
| D04 · 在途终点pinning | 机动过程中多次query同一目标时刻，然后完成或改version。 | 固定终点结果复用；结束/变版后正确失效；状态修改仍串行。 **证据：**pin命中、失效与原结果。 |
| D05 · 不可撤销机动 | 应急在跨环途中触发并延续到目标环。 | 原航段不变；pending、声明时刻、实际应急源环和收费一致。 **证据：**机动hash与事件流。 |
| D06 · pad未来释放 | pad此刻忙但下降开始前释放；另设下降早于释放反例。 | 前者不能被free_now预检删除，后者按保护区间拒绝；保持R0。 **证据：**资源区间与submit结果。 |
| D07 · soft和committed保护 | 同pad不重叠soft、重叠soft、未来committed三种预约。 | 只按R0允许范围重排；不取消已开始航段，不删除不重叠soft。 **证据：**原子提交前后日历和依赖。 |
| D08 · 上游等待与守恒 | 空槽但轨迹冲突、永久满槽、同gate多请求。 | WAITING/FIFO/容量分流及initial_soc语义不变；不添加物理队列飞机。 **证据：**上游、物理机群及成本记录。 |
| D09 · 拒绝原因与trace | 候选同时可能违反运动/能量/资源，启用资源预检。 | 主reason、逻辑序列、final_attempt_rejected事件符合R0；无法保持则预检不开启。 **证据：**原/新rejection逐项差分。 |
| D10 · 多机与平局顺序 | 相同SOC、同参考代价与同时资源释放。 | 按原飞机/候选顺序处理；一次性应急费、正常/应急机会不被并行时序改写。 **证据：**精确事件及选择结果。 |

## E. Checkpoint与恢复（P1）

| ID | 测试情形 | 预期行为及最低证据 |
|---|---|---|
| E01 · 全状态round-trip | 混合active/soft/依赖/上游/peak/服务状态。 | 恢复所有语义字段、容器顺序、时间控制游标和轨迹hash；无默认值补猜。 **证据：**checkpoint schema与逐字段diff。 |
| E02 · 中断续跑对照 | 同一运行在自然安全切点保存后恢复。 | 与不间断运行逐帧、事件、候选和最终结局相同，费用不重复。 **证据：**完整连续性报告。 |
| E03 · 同刻事件恢复 | 触地、服务释放、请求、软计划启动在同刻或极接近。 | 原事件优先与精确边界不变；不重复_settle/_update_emergency。 **证据：**事件时序trace。 |
| E04 · checkpoint不切步 | 请求在非网格时刻或kernel运行中保存。 | 延后到原自然切点；没有额外advance目标与decision_times。 **证据：**hook开关对照。 |
| E05 · 历史依赖不丢 | source-clear事件已记录，历史流外置后恢复。 | 去重与支持依赖仍正确，不能重复清空/解绑或漏事件。 **证据：**事件处理集合/前缀恢复测试。 |
| E06 · 无穷端点与顺序 | 含inf的永久预约、tuple/dict与接近阈值浮点。 | 无损round-trip；inf不变null；insertion order、整数key不被错误更改。 **证据：**二进制/结构往返检查。 |
| E07 · 断电与半条日志 | 在写checkpoint、写JSONL和flush前后注入中断。 | 只读取完整原子checkpoint；新attempt回滚未提交尾部，旧证据只读。 **证据：**故障注入与prefix hash。 |
| E08 · 损坏与错版本 | 错schema、配置、policy、plan hash或实现迁移标签。 | 恢复前拒绝；不能恢复后补记不一致。 **证据：**所有负向退出原因。 |
| E09 · 冷热cache恢复 | 同一checkpoint清空/恢复派生cache，含worker重建。 | 冷/热物理结果同一；canonical选择不受首次访问顺序影响。 **证据：**多cache模式结果对照。 |
| E10 · 归档不是checkpoint | 仅有states.jsonl或旧策略replay，尝试标为R0连续恢复。 | 入口拒绝或明确标为单点重建fixture；不能形成伪完整episode。 **证据：**类型和来源边界检查。 |

## F. 单种子内部并行（P4）

| ID | 测试情形 | 预期行为及最低证据 |
|---|---|---|
| F01 · worker数量与平局等价 | 纯serial、进程池1/2/4/6，扰动任务完成顺序。 | 同query/pad归并；qE、最优plan、候选和world事件同R0。 **证据：**完整payload与事件diff。 |
| F02 · worker cache隔离 | 同query重复、跨worker分派与重新启动。 | cache只影响工作量；无可变对象污染，结果不随历史变化。 **证据：**重复/重建对照。 |
| F03 · 父进程独占world | 大量并行pad任务，在执行返回前观察world。 | worker不揭示需求、不提交或写父资源；状态仅按原顺序修改。 **证据：**mutation边界测试。 |
| F04 · 任务故障传播 | worker异常、死亡、错误query身份或错profile。 | 报告实现故障，不转None/NOT_FOUND/物理失败；保留原诊断。 **证据：**故障类型与处理记录。 |
| F05 · 进程完整退出 | 正常、取消、异常、外部时间预算终止。 | 全部子进程退出或被supervisor回收；无孤儿继续消耗CPU。 **证据：**进程树前后记录。 |
| F06 · 完整CPU与RSS核算 | 包含spawn、IPC、结果序列化、任务和join。 | 父/子总CPU与同步RSS报告完整；不只用completed-pad-task计时。 **证据：**OS计数与测量覆盖说明。 |
| F07 · benchmark无逐任务profile | 启用benchmark与profile分别执行同一批query。 | benchmark不创建逐pad cProfile/完整function表，语义一致；开销单列。 **证据：**执行路径计数与on/off对照。 |
| F08 · 资源预算不超配 | 多个独立job请求内部worker。 | 1父+最多6子，native线程1，总physical allocation可核算；无20×6隐式超配。 **证据：**环境和affinity/预算报告。 |

## G. 性能测量与门槛（P5）

| ID | 测试情形 | 预期行为及最低证据 |
|---|---|---|
| G01 · 低开销仪器资格 | 同输入关闭计数与benchmark低成本计数对照。 | 中位wall额外开销≤5%，物理输出和记录策略相同。 **证据：**原始计时与比率。 |
| G02 · profile与benchmark隔离 | 三个共同窗口各跑profile on/off。 | 仪器开销单列；两边均off的数据才进入单核加速。 **证据：**模式标识与主表过滤测试。 |
| G03 · 六窗口与重复完整 | 第15.2节六个共同checkpoint及3次配对。 | hash冻结、AB/BA/AB顺序、未因速度挑样；TIMEOUT不冒充完整值。 **证据：**input manifest、所有重复原值。 |
| G04 · 单核提升门槛 | R0与OPT单核10s六窗。 | 几何平均加速≥2.0，各窗中位劣化不超过max(10%,0.05s)规定。 **证据：**公式重算与逐窗判定。 |
| G05 · 持续窗口绝对门槛 | 选定后端六个60s窗口各3次。 | 各窗中位wall/sim≤10，六窗几何平均≤5；完整记录开启。 **证据：**逐窗原值及gate。 |
| G06 · 完整15项端到端 | 3 rates×5 seeds，从0到5400s或相同物理失败。 | 每项有效累计wall≤7200s；同policy oracle/逐段链完整；恢复不重置计时。 **证据：**result、链、回放、实际停止依据。 |
| G07 · 内存和cache有界 | 长运行、cache churn、6worker与多次恢复。 | serial≤2GiB、并行tree≤8GiB；派生cache预算执行；无持续无界增长。 **证据：**RSS时间序列/eviction/peak。 |
| G08 · 异常与统计诚实 | 人为卡慢、到达wall预算、重复测量噪声与物理失败。 | 明确PERFORMANCE_TIMEOUT/NOISY/物理结局；不算零耗时、无假清空，无漏总CPU。 **证据：**统计器与gate负向测试。 |

## H. 原场景回归、回放和发布（P6）

| ID | 测试情形 | 预期行为及最低证据 |
|---|---|---|
| H01 · 原94项有效准则映射 | 旧目录与R0修订影响项。 | 原目录不改；每项明确effective criteria、实测状态、源码来源；旧PASS不自动继承。 **证据：**mapping和当前回归报告。 |
| H02 · 既有平台成对回放 | 正常入场/跨环、上游等待、提前出发到未来pad。 | reference/OPT同时间轴可观察；未来committed显示正确，物理轨迹不重生成。 **证据：**回放索引与观察清单。 |
| H03 · 应急与边界可视化 | 外环直达、繁忙pad救援、80m附近与能源失败。 | 真实SOC/功率/peak、源环代价和拒绝原因可追溯；不使用平滑掩盖差异。 **证据：**真实事件链接与人工备注。 |
| H04 · 恢复接缝与全量记录 | checkpoint前后、seek/逐步/事件跳转和记录flush。 | 无缺帧、重复事件或轨迹跳变；server viewer关闭不影响可回放完整性。 **证据：**流hash、接缝对照与交互检查。 |
| H05 · 人工验收不预签 | 自动测试已通过但研究负责人尚未观看。 | 保持HUMAN_PENDING；只有真实observer/时间/hash/结论齐全才VISUAL_ACCEPTED。 **证据：**签收记录与gate负向样例。 |
| H06 · 发布与交接不越界 | 全部gate结果及未完成项汇总。 | 任何缺证不PERF_ACCEPTED；不启动阶段2/学习；交付真实运行命令、开关和性能边界。 **证据：**最终状态机、交接报告和回滚说明。 |


---

# 附录B：实施者不可越过的停机线

遇到任一情形，停止合并相关优化，保留失败证据，不自动改研究规则：reference并非R0；规则hash变化；原SAFE/UNKNOWN判定被无声明改变；漏掉pad/slot候选；近边界动作/应急事件翻转；回放与执行轨迹不同；checkpoint重复事件或费用；worker失败被吞掉；纯CPU时间只报父进程；缺少慢窗却宣称加速；通过profile开关制造“算法加速”；把上游请求加入物理空域；为了加速关闭原完整记录；性能门槛未达却自动进入阶段2。

**最终验收回答的是“同一仿真是否被正确、可恢复、可核算地加速”，不是“新的运行规则或更强调度是否得到更好结局”。**

## 固定来源链接

| 来源 | 固定资源 |
|---|---|
| S1 | [1c5afd37c891a0ed4776f64367fc7264db3c1c54](https://github.com/Michae1JeeSEU/TerminalRL/commit/1c5afd37c891a0ed4776f64367fc7264db3c1c54) |
| S2 | [docs/rotating_slots_stage1/amendment_20261018.md](https://github.com/Michae1JeeSEU/TerminalRL/blob/1c5afd37c891a0ed4776f64367fc7264db3c1c54/docs/rotating_slots_stage1/amendment_20261018.md) |
| S3 | [terminal_marl/core/rotating_slots/world.py](https://github.com/Michae1JeeSEU/TerminalRL/blob/1c5afd37c891a0ed4776f64367fc7264db3c1c54/terminal_marl/core/rotating_slots/world.py) |
| S4 | [terminal_marl/core/rotating_slots/trajectory.py](https://github.com/Michae1JeeSEU/TerminalRL/blob/1c5afd37c891a0ed4776f64367fc7264db3c1c54/terminal_marl/core/rotating_slots/trajectory.py) |
| S5 | [terminal_marl/core/rotating_slots/energy.py](https://github.com/Michae1JeeSEU/TerminalRL/blob/1c5afd37c891a0ed4776f64367fc7264db3c1c54/terminal_marl/core/rotating_slots/energy.py) |
| S6 | [terminal_marl/core/rotating_slots/safety.py](https://github.com/Michae1JeeSEU/TerminalRL/blob/1c5afd37c891a0ed4776f64367fc7264db3c1c54/terminal_marl/core/rotating_slots/safety.py) |
| S7 | [scripts/server/parallel_rotating_slots_runtime.py](https://github.com/Michae1JeeSEU/TerminalRL/blob/b936f6ef7b47156f9c3d13ee941580a5da902ebe/scripts/server/parallel_rotating_slots_runtime.py) |
| S8 | [scripts/server/recover_rotating_slots_parallel.py](https://github.com/Michae1JeeSEU/TerminalRL/blob/b936f6ef7b47156f9c3d13ee941580a5da902ebe/scripts/server/recover_rotating_slots_parallel.py) |
| S9 | [terminal_marl/core/rotating_slots/qualification.py](https://github.com/Michae1JeeSEU/TerminalRL/blob/1c5afd37c891a0ed4776f64367fc7264db3c1c54/terminal_marl/core/rotating_slots/qualification.py) |
| S9-driver | [terminal_marl/core/rotating_slots/driver.py](https://github.com/Michae1JeeSEU/TerminalRL/blob/1c5afd37c891a0ed4776f64367fc7264db3c1c54/terminal_marl/core/rotating_slots/driver.py) |
| S10 | [docs/rotating_slots_stage1/README.md](https://github.com/Michae1JeeSEU/TerminalRL/blob/1c5afd37c891a0ed4776f64367fc7264db3c1c54/docs/rotating_slots_stage1/README.md) |
| S10-history | [docs/rotating_slots_stage1/diagnostics_20261016.md](https://github.com/Michae1JeeSEU/TerminalRL/blob/b936f6ef7b47156f9c3d13ee941580a5da902ebe/docs/rotating_slots_stage1/diagnostics_20261016.md) |
