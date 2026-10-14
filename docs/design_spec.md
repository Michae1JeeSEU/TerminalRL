# 固定多环与学习辅助调度设计

## 阶段7用户授权状态（2026-10-14）

用户已单独授权[阶段7Learning to Search计划](fixed_terminal_stage7_learning_to_search_plan.md)。**正式data完成，Step7.1已通过主对话审查；唯一正式模型已完成冻结并通过主审，正式模型单决策接缝也已通过；264episodes已正常唯一启动并通过启动核验，运行中、尚未终态；15隔离paneljobs未启动且不重叠。** 数据/分析实际源码为`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`；36/36任务exit0，144context（train108/val36）、stored scenario/SOC/config/times与source hash核验通过。完整字段及边界见[搜索瓶颈记录](evidence/fixed_terminal_stage7/search_bottleneck.md)与[数据/分析证据](evidence/fixed_terminal_stage7/data_analysis_evidence.json)。

首次data的33成功/3失败、132context和全部日志保留在ROOT/data_attempt1，936files SHA/bytes一致；旧source仅SOC末位差的兼容性修复及失败事实继续见[运行状态](evidence/fixed_terminal_stage7/run_status.md)。Step7.1的negative只表示未贡献有限width128最佳叶，不是反事实无价值；312,713个eligible节点含train与val，positive1.463%，accuracy不能独证学习有效。不修改参数、class weight、threshold、dataset或矩阵，不预判A/B/C。

一次正式fit已使用c892源码、ROOT/data/train、ROOT/data/val和ROOT/models完成，无smoke，fit86.591769 s、wall96.33 s；四文件hash及144context绑定见[模型冻结证据](evidence/fixed_terminal_stage7/model_freeze.json)。原fit T0=5分钟，仅一个顺序boosting模型，无独立副本可并行，threads1。已推送c892运行源码冻结、数据/模型/接缝主审通过后，主对话已授权Luna在纯文档整理并行期间启动264episodes（workers20、threads1，T0约100分钟），随后15隔离paneljobs（worker1，T0约52分钟）、严格summary、文档与结果包；当前模型终态与单决策API接缝已通过主对话核验，EP实际source仍为c892，11:55:32.967143北京时间启动，wrapper/dispatcher PID3478486/3478487；11:56:11观测9完成/20运行/235待启动/0失败，共264，T0=100分钟、预计13:35:32结束；后续文档SHA不改写actual run来源，启动原件见[episode_launch.json](evidence/fixed_terminal_stage7/episode_launch.json)。模型、种子、物理与范围保持[阶段7方法](arrival_restructure/stage7_learning_to_search.md)及[协议](experiments/fixed_terminal_stage7_protocol.md)定义，沿用当前chat heartbeat。

阶段6停止扩展当前学习排序方案的结论及阶段1–6全部历史证据保留；本次只执行新用户计划明确授权的阶段7，不自动扩展长期Q、模型或改变几何。Pilot训练val同pilot仅为流程，data多width累计进程测时及不同profile/并发episode只作排程预估，不证明方法差异。

## 阶段6研究验收完成（2026-10-14）

[阶段6用户计划](fixed_terminal_stage6_learning_assisted_scheduling_plan.md)的264episode、15个独立panel jobs（30states/240方法行）及严格汇总全部完成，模型hash未变。**B未成立；相对同机制beam_adaptive未见学习增量计算收益，当前学习排序配置按A停止扩展，保留实验资产。** 学习相对full的非退化根数301→202（减少32.9%），完整wall仅减少约6.4%–6.6%，未达预定双10%门槛；相对beam_adaptive三学习根数都相同，wall比约0.9994/1.0015/1.0013。C的大量验证现象存在，但不支持扩大value或长期Q路线，不启动新训练或下一阶段。

每方法33场/1163请求，仍各有6次能源失败；全部间隔违反/未认证为0、最终执行验证通过，不能写成零能源失败。结论限定pool14、已到达FCFS闭合值、单模型seed5401和本profile新seed及初态占用变体，不泛称所有学习无效。主表、15种质量/计算条件、泛化和成本见[阶段6结果](evidence/fixed_terminal_stage6/results.md)；[方法](arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](experiments/fixed_terminal_stage6_protocol.md)、[运行状态](evidence/fixed_terminal_stage6/run_status.md)保留完整范围与执行事实。源码冻结`f1018d8aaeff03836b13475301087f83ab70a155`。交付版本、ZIP条目与校验结果由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，包链接与最终交付状态见交付回复。

阶段1–5历史计划和结论保持原样；下文“下一阶段另定范围”等属于当时历史状态。后续可另定高效传统优化范围，本次不自动开展新实验、actor/PPO、teacher动作模仿、offline RL、GNN/attention或改变几何。


阶段4实现、固定数据与训练、108context离线评价和162个完整episode均已完成。设计集中于[阶段4方法](arrival_restructure/stage4_learning_scheduler.md)，预算与执行顺序见[阶段4协议](experiments/fixed_terminal_stage4_protocol.md)；下列阶段1–3内容保留既有合同和历史事实。

阶段4完整结果与结论见[结果报告](evidence/fixed_terminal_stage4/results.md)，rank显示可学习的离线排序信号，但尚无优于同预算beam筛选的综合证据；学习必要性仍未证明。

## 模块边界

新模型放在 `terminal_marl/core/fixed_terminal/`，与旧动态空域及 RL 栈分离。阶段1核心不导入 ROS2、Crazyswarm2、策略训练器或 teacher。

| 组成 | 职责 |
|---|---|
| 固定几何与配置 | 环、节点、路径、pad、几何及运营容量；运行中不可变 |
| 运动学轨迹 | 环内保持、向内跨环、最后进近、着陆与应急路径；声明速度及下降率界限 |
| 能源 | 按运动模式积分 SOC 消耗，计算正常/应急路径需求和储备 |
| 可行性 | 验证连续安全、窗口、容量、服务占用和能源，返回可解释结果 |
| 场景 | 独立到达/SOC随机流、可行初始占用和外部持续需求 |
| 环境 | 执行已验证计划，推进时间、资源和请求状态 |
| 指标与评价 | 全请求结局、延迟、能源、安全、计算和截尾统计，序列化输出 |

模块内具体公开符号以[技术规范](technical_spec.md)和实际实现为准；本表不规定尚未实现的类名。

## 数据流

冻结配置 → 生成初态与到达请求 → 在可声明的正常候选域生成联合窗口计划 → 独立可行性检查 → 预约资源并执行 → 更新 SOC 与请求结局 → 汇总全请求指标。

联合计划包含向内多个环的等待及转移，而非只选择首次离环时刻。可行性层应可被未来任意调度方法调用，不能把某个启发式策略未找到计划当成物理上无可行方案。正常域被穷尽且为空后，才检查应急路由及其独立资源；未完成的搜索保留未知状态。

外部准入是系统边界的一部分。请求产生后即开始累计时间与声明的外部能耗；资源不足可推迟准入，但必须保留外部积压、失败和截尾记录。

## 模型边界

连续安全检查覆盖采样点之间的轨迹，不能仅逐仿真步检查距离。运动学路径须服从已声明的界限；未建模的加速度、空气动力学和跟踪闭环不能被宣称已验证。归一化能耗用于约束调度研究，不代表 Crazyflie 或载人 eVTOL 的实测消耗。

阶段1简单可行性驱动器保留原用途；阶段2六方法比较已完成，不能把原驱动器当作强调度基线。阶段2不做网络优化或几何重构。旧证据按[历史索引](history/README.md)解释。

## 阶段2调度与联合提交

调度层只产生现有固定路线意图及候选批次，不拥有安全规则。原子批次检查同时验证选中计划之间、选中计划与既有预约、以及未选保持飞机的连续间隔、资源、容量和能源；只有整批可行才承诺，不撤销旧承诺。

排序规则调用正常候选可行性分析以得到能耗/最后完成窗口及搜索状态。联合窗口在相容候选中选择批次。滚动方法复制仅含可观测飞机的状态，推进20s并包含HOLD，后续每5s用联合策略继续；最终仅提交第一批完整路线。不得将预采样未来请求泄漏到分支评估。

小规模oracle把明确有限目录转换为MILP约束，并独立复核整数解与目标；受限求解保留界限/状态。它与六方法的同目录评分独立于完整到达流 rollout。预算与方法语义见[阶段2协议](experiments/fixed_terminal_stage2_protocol.md)，公开API已按实现核对并列于[技术规范](technical_spec.md)。

阶段2实现与冻结实验已完成，47项针对性测试及2项元数据修正测试通过；162场完整比较和12个有限目录参考的事实与边界见[结果报告](evidence/fixed_terminal_stage2/results.md)。本轮尚未建立学习必要性证据，不自动启动后续训练。

## 已完成阶段3机制层

阶段3机制层与15context实验已完成。数据流为既有场景/共同FCFS prefix → 根前完整状态 → 可观测副本与确定性候选池 → 独立批次检查 → 固定FCFS continuation的闭合评估与原需求ex-post重放 → 原始累计J和条件边际。定义见[阶段3协议](experiments/fixed_terminal_stage3_protocol.md)。

闭合副本只保留根时已到达请求及终态历史；完整未来只留在ex-post评估器，不能进入候选或决策观察。两类世界各有请求分母，不能直接混用。候选评分与可行性/执行职责分离，既有预约不撤销。remove-one先检查两批可行；不以预测值替代安全证明。

学习接口定义窗口条件价值或联合批次价值，不设计网络/actor，也不把机制数据库叫teacher。扩大候选和延长窗仅是预算受限参考。公开机制接口已按实现事实列于[技术规范](technical_spec.md)，正式证据见[阶段3结果](evidence/fixed_terminal_stage3/results.md)。13可行/20不可行的局部remove-one不能默认可加；undefined保留状态。输出可为声明窗口/continuation/需求假设后的结果分量向量或批次比较分数，不是固定单机urgency或optimal/expected Q；阶段3当时不开展训练，其后授权的阶段4实现和结果见下节。

## 阶段4新增职责

数据层只从joint behavior共同prefix的可见snapshot生成手工状态/候选特征和闭合J/Δ；模型层仅筛选批次。统一池≤14，学习或beam前3+FCFS/HOLD后≤5根精确rollout，最终按真实key选根并独立复查/共用应急。单根快路径不推理/rollout，完整承诺不撤销。生成中的feasibility开销须计入在线链，不以模型分数替代安全。

train/val/test按新seed场景分组，scaler只拟合train。两简单MLP分别回归七维ΔJ、学习同context非tie pair顺序；输出不是动作模仿。公开接口与93个FEATURE_NAMES已按实现列于技术规范和协议附录。

## 阶段5筛选与预算职责

阶段5在固定93维表示上比较 linear_rank、mlp_rank、gradient_boost 与 value；训练读取阶段4既有train/val，模型冻结后才读取新5301–5303测试场景。当前接口与实验矩阵以[阶段5协议](experiments/fixed_terminal_stage5_protocol.md)为准，不能把实现或预检完成写成正式效果结论。

`validation_models.ValidationScorer`只输出批次排序/价值预测；`validation_scheduler`负责现场候选池、TOTAL预算筛选、闭合评估及提交。TOTAL B=3/5/8先计入去重后的HOLD和现场FCFS anchors，再按排序补足，不能在B之外额外添加anchors。有限池参考评估全部根；rolling20/60保留旧joint continuation及承诺代理目标，其内部搜索与仿真步数另记。网络不替代候选生成、安全、能源、资源、pad或应急证书，已承诺路线不撤销。

独立进程dispatcher只执行冻结任务并记录退出状态，不自动重训、重试或恢复已有场景。交付工具只在正式执行、汇总和模型hash核对完成后生成ZIP；原始数据、模型与日志和紧凑研究报告分目录保留。
