# 固定终端区需求：阶段4学习辅助调度

## 阶段6研究验收完成（2026-10-14）

[阶段6用户计划](fixed_terminal_stage6_learning_assisted_scheduling_plan.md)的264episode、15个独立panel jobs（30states/240方法行）及严格汇总全部完成，模型hash未变。**B未成立；相对同机制beam_adaptive未见学习增量计算收益，当前学习排序配置按A停止扩展，保留实验资产。** 学习相对full的非退化根数301→202（减少32.9%），完整wall仅减少约6.4%–6.6%，未达预定双10%门槛；相对beam_adaptive三学习根数都相同，wall比约0.9994/1.0015/1.0013。C的大量验证现象存在，但不支持扩大value或长期Q路线，不启动新训练或下一阶段。

每方法33场/1163请求，仍各有6次能源失败；全部间隔违反/未认证为0、最终执行验证通过，不能写成零能源失败。结论限定pool14、已到达FCFS闭合值、单模型seed5401和本profile新seed及初态占用变体，不泛称所有学习无效。主表、15种质量/计算条件、泛化和成本见[阶段6结果](evidence/fixed_terminal_stage6/results.md)；[方法](arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](experiments/fixed_terminal_stage6_protocol.md)、[运行状态](evidence/fixed_terminal_stage6/run_status.md)保留完整范围与执行事实。源码冻结`f1018d8aaeff03836b13475301087f83ab70a155`。交付版本、ZIP条目与校验结果由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，包链接与最终交付状态见交付回复。

阶段1–5历史计划和结论保持原样；下文“下一阶段另定范围”等属于当时历史状态。后续可另定高效传统优化范围，本次不自动开展新实验、actor/PPO、teacher动作模仿、offline RL、GNN/attention或改变几何。


阶段4实现、固定数据与训练、108context离线评价和162个完整episode均已完成。设计集中于[阶段4方法](arrival_restructure/stage4_learning_scheduler.md)，预算与执行顺序见[阶段4协议](experiments/fixed_terminal_stage4_protocol.md)；下列阶段1–3内容保留既有合同和历史事实。

阶段4完整结果与结论见[结果报告](evidence/fixed_terminal_stage4/results.md)，rank显示可学习的离线排序信号，但尚无优于同预算beam筛选的综合证据；学习必要性仍未证明。

本需求继承阶段1/2固定物理合同，落实用户授权的[阶段3机制协议](experiments/fixed_terminal_stage3_protocol.md)；物理与原始统计以[问题定义](fixed_terminal_problem_definition.md)为准，价值接口见[学习问题定义](fixed_terminal_learning_problem_definition.md)。

## 固定结构与联合决策

1. 固定多环、向内相邻跨环节点、最后进近路径、正常与应急 pad；运行中不得改变半径、高度、容量或路径结构。
2. 分别报告由空间间隔推导的几何容量和扣除运营余量后的运营容量。高负载初始各环非空且接近运营容量，持续产生外部请求，初态仍须可行。
3. 调度接口必须表达各环等待、跨环窗口、最后进近和 pad 占用的联合计划，不能退化成单一初始放行顺序。

## 独立可行性

- 可行性层独立于调度器与任何未来学习模块，验证连续轨迹间隔、运动学界限、环容量、转移合法性、最终进近时间间隔、pad 服务占用和能源储备。
- 正常路径能源随位置、等待和路径变化；按环内等待、转移、下降和着陆分别累计归一化消耗。
- 应急必须以已定义有限正常候选域内不存在可行计划为前提，并满足独立应急路径、资源和储备要求。搜索预算耗尽只能表示未知，不能证明正常不可行。
- 无可行应急、能源不足或其他失败应保留明确原因；低 SOC 本身不是应急标签。

## 需求、随机性与观测

- 低、中、高负载与高、中、低 SOC 有固定定义；SOC 用拒绝采样截断正态，不能用截断裁剪代替。
- 到达流与 SOC 使用相互独立、可复现的随机流。相同需求种子改变 SOC 情景时不得改变到达时刻。
- 外部请求从产生时刻进入统计；显式定义准入前等待的时间、位置边界与能耗，禁止用无限外部队列隐藏低吞吐或耗能。
- 明确有限观察窗及截尾，不把未完成请求算作成功或从分母删除。

## 验证与交付

新核心配套针对性测试，覆盖几何容量、连续安全、联合窗口、能源、资源、应急语义、随机性和指标。先小样本核对输出，再运行冻结场景。阶段1交付已完成且证据保持不变；阶段2六方法实现、162场相同输入比较、12个有限目录oracle与计算/结局核对已完成，见[结果报告](evidence/fixed_terminal_stage2/results.md)。本轮学习必要性尚未建立，停止于研究判断，不自动开始学习阶段。旧实验、策略权重与硬件适配不构成本阶段依赖。

## 阶段2共同调度要求

1. 普通联合批次独立原子验证，保留已承诺计划和未选飞机保持；不得通过顺序部分提交伪装联合兼容。
2. 能源余量基于最低可行正常耗能加储备，deadline余量基于最后可行正常完成窗口；截断时明确未知/界限。
3. 滚动优化必须实际推进可观测状态并包含 HOLD/延后分支，仅承诺首批完整路线，不读取未来 `pending` 请求。
4. 方法共用应急证书和所有安全/能源/资源规则；预算不足不等于正常不可行。
5. 162场完整比较与12个闭合 oracle实例按[冻结协议](experiments/fixed_terminal_stage2_protocol.md)执行；小实例共用候选目录并按共同目标重算评价，各方法内部评分保持各自定义。有限目录最优不得称为动态全局最优。

## 已完成阶段3要求

1. 只使用已冻结15context；t60取共同FCFS prefix后的决策前状态，所有根共享状态、候选池和FCFS continuation。案例依旧结果选择，不作heldout宣称。
2. 根池为扩展joint前12唯一批次，加HOLD和六控制根，去重最多19；全部由独立可行性执行，不改变应急证书或已有承诺。
3. J采用episode累计实际分量`(-F,N,-E,-C,-T,-D,-energy)`，不拟合权重、不用计划完成或激活代替实际着陆。保留原始分量及条件均值。
4. 闭合观测世界和ex-post固定原需求世界分别记录分母、J与差分；未到达身份/SOC/time及结果标签不得进入决策输入。
5. 单机边际条件在其余批次，两批都可行才定义，否则undefined+状态；不得假定价值可加或用整策略差异替代单次干预。
6. t0控制为预计算warm-start，保留原求解计时；t60重新求六控制根。分项记录候选生成、控制获取和两类rollout成本。
7. 阶段3已完成15context与定义核对，见[结果](evidence/fixed_terminal_stage3/results.md)；学习必要性仍未证明。undefined不得作为0或负奖励，词典序收益不得写成Pareto改善。
8. 后续接口须明示continuation、horizon、需求假设，允许结果分量向量或批次比较分数；阶段3当时不开展网络训练或调奖励；其后授权的阶段4范围见下节，仍不实施actor/PPO/IL/teacher。

## 阶段4新增要求

阶段4固定数据、训练与全部评价已完成。必须遵循冻结train/val/test seed组，所有同场景候选归同集合，scaler只用train；两checkpoint确定后才首次生成test。只近似闭合参考，不声明期望未来价值，不用阶段3数据训练。

六方法使用相同物理/安全合同；新统一池无warm-start，full/beam/value/rank共享定义和anchors预算，原rolling目标差异明确标注。共同单根快路径与非退化表现分开，模型仅筛选，最终精确J选择及独立提交复查不可省略。

报告全请求结局、应急分母、暴露/条件延迟/能耗、安全及完整链成本，不能以推理耗时替代在线总耗时。仅两固定简单MLP各训练一次，不调奖励、不按结果改模型/seed/预算，offline RL及复杂架构不实施。

## 阶段5具体要求

按[阶段5冻结协议](experiments/fixed_terminal_stage5_protocol.md)验证当前已到达机群的联合批次筛选，保持93特征和阶段4train/val；新test seeds5301/5302/5303只在四模型冻结后生成，pilot只用5001。linear_rank、mlp_rank、gradient_boost、value分别按验证集TOTAL B5保优率、各自loss、最早迭代选checkpoint；train-only scaler及明确smoke标记不可省略。

TOTAL B3/5/8包含去重anchors；统一普通池最多14、敏感性池最多32。30场景含27标准与3个medium/medium初相变体；线上11 arms共330场，固定行为90 snapshots；同状态实际计时24状态×11 arms独立重建，不使用标签查表时间冒充求解时间。初相变体只改变指定初始相位，不改变SOC、请求及物理参数。

报告区分闭合参考质量、实际完整episode结局、同状态计算量和continuation敏感性。完整在线计时保留生成/特征/推理/精确rollout/最终验证及应急；rolling内部搜索和仿真步数不能被相同根数掩盖。既有stage4诊断不是新test，也不是长期训练标签。正式结果尚须依据实际运行与汇总证据记录，不按结果改模型、预算、种子或删除失败。
