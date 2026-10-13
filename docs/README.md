# 固定终端区文档入口

## 阶段6当前范围（2026-10-13）

用户已授权[阶段6学习辅助联合调度](fixed_terminal_stage6_learning_assisted_scheduling_plan.md)，当前进入implementation/preflight，正式264episode和30同状态面板（15jobs、240方法行）尚未启动。复用阶段5冻结MLP rank/GB，无需重训；固定物理、93维特征、pool14和FCFS闭合J。八方法比较采用TOTAL B3→B5→B8→full及已到达机群正常完成上界证书，证书不保证未来到达或全J最优。质量条件epsilon_normal=0与同状态非退化计算减少10%门槛预先冻结，A/B/C判断等待正式证据。

当前依据为[方法](arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](experiments/fixed_terminal_stage6_protocol.md)、[运行状态](evidence/fixed_terminal_stage6/run_status.md)。阶段1–5历史事实与结论保持；下文“下一阶段另定范围”等表述属于阶段5结束时的历史状态，本节记录后续明确授权。阶段6仍不开展PPO、actor、teacher动作模仿、offline RL、GNN/attention或改变几何。


阶段5冻结实现、评价与研究验收已完成。存在离线排序信号和部分在线收益，但尚未达到不退化系统质量并明显降低计算的B类门槛，也无长期价值C类证据；不能据此笼统宣称学习完全无效或普遍无必要。保留beamB5/B8及full工程参考，rank/GB保留实验性结果；下一阶段另定范围，不自动扩大网络或启动新实验。版本与交付校验信息由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，交付状态见最终交付回复。

## 当前阅读顺序

| 文档 | 职责 |
|---|---|
| [阶段5结果与研究判断](evidence/fixed_terminal_stage5/results.md) | 完整分母、系统质量/计算权衡及结论边界 |
| [阶段5运行状态](evidence/fixed_terminal_stage5/run_status.md) | 训练/评价完成、失败保留、修复重放及交付状态 |
| [阶段5用户计划](fixed_terminal_stage5_learning_validation_plan.md) | 用户授权范围，原样保存 |
| [阶段5方法](arrival_restructure/stage5_learning_validation.md) | 联合批次筛选、TOTAL预算和证据关系 |
| [阶段5冻结协议](experiments/fixed_terminal_stage5_protocol.md) | 四模型、数据、330episode及质量/计时面板 |
| [阶段4前置诊断](evidence/fixed_terminal_stage5/diagnosis.md) | 旧数据复现、anchors、误差与结论边界 |
| [阶段4结果与运行证据](evidence/fixed_terminal_stage4/results.md) | 完整数据、训练、离线/episode结果、计算成本与结论 |
| [阶段4用户计划](fixed_terminal_stage4_learning_scheduler_plan.md) | 阶段4用户任务，原样保存 |
| [阶段4方法](arrival_restructure/stage4_learning_scheduler.md) | 六方法、统一候选池与学习筛选职责 |
| [阶段4协议](experiments/fixed_terminal_stage4_protocol.md) | seed分组、标签、模型/选择预算和评价顺序 |
| [阶段3用户计划](fixed_terminal_stage3_learning_problem_definition_plan.md) | 用户指定任务；开头方法有效性和传统边界是待检验动机，不是既有证据结论 |
| [阶段3协议](experiments/fixed_terminal_stage3_protocol.md) | 15context预定方案、根池、价值、闭合/真实重放与边界 |
| [阶段3结果](evidence/fixed_terminal_stage3/results.md) | 15context、209根、选择/成本/条件边际与正负结果 |
| [学习问题定义](fixed_terminal_learning_problem_definition.md) | 受约束条件价值或联合批次评分，明确需求/窗口/continuation假设 |
| [冻结物理定义](fixed_terminal_problem_definition.md) | 几何、路径、安全、能源、资源、外部与应急合同 |
| [当前状态](CURRENT_STATUS.md) | 已执行与未执行工作，区分历史测试和新实验 |
| [研究定位](research_positioning.md) | 问题、事实基础和结论范围 |
| [需求](requirements.md)、[设计](design_spec.md)、[技术](technical_spec.md) | 接口责任、信息隔离和输出口径 |
| [执行计划](execution_plan.md) | 阶段5执行状态与阶段4历史记录 |

## 保留的既有结果

[阶段2结果](evidence/fixed_terminal_stage2/results.md)记录完整162场、12个有限目录参考与学习必要性尚未建立的判断；[方法](arrival_restructure/stage2_traditional_baselines.md)、[协议](experiments/fixed_terminal_stage2_protocol.md)和[阶段2计划](fixed_terminal_stage2_traditional_baseline_plan.md)保留原阶段语义。阶段3只读inventory是独立策略轨迹差异，不能替代共同状态的一次根干预。

[阶段1计划](fixed_terminal_reconstruction_plan_stage1.md)与[验证](evidence/fixed_terminal_stage1/validation_20261013.md)保持不变。旧动态空域/RL/arrival资产按[历史索引](history/README.md)解释，不构成新机制结论依据。代码仍由独立固定核心、可行性层和环境承接，公开机制接口已按实现事实记录于技术规范，smoke与正式实验状态单独列明。
