# 固定终端区文档入口

**阶段5已授权，当前处于实现与预检；正式训练和新test尚未开始。** 当前范围是完整联合批次的闭合参考筛选与计算预算验证，详见方法和冻结协议。阶段4实现与冻结实验已完成。rank显示可学习的离线排序信号，但尚无优于同预算beam筛选的综合证据；学习必要性仍未证明。学习仅近似已到达机群的预算闭合参考，确定性生成与安全职责不变，阶段3闭合与实际未来排序反转的限制继续保留。

## 当前阅读顺序

| 文档 | 职责 |
|---|---|
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
