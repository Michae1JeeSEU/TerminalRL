# 固定多环终端区文档入口

用户授权的**阶段2传统调度基线**已完成162场冻结比较和12个有限目录参考；本轮未建立学习必要性证据。阶段1已完成固定问题定义与内部验证；阶段2继承全部物理参数、容量、SOC、负载和生成器，只研究调度方法。尚不设计学习方法或改变几何。

## 阅读顺序

| 文档 | 职责 |
|---|---|
| [阶段2总体计划](fixed_terminal_stage2_traditional_baseline_plan.md) | 用户指定阶段2范围与研究问题 |
| [问题定义](fixed_terminal_problem_definition.md) | 固定几何、运动、能源、资源、外部需求与应急合同 |
| [阶段2方法](arrival_restructure/stage2_traditional_baselines.md) | 六种方法、联合批次、可观测信息与有限目录 oracle |
| [阶段2实验协议](experiments/fixed_terminal_stage2_protocol.md) | 比较输入、种子、预算、目标、输出与验证安排 |
| [阶段2结果](evidence/fixed_terminal_stage2/results.md) | 完整比较、研究判断及紧凑证据入口 |
| [当前状态](CURRENT_STATUS.md) | 已观察到的实现、测试与结果 |
| [研究定位](research_positioning.md) | 当前研究问题与结论边界 |
| [需求](requirements.md)、[设计](design_spec.md)、[技术规范](technical_spec.md) | 行为、接口与模块职责 |
| [执行计划](execution_plan.md) | 阶段2实现、预检、比较与审查顺序 |

最新用户指令和阶段2计划定义当前范围；阶段1问题合同中的物理规则保持冻结。协议在 pilot 前明确调度预算，正式比较不能按结果调整参数。状态页只报告实际进度，不授权未运行的实验结论。

## 保留的阶段1入口

[阶段1总体方案](fixed_terminal_reconstruction_plan_stage1.md)与[阶段1验证](evidence/fixed_terminal_stage1/validation_20261013.md)保留原阶段语义。新核心仍位于 `terminal_marl/core/fixed_terminal/`，环境为 `terminal_marl/envs/fixed_terminal_env.py`；调度器通过独立可行性层执行。

旧动态空域、RL 和前次 arrival 实验位于原 `reference/`、`evidence/arrival_restructure/` 和[历史索引](history/README.md)，不构成本次阶段2证据。[旧研究重构计划](research_restructure_plan.md)仍为已替代说明。
