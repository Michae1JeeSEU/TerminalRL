# 固定多环终端区阶段1文档入口

当前唯一研究主线是固定多环终端区中的联合到达调度：在随机 SOC 下，联合决定向内跨环窗口、最后进近与 pad 服务时间，并由独立的安全、能源和资源可行性层约束。阶段1建立问题定义与仿真运行域，尚不比较算法。

## 阅读顺序与职责

| 文档 | 用途 |
|---|---|
| [阶段化总体方案](fixed_terminal_reconstruction_plan_stage1.md) | 用户指定的研究顺序与阶段1范围 |
| [问题定义](fixed_terminal_problem_definition.md) | 几何、运动、能源、资源、外部需求、应急和指标语义 |
| [当前状态](CURRENT_STATUS.md) | 实际实现与验证进度 |
| [研究定位](research_positioning.md) | 研究问题与可支持的结论 |
| [需求](requirements.md) | 必须满足的模型行为 |
| [设计](design_spec.md) | 模块边界和数据流 |
| [技术规范](technical_spec.md) | 配置、接口、运行与输出约定 |
| [执行计划](execution_plan.md) | 阶段1实现、验证和审查顺序 |

发生冲突时，以最新用户指令和阶段化总体方案为研究范围，以问题定义为当前模型合同；同步修订其余规范。状态页只记录已观察到的进度，不改变模型或授权。数值参数在 pilot 前冻结；不得为改善结果调整场景。

## 代码入口

- `terminal_marl/core/fixed_terminal/`：固定几何、运动学轨迹、能源、资源与可行性、场景和指标。
- `terminal_marl/envs/fixed_terminal_env.py`：阶段1仿真环境。
- `terminal_marl/configs/fixed_terminal_stage1.py`：阶段1冻结参数。
- `terminal_marl/scripts/evaluate_fixed_terminal_stage1.py`：复现实验与输出。
- `tests/test_fixed_terminal*.py`：新模型的针对性测试。

具体已实现接口和命令见[技术规范](technical_spec.md)；完成度见[当前状态](CURRENT_STATUS.md)。旧 `src/`、动态空域、RL 和 arrival 调度实现不是阶段1运行依赖。

## 历史资产

旧权威文档已保存于 [pre_fixed_terminal_stage1](history/pre_fixed_terminal_stage1/README.md)。[旧研究重构计划](research_restructure_plan.md)已被替代。`reference/`、`evidence/`、`baselines/` 和其他历史训练、资格认证、硬件报告保留其原始实验语义，不能证明新模型的有效性，也不能恢复旧路线中的执行要求。完整历史入口见[历史索引](history/README.md)。
