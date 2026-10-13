# TerminalRL

当前研究：**固定多环终端区内，在随机剩余电量（SOC）与安全间隔约束下，联合调度向内跨环转移、最后进近和着陆资源。**

阶段1建立独立的运动学仿真、可行性层和可复现场景，检验运行域与指标。暂不训练学习模型、不设计 teacher、不比较调度算法，也不改变空域几何。能源参数是归一化研究假设，不是实机标定。

- [当前状态](docs/CURRENT_STATUS.md)
- [阶段化总体方案](docs/fixed_terminal_reconstruction_plan_stage1.md)
- [问题定义](docs/fixed_terminal_problem_definition.md)
- [文档与代码入口](docs/README.md)

新主线代码位于 `terminal_marl/core/fixed_terminal/`，环境为 `terminal_marl/envs/fixed_terminal_env.py`。旧动态空域、RL、arrival 重构代码和报告保留为历史资产，不作为本模型的验证证据。阶段1完成后先审查，再决定阶段2；不自动启动后续研究。
