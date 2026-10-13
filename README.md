# TerminalRL

当前研究固定多环终端区内，在随机剩余电量（SOC）与安全间隔约束下，联合调度向内跨环转移、最后进近和着陆资源。

阶段1独立运动学仿真与内部验证、**阶段2传统调度基线的162场比较与12个有限目录参考**已完成：在完全相同的几何、安全、容量、能源、负载与 SOC 生成规则下，比较四种排序规则、联合窗口和滚动优化，并建立小规模有限目录 MILP 参考。尚不引入 RL、IL、teacher 或网络，也不调物理参数。

- [阶段2结果与结论](docs/evidence/fixed_terminal_stage2/results.md)
- [当前状态](docs/CURRENT_STATUS.md)
- [阶段2总体计划](docs/fixed_terminal_stage2_traditional_baseline_plan.md)
- [阶段2方法说明](docs/arrival_restructure/stage2_traditional_baselines.md)
- [阶段2实验协议](docs/experiments/fixed_terminal_stage2_protocol.md)
- [固定问题定义](docs/fixed_terminal_problem_definition.md)
- [文档入口](docs/README.md)

本轮尚未建立学习必要性证据，保留服务结局和计算代价的权衡，不启动第三阶段训练。仿真采用合成归一化能源和有界运动学路径，不是实机标定模型。阶段1证据保留；阶段2结果必须独立记录，不能据有限目录最优性宣称全动态问题最优。
