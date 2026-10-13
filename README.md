# TerminalRL

研究固定多环终端区中，随机剩余电量和独立安全/能源/资源约束下的向内跨环、最终进近与pad联合调度。

**阶段4实现与冻结实验已完成。** rank显示可学习的离线排序信号，但尚无优于同预算beam筛选的综合证据；学习必要性仍未证明。学习仅近似已到达机群的预算闭合参考，确定性生成与安全职责不变，阶段3闭合与实际未来排序反转的限制继续保留。

- [阶段4结果与运行证据](docs/evidence/fixed_terminal_stage4/results.md)
- [阶段4方法](docs/arrival_restructure/stage4_learning_scheduler.md)
- [阶段4冻结协议](docs/experiments/fixed_terminal_stage4_protocol.md)
- [阶段4用户计划](docs/fixed_terminal_stage4_learning_scheduler_plan.md)
- [阶段3结果与证据](docs/evidence/fixed_terminal_stage3/results.md)
- [当前状态](docs/CURRENT_STATUS.md)
- [阶段3用户计划](docs/fixed_terminal_stage3_learning_problem_definition_plan.md)
- [阶段3机制协议](docs/experiments/fixed_terminal_stage3_protocol.md)
- [学习问题定义](docs/fixed_terminal_learning_problem_definition.md)
- [冻结物理合同](docs/fixed_terminal_problem_definition.md)
- [文档索引](docs/README.md)
- [阶段2已完成结果](docs/evidence/fixed_terminal_stage2/results.md)

阶段2的162场及12个有限目录参考已完成：FCFS正常444/失败7、deadline445/7、joint429/9、rolling441/9。rolling相对joint正常+12，但不支配FCFS，未建立学习必要性证据。阶段3检验机制，不能把这些独立rollout差异当作单步因果。模型是合成运动学与归一化能源研究模型，不是硬件标定或实时部署认证。
