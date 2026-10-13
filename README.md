# TerminalRL

## 阶段6当前范围（2026-10-13）

用户已授权[阶段6学习辅助联合调度](docs/fixed_terminal_stage6_learning_assisted_scheduling_plan.md)，当前进入implementation/preflight，正式264episode和30同状态面板（15jobs、240方法行）尚未启动。复用阶段5冻结MLP rank/GB，无需重训；固定物理、93维特征、pool14和FCFS闭合J。八方法比较采用TOTAL B3→B5→B8→full及已到达机群正常完成上界证书，证书不保证未来到达或全J最优。质量条件epsilon_normal=0与同状态非退化计算减少10%门槛预先冻结，A/B/C判断等待正式证据。

当前依据为[方法](docs/arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](docs/experiments/fixed_terminal_stage6_protocol.md)、[运行状态](docs/evidence/fixed_terminal_stage6/run_status.md)。阶段1–5历史事实与结论保持；下文“下一阶段另定范围”等表述属于阶段5结束时的历史状态，本节记录后续明确授权。阶段6仍不开展PPO、actor、teacher动作模仿、offline RL、GNN/attention或改变几何。


研究固定多环终端区中，随机剩余电量和独立安全/能源/资源约束下的向内跨环、最终进近与pad联合调度。

阶段5冻结实现、评价与研究验收已完成。存在离线排序信号和部分在线收益，但尚未达到不退化系统质量并明显降低计算的B类门槛，也无长期价值C类证据；不能据此笼统宣称学习完全无效或普遍无必要。保留beamB5/B8及full工程参考，rank/GB保留实验性结果；下一阶段另定范围，不自动扩大网络或启动新实验。完整结果见下方阶段5报告。版本与交付校验信息由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，交付状态见最终交付回复。

- [阶段5结果与研究判断](docs/evidence/fixed_terminal_stage5/results.md)
- [阶段5运行与失败记录](docs/evidence/fixed_terminal_stage5/run_status.md)
- [阶段5方法](docs/arrival_restructure/stage5_learning_validation.md)
- [阶段5冻结协议](docs/experiments/fixed_terminal_stage5_protocol.md)
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
