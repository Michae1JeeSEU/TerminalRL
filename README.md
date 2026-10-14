# TerminalRL

## 阶段7用户授权状态（2026-10-14）

用户已单独授权[阶段7Learning to Search计划](docs/fixed_terminal_stage7_learning_to_search_plan.md)。**pilot已通过流程核验，准备正式搜索数据生成；正式模型、264episodes及15paneljobs尚未运行。** 八方法、一次正式GradientBoostingClassifier fit、train108/val36与30states/240panelrows范围保持[阶段7方法](docs/arrival_restructure/stage7_learning_to_search.md)及[冻结协议](docs/experiments/fixed_terminal_stage7_protocol.md)定义。Pilot模型的train/val使用同一pilot，仅验证流程，不作为正式训练或泛化证据；当前没有新fit、重训或调参。

主对话已接受C三episode和D隔离panel的实际退出、完整产物、请求守恒/最终检查、模型hash及无残留任务证据，紧凑记录见[运行状态](docs/evidence/fixed_terminal_stage7/run_status.md)与[pilot执行证据](docs/evidence/fixed_terminal_stage7/pilot_execution.json)。已授权下一步依次为正式data36jobs→Step7.1主审→一次正式模型fit并冻结→264episodes（workers20、threads1）→15隔离paneljobs（worker1）→严格summary、文档与结果包。正式run_sourceSHA将在本轮文档提交后设置真实最新SHA，实际启动时间尚未知；继续沿用当前chat heartbeat `7-learning-to-search`，不重复创建。

阶段6停止扩展当前学习排序方案的结论及阶段1–6全部历史证据保留；旧文档中“不自动启动下一阶段”描述的是当时边界，本次只执行新用户计划明确授权的阶段7，不自动扩展长期Q、模型或改变几何。Pilot并发计时和不同profile只用于流程与耗时预估，不证明方法差异。

## 阶段6研究验收完成（2026-10-14）

[阶段6用户计划](docs/fixed_terminal_stage6_learning_assisted_scheduling_plan.md)的264episode、15个独立panel jobs（30states/240方法行）及严格汇总全部完成，模型hash未变。**B未成立；相对同机制beam_adaptive未见学习增量计算收益，当前学习排序配置按A停止扩展，保留实验资产。** 学习相对full的非退化根数301→202（减少32.9%），完整wall仅减少约6.4%–6.6%，未达预定双10%门槛；相对beam_adaptive三学习根数都相同，wall比约0.9994/1.0015/1.0013。C的大量验证现象存在，但不支持扩大value或长期Q路线，不启动新训练或下一阶段。

每方法33场/1163请求，仍各有6次能源失败；全部间隔违反/未认证为0、最终执行验证通过，不能写成零能源失败。结论限定pool14、已到达FCFS闭合值、单模型seed5401和本profile新seed及初态占用变体，不泛称所有学习无效。主表、15种质量/计算条件、泛化和成本见[阶段6结果](docs/evidence/fixed_terminal_stage6/results.md)；[方法](docs/arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](docs/experiments/fixed_terminal_stage6_protocol.md)、[运行状态](docs/evidence/fixed_terminal_stage6/run_status.md)保留完整范围与执行事实。源码冻结`f1018d8aaeff03836b13475301087f83ab70a155`。交付版本、ZIP条目与校验结果由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，包链接与最终交付状态见交付回复。

阶段1–5历史计划和结论保持原样；下文“下一阶段另定范围”等属于当时历史状态。后续可另定高效传统优化范围，本次不自动开展新实验、actor/PPO、teacher动作模仿、offline RL、GNN/attention或改变几何。


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
