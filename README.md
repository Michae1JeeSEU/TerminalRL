# TerminalRL

## 阶段7用户授权状态（2026-10-14）

用户已单独授权[阶段7Learning to Search计划](docs/fixed_terminal_stage7_learning_to_search_plan.md)。**正式data完成，Step7.1已通过主对话审查；唯一正式模型已完成冻结并通过主审，正式模型单决策接缝也已通过；264episodes已完成终态核验（264unique、0jobfailed、exit0）；15隔离paneljobs已唯一启动，运行未终态，两个phase无重叠；严格summary尚未生成、A/B/C未判断。** 数据/分析实际源码为`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`；36/36任务exit0，144context（train108/val36）、stored scenario/SOC/config/times与source hash核验通过。完整字段及边界见[搜索瓶颈记录](docs/evidence/fixed_terminal_stage7/search_bottleneck.md)与[数据/分析证据](docs/evidence/fixed_terminal_stage7/data_analysis_evidence.json)。

首次data的33成功/3失败、132context和全部日志保留在ROOT/data_attempt1，936files SHA/bytes一致；旧source仅SOC末位差的兼容性修复及失败事实继续见[运行状态](docs/evidence/fixed_terminal_stage7/run_status.md)。Step7.1的negative只表示未贡献有限width128最佳叶，不是反事实无价值；312,713个eligible节点含train与val，positive1.463%，accuracy不能独证学习有效。不修改参数、class weight、threshold、dataset或矩阵，不预判A/B/C。

一次正式fit已使用c892源码、ROOT/data/train、ROOT/data/val和ROOT/models完成，无smoke，fit86.591769 s、wall96.33 s；四文件hash及144context绑定见[模型冻结证据](docs/evidence/fixed_terminal_stage7/model_freeze.json)。正式EP外层11:55:32.967143、内部dispatcher11:55:33.018977北京时间开始，12:34:44.727322结束，duration2351.708353 s、time2351.77 s（39.2分钟），workers20/threads1，48decisions各场，共12,672条，1,320个预期文件齐全、finite、final feasible与守恒通过，四模型hash不变，无残留EP任务。全八方法累计N/E/F/C=4524/197/16/3791；0jobfailed不表示0能源失败，最终每arm指标待严格summary。原件见[EP终态](docs/evidence/fixed_terminal_stage7/episodes_terminal.json)。

15隔离paneljobs于13:36:16.418133北京时间正常唯一启动，wrapper/dispatcher/worker PID3487053/3487054/3487062，worker1/threads1；13:36:33观测0完成/1运行/14待启动/0失败，首场t0 snapshot、8arm JSON与full_values非空，stderr0。T0=52分钟，预计14:28:16完成，仍未终态；原件见[panel启动](docs/evidence/fixed_terminal_stage7/panels_launch.json)。Data、模型、EP及panel实际source均保持c892，后续文档SHA不改写actual run来源；沿用当前chat heartbeat按估时完成复查，无常驻monitor。初EP100分钟估计保留为历史，实际39.2分钟已记录。模型、种子、物理与范围保持[阶段7方法](docs/arrival_restructure/stage7_learning_to_search.md)及[协议](docs/experiments/fixed_terminal_stage7_protocol.md)定义；后续继续既定汇总与研究验收，不扩展实验或重跑EP，当前尚无最终研究结论。

阶段6停止扩展当前学习排序方案的结论及阶段1–6全部历史证据保留；本次只执行新用户计划明确授权的阶段7，不自动扩展长期Q、模型或改变几何。Pilot训练val同pilot仅为流程，data多width累计进程测时及不同profile/并发episode只作排程预估，不证明方法差异。

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
