# 阶段6执行记录：研究验收完成

## 阶段7用户授权状态（2026-10-14）

用户已单独授权[阶段7Learning to Search计划](fixed_terminal_stage7_learning_to_search_plan.md)。**正式data完成，Step7.1已通过主对话审查；唯一正式模型已完成冻结并通过主审，正式模型单决策接缝也已通过；264episodes已正常唯一启动并通过启动核验，运行中、尚未终态；15隔离paneljobs未启动且不重叠。** 数据/分析实际源码为`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`；36/36任务exit0，144context（train108/val36）、stored scenario/SOC/config/times与source hash核验通过。完整字段及边界见[搜索瓶颈记录](evidence/fixed_terminal_stage7/search_bottleneck.md)与[数据/分析证据](evidence/fixed_terminal_stage7/data_analysis_evidence.json)。

首次data的33成功/3失败、132context和全部日志保留在ROOT/data_attempt1，936files SHA/bytes一致；旧source仅SOC末位差的兼容性修复及失败事实继续见[运行状态](evidence/fixed_terminal_stage7/run_status.md)。Step7.1的negative只表示未贡献有限width128最佳叶，不是反事实无价值；312,713个eligible节点含train与val，positive1.463%，accuracy不能独证学习有效。不修改参数、class weight、threshold、dataset或矩阵，不预判A/B/C。

一次正式fit已使用c892源码、ROOT/data/train、ROOT/data/val和ROOT/models完成，无smoke，fit86.591769 s、wall96.33 s；四文件hash及144context绑定见[模型冻结证据](evidence/fixed_terminal_stage7/model_freeze.json)。原fit T0=5分钟，仅一个顺序boosting模型，无独立副本可并行，threads1。已推送c892运行源码冻结、数据/模型/接缝主审通过后，主对话已授权Luna在纯文档整理并行期间启动264episodes（workers20、threads1，T0约100分钟），随后15隔离paneljobs（worker1，T0约52分钟）、严格summary、文档与结果包；当前模型终态与单决策API接缝已通过主对话核验，EP实际source仍为c892，11:55:32.967143北京时间启动，wrapper/dispatcher PID3478486/3478487；11:56:11观测9完成/20运行/235待启动/0失败，共264，T0=100分钟、预计13:35:32结束；后续文档SHA不改写actual run来源，启动原件见[episode_launch.json](evidence/fixed_terminal_stage7/episode_launch.json)。模型、种子、物理与范围保持[阶段7方法](arrival_restructure/stage7_learning_to_search.md)及[协议](experiments/fixed_terminal_stage7_protocol.md)定义，沿用当前chat heartbeat。

阶段6停止扩展当前学习排序方案的结论及阶段1–6全部历史证据保留；本次只执行新用户计划明确授权的阶段7，不自动扩展长期Q、模型或改变几何。Pilot训练val同pilot仅为流程，data多width累计进程测时及不同profile/并发episode只作排程预估，不证明方法差异。

## 阶段6研究验收完成（2026-10-14）

[阶段6用户计划](fixed_terminal_stage6_learning_assisted_scheduling_plan.md)的264episode、15个独立panel jobs（30states/240方法行）及严格汇总全部完成，模型hash未变。**B未成立；相对同机制beam_adaptive未见学习增量计算收益，当前学习排序配置按A停止扩展，保留实验资产。** 学习相对full的非退化根数301→202（减少32.9%），完整wall仅减少约6.4%–6.6%，未达预定双10%门槛；相对beam_adaptive三学习根数都相同，wall比约0.9994/1.0015/1.0013。C的大量验证现象存在，但不支持扩大value或长期Q路线，不启动新训练或下一阶段。

每方法33场/1163请求，仍各有6次能源失败；全部间隔违反/未认证为0、最终执行验证通过，不能写成零能源失败。结论限定pool14、已到达FCFS闭合值、单模型seed5401和本profile新seed及初态占用变体，不泛称所有学习无效。主表、15种质量/计算条件、泛化和成本见[阶段6结果](evidence/fixed_terminal_stage6/results.md)；[方法](arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](experiments/fixed_terminal_stage6_protocol.md)、[运行状态](evidence/fixed_terminal_stage6/run_status.md)保留完整范围与执行事实。源码冻结`f1018d8aaeff03836b13475301087f83ab70a155`。交付版本、ZIP条目与校验结果由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，包链接与最终交付状态见交付回复。

阶段1–5历史计划和结论保持原样；下文“下一阶段另定范围”等属于当时历史状态。后续可另定高效传统优化范围，本次不自动开展新实验、actor/PPO、teacher动作模仿、offline RL、GNN/attention或改变几何。


## 阶段5历史执行记录

阶段5冻结实现、评价与研究验收已完成。存在离线排序信号和部分在线收益，但尚未达到不退化系统质量并明显降低计算的B类门槛，也无长期价值C类证据；不能据此笼统宣称学习完全无效或普遍无必要。保留beamB5/B8及full工程参考，rank/GB保留实验性结果；下一阶段另定范围，不自动扩大网络或启动新实验。详细结果见[结果报告](evidence/fixed_terminal_stage5/results.md)，固定参数见[协议](experiments/fixed_terminal_stage5_protocol.md)，原始运行与失败处理见[运行记录](evidence/fixed_terminal_stage5/run_status.md)。

1. **旧证据诊断与冻结已完成。** 保留阶段4诊断及用户计划；固定93维表示、物理、模型seed5401、train/val和新test分组，TOTAL预算含anchors。
2. **实现、pilot与训练已完成。** 四模型各唯一实际训练并冻结；首次cwd启动失败和pilot输出目录失败保留。101项原检查与13项summary/package检查分别通过，不作为一次合并检查。
3. **正式test已完成。** 360jobs全部exit0，330episode、30snapshot jobs、90base contexts、9large contexts、24计时面板/264行，21:17:04.821508北京时间结束，耗时4317.407811 s，四模型hash不变。
4. **continuation修复与完整重放已完成。** 首轮3成功6失败的33files/133416 bytes逐hash保存；仅旧artifact浮点比较修复，26项检查0.79 s通过，提交b7233cf已推送。第二轮9/9成功，21:37:28开始，外层10.20 s、exit0。正式test与模型没有重跑或改变。
5. **严格汇总与研究验收已完成。** summary一次`completed`且`allow_incomplete=false`，完整分母核对通过。学习离线筛选和部分正常完成收益伴随额外能源失败，同状态计时没有明显加速；32上限未实际扩池，continuation全HOLD不能识别排序变化。这些负面与退化证据全部保留。
6. **版本与交付记录。** 版本与交付校验信息由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，交付状态见最终交付回复。

部署边界维持有限池FCFS闭合参考；仅3个初相、同profile family新seed和单一模型seed，20并发耗时不是实时认证。后续研究另定范围。

## 阶段4已完成的执行记录

用户[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)已按[方法](arrival_restructure/stage4_learning_scheduler.md)与[冻结协议](experiments/fixed_terminal_stage4_protocol.md)完成。完整事实和研究结论见[结果报告](evidence/fixed_terminal_stage4/results.md)。rank有离线排序信号，但尚无优于同预算beam的综合证据；学习必要性仍未证明。

1. **准备与冻结已完成。** 原样保存计划，核对运行环境，冻结新seed分组、统一池、简单模型及评价预算。阶段3闭合/实际排序反转限制保留。
2. **实现与tiny预检已完成。** 93维可见特征、train-only scaler、闭合J/Δ标签、两类MLP、top3+anchors及独立复查完成；10项新增、67项合并测试通过。pilot仅验证流程，不计正式集。
3. **train/val与固定训练已完成。** 36个数据任务exit0，train108/val36 contexts；两模型各seed4401、300epoch训练一次。以informative val top3召回、val loss、最早epoch选择value17/rank35后冻结。
4. **首次test已完成。** 27个新场景生成108个离线snapshot，六方法162episode；合计189任务全部exit0。独立offline评价exit0、6.014 s。四模型文件hash未变，无补采样、重跑或追加计时矩阵。
5. **汇总与研究判断已完成。** 保留全部分组、配对J、全请求结局、安全与完整链成本。rank原始排序提高未形成优于beam的最终保优率，在线结果有失败与耗时权衡；不按结果更改预算或删除数据。

阶段4工作已完成。当前优先保留确定性full_reference与beam_topk，rank保留实验性结果；已授权阶段5范围见上文，不扩展GNN/attention、offline RL、actor/PPO、teacher动作模仿或几何变化。阶段1–3计划和证据保持原样，状态见[CURRENT_STATUS](CURRENT_STATUS.md)。
