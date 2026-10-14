# 当前状态：pilot流程核验通过，准备阶段7正式数据

## 阶段7用户授权状态（2026-10-14）

用户已单独授权[阶段7Learning to Search计划](fixed_terminal_stage7_learning_to_search_plan.md)。**pilot已通过流程核验，准备正式搜索数据生成；正式模型、264episodes及15paneljobs尚未运行。** 八方法、一次正式GradientBoostingClassifier fit、train108/val36与30states/240panelrows范围保持[阶段7方法](arrival_restructure/stage7_learning_to_search.md)及[冻结协议](experiments/fixed_terminal_stage7_protocol.md)定义。Pilot模型的train/val使用同一pilot，仅验证流程，不作为正式训练或泛化证据；当前没有新fit、重训或调参。

主对话已接受C三episode和D隔离panel的实际退出、完整产物、请求守恒/最终检查、模型hash及无残留任务证据，紧凑记录见[运行状态](evidence/fixed_terminal_stage7/run_status.md)与[pilot执行证据](evidence/fixed_terminal_stage7/pilot_execution.json)。已授权下一步依次为正式data36jobs→Step7.1主审→一次正式模型fit并冻结→264episodes（workers20、threads1）→15隔离paneljobs（worker1）→严格summary、文档与结果包。正式run_sourceSHA将在本轮文档提交后设置真实最新SHA，实际启动时间尚未知；继续沿用当前chat heartbeat `7-learning-to-search`，不重复创建。

阶段6停止扩展当前学习排序方案的结论及阶段1–6全部历史证据保留；旧文档中“不自动启动下一阶段”描述的是当时边界，本次只执行新用户计划明确授权的阶段7，不自动扩展长期Q、模型或改变几何。Pilot并发计时和不同profile只用于流程与耗时预估，不证明方法差异。

## 阶段6研究验收完成（2026-10-14）

[阶段6用户计划](fixed_terminal_stage6_learning_assisted_scheduling_plan.md)的264episode、15个独立panel jobs（30states/240方法行）及严格汇总全部完成，模型hash未变。**B未成立；相对同机制beam_adaptive未见学习增量计算收益，当前学习排序配置按A停止扩展，保留实验资产。** 学习相对full的非退化根数301→202（减少32.9%），完整wall仅减少约6.4%–6.6%，未达预定双10%门槛；相对beam_adaptive三学习根数都相同，wall比约0.9994/1.0015/1.0013。C的大量验证现象存在，但不支持扩大value或长期Q路线，不启动新训练或下一阶段。

每方法33场/1163请求，仍各有6次能源失败；全部间隔违反/未认证为0、最终执行验证通过，不能写成零能源失败。结论限定pool14、已到达FCFS闭合值、单模型seed5401和本profile新seed及初态占用变体，不泛称所有学习无效。主表、15种质量/计算条件、泛化和成本见[阶段6结果](evidence/fixed_terminal_stage6/results.md)；[方法](arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](experiments/fixed_terminal_stage6_protocol.md)、[运行状态](evidence/fixed_terminal_stage6/run_status.md)保留完整范围与执行事实。源码冻结`f1018d8aaeff03836b13475301087f83ab70a155`。交付版本、ZIP条目与校验结果由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，包链接与最终交付状态见交付回复。

阶段1–5历史计划和结论保持原样；下文“下一阶段另定范围”等属于当时历史状态。后续可另定高效传统优化范围，本次不自动开展新实验、actor/PPO、teacher动作模仿、offline RL、GNN/attention或改变几何。

## 阶段5历史状态

更新日期：2026-10-13。阶段5冻结实现、评价与研究验收已完成。存在离线排序信号和部分在线收益，但尚未达到不退化系统质量并明显降低计算的B类门槛，也无长期价值C类证据；不能据此笼统宣称学习完全无效或普遍无必要。保留beamB5/B8及full工程参考，rank/GB保留实验性结果；下一阶段另定范围，不自动扩大网络或启动新实验。版本与交付校验信息由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，交付状态见最终交付回复。完整数据见[阶段5结果](evidence/fixed_terminal_stage5/results.md)，执行与失败记录见[运行状态](evidence/fixed_terminal_stage5/run_status.md)，方法和冻结范围见[方法](arrival_restructure/stage5_learning_validation.md)、[协议](experiments/fixed_terminal_stage5_protocol.md)。

## 已完成的执行证据

- 四模型训练一次并冻结，train108/val36，seed5401；正式test360/360成功、0failed、exit0，21:17:04.821508北京时间结束，duration4317.407811 s。完整覆盖330episode、30snapshot jobs、90base contexts、9large-pool contexts和24真实计时面板/264行；四模型hash不变。
- continuation首轮3成功6失败全部原始证据33files/133416 bytes逐hash保留至`continuation_attempt1/`。旧artifact浮点比较修复提交`b7233cfb58b1b7ebeb534ad095a6373516fac84c`已推送；仅finite float abs1e-12/rel0，config/scenario、类型、离散和结构exact，其他17函数AST一致，不改变物理、安全、J、模型或正式test计算。第二轮21:37:28开始，启动/dispatcher PID3424338/3424339，外层10.20 s、exit0，9/9成功。
- 严格summary一次完成，`status=completed`、`allow_incomplete=false`，覆盖330/30/90/9/24/264/9。正式test源为`4811abb805ab03997f5ce4186c6b2610f3c6b3b3`，修复诊断源为`b7233cf...`。首次训练cwd失败、pilot预建目录失败和continuation浮点比较失败均保留，没有据结果重训或改变评价矩阵。
- 检查分别为原阶段1–5合并101项（17.21 s）、summary7+package6共13项（1.10 s）、修复相关26项（0.79 s），均通过；它们是三次范围不同的检查，不是一次合并运行。正式输出在`D:\7_codespace\training_outputs\terminal\stage5_validation_20261013`。

## 研究判断

B5离线保优为beam80/90、linear87、MLP88、GB89、value77、hybrid87；90context含59 informative和23单根。30episode每方法1010请求，beamB5 N/E/F/C=528/24/4/454，MLP534/25/5/446，GB535/23/5/447。MLP与GB对beam的词典序配对为13/15/2、14/14/2，但在high/low/5303均增加一次能源失败（uav-0026）。正常数或多数胜场增加不能抵销这一代价。

真实同状态24面板的beamB5/MLP/GB总时间为128.355/128.106/128.280 s，均精确评估80根、24/24保留闭合最佳，beam已有同等质量，没有明显加速；beam生成耗时约占76%。各自episode轨迹的MLP/GB累计成本反而更高，但不同状态与并发不能解释为同状态因果。学习B3离线87/90对beamB8的84/90表明筛选潜力；未运行学习B3在线，不能据此声称真实加速。

## 关键边界与下一步

上限32的9个t15面板实际仅1/8/9根，与原上限14池的root ID集合全部相同，未实际扩大池。9个continuation的selected union均仅HOLD、pair_count=0，不能检测排序反转或证明continuation无偏差，closed与realized两个world的请求分母不同。330episode spacing违反/uncertified均0，但能源失败真实存在。仅3个初相场景、同profile family新seed、单一模型seed5401；20并发墙钟不证明实时性，有限池FCFS闭合值不等于长期Q。

保留现有结论，后续研究另定范围，不自动新增实验或扩模型。阶段1–4历史计划和证据保留。

## 阶段4已完成事实

[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)的实现、冻结数据与训练、108context离线评价及162个完整episode均已完成，主审已完成研究判断。**rank存在可学习的离线排序信号，但尚无优于同预算beam筛选的综合证据；学习必要性仍未证明。** 完整分母、表格与边界见[结果报告](evidence/fixed_terminal_stage4/results.md)，设计见[方法](arrival_restructure/stage4_learning_scheduler.md)和[协议](experiments/fixed_terminal_stage4_protocol.md)。

### 阶段4执行事实

- 新增10项测试通过（3.28 s），阶段1–4合并67项通过（12.64 s），均exit0，生产代码未再改变。pilot4001 low/medium只作流程检查：25根，2epoch且同一pilot复用train/val，smoke_only，不能作性能结论。
- train/val共36数据任务全部exit0：train108context/1117根/64informative/22单根/3332非tie pairs；val36/384/25/2/1032。两模型各固定seed4401训练300epoch一次，wall30.0575322 s、exit0；value选17、rank选35，val informative召回均24/25，仅用于选checkpoint。
- 冻结后首次test生成27场景、108context/1012根/60informative/30单根/2813pairs，随后完成六方法162episode。189任务于UTC 06:52:40.609609至09:17:13.626789全部exit0；独立offline评价exit0、6.014 s。四模型文件hash不变，无实验重跑或补采样。
- 物理、安全、能源、SOC与负载设置保持不变。全部方法执行spacing违反与未认证均0，最低着陆SOC0.06067783709660538；每方法仍有8–9例能源失败，不能写成零失败或硬件安全认证。

实现版本`08235fb476f4bf483fecd71fdec5dfe9a8e1778e`，原运行工作树`c0f02fbcc5dd078c34c8211bd47b71bb98caebce`，各日志保留实际workspace_head。当前分支在主目录`D:\7_codespace\terminal`；raw保留在`.worktrees/arrival-restructure`的detached `d511438`目录。分支归属调整没有改变实验实现或模型字节。

### 阶段4研究结论与保留边界

informative离线rank top3命中54/60、beam48/60，但加共同anchors并精确评估后两者保优均55/60。在线每法795请求：rank N/E/F/C=439/16/9/331，full438/17/8/332，beam436/17/8/334；rank多1正常不能抵销对full多1失败。rank对beam实际J为9赢/11平/7输，尚未形成综合支配。

rank累计在线决策5659.16 s、beam3575.93 s，>5 s分别274/1296与208/1296；状态轨迹、并发竞争与退化比例不同，不解释为纯网络因果贡献。仿真时钟不计CPU等待，不能据服务结果宣称实时可用。标签仅近似已到达闭合FCFS continuation，非真实未来Q；单一模型seed也不证明训练稳定性。

当前工程参考优先保留full_reference与beam_topk，rank保留实验性可学习信号。阶段4已完成；阶段5按本页列明的已授权范围推进，不扩展GNN/attention、PPO、offline RL、teacher动作模仿或动态几何。

## 保留的既有证据

[阶段3结果](evidence/fixed_terminal_stage3/results.md)：15context、209root、57项合并测试通过；15个FCFS逐请求重放一致，执行安全违反/未认证0，着陆最低SOC0.0602558271。三个t60为单HOLD退化对照。until240对FCFS实际词典序6赢/8平/1输；13可行/20不可行remove-one；闭合与实际未来存在排序反转。学习必要性仍未证明，旧数据只作动机，不用于阶段4训练。实现版本`5a67b9ac6d8421ac31172a00ce0de851ed0a4c14`，文档证据版本`3fb624b80b93ffc11ecbc8761d4309bbe7a89e74`。

[阶段2结果](evidence/fixed_terminal_stage2/results.md)保留162场/12有限oracle：FCFS444正常/7失败、deadline445/7、joint429/9、rolling441/9；rolling不支配FCFS。以上历史测试不作为阶段4通过证据。阶段1固定物理合同、SOC、负载和原阶段证据不变。
