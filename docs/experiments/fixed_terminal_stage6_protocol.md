# 阶段6冻结评价协议

更新日期：2026-10-13。范围依据[用户阶段6计划](../fixed_terminal_stage6_learning_assisted_scheduling_plan.md)，具体方法见[方法](../arrival_restructure/stage6_learning_assisted_scheduling.md)。本页冻结数值保持不变。正式264episode、15panel jobs及严格汇总已完成；研究判断见[结果](../evidence/fixed_terminal_stage6/results.md)，不以pilot支持效果结论。

## 冻结输入与矩阵

沿用阶段1物理配置、固定93维特征、上限14根池、FCFS闭合J，以及阶段5四模型manifest与文件hash；只加载mlp_rank和gradient_boost作排序，不重训。正式test seeds为6301、6302、6303，pilot为6001。8arms固定为beam_b3、beam_b5、beam_b8、full_reference、beam_adaptive、rank_adaptive、gb_adaptive、hybrid_adaptive；TOTAL自适应预算B3→B5→B8→full，unique anchors计入预算，缓存避免重复根rollout。

正式33场景包括27个standard（3负载×3SOC×3seed）与6个medium/medium初始占用变体（各seed的initial_sparse/initial_dense）。稀疏/密集初态位置与人数分别取同seed下low/high初态，占用(2,2,3)/(6,8,9)。前min(变体初态数,15)个初态SOC沿用medium基准初态；dense新增8个SOC使用独立Random(stream_seed(seed,'stage6-extra-initial-soc'))及同profile拒绝采样截断正态，避免复用medium未来请求已占用的SOC draw。future arrival time和SOC完整保持medium基准，最终重新分配唯一ID。scenario_metadata记录初態来源、占用、额外SOC独立流及未来配对方式。各方法共享冻结完整场景，禁止按结果改变物理、种子或场景。

episode阶段264独立jobs。panel阶段15场景jobs：standard的low/high、medium/medium、high/low各3seed，加6个初态变体；每场景joint_window前缀t0和t60，共30面板×8方法=240行。arm顺序按完整33场config索引×2+time_index轮转。所有方法克隆同状态、独立生成相同池、无跨方法缓存；full值仅作事后标注。panel workers=1，各数值库threads=1；episode建议workers=20，实际值由服务器资源预检确定，两phase不重叠。

## 预先冻结的判断口径

对rank_adaptive、gb_adaptive、hybrid_adaptive分别与beam_b3、beam_b5、beam_b8、full_reference、beam_adaptive逐一配对。epsilon_normal=0：聚合正常完成比例不低于参考，每一对应场景F不增加，聚合E不增加，全部请求守恒，安全间隔违反/未认证均为0，最终执行验证全部通过。请求分母包括N/E/F/C，不能删除失败或截尾。延迟以正常完成N为条件分母；暴露、能耗、最小着陆SOC及所有J分量完整报告。通过该条件不意味着延迟、截尾或能耗均无损。

计算门槛：同状态非退化面板累计unique rollout和完整decision wall总和同时不超过参考的90%。该10%为描述性阈值，不作统计显著性宣称。另报所有面板、单根面板、逐state比值及各方法独立episode轨迹成本，不让单根稀释非退化差异；零参考分母比值为null，不能判为节省。

报告按standard、initial_sparse、initial_dense及负载/SOC分组，分组相互重叠，不能跨轴相加。报告停止阶段、扩搜率、全池耗尽率、证书率、累计真实unique根数、池规模和完整组件时间。学习贡献单独对beam_adaptive判断。A/B/C最终解释由主审在正式结果后完成，本协议不预写成功或模型必要/无必要。

## 输出与完整性

`episodes/<arm>-<load>-<soc>-<seed>-<variant>/`保存frozen_input.json、decisions.jsonl、events.jsonl、requests.jsonl、result.json。
`panels/<load>-<soc>-<seed>-<variant>/`保存frozen_input.json、generation.json；t0/t60各保存snapshot.json、8个arm JSON、full_values.json、summary.json。full_values中的analysis_label_cost排除于方法计时。
根目录execution_episodes.json与execution_panels.json保存264/15 jobs和退出状态；progress与日志沿阶段5结构。decision adaptive含enabled/stages/stop_reason/fallback_count/quality_certified/visible_counts；各stage保留target_budget/new_root_ids/cumulative_count/chosen_root_id/certificate/elapsed_seconds。budget为累计真实精确根数，configured_budget为原始配置预算。

默认严格汇总检查正式数量、唯一性、finite、配对输入、请求守恒、决策成本和模型hash一致。仅显式`--allow-incomplete`接收标记fixture/pilot的部分产物，输出状态不得completed。汇总写summary.json及episodes/groups/paired/same_state/fallback五类CSV。打包默认dry-run；只有正式矩阵、严格汇总、冻结模型一致且最终提交为实际HEAD才能build，禁止覆盖。包包含当前必要文档、用户阶段1–6计划、阶段6全部raw、阶段5四冻结模型、manifest及其files引用的全部安全辅助文件、当前fixed核心/config/scripts/tests和依赖，不包含旧阶段4/5全部raw；ZIP核验CRC与每entry hash，包外保存validation.json和zip.sha256。
