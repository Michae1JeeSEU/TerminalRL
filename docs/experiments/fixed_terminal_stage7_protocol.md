# 阶段7冻结协议

授权来源：[用户阶段7计划](../fixed_terminal_stage7_learning_to_search_plan.md)。方法定义见[方法](../arrival_restructure/stage7_learning_to_search.md)。阶段1–6历史结果保持原样，阶段6停止扩展学习排序是本阶段出发点。

## 冻结参数与任务矩阵

- Catalogue：8holders、每holder cap8、scan48；传统参考partial beam128，top12+HOLD/liveFCFS，单stage≤14根。
- train seeds4101–4103，val4201，标准9load/SOC×4times(0,5,15,90)：36scenariojobs、108train/36valcontexts。从阶段4冻结snapshot重新插桩，旧日志无partial trace。每context真实记录partial widths3/5/8/128，trace_widths=[3,5,8,128]；共用该decision的catalogue/check cache，只为full128最终≤14根运行闭合label，窄宽度不额外rollout。以有限full最佳真实叶lineage prefix标注各stage祖先贡献，anchor不伪造树路径；插桩真实成本全部计入。
- 模型：单GradientBoostingClassifier，100/depth3/lr.05/seed7401；一次fit，threshold.5；train/val信息与feature schema及所有aux hash入manifest，不访问test调参。
- pilot7001，test7301/7302/7303；33正式scenarios×8arms=264episodes。
- panel15scenariojobs×2times(0,60)×8arms=30states/240rows。joint_window前缀；旋转方法顺序，独立catalogue生成，允许各方法pool不同；单worker、数值库threads1。
- episodes workers20、threads1；panels workers1且与其他phase不重叠；data按独立scenario并行，不改变随机流、snapshot与标签定义。

## 执行与产物

服务器仅运行，源码只在本地修改并经Syncthing同步；服务器Git或手工源码传输不属于流程。服务器输出根`/home/user/training_outputs/terminal/stage7_learning_to_search_20261014`，同步本地`D:\7_codespace\training_outputs\terminal\stage7_learning_to_search_20261014`。

Dispatcher使用参数列表、显式项目cwd、线程环境1；每output-root互斥phase，独占state/log/output，重复启动拒绝，无自动retry。记录source SHA（TERMINAL_SOURCE_COMMIT）、UTC时间、PID、每job实际exit及预期产物。状态completed须同时exit0、完整产物和模型前后hash一致；百分比不能单独证明成功。

Data每context保存snapshot/context/trace/features/labels/leaves，node trace为JSON/JSONL。训练只执行一次独立CLI；分析只读取train/val。Episode保存frozen_input、decisions/events/requests JSONL及result；panel保存原snapshot、各arm原始trace、summary及full_values。节点与leaf/root、精确闭合value/certificate保持可追踪链接。

## 质量、计算和有限参考

质量按配对每场F不增、aggregate E不增、aggregate N/全部请求比例不降（epsilon0）、间隔违反/未认证0与最终执行验证通过。保留所有normal/emergency/failed/censored结局、全部J分量、正常条件delay/energy，不能把真实能源失败写成零。

主计算接受条件`primary_compute_condition_passed`要求共同非退化同状态的实际candidate_checks与完整wall均至少降10%；共同退化状态由full_reference单根定义，方法自身单根指标另外保留。`compute_condition_passed`为该主条件别名，主条件只在nondegenerate分层判断。累计unique exact rollout与wall的旧双10%条件保留为辅助`rollout_wall_condition_passed`，不作为阶段7唯一门槛。expanded_nodes、unique_rollouts及candidate_checks各自报告ratio、总计与均值，不能把主条件成立解释为所有搜索资源同时降低。完整wall包括初次与回退所有catalogue生成、partial展开和checks、features、inference、rollouts、最终检查/submit、应急、bookkeeping；加载、IO、环境step及事后analysis labels单列。缓存仅同decision同方法内部。对5传统reference分别输出指标；重点比较beam_adaptive判断学习增量，full_reference作为次级参照，不把相对full收益直接解释为学习增量。

full_reference仅有限catalogue上的128宽度参考，不能称为global optimum。事后精确值标注学习新root、是否超过有限reference最佳和reference最佳保留率；不能据事后full值触发在线回退。C类现象不自动授权长期Q或扩大value路线。

严格summary禁止用allow-incomplete作为正式交付：核对264/15/30/240、数据train108/val36、split、modelhash、重复/缺失、请求守恒和最终执行检查。完整结果包保存用户计划、重要文档、源码/tests、raw/训练/模型及manifest引用aux；finalcommit须为真实本地HEAD，ZIP独占创建并检查CRC/entry SHA256。

## 当前执行状态

Pilot已通过流程核验；正式data attempt1已终态，33/36成功、3val失败，132/144context，原件归档完成，source验证修复已通过主审及33项检查。analyze、attempt2、正式模型训练、episodes及panels尚未启动；实际失败及恢复状态见[运行状态](../evidence/fixed_terminal_stage7/run_status.md)。

训练只有一个class时，记录实际positive/negative计数，并诚实使用constant DummyClassifier诊断回退，一次fit与冻结hash仍保留；非单class使用预定GradientBoostingClassifier固定参数。该回退不能报告为学习模型存在有效信号，最终研究解释由主对话依据实际证据给出；不额外重训或根据test调整。

## 主计算条件预先确定的理由（2026-10-14）

该口径由主对话在正式data、训练及test开始前确定，未读取当前pilot功效，也不依据test。阶段7检验真实partial展开/剪枝，候选检查可能显著减少而最终rollout数量持平；因此使用candidate_checks+完整wall作为primary，保留rollout+wall作辅助。COMPUTE_REDUCTION_TARGET=.10仍为描述性阈值。质量条件保持配对每场F不增、aggregate E不增、aggregate N比例不降（epsilon0）、间隔/未认证0和最终执行验证通过；模型、搜索、训练、seed、物理与任务矩阵均不变。当前pilot继续使用`ee6d`源码；本地接受口径修订不改变该运行。正式来源提交在pilot终态之后另行记录。

## source artifact一致性检查修复范围（2026-10-14）

[全36场景对照原件](../evidence/fixed_terminal_stage7/source_scenario_comparison.json)显示config、times、类型与离散字段exact，仅三val场景的4个SOC float末位不同（max1.1102230246251565e-16）。主对话限定修复的旧generator/source比较仅允许同类型且finite的`requests[*].soc`绝对差≤1e-12、relative0，其他scenario字段及config全部exact；实际执行严格采用stored原SOC/场景并保留原source字节，不以Linux重建场景替换。该规则是source artifact兼容性验证，非物理容差、非test调参；旧snapshot artifact float容差仍原规则，物理、搜索、模型、数据种子与任务矩阵不变。

Attempt1已归档到输出ROOT/data_attempt1，保留33成功与3失败、旧data/state/progress、全部36job日志及launch logs。源码修复已通过主审及data/models33项验证（8.09 s、exit0）；提交推送和Syncthing核对后，只启动一次完整36任务重放；严格核对144context后执行分析，再进行Step7.1主审及唯一正式fit。当前归档已完成，提交推送、Syncthing核对与attempt2尚未完成，不将132context当作正式训练完整证据。
