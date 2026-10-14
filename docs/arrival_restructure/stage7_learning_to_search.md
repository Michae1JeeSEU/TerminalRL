# 阶段7：Learning to Search

本阶段由用户[阶段7计划](../fixed_terminal_stage7_learning_to_search_plan.md)单独授权。阶段6学习排序停止扩展的结论保留；本阶段检验在候选完整形成之前控制真实partial计划搜索，不复用阶段6TOTAL根排序作为新方法。

## 研究对象与信息边界

沿用固定多环几何、运动、SOC、容量、pad、确定性连续间隔与应急规则。每次决策仅读取已到达机群与固定承诺，目标仍为已到达机群FCFS闭合真实词典序J，不读取未来draw、seed/profile/method标识、真实结局或全参考标签。闭合证书仅证明当前已到达机群达到正常完成上界；不证明全J最优、未来到达安全或长期Q。

Catalogue沿用REFERENCE_SETTINGS：最多8holders，每holder最多8候选，扫描48。学习介入joint_choices层逐depth父partial的展开/剪枝/宽度。full_reference是传统partial beam128，最终保留top12并加入HOLD和同状态liveFCFS，单stage至多14根；其有限池不等于整个正常可行域。候选池随partial控制而改变。

## 八个方法

| 方法 | partial搜索控制 |
|---|---|
| beam_b3 / beam_b5 / beam_b8 | 固定真实partial宽度3 / 5 / 8 |
| beam_adaptive | 传统宽度3→5→8→128，以当前可计算闭合证书停止 |
| full_reference | 固定传统宽度128的有限参考 |
| learned_pruning | 父partial展开概率p<.5暂停展开，宽度8 |
| learned_beam_width | 不hard prune；每depth父frontier概率和≤3时宽度3，≤5时5，否则8 |
| hybrid | 同时应用以上概率剪枝与宽度规则 |

学习方法若未达到已到达FCFS闭合正常完成上界，或预测/最终验证异常，则恢复传统128。回退判据不读取full或test事后标签。每decision缓存只在该方法内部共享；累计记录首次搜索与全部回退的generation、partial checks、features、inference、unique exact rollouts、最终submit、应急和bookkeeping。每stage至多14根，但decision累计union可超过14，不把partial宽度当TOTAL rollout预算。

## 训练信号与证据

旧阶段4日志没有partial node记录。对旧train/val冻结snapshot重新插桩产生阶段7newly generated trace；每个context记录传统width3/5/8/128的真实traversal（trace_widths固定[3,5,8,128]），共用同decision内部catalogue和validation缓存，保留原snapshot、物理、候选域与split。仅full128最终≤14根执行closed rollout标签，不为窄宽度增加rollout。各stage节点以实际lineage prefix映射有限full最佳真实叶祖先贡献，独立anchor无tree时不伪造lineage；全部真实插桩成本计入Step7.1。每node记录depth、candidate数量、父节点、validation、beam筛选、学习剪枝和实际预测；leaf_links链接真实叶、闭合rollout与certificate。有限参考最终最佳叶的祖先membership构成展开训练信号。“无贡献”只表示不属于这个有限参考最佳叶祖先，不能称为全域节点无价值。

单GradientBoostingClassifier，100 estimators、max_depth3、learning_rate.05、seed7401；仅一次fit，threshold=.5，无test调参或重训。train=4101/4102/4103，val=4201；标准九load/SOC组合，每场t0/5/15/90，共108/36contexts。feature schema及模型hash随manifest冻结。

## 评价与成本

test=7301/7302/7303，pilot=7001。正式33场：9load/SOC×3seed，加medium/medium的initial_sparse/dense各3场；初态占用变体直接复用阶段6生成器及独立SOC规则。八方法共264episodes。隔离panel是low/high、medium/medium、high/low×3seed加6变体，共15jobs，joint_window前缀的t0/t60共30states、240方法行。

Panel每方法独立生成catalogue和partial pool，原snapshot、物理与catalogue域相同，不要求pool相同。方法按scenario/time旋转顺序，单线程逐个执行，无跨方法缓存。所有方法完成后才为新根与缺失chosen value补精确label，analysis_cost单列，不计方法成本；比较full_reference有限池最佳、根贡献和是否出现新高值，不宣称global oracle。

质量门槛对每个传统reference分别核对：配对每场F不增、aggregate E不增、aggregate正常比例不降（epsilon_normal=0），间隔/未认证为0且最终执行验证通过。主计算门槛`primary_compute_condition_passed`冻结为共同非退化同状态的实际candidate_checks和完整wall均≤reference的90%；共同退化分层由full_reference的pool_count==1定义，窄方法自身单根另列。旧unique rollout与wall双90%条件保留为辅助`rollout_wall_condition_passed`，不能作为本阶段唯一门槛；`compute_condition_passed`为主门槛别名。expanded_nodes、unique_rollouts、candidate_checks分别报告ratio、总计与均值，并报告单根/非退化分层、>5s比例、各时间分量和恢复率，不能据主门槛声称所有资源同时降低。重点比较beam_adaptive识别学习增量，full_reference比较为次级参照；并发episode wall不能解释为纯方法成本或同状态因果。A/B/C解释由主对话审查证据后给出。

不自动扩大value、长期Q、模型、teacher动作模仿、actor/PPO、offline RL、GNN/attention或改变几何。

训练只有一个class时，记录实际positive/negative计数，并诚实使用constant DummyClassifier诊断回退，一次fit与冻结hash仍保留；非单class使用预定GradientBoostingClassifier固定参数。该回退不能报告为学习模型存在有效信号，最终研究解释由主对话依据实际证据给出；不额外重训或根据test调整。

## 正式实验前确定计算接受口径（2026-10-14）

主对话在正式data、训练和test开始前确定上述主门槛，没有读取当前pilot的功效或据test结果调整。原因来自阶段7用户计划的真实partial展开/剪枝目标：方法可能保持同样的最终unique rollouts，却减少大量候选检查；阶段6的rollout+wall双10%条件因此保留作辅助，阶段7主条件改为实际candidate_checks+完整wall双10%。COMPUTE_REDUCTION_TARGET仍为.10，指标描述性；质量门槛、模型、搜索、训练、种子与评价矩阵全部不变。本段只记录接受口径，不给出A/B/C结果。

## 旧source场景一致性验证的限定修复（2026-10-14）

正式data attempt1的三val任务在仿真前因旧source比较失败；[全36场景对照](../evidence/fixed_terminal_stage7/source_scenario_comparison.json)仅发现4个`requests[*].soc`有限float末位差，max1.1102230246251565e-16，config、times、类型与离散字段一致。主对话已审查并接受最小源码修复：旧generator/source比较仅允许同类型有限SOC float的abs≤1e-12、relative0；所有其他scenario字段与config exact。实际仿真仍使用stored原SOC/场景，保留原source artifact字节，不用Linux重建值替换。

这一比较规则仅用于source artifact兼容性验证，不作为物理、安全、能源或研究参数容差，也不是test调参；搜索、模型、data种子与冻结矩阵保持原定义。旧snapshot artifact float比较维持原规则。源码修复已通过主审及data/models33项验证（8.09 s、exit0），c892提交同步后已完成唯一完整36任务重放、144context核验和分析。attempt1原件完整归档；Step7.1已通过主审，唯一正式模型已完成冻结并通过单决策接缝，正式episodes已正常唯一启动并通过核验，仍运行未终态；panels未启动且不重叠，执行事实见[运行状态](../evidence/fixed_terminal_stage7/run_status.md)。

## Step7.1与正式模型当前证据（2026-10-14）

正式数据/分析与唯一fit来源均为`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`。144context累计candidate checks2,068,942，其中partial1,613,909约78.0%；actual uncached partial节点batch调用391,098，未贡献有限full最佳叶的389,371约99.56%，cached revisits137,515单列，不能把batch与candidate单位混加。Eligible312,713节点为所有train+val，positive4,575（1.463%）；negative不证明反事实无价值，accuracy单独不能证明模型有用。完整depth、timing、context分布及边界见[Step7.1记录](../evidence/fixed_terminal_stage7/search_bottleneck.md)。

一次正式GradientBoostingClassifier fit与冻结核验已通过主审：100/depth3/lr.05/seed7401、threshold.5、109维、fit_count1、roundtrip=true、test_read=false，val只报告。Train108context/228,516节点，val36/84,197；val precision.928962、recall.438898、Brier.0073096，未因此改阈值或重训。[模型冻结原件](../evidence/fixed_terminal_stage7/model_freeze.json)保存四文件hash与144context SHA核对。正式模型在train/low-high/4101 t0一次hybrid决策的API、feasible、hash及原snapshot不变接缝已通过；该流程检查不替代正式264episode/15panel评价。已推送c892运行源码冻结、数据/模型/接缝主审通过，EP获授权与纯文档整理并行，actual EPsource仍c892；11:55:32.967143北京时间启动，wrapper/dispatcher PID3478486/3478487，启动核验已通过，仍未终态。冻结参数、数据与矩阵保持不变。
