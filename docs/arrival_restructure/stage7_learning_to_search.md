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

质量门槛对每个传统reference分别核对：配对每场F不增、aggregate E不增、aggregate正常比例不降（epsilon_normal=0），间隔/未认证为0且最终执行验证通过。计算门槛为非退化同状态累计unique rollout及完整wall都≤reference的90%；同时报告单根和非退化分层、总和/均值、>5s比例、expanded nodes、candidate checks、全部时间分量及恢复率。重点对full_reference与beam_adaptive分别核对；并发episode wall不能解释为纯方法成本或同状态因果。A/B/C解释由主对话审查证据后给出。

不自动扩大value、长期Q、模型、teacher动作模仿、actor/PPO、offline RL、GNN/attention或改变几何。

训练只有一个class时，记录实际positive/negative计数，并诚实使用constant DummyClassifier诊断回退，一次fit与冻结hash仍保留；非单class使用预定GradientBoostingClassifier固定参数。该回退不能报告为学习模型存在有效信号，最终研究解释由主对话依据实际证据给出；不额外重训或根据test调整。
