# 阶段7 Step7.1：正式搜索数据与瓶颈记录

更新日期：2026-10-14。主对话已审查并接受Step7.1证据；唯一一次正式fit已完成冻结，训练终态与正式模型单决策API接缝均通过主审。已推送c892运行源码冻结且数据/模型/接缝主审通过，正式episodes已正常唯一启动并通过核验，与纯文档整理并行；actual EPsource仍c892，仍运行未终态，panels未启动。本记录解释已观测数据及其边界，不作A/B/C结论。

数据和分析源码均为`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`，不是后续文档提交SHA。紧凑原始字段、144context分布、collector与完整bottleneck来源hash见[data_analysis_evidence.json](data_analysis_evidence.json)；完整原始分析为输出ROOT的`analysis/bottleneck.json`。

## 完整性与执行事实

正式data attempt2于11:35:09北京时间启动，wrapper/dispatcher PID3475605/3475606；workers20、threads1。36/36任务成功、exit0，dispatcher156.193307 s、外层time156.24 s。27train/9val场景共144context（108/36），每context六个预期JSON齐全、finite、109维features与trace widths3/5/8/128一致，没有重复、source hash不匹配或stored scenario/config/times不匹配。旧source generator比较audit为33场exact、3场SOC末位差，共4处，max1.1102230246251565e-16；实际使用stored原SOC/场景。

分析于11:40:20启动，PID3476384，exit0、5.19 s，144context结果finite。首次data失败仍完整保留：attempt1的33成功/3失败、132context、state/progress、72job logs与launch logs共936files在`data_attempt1/`逐SHA/bytes核对，不以attempt2替代或删掉失败记录。详细执行事实见[运行状态](run_status.md)。

## 搜索与检查单位

每个context真实遍历传统width3/5/8/128，catalogue与检查cache仅同decision内共享；只有full128最终保留根执行闭合label。以下为144个train+val context的累计记录。

| 观测量 | 数量 | 单位与范围 |
|---|---:|---|
| 保留并闭合评估的roots | 1,501 | 有限width128参考根 |
| expanded nodes | 60,489 | 四宽度真实展开记录 |
| child/partial-check attempts | 528,613 | 包括cache命中的尝试 |
| 全部candidate_checks | 2,068,942 | 实际候选检查调用 |
| partial candidate_checks | 1,613,909 | 占全部candidate调用约78.0% |
| 全部batch_checks | 469,181 | batch调用，不能与candidate调用相加 |
| 实际uncached partial node batch-check calls | 391,098 | 排除cache命中的partial节点batch调用 |
| 其中未贡献有限full最佳叶的实际调用 | 389,371 | 占上述391,098调用约99.56% |
| cached partial revisits | 137,515 | 单列，不当作实际batch调用重复计入 |
| unknown validation | 0 | 本组已记录partial validation |

389,371/391,098的99.56%对应实际uncached partial节点batch调用；它不以1,613,909个partial candidate调用为分母。在线主计算指标仍为actual candidate_checks与完整wall；batch单位独列。

按depth记录的validated节点不贡献比例（包含cached记录）在depth1为78.36%、depth2为96.06%，depth3–8约99%。这些比例仅说明未关联到有限width128最佳真实叶的lineage prefix，不能证明被剪枝节点在反事实展开中没有价值、不可行或不属于全域最优。完整depth计数、实际调用与cached revisits分开保存在JSON。

## 训练标签与叶值分布

所有144context合计312,713个eligible节点，positive4,575、negative308,138，positive fraction1.463%；该分母包含train和val，不能写成仅train数量。类别明显不平衡，因此accuracy单独不能证明展开预测有用；本阶段不据此修改参数、class weight、threshold或数据。

144context中89个有多个叶J，55个叶J全部相同，其中24个只保留单叶。另有66个unlinked best anchors；单位是anchor，不是66个context。独立anchor没有真实树路径时不伪造祖先标签。negative标签只表示未贡献当前有限width128参考最佳叶，不表示全域无价值。

## 累计计时边界

| 进程内累计时间分量 | 秒 | 对trace-and-label累计时间比例 |
|---|---:|---:|
| generation | 803.319 | 55.13% |
| exact rollout | 558.017 | 38.29% |
| dataset features | 94.045 | 单列 |
| trace-and-label elapsed | 1,457.176 | 分母 |
| serialization | 51.381 | trace之后单列 |

这些是多个context、四种width的进程内累计计时，来自20worker数据任务；它们不是一次程序总wall，不等于隔离在线决策wall，也不能用并发episode和不同profile计时证明方法差异。完整timing字段保存在JSON，正式panel仍按一worker独立同状态执行。

## 已授权接续

主对话已接受Step7.1。唯一正式fit使用c892源码、ROOT/data/train、ROOT/data/val与ROOT/models，无smoke，现已完成并冻结；完整退出、参数、144context hash绑定、四模型文件hash及原始metrics见[model_freeze.json](model_freeze.json)。fit的初始T0取5分钟：smoke2.53 s/9,689 rows乘全部312,713 rows作为上界约82 s，再留3倍余量。协议只训练一个顺序boosting模型，没有独立副本可并行，threads1；data/episodes保持20进程，panel隔离1进程。

正式fit于11:43:40.751676北京时间启动，wrapper/Python PID3476873/3476874，exit0，wall96.33 s、fit86.591769 s、RSS2204180 kB，stderr0、无残留。模型是100 estimators、depth3、lr.05、seed7401的GradientBoostingClassifier，threshold.5、109维、fit_count1、roundtrip=true、smoke_only=false、test_read=false；val仅报告，不用于阈值、参数或checkpoint选择。

| split | contexts / eligible nodes | positive / negative | precision | recall | Brier |
|---|---:|---:|---:|---:|---:|
| train | 108 / 228,516 | 3,413 / 225,103 | .983264 | .413126 | .0067902 |
| val | 36 / 84,197 | 1,162 / 83,035 | .928962 | .438898 | .0073096 |

Val accuracy为.991793，但按类别分母，全预测negative的accuracy已约98.62%；高accuracy不能独自支持模型有用。保持原threshold与参数，不据val重训或修改数据。冻结训练记录只建立正式模型与输入对应关系，不替代在线质量、candidate_checks及完整wall评价。

正式模型在train/low-high/4101 t0仅运行一次hybrid决策的接缝已通过主对话审查：exit0、1.10 s、RSS135760 kB，109维API接受，final_check为feasible/joint_normal_batch，模型hash与原environment snapshot不变，两个输出JSON finite。原collector与产物hash保存在data_analysis_evidence.json；没有运行新fit、完整episode、panel或调参。这只验证接口与执行流程，不替代正式评价。

在已推送c892运行源码固定、数据/模型/接缝主审通过后，主对话已授权Luna并行于纯文档整理启动264episodes，workers20、threads1；初估T0约100分钟，依据pilot最慢full high/high366.67 s×264/20约80.7分钟，再乘1.25余量。15隔离paneljobs保持worker1，初估约52分钟（138.64 s×15×1.5）。11:55:32.967143北京时间正式EP唯一启动，wrapper/dispatcher PID3478486/3478487，20workers/threads1；启动核验通过，后续文档SHA不改写actual EPsource c892，尚未终态。以上仅用于排程与监控，不新增矩阵、不重训调参、不预判学习收益。
