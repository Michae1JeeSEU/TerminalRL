# 阶段4学习辅助调度：结果记录

状态：**实现、预检、正式train/val数据与两模型训练已完成并冻结；离线test与完整episode评价待证据。** 本文不提前判断学习效果。研究问题与固定设计见[方法](../../arrival_restructure/stage4_learning_scheduler.md)、[协议](../../experiments/fixed_terminal_stage4_protocol.md)；[阶段3闭合与实际未来排序反转](../fixed_terminal_stage3/results.md)仍约束解释。

## 版本与已执行范围

实现提交`08235fb476f4bf483fecd71fdec5dfe9a8e1778e`；正式生成时的协议/工作树提交`c0f02fbcc5dd078c34c8211bd47b71bb98caebce`，两者均已普通push且远端核对一致。未改变物理、SOC、负载、候选/模型预算或固定种子，没有补采样或重跑实验。

最终新增10项测试通过（3.28 s），阶段1–4合并67项通过（12.64 s），均exit0。pilot4001 low/medium四时刻14/9/1/1根；2epoch模型复用同一pilot train/val，smoke_only，只验证流程，不提供验证集性能或泛化结论。模型重载预测一致，CLI与finite序列化检查完成。完整预检记录在[tmp_stage4_preflight](../../../tmp_stage4_preflight)的validation.json及原始日志，早期失败日志仅恢复摘录的限制在记录中保留。

## 正式train/val数据

36任务全部真实exit0，2026-10-13 06:36:12.519798至06:49:41.493900 UTC，总执行墙钟约808.974 s。最多4并发，OMP/MKL/OPENBLAS线程各1；每任务为九profiles中的一个seed场景，joint behavior在0/5/15/90 s决策前取快照。原样保留退化状态。

|split|seed|场景|context|根|informative|degenerate|nondegenerate|非tie pairs|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|train|4101、4102、4103|27|108|1117|64|22|86|3332|
|val|4201|9|36|384|25|2|34|1032|

总144context、1501根。informative表示池内至少两种真实comparison key；degenerate表示单根，其余为nondegenerate，不能混为同一分类。以上是数据分布事实，不因分布补采样。

2293个正式JSON严格finite解析通过；固定seed分组、物理配置、四时刻、唯一context、93维名称、根/特征/标签顺序与维度核对通过。snapshot只含已到达非pending请求，全部保存根的闭合rollout状态feasible。[紧凑inventory](trainval_inventory.json)保存逐context来源、根数和分类；原始记录在[tmp_stage4_results/data](../../../tmp_stage4_results/data)。

只读inventory脚本首次调用因临时脚本未加入工作树import路径出现ModuleNotFoundError，修复该临时路径后汇总exit0；未修改仿真源码/数据，没有实验重跑。全部36个真实生成任务一次完成。

## 固定正式训练与checkpoint

两模型各按固定seed4401训练300epoch，共600条曲线，一次训练执行exit0、墙钟30.0575322 s；smoke_only=false、test_read=false。数据scaler仅用1117个train候选拟合；rank有3332非tie pairs、64个informative train contexts。选中epoch依据预定validation top3召回、val loss及最早epoch顺序，后续test不再改变checkpoint。

|模型|选中epoch|train loss|val loss|informative val top3命中|
|---|---:|---:|---:|---:|
|value|17|0.254021227360|0.211728215218|24/25=0.96|
|rank|35|0.188442707062|0.607461452485|24/25=0.96|

value为标准化七维MSE、rank为context等权非tie pair logistic，loss单位/目标不同，不据两列数值比较方法优劣。验证召回分母是25个informative context，既不等于全部36个validation context，也不是test泛化结果。仅一个模型seed、每类一次训练，不能声称训练稳定性或学习有效。

[训练摘要](training_summary.json)含命令、来源和模型哈希。冻结value.pt SHA256为`004d9f1f6054a1f9109ee4a3b64f337045599eb5b91577fafc77d339375b0d2a`；rank.pt为`a1c298f294143e8d2e4330d748648a2006978c0e9db4be0a32b2aa913f636c38`。完整checkpoint、scaler、600条曲线与日志留在tmp_stage4_results，进入最终包，不复制进Git。

## 正式test数据分布

checkpoint冻结后生成的27个test场景（4301/4302/4303×九profiles）已完成，108context、1012根；60informative、30degenerate、78nondegenerate、2813非tie pairs。1633个JSON严格finite解析，seed/配置/时刻/93特征及根标签对齐通过，27场frozen_models记录均与冻结四文件hash一致。只读记录见[test_inventory.json](test_inventory.json)，本统计未运行预测或rollout。

离线效果表后续优先报告60个informative context，同时保留108all、78nondegenerate和30degenerate；当前尚无离线预测评价结论，不能从数据分布推断模型效果。

## 后续结果位置与当前空缺

|证据|入口|状态|
|---|---|---|
|预检与67测试|tmp_stage4_preflight/validation.json及日志|已完成，流程证据|
|正式train/val|[trainval_inventory.json](trainval_inventory.json)|已完成|
|命令、进程、起止、exit、stdout/stderr|[正式运行目录](../../../tmp_stage4_results)|execution.json、logs/、progress.jsonl已保存|
|模型、曲线、checkpoint选择|[training_summary.json](training_summary.json)、tmp_stage4_results/models|已完成并冻结|
|test数据分布|[test_inventory.json](test_inventory.json)|已核对，仅数据事实|
|离线test排序/误差/regret|待正式产物|尚无评价结论|
|27场景×六方法episode|待正式产物|尚无结果|
|最终科学结论|本文后续补充|待完整证据与主审|

正式训练已按固定300epoch、两个模型各一次完成；checkpoint固定后，主线程安排首次test数据与完整episode评价。当前不能用pilot或train/val分布推断性能、泛化、实时可用或学习必要性。后续继续完整报告闭合参考范围、同预算beam对照、全请求结局及含生成的在线成本。
