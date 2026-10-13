# 阶段4学习辅助调度：结果与结论

**阶段4实现与冻结实验已完成。rank存在可学习的离线排序信号，但尚无优于同预算beam筛选的综合证据；学习必要性仍未证明。** 60个informative test context中，rank原始top3命中54个，高于beam的48个；加入共同HOLD/FCFS anchors并精确评估后，两者都在55个context保留全池最佳key。在线结局和完整成本存在权衡，不能用推理很快或正常数增加掩盖失败与截尾。

阅读顺序：[方法](../../arrival_restructure/stage4_learning_scheduler.md) → [冻结协议](../../experiments/fixed_terminal_stage4_protocol.md) → 本报告。学习只近似已到达机群在FCFS continuation下至240 s的闭合参考，确定性层继续负责生成、安全、能源、容量和pad。[阶段3闭合与实际未来排序反转](../fixed_terminal_stage3/results.md)仍约束解释；full_reference不是全问题oracle或真实未来Q。

## 版本与执行范围

实现提交`08235fb476f4bf483fecd71fdec5dfe9a8e1778e`；正式生成/运行记录中的工作树提交`c0f02fbcc5dd078c34c8211bd47b71bb98caebce`，各日志保留实际workspace_head。27个test数据任务与162个完整episode共189任务全部exit0，UTC 2026-10-13 06:52:40.609609至09:17:13.626789；独立offline评价108context，exit0、6.014 s。四个冻结模型文件hash未变，没有补采样、调参或实验重跑。

运行期间仅调整checkout分支归属：当前`codex/arrival-restructure`在主目录`D:\7_codespace\terminal`，原运行目录`.worktrees/arrival-restructure`在`d511438`保持detached。该操作未改变实验实现或模型字节；`33eff25`另记录用户新AGENTS流程。原始数据留在原运行目录，不将当前文档HEAD冒充运行版本。

最终新增10项测试通过（3.28 s），阶段1–4合并67项通过（12.64 s），均exit0，覆盖未再改变的生产代码。pilot4001 low/medium四时刻14/9/1/1根；2epoch模型复用同一pilot train/val，smoke_only，只验证流程。模型重载预测一致，CLI与finite序列化检查完成。预检`tmp_stage4_preflight/validation.json`及日志单独保留；早期失败日志部分被覆盖、仅恢复工具输出摘录的限制如实保留，摘录不是完整原日志。

## 正式train/val数据

36任务全部真实exit0，2026-10-13 06:36:12.519798至06:49:41.493900 UTC，总执行墙钟约808.974 s。最多4并发，OMP/MKL/OPENBLAS线程各1；每任务为九profiles中的一个seed场景，joint behavior在0/5/15/90 s决策前取快照。原样保留退化状态。

|split|seed|场景|context|根|informative|degenerate|nondegenerate|非tie pairs|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|train|4101、4102、4103|27|108|1117|64|22|86|3332|
|val|4201|9|36|384|25|2|34|1032|

总144context、1501根。informative表示池内至少两种真实comparison key；degenerate表示单根，其余为nondegenerate，不能混为同一分类。以上是数据分布事实，不因分布补采样。

2293个正式JSON严格finite解析通过；固定seed分组、物理配置、四时刻、唯一context、93维名称、根/特征/标签顺序与维度核对通过。snapshot只含已到达非pending请求，全部保存根的闭合rollout状态feasible。[紧凑inventory](trainval_inventory.json)保存逐context来源、根数和分类；原始记录在`tmp_stage4_results/data`。

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

## 离线test：排序信号与筛选后保优

主要分母是60个informative context、2813个非tie pairs；全部108个context来自27个新seed场景的共同joint behavior快照，并非108个独立场景。top1/top3以context计；pair accuracy以非tie pair计，包含固定action_key tie-break形成的最终诱导排序。top3在加入anchors前计算，筛选后比较在共同anchors和精确rollout后计算；真实key沿用整数/round8/round10。

|组（context数）|方法|pair accuracy|top1命中|top3命中|筛选后对全池赢/平/输|
|---|---|---:|---:|---:|---:|
|informative（60）|beam_topk|0.636331|37/60|48/60|0/55/5|
|informative（60）|value_topk|0.590828|40/60|44/60|0/50/10|
|informative（60）|rank_topk|0.830786|48/60|54/60|0/55/5|
|all（108）|beam_topk|0.636331|85/108|96/108|0/103/5|
|all（108）|value_topk|0.590828|88/108|92/108|0/98/10|
|all（108）|rank_topk|0.830786|96/108|102/108|0/103/5|
|nondegenerate（78）|beam_topk|0.636331|55/78|66/78|0/73/5|
|nondegenerate（78）|value_topk|0.590828|58/78|62/78|0/68/10|
|nondegenerate（78）|rank_topk|0.830786|66/78|72/78|0/73/5|
|degenerate（30）|beam_topk|—（0 pairs）|30/30|30/30|0/30/0|
|degenerate（30）|value_topk|—（0 pairs）|30/30|30/30|0/30/0|
|degenerate（30）|rank_topk|—（0 pairs）|30/30|30/30|0/30/0|

rank排序正确率0.830786高于beam的0.636331，但最终保优同为55/60；value为50/60。因此原始排序改善没有转化为优于同预算beam的最终保优率。单根30个context无需在线推理/rollout；另18个非退化context中所有真实key相同，不能用这些容易保优的状态替代informative结果。

value反标准化ΔJ的逐分量MAE如下。按组内所有候选等权；前四列单位为请求数，T/D为秒，energy为归一化SOC。J中计数的正负号不影响绝对误差。MAE使用未取整预测，排序时前四维另按Python round（.5 ties-to-even）处理。

|组|候选分母|F|N|E|C|T（s）|D（s）|energy（SOC）|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|informative|730|0.091449|0.716611|0.545472|0.404180|97.649984|81.086218|0.073861|
|all|1012|0.070799|0.583421|0.468949|0.328431|78.549318|68.488080|0.060622|
|nondegenerate|982|0.072554|0.593811|0.478966|0.335426|80.465637|70.007303|0.061980|
|degenerate|30|0.013372|0.243329|0.141056|0.099441|15.821796|18.758821|0.016168|

[offline_metrics.json](offline_metrics.json)保留全部组、逐context根ID及raw regrets。raw regret是full减filtered的逐分量差，既有带符号J差也有物理分量差，并非逐维非负标量；原始浮点微差保留，胜负按冻结comparison key判定。不能把同池闭合误差当成真实未来价值误差。

## 完整episode：全请求结局与配对比较

每方法为相同27个新场景、795个请求，观察窗均240 s。N/E/F/C为正常着陆/应急着陆/能源失败/截尾，四项闭合至795。应急比例分母为全部795请求；正常延迟是完成者条件均值，各方法完成的人群不同。暴露与能耗为全请求总和，不能只用正常延迟或耗能较小宣称整体改善。

|方法|N|E|F|C|E/795|正常条件延迟（s）|暴露总量（请求·s）|能耗总量（SOC）|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|fcfs|429|21|9|336|2.642%|70.99998|85210.38661|72.060721|
|rolling_horizon|424|54|8|309|6.792%|68.76743|81084.37934|68.561155|
|full_reference|438|17|8|332|2.138%|69.64524|84302.36318|71.210866|
|beam_topk|436|17|8|334|2.138%|69.63134|84639.35389|71.580127|
|value_topk|433|19|8|335|2.390%|71.35306|85422.87977|72.137958|
|rank_topk|439|16|9|331|2.013%|69.98140|84663.67793|71.536181|

六方法执行间隔违反与未认证均为0，所有着陆记录最低SOC为0.06067783709660538，高于reserve0.06。这不抵销每方法8–9例能源失败，也不代表硬件安全认证。rank比full多1正常，也多1失败；冻结J优先减少F，不能用正常数抵销失败数。具体在medium-load/low-SOC/seed4303，rank为17/0/1/7，full与beam均为17/1/0/7；应急总数减少不自动等于改善，此处只记录观测，不推断根因。

下表按27个场景逐一比较实际episode J的赢/平/输，不用跨场景总数替代配对，也不把词典序win称为Pareto收益。

|方法|对full_reference|对beam_topk|对FCFS|对rolling_horizon|
|---|---:|---:|---:|---:|
|beam_topk|4/13/10|0/27/0|18/7/2|17/3/7|
|value_topk|4/8/15|6/11/10|19/6/2|17/1/9|
|rank_topk|3/16/8|9/11/7|21/4/2|19/3/5|

rank对beam为9赢/11平/7输，不能视为综合支配；value为6/11/10。实际episode中筛选法偶尔赢full不与离线同池结果矛盾：各策略改变后续状态，未来请求继续到达，full仅对当时有限候选池的闭合世界取最佳。[online_metrics.json](online_metrics.json)保存九个load×SOC分组和配对赢/平/输计数；完整配对raw J、分量差和来源保留在`tmp_stage4_results/summary/online_summary.json`；[online_summary.csv](online_summary.csv)包含六方法各总体及九格，共60行。

## 完整在线成本

以下是每方法1296次决策（27×48，含单根快路径及空闲）的本机墙钟；最多4并发、每进程线程1，无跨方法缓存。累计值是决策耗时之和，不是并行矩阵历时。p95采用位置0.95×(n−1)线性插值；I/O、模型冷加载和环境推进另列。FCFS/旧rolling内部工作按legacy_scheduler记录，其分项0不表示没有安全工作。

|方法|累计（s）|p50（s）|p95（s）|max（s）|>5 s/1296|
|---|---:|---:|---:|---:|---:|
|fcfs|10.067045|0.005931|0.016333|0.250359|0|
|rolling_horizon|12731.358036|3.037544|36.392660|64.803981|590|
|full_reference|6881.220955|1.565572|30.234730|47.742662|367|
|beam_topk|3575.930901|0.457973|13.623458|44.458134|208|
|value_topk|3238.706002|0.489815|11.047366|44.819065|177|
|rank_topk|5659.162100|0.777335|25.773932|44.175301|274|

完整链分项累计如下（s）。total另含少量调度封装开销，不用分项相加替换实测total。

|方法|generation|features|inference|exact rollout|final validation/emergency|legacy scheduler|
|---|---:|---:|---:|---:|---:|---:|
|fcfs|0.000000|0.000000|0.000000|0.000000|0.000000|10.062311|
|rolling_horizon|0.000000|0.000000|0.000000|0.000000|0.000000|12731.351052|
|full_reference|4198.990003|0.000000|0.000000|2659.328095|22.899689|0.000000|
|beam_topk|2892.987253|0.000000|0.000000|665.871701|17.063394|0.000000|
|value_topk|2515.958385|2.678795|0.321550|704.701598|14.991822|0.000000|
|rank_topk|4719.963425|3.171221|0.337222|916.653646|18.998174|0.000000|

单根与非退化分别列明；原FCFS/rolling没有统一池分类，不套用该分母。

|方法|组|决策数|累计（s）|p50（s）|p95（s）|max（s）|>5 s|
|---|---|---:|---:|---:|---:|---:|---:|
|full_reference|nondegenerate|918|6879.266723|3.586055|32.651824|47.742662|367|
|full_reference|degenerate|378|1.954232|0.004183|0.011211|0.018683|0|
|beam_topk|nondegenerate|878|3573.133449|1.353678|23.462226|44.458134|208|
|beam_topk|degenerate|418|2.797453|0.006310|0.013762|0.031570|0|
|value_topk|nondegenerate|819|3235.906136|1.563602|21.170710|44.819065|177|
|value_topk|degenerate|477|2.799866|0.005640|0.011133|0.041127|0|
|rank_topk|nondegenerate|912|5657.037994|1.951941|27.108563|44.175301|274|
|rank_topk|degenerate|384|2.124106|0.004854|0.011113|0.034858|0|

value/rank推理p50约0.32 ms、p95约0.48–0.52 ms，但生成仍耗时且包含候选可行性工作，不能把网络推理速度当成完整决策速度。非退化决策数full/beam/value/rank为918/878/819/912；所到状态不同，加上并发竞争，以上不是同状态微基准，累计时间差不能单独归为网络贡献。

|模型冷加载（每方法27进程）|累计（s）|p50（s）|p95（s）|max（s）|
|---|---:|---:|---:|---:|
|value_topk|46.119271|1.623973|2.012308|3.479088|
|rank_topk|42.466083|1.571090|1.678345|1.704637|

其他方法模型冷加载为0。所有分项p50/p95/max及环境推进时间均在online_metrics.json。value、rank、beam、full分别有177、274、208、367次完整决策超过5 s；离线交通时钟不随真实CPU等待推进，服务结局未包含这些等待造成的后果，不能据本轮结果宣称实时部署资格或算法固有复杂度。

## 结论与下一步边界

rank学到了可用于离线排序的信号，但共同anchors与精确筛选后的保优率未超过beam，在线结局与耗时也未形成综合优势。value有更低的本轮累计耗时，同时保优和若干服务指标较弱；这些混合结果不支持学习必要性。当前工程参考优先保留确定性full_reference与beam_topk加速路径，rank保留为实验性可学习信号。

本阶段已完成，负结果和权衡也是有效研究结果。标签仅针对指定闭合FCFS continuation；单一模型seed不能证明训练稳定性，joint behavior离线状态不等于在线策略状态，合成能源与运动学不等于硬件校准。后续若另行确定范围，可考虑候选生成或简单搜索加速；本轮不自动启动新实验、GNN/attention、actor/PPO、offline RL、teacher动作模仿或动态几何。

## 产物与复现入口

|内容|紧凑入口/原始相对路径|
|---|---|
|train/val、test分布|[trainval_inventory.json](trainval_inventory.json)、[test_inventory.json](test_inventory.json)|
|固定训练与模型哈希|[training_summary.json](training_summary.json)，`tmp_stage4_results/models`、`training_run`|
|离线分组MAE与逐context regret|[offline_metrics.json](offline_metrics.json)，`tmp_stage4_results/summary/offline_test.json`|
|162episode与配对、分项时间|[online_metrics.json](online_metrics.json)、[online_summary.csv](online_summary.csv)，`tmp_stage4_results/episodes`|
|执行、模型冻结与安全核对|[execution_summary.json](execution_summary.json)，`tmp_stage4_results/test_logs`、`test_execution_all.json`|
|原始数据、逐根标签和rollout|`tmp_stage4_results/data/{train,val,test}`|
|预检与测试|`tmp_stage4_preflight/validation.json`及原始/摘录日志|

上述raw路径在本地以`D:\7_codespace\terminal\.worktrees\arrival-restructure`为根；交付ZIP保留`tmp_stage4_results`、`tmp_stage4_preflight`等统一相对布局，docs/source从主目录取。这里使用代码路径而非指向主目录不存在目录的链接。全部raw、模型、600条训练曲线与日志进入交付包，Git只留紧凑证据；分支交接诊断不混入科研证据包。

公开CLI的`--aggregate`导出每episode简表；本报告的加权分组、pooled timing和paired统计由包内`tmp_stage4_analysis/summarize_stage4.py --execute`生成。该脚本默认路径固定，复现前须阅读并为新的输出目录配置路径，不能覆盖交付结果。

复现使用实现版本08235fb及冻结依赖，按[协议CLI与顺序](../../experiments/fixed_terminal_stage4_protocol.md)先train/val、再固定训练、最后test与六方法episode。单个test/episode和离线汇总示例（输出必须为新路径）：

```powershell
python -m terminal_marl.scripts.generate_fixed_terminal_stage4 --split test --seed 4301 --load low --soc high --models MODEL_DIRECTORY --output NEW_TEST_SCENARIO
python -m terminal_marl.scripts.evaluate_fixed_terminal_stage4 --method rank_topk --seed 4301 --load low --soc high --models MODEL_DIRECTORY --output NEW_EPISODE
python -m terminal_marl.scripts.evaluate_fixed_terminal_stage4 --offline TEST_DATA_ROOT --models MODEL_DIRECTORY --output NEW_OFFLINE.json
python -m terminal_marl.scripts.evaluate_fixed_terminal_stage4 --aggregate EPISODE_ROOT --output NEW_SUMMARY_PREFIX
```
