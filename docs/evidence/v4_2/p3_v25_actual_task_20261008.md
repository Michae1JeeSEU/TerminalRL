# V2.5 真实P3任务数据与训练衔接（2026-10-08）

## 改动依据与边界

用户授权持续分析、修复、验证至新的P3正式训练就绪，并启动正式训练；最后正式训练不设heartbeat。本文衔接`p3_v25_formal_readiness_20261008.md`，不继承旧checkpoint的资格。

之前direct C²及未来合流成员修复针对物理基础，`p3_integrated_repair.build_config`刻意关闭task priority；其结果支持基础完整性，不能代替P3优先级任务。primary核对execution_plan与既有P3配置后新建显式`configs/p3_v25_task.py`，保留全部已修复物理/安全/能量/应急/观测/K3契约，恢复priority 1/3/5、mission概率(.45,.25,.10,.20)、SOC uniform .55–.90、downstream_priority、priority urgency权重1和SOC urgency权重2/阈值.65。原基础profile不变。

正式任务场景为flow6（空启动、10+60i秒入场、1210tick完整上限）、dense24（12架warm 6/4/2、另12架每15s到达、1800tick）和flow12（空启动、10+60i、1570tick）。这些到达计划与历史正式任务一致，不因修物理而改成全部warm或高SOC。

当前105D ego保留自己的动态优先级、SOC、mission与服务信息；原五个拓扑邻机槽包含priority_delta，附加scheduler context提供同接口竞争数与执行顺序前驱数。全邻机50D提供几何与承诺轨迹。此次无需把未知未来动作/teacher标签塞入观测，也不预先宣称这些输入足以泛化；真实任务拟合与student闭环将提供判断依据。

## 先验冻结的最小数据范围

输出根`/home/user/training_outputs/terminal/p3_v25_task_20261008`。基础train_A893301已完整结束；其64标签只保留为task-neutral桥接证据，**不并入本真实P3 train/Q**。原893302/893303与旧Q895301–3未消费。以下新seed在首次reset前冻结；本地精确值预检无历史使用，服务器目标根不存在，无旧采集进程；每次启动仍检查源码同步、路径/协议身份。

|用途|seed|场景|目录|
|---|---:|---|---|
|development smoke|892808|真实P3属性，3/warm3，2tick|smoke|
|train 1|893401|flow6|train_flow6|
|train 2|893402|dense24|train_dense24|
|train 3|893403|flow12|train_flow12|
|Q 1（仅保留）|895401|flow6|Q_flow6|
|Q 2（仅保留）|895402|dense24|Q_dense24|
|Q 3（仅保留）|895403|flow12|Q_flow12|

新collector `training.p3_v25_task_data`，协议`p3_v25_actual_task_data_20261008_v1`。按表顺序一次一臂，primary完整分析再推进。环境最多各完整场景ticks/3600s，2048个全部actor pre-state（tick/数值UID顺序）/51200 RELEASE意图，标注7200s、上下文重建900s；多LZ plan调用单列。超限保留INCOMPLETE，不续标、不复跑、不挑成功前缀。smoke环境180s/标注180s，只验输出闭环，不进train。

单机teacher仍逐意图完整生成/证书检查、首可行的最小干预请求，非联合批准或全局最优。原raw teacher_label保持null、forced不入actor loss，只有独立OK标签可监督；规则behavior、teacher请求与实际execution分开。原始90D与105+N×50从同可信pre-state重建。新loader严格检查profile/真实任务参数、partition/seed/完整周期/context字节，分别提供train和只读Q解码，拒绝task-neutral旧数据与Q混入train。

每臂检查完整N/E/F/D/U、actual safety/checked-executed hash/能量/技术censor，原始行/mask/全部actor-selected-OK label集合和标签证书；统计三接口、priority、SOC、mission、资源、retry与WAIT→RELEASE覆盖。E须已提交并完成且单列，不新增零E门槛；F/D/U、技术/实际安全/hash或标注不全阻止后续臂，最小诊断后再改。rule访问分布和三个周期不会自动赋予student泛化资格。

## 接下来的训练顺序

三个train完整后先依据训练内残差做固定预算fresh imitation（预留初始化91901/91902/91903，不选最好seed；沿用5000updates/300s、Adam3e-4、batch128、clip.5），需要的student分布补充只进入新train/development。真实训练前将具体输入和有界方案写入；此处不消费Q。旧I1中先收Q才能fit的入口不直接执行，新的真实任务入口须独立命名，保留旧方案未消费状态。

候选固定后才采集/一次性评估独立Q；沿用已记录的WAIT recall .90、RELEASE .70、macroF1 .80、proposal top1 .85/top3 .95、joint .85等阈值，不按Q结果降低。其后teacher-free完整任务周期决定student-only资格；有界真实critic初始化和PPO更新验证requested likelihood、冻结proposal、KL回退、安全/完成与退化表现。任何不通过均由primary分析具体证据，再修复，不直接跳到formal。Q失败保留，不加入train或重复选择。

最终正式P3使用此真实任务profile、合格的新source、独立训练seed与fresh optimizer，按任务要求事先冻结周期/评估/停止规则；旧formal资格不能继承。正式协议与具体命令在这条验证链完整后落盘，不在此假装已准入。前序长任务60min heartbeat正常静默；真正formal启动、核对PID/cwd/argv/初产物后删除前序heartbeat并结束，正式运行不建heartbeat。

## 实际执行证据

这里仅追加已发生结果。此协议写入时新smoke/train/Q未reset，无新的BC/PPO，I0_QUALIFIED=false/PPO_ALLOWED=false。

任务profile/数据入口/旧接口定向28项local/server tests全部通过（66.43s/34.20s），四个新文件两端SHA一致。覆盖真实priority影响观测/仲裁/reward、dense 6/4/2分配、空启动12tick延迟入场与首tick重放、新任务标签→105+N×50重建、原始null与partition隔离，以及旧forecast/Q接口回归。两个test fixture仅development用途，不消费正式seed。下一步提交此版本后仅启动独立892808 smoke；尚无真实train或Q结果。

代码/协议已提交a8c821be。独立smoke892808为SMOKE_OK/exit0，2ticks/6raw/3actor，3/3原teacher RELEASE proposal12均获证/执行；三接口各1，priority1两行/3一行、三行SOC均<.65。原null/forced mask、90D→105+2×50/标签join及首tick重放全部通过，无实际安全/hash/能量异常。此随机小样本没有priority5或非standard mission，不声称覆盖充分；真实属性由配置、定向tests及后续完整回合共同检查。result SHAd076f4197c7d2e2ed64095a7b5af35c33b7a9f8418f08a4e1da8ca3d4cf53057，context SHAd1f9341594650a6c029cae624d4d683ef2b34b3c7b75444edc3cf81ae5815199，终态已同步/无残余进程。primary全量分析器输出SHA fd90fdd7af95ed8f80cb722dac6634748e530ee50387f3eec0ec4dfa3bf4331f。接受仅启动train_flow6，禁止重跑smoke。

primary另建`training/p3_v25_task_imitation.py`，仅为后续入口准备：fit只能读三个真实P3 train，固定final/全部三seed，完全不打开Q；Q需在候选冻结后由独立evaluation-only loader读取，一次性标记消费，评分0参数更新，保持student/PPO未准入。旧I1文件和未消费895301–3不动。3项本地/服务器合成测试通过3.03s/1.20s，包括无Q访问、冻结checkpoint字节和重复Q拒绝；测试没有实际train fit、正式Q采集或环境reset。所有真实学习仍等待三个train完整分析后再执行。

### train_flow6终态（newly_run）

893401已COLLECTION_OK/exit0，692ticks、6N/0E/F/D/U，所有到达/服务结束，技术/censor/实际安全/hash/负能量异常0；1896raw=20actor+1876forced，20/20标签=19R1W，五种proposal12/22/13/7/14，三接口7/7/6。actor中priority1/3/5为11/3/6，standard/emergency/cargo为9/3/8（任务名emergency不等于应急落地）；SOC<.65为14行，<.55为6行，32条基础臂retry统计不混入本臂，本臂after-retry2次、WAIT→RELEASE1次。最小实际/扫掠81.089035/81.053866m、最低完成SOC .401436、应急提交0。环境48.583s/标注78.668s。

primary全量行/teacher证书/原候选与执行/selected快照hash/105+N×50 join断言通过；一条teacher proposal13对规则nominal的执行WAIT差异保留，另有teacher proposal7（速度残差−.5）而规则实际速度残差−1的候选差异，不改标。result SHA165fb4f0d4f8eb3dbde4b05b713b758500408ae7a3f82017383af589ad21f608，context SHA39e77e92ed97938d25d3a7068feaf9deb76da1cfcea17880e06469bdc9f22180，终态两端已同步；Python2819036/timeout2819035及child退出。分析文件SHA b713d9cf44afe99332cd2e6c7606646d3c13523fcdaee6328cad9351b9be61b4。接受继续唯一train_dense24；flow6禁止重跑。仍未实际BC/Q/PPO。

后续真实PPO记录准备：primary在未被当前collector导入的`p3_v25_rollout`中保存完整physics config、每tick原requested likelihood/mask/K3执行trace和奖励分量，并核对各分量之和等于team reward。新增`p3_v25_task_metrics`只读取既有completed_uavs，统计priority加权等待、按初始SOC<=.65分组的等待以及N/E分项；不把服务完成混为触地时刻，不按终态SOC重分初始低电量组。14项local/server回归通过12.90s/5.43s（一个既有test tensor-to-scalar warning）；物理/teacher/当前采集核心未改，无真实PPO更新。这是为后续实际任务验证补齐结果口径，不是准入结果。

train_dense24已启动：Python2819670/timeout2819669，真实cwd=/home/user/projects/terminal、argv指向新task_data/index2；protocol 893402/mixedSOC/priority任务/24warm12已核对。250tick/4286raw/124selected/4N/340.869s及steps、progress、first产物仅启动前缀，尚无exit/完整标签。当前一次一臂核心冻结，不重复启动；完整分析后才继续flow12。

后续teacher-free学生开发入口`training/p3_v25_task_student.py`已准备：只加载新任务fit中预定canonical91901，不挑选最好的seed；记录来源/真实profile后执行完整任务，以自然终态/实际安全/重放/无FDU为继续依据，E安全完成单列。零优化器、零teacher，保存tensor records及原请求trace，可供同一合格source后续critic初始化使用；不会自动授予student/PPO资格。6项local/server开发fixture tests通过15.33s/8.43s，包含原runtime/K3/likelihood回归、三接口smoke和禁止构建Adam。尚无真实student任务或新的模型权重。

进一步核对统计定义发现：现有internal_wait仅在未承诺的ring hold推进中累加，已承诺轨迹的前段环上等待/长飞行不进入该项。因此metrics显式保留此口径，同时新增priority加权request-to-touchdown/request-to-completion与初始低SOC组request-to-touchdown，后续不能仅凭旧waiting下降宣称整体延误改善。没有修改环境计时、调度优先级、奖励或运行中采集。新增字段2项本地tests通过.23s；worker服务器实际重跑含PPO的14项通过5.38s（原指令只需2项metrics，多做的12项不重复计作新用例）。
