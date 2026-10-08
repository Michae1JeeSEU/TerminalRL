# V2.5 当前中性课程与P3继续验证（2026-10-08）

## 范围纠正

primary上一段把`execution_plan.md`历史P3表和旧priority-aware formal配置当成当前规范，新增了priority-enabled数据分支。这是解释错误。`README.md`明确证据文件不能改写规范；`requirements.md`5.2、design/technical与execution_plan0.1/0.2一致规定当前主课程`task_priority_enabled=false`、priority shaping=0，mission只保留物理service/兼容性，等待与能量紧迫度继续参与调度。该规则不是本轮新增，也不因用户要求“正式P3”自动恢复历史任务等级偏好。

保留priority分支的全部代码/结果：flow6为6N完整；dense24环境915ticks/19N5E、无实际安全/hash问题，776actor待标注。primary已授权只停止其未完成标注并保留原ANNOTATING与行政停止说明，不伪造成功、不继续flow12。未训练91901–3、未消费旧Q；没有checkpoint需要撤回。原neutral forecast train_A仍是有效开发证据，但它高SOC/warm的场景不等于以下三条件，不冒称正式三场景采集完成。

## 当前配置与最小数据范围（首次reset前冻结）

新增`configs/p3_v25_course.py`，profile=`p3_v25_neutral_course_committed_merge_20261008_v1`。保留已验证direct C²、committed incoming预测、105+N×50观测、K3、pre-action mask、真实能量/应急和资源服务，沿用foundation的downstream_soc_priority（energy slack、waiting/dynamic priority与可行性共同决定，非只有SOC）。有效base=1、policy mission token=STANDARD，但raw mission和30/120/10/60s service不变，SOC urgency weight2不变。三个条件来自execution_plan0.2：

|条件|请求/初始warm|SOC|到达/周期上限|train seed|Q seed（保留）|
|---|---|---|---|---:|---:|
|flow6_mixed|6/0|uniform .55–.90|10+60i秒；1210ticks|893501|895501|
|dense24_fixed|24/12（6/4/2）|fixed .8|剩余12架每15s；1800ticks|893502|895502|
|dense24_mixed|24/12（6/4/2）|uniform .55–.90|同dense fixed；1800ticks|893503|895503|

fixed .8沿用固定SOC开发值，不提高mixed下限或改变能量阈值来制造通过。每臂一次；自然完成或既定上限。源码、服务分布、能量、安全与调度语义先定向检查，独立smoke892851/3warm3/2ticks验实际入口和输出，永不进入训练。首次预检server新根不存在、7个seed在已见protocol元数据无碰撞；每次启动仍核对无重复进程/输出。

输出根`/home/user/training_outputs/terminal/p3_v25_neutral_course_20261008`，目录`smoke/train_flow6_mixed/train_dense24_fixed/train_dense24_mixed`。入口`training.p3_v25_course_data`。环境3600s、全部actor pre-state按tick/数值UID、cap2048query/51200 RELEASE意图、离线标注7200s、rebuild900s；smoke环境/标注各180s。仅一个父进程和一个顺序query child。超限保持INCOMPLETE，不补标/延时/重跑。原null不作WAIT、forced不入actor loss、单机teacher不当联合执行标签，沿用已测试collector/rebuild语义。

每臂完整后primary核对N/E/F/D/U、technical/censor、安全/hash/能量、全部actor-selected-OK label集合及105+N×50/teacher证书join和成本；安全完成E单列，无新增零E门槛；FDU/技术/实际安全/hash/不完整标注阻止后续。三条件完成再判断训练适用范围，不能因少量train标签就宣布泛化。当前不扩大数据矩阵；需要补充时由实际残差/闭环证据决定。

## 接续学习与正式启动

未曾真实执行的`p3_v25_task_imitation`/`p3_v25_task_student`入口现显式切到本profile及新protocol；priority的原collector/数据不改。拟合仍预定fresh91901/2/3、各5000updates/300s、固定final/canonical首seed、Adam3e-4/batch128/clip.5；输入仅上述三个完整train，fit不打开Q。必须先记录实际输入分析后执行。旧895401–3与895301–3仍未消费，不移作train。

候选按训练内残差和teacher-free开发闭环作必要的Safe DAgger修复，补充只进入新train/development，不借Q调参。新Q895501–3在候选冻结后采集并仅评分一次，保持原独立imitation判据，不事后降低。student-only资格要求完整任务/原请求与likelihood/实际安全/无FDU，原actor takeover<=.05恢复为显式检查；普通候选拒绝不等同安全违规。再做真实critic-only warmup、有界PPO更新与源/更新后同任务对照，检查退化及收益边界。learning预算与具体开发seed在运行前记录；此处不声称任何准入。

用户授权持续修复至新的P3正式训练启动仍有效。必要有界步骤失败则primary诊断修改，不以“等待批准”代替工作；也不以目标期限跳过证据。最终正式配置按本中性课程冻结，核对服务器启动PID/cwd/argv/初始产物后结束，删除前序60min heartbeat，**正式训练不设heartbeat**。无Git remote，代码通过Syncthing同步，不能称Git已推送。

## 实际执行

协议写入时：新profile/入口正在定向测试；新smoke/train/Q未reset、BC/PPO未执行。I0/I1/STUDENT/PPO均未准入。

定向11项local/server tests均通过40.33s/21.11s，8个代码/测试文件SHA相同，实际reset确认fixed.8、effective base1、多物理mission与STANDARD policy token；原始null/完整标注/105+N×50、严格train/Q隔离与无优化器student检查通过。恢复既有takeover<=.05检查，未新增门槛。priority dense父2819670/所属query2821768已SIGTERM，exit143，62个label文件及原ANNOTATING保留、无残余进程，行政停止单列；环境19N5E不是完整采集成功。下一步仅独立892851 smoke。

实现6dd89d98后独立892851 smoke已SMOKE_OK/exit0：2ticks/6raw/3actor，三接口各一条teacher RELEASE12并实际批准；有效base全1、物理standard/cargo/surveillance各一、2行SOC<.65。原null/forced、90D→105+N×50/原label join及首tick重放通过，实际安全/hash/负能量0。环境3.479s、标注4.558s；result SHA b9a66036e8a407dad600449329470b6f4b0c0a2fe3a2f586910dc51ca1b28bd1，关键文件已同步，无残余进程。primary读完整分析后仅接受train_flow6_mixed；smoke不重跑、不进train。

train_flow6_mixed/893501已COLLECTION_OK/exit0，694ticks/6N/0E/F/D/U，2026raw=19actor+2007forced，19/19原teacher均R，proposal12/13/11为17/1/1。三接口6/6/7，effective base全部1，10行SOC<.65、5行<.55；物理standard/cargo/surveillance覆盖，busyLZ13、afterretry1。没有WAIT标签是此完整低并发回合的实测分布，不补造WAIT；尚不能单独训练合格gate。规则19R获18批准，唯一teacher11对规则nominal拒绝差异保留，不改标。最小实际/扫掠111.112/111.003m，最低完成SOC.439628，应急0，技术/censor/hash/负能量0；首tick重放与全量raw/selected/labels/context检查通过。环境49.805s/标注30.168s。result SHA7edc78c9330e770a2ce01fed62ab5906c9c9c33a7682113b01bc72a2d9e39aea，context SHA1ab2df8c332c83e8efb06950e1920c82829385c6c36da8be158162fdaf026c1c；关键终态两端同步、父2822456及child退出。首个分析因Syncthing result滞后失败，确认hash后只重做读取分析，未重跑环境。primary接受下一臂dense24_fixed/893502，其他臂仍待结果。

### 后续学生开发与critic入口准备（未执行）

仅在三个train完整分析、固定fresh fit完成后，canonical91901先做teacher-free两tick smoke（env892852/policy929852、3warm3 mixed），再顺序三个完整随机策略周期：flow6_mixed 892853/929853、dense24_fixed 892854/929854、dense24_mixed 892855/929855。这些为development，可用于诊断/DAgger与同源critic MC，不作为untouched Q。各rollout/replay各3600s/场景完整tick上限，全量请求重放；不选最好policy seed、不直接加载旧source。业务异常时只加必要同状态/同seed规则诊断，不能靠重复抽seed掩盖失败。新helper只准备未执行；首次reset前还核对seed元数据/输出无重复和source。

primary新增`p3_v25_critic_initialization`入口：只接收与同一冻结canonical源/本profile一致、三个完整随机student周期，拒绝其他课程/确定性/不完整记录；逐条核对stored原分布后，以gamma.99完整terminal MC进行500次critic-only更新（batch32/Adam3e-4/seed93001/600s），actor与零gate adapter不变，丢弃warmup optimizer。PPO必须重新采样更新后的value，不把旧critic值当on-policy新rollout。入口不授予PPO资格，真实调用仍须先完成source/student判断；warmup MSE下降也不是性能通过。当前只准备代码及合成测试，没有真实critic更新。

critic入口与PPO构件16项local/server定向测试通过10.36s/5.45s，包括序列化checkpoint保留actor/零adapter、无optimizer继承、错误课程/确定性/failed-energy拒绝；一个既有test scalar warning不影响通过。4个源码/测试文件两端SHA一致。仍只有合成测试，无真实critic/PPO更新，运行中course collector所用核心未改。

当前唯一dense24_fixed/893502已启动：Python2822837/timeout2822836，真实cwd与argv匹配course_data/index2；protocol确认task_priority=false、fixedSOC.8。初次snapshot仅首tick12raw/0steps，随后本地同步progress100ticks/1409raw/41selected/227.497s，属于正常推进前缀，不记终态。核心冻结，不重复启动；完整结果分析后才可继续mixed。

为对应中性课程要求，student/后续PPO结果增加全已完成请求的holding wait、request-to-touchdown、request-to-completion的mean/p95/max；完整回合才解释为全请求，部分回合明确completed_count，不把服务完成当触地。尾部用于监测，不新增逐子组不退化门槛。2项local/server metrics测试通过.22s/.11s；只补结果字段，当前collector/teacher/物理不变。

### dense24_fixed环境终态与标注进度

一次低频检查读到环境已自然结束：1157ticks/2189.094s、16N8E、0F/D/U，24/24请求生成，active/queue/service均0。8次应急已完整触地并完成服务；最小实际/扫掠82.045532/82.030897m、最低完成SOC.212100，实际碰撞/NMAC/间隔/takeover/hash/负能量均0。391次safety_interventions是计划过滤累计，不是391次实际安全违规。固定.8仍有8E及377个LZ idle-with-inner-demand ticks，说明密集业务效率仍有代价，不能写成24N或已解决调度问题；这不触发新造的零E门槛。

原始12151行=500actor+11651forced；root仍ANNOTATING，检查时23/500标注完成、350.669s，query window23在运行。annotation_progress中的单次COMPLETED不代表整臂完成；继续原7200s上限，不提前运行mixed或学习。上述为环境终态/标注前缀，不是完整采集结论。result SHA714b25a0df8936cb4611590cacb1904824901ea1d905344e58024e3f5b67e62b对应ANNOTATING快照。

### 有界PPO入口准备（尚未准入/执行）

primary实现`training.p3_v25_ppo_block`作为后续首个三周期block，不启动formal。它要求新fit三seed对应的独立Q判据通过、同canonical源的三个完整随机student周期及critic初始化；Q只读评价结果，优化器reference只读strict train观测，不读Q行/teacher。首次实际调用之前仍由primary审查真实source/student证据。本节预留train env896501/2/3、policy936501/2/3，分别三课程条件；首次reset前须做历史查重。

从critic_initialized加载同源actor/零adapter，fresh Adam3e-5；每完整周期后epochs2、minibatch32、gamma.99/lambda.95、clip.1/valuecoef.5/gateentropy.001，exact rollout gateKL<=.01、fixed-source train-referenceKL<=.02/coefficient1、globalclip1，冻结全部actor/proposal只更新gate_adapter与critic。每次更新600s在minibatch边界检查，超限保存已接受更新的checkpoint并将block标为BLOCKED，不延预算；优化器/KL拒绝沿用原回退。保存每次update及model+Adam，不覆盖source。

三个新training周期完成后，仅在相同development env/policy seeds评估候选，与原source三个完整周期按请求汇总N/E、reward、energy、holding/full-lifecycle mean/p95/max；不得把这些配对development当heldout。PILOT_COMPLETE只表示协议完整，不自动candidate_retained或formal_allowed；primary根据真实训练更新及退化/业务证据决定下一步，有需要再预注册新block，不能通过no-op更新宣称PPO修复。任何技术/实际安全/不完整/FDU/takeover超限均停止后续保留产物。

此入口及PPO构件16项local/server合成测试通过10.87s/5.43s（3个新orchestration用例+13个PPO用例，和此前tests有重叠）：失败Q在optimizer前拒绝、真实测试梯度更新冻结actor/proposal、按三train→三paired-eval保存artifact、超限留checkpoint并停止、重复输出不覆盖。无真实BC/Q/critic/PPO，当前collector未导入这些新模块，物理与标注核心保持冻结。

### 标注成本检查（原任务不变）

下一次60min检查仍为ANNOTATING，130/500个label已完成、4354.579s，正在window130；最后完整window129为tick218/UID49、25个RELEASE意图均拒绝。该前缀已显示标注成本明显高于此前低并发臂，但不能据线性外推提前宣布超时或改变本次7200s上限。原任务继续，源码/输入/标签冻结。

primary使用既有`diagnose_p3_expert_cost.profile_query`，仅在本地读取已同步的可信快照，预定window0作输出smoke、window129作一次耗时profile，各最多120s，总最多2个teacher query/50 RELEASE意图，0reset/step/学习。输入selected SHA76f505d230601fc3037282aed841021f1caf421a8f0ae4e6085e4cb033531130；输出`.tmp/p3_v25_dense_cost_{smoke,profile}`，不写服务器/原采集目录。必须复现原label及逐候选audit并保持原状态/文件不变；这里只判断计算成本，不补标、不将profile结果加入train，也不影响正在运行的query child。

本地smoke为SMOKE_OK/2.078s，原label/audit精确相同；完整window129 profile触及120s后停止，只有progress保留，未得到终态统计。原server该query本身45.594s，不能把跨平台且启用cProfile的120s说成原服务器query超时。下一最小诊断缩小到**同window129的第一个nominal候选joint13一次**，最多60s、新目录`.tmp/p3_v25_dense_cost_candidate`；复用已经通过的真实smoke，不重复全query、不增加原标注预算、不做reset/step。

单候选已PROFILE_OK/5.578s，原`predicted_holder_reachable_tube`拒绝精确复现。4.588s（约82%）在ring_transfer，其中19次state_at/ring_hold反复构建/验证候选等待段耗时2.838s；prime_quintic_bounds累计2.966s，不能把嵌套时间重复相加。完整SafetyFilter为.887s。证据说明本状态的主要计算开销在搜索轨迹的生成/等待段验证，不是神经网络学习，也不是无根据地把SafetyFilter判成错误。

primary据此只准备**未接入生产的本地候选**：对环上等待段的水平速度/加速度多项式，用Bernstein控制点凸包给出保守上界；只有带向外数值余量的上界已小于原限值时，才省去同一等待段的极值求根。不能证明时完整调用原精确求根路径；不改变任何点、候选顺序、动力学限值、安全过滤或最终计划证书，不改`segment.bounds()`语义。先做边界/随机多项式和真实等待段等价检查，再同状态单候选比较；当前正式标注仍用旧冻结源码。本候选只有通过证据检查且当前任务退出后才可能接入，不预先宣称提速或解决采集。

等待段候选的7项本地检查通过：300个随机多项式保守证书、阈值/内部峰值/极端值回退，以及12组真实圆环等待点逐数值完全相同。两个既有状态各单候选的同机无profiler对照：slot0为1.174→.819s，slot129为3.220→2.147s；安全结论和两个生成轨迹hash均完全相同。该局部约30%–33%节省不能外推为全采集节省率。

同一数学证书的下一项局部候选扩到parameterizer的逐段速度/加速度/高度筛选：确定满足全部原限值的段仍保留原三点finite sanity检查；不确定段完整走原解析极值检查，最终SafetyFilter/动力学与exact bounds缓存不改。先额外150个随机段和原判据逐项比对，再另目录`.tmp/p3_v25_dense_cost_hull_combined`复用同两个既有状态的4个单候选调用，总120s，0reset/step/完整teacher query。生产仍冻结；这不是新的采集臂或安全阈值修改。

组合候选的8项测试全部通过（新增150段保守上下界及接受/拒绝等价），两个同机单候选为1.232→.651s、3.253→1.744s，原feasible/reason及生成轨迹hash完全一致。接下来只对新候选在原slot0/129执行各一次完整query（各120s上限、0reset/step、新`.tmp/p3_v25_hull_query_{0,129}`），逐字段对照原已完成label、audit与checked certificate；不重跑baseline全query，不据两个局部点宣称全采集通过。核心仍未改。

slot0完整query的原label、proposal及audit相同，但与服务器原记录比较的certificate哈希断言失败（exit1，query.json保留，未写成功result）。本地**旧实现与候选**同为9c482a0f…，服务器原记录为e66fade2…；目前不能把跨平台不一致归因于候选，也不能宣称服务器证书等价。单次只读核对六个相关生产文件两端SHA完全一致；local Python3.12.14/NumPy2.3.5，server Python3.12.3/NumPy2.5.1。数值平台差异是待验证解释，不能据此放宽checked/executed校验。生产核心仍冻结；原任务退出后再用服务器原平台检查候选。slot129为WAIT无放行证书，继续原定独立query/audit比较，各上限不变，不重复slot0。

slot129本地候选完整query已QUERY_EXACT/exit0、43.191s：25个RELEASE意图全部仍拒绝，所有audit/reason及WAIT标签与原记录逐字段相同，0reset/step。这里不把跨平台耗时当提速比。已准备原平台验证脚本，只在当前父/child全部退出后分别运行slot0、129一次、各120s；输出新`p3_v25_hull_query_20261008/query_{0,129}`，只在私有Python进程挂接候选，不改生产文件/原数据。slot0必须先通过精确原证书比较，再执行129；不放宽哈希或原安全判据。

### 固定SOC标注终态与计算修复

原train_dense24_fixed自然结束，服务器exit文件为2，INCOMPLETE/error=null，唯一blocker=`incomplete_annotation`。环境结论不变；annotation在7200.0135s封顶：201OK（30R/171W）、window201超时、298未尝试，合计500。原失败与所有前缀保留，未延时；关键终态两端SHA一致，原父及child均已退出。root result SHA943e18142b78d2c5f64c4299334724e4b2b257f78f4c6eaeff4cf18de53c7a8b。局部201标签不作为完整训练集、超时/null不转WAIT。

退出后原平台两个候选query均QUERY_EXACT：slot0 .382s，放行certificate逐字等于原e66fade2…；slot129 24.741s，20个holder reachable tube、3个holder separation、2个predicted separation拒绝及WAIT完全相同。slot129原记录45.594s，相同服务器状态下约46%节省；两点不代表全回合提速。跨平台哈希失败不再构成候选改变轨迹的证据，原checked/executed严格相等仍保留。

primary随后接入`core/quintic_hull.py`、环等待构建充分条件、parameterizer充分条件及原解析fallback。没有改轨迹点/顺序、能量/安全限值/最终certificate，也不改analytic bounds缓存。43项local/server定向tests通过9.96s/5.33s：随机保守边界、内峰值/阈值/非有限回退、12组圆环点逐值相等及已有V3生成/parameterization/dynamics/合同。下一步是已接入生产模块的同两条原平台query，各120s、新`p3_v25_hull_production_query_20261008`；不使用临时monkeypatch，原数据不改。

若生产查询通过，下一项是**新协议的完整离线标注修复**：复用原已完整的893502环境全部12151raw/500actor及可信快照，0reset/step，不重复环境、不改变16N8E；在独立`train_dense24_fixed_annotation_v2`重新标注全部500条，不沿用201条已完成label作为可选样本、不只补容易/剩余状态。原采集永远保持INCOMPLETE；新产物明确引用原rollout及失败annotation。依据已观测201条耗时2h、同状态约46%节省，预设新标注上限21600s、全部actor/2048query上限不变；不声称6h必定够。所有原201个OK标签应与重算标签逐字段相同（耗时字段除外）；任何语义/证书差异须诊断，不能覆盖改标。全部完成、重建及primary分析通过后才能继续mixed。未来尚未reset的mixed也显式使用21600s标注上限，环境仍3600s；这是计算预算修订，非改变业务/标签或选择seed。Q与学习尚未执行，保持分离。

生产同状态查询已全部QUERY_EXACT/exit0：slot0 .382s、slot129 24.618s，label/audit/certificate与旧记录逐字段相同。新`p3_v25_annotation_repair`仅接收annotation-only CENSORED的当前train完整rollout，独立复制raw/snapshots、全量重新标注，不复制原label，不接受Q/部分actor/有业务阻塞的环境；原标签所有非耗时字段必须相等。全部成功才重建105+N50并允许后续loader读取新目录；没有重置/步进/优化器。新入口及课程8项local/server测试通过21.69s/10.97s，包含实际文件复制/新labels与result输出路径、原失败保持、错误输入/变更label停止和21600预算在reset前落盘。连同43项生产回归，本轮51个不同测试通过；未重跑已有smoke或环境。

本轮证据类型：上述运行/测试/分析为newly_run；原环境与旧标签作为preexisting_artifact在新查询中只读引用；没有将user_claim当测量。两端代码SHA相同。下一运行命令为`python -u -m terminal_marl.training.p3_v25_annotation_repair --source-directory /home/user/training_outputs/terminal/p3_v25_neutral_course_20261008/train_dense24_fixed --output /home/user/training_outputs/terminal/p3_v25_neutral_course_20261008/train_dense24_fixed_annotation_v2`，cwd与Python环境按项目准则、外层24000s；source result SHA在启动前再次核对。源码/测试提交后才启动，初始产物确认后60min监控；主目标继续至正式P3实际启动，最终formal不设heartbeat。

实现提交`dbc2dd3296522530c1fa52d9f814aa0848d557f6`后，新完整标注已启动：Python2833606/timeout2833605、SSH session33746，真实argv/cwd已核对，初始query2833639/window6；protocol记录原失败SHA、0reset/step及21600s。初始snapshot为6条已完成/500、12.630s，firstlabel为原R12；随后本地同步文件到达。root仍ANNOTATING，无annotation_result/exit，不当完成。运行中核心及重建模块冻结，60min heartbeat已更新；只有全量通过后才启动mixed，旧失败目录绝不再用于fit。新的fit/monitor/analyzer helper已显式指向修复目录；没有真实BC/Q/critic/PPO。无Git remote，源码通过Syncthing两端SHA检查，未Git推送。
