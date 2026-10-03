# V3/90D稀疏专家标注开发pilot（2026-10-04）

## 目的与范围

用户授权继续下一步，并允许长耗时任务每60分钟heartbeat检查，不要求第一时间读取结果。primary继续采用research-pragmatist/academic-experiments的最小开发验证：实现上一轮记录的“完整规则回合→事先固定pre-state离线标注”，亲自负责代码、文档、分析及准入判断；worker仅执行指定SSH、测试、监控和本地Git。

前轮892201 expert-on-policy采集在298ticks/3625.746s被截断，3N/9U，无实际安全/hash异常；固定状态25候选查询的主要成本是轨迹生成。缓存试验等价但收益不足，已撤回，诊断commit `f542ba485456e7d7892cda6986f69fdcf928e80b`。这些是preexisting_artifact，详见[前轮报告](p3_v3_i0_pilot_20261003.md)，本轮不重跑旧pilot或改写其CENSORED。

新脚本`terminal_marl/scripts/collect_p3_v3_sparse_expert.py`只做开发数据流程验证，不训练。轨迹、安全、动力学、能量和奖励核心保持原版本；沿用`p3_integrated_repair.build_config`的static/direct_c2/nominal-holder/90D配置，以及此前integrated repair验证中的名义RELEASE、speed 0/-1/+1三个候选的rule K3行为。专家标签只在私有快照中查询，不进入规则行为动作。

## 首次reset前固定的协议

| case | seed | 负载/SOC | 环境预算 | 标注预算 |
|---|---:|---|---|---|
| smoke | 892401 | 3请求/warm3，.90–1.0 | 2ticks/180s | 180s，本窗口最多1个UID |
| high | 892402 | 12请求/warm6，.90–1.0 | 完整终态或1800ticks/3600s | 1800s，最多24次query |
| mixed | 892403 | 12请求/warm6，.55–.90 | 同high | 同high |

上述seed本地configs/scripts/tests在代码写入前扫描无匹配。server历史输出检查：候选seed路径无匹配；4531个protocol/manifest/config/result/summary的JSON/YAML及小型log扫描完成，排除小数片段的精确整数匹配为0，41个大于2MiB文件未读。先前全量raw行数字扫描在55/60s上限停止，本地出现无文件路径的数字片段，不能认定是seed使用；本轮以已完成的元数据/路径核对为依据，不声称已穷尽巨型raw文件。扫描不消费环境seed。892401–892403为新development分区，永不并入train/audit/Q；原893001–893024/894001–894006/895001–895012仅预留，未使用。

采样在标签、终态结果和耗时出现前决定：1800tick划为24个75tick窗口；每窗口第一个至少有一架actor eligible的pre-state，选择**按数值UID升序**列表中的`window_index % eligible_count`对应一架。一个窗口最多一条，空窗口不补额，不按teacher gate、拒绝理由、episode成败或事后耗时选样。自然终态之后的窗口不虚构状态。固定窗口的短暂事件漏采风险按实际覆盖报告，不为凑覆盖临时追加查询。

每个query沿用原state-only teacher的25种RELEASE意图以及最后WAIT顺序，遇首个可行候选即停止。最多24×25=600个**RELEASE意图检查**。primary代码检查发现final候选可能遍历多个LZ，所以这不等于600次`_candidate_plan`或轨迹构造；新输出分别统计release_intents_checked与candidate_plan_calls，保留teacher audit中的LZ检查。不得为了满足虚假的600计划次数门槛截断原专家搜索。

所有active每tick的90D原始观测、actor_mask、规则请求和候选、pre-state hash都保留。原始`rows.jsonl`中teacher_label始终null；实际标注写入独立`labels/window_XX.json`，按decision_id关联，只有status=OK才是有效标签。未选、未执行、超时和错误的标注不能填成WAIT；forced-WAIT不能进入actor损失。后续proposal监督仅限有效teacher RELEASE。标签单机可行性与规则K3批量执行不混淆，不把teacher request当作已执行请求。

环境循环不查询专家，不因标注额度用完而停；保留完整steps/action traces/N/E/F/D/U、checked/executed与能量。首tick保存before/request/live/replay并严格检查重放；每step复用已测execute_tape检查实际安全、hash、有限obs/reward、请求与执行契约。异常留存pre-state及请求，failed_energy立即按ERROR停止；wall/tick预算到达而未自然终态保持CENSORED。环境墙钟在tick边界检查，单tick可能略越界，不事后延长预算。

正常完整终态后才开始离线标注（smoke允许规定2tick截断）。每个已选状态一个顺序子进程，parent以剩余标注总预算为timeout；超时杀死并回收该child，记录TIMEOUT并让后续项保持NOT_ATTEMPTED。标签文件、日志及快照保留，错误不得升级成功。环境完成和标注完整性分别输出；PILOT_OK只要求环境按本case完成、至少一条有效标注且所有已选标注完整，无I0覆盖/性能资格含义。若环境存在D或其他业务阻塞，由primary分析后决定，不自动启动mixed。

执行顺序：定向tests→独立smoke→primary检查smoke原始行/标签/trace→只启动high。high自然结束且无技术/安全/耗尽/未决/业务阻塞，primary分析后才启动mixed。任何失败保留原输出，完成最小必要诊断，不同目录另记后续试验；不重启覆盖。最多一个本轮父进程及其一个标注child（顺序运行），不同时开两个环境回合。

## 输出及执行

server cwd `/home/user/projects/terminal`，Python `/home/user/envs/terminal_env/bin/python`，`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。输出根`/home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004`，各case独立新目录；相邻.log/.pid/.exit保存真实进程与状态。代码/docs仅本地修改，经Syncthing核对后执行；无源码手传、无server Git。

```bash
python -u -m terminal_marl.scripts.collect_p3_v3_sparse_expert --case smoke --output /home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004/smoke
python -u -m terminal_marl.scripts.collect_p3_v3_sparse_expert --case high --output /home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004/high
python -u -m terminal_marl.scripts.collect_p3_v3_sparse_expert --case mixed --output /home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004/mixed
```

每臂运行预算最多60分钟环境阶段＋30分钟标注阶段（环境tick边界可能略超），实际预计几十分钟内。启动后必须核对真实argv和至少一份预期产物；之后按用户偏好设置60分钟heartbeat，正常推进静默。high完成后继续已授权mixed；全轮终态/必要诊断完成、记录与提交后删除heartbeat。

## 结论边界

本轮rule occupancy与上一轮expert occupancy不同，因此不能把新旧总耗时或成功率差异直接当作同分布算法改进；本轮检验的是完整周期与有界标注能否分别完成、字段是否正确。先前完整环境开发验证已通过；新fresh90D student还未训练/qualified。正式I0仍需冻结采样及覆盖协议，后续fresh BC→Safe DAgger→student-only资格后才讨论有界PPO。PPO_ALLOWED=false；本轮不自动学习、full training、held-out或硬件，不创建/覆盖checkpoint，不声称训练退化已解决。

## 执行记录

代码、测试和以上协议由primary编写。测试与独立输出检查的实际结果随后追加，未运行项目不写PASS。

### 定向测试与独立smoke（newly_run）

`tests/test_p3_v3_sparse_expert.py tests/test_p3_v3_expert_collector.py tests/test_joint_teacher_state_only.py`共17 passed in 39.23s、exit0，日志`/home/user/training_outputs/terminal/p3_v3_sparse_tests_20261004.log`。覆盖窗口选择/数值UID/不补额、90D与mask/null、环境循环不调用teacher、真实快照标签与原teacher逐项相同、原始文件不变、真实子进程CLI与JSON、环境cap/error留证、标注总预算/单child timeout及hash校验。collector SHA`e46d4f2176f186ccf0b9650845ef1885419af984bdd0a154f3b27ce8c152cf35`，test SHA`1de56ddc67430d75965aa178d99e5258291f84cd2d298721ac81253bce59e608`，local/server一致。

独立CLI smoke892401已SMOKE_OK/exit0结束，原PID2478893退出。primary亲自读取全部6行/2steps/1selected/1label：6个decision_id唯一，90D及原始teacher_label=null、规则request与tape、row/step pre-state hash均相同；3 actor行/3 forced WAIT，选中tick0 UID0。单机teacher joint13/proposal12、request[1,0,0]，1个意图/1次plan检查，ring_transfer checked/content hash相同。first_live/replay完整相同；live规则批准3次，checked/executed samples30，所有实际安全/hash异常计数0，查询未计入live候选统计。环境阶段4.9541s、标注阶段2.8732s（child query2.1093s）。terminal=false/censored=true仅为规定2tick smoke，不是完整业务结果。

smoke result为4135bytes，SHA`106284862aa5385dd0bd6f847dd28c24ca31c9be469cbadb2b1e8b0ae965fab8`；目录`.../p3_v3_sparse_pilot_20261004/smoke`及其相邻log/pid/exit保留。primary据此接受新输出路径检查，只准许接着启动预注册high一臂；mixed仍待high终态分析。

### High启动核对，完整结果未定

代码/测试/协议与CURRENT_STATUS最新块已本地提交`411c4019540e55a2cd54ff117a59ce9e0a4b4035`，index空，其他dirty保留；无remote，未推送。启动前collector、config、polynomial core本地/server SHA一致，旧cache仍未采纳；high路径及进程无重复。

唯一high Python PID2479355，wrapper2479354；actual argv为本报告`--case high`命令，相邻high.pid保存真实Python PID，high.log/high.exit记录输出与退出。初次核对elapsed1:03、CPU99.9%、RSS521096KB；真实progress为rollout75ticks/588行/1selected，elapsed53.741s。rows当时597行、steps75行，是下一tick原始行先写入的正常顺序，不是额外完成9行的step。

primary随后读取同步产物：progress200ticks/1821行/3selected、completed3、elapsed118.8665s；这只是正在运行的前缀，尚无完整N/E/F/D/U或标注结论。首tick规则6请求均执行RELEASE，UID2/4实际选择rank2速度-1，其余rank1；原请求仍为名义[1,0,0]，teacher_used=false。这是rule K3的实际分支记录，不回溯升级冻结source或fresh student资格。

heartbeat `p3-v3-sparse-pilot`已创建并回读确认ACTIVE、每小时一次，目标为当前thread；按用户要求正常推进静默、重要失败/终态才报告。不再额外频繁轮询。high与其标注完整结果由primary分析后才继续mixed；运行中核心冻结，mixed尚未启动，所有学习仍禁止。

### High终态及mixed继续判断（2026-10-04，newly_run）

小时heartbeat读取时high已自然结束，原Python2479355及所有query child均退出，high.log为PILOT_OK/605ticks/6labels，exit0；mixed路径/进程尚不存在。环境阶段193.9813s、605ticks，N/E/F/D/U=12/0/0/0/0，generated=requested=12，active/queue/service=0，error=null、censored=false。标注阶段70.0351s，6/6已选状态均OK；两个阶段合计264.0164s（约4.4min，不含进程初始化）。本轮流程在这一个开发场景中完成，但其rule occupancy与原expert采集不同，不能据此报告同分布加速比。

实际安全：collision/NMAC/hard separation/buffer/runtime takeover/shield fail-closed/joint fail-closed/hash mismatch均0；checked/executed samples14339、匹配计划38，最小actual/swept separation=84.04323/83.82647m。`safety_interventions=35`是候选处理计数，实际shield_interventions=0，不将两者混写。12个ground_service_complete终态SOC范围.645182–.905261，最低剩余能量967.7737；应急尝试/提交/耗尽全0。

primary亲自检查全部3204行/605steps与6条label：decision_id唯一，90D有限，原始teacher_label全null，rule request/shortlist和step tape一致，row/step pre-state hash一致，执行trace保持原请求、teacher_used=false，所有实际rank对应原K3项。逐tick按预注册窗口重新构造的选择与selected完全相同；各选中快照SHA、90D、pre-state和label join一致，teacher joint/gate/proposal及audit末项一致，两个RELEASE certificate均checked/content hash相同。first_live/replay完全相同，无技术或标注截断。

64条actor行、3140 forced WAIT。64次规则RELEASE请求中32执行RELEASE、32执行WAIT；批准rank1/2/3分别26/5/1，32条原active被拒请求的主要原因PREDICTED_SEPARATION28/NO_MERGE_GAP3/HEADWAY1。此为pre-action active行口径，与包含本tick新入场trace的环境总计分开。6条标注的RELEASE2/WAIT4与这6个状态的实际gate一致，但不外推单机标签与批量执行总能一致，也不把这些行当训练数据。

| window | tick/UID | 接口 | teacher | 意图/plan调用 |
|---|---|---|---|---:|
| 0 | 0/0 | o2m | RELEASE proposal12 | 1/1 |
| 1 | 75/3 | i2f | RELEASE proposal12 | 1/1 |
| 2 | 151/8 | o2m | WAIT，25 predicted_separation | 25/25 |
| 3 | 230/8 | m2i | WAIT，25 predicted_separation | 25/25 |
| 4 | 314/8 | i2f | WAIT，25 predicted_swept_separation | 25/75 |
| 5 | 392/46 | i2f | WAIT，16 swept/9 predicted separation | 25/50 |

合计102个RELEASE意图、177次plan调用，验证了“意图数≠多LZ计划数”的计数区别。窗口6–8在自然终态前没有actor eligible，不补样；后续窗口不产生虚构行。6条标注只支持proposal12，数据规模/支持度远不足以宣布I0覆盖通过。

raw fast_inner/busyLZ/retry/lowSOC=165/3198/295/0；actor行相应23/58/0/0；被标注状态相应3/5/0/0，均为相关状态行数而非独立episode。规则请求WAIT→RELEASE58次，其中前一行处于retry的恢复32次；这是behavior请求恢复，不是teacher标签恢复证据。high无低SOC覆盖是预期，下一步使用已登记mixed892403检验，不临时更改high采样。

本地/server主产物SHA一致：result9832bytes `fcf5d73b88f7d87bc321259626656a2e917804fdc946c93716516978ea360ca7`；rows4397300bytes `0d35fc171d90a86327c7f988f15932b38dc9a2a79e8438d1dd9cf98de51d9728`；steps6040262bytes `c219db9f8e539b13697888268cdc47d72eb6ecbe96c685bfc1de689e7263c32a`；selected16161bytes `5b52864fccfa5fcb01bce0b8f6572ab099eb4f370a9b42f8fbdc6da97bd77583`；annotation_result2079bytes `d6177a8659e2f8d771cf49e41168d8e0a756b89e06ed5030a4def38113aa49fc`。

判断：high没有阻止既定mixed开发验证的技术/安全/耗尽/未决/业务问题。primary接受high流程验证，继续原协议mixed，核心不改、不重新运行high。17tests/smoke证据仍适用，文档追加以diff检查验证。任何学习、I0/student qualification及PPO仍未开始。

### Mixed已启动，终态待查

启动前collector/config/polynomial core与预注册SHA保持一致，mixed输出与相邻pid/log/exit原先均不存在；确认无high或query child残留后只启动一次mixed。真实Python2483247、wrapper2483246，argv为既定`--case mixed`命令，输出`.../p3_v3_sparse_pilot_20261004/mixed`，活动seed892403。初次核对进程R、elapsed00:57、CPU99.9%、RSS516240KB；progress为rollout75ticks/579行/1selected、54.7125s，文件实际已80steps/637raw行，result/exit尚不存在。这些均为运行前缀，不报告完整成功率或耗尽结论。

当前唯一活动臂是mixed；接续先读mixed.pid及实际argv、progress/rollout_result/annotation_result与相邻log/exit，禁止重复启动high/mixed。heartbeat保留60分钟，正常推进静默；mixed终态由primary亲自核对全部raw/label/选样/执行/能量安全后形成两臂结论和有界后续方案，再提交文档并删除监控。运行中不修改核心。

### Mixed终态（2026-10-04，newly_run）

下一次小时heartbeat确认mixed自然结束：根PILOT_OK，rollout COMPLETED，error=null、censored=false，mixed.log为572ticks/7labels，exit0。Python2483247/wrapper2483246及所有本轮collector/query child均退出，high也无残留。12请求全部正常完成，N/E/F/D/U=12/0/0/0/0，active/queue/service=0；环境阶段124.6635s，7/7标注阶段42.1199s，两阶段合计166.7834s（约2.8min，不含初始化）。

实际安全异常及checked/executed hash mismatch全部0，checked/executed samples15448、匹配计划38；最小actual/swept separation=84.56070/84.54101m。12个ground_service_complete终态SOC范围.320021–.649963，最低剩余能量480.0313，无耗尽。应急attempts=8、blocked_safety=16、committed/touchdown=0；primary核查实现，attempts在进入应急尝试时加1、blocked_safety在每个被拒LZ计划上加1，因此二者计数单位不同。该回合最终全部走正常完成；不能写成“从未触发应急判断”，也不以候选安全拒绝冒充实际违规或据此放宽安全层。safety_interventions=31是候选处理计数，实际shield_interventions=0。

primary运行自己的逐行分析并检查实际label原文：3383个decision_id唯一，57 actor/3326 forced WAIT，全部90D有限、raw标签为null、rule请求/shortlist与tape和执行trace一致，teacher_used=false。窗口/UID从全部raw独立重建与7个selected一致；所选快照SHA、pre-state、90D与label join正确，joint/gate/proposal/audit末项、5条RELEASE证书正确；first_live/replay相同。原active规则57 RELEASE请求中32执行RELEASE、25执行WAIT，实际rank1/2/3为21/10/1，拒绝原因PREDICTED_SEPARATION23、HEADWAY2。此口径仍不把同tick新入场trace当作pre-action actor行。

| window | tick/UID | 接口 | teacher标签 | 意图/plan调用 |
|---|---|---|---|---:|
| 0 | 0/0 | o2m | RELEASE proposal12 | 1/1 |
| 1 | 75/4 | i2f | RELEASE proposal12 | 1/1 |
| 2 | 152/6 | m2i | RELEASE proposal20 | 24/24 |
| 3 | 228/8 | m2i | RELEASE proposal12 | 1/1 |
| 4 | 302/8 | i2f | WAIT | 25/75 |
| 5 | 377/24 | i2f | WAIT | 25/25 |
| 6 | 451/26 | i2f | RELEASE proposal13 | 3/11 |

合计80个RELEASE意图、138次plan调用。5 RELEASE/2 WAIT，proposal12/20/13分别3/1/1。tick152 UID6的teacher proposal20为`[1,1,-1]`，存在checked轨迹；规则实际请求`[1,0,0]`，K3仅含timing=0的三个速度，最终执行WAIT。**该teacher动作没有送入执行层，不能把此处写成“专家RELEASE被拒绝”或标签错误**；它只显示本单机查询找到规则固定K3之外的候选，尚未证明将它加入联合批次必获批准。tick451 UID26的teacher proposal13为`[1,0,.5]`，而规则实际执行rank2的`[1,-1,0]`；两个不同意图均保留，不按实际行为改teacher标签。

raw fast_inner/busyLZ/retry/lowSOC=249/3377/263/1352，actor行28/51/0/28，被标注状态4/6/0/2。两个低SOC标注分别为SOC .504364的WAIT和.411558的proposal13 RELEASE；这些仍是两个相关开发状态，不是低SOC泛化资格。规则请求WAIT→RELEASE51次、前一行retry后的恢复25次；都是behavior请求统计。结合high逐行检查，有3个selected状态紧随同UID retry行之后：high151/8为teacher WAIT，mixed377/24为WAIT，mixed451/26为RELEASE。当前retry=true行被actor mask排除是预期，不能把“0个retry中的actor标签”当作缺陷；实际需要保留并检查的是retry结束后的决策。

本地/server主产物大小及SHA一致：result10172bytes `ec803095a6355b51c167fcdbd15f6e20906675d930f8989814736ed6038a693a`；rows4617158bytes `0e70955e9508261326ed832021fee2b81fb9b038f6c8b369ba25964dc3c0e4de`；steps6347175bytes `5d4224482035d909b94f9dd5c4b87b53088afb0952a6f99052921c869d80a5b1`；selected18523bytes `44462d030fc17456a4edf8c3f016fb2378b5eb4d7fe9c3d696a89f2c2f5c950a`；annotation_result2395bytes `0d096f75171ac3851e54d1e2644342a9edd0629783ade4167920ed056533c9cd`。

## 两臂结论与后续有界工作

| 开发场景 | N/E/F/D/U | 环境终态 | 环境/标注秒数 | raw/actor/有效标签 |
|---|---|---|---|---|
| high892402 | 12/0/0/0/0 | 完整、无censor/error | 193.98 / 70.04 | 3204 / 64 / 6 |
| mixed892403 | 12/0/0/0/0 | 完整、无censor/error | 124.66 / 42.12 | 3383 / 57 / 7 |

**接受本轮稀疏采集流程验证。** 两个完整开发回合共24N、0E/F/D/U，13/13预选标注完成，无真实安全/hash异常。把完整环境推进与有界专家查询分开后，这两个场景没有再因专家扫描成本丢失完整业务终态；没有通过改物理、安全、标签或停止规则获得PASS。此前旧expert-on-policy失败仍为CENSORED。新旧行为分布/seed不同，不给“快了多少倍”或优于原策略的因果性能结论，也不为已闭合的开发采集流程再扩大诊断矩阵。

**仍未解决的任务是新MDP下的数据充分性及训练后表现验证。** 6587条raw中只有121条actor eligible，6466条forced WAIT占98.163%；采样取得13条有效label，只占eligible行10.744%，绝不能把6587行当作6587个监督样本。13标签为7 RELEASE/6 WAIT，RELEASE proposal12/20/13为5/1/1；各接口均出现，但仅有2个独立episode、2个低SOC标注、1个retry恢复后的RELEASE标注。规模、时序恢复覆盖和proposal支持都不能据此宣称I0通过。低SOC场景本轮正常结束也不能覆盖此前24请求低SOC场景中的应急兜底需求。

当前边界：环境开发闭环及数据语义检查通过；fresh90D student尚未训练、student-only未qualified、PPO_ALLOWED=false。要回答“P3训练后是否仍退化”，仍需在新MDP上完成fresh BC及Safe DAgger/student-only资格，随后才可按另行批准的有界PPO协议做更新前后配对；本轮无optimizer、checkpoint更新、held-out或硬件。

下一项工作限定如下（记录方案，本轮不执行）：

1. **先实现V3/90D数据接入，不开始学习。** 使用独立V3 loader/adaptor，保留历史88D入口。明确按decision_id关联status=OK的原teacher label，校验90D有限值/索引/seed与partition，gate loss只选有效label且actor_mask=true；proposal loss只选teacher RELEASE。null、TIMEOUT、ERROR、NOT_ATTEMPTED不能转换为WAIT，behavior/actual不能覆盖teacher target。训练入口必须拒绝development/audit/Q partition；本pilot只能用于只读字段检查，不能进入优化器。该项预算为一个loader及定向测试/序列化检查，0个环境reset、0个optimizer step、0个checkpoint写入。测试覆盖以上具体错误路径及actor样本计数，随后再决定数据采集。
2. **正式I0在首次reset前另冻协议。** 既有reserved train893001–893024/audit894001–894006/Q895001–895012继续未消费。候选采集上限仍为24train+6audit、每回合1800ticks，三种预定负载/高低SOC场景；这些是预算上限，不是要求立即跑满。pilot的每75tick仅一条已表现为数据过薄，不直接把这一采样密度升级成正式训练方案。先根据121个eligible行的时序分布制定固定、与标签无关的更充分选样和query预算，明确各接口、低SOC及retry结束后决策的记录口径、异常/截断留存规则与I0判定；必须在收集新分区前确定，不能看到正式标签后挑seed、补成功样本或改阈值。先执行协议中首个固定三场景cycle检查实际数据完整性/成本，再按同一冻结协议继续其余上限，不自动进入BC。不能把“每类都出现过一次”当作可学习性证明，也不沿用旧88D特定proposal计数当新90D门槛。
3. **学习证据独立建立。** I0达到预注册要求后，另冻fresh90D多seed BC/内部audit与一次性untouched资格协议，再按研究规范I1→Safe DAgger→student-only推进。学习曲线、WAIT/RELEASE与proposal条件指标、无teacher替代的完整N/E/F/D/U和安全能量需要同时观察；此时才能判断是否存在退化，不能从当前规则回合推断优化器更新后的行为。

本轮没有发现需要再改物理或安全核心的具体失败，因此不新增修复或额外仿真。collector/test/core SHA仍与411c401版本一致，既有17tests及smoke结果适用；本次只更新结论文档，diff检查通过。全部源代码/文档判断由primary负责，worker仅执行指定命令。本地显式提交本报告与CURRENT_STATUS最新块，保留其他dirty；仓库无remote，不推送。两臂及必要检查已完成，提交后删除`p3-v3-sparse-pilot` heartbeat。
