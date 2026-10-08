# P3 nominal curve / total wait 修复后的 fresh 数据与训练方案

本文件承接 [中性课程诊断证据](p3_v25_neutral_course_20261008.md)。用户已授权持续修复、验证到新 P3 正式训练实际启动；最后正式训练不设 heartbeat。本文在新的数据首次 reset 前记录后续范围，执行仍取决于前一步实际结果。

## 修复后的数据边界

核心 `55743e9d6a567a4d39424f0f6020f0a9d3c4e77d` 保持 80 m 安全间隔：仅已知 nominal、零宽 holder 轨迹改用相对弦段和加速度误差界；真实预测交叉仍拒绝。`total_wait_v1` 使用 external+internal 等待。当前课程显式开启两项，历史默认保留；profile 为 `p3_v25_neutral_course_nominal_curve_total_wait_20261008_v2`。

任务等级继续关闭：有效 base=1、priority reward=0、policy mission=STANDARD；保留 SOC 紧迫度、总等待、raw mission 的物理服务时间。历史 `random_135` 的 RNG 抽样只保持种子流，不是当前有效随机任务等级。

原 893501–3 的 1003 标签和 fit 保留为旧实现证据，不能混入新轮。旧 dense fixed 的 424 WAIT 中 196 条、mixed 的 406 WAIT 中 119 条含至少一次 holder-tube 拒绝；这 315 条只是可能受影响，并非 315 个已证实错误。两个同状态查询才具体证明旧 WAIT 被过保守判据影响。重新采集采用新物理/调度合同下的完整访问分布，不能只改已知错误标签或保留容易样本。

## 顺序与预算

先完成 `p3_v25_nominal_curve_validation_20261008/source_dense_fixed`：沿用旧失败 env892854/policy929854 和固定旧 actor，以新合同做完整迁移诊断，不授予新 student 资格。只有技术、完整周期、实际安全/hash、F/D/U 检查无阻塞后，才执行下表。若仍失败，保留失败并依具体状态完成最小诊断。

新输出根：`/home/user/training_outputs/terminal/p3_v25_neutral_course_v2_20261008`。数据协议 `p3_v25_neutral_course_data_20261008_v2`。三个条件、服务分布、动作与观测架构均不扩展。

|阶段|seed|条件|目录|
|---|---|---|---|
|独立入口 smoke|892863|3 warm、mixed SOC、2 ticks|smoke|
|train 1|893511|6 请求/0 warm，SOC .55–.90，每 60s 入场|train_flow6_mixed|
|train 2|893512|24 请求/12 warm，fixed SOC .8|train_dense24_fixed|
|train 3|893513|24 请求/12 warm，SOC .55–.90|train_dense24_mixed|

每个正式数据周期分别为 flow 1210 或 dense 1800 ticks、环境 3600s、全量 actor 标注 21600s、观测重建 900s，外层 27000s。全部 eligible pre-state 按 tick/数值 UID 排序，最多 2048 query/51200 RELEASE 意图，多 LZ plan 次数另计。smoke 的环境/标注各 180s、外层 600s，永不并入 train。超过上限原样保留 INCOMPLETE，不临时补标、延时或抽新 seed 替换失败。

运行前核对源 SHA、进程/输出与 seed 元数据；代码只本地修改，经 Syncthing 同步。最多一个环境父进程与其一个顺序标注子进程。各臂终态后 primary 核对完整 raw/actor/selected/OK-label 集合、105+N×50、null/mask、原 teacher/behavior/实际执行分离、三接口、proposal、busy LZ/低 SOC/retry/WAIT→RELEASE 与成本，再决定下一臂。无新增“零应急”门槛；E 是完整应急服务，仍必须如实报告其能耗/等待代价，F/D/U 和实际安全/技术阻塞不得掩盖。

## 拟合、闭环与独立评价

三个 train 完整后，沿用已验证网络和 loss：91901–3 各 fresh 初始化、固定 5000 updates、300s/seed、batch128、Adam3e-4、clip .5、未加权 gate CE + 条件 RELEASE proposal CE。不选最好 seed/epoch，canonical 固定 91901，输出新的 `fit_initial`，不覆盖旧模型。首先检查训练内残差；模型/损失只在有具体残差证据时修改。

随后新模型执行独立 student smoke（892864/929864，3 warm mixed、2 ticks），以及三个 teacher-free 随机完整开发周期（flow6_mixed 892853/929853、dense24_fixed 892854/929854、dense24_mixed 892855/929855），输出新根下 `student_initial/<scenario>`。这些完整场景沿用原开发协议：flow/fixed 已消费，mixed 原先因前臂失败未运行，全部为 development，不能冒称 untouched。fixed还可与本轮旧actor迁移诊断作同合同对照。每臂最多完整flow1210/dense1800ticks，rollout/replay各3600s、外层7600s；保留全部请求、likelihood、mask、奖励与精确 tape replay。无实际安全/hash/F/D/U/技术/周期阻塞，actor takeover 沿用 <=.05。有分布外误决策才实施有界 Safe DAgger，不直接把运行失败归因 PPO。

Q895501–3 保持未消费，仍按同三个条件保留。候选与训练数据最终冻结后才收集独立 Q，一次评价全部三 seed final；沿用既有 imitation 判据，不用 Q 拟合、选择阈值或反复挑模型。state-only teacher 是当前候选集首可行，不是联合批准或全局最优；学习前和评价解释中都保留此边界。

只有新学生完整周期和冻结 imitation 证据通过，才执行已准备的同源 critic 初始化及三周期 PPO block（预留896501–3/936501–3，尚未消费）。PPO 后在相同开发场景比较 N/E、全请求等待/落地/服务周期、能量、reward、安全和真实参数更新；不能凭 no-op、训练内拟合或一个周期宣称训练退化已解决。正式 runner、预算和 seed 在实际有界 PPO 结果审查后冻结。最终确认正式训练 PID/cwd/argv/初始产物，再删除本任务 heartbeat 并结束，正式训练不设监控。

本方案冻结时尚未产生新轮采集/拟合/Q/PPO 数据，后续实际状态见执行记录。证据来源：既有失败及旧标签为 preexisting_artifact，同状态查询与测试为 newly_run；用户授权不是性能证据。

## 执行记录

实现提交`e44e76d7792ac2a6812080e3f80005da163639ec`，15项local/server定向测试通过29.97/15.37s，源SHA同步，无Git remote/未推送。独立892863 smoke已SMOKE_OK/exit0：2tick/6raw/3actor、三接口各1条teacher R12，实际均批准；null/forced与105+N50原标签join逐行检查通过，首tick重放一致。base全1、policy mission STANDARD，raw cargo/standard物理服务保留；无实际安全/hash/能量异常。环境2.136s/标注3.606s，result SHA`425eaf71c9834dae8e52a91e0e07832a0c62c1dd2555a4dd1bce161c3ac2b363`，关键产物两端同步。进程已自然退出，速度较快未捕获live PID/cwd，不能补称捕获过；显式cwd启动器和完整终态/序列化证据均保留。该smoke不重跑、不入train。

primary接受唯一下一臂train_flow6_mixed/893511。后续拟合/student助手已编译，只准备未执行；其student开发种子如上固定，入口只读新根，不覆盖旧结果。

train_flow6_mixed已COLLECTION_OK/exit0：702ticks、5N1E/0F/D/U，2050raw=30actor+2020forced、30/30 OK标签（23R/7W，4种proposal）。三接口actor15/6/9；busyLZ18、retry恢复12、SOC<.55为5/<.65为17，3次同接口WAIT→RELEASE，原始null/actor mask/全部label与105+N50 join检查通过。teacher23R中5次原规则执行WAIT，保留原pre-state候选标签与批量执行差异。17次正常K3批准及1条应急无rank trace分开记录，不按执行改标。

最小实际/扫掠81.145833/81.133264m，最低完成SOC .354109/剩余能量531.164；应急1次尝试/提交/触地，实际安全/hash/负能量/技术/censor均无异常。环境45.167s、标注93.044s；210个RELEASE意图、386次plan调用。不同于旧flow seed的6N，不能把本5N1E差异归因修复。root result SHA`2ba05115052825fb8d105415c0a2eb15f3914cc9eb02d041c79bbd9c79ef666c`、context SHA`b3ee0bf5dfcd438bdfddee9a12120ba5ea22fed47746a64bbd2de780d92c2717`，终态两端一致；首次同步缺3个尾部文件，等待10s后齐全，未重跑环境。timeout2858974/Python2858975均已退出。

primary接受该完整训练周期，下一唯一train_dense24_fixed/893512，既定24/warm12/fixed.8、1800ticks/3600s环境/21600s完整标注，所有上限不变。尚无新fit/student/Q/PPO；dense须终态分析后才继续mixed。由于可能持续数小时，保持60min监控。

### 正式训练入口准备（无真实formal运行）

在当前collector没有import的新模块`training.p3_v25_formal`实现正式运行入口。它先核对primary明确的候选保留/正式准入说明、冻结fit/Q评价/同源student与critic、完整三周期PPO及配对结果和指定candidate SHA；不能只凭PILOT_COMPLETE自授资格。优化器只读取strict train reference，延续pilot的同一actor、gate adapter、critic与Adam状态，沿用已有PPO超参、逐minibatch exact gate/reference KL约束及回退；全actor/proposal始终冻结。没有加载Q观测或teacher，未创建新的实际准入manifest。

正式周期/seed/总墙钟预算由之后的结果审查冻结，入口要求按三条件顺序、完整循环及不与train/Q/pilot/student env seed重合。总墙钟在新episode开始前检查；每episode原rollout/replay各3600s、update600s不变。每周期独立保存完整请求/重放结果和model+Adam checkpoint；发生实际安全/hash/F/D/U/技术/超限立即保留已存结果并停止后续，不覆盖source，不自动重启/延时/选择最佳模型。正式入口不创建heartbeat或heldout评估。

本地和服务器16项合成测试通过6.09s/2.24s（含3项既有PPO block回归）：确实执行mock环境下的梯度更新并验证连续optimizer step、固定actor/source、三周期顺序、准入/profile/hash/seed拒绝、耗尽周期在更新前停止、预算截止保留checkpoint、异常保留完整rollout及重复目录不覆盖。初次测试因test helper的包导入路径错误而collection失败，改为项目已有`tests.*`方式后通过，失败日志保留；不涉及实际仿真失败。相关4文件SHA两端相同。当前collector核心与数据合同未改；入口测试不构成真实PPO效果或formal资格。

当前train_dense24_fixed真实timeout2859845/Python2859846，cwd和argv已核对，新profile/893512/fixed.8与全部上限吻合；first_request/before/live/replay和raw已生成。一次启动确认时400ticks/5completed/451selected、602.47s，是运行前缀而非全臂结果。正在写入的raw与progress时间不同，不能将两者瞬时行数差当终态不一致。核心冻结、后续仅60min监控；必须完整终态核对后再mixed。

### dense固定SOC已完整完成

train_dense24_fixed/893512已COLLECTION_OK/exit0，1003ticks/20N4E/0F/D/U；环境851.469s、标注4891.094s（约81.5min，低于21600s）。11448raw=721actor+10727forced，721/721标签全部OK：94R/627W、14种proposal。接口o2m为25R119W、m2i44R80W、i2f25R428W；busyLZ709、高速inner450、SOC<.55为238/<.65为555、retry恢复302，37次同接口WAIT→RELEASE。16084个RELEASE意图/9968次实际plan，资源等早期拒绝不必进入plan，所以后者可以更少。

全部raw90D/null/pre-mask、actor-selected-label集合、原证书与105+N50 join及快照config/effective base1检查通过。teacher/actual分别57R/R、37R/W、3W/R、624W/W，保留pre-state单机teacher与实际顺序/资源更新语义，不按执行改标；59次有rank正常K3批准（41/8/10）与1条无rank应急trace分开。原始physical mission包含emergency，并不恢复任务等级偏好，policy token仍STANDARD。

最小实际/扫掠80.797069/80.768339m，最低完成SOC .298050/能量447.075；4次应急完整触地服务，27次尝试、36次安全拒绝/7次LZ拒绝、0能量拒绝。实际安全/碰撞/NMAC/hash/shield/负能量/技术/censor均无异常；146次safety_interventions为计划拒绝。20N4E与其他seed结果不作修复因果收益比较。root result SHA`347dc1315233a75db77b80c7d1baeb3aa480aab883e5fc3311f8d3a7b3bdce62`，context SHA`91acf84e9916cf30c37bf75fdb904308e81d974e349ceefa14e18eb77be43743`，关键产物两端同步，原父/child已退出。

全量只读分析报告`.tmp/p3_course_v2_dense_fixed_analysis.json`已生成（SHA`174f292baab976cf7ac46182fabc559f2d648c429bfb5be9998c3b6c80b08f14`），primary亲自读回核对。执行worker未捕获首次分析的明确stdout/exit，在报告已存在后重复调用，触发第13行防覆盖assert而exit1；该错误不是数据检查失败。脚本只在全部断言后写最终报告，原报告完整、未覆盖，不重跑仿真或再次重复分析。

primary接受该完整训练输入，继续唯一train_dense24_mixed/893513，24/warm12、SOC .55–.90、1800ticks/3600s环境/21600s全actor标注/900s重建，其他合同不变。当前仍无新fit/student/Q/PPO，formal入口2c4384c及16项local/server合成测试只是准备。mixed终态分析后才合并三个严格train输入开始既定fresh拟合。

mixed已启动timeout2869692/Python2869693，实际cwd=/home/user/projects/terminal、argv/protocol确认index3/893513、uniform SOC .55–.90及新profile/全部预算；25tick/309raw/26selected、85.83s只是初始前缀。first_before/live/replay及请求步骤已生成，无exit或终态。原fixed完成后才启动该臂，最多一个环境父与其标注child；核心冻结，下一检查仍60min。完整后primary分析，再决定既定fresh fit。

### 三个新训练周期完整，开始固定预算拟合

train_dense24_mixed/893513已COLLECTION_OK/exit0，957ticks、14N10E/0F/D/U，原timeout2869692/Python2869693及标注child自然退出。环境817.385s、标注4546.196s、105+N50重建22.472s，均在预登记预算内。11370raw=547actor+10823forced，547/547标签全OK，87R/460W、17种proposal；o2m为27R104W、m2i42R104W、i2f18R252W。busyLZ535、高速inner263、SOC<.55为339/<.65为453、retry恢复365，40次同接口WAIT→RELEASE。11947个RELEASE意图、11613次plan调用。primary已亲自核对全量分析；null/forced mask、全部actor-selected-label集合、原pre-state证书、snapshot config、policy STANDARD/base1、105+N50 join与首tick重放通过。

teacher/实际trace为48R/R、39R/W、6W/R、454W/W；52次正常K3批准rank1/2/3=35/11/6，另有2条无rank应急trace。原始单机teacher与批量执行语义继续分开，不按执行结果改标，也不把这些差异概括为错误监督。最小实际/扫掠81.238699/80.871572m，最低完成SOC .126563、能量189.844；10次应急提交/触地，151次尝试、297次候选安全拒绝、2次LZ拒绝、0能量拒绝。实际安全/hash/技术/censor/负能量无异常，506次safety_interventions是候选拒绝；应急服务占比仍高，不能据此声称业务最优或训练退化已解决。

终态关键文件两端SHA相同：root result `47ee6e33a01cf4402b85369dca5e3048bdcd14491b305865a496d97eb6633338`，context `32df9fdcd37c8c3821df18e3d29c75bcf2a6f5fc2986028016a207f1157432b5`。全量分析`.tmp/p3_course_v2_mixed_analysis.json`首次执行exit0，SHA `9e2bf7ceb9a27cd6fc36324056e35f0d0b25910ea2a7146e42de5c8d78c98912`。运行中曾读到不同时刻的label数量与hash差异，终态一致；单条annotation_progress的COMPLETED不是全臂终态。中途一个监控worker因工具上下文超长退出，换成只输出紧凑字段的只读监控，服务器父/child持续正常，未重启实验。

三组严格train共54请求39N15E/0F/D/U、24868raw，其中23570forced不入actor loss，1298有效标签=204R/1094W、19种proposal。没有纳入旧合同数据、开发smoke或Q。primary接受新数据作为本轮拟合输入；按既定91901–3各fresh5000updates/300s、固定final/canonical91901执行新的fit_initial，随后先核对拟合残差再逐臂student闭环。Q895501–3仍未消费，critic/PPO/formal未启动；采集完整不等于学生资格。

### 固定预算 fresh fit 完成，进入 student 开发闭环

数据接受记录提交`56ad3cbe5f17795b10b23792e1bd52a0538db54c`后启动唯一fit_initial，入口SHA同步检查通过，exit0/FIT_FROZEN。91901/2/3各完成5000次更新，耗时15.870/15.371/15.448s；输入恰为三组1298个唯一train ID，所有actor子模块确实更新。运行很快，未捕获live argv/cwd，只记录了PID文件2887166与显式cwd启动器和完整终态，不能补称曾观察live进程。三个checkpoint实际SHA与manifest一致，canonical91901为`16d3ee800263023386341cb92a6335bb63dd487a346181a0275bf4c3ecb63600`。

primary检查固定final训练内残差：91901 gate错18条（全部teacher R预测W），WAIT/RELEASE召回1.000/.911765，proposal top1/top3均1.000，joint .986133；91902 gate错4条（3W→R、1R→W），proposal top1 .990196、joint .995377；91903 gate错16条（全部W→R），proposal top1 .990196、joint .986133。没有挑选表现最好的91902，也没有延长训练、改loss或用训练内指标授予资格。canonical的18条漏放行中12条属于低SOC，需由下一步完整随机student闭环检查实际等待/能量后果，不能仅凭少量训练残差归因新的退化。

继续既定新smoke892864/929864；其通过后按顺序flow892853、fixed892854、mixed892855开发闭环，每次由primary核对终态。当前Q未消费、尚无新学生资格/PPO/formal。fit只证明已产生并冻结可检验的候选，不证明泛化或业务收益。

smoke已SMOKE_OK/exit0：2ticks、3actor全部RELEASE并批准，三接口覆盖，actual/hash/shield无异常，完整2tick tape精确重放，原请求/条件likelihood/mask/奖励记录全量检查通过；按2tick边界censored是预定smoke语义，不当完整周期。根result SHA`90ed34ec7e3767f44506f45daa816d529a466efff3c090e032508632ec007ae1`，终态两端一致，不重跑、不入train。

flow6_mixed/892853/929853已DEVELOPMENT_COMPLETE/exit0，667ticks、6N/0E/F/D/U，63.450s（含完整重放）。98actor=18R/80W，shield0，实际安全/hash/负能量/技术/censor均无异常，最小实际/扫掠111.691875/111.627825m，最低完成SOC .437690。6个请求的全口径等待mean/p95/max=14.893084/37.672412/39.760307s，request→touchdown=294.833333/327/327s，request→service完成=359.833333/402/407s；team reward130.069398。policy任务等级仍全1，各物理服务时长保持原任务类型。primary已逐条核对667请求记录/steps与原始请求likelihood、mask、teacher-off、episode end、奖励和全请求能量；`.tmp/p3_course_v2_student_flow_analysis.json`首次exit0。根result SHA`04ef358e75525d79bf68d24d0b931954e2ee2f27f19ceccb2ebe1c138bda9dae`，全部关键文件两端一致。此开发seed复用不冒称独立Q。接受下一唯一dense24_fixed/892854/929854，再视结果继续mixed；当前不提前访问Q或启动PPO。

fit/smoke/flow记录已提交`a32f6dd6f59132226852a93a30050964c993e1d9`，无remote/未推送。新根`p3_v25_neutral_course_v2_20261008/student_initial/dense24_fixed`已实际启动timeout2888670/Python2888671，cwd/argv/new profile/canonical16d3ee与892854/929854核对一致；首次真实PID检查375tick、271.327s，requests已399行，无root result或exit，属于运行前缀。启动worker误读了不含`_v2_`的旧根18N4E1F1U失败，primary立即根据新根protocol/progress核对，并让另一worker检查真实新PID；这是路径识别错误，没有新失败、没有重复启动或终止新进程。核心冻结，下一监控仍按60min，完整终态后才接受mixed。

### 新 student dense fixed 完整结束及业务比较

新dense24_fixed/892854/929854已DEVELOPMENT_COMPLETE/exit0：1024ticks、15N9E/0F/D/U，842.401s含完整精确重放；timeout2888670/Python2888671已退出。3477actor=156R/3321W，0shield；24个完整请求的最低SOC .246556、最低能量369.834、总耗能11603.367；最小实际/扫掠均83.134983m，actual安全/hash/负能量/技术/censor均无异常。9次应急提交/触地、87次尝试、0能量拒绝；363次安全干预是候选拒绝，未当作实际违规。原请求/likelihood/mask/奖励/complete-cycle及全部能量记录已由primary核对，完整分析`.tmp/p3_course_v2_student_fixed_analysis.json`exit0；root result SHA`57eb8864d7655dbf8eadd6aa52e645c946ea1e878e1682c535081ffbc8db8efc`，7项关键产物两端一致。

与同env/policy seed、同修复合同的旧5ca actor迁移诊断相比，新16d3ee不是全面改进：N/E由17/7变15/9，reward377.702031→365.351956，mean wait181.643113→189.511015s；但wait p95为471.553637→433.839169s，touchdown p95/max为857.2/970→770.8/850s，service完成p95/max899.35/1000→863.55/880s，总耗能11624.850→11603.367。mean touchdown/service完成均增加2s。保留这些方向相反的业务结果，单个开发场景不能归因或证明新拟合性能优于旧actor，更不是PPO退化恢复证据。原未修复失败18N4E1F1U没有完整24请求口径，不能拿其已完成均值作公平收益比较。

当前完整周期/实际安全/能量与原有teacher-free准则无阻塞，E并非F且未预登记零应急门槛；primary接受继续唯一dense24_mixed/892855/929855。三种学生开发条件完成后再决定冻结候选并收集Q；不会凭当前业务混合结果自动准许正式PPO。Q启动/全量只读分析helper已准备并仅语法编译，尚未调用或读取Q；它要求三种学生完整、固定fit source，并沿用预留Q895501–3、每臂3600s环境/21600s标注和原数据上限。

固定场景记录提交`cf445de51b32f13bc5e476fb470139541658a616`后，新mixed已启动timeout2894808/Python2894809，真实cwd/argv、892855/929855、新profile/canonical16d3ee均核对；首次progress25ticks/36.381s、requests27行，无result/exit，是启动前缀。源SHA/Syncthing与seed查重通过，没有旧根复跑。核心冻结、60min监控；mixed完整后才判断Q。无remote/未推送。

### mixed运行时证明分辨率失配：失败保留、同状态诊断与修复

mixed/892855已BLOCKED/exit1，489个成功tick后在now489抛出`SafetyInvariantError: Nominal holding contract is not recursively safe; physics execution blocked`，原Python2894809和timeout2894808退出。当时11N/13U、12active/1service，1548actor/152R，既有物理轨迹最小间隔81.043160m、实际违规/hash/能量耗尽均0，但完整周期与精确重放没有完成。失败不能计入学生资格。原episode telemetry是最后成功step的汇总，shield_fail_closed仍0；`failure_after.pkl`的runtime原始counter已为shield_fail_closed/joint_safety_fail_closed/unrecoverable各1，必须保留该执行屏障事实。

primary亲自比较可信失败前后快照：本step自动应急层为middle UID31提交了到LZ5的direct-C²轨迹（489→581.520768s），随后runtime在其与inner holder UID2的当前30s horizon拒绝。UID2保持16m/s标称等待，提交/执行所指计划hash一致`2de9b267c60c790118908c1cda77503dcd0b6d57f4bbc89cf32b7621a8962889`。failure_before SHA`f703f5ae408010c4fe62f74f5c2e70ad7657ebd974f4787f2c3c0fba0c504b04`、after SHA`e8280556cb012bfde56d1b540708bb68f62e42a5af623a92664015c517b1c517`均两端核对；原失败root result SHA`82b77904e3b180b508f3fdfa75dabd29b56de3a56600956158e21d981f6a69d8`。这些preexisting_artifact被最小只读探针复查，不消费新reset/teacher/学习。

`.tmp/probe_p3_mixed_certificate.py`首次exit0（newly_run），同一轨迹SafetyFilter仍approved。冲突区间518–519s：runtime的1s网格原始扫掠间隔80.410774m，减曲率和源插值余量.420650m后变79.990124m；physics_dt=.2s而safety_sampling_dt=1s。直接消除重复插值余量也只有79.997629m，说明不能只删该余量解决根因。提交检查的.2s连续曲线下界为80.363964m；.1/.02/.005s的下界分别80.376076/80.379553/80.379685m，.005s密采样实际最小80.379696m（518.825s）。holder两种采样坐标差最大7.1e-13m。结论为**不同证明分辨率导致的保守拒绝**，未发现这一对真实低于80m；密采样用于诊断，准入仍必须有连续曲线下界。

primary选择最小运行时修复：只对当前V3 nominal+swept_curve_v1合同下粗网格未获证的pair，在原physics_dt上重建两条轨迹并重新进行相同加速度余量的point/swept检查；物理网格上的holder直接用原积分点，不再扣不存在的源插值误差。已获证pair不细化，缓存保存最终判定；未知轨迹、实际冲突仍fail closed，80m阈值/动力学/teacher候选/轨迹生成/原监督标签与优先级合同均不改。旧mode保留原行为。新回归包含该应急几何、真实冲突、样点间内凹冲突、未知轨迹、有限加速度/末尾分数步、缓存与旧合同隔离。首次65tests中64通过，1个新fixture因env构造自动生成warm holders、未清空而失败；primary修正fixture，不改生产约束，失败日志保留。

验证顺序冻结：定向tests、可信完整失败状态的current+successor proof；再同892855/929855/canonical16d3ee完整重跑到新`student_runtime_refined/dense24_mixed`，原失败不覆盖。采样证明修复不修改teacher/filter或训练输入，所以沿用已冻结1298标签和三个final模型，不重新拟合或挑模型。既有flow/fixed在更粗证明下已完整通过，继续保留；mixed新证据须完整精确重放且全N/E/F/D/U、能量与业务指标核对后才判断学生资格。Q895501–3仍未消费，当前禁止Q/critic/PPO。若重跑仍有阻塞，继续具体同状态诊断；用户授权的最终目标仍是正式训练就绪并实际启动，不在本次失败处结束。

修后65项local/server定向tests均通过（11.22/6.35s），8项源码/测试SHA同步，服务器无旧P3进程。primary的完整可信失败状态探针`.tmp/verify_p3_mixed_refinement.py`首次exit0：12active/42相关pair，current+successor证明全部approved，最小连续下界80.363964m，耗时.047s，原now489与应急plan hash不变。没有实际推进、改计划、改阈值或训练。该单状态结果解除本次已定位的证明矛盾，但仍须完整重跑验证后续状态；接受提交后启动新mixed路径，预算/actor/seeds不变。

修复提交`e340a8947be10c0d601c73d32782514c20ce7ee3`（无remote、未推送）后新mixed已启动：timeout2901702/Python2901703，真实cwd=/home/user/projects/terminal、argv892855/929855及新student_runtime_refined目录核对一致。protocol/first_before/requests/progress均已生成，相邻exit尚无；不是终态或学生准入。Syncthing源码SHA先核对，最多一个环境父，核心冻结；原student_initial失败及其他已完成臂不重启。

### 修复后mixed完整通过；冻结候选进入独立Q检查

新`student_runtime_refined/dense24_mixed`已DEVELOPMENT_COMPLETE/exit0（newly_run），898ticks、17N7E/0F/D/U，24请求全部触地并完成物理service；2500actor=163R2337W，shield takeover/fail_closed均0，772.015s含完整精确tape replay。minimum actual/swept=80.413918/80.410774m，actual hard/NMAC/collision/checked-executed hash全部0，75次plan一致提交；401次safety_interventions是候选拒绝，不是物理违规。7次应急提交/触地、135次尝试、0能量拒绝/负预算/耗尽，最低完成SOC .181618。全请求wait mean/p95/max=143.906689/346.158118/359.437154s；request→touchdown=423.083333/687.45/690s；request→service完成=482.666667/759.10/793s，reward392.094247。

primary亲自核对所有898条请求、tensor、actor mask、请求likelihood、teacher-off、奖励、terminal语义和24个能量结局；全量分析`.tmp/p3_runtime_refined_student_mixed_analysis.json`首次exit0（SHA`d04f9f9bcb2ab0873ed3537d1d1e785ddce5b6464411950ea73e606ad2539cf2`）。7个关键文件两端同步，root result SHA`224443220ec6d7a842acee0315b876e87354bed8d9800e8ae16570c76d7f19ca`，requests SHA`db1aa03a3a18724e8b32a03abbeeeb46547ad7ae4725ce408de14b3b4dca10fe`，steps SHA`e7f2b759b2b3c6a7a348224f1c201a2fac470e6e611cfc6e796b91e70c0bc64a`。原2901702/3已退出。运行时实际采样点最小距离高于密采样连续极小值，不能把80.413918误写成全连续时间最小值；安全依赖修后的连续下界证明。

当前同canonical source的三个完整开发周期共54请求38N16E/0F/D/U：flow6N，fixed15N9E，mixed17N7E；均teacher-free随机请求且完整重放，实际安全与能量无阻塞。它们支持这些条件下的环境可执行性与学生闭环完整性；应急仍16/54，规则采样数据与学生访问分布也不同，不能称业务最优、独立泛化或P3训练退化已解决。原失败prefix与完整结果分别保留，未将失败改标。

primary据上述证据接受下一项**既有预留Q的采集和一次冻结模型评价**，不授PPO资格。三模型维持fit_initial的91901/2/3固定5000step final，不再调参/挑seed，canonical仍91901。Q按895501 flow6_mixed→895502 dense24_fixed→895503 dense24_mixed顺序，输出`Q_<scenario>`；三臂此前未reset/读取。每臂rule行为、全actor原state-only teacher标签，完整周期（flow1210/dense1800ticks）/3600s环境/21600s标注/900s重建、2048query/51200RELEASE意图、outer27000s，超限保留失败不补标。原单机teacher非联合批准/非全局最优边界不变。逐臂primary先查业务/安全/完整行与独立evaluation decoder，再继续下一臂；全部齐全后仅一次`p3_v25_task_imitation --mode evaluate`评价全部三seed final，Q不进训练/阈值选择。沿用已有WAIT≥.90、RELEASE≥.70、macroF1≥.80、proposal top1≥.85/top3≥.95、joint≥.85及支持/非坍缩判据，不因结果弱而事后降低。

新runtime修复不改变SafetyFilter/teacher候选或采样状态语义；原train/flow/fixed保留其核心版本身份，后续critic输入路径明确为student_initial下flow/fixed以及student_runtime_refined下mixed。Q/helper已包含runtime源码SHA并检查这三条成功路径。此时critic/PPO/formal仍未执行，只有Q完整且独立拟合资格实际通过后才进入已登记有界学习步骤。正式训练最后启动后不设heartbeat；当前长标注继续60min静默监控。

附加同seed前缀检查：原失败的489个成功记录与新完整臂前489条的tape、pre_hash、post_hash、reward、actor_mask、likelihoods全部逐字段一致（各0差异）。first_before pickle bytes不同，但整个可观察状态/决策轨迹前缀精确一致，不用pickle字节相等代替语义相等。这支持本次修改确实在原执行屏障处解除阻塞，未通过改变早期动作绕开该失败状态。

### Q第一臂完整采集（尚未评价模型）

学生接受/冻结说明提交`0ee4574de9b9b2086a486bf4f7f54be1ec7133cd`后，Q_flow6_mixed/895501唯一启动；timeout2903794/Python2903795、cwd/argv/Syncthing及未消费seed检查通过，随后自然COLLECTION_OK/exit0退出。683ticks、6N/0E/F/D/U，1836raw=19actor+1817forced，19/19 OK独立标签均RELEASE，proposal12/13/17为16/2/1。flow没有WAIT标签是这一个预留场景的结果，不补样，不单凭该臂做二分类资格判定。o2m/m2i/i2f标签7/6/6；busyLZ13、SOC<.55为4/<.65为9、高速inner6、retry后1。teacher19R中1条与实际WAIT不同（tick315 UID4 teacher proposal13、行为nominal12被HEADWAY拒绝），原标签保留，未混为联合批准。

环境32.404s、标注26.571s，29个RELEASE意图/43次plan；实际/扫掠最小95.209671/95.062213m，最低完成SOC .430094、剩余能量645.141，技术/censor/实际安全/hash/应急/耗尽均无异常。primary完整检查1836raw/683steps/19selected-label/90D null/mask/105+N50 join、可信快照及首tick重放通过，Q只用evaluation decoder，模型未加载/评分/修改。分析`.tmp/p3_course_v2_Q_flow_analysis.json`首次exit0，SHA`2ffb0791ce0afd9cdeb8e199f6b7b83197d5179b19afec7d4cdc578d23fe0ffd`。标注实际位于`labels/window_XX.json`，没有通用labels.jsonl不是缺标签。关键文件两端同步，root result SHA`05f502b625e7a14257036709007c3b26ab51291c75676da2a84c2844a85507fe`、context SHA`25a07fc9948e8200fc53b13f2e1e24be7ccb7bb4ed580a5065434954e0c65c83`。

primary接受继续唯一Q_dense24_fixed/895502，原预算/固定候选/一次最终评价不变；Qmixed895503尚未消费。其他成功/失败不复跑，当前仍无真实critic/PPO/formal。Q标注预计dense约1–2小时，保持60min监控，不因为首臂快而提高轮询频率。
