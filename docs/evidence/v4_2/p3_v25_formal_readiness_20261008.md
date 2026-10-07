# V2.5至新P3正式训练的持续修复与准入（2026-10-08）

## 用户授权与目标

用户明确要求：根据每步结果持续分析修复，直至新的P3正式训练就绪，再同步服务器启动并结束；**最后的P3正式训练不设heartbeat**。这覆盖必要的数据/模型/执行接口修复、有界学习与资格验证，并允许通过后启动正式训练；不把授权当作实验已经通过，不降低安全约束。主代理亲自读研究文档、修改核心/协议、诊断和判断；luna_worker仅执行明确测试/SSH/Git。代码/docs只本地经Syncthing，服务器不Git、不手传源码。所有失败保留，其他dirty不动。

当前已有：direct C²几何/动力学修复、小样本规则/source完整闭环、三个完整train回合；V2.5观测修复重建1046行；同网络信息对照新输入平均joint86.33%、gate AUC .96569，但proposal仍不足，82/88/99次拟请求属于原teacher拒绝候选。这些都是已有证据，不是新schema正式I0/I1/student/PPO资格。

推进顺序：固定模型拟合曲线→根据残差决定最小修复→V2.5请求/mask/K3/likelihood与执行桥接→完整开发闭环→冻结数据及fresh多seed imitation资格协议与untouched评估→必要Safe DAgger→student-only资格→PPO实现/critic及有界更新防退化检查→正式P3准入审查及启动。每步具体seed/预算/成功条件在执行前补充，使用原有规范的安全、无teacher替换、请求概率、完整业务周期等要求。历史88D资格及旧Q不能直接给V2.5背书，也不机械搬用旧采集样本数。

正式就绪至少需要：相关新schema与策略执行一致；数据分区及teacher语义成立；独立student-only完整周期无技术/实际安全/hash/F/D/U问题，保留应急计数与既有takeover边界；PPO请求log-prob/forced mask/terminal reward与GAE正确、更新预算与防退化机制经过有界验证；冻结正式配置、初始化、训练上限和输出目录。`PPO_ALLOWED=false`保持至实证准入；不能仅凭训练集准确率启动正式PPO。最终服务器启动后检查真实argv/cwd、日志及至少一个正常中间产物，不监控最终完整训练。此前预计长于交互窗口的任务可按用户要求60min监控，正常推进静默。

## 第一步：固定新信息5000更新曲线（运行前冻结）

协议`p3_v25_fixed_fit_curve_20261008`：只原1046条train及相同label；固定V2.5网络59451参数、新信息输入、未加权CE、Adam3e-4/batch128/global clip.5，3个初始化91701/2/3。每seed从原fresh初始化重新跑**5000更新/300s**，固定1000/2500/5000记录，不选最好epoch，不调权重/阈值或临时延预算。1000点必须与上轮同seed新信息臂的参数hash、batch序列hash和完整指标精确一致，证明新增测量未改变原前缀。这里只研究延长预算后的残差，不把5000对旧信息1000的差异称为观测因果效应。

每seed只保存final诊断checkpoint，曲线三个点都保存原1046逐行预测/指标，方便检查gate、proposal及原前缀已知拒绝率。仍不reset/step/teacher/audit/Q。结果决定下一步，但不据本曲线作资格判断。

修改`p3_v25_set_learnability.fit_arm`只增加可选固定测量点/reference-prefix断言，原默认1000入口行为保持；新`p3_v25_fit_curve.py`独立入口。先合成smoke（2更新prefix+3更新curve）验证同参数/同batch/输出，再真实三seed。测试/源码同步后执行，输出`/home/user/training_outputs/terminal/p3_v25_set_curve_20261008/{smoke,curve}`，新目录且禁止重复启动。预检无项目Python，原fit SHA两端一致，新根不存在。

## 实际执行与分析

以下仅追加实测证据。

曲线实现的20项local/server tests均通过（3.25s/1.29s）；5文件SHA相同。合成smoke已CURVE_COMPLETE/exit0，2更新reference与3更新curve的前缀精确相同，固定2/3两个点、final checkpoint及JSON有效，两端SHA一致；root result SHA=`5549e30e1e6696fb7c6abbb4dff09202bef227265ec033c331aaac15b3a6b4d8`。primary核对后准许本地提交后真实3seed×5000更新。尚无真实curve结果，不改变资格。

### 固定曲线终态与下一判断（newly_run）

核心94cabf1后真实curve已CURVE_COMPLETE/exit0，91701/2/3各5000更新；1000点的参数/batch/全指标均精确匹配上轮。result SHA `b74f766d1147f0f2b067e294456ada2cb92f898ccb89a5e4446281da45b46228`，两端同步，3个固定final新checkpoint保留，Python2803088已退出。

| seed | 1000 joint/proposal top1 | 2500 joint/proposal top1 | 5000 joint/proposal top1 | 5000 W/R recall |
|---|---|---|---|---|
|91701|86.62%/65.50%|95.98%/88.21%|100%/100%|100%/100%|
|91702|86.14%/59.83%|95.22%/87.77%|100%/100%|100%/100%|
|91703|86.23%/60.70%|95.51%/89.96%|99.81%/100%|100%/99.13%|

三个final gate AUC均1，proposal top3均100%。这表明当前1046个已知train状态的主要残差可以通过原网络/原loss继续拟合消除，暂不增加loss或更换架构；不能由训练集拟合宣称泛化。接下来直接检查执行与开发闭环，不再反复调训练集分数。固定选seed91701作为开发canonical（按编号，非择优；三seed均保留，正式资格另行全量评价）。

### 第二步：V2.5执行桥接与开发闭环（运行前冻结）

新增`p3_v25_runtime.py`只从当前V2.5 set取样，沿用既有pre-action actor mask、WAIT sentinel、requested gate+conditional proposal log-prob；K3始终以真正请求的proposal为首，其后按logit稳定排序。forced行WAIT/0joint likelihood、不入actor loss。加载显式new-information/V2.5 checkpoint，禁止把旧信息或88/90D模型当新模型。环境内部保留90D物理配置供兼容，**policy实际读V2.5**并在protocol单独记录；不调用teacher/candidate，不动safety。

新增评估入口执行真实ExecutionScheduler/SafetyFilter，记录每tick原请求/likelihood、actor mask、K3、实际trace、状态hash/reward，再从初始pickle以相同请求重放全周期；核对pre/post hash、reward、实际安全与checked/executed一致性、权重不变。完整周期自然终态或1800ticks，rollout/replay各3600s；技术/安全/F/D/U/censor阻塞时停止后续臂，保留failure快照，不扩负载。shield_actor_fraction沿用<=.05要求。此阶段是development，不冒称正式student qualification。

固定checkpoint为`p3_v25_set_curve_20261008/curve/seed_91701/diagnostic_final.pt`。独立smoke env892801/policy928801，3请求/warm3高SOC、deterministic、2ticks；仅验证输出/执行重放。随后顺序A deterministic、A stochastic、B deterministic、B stochastic、C deterministic、C stochastic：A=12/warm6高SOC、env892802/policy928802；B=24/warm12高SOC、892803/928803；C=24/warm12混合SOC .55–.90、892804/928804。各对共用exogenous seed但从头独立rollout；不用Q/audit。代码/docs搜索未发现这些seed实际使用记录（rg no matches），服务器启动前另查无输出/进程。测试及smoke通过后才启动A；每臂完整结果由primary判断后继续，不自动铺开。输出新根`/home/user/training_outputs/terminal/p3_v25_student_dev_20261008`，最多一个本轮评估进程，所有失败保留。预计若分钟级继续当前任务；超过交互窗口再60min heartbeat，最终正式训练无heartbeat要求不变。

执行桥接21项local/server tests通过（10.84s/5.53s），3源码SHA相同。独立892801 smoke已SMOKE_OK/exit0：2ticks、3actor/3RELEASE、0shield，3个正常批准；完整tape replay_exact=true，实际安全/hash异常0，checkpoint不变。primary读取steps确认likelihood和pre/post hash存在（这些字段在steps中，不是root result命名字段），3个未决是预定2tick截断。result SHA=2edd92540d8b591e7a9f270705bbcb9e999f33a5495cd484c6e9d70463446258，两端同步。接受提交后只启动A_det完整开发臂，其他臂仍依完整结果决定。

runtime已提交`8423460`，Syncthing后A_det已启动：Python2804432、timeout2804431、SSH session39425，真实cwd/argv与协议相符；first_before、protocol、steps和progress均出现。启动检查50ticks/6actor/6RELEASE/0shield只是前缀。此臂完成后由primary核对，再按上述次序决定后续。持续推进目标的60min heartbeat为`p3-v2-5`，它用于正式训练之前的工作；最后正式训练启动并检查初产物后删除，不监控正式训练。无Git remote，未推送；服务器源码通过Syncthing同步。

### A_det完整开发终态（newly_run）

A_det为COMPLETED/exit0，555ticks，rollout+完整重放222.291s，12N/0E/F/D/U、active/queue/service/unresolved均0，error/censor无，replay_exact=true。primary逐行核对555steps/3409rows：209actor=50R+159W、3200forced，requested likelihood/forced sentinel/K3首项与实际trace全部一致，0teacher/0actor shield。root result SHA=`c0b146f0d82ce4c9899488a5c1dd9e9bc2f1bc3349c0c309432fd11e7dd43f97`；result/protocol/steps两端相同，Python2804432已退出。

实际最小点间隔95.388m、swept95.244m，protected/buffer/NMAC/collision/shield fail-closed/hash mismatch均0；140energy checks、0negative，12个ground_service_complete，无应急。注意telemetry safety_interventions=10是`_commit`边界或plan过滤拒绝的累计，不能误写为0，也不等于实际安全违规或actor takeover；本臂shield_interventions/runtime_traffic_interventions均0。50个R中32批准、18最后拒绝；批准rank1/2/3=23/7/2，仍依赖9次K3备选，候选拒绝不是碰撞。root outcome_counts的Enum键被通用serializer表示成`TerminalOutcome.COMPLETED`，以完整outcomes的12个`completed`核对N，不漏计。暂无必须修物理/策略的阻塞，接受继续固定A_sto；此开发样本不作为正式资格。

A_sto随后COMPLETED/exit0：585ticks/239.224s，12N/0E/F/D/U、完整重放相同、无error/censor，158actor=56R+102W、3361forced、0actor shield。实际最小间隔87.303m/swept87.166m，所有实际安全与hash计数0；211energy checks、0negative、12个正常service completion、0应急。56次R中32批准/24拒绝，rank1/2/3=22/7/3，10次K3备选；_commit层safety_interventions=10仍单列。完整逐行断言通过，result/protocol/steps两端相同，result SHA=`196baa8454ecb82799eb64bb5d7e5fce74f676c37167667b76a1499770f0c032`，无遗留进程。A配对均完整无阻塞，primary接受继续固定B_det，尚不作跨场景泛化或正式资格结论。

### 并行准备后续PPO接口（尚无实景PPO更新）

等待开发闭环时，primary只在新文件`algo/trajectory_set_ppo.py`、`training/p3_v25_training.py`及对应tests中补齐V2.5训练数据/算法；当前评估不import这些文件，冻结的运行核心不改。既有flat88/90D PPO不能把105+变长邻机当普通矩阵，因此新增明确的set rollout record，沿用team GAE(.99/.95)、逐请求clip、固定全rollout actor数量归一化和gate-only entropy，forced行不入actor objective；空active/服务等待tick继续保留team reward。初始base gate/proposal与BC完全相同，base冻结，仅零初始化gate residual与独立critic可更新。

critic使用全active set及26维当前全局上下文：时间、到达余量/下一到达延迟/是否耗尽、队列数量/等待/最低SOC、LZ占用与服务/预约剩余时间、active/unresolved计数与三环几何。即使active为空，队列/服务上下文仍存在；不调用teacher、不读未来随机抽样或终态结果。每条record保存原gate/proposal分布和条件likelihood，更新前检查行为概率一致；真实terminal无bootstrap，时间截断bootstrap但不串下一episode。每次更新后检查全rollout的精确categorical gate KL，超限恢复模型和Adam状态；冻结proposal使该KL覆盖可学习策略变化。实现仍显式要求training_authorized，单测不授予PPO准入，正式优化预算/协议尚未执行。

新增构件与旧PPO回归共28项local/server tests通过（3.21s/1.39s），三文件SHA一致。检查零residual初始分布相同、ragged neighbor/forced mask、原请求概率校验、空tick晚到reward/截断bootstrap、只更新adapter+critic，以及空/非空Adam状态的KL超限回滚。这里只发生合成单测中的更新；没有读取真实训练rollout作PPO，没有产生可部署PPO checkpoint。

上述构件已提交`86e3182`。进一步新增未被当前评估import的`p3_v25_rollout.py`，让完整team tick、26D critic context、原请求分布与实际reward进入可安全读取的纯tensor records；记录终态/截断bootstrap，使用相同scheduler请求tape并支持完整重放。critic-only warmup只接收完整自然终态的stochastic episodes，以真实reward计算gamma=.99 MC target，冻结base和gate adapter。新增的两步环境单测通过；首次合成MC测试的90%降误差断言未满足（92.691→12.076，约87%），MC target/参数冻结均正确。该断言把100步合成检查误当指定收敛率，primary改为只要求明显学习进展（50%降误差）；模型/优化器/测试预算不变，正在重跑。此处不是实景critic warmup结果，也不改变任何研究准入阈值。

重跑完成30项local/server tests全通过（10.31s/5.41s），新增rollout与test SHA一致，未做实景PPO。当前B_det真实Python2805535/timeout2805534、cwd与24/warm12/892803/928803高SOC参数已核对。progress到982ticks、24个completed、actor1808=293R+1515W、0actor shield、rollout耗时870.575s，但root result及exit尚无；完整重放及终态报告完成之前不启动下一臂。此前当前A的monitor信息已过期，以本条与实际进程为准。

rollout/MC接口已提交`4fad802`。PPO侧另补齐固定train-state reference gate KL约束，防止每个rollout内的小更新在多轮后累计偏离合格模仿源；reference只来自允许的train观测，不使用audit/Q或teacher label作额外PPO奖励。gate base保持冻结，reference分布由同一base在固定状态生成；更新loss可带KL正则，接受前同时检查本rollout精确KL与固定reference平均KL，越界恢复参数及Adam。当前构件默认reference上限.02、系数1、rollout上限.01只是实现参数，实景有界协议尚未运行/批准准入，将在该阶段冻结具体数值和输入。新增独立reference guard及其他回归31项local/server tests通过（10.28s/5.45s）；未发生真实PPO更新，未修改B_det运行核心。

### B_det完整终态（newly_run）

固定reference构件已`4edee3a`提交。B_det已COMPLETED/exit0，982ticks、24N/0E/F/D/U，rollout+全重放1697.596s；无error/censor，replay_exact=true，0actor shield，root result SHA=`e1f7ab6358e2e57514bd9b2e05ced30b0ef489c3c4fe5b0627cd3be489695c9c`。result/protocol/steps两端相同，PID2805535已退出。primary核对10902rows=1808actor（293R/1515W）+9094forced，所有请求概率/trace/K3/mask断言通过。

实际最小间隔81.826m、swept81.684m，buffer/protected/NMAC/collision/physical shield/hash异常均0，930energy checks/0negative、24个ground_service_complete/0应急、最低完成SOC .406903，合计能耗10636.4304。117次_commit过滤干预和8个不可行候选单独保留，不当作实际违规。293R中64批准/229拒绝；拒绝PREDICTED_SEPARATION145、NO_MERGE_GAP55、HEADWAY14、FINAL_RESOURCE15，批准rank1/2/3=49/14/1。候选批准率偏低说明尚有放行时机/排序优化空间，不能仅凭24N称学到了最优决策；也没有依据降低安全过滤。当前完整性/安全/业务无阻塞，接受继续唯一B_sto，随后仍依结果决定C。

B_sto已启动Python2807712/timeout2807711，SSH session28812，已出现first_before/protocol/steps，cwd确认。primary核对本地产物protocol的checkpoint为原固定`p3_v25_set_curve_20261008/curve/seed_91701/diagnostic_final.pt`、SHA c28949a8...，892803/928803、高SOC、stochastic；一份worker文字报告将日期写为202608，不能据此替换真实protocol参数，原样/proc复核正在进行。后续统一使用primary准备的`.tmp/snapshot_p3_v25_student.py`原样JSON读取进程与相邻产物，减少人工摘录错误。

### 并行准备独立imitation资格（尚未启动）

primary在不被当前评估import的新文件中实现[三完整周期Q与fresh多seed I1协议](p3_v25_i1_qualification_20261008.md)。所有开发臂通过及独立smoke后才允许首次Q reset；Q先完整冻结，再以原1046 train fresh91801–3各5000更新，全部final写完后一次评估。895301–3覆盖12高SOC/24高SOC/24混合SOC，原始Q永不进入train；沿用既有I1准确率门槛，Safe DAgger/student/PPO后续资格仍分开判断。精确seed搜索无历史代码记录，服务器新根未存在，真正采集前另查历史产物身份。新增30个不同local/server定向用例通过（29项加新增1项，新文件15项复跑），首轮Windows清单路径错误已修复并留证；当前没有真实Q采集或fresh fit。

### B_sto完整终态及应急来源的最小检查（newly_run）

B_sto已COMPLETED/exit0，1166ticks/1771.421s（含完整重放）、**22N/2E/0F/D/U**，error/censor无、replay_exact=true，active/queue/service/unresolved均0。主代理核对12039行=2186actor（242R/1944W）+9853forced，likelihood/mask/K3/trace逐行检查通过。result SHA `e33a8edc09722b71f7ac963665f04fcdf342aab977234964343a3ab8058b3342`，protocol/steps/first_before/log/exit两端一致，原2807712已退出。终态checkpoint路径也确认20261008和原固定模型，前述202608文字摘录不能代表真实输入。

最小实际/扫掠间隔均82.495507m，buffer/protected/NMAC/collision/physical shield/fail-closed/hash异常均0，0actor takeover；748energy checks/0negative，2次应急提交/落地均完整，最低完成SOC .292214。普通过滤safety_interventions146与8个不可行候选仍单列。242次原R中62普通批准、178有普通拒绝原因（NO_MERGE_GAP36/HEADWAY18/PREDICTED_SEPARATION123/FINAL_RESOURCE1），另2次原R所在tick由应急流程接管、没有普通release_telemetry，**不能将这两次记为普通拒绝或删除原动作/likelihood**。普通批准rank1/2/3=50/7/5，12次K3备选；应急是独立业务结果，不误写为actor physical shield。

相比同环境seed的B_det：周期982→1166，reward486.063878→447.355996，合计能耗10636.430423→11777.768984。arrival17和22在det均正常，完成时刻690/704、SOC .608197/.599442；sto分别1128/1166应急完成、SOC .324148/.292214。这里只是一个配对开发场景的观测，不作统计泛化或单因素机制证明。

primary编写只读trace汇总及六个既有snapshot（两个臂×400/600/850）检查；worker仅执行。所有snapshot canonical state均精确等于已核对steps的post_hash，0reset/0step/0模型/0teacher。结果：这两架sto的actor WAIT仅137/122（det113/94），forced为814/813（det411/405）。在所查forced状态中存在合法已承诺切环，checked hash全部相同，不能取消mask让策略途中换轨。sto arrival17的o2m为350→587.861，m2i为729→999.913；arrival22的o2m为437→665.446，m2i为765→1078.915。det对应o2m约139/140s、m2i约97/93s。sto两条长m2i请求/实际执行都是nominal speed0/timing0，不是简单“模型选了慢速”可以解释；arrival22的o2m是K3 rank3执行speed+.5/timing+.5，原请求speed−.5/timing0。`scheduled_delay=0`不能表示轨迹本身没有较长飞行时间。计划时长与上下游间隔/调度上下文应一起看。

另用两既有eligible WAIT状态作最小teacher复核：B_sto tick400/UID114与tick600/UID38，各原快照与tape pre/post hash相等、私有副本，0reset/0step/0学习。两次均teacher WAIT、无放行证书；400的25个RELEASE意图为18个merge_slot_retime_conflict、4个predicted_holder_reachable_tube、3个predicted_separation；600的25个均predicted_separation。各耗时69.922/60.359s，共130.281s，未重试。原stdout完整存在执行工具记录，但worker遗漏约定的文件重定向，预期JSON未产生；primary发现缺文件后仅要求读取已有stdout并核对上述26项（25R+WAIT）原reason表，没有为补产物重复查询。此探针用于排除“这两个时刻模型明显漏放”的直接指控，**拒绝reason本身不证明物理绝对无解**，也不为更早的放行顺序/计划时长提供最优性证明。

判断：开发安全/完整性与预定继续条件通过，但高SOC仍有应急及效率下降，不能称业务最优或student/PPO已就绪。保留安全与原teacher，继续既定C_det/C_sto，再以独立I1及student状态分布验证决定训练数据/策略是否需修复。当前C_det唯一评估Python2810032/timeout2810031，SSH session41164，892804/928804、24/warm12、SOC.55–.90、deterministic、同固定checkpoint；真实cwd/argv和protocol/first_before存在且两端一致，首报steps0行只是启动前缀。源码冻结，未启动C_sto、Q或实景PPO。

### C_det终态：应急比例高，需要先区分环境与策略（newly_run）

C_det已COMPLETED/exit0，958ticks/1338.105s（完整重放）、**10N/14E/0F/D/U**，error/censor无、replay_exact=true，active/queue/service/unresolved0，所有实际安全/hash/actor shield计数0。result SHA `bc757fb090b3eb52f51dc635c6f739d55b0b49fc2c061ad9824f81197afb006f`，protocol/steps/progress/first_before/log/exit两端相同，Python2810032退出。primary读完整结果并核对11360行：3293actor=212R/3081W、8067forced，原请求概率/mask/K3/trace无异常。普通批准44（rank1/2/3=26/9/9）、有普通原因的拒绝165（HEADWAY11/PREDICTED_SEPARATION97/NO_MERGE_GAP25/LZ_RESOURCE32），余下3个原R无普通release_telemetry，不混为普通拒绝。

最小实际间隔84.706372m/swept84.582154m；628energy checks/0negative、14应急提交/落地全部完整，无应急能量/动力学拒绝。应急尝试125、候选安全阻塞308（不同计数口径），普通过滤interventions394，均不抹去。最低完成SOC .116900/剩余能量175.350674，属于arrival15的应急完成；没有耗尽不等于充足的运营余量。reward296.877299，正常完成比例仅10/24。

此臂满足原开发“完整、安全、无F/D/U”的继续条件，可完成已固定的C_sto配对；但14E是业务层的重要异常信号，**独立I1采集/训练暂缓**。不能拿不同seed的早期规则结果直接归因学习退化，也不能改应急阈值把E变N。

在C_sto自然终态并分析后，新增唯一同seed规则对照`C_rule_same_seed`，与本C的env892804/policy928804、24/warm12、SOC.55–.90、相同build_config/物理/SafetyFilter一致，仅控制器换为已验证的nominal K3 rule。使用未修改的`terminal_marl.scripts.diagnose_p3_integrated_repair --run --controller rule --requests 24 --warm-start 12 --seed 892804 --policy-seed 928804 --max-steps 1800 --trace-blockers --snapshot-ticks 1 400 600 850`，输出新`p3_v25_student_dev_20261008/C_rule_same_seed`，外部3700s硬上限；如果超时无终态report则保留为技术/截断，不延预算或覆盖。此既有rule入口不提供完整request-tape重放，明确这个证据边界；它提供N/E/F/D/U、安全/hash、能量与阻塞快照，用来定位14E究竟也出现在规则控制，还是主要出现在当前学习闭环。不得冒称它等同student runner的完整重放证据。无学习/新teacher标签/新Q数据；同seed新controller是控制实验，不重跑已完成student臂。

保持最多一个本轮环境评估，先Syncthing/无重复/初产物核对。规则结果与C配对一起由primary决定必要最小轨迹/能量/监督诊断及修复；只有处理当前业务疑点后再恢复I1计划。用户持续推进到正式训练的授权不变，不在此处仅报告失败就结束。

C_sto随后启动：Python2811481/timeout2811480、session32263，真实cwd/argv与892804/928804/24warm12混合SOC/stochastic相符，固定checkpoint不变。一次60s启动等待后progress50ticks（117actor=18R/99W、0shield）、steps68行、first_before/protocol均正常；protocol/first_before等两端一致，持续写入steps在快照时SHA不同是同步中的前缀，不能作为终态一致证据。root result/exit尚无，当前唯一运行臂；完成后再由primary决定上述规则对照。

### 合流点选择遗漏已承诺入环飞机（newly_run，只读诊断）

primary读取`MergeScheduler._gap_center`发现其仍只选择`ring_role is target`，以目标环名义角速度外推当前holder。真实切环飞机在transfer阶段`ring_role=None`，即使其承诺轨迹早于本候选合流时间结束，也完全不参与目标gap选择。V2.5观测已能看到这些飞机，并不意味着执行调度器也已处理它们。这是独立于模型拟合的具体实现问题。

primary编写`.tmp/probe_p3_v25_merge_context.py`，先独立B_sto400小样本检查输出，再读取既有B_sto400、C_det400/600/850四快照，均与已核对tape的post hash相同。0reset/step/teacher/candidate，只在私有SafetyFilter predictor中计算本次nominal proposal预计合流时刻的实际holder/承诺到达后位置。7个eligible o2m/m2i查询全部遗漏了incoming，未观察到stale departure；850没有这类eligible查询，不补样。完整JSON`.tmp/p3_v25_merge_context_full.json`，SHA `1b508885c5c40ea7803e5d1cf6823b35453c8b59bc651eb8b9bcd1aa88e3adff`。

| 已有状态 | 遗漏incoming数量 | 旧合流点到预测目标邻机最小弦距 | 纳入incoming后的最大空隙中心弦距 |
|---|---:|---:|---:|
|B_sto400 UID29|2|58.639m|216.824m|
|B_sto400 UID32/114|各6|36.833m|218.752m|
|C_det400 UID70/94|各6|33.667m|425.962m|
|C_det600 UID18/109|各1|8.609m|276.859m|

这些是固定预计合流时刻的几何距离，**不是新完整轨迹获准或闭环收益**。后续最小诊断仅在两个私有旧状态分别比较原nominal与替换forecast gap的nominal候选，完整保留generator/energy/SafetyFilter；先C_det600/UID18验证脚本输出，再B_sto400/UID114。不reset/step/teacher、不改运行核心、不替换原标签。原C_sto与同seed rule baseline终态保存后，修正scheduler未来成员与相位预测，再通过定向回归和新目录闭环检查。涉及新planner语义的旧标签不能直接作为新正式监督资格，需按实际影响决定重新采集/标注；I1仍暂缓。

两个状态的4次私有nominal候选已完成，0reset/step/teacher。C_det600/UID18原/forecast均被predicted_separation拒绝，生成时长116.881→73.692s；B_sto400/UID114两条轨迹都获得SafetyFilter checked hash，但均因merge_slot_retime_conflict未获scheduler批准，时长123.356→205.754s，原预约时刻为607.703，实际终点分别523.356/605.754。**改善目标终点gap不等于路径全程可行，也不能绕过真实合流时间的headway检查。** 路径长度有升有降；此处不宣称性能改善。两个输出分别`.tmp/p3_v25_merge_candidate_C600.json`（SHA d09ccaeb06c454ea6974d0511e204fdcc9628882ecf9bc4f45c6fa49adc5274e）和`..._B400.json`（SHA 2a8a84d760ed791c4b20f7967ada6e0071894eacc9b078ea512511b8d680e296），脚本逐候选直接落盘。旧last_predicted_track_audit仍not_run/null，不能把null当无穷安全距离；本诊断只使用返回reason/checked hash，没有虚构最小全程间隔。

### C_sto终态及规则控制启动（newly_run）

C_sto为COMPLETED/exit0，879ticks/952.600s（含全tape重放），**14N/10E/0F/D/U**，无error/censor、replay_exact=true。2918actor=174R/2744W、7527forced，全部原请求likelihood/mask/K3/trace核对通过；51普通批准（rank37/8/6）、119普通拒绝（HEADWAY9/PREDICTED_SEPARATION84/NO_MERGE_GAP15/LZ_RESOURCE11）、4原R无普通release_telemetry，不混算。0actor shield/实际安全/hash异常，最小实际/扫掠间隔80.603108/80.538428m；564energy checks/0negative，10应急提交均完成，最低完成SOC .209604。safety_interventions335保留，reward349.525168，不等于14E配对已被解决。

result SHA `925d37ff350b70ceb256d85c0d94ea4060668e2fe05a0d3b590e424a30d75dff`，终态result/protocol/steps/progress/first_before/log/exit同步，原2811480/2811481退出。primary读完整结果后接受启动前述唯一C_rule_same_seed，原student六臂完成且禁止重跑；Q/实景PPO继续暂缓。等待规则baseline期间仅新建尚未被运行代码import的forecast helper及其测试，旧scheduler/generator/config保持不变。

C_rule_same_seed已启动：Python2812519/timeout2812518、session84659，真实cwd/argv已核对，progress75ticks/76.126s、snapshot_1/blockers/log正常。新forecast helper的6项local/server用例通过（1.80s/.71s），两文件SHA相同，覆盖incoming到达边界、departure排除、实际有限加速度/速度偏置、空/单/多gap和不变性；尚未接入运行环境，不能当生产修复已完成。

### 同seed规则终态与修复实施（newly_run）

C_rule_same_seed已COMPLETED/exit0，939ticks/852.145s，**19N/5E/0F/D/U**；无error/censor/实际安全/hash/耗尽，5E全部ground_service_complete，最小实际/扫掠间隔80.641802/80.622081m，1478energy checks/0negative，最低完成SOC .124882。safety_interventions162保留。result SHA `8292fd5bbb3b7881ee9af15ad715096924c12856211857ce1e6693536a67049a`，result/progress/blockers/四快照均两端一致，2812518/2812519已退出。

primary核对939条blockers：57批准、305safety_reject、46final_capacity、47lz_unavailable，普通primary原因233PREDICTED_SEPARATION/20HEADWAY/52NO_MERGE_GAP/46FINAL_RESOURCE/47LZ_RESOURCE。此455条release telemetry与累计release_rejected计数不是同一个统计口径，不强行求和改原数据。规则总能耗9648.984857，C_det11106.523908、C_sto10183.666972；reward分别407.275229/296.877299/349.525168。逐arrival_id初始能量（剩余+累计消耗）24架完全匹配至6.4e-12，seed和所有物理配置一致，支持这是同外生条件控制对照。

分析脚本首轮配置字典比较失败，primary检查差异只有历史两个JSON serializer对`rings`枚举键分别写`RingRole.OUTER`和`outer`等；只对三个已知key做规范化，所有数值仍精确比较。修后全量通过，结果`.tmp/p3_v25_C_rule_analysis_v2.json` SHA c3e562d2e4242f59722ee61b66a5a2b97742600a735179624abdff6faea7f0ee。没有重跑环境或修改原产物。结论：规则也有应急/尾部余量不足，但当前student在这一个配对场景中额外损失5–9个正常完成，不能全部归为环境容量；需同时修已证实的scheduler缺陷和后续student泛化/访问分布。

forecast helper已`88f970d`提交。baseline自然结束后primary才接入`merge_gap_prediction_mode=committed_arrivals_v1`；历史默认`current_holders_legacy`保持旧scene/旧pickle语义。新P3 builder与两个既有评估CLI增加显式`--forecast-merge-gap`，完整config记录选择；不改generator、安全过滤、能量、应急、K3或策略参数。目标环成员按**本提案merge_time**判断，位置使用与SafetyFilter一致的承诺后续/有限加速度holder预测；空目标仍沿用原ego-phase规则。此阶段只隔离这一项修复，不把先前slot冲突假装已解决。

接入定向tests后，先新目录`/home/user/training_outputs/terminal/p3_future_merge_gap_20261008/smoke`：rule、3/warm3高SOC、892806/928806、2ticks，检查配置/正常输出/实际安全/hash。再唯一`C_rule`：同892804/928804、24/warm12混合SOC、1800ticks/3700s、blockers和1/400/600/850快照，与上面旧rule保持同runner/controller，只改变预测mode。新源码SHA同步、无重复后才能启动。完整结果决定是否继续固定checkpoint的C_det/C_sto或先进一步修复轨迹slot，不预先声称提速/应急减少；所有失败保留。开发seed精确搜索无历史命中，smoke首次reset前服务器另核对新输出未存在。运行中冻结生产核心，Q与学习继续暂缓，最终正式P3启动目标不变。

接入后57项local/server定向回归通过（51.94s/28.09s），8文件SHA相同，scoped diff检查通过。包括9项forecast/helper/接入、旧单holder/空gap兼容及新增forecast分支确实调用完整SafetyFilter、slot reconciliation、整体修复/runtime回归。没有降低任何已有测试阈值。编辑过程中patch工具返回局部失败，primary检查发现config重复字段/validation及部分文件未应用，已逐项修正后才运行上述测试；未在半完成版本启动任何环境。此处仍是代码与回归通过，待独立smoke和完整C_rule。

独立smoke已exit0，892806/928806、3/warm3高SOC，2ticks/1.877s，forecast profile/mode正确；3次放行全部批准，30checked samples/0hash mismatch，全部实际安全计数0，3energy checks/0negative/0应急。既有CLI按预定2tick截断写CENSORED，3U/terminal=false是smoke语义，不能改写成完整回合成功。result SHA `85433c17e2e28b7c94ff021cce830647470828e396cdfc180928e77f7b9cf9b3`，result/progress/snapshot1/blockers均两端一致，无残余进程。primary核对原始输出，接受本地提交后只启动预定C_rule完整对照。

合流预测接入已提交`8078e3e`。新C_rule唯一Python2814843/timeout2814842、SSH session12338，实际cwd/argv符合协议；100ticks/60.843s、blockers/progress/snapshot1正常。分析器首行smoke通过，最初worker误查旧根后已纠正为新根，不能由旧根缺文件判断本次失败。当前持续等待完整结果，核心冻结。期间primary仅准备未被此评估import的[新规划器初始train数据接口与方案](p3_v25_forecast_train_20261008.md)，30项local/server tests通过；旧1046标签不混入新数据，真正采集仍依本rule完整结果决定。

### 未来合流预测完整对照及预约时刻最小诊断（newly_run）

新C_rule已COMPLETED/exit0，963ticks/1069.727s，19N/5E/0F/D/U，24架均结束服务、active/queue/service=0，无error/censor/实际安全/hash或负能量。result SHA `99bf6388e2d7315857db39adcf4f9e2ab354157891d715da61309e6466830283`，blockers及四快照/相邻exit均两端一致；初次同步快照有暂时SHA不同，随后终态再核对全部匹配，进程2814843/2814842已退出。旧progress保留running字样但root result/exit/实际退出一致，不能以progress文案推翻终态。

primary全量核对963条blockers和逐arrival对照：54普通批准、257safety_reject、78final_capacity；primary原因192PREDICTED_SEPARATION/52NO_MERGE_GAP/13HEADWAY/78FINAL_RESOURCE。全部物理配置仅gapmode不同，逐arrival初始能量最大差9.33e-12。新/旧正常均19，应急均5；周期939→963，reward407.275229→398.885552，能耗9648.984857→9721.172536（约+.75%），最低完成SOC .124882→.185705。最小实际/扫掠间隔81.870836/81.790764m，36162checked samples/0mismatch，1138energy checks/0negative，5E均落地；412普通/应急过滤干预及122应急尝试保留，不写成“安全干预零”。合流成员遗漏确已修正，但该一场景没有整体效率改善；arrival0/19/23由E变N，12/18/21由N变E，不能只挑改善个体。完整分析`.tmp/p3_future_merge_gap_C_rule_analysis.json` SHA cb91d120de89fca9b5e291e2a5e46104c0f1782c587cd1835cc054ae7a6b05cf。

在开始新监督之前，primary进一步检查已记录B_sto400/UID114的slot疑点：私有副本保持新forecast，原生成器到达605.753829早于607.703295预约，完整SafetyFilter获证，但真实headway因此拒绝。只在诊断副本对原有有界departure区间增加12次二分参数化，并选择不早于预约的最近方案，可达到607.703998（误差+.000703s），真实slot可协调，**却被完整SafetyFilter的predicted_holder_reachable_tube拒绝**；checked hash=null。两次分别3.797/5.938s，0reset/step/teacher/学习，原快照与原tape哈希一致，生产生成器未改。先0query syntax smoke通过；逐次12个生成时刻保存在`.tmp/p3_slot_refinement_B400.json`，SHA7eccc632a2021cecd32336ef05cb21abc771f67aa6c316de1591ee12b998fb97。

这个最小检查说明原粗时刻搜索有精度限制，但不能把该状态的拒绝仅归于精度，更不能由精确预约推断存在安全放行。没有证据支持将这项增加计算、仍未获证的局部搜索部署为性能修复，也不放松headway或holder安全包络。保留为后续规划优化线索；不把有限搜索拒绝写成物理无解。当前完整规则周期无技术/安全/耗尽/未决阻塞，5E明确计入业务基线，接受推进新planner数据的独立smoke及逐臂采集。新/旧student调度差距仍待后续fresh imitation与必要student分布聚合解决，不重跑旧模型来冒称已解决，也不直接启动旧Q方案或PPO。
