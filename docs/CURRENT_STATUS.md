# 项目当前状态

## 2026-10-03 P3修复效果逐步验证（最新）

- 本轮A/B/C完整结束：新A两臂均12N，B两臂均24N，C规则19N5E、冻结source20N4E；共120请求111N9E、0F/D/U。所有臂自然终态、error/censor/实际安全违规/hash mismatch均无异常。C的5/4个应急提交全部完成，无耗尽；两个原C进程已退出，无本轮评估进程。详见 [完整验证记录与后续方案](evidence/v4_2/p3_direct_c2_repair_validation_20261003.md)。
- 已定位的原内环退出阻塞在已测开发场景中消失：原A为5N7F/4N8F。primary实现直接C²进近、途中平滑调速再接垂直下降，并修复近零时长切环段；不强制独立200→100m收缩或占用环上先减速。核心commit a7096fb，66项相关回归通过；机制证据见 [退出阻塞报告](evidence/v4_2/p3_inner_exit_blocker_findings_20261003.md)。
- 结论边界：A/B/C无参数更新，尚不能证明P3训练退化已解决。审查发现旧source loader曾创建空Adam，90D推理容器自身无优化器；本轮已改为纯模型加载，纠正此前“全过程无优化器”的表述，不将空对象创建归因为退化。V3/90D环境开发闭环通过，冻结source仍NOT_QUALIFIED，PPO_ALLOWED=false。C仍有4/5次应急兜底，不称低SOC全正常或业务最优。
- 用户授权的3个source snapshot_400预检已完成：每态1次固定请求、两副本各1step，总3个逻辑tick/6次step；59 tests通过，A/B/C均exit0、迁移输出差0、重放相同、权重/checkpoint不变、安全/hash无异常。[详细协议及结果](evidence/v4_2/p3_direct_c2_contract_preflight_20261003.md)。30行中28 forced WAIT、2 actor WAIT、0 RELEASE：已观察契约通过，但真实RELEASE条件概率/K3执行分支覆盖未建立，不能称完整接口资格通过；无参数更新，P3训练退化仍未验证解决。
- primary已亲自核查旧mask原因，未发现误屏蔽。用户授权的[真实RELEASE/K3事件预检](evidence/v4_2/p3_release_event_contract_20261003.md)已完成：18 tests通过，独立2tick smoke通过；固定12/warm6高SOC、seed891901/891902的run在第1主tick自然覆盖o2m/m2i/i2f三接口首次请求/批准，4 RELEASE+2 actor WAIT、3个rank1批准、1个HEADWAY拒绝后canonical WAIT。1主step+1副本step，完整重放相同、实际安全/hash无异常、权重未更新。被拒RELEASE的原动作及likelihood保留，没有被执行WAIT替换；成功选中rank2/3未出现，不声称覆盖。
- 当前已测环境闭环与source请求接口未见新阻塞，不再重复扩大环境预检。primary检查发现历史R4 collector硬编码88D/旧配置，已另写[V3/90D expert pilot](evidence/v4_2/p3_v3_i0_pilot_20261003.md)：复用state-only teacher，在私有副本用当前direct_c2生成标签，原始行、forced-WAIT mask、请求/执行拒绝全部保留。9项tests通过；独立smoke892101的2tick/6行通过90D、标签、checked hash、重放和实际安全检查。代码/预注册commit13a65b7，已准许按既定上限运行两回合pilot892201/892202；future train/audit/Q仅保留互斥seed范围，未采集。fresh actor资格前不恢复PPO，本轮无训练或optimizer更新，不能宣称退化已解决。无full training、held-out、硬件或checkpoint覆盖。
- V3专家pilot已按wall_limit截断：892201为298ticks/2757行/3625.746s，N/E/F/D/U=3/0/0/0/9，error=null、安全/hash无异常，status=CENSORED、根INCOMPLETE/exit1。原Python2469251及wrapper已退出，892202未启动，禁止重启补PASS。primary逐行核对全部90D/mask/索引/请求/state hash通过；2643 forced WAIT，真正actor行114（26 RELEASE/88 WAIT），专家RELEASE22批准/4拒绝。9U不是能量失败或不可达证据。
- [采集成本诊断及下一步方案](evidence/v4_2/p3_v3_i0_pilot_20261003.md)已完成：原tape150步逐步重放相同，同tick150/UID10一次25候选查询带profiler耗时84.8986s，其中轨迹生成77.8834s。完全相同多项式bounds复用试验虽69tests及重放/全部25候选/label等价检查通过，query仍82.0386s，不足以解决瓶颈；primary已撤回cache，核心SHA恢复832a88f，保留全部试验输出与新诊断脚本。最终保留代码相关12tests通过（31.85s），所有本轮采集/诊断进程均退出。
- 下一步为有界的“完整rule回合→事先固定pre-state稀疏离线专家标注”开发collector，最多24次query/600候选，不按标签或成功结果选样，不把未标注或forced WAIT行塞入actor监督；完整业务终态与标注预算分别判断。新seed须先查重预注册；旧88D BC loader不能直接接90D/mask，fresh student尚未训练或qualified。future train/audit/Q与所有学习均未启动，PPO_ALLOWED=false；本轮结果/必要诊断已收束，heartbeat `p3-v3-expert-pilot`将在提交后删除。
- 用户2026-10-04继续授权后，primary已实现[稀疏专家开发pilot](evidence/v4_2/p3_v3_sparse_pilot_20261004.md)，新seed892401 smoke、892402 high、892403 mixed已完成元数据查重并预注册；旧expert heartbeat已删除。17 tests通过39.23s；独立smoke为2ticks/6raw行/1label，90D/mask/null、原请求、checked hash及完整重放通过，实际安全/hash无异常。primary准许下一步仅启动high，完整结果无阻塞再mixed。75tick窗口最多一条、全回合最多24次query；600是RELEASE意图数，final多LZ计划调用另行记录。环境与离线标注解耦、原始未标注行保留null，行为请求/teacher标签/执行严格分开。长任务按用户偏好60分钟heartbeat；无训练，PPO_ALLOWED=false。
- 稀疏collector commit411c401的high已PILOT_OK/exit0自然结束：605ticks、12N/0E/F/D/U、error/censor/实际安全/hash无异常，原PID2479355及child已退出。环境193.9813s，6/6离线标注70.0351s；primary逐行核对3204raw/605steps/6labels与固定窗口选择通过，64 actor/3140 forced WAIT，标注2 RELEASE proposal12/4 WAIT，102意图/177多LZ plan调用。末SOC最低.645182，无应急/耗尽。该开发采集流程可继续既定mixed892403；仍不等于I0或student资格。heartbeat `p3-v3-sparse-pilot`保持60分钟正常静默，无训练/PPO。详情见稀疏pilot报告；无remote未推送，其他dirty保留。
- mixed892403已PILOT_OK/exit0自然结束，572ticks、12N/0E/F/D/U，error/censor/实际安全/hash无异常；环境124.6635s、7/7标注42.1199s，primary逐行核对3383raw/572steps/7labels及选样通过。末SOC最低.320021；8次应急尝试/16次LZ计划安全拒绝、0应急提交，最终均正常完成。真实Python2483247/wrapper及所有child均退出，禁止重启两臂。
- [稀疏pilot两臂结论](evidence/v4_2/p3_v3_sparse_pilot_20261004.md)：共24N、0E/F/D/U，13/13标注完成，接受开发采集流程验证。6587raw只有121 actor/13有效label，标签7 RELEASE/6 WAIT，proposal12/20/13为5/1/1，不能宣称I0充分或训练退化解决。mixed存在teacher动作不在rule K3中的案例，原teacher/behavior/execution均分开保留，不能误报为专家标签被执行拒绝。下一项仅为独立90D loader与mask/partition/空标签定向检查（0reset/0optimizer），再在fresh分区首次reset前冻结更充分的正式I0选样及预算；不直接把75tick一条的密度升级成正式训练协议。所有reserved train/audit/Q仍未消费，无新训练/PPO。核心不变，17tests及smoke证据保持；本轮仅文档检查/提交，之后删除已完成的heartbeat，无remote未推送。

- 用户2026-10-04授权下一步后，primary已完成[独立V3/90D数据接入](evidence/v4_2/p3_v3_loader_validation_20261004.md)，旧88D入口和collector/物理安全核心保持原实现。新loader按decision_id关联OK标签，gate仅取有效标签且actor_mask=true，proposal仅取teacher RELEASE；null/超时/错误不变成WAIT，behavior/actual不改teacher target，训练接口拒绝development/audit/Q。33项数据专用tests通过（0.22s），无环境reset。两臂真实文件只读CLI均exit0，重得3204/3383raw、64/57actor、6/7标签、2/5proposal监督计数，与原分析相同；真实开发分区训练入口均拒绝。0reset/0teacher query/0optimizer/0checkpoint写入，所有reserved train/audit/Q未消费。数据接口检查完成，下一步冻结正式I0更充分的固定选样、场景与query预算，然后运行其首个三场景cycle；不把接口通过当I0/student资格或训练退化解决。没有活动任务/heartbeat，PPO_ALLOWED=false，无remote未推送。

- 用户继续授权后，primary已冻结[V3 fresh I0完整决策采集协议](evidence/v4_2/p3_v3_i0_protocol_20261004.md)：所有actor pre-state按tick/数值UID顺序离线标注，最多256 query/6400 RELEASE意图，完整周期1800ticks/3600s与标注7200s分别封顶；溢出/失败/截断保留且阻止后续臂，不选标签或补成功seed。此前121 actor只留13标签的数据过薄问题由取消第二次稀疏采样处理，forced仍排除。新runner、旧sparse回归及loader共47tests通过29.65s；独立smoke892501为2ticks/6raw/3actor/3OK标签，primary核对90D/全部UID/label join/checked hash/重放通过，无实际安全/hash异常。协议和代码提交后先启动train01（893001，12/warm6高SOC），其完整无阻塞后依次train02/03（893002/3，24/warm12高/混合SOC）；首cycle不运行audit/Q或学习。I0_QUALIFIED=false/PPO_ALLOWED=false，长任务按用户偏好60分钟静默heartbeat。此条记录时正式seed仍未reset，无remote不推送。

- I0代码/冻结协议已提交`2931e4c`；首正式臂train01已启动，Python2521016/wrapper2521015，输出`/home/user/training_outputs/terminal/p3_v3_i0_20261004/train_01`。primary读回环境阶段COMPLETED：893001、504ticks/145.7804s、12N/0E/F/D/U、无技术/censor/实际安全/hash异常，3184raw中59actor全部选中、无overflow；根status仍为ANNOTATING，59个独立专家查询尚待完整结果，不称COLLECTION_OK。heartbeat `p3-v3-i0-first-cycle`已确认ACTIVE/每60分钟/正常静默；禁止重复启动train01或smoke。完整无阻塞后primary继续预注册train02→03；首cycle以外train/audit/Q不自动启动，任何学习仍禁止。893001已消费，其余正式seed未reset。代码冻结、docs本地提交，其他dirty保留，无remote不推送。

- I0 train01已COLLECTION_OK/exit0自然结束，Python2521016/wrapper及child退出。primary亲自核对3184raw/504steps/59selected-label与快照、全部90D/索引/mask/行为/原teacher/执行、coverage和重放通过：12N/0E/F/D/U，无error/censor/安全/hash/能量异常，59/59有效标签=37 RELEASE+22 WAIT，8类proposal、25次retry结束后标注与8次同UID同接口teacher WAIT→RELEASE。环境145.7804s、标注468.6766s，合计约10.24min。5次teacher RELEASE/实际WAIT均为rule K3外的timing意图，未提交给执行层，不改标、不误称专家动作被拒绝。primary接受单臂完整采集，继续预注册train02（893002，24/warm12高SOC）；保持60分钟heartbeat、核心冻结、无学习/I0资格/PPO结论。详见[I0记录](evidence/v4_2/p3_v3_i0_protocol_20261004.md)，无remote不推送。

- I0 train02启动记录：Python2524958/wrapper2524957，`.../p3_v3_i0_20261004/train_02`，893002/24请求/warm12高SOC；首次57s确认真实argv及12raw/1step。启动前冻结源码SHA同步且无重复输出/进程，按60分钟heartbeat检查。完整终态及当前判断见下一条，不再视为运行中。

- **当前终态：I0 train02为INCOMPLETE/exit1，首cycle因查询数超限停止。** 环境943ticks/1553.7767s自然完成23N/1E/0F/D/U，无error/censor/实际安全/hash/耗尽，原父/child均退出；772actor超过256上限，overflow516，annotation按协议NOT_RUN/0labels。primary逐行核对10769raw/943steps/256selected、177快照SHA、90D/mask/null/请求执行分离通过。435个资源拒绝pre-state均已有3架final；最早此类可信快照的一次独立诊断得到原teacher WAIT、25个final_capacity拒绝、0plan调用/0.05498s，保持diagnostic NEVER_TRAIN。前256条位于tick443及以前，遗漏全部108个低SOC actor，不能作为完整I0。具体是采集数量预算不足，未发现应削弱容量、安全或actor mask的依据。[完整结果、必要诊断及有界后续方案](evidence/v4_2/p3_v3_i0_protocol_20261004.md)：另冻新协议后建议2048query但保留7200s标注墙钟，先一臂fresh 24请求；本轮未实施、未补标/复跑或启动train03。train01仍59个有效标签，其余train/audit/Q未消费；I0_QUALIFIED=false/PPO_ALLOWED=false，无学习或P3退化解决结论。heartbeat `p3-v3-i0-first-cycle`已删除，无遗留任务；冻结源码未改，47tests/smoke为已有通过证据，本轮全量产物断言和单点query通过。仅本地文档提交，保留其他dirty，无remote不推送。

- 用户要求“解决第二臂问题，然后继续”后，primary实现独立`--protocol-version v2`与[新查询预算协议](evidence/v4_2/p3_v3_i0_v2_budget_20261004.md)：2048 query/51200 RELEASE意图，保留1800ticks/3600s环境与7200s标注上限；原v1/256默认、已失败train02、mask/安全/物理/teacher均不改。fresh smoke892601、dense_high893101及条件后续dense_mixed893102已在首次reset前冻结，原v1其余seed/audit/Q不挪用。51tests通过36.76s；独立smoke已SMOKE_OK/exit0，2ticks/6raw/3actor/3有效RELEASE标签，primary亲自核对真实V2 cap、90D/快照/请求执行/label join和完整重放通过，实际安全/hash无异常。代码与协议提交后准许只启动dense_high，完整无阻塞后primary继续唯一mixed；暂无学习/I0/student/PPO资格结论，长任务保持60分钟静默监控，其他dirty保留，无remote不推送。

- **V2 dense_high_launch2启动记录（现已自然完成，终态见下条）**：Python2534121/wrapper2534120。 核心/协议commit`6863caa`，51tests及smoke通过。首次dense_high启动因worker漏cd在module导入前exit1，没有reset；原失败保留，primary准备显式cwd/源码SHA/无重复检查的启动器，解决默认沙箱无法读取SSH配置后在新目录启动同一未消费seed893101。真实argv/cwd已核对，24/warm12高SOC、2048query/51200意图/1800ticks/3600s/7200s，primary确认首tick重放一致，初读75ticks/1035raw/13selected仅运行前缀。heartbeat `p3-v3-i0-v2-budget`已确认ACTIVE/每60分钟/正常静默；禁止重跑旧臂或改运行中核心。完整high无阻塞且全部actor标注完成后primary继续唯一预登记mixed893102，出现阻塞则先完成最小必要诊断。当前无学习/I0/student资格或P3解决结论，其余train/audit/Q不自动执行。详见[V2执行记录](evidence/v4_2/p3_v3_i0_v2_budget_20261004.md)，无remote未推送、其他dirty保留。

- **V2 dense_high已COLLECTION_OK/exit0，查询预算修复在本密集回合有效。** 893101、863ticks、24N/0E/F/D/U，无技术/censor/实际安全/hash/能量异常；10040raw=613actor+9427forced，613/613原teacher标签全OK、overflow0。primary逐行核对90D/mask/null/快照/集合与loader通过，local/server六终态文件SHA一致；环境1000.3780s、标注5616.8450s，未延预算。标签90 RELEASE/523 WAIT、16类proposal、9个低SOC、223个retry恢复、37次同机同接口WAIT→RELEASE；相关状态不是独立场景，rule占用与高WAIT比例不等于student资格。原父child均退出，high禁止重跑。
- primary完成3个差异状态的原tape精确重放（0reset/0teacher query/3step，全部exit0、post-state hash/trace/telemetry相同）：tick211/UID9在服务释放LZ3后实际可放行；tick515/UID42的单机可行LZ0先被UID13占用；tick610/UID91放行前UID69/42已由holder变为已提交final。由此记录单机pre-state标签与服务推进/逐机批准上下文的区别，不能把WAIT当联合物理无解或用实际动作改标。没有新的环境或采集完整性阻塞；primary接受high并继续唯一预登记mixed893102，任何学习前仍须处理该监督语义边界。详见[V2终态与最小诊断](evidence/v4_2/p3_v3_i0_v2_budget_20261004.md)。核心未改、51tests/smoke为已有证据；I0_QUALIFIED=false/PPO_ALLOWED=false，无其他train/audit/Q、BC/PPO或P3解决结论。

- **V2 dense_mixed启动记录（现已终态，见下条）**：893102，Python2543375/wrapper2543374。 显式cwd/Syncthing五源码SHA/无重复预检后启动，真实argv及/proc cwd已确认；24/warm12、SOC.55–.90、2048query/51200意图、1800ticks/3600s环境/7200s标注。primary已核对初始产物和首tick完整重放；100ticks/1418raw/19selected为运行前缀，不能当终态。既有heartbeat `p3-v3-i0-v2-budget`已更新为mixed、保持每60分钟正常静默。完整后总结两臂成本/覆盖/监督边界；失败则保留并做最小必要诊断，不自动扩展train/audit或学习。禁止重启smoke/high/原v1，无remote不推送、其他dirty保留。

- **当前结论：V2预算修复部分有效，完整fresh I0仍未通过；本轮无运行进程。** mixed893102已INCOMPLETE/exit1，环境892ticks/1458.1460s自然完成21N/3E/0F/D/U，技术/censor/实际安全/hash/耗尽均无异常；482actor全部selected、overflow0，但标注7200.0166s触顶，300OK+1TIMEOUT+181NOT_ATTEMPTED。primary全量核对9932raw/892steps/482selected/335快照及300标签通过，local/server七终态文件SHA一致；未标注后缀缺139个低SOC、101高速inner、115个retry恢复状态，禁止拿300条时间前缀冒充完整I0、恢复标注或复跑。
- [V2终态与成本诊断](evidence/v4_2/p3_v3_i0_v2_budget_20261004.md)：已完成query总6937.7360s，o2m/m2i的156个WAIT穷举25候选占约93.80%。primary机械选择最早耗时>=30s的已完成slot19/tick100/UID13，先slot0 smoke后一个私有副本profile（合计2query、0reset/0step）；原完整audit/label/输入均不变。profile中切环轨迹生成57.8502/63.8815s约90.56%，450次出发时刻build、520次parameterize、1175次prime_quintic_bounds；主要成本在轨迹数值计算，不是提高query数量就能解决。cProfile增加耗时，不能把64s对原33.5s当成退化。worker预建目录及外层CR退出错误均另行留证，实际slow结果PROFILE_OK/.exit0，未重复query或改原失败。
- 两臂环境共48请求=45N/3E、0F/D/U，无原inner耗尽复现；1095actor中913有效标签（148R/765W）但mixed缺182条，不能称完整两臂训练集。已有直接C²环境改善证据继续成立，尚无fresh student-only或训练退化解决证据，I0_QUALIFIED=false/PPO_ALLOWED=false。下一步建议只优化批量多项式bounds中点积/标量裁剪去重的精确计算开销，保留所有候选、搜索顺序、容差和安全判据；先4个既有固定状态原/新实现最多8query及4step等价/成本验证，再决定新正式采集协议。此方案尚未执行，不自动扩展train/audit或学习。所有本轮collector/profile已退出，heartbeat `p3-v3-i0-v2-budget`已删除。仅文档本地提交，保留其他dirty，无remote不推送。

- **2026-10-05当前阻塞与数值优化候选：** 用户继续授权后，primary准备了[多项式bounds优化及验证记录](evidence/v4_2/p3_numeric_bounds_optimization_20261005.md)和[待应用补丁](evidence/v4_2/p3_numeric_bounds_candidate_20261005.patch)。优化仅减少数组装箱、标量clip、通用polymul/pad和重复逐段求导开销；候选/搜索顺序、容差、安全/容量/mask不变。正确加载候选的64项回归全部通过（7.76s）；512合成段×4轮的bounds全部精确相同，纯bounds中位速度约1.83倍，不能外推完整query。首个加载器混用新旧模块导致7失败的日志和原版7项对照均保留，修正的是测试加载器，未掩盖失败。
- 服务器三次SSH均在建连时关闭（exit1，连pwd未执行）；不能确认当前服务器进程/同步。本地原版固定mixed slot0只执行1query/1RELEASE意图/1plan、0reset/0step，即因轨迹证书hash不同于服务器原产物而失败；动作及完整audit等其余字段一致，自己的checked hash与trajectory hash相同。候选query、其余状态、4step及所有正式采集/训练未执行，失败目录保留。该差异发生在候选之前，不能归因优化，也不能在缺少证据时唯一归因某个平台因素。生产polynomial源码仍为832a88f原版；候选603a299保存为未应用补丁。连接恢复后须在原服务器先验证原版/候选smoke，再复用其结果完成既定4状态/8query/4step等价和同机耗时验证，历史本地失败1query另计。通过前不启用优化或重启I0；I0_QUALIFIED=false/PPO_ALLOWED=false，P3训练退化未解决。本轮未启动后台任务/heartbeat，无remote不推送，其他dirty保留。

- **2026-10-06连接复查：** SSH仍不可用。普通连接及一次verbose诊断确认TCP22连接成功，但SSH标识交换阶段收到`kex_exchange_identification: Connection closed by remote host`（verbose进程exit255），远程命令未执行。不能据此唯一归因sshd/网络，也未取得实际进程或同步SHA。详见[数值优化记录的连接复查](evidence/v4_2/p3_numeric_bounds_optimization_20261005.md)。本次没有应用候选、启动查询/采集/训练或新增性能结果；保留原服务器验证计划，待SSH恢复后继续。只更新文档，无新heartbeat，无remote不推送。

- **2026-10-07服务器恢复，数值优化已应用：** 原服务器固定mixed0/19/299与high177的8query/4step全部通过，0reset，136个RELEASE意图/plan；完整audit、选中证书、原tape及安全一致，4个step输出原/候选完全相同。四状态query合计146.4656→100.8492s，减少31.14%（约1.45倍）；这是固定开发状态收益，不能外推完整I0成本。此前本地不同证书hash失败保留且另计1query，原服务器slot0已复现原证书。确认诊断退出后primary本地应用冻结603a299候选，Syncthing后实际生产路径90tests通过29.85s，无遗留本轮进程。详见[原服务器验证及结果](evidence/v4_2/p3_numeric_bounds_server_validation_20261007.md)。下一步另冻一个fresh密集mixed采集，维持原2048query/7200s；不恢复旧失败、不启动学习。I0_QUALIFIED=false/PPO_ALLOWED=false，P3训练退化未验证解决，其他dirty保留，无remote不推送。

- **2026-10-07 V3 fresh mixed采集准备完成：** 数值核心已提交`0804d77`，原服务器固定状态与90项生产回归通过。primary另冻[单臂V3协议](evidence/v4_2/p3_v3_i0_v3_numeric_20261007.md)：fresh 893201、24/warm12、SOC.55–.90，仍2048query/51200意图、1800ticks/3600s环境和7200s标注，原v1/v2及失败产物不改。新入口/停止条件/90D loader共54tests通过27.95s；独立892701 smoke已SMOKE_OK/exit0，2ticks/6raw/3actor/3OK标签。primary亲自核对全部90D/mask/null、原请求、快照、标签join、teacher证书和首步完整重放通过，实际安全/hash无异常；预设2tick截断不称完整业务通过。smoke父子均退出，禁止重跑；提交后只启动893201一次。预计环境15–30min、标注最多2h，按60分钟heartbeat监控。没有学习、student或I0资格结论，PPO_ALLOWED=false，未消费的其他train/audit/Q保持原样；保留其他dirty，无remote不推送。

- **2026-10-07当前唯一运行：V3 dense_mixed893201。** 入口/冻结协议commit`1663600`后，经Syncthing五源码SHA和无重复检查启动Python2759659/wrapper2759657，输出`/home/user/training_outputs/terminal/p3_v3_i0_v3_20261007/dense_mixed`；真实cwd/argv已核对。primary核对protocol及首tick12个90D行与tape/state hash通过，完整first_live/first_replay相同；初始progress1tick/12rows/12selected，rows随后120行，均为运行前缀。新heartbeat `p3-v3-soc`已ACTIVE/每60分钟，正常静默，终态须本人全量核对成本/完整标签/业务安全与覆盖。仅运行此fresh混合SOC臂，原smoke/旧臂禁止重启，核心冻结，不延预算补标；I0_QUALIFIED=false/PPO_ALLOWED=false，无自动学习或额外train/audit/Q。详见[V3协议及启动记录](evidence/v4_2/p3_v3_i0_v3_numeric_20261007.md)，其他dirty保留，无remote未推送。

- **2026-10-07 V3 mixed893201已COLLECTION_OK/exit0，无本轮运行进程。** 921ticks/1157.1544s，16N/8E/0F/D/U，error/censor/实际安全/hash/能量耗尽均无异常；374actor全部selected并取得374OK标签，overflow/缺失0，标注6248.8170s（104.15min）未超7200s。primary逐行核对11136raw/921steps/268快照/90D/mask/null/原teacher/loader集合通过，local/server七终态文件SHA一致。标签102R/272W、19类proposal；179低SOC、150高速inner、265 retry恢复及42次同接口WAIT→RELEASE均完整标注。本臂解除采集成本阻塞；不同seed不能当完整配对加速实验，8/24应急说明仍非全正常业务，P3训练退化未验证解决。
- primary又完成唯一teacher WAIT/trace RELEASE状态slot287的一次原tape精确重放（0reset/0query/1step、EXACT_REPLAY_OK/exit0）：UID21已先被应急层接管，正常release返回plan_active；trace默认继承请求，不是策略批准。54个真实K3批准与4个无rank RELEASE trace必须分开，原teacher/raw均不改。完整结论与[有界后续方案](evidence/v4_2/p3_v3_i0_v3_numeric_20261007.md)已记录：三个完整train回合共1046标签（229R/817W，仍各保留原协议身份），下一项建议0reset/0optimizer准备90D BC清单和入口、审查I0支持范围，再另冻最多3初始化seed×1000更新的监督可学习性小试；本轮未执行或准入学习，不继续铺开采集pilot。I0_QUALIFIED=false/PPO_ALLOWED=false，audit/Q untouched。进程已退出，监控`p3-v3-soc`将在结果提交后删除；其他dirty保留，无remote不推送。

## 2026-10-03 P3 统一实现修复（最新）

- 用户授权 primary 亲自完成重要实现，再逐步验证效果；详细问题、证据、修改和限制见 [统一修复记录](evidence/v4_2/p3_integrated_repair_20261003.md)。
- 旧 V3 R2 在完成2/8臂后主动停止：legacy88=6N/0E/2F/0D/16U并异常；repaired88自然结束=2N/1E/20F/1D/0U。repaired90为未完成分支。PID2323277于UTC2026-10-02 17:45:57确认退出；旧产物不改写。旧heartbeat确认已不存在。矩阵对照不完整，不能报全矩阵PASS。
- 已实现边界可恢复的V3速度过渡、nominal holder执行/预测约定、实际应急路线资源优先、路线能量期限、基于V3路线的入场估计、同tick批量容量和入场/队列互斥计时修复；统一90D/pre-action forced-WAIT、队列team reward与gate-only entropy选项。旧V2动力学和历史profile保留；新组合使用 `terminal_marl/configs/p3_integrated_repair.py`。
- 当前状态：22文件最终相关回归196 passed；check-only和规则/冻结source两步smoke通过；3请求高SOC规则完整周期475 ticks、3正常完成、无碰撞/硬间隔违规/hash mismatch。相同场景冻结source540 ticks、3正常完成、同样无安全/hash违规；两次验证监控均已删除。尚未验证dense/低SOC和训练效果。没有新训练、held-out、checkpoint写入或默认promotion，`PPO_ALLOWED=false`。

## 2026-10-02 V3 修复审核状态（最新）

- 用户已同意本轮重大修复；opt-in `executable_trajectory_v3` 已实现并限定为 static geometry，V2 默认路径与 50 条历史 hash 回放保留。
- 四 fixture corrected V3 diagnostic 为 `KINEMATIC_CONSISTENCY_PASS`、`finding_count=0`、`technical_error_count=0`；同一 quintic 曲线、全曲线 bounds、live initial/ring-handoff guard 与既有 Safety/runtime budgets 已保留。
- 当前 core implementation commit 为 `f1dfb1034d6517fb1b5197035dc92e3eabe482ba`（24 files）；remote-v 为空，未 push。Safety/dynamics regression 为 `123 passed / 1 skipped`，`89.26 s`。
- V3 check-only preflight 已通过，source/load 为 V2、runtime 为 V3，使用 frozen source-to-runtime transfer；unified retry 为 `298 passed / 1 skipped`，focused diagnostic `6 passed`。
- Corrected V3 smoke 已接受：三 source 各 `2 ticks`、`SMOKE_OK`、`censored=true`，validity valid、technical errors 为空，safety none、uncertified/invariant 为 `0`，energy coverage valid，V2/V2/V3 bindings 与 fixed source 一致，repaired88/90 actions、metrics、hashes、outcomes、request identity 相等。
- 此前 compact/strengthened smoke 失败 provenance 保留；counter mapping 已修正为 runtime `joint_safety_fail_closed`，缺失字段保持 `None/reject`，不弱化安全规则。
- Corrected evaluator tests `16 passed`；evaluator/docs commit 为 `9b0d13cdd85a7c59170b68a25f8e12a1dfbad22e`，index empty，remote-v 为空，未 push。
- Batch bounds helper targeted qualification 已完成：server NumPy `2.5.1`、`256` segments 为 scalar `0.0893419 s` / batch `0.0448305 s`（约 `1.9929×`），`max bounds diff=0`；local NumPy `2.3.5`、`64` segments 亦为 `max diff=0`，focused `19 passed`、caller integration `55 passed`。
- True batch smoke 已接受：三臂各 `2 ticks`、`SMOKE_OK`、V2/V2/V3 source/load/runtime bindings、fixed source、frozen flags 全 false；除 UUID plan id 外 requested/accepted/actual metadata、hashes、safety、validity、wait/elapsed/energy 非计时字段与 corrected smoke 相等。
- Final `38`-file unified regression 已完成：`319 passed / 1 skipped`、`99.25 s`、exit `0`；四个既有 before/after JSON compare 的 field/PVA-time/segment/bounds diff count 均为 `0`，max bounds diff `0`，validation errors 为空。
- Registered 8-arm zero-training matrix comparison 已结束为 `COMPLETED_WITH_ISSUES`：artifact SHA=`d7cee7a34092cf885aa213961fa86ceba68d5dfb3b125607164d4e126183381b`，`executed=8/8`、`stopped_early=false`、wall=`1542.282646 s`；UTC `12:44:37` 时 PID=`2303790` absent，OS exit 未观测。execution/performance commit=`e5aa7af315ecefb786464740122d08d8a6816c90`（7 files，remote-v 为空，未 push）。
- 八臂均 `status=error`、`stop_reason=exception`、`resolved=0`；ledger-only validity=`valid`、errors 为空，但 arm error 不构成 qualification。C1/C2 的 ring-hold 与 continuous-landing 错误明细见结果文档；hard/invariant/uncertified 均为 `0`、event lists 为空，zero-initialized partial tape/metrics/hash comparison 不能作为 performance qualification。
- heartbeat `p3-v3` 已由 primary 在旧 matrix 终态删除；旧失败 artifacts 与 source binding 保留。primary 已亲自修复三个 core 文件：V3 altitude-bounded interior slope、与实际执行相同的 partial ring-wait projection，以及明确不可行 fallback 的保守返回；安全、高度、SOC/energy、reward 和 authority 门槛均未调整，V2 默认路径保留。code/tests commit=`5bd9b62e218d71640deb8bd73a9c691af454052b`（6 files），remote 未配置，未 push。
- 本轮 targeted 首次 `56 passed / 1 failed`，新增 cases 全通过；唯一旧 warm-start fixture 把可行 SOC `.35` 错当不足。冻结旧 helper/当前 helper 都为 required SOC=`.2060807011022114`；primary 将拒绝 fixture 改 `.15` 后 `57 passed in 3.38 s`、exit `0`。既有四条 V3 轨迹除 UUID/source hashes 外全部 fields、p/v/a/time、bounds 和 trajectory hash 完全相同。
- 两个原异常 prefix replay 为 `PREFIX_SMOKE_OK`、exit `0`（C1 legacy `14 ticks`、rule-repaired `10 ticks`），error=null、technical errors=[]、safety none、uncertified/invariant=0；均 censored，不能替代完整业务闭环。明确 `33`-file regression 为 `280 passed / 3 failed / 1 skipped`；三个旧 emergency-energy fixture 的修复前后执行结果相同，primary 修正前提后 focused 3-file retry 为 `27 passed`、exit `0`。三处失败已闭合，不声称重新运行完整集合；旧 `38`-file 命令清单未保存，故不声称重复了原清单。
- 相同八臂协议的 R2 已 exact-one launch：UTC `2026-10-02T13:28:22.462494+00:00`、PID=`2323277`、execution commit=`592d9758469130062540793c699964c198249eca`；独立 output root=`/home/user/training_outputs/terminal/p3_foundation_repair_v3_r2_20261002`。原 checkpoint、seeds、arms、24 requests/1800 ticks 和 hard-safety 停止规则保持固定，15/15 local/server source SHA 一致。初始真实 tick=`1`、simulation=`1 s`、active/requests=`12/12`、safety none，进程/argv/cwd 已核对；最新服务器快照（UTC `13:33:48`）为 PID alive/Rsl、CPU=`106%`，C1 legacy88=`100 ticks / 100 s`、`0/8` arms complete、requests/active/queue/service=`18/16/1/1`、safety none，尚无完整终态。
- heartbeat `p3-v3-r2` 已创建为 ACTIVE、每 `10 min`，正常进度静默、完成/新异常/需要用户决定时通知，终态删除。运行中保持 sources 不变；held-out、training、optimizer step、checkpoint write、default promotion 均未执行，`PPO_ALLOWED=false`。
- 用户要求等待期间先修复 R2 新异常：C1 legacy88 在 `1017 ticks` 后报 `ring_hold_tangential_budget_unavailable`（partial normal/failed/unresolved=`6/2/16`，hard safety 为 0）。primary 已准备一行异常分类修复及 8 个真实调用测试，隔离候选 `8 passed`、扩大精确候选回归 `50 passed in 3.23 s`；15 个 R2 source hash 前后相同，production source 保持 frozen，待 R2 终态后由 primary 应用，完整业务结论仍待 comparison。详见 [R2 ring-budget 待应用修复](evidence/v4_2/trajectory_v3_r2_ring_budget_repair_20261002.md)。当前 heartbeat 实际配置为每 `20 min`。
- P3 尚未解决或准入，dynamic geometry 与真实 hardware adapter 仍未 qualification。详见 [V3 修复结果](evidence/v4_2/trajectory_v3_repair_results_20261002.md)、[运动学修复审核方案](evidence/v4_2/trajectory_kinematic_consistency_repair_review_20261002.md) 与 [基础修复协议](evidence/v4_2/p3_foundation_repair_protocol_20261002.md)。

## 2026-10-02 已批准首阶段修复与新增运动学阻塞（独立块）

已批准 opt-in final proposal 执行与显式 88→90D 推理 initializer；core commit `aff605e4f74a560853ef111d6f8c7c4900d12c1c`、runner/protocol commit `848b2d8570f8ae9b680b98273a45e4e34fd93bec`。`116` 项 core、`10` 项 runner/protocol 与 `4` 项运动学诊断测试通过；legacy 50 条历史 hash、duration、departure 保持相同。

真实 preflight 为 `CHECK_ONLY_OK`，三臂各 2 tick smoke 为 `SMOKE_OK`；repaired88/90 requested action tape 最大 diff `0`，metrics/hash 一致。新运行的四 fixture 旧 `validate_kinematics=[]`，但 actual-position-derived 水平加速度为 `22.3043–28.0235 > 1.5`、垂直加速度为 `5.8505–6.5178 > 1.0`，compact CLI 为 `KINEMATIC_CONSISTENCY_FAIL`，`technical_errors=0`。

完整八臂 `NOT_RUN`。新曲线/动力学方案待本次用户审核；`PPO_ALLOWED=false`，新 mode 未 default promotion，未执行 training、held-out、optimizer step 或 checkpoint 写入。边界事实：当前没有真实 Crazyflie/Crazyswarm2 hardware adapter；这不作为 bug 结论。详见 [运动学修复审核方案](evidence/v4_2/trajectory_kinematic_consistency_repair_review_20261002.md)、[P3 基础复核报告](evidence/v4_2/p3_foundation_review_20261002.md) 与 [基础修复协议](evidence/v4_2/p3_foundation_repair_protocol_20261002.md)。

## 2026-10-02 基础实现复核更新（独立块）

本块只补充本轮基础实现核查，不改写下方历史状态段。详细证据见 [P3 基础实现复核与修复审核方案](evidence/v4_2/p3_foundation_review_20261002.md)。

- 已确认两个 isolated implementation defects：① final speed/timing proposal 在当前 final-approach generator fixture 中未形成不同 physical trajectory/hash（两种 geometry 各 25 rows 均为单一 hash）；② 最近 task-neutral、`task_priority_enabled=false` 的 88D observation 中 STANDARD/CARGO/EMERGENCY/SURVEILLANCE 在同一 physical fixture 下 byte-equal；显式 own service duration 与 earliest compatible-LZ release wait 两个新增字段在 90D append 中提供。priority-enabled 88D raw mission token 可区分 mission；因此该 alias 不泛化为全项目 88D 问题，也不回溯解释最早 priority-aware P3 的全部退化。两项均为 isolated contract 事实，不是 closed-loop performance 或全局根因结论，因果份额尚未测定。
- `diagnose_p3_foundation_contracts_20261002.py` v3 正常 CLI 在 Windows 11、CPython `3.12.14`、NumPy `2.3.5` 上为 `CONTRACTS_READY`、exit `0`、`technical_errors=[]`；最新本地 artifact 为 `.tmp/p3_foundation_review_20261002_contracts_v3_local.json`，v1/v2 保留。
- 初始基础 suite 为 `67 passed / 4 failed`；主审复核 numerical fixture 后确认旧 SOC `.37` 的 negative 断言不成立，`.25` fixture 才产生负 route slack，而立即应急 required SOC `.2060807` 仍可行。修复仅涉及两个测试文件，生产规则未改；targeted `16 passed`，原 9-file suite `71 passed`、exit `0`，SHA parity 已核验。P3 foundation report 保留初始失败表及修复依据。
- 最新 task-neutral dense24 的 `_build_neutral_inputs()` builder-only runtime verification 已在服务器 `/home/user/envs/terminal_env/bin/python` 执行并 exit `0`；无 environment reset/step、模型或训练。实际 `neutral[dense24]` 确认 energy/emergency priority guards=true、trigger `.40`、`continuous_inward_v1`、final capacity `3`、nearest-available、downstream-priority、canonical WAIT、legacy student decision mask、SOC urgency time weight `2`，以及 `observation_schema_version=phaseA_low_obs_v2_1_retry_state`（88D）。这确认两个 guard 已打开；不是 runtime rollout 或 performance result。
- V4.2 gate/proposal decode、WAIT zeroing、sampled requested proposal + K3 fallback、requested PPO likelihood 保存、scalar team reward、actor-mask 和 complete-cycle aggregation 已核查。`gamma=lambda=1`/full-MC 旧证据排除了已记录的折扣误用，但不能据此确立 team-credit、entropy 或 clipping 根因；queue/service business cost 在当前 scalar reward 中的错位贡献尚未量化。
- P3 仍 unresolved，`PPO_ALLOWED=false`；无 training、held-out 或新 checkpoint。首阶段生产修改与 90D schema/权重迁移仍待用户审核，先做 opt-in 小样本无训练闭环验证。

> **LATEST P3 STATUS — 2026-10-01 normal-inner gate-value FULL terminal:** The mechanical run is terminal FULL_OK; PID 2191437 exited. Full artifact SHA=1280dc0a5697913749f22982cff89aa7f0ff2a8075f4bb05b02331033ff7cdba. The immutable pilot was reused and 312 new branches were run, for 416 natural branches over 2749 exact-replay ticks; censor/error/hardsafety counts were zero. Source/model/config/controller/caller-RNG and replay checks passed. PPO_ALLOWED=false; no background job remains.

All four selected states have C=1. The exact full-cycle values are:

| cell / selected state | p_before -> p_after | delta p | D by continuation seed | mean D | local_m |
|---|---:|---:|---|---:|---:|
| C1 (step=307, UID=15, arrival=15, SOC=0.7262173226131179) | 0.9952911138534546 -> 0.9953116774559021 | +2.0563602447509766e-05 | -17.633685634105575, -0.03883038093772484, 16.457156780716787, -9.668157336944148 | -2.7208791428176653 | -5.595107700062362e-05 |
| C2 (step=321, UID=25, arrival=15, SOC=0.6731115455491892) | 0.06313349306583405 -> 0.054939594119787216 | -0.00819389894604683 | 8.923167075767708, -13.078767221640362, 16.823774088948483, 0.32785041685423266 | 3.2490060899825153 | -0.026622027576407463 |
| C3 (step=303, UID=20, arrival=15, SOC=0.6404768921709051) | 0.15514136850833893 -> 0.14121808111667633 | -0.013923287391662598 | 12.306142033410083, 4.107853647160425, 11.396477786987532, -0.6990076360662556 | 6.777866457872946 | -0.09437018259527523 |
| C4 (step=365, UID=46, arrival=15, SOC=0.47479269296572224) | 0.9949211478233337 -> 0.9941213726997375 | -0.0007997751235961914 | 0.1476743613766871, 0.020493698685015715, -7.043622299281537, 0.08259313843001337 | -1.6982152751974553 | 0.001358190331613985 |

The four seed signs are mixed in every state. wait and linear_residual are unavailable (None); local_m is state-local and is not summed into a global gain. The earlier startup/progress paragraphs below this block are historical snapshots, not the current run. The prior six-scene AP/BP screen remains failed and P3 remains unresolved; these measurements do not establish a team-credit, entropy, clipping, or universal training cause.

ROOT has selected preparation of a pure-Python soft-target tool only: p0=BEFORE, mu=mean(D), sampleSE=std(ddof=1)/sqrt(4), w=mu^2/(mu^2+SE^2), and heuristic D_eff=w*mu; fixed eta=0.02, per-state KL<=0.001, up to 60-step bisection, no new floor, and endpoints retained. This is not a sign-confidence rule and involves no fitting, optimizer, teacher, PPO, held-out data, weight change, root-cause claim, or global guarantee. ROOT accepted the pure target preparation. Server code commit 3852587 completed 16 distinct tests with exit code 0 and CLI status TARGETS_READY (4 known, 0 unknown). The server output is /home/user/training_outputs/terminal/p3_followup_20260930/counterfactual_gate_targets_20261001/targets_v1.json, SHA=4085299501758c55c476fac9676092ace9d9c69f4ea5d3a68b74f786eaaf1473; maximum per-state KL is 0.0008676730314016359, with server/local parity. This remains target preparation only: no pilot/full phase, optimizer, teacher, PPO, held-out data, weight change, or P3 resolution claim. ROOT's next preparation is full actor anchor collection over the existing four BEFORE-prefix tapes and 2749 exact-replay ticks, retaining every tick's all-UID rows with WAIT/RELEASE/masked/unknown states and an explicit UID-to-row mapping. The scope is motivated only by incomplete coverage of actor rows in the 249 normal requested_gate=1 records; it is not a new root-cause or training-repair claim. The anchor collector reached authoritative terminal status FULL_OK. Corrected proof records full SHA=1129676804003dd5a569a7723015ef9ceddd8c1cc016cb1ffca02c3cd4318d35, PID 2204414 exited, and heartbeat p3-gate-anchor-collection deleted. The four existing tapes covered 2749 ticks and actor-row counts 7378, 6909, 7070, and 7213, totaling 28570 rows. Gate counts were 1145 true, 27425 false, and 0 unknown; actor-mask counts were 9320 true, 19250 false, and 0 unknown. There were 224 empty ticks, not empty rows; the earlier C4 empty-count report of 70 was superseded by the corrected proof value 93, giving 60+30+41+93=224. All maps, source/replay/reset/config, caller-RNG before/after, controller, and source-certificate checks passed. The pilot remained immutable; no model fitting or new population was used. Stale RUNNING progress is historical. Fit-readiness inputs and metadata are READY_FOR_ROOT_REVIEW: full manifest SHA=1129676804003dd5a569a7723015ef9ceddd8c1cc016cb1ffca02c3cd4318d35, readiness SHA=646f95f7d944a58ab2c6a0950b48fd4e1f5f99cc8f838d826ce8b0ca37ee809a, four target maps, 2749 ticks, 28570 rows, and all certificate/mapping checks passed. Safe preflight commit 92aa86d3d71afec368acfdcfcba3a8ed3c383465 completed 14 local/server tests; the collector commit 8fcbd6e and the soft-target commit 3852587 retain their separate passing test records, with the latter counted as 16 distinct tests. The recorded triplets are p3_counterfactual_gate_targets_20261001.py / test_p3_counterfactual_gate_targets_20261001.py / p3_counterfactual_gate_targets_protocol_20261001.md; collect_p3_gate_anchor_states_20261001.py / test_p3_gate_anchor_states_20261001.py / p3_gate_anchor_states_protocol_20261001.md; and p3_counterfactual_gate_fit_preflight_20261001.py / test_p3_counterfactual_gate_fit_preflight_20261001.py / p3_counterfactual_gate_fit_preregistration_20261001.md. The 16, 9, and 14 counts are distinct per triplet and are not cumulative. No optimizer, model, gradient, checkpoint-writer, or held-out operation was used. P3 remains unresolved and no job or monitor remains. The single user-approved finite fit has now executed once and terminated with FIT_SCREEN_FAIL/ROLLBACK_STOP. Output is counterfactual_gate_fit_20261001; preflight SHA=a3aee30499e5890b1597057ffc1f2a9f1d65fe5dd4d68e398379ef3d4815c8aa, fit SHA=dc8890b56cf0cc6f44bea5cdf7f832a573ccc43670717d3c4dc6a892a136fab5, code commit=511ca42, and the local/server focused suite has 10 distinct tests with matching results. PID 2209548 ended with 0 accepted and 1 rejected candidate. The first rejection had candidate KL=0.03208379613660517, target KL=0.011689337256158998, and cap=0.001; final rollback restored both KL values to 0. Initial and final BCE were both 0.1893708452074146; initial and final p were [0.9952911425998857, 0.06313348800762023, 0.15514133644666978, 0.9949211148729823], while p-star was [0.9952601652648472, 0.06392553361964069, 0.1704190621781801, 0.9948388080341942]. Frozen/q/source/RNG checks were true; optimizer_called was true and weights_changed was false. Candidate, error, and partial artifacts were absent. No environment, teacher, held-out, PPO, or performance evaluation was used; elapsed time was 0.7480152603238821 seconds. ROOT accepted this negative screen result without relaxing the gate, retrying with another learning rate, choosing a seed, or resampling; P3 remains unresolved. No background job remains. Current work is read-only source-gradient diagnosis; no second fit has been admitted. Read-only source-gradient diagnostic: first-order gate-logit deltas were [0.1780496443, 0.3744040335, 0.3879471889, 0.3925988156], all positive. C1 and C4 soft targets therefore require a decrease in this first-order direction. Anchor KL was 0, gradient L2 was approximately 1e-18, and source-q hash/RNG checks were unchanged. ROOT records this as evidence of a current shared-gradient direction conflict and first-step overshoot; it is not a global P3 root-cause claim, and the first-order approximation is not a post-step probability measurement. The read-only structural proof reports H_t shape 4 x 97 with rank 4 and singular values [13.5657020230, 5.8354903501, 2.0122989667, 1.3534608620]; hidden shape is 4 x 96 and the last-layer weight/bias shapes are 2 x 96 and 2 in float32 at T=2.5. Global anchor indices are [4873, 12277, 18971, 26870]. This is structural solvability evidence only; source/RNG were unchanged and no autograd, optimizer, weight update, candidate, or performance result occurred. Structural proof SHA=a03ded5c682ae909c3ea0ab91cc31ba45a29136fd5c3b000b03e992113012127. The new last-layer projection protocol is DRAFT_UNADMITTED at docs/evidence/v4_2/p3_counterfactual_gate_last_layer_projection_protocol_20261002.md; it has not been implemented or executed. The 249 normal events from the separate event diagnostic contain embedded 88D worlds, not NPZ files, and are only future anchor inputs; no fitting has been done.

Pure NumPy helper preparation has completed: `terminal_marl/training/p3_gate_last_layer_projection_20261002.py` with `tests/test_p3_gate_last_layer_projection_20261002.py` passed 9 synthetic tests. This did not solve real data, apply weights, load or modify a model/checkpoint, generate a candidate, or start a new job. The protocol remains `DRAFT_UNADMITTED`; one new closed-form fit remains pending explicit human approval and is not execution admission.


> **历史证据（fresh six-scene 与旧 trained gate-value C=0 cohort；与本段新 normal-inner C1–C4 不同 cohort，2026-10-01）：** fresh V4 six-scene screen 为 FAIL，P3 仍未解决。四个既有 trained gate-value states 均为 `C=0`。沿用现有标签，AP=`132N/12E`、BP=`134N/10E`，两者 `F/D/U=0`；AP 相对 BP 的三项成本差为 `+2.9843324913833236%`、`+0.5281622267627961%`、`+0.7351975729166904%`。prefix 修复的业务差异属于执行比较证据，不是训练修复证据。same-prefix 在扣除 rejection 后的 raw return 为 AP `2967.1651355245745`、BP `2987.9587313050633`；AP 小于 BP，不能只解释为 rejection 处罚。
>
> batch-04 larger-batch 协议与 metadata 一致：`gamma=lambda=1`、`full_episode_mc`、joint-team actor aggregation、minibatch `128`。此前与 learning-curve 协议的错误关联已在 [.tmp objective correction](../.tmp/p3_prefix_objective_protocol_correction.json) 中纠正；未发现折扣参数误用。scalar team reward 没有单独的 gate-outcome 通道，normal、emergency、failed 信号经环境 scalar reward 进入；不据此称 team credit 或 energy absence 为根因。
>
> 新增 [prefix-feasible-gate protocol](evidence/v4_2/p3_prefix_feasible_gate_protocol_20261001.md)，只回放四个固定、既有的 BEFORE-prefix scenes，每个 cell exact replay 一次，full input SHA=`30c28c03d26466eabeab2a35c9fc117424648f1a3ae17be6f8d69b6b92cb2195`。按 `(step, numeric UID)` 原始顺序，先保存 pre-state、执行原 tape action 一次并确认 `last_release_decisions[uid]=approved` 且 rank `>3`；选择窗口还要求 `ring_inner`、`0.4<SOC<=0.65`、actual `actor_mask=true`、retry-free `now>=safety_retry_after`、`committed_plan=None`。在保存的 88D world 上以 no-grad 比较 BEFORE/AFTER 的 gate `p` 与全 25-way `q`；`D/local_m` 保持 unknown。被物理条件排除的候选仅记录计数/原因，不计算 `p/D`，缺失不替换 scene 或扩大预算。
>
> `19` 个 target tests、compile、local serialized preflight 已 PASS；server/local 3-file SHA equality 已确认（代码 commit `afa9bfaab1e9e8f2ae43f661b8f24d890eb79cd5`），server preflight=`PREFLIGHT_OK`、phase validation=`PHASE_VALIDATION_OK`。ROOT 已完成 cell 1 pilot 并准入 FULL：cell 1 immutable reuse 加 cells 2–4 均已完成，FULL=`FULL_OK`，full SHA=`9188e0097acf7c9209cea0e51d890c1de0e9663860f9eb4672601e2ac8e01b6c`；四个 cell 均为 `COMPLETED_MISSING`，exact replay ticks=`2749`（`708+674+644+723`），potential candidates=`20/22/23/24`（合计 `89`）全部因 `physical_window` 排除，四个 cell 均 selected=`null`；model/config/reset/caller-RNG/controller checks 通过，无 error/partial artifact，未生成 selected NPZ，`p/q/D/m` 保持 unknown。PID `2179064` 已退出，FULL heartbeat 已删除；该 narrow-window 阶段为历史终态，不作 gate causal conclusion；其后续 event diagnostic 已完成，见下方 terminal proof。下一步由 ROOT 决定另行 preregister mechanically first normal inner event per cell full-cycle gate-value under SAME prefix25 executor；尚未准入训练，`PPO_ALLOWED=false`。

> 新 event diagnostic runner（code commit `45484dba4377f9c46eb996f485bdbd103e6b5278`；protocol=`docs/evidence/v4_2/p3_prefix_gate_events_protocol_20261001.md`；runner=`terminal_marl/training/diagnose_p3_prefix_gate_events_20261001.py`；tests=`tests/test_p3_prefix_gate_events_20261001.py`）：`16` 项定向 tests、compile、local serialized preflight 已 PASS；server `PREFLIGHT_OK`、serialized validation exit `0`，实际 preflight SHA=`633c2a525c7a797b6adb4494b3adae7121f79110fc69042bab1f18ae90933150`，runner SHA=`8627e72f2a3aa66c917f65673b1f63f13a326c7f9a65416a837d24972b30cdef`，protocol SHA=`59d812d5ff1a53d18b7407dedf25a2b7d107902087796e4bd3a97f41910ff6e5`，server/local 3-file SHA equality 已确认。cell 1 pilot PID `2182083` 已退出 `PILOT_OK`，FULL PID `2183147` 已退出 `FULL_OK`；四 cell exact replay 合计 `2749` ticks，full SHA=`fb89f62339755f7ebcd2bb0f9672438000fe8b754067cd2cdb28bf8407feb3e6`，pilot cell immutable reuse，model/RNG/config/reset/controller checks 通过，q/source logits/source gate-logprob 对全部 `249` 个 normal events 匹配，`D/local_m=None`。逐 cell 事实为：C1 potential/normal/excluded/auto=`314/62/251/1`，phase inner/middle/outer=`121/128/65`，rank first3/late/unknown=`42/20/252`，inner `n=22` mean Δp=`-0.01187748529694297`（20 negative/2 positive）；C2=`267/62/202/3`，phase=`87/102/78`，rank=`41/22/204`，inner `n=22` mean=`-0.02123798243701458`（21/1）；C3=`290/63/227/0`，phase=`87/141/62`，rank=`40/23/227`，inner `n=23` mean=`-0.013047807242559351`（21/2）；C4=`274/62/209/3`，phase=`85/113/76`，rank=`38/24/212`，inner `n=22` mean=`-0.01429183374751698`（21/1）。总计 potential/normal/excluded/auto=`1145/249/889/7`，inner=`89`（83 negative/6 positive）；均为四个固定场景的描述性结果，不是独立样本，不产生 root-cause 或 global-gain 结论。`p/q` 已测，未生成 `D/m`；FULL heartbeat 已删除；该 event 阶段终态时无 active job。其后续 normal-inner pilot 正在运行，状态见下段。

> **2026-10-01 normal-inner gate-value pilot (mechanical selection; startup/progress snapshot)** The normal-inner gate-value runner/protocol was prepared under code commit `7e5d531`; `12` distinct local/server tests passed. The server no-env preflight SHA is `a247e7cdbaee8f9686b67cf6506ba2ad5e7d59f5d38df09a0088f7c6629cd5f8`; runner=`0a300fbef347bc8b428619d158baf9c85cd0deb3343d213e6b4ca3f0f5084e4d`, test=`5b1ec7bccbe4dfc3bde40202ed5db7a4d47cd3672a59b19c29d69d2ac98a7015`, protocol=`086f6e5cc42941cf2e564d7cfcbb44c057d3cdeba0e2936340d6ece80dcd6be6`.
>
> ROOT admitted only the `pilot` phase (`104` branches: `WAIT + 25 RELEASE = 26 branches per seed x 4 continuation seeds`); the `312` new full-phase branches remain unadmitted. The first startup snapshot at `2026-10-01T19:38:52+08:00` showed PID `2187800` (parent `2187799`) running in `/home/user/projects/terminal` with the registered module, `/usr/bin/python3.12`, and `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Output root: `/home/user/training_outputs/terminal/p3_followup_20260930/prefix_normal_gate_value_20261001`; startup proof=`.tmp/p3_prefix_normal_gate_value_pilot_first_output_proof.json`.
>
> The fixed states are cell steps `(307, 321, 303, 365)` with `arrival_id=15`; cell 1 is step `307`, numeric UID `15`, SOC `0.7262173226131179`. Selection used the first normal-inner event rule and was not selected by p delta, reward, or terminal outcome. The first observed branch was WAIT, continuation seed `9303401`, `436` steps, `natural_terminal`, `22N/2E/0F/0D/0U`, safety violation false, undiscounted return `289.6397535818527`, and request elapsed time `5653.0` seconds. `D` and `local_m` are unknown.
>
> The subsequent snapshot at `2026-10-01T19:44:54+08:00` still showed PID `2187800` running after `00:10:57`, with `32/104` pilot branches completed (cell 1, continuation seed `9303402`, proposal `4`). `pilot.json`, `cell_01.json`, `error.json`, and `partial.json` were absent at that snapshot; the log was `0` bytes. Full phase and group metrics had not started/been produced; progress proof=`.tmp/p3_prefix_normal_gate_value_pilot_snapshot_194454.json`. Monitor `p3-normal-inner-gate-value` remained active; no PPO or training phase was admitted (`PPO_ALLOWED=false`).
>
> **2026-10-01 normal-inner pilot terminal and FULL startup** Pilot terminal proof=`.tmp/p3_prefix_normal_gate_value_pilot_terminal_proof.json` reports `PILOT_OK`; PID `2187800` exited. The pilot artifact SHA is `b5138ee3493fbfeaf22172f0107251a1897175878914727c21bd52e424a85332`, cell 1 SHA is `ace3e81b49a05011497fd5a3f6cb3a3c90790d6b114a67d41a325c77f509605c`, and all `104` branches were natural with zero censor/error/hardsafety; exact replay used `708` steps and model/RNG/source checks passed.
>
> Cell 1 selection was step `307`, UID/arrival `15`, `ring_inner`, actor mask true, committed plan false, retry-free. It had `C=1.0`, `p_before=0.9952911138534546`, `p_after=0.9953116774559021`, Δp=`2.0563602447509766e-05`, `D=-2.7208791428176653`, and `local_m=-5.595107700062362e-05`. Per-seed D was `[-17.633685634105575, -0.03883038093772484, 16.457156780716787, -9.668157336944148]`; wait and linear-residual fields were unavailable. These are one selected-state measurements and do not form a global gain.
>
> ROOT then admitted FULL with `FULL_GUARD_OK`, reusing immutable cell 1 and running only cells 2–4 (`312` new branches, `max_steps=1800`); cell 1 was not rerun. FULL startup proof=`.tmp/p3_prefix_normal_gate_value_full_startup_proof.json`; first-output proof=`.tmp/p3_prefix_normal_gate_value_full_first_output_proof.json`. At `2026-10-01T20:16:04+08:00`, PID `2191437` (parent `2191436`) was running with the registered `/home/user/projects/terminal` module, `/usr/bin/python3.12`, and all three runtime flags set to `1`; `PPO_ALLOWED=false`.
>
> The first new FULL branch was `/home/user/training_outputs/terminal/p3_followup_20260930/prefix_normal_gate_value_20261001/branches/cell_02_001.json`: WAIT, seed `9303401`, proposal `null`, `419` steps, natural terminal, `23N/1E/0F/0D/0U`, safety violation false, return `281.56366903465454`, request elapsed `5608.0` seconds, branch wall `18.729321520775557` seconds. Progress was cell 2 `2/104`; full/error/partial and cell 2–4 summary artifacts were absent at that snapshot. C/D/local_m for the new cells were not yet available.
>
> **2026-10-01 normal-inner FULL progress: C2 completed, C3 running** Proof=`.tmp/p3_prefix_normal_gate_value_full_cell02_proof.json` records the same PID `2191437` and runtime bindings; cell 2 completed exact `674`-step replay with `104` natural branches and zero censor/error/hardsafety. Its selected state is step `321`, UID `25`, arrival `15`, SOC `0.6731115455491892`, `ring_inner`, actor mask true, retry-free, and no committed plan. Cell 2 has `C=1.0`, `p_before=0.06313349306583405`, `p_after=0.054939594119787216`, Δp=`-0.00819389894604683`, `D=+3.2490060899825153`, and `local_m=-0.026622027576407463`; wait and linear-residual fields are unavailable. Per-seed D is `[8.923167075767708, -13.078767221640362, 16.823774088948483, 0.32785041685423266]`; the q-weighted recomputation matched within `5.7e-14` per seed and mean `+3.249006089982558`. This is a state-local measurement; C3/C4 D and local_m remain unknown.
>
> The same snapshot showed C3 running at `2/104` branches (seed `9303401`, proposal `0`), with full/error/partial and C3/C4 summaries absent. Cell 1 remained immutable and was not rerun; `PPO_ALLOWED=false`; the active monitor remained `p3-normal-inner-gate-value`. The C2/C3 branch and cost fields remain descriptive counterfactual results, not a global conclusion.
>
> **2026-10-01 normal-inner FULL progress: C3 completed, C4 not yet observed** Proof=`.tmp/p3_prefix_normal_gate_value_full_cell03_proof.json` records PID `2191437` with the same cwd/argv/exe/runtime flags. Cell 3 completed exact `644`-step replay with `104` natural branches and zero censor/error/hardsafety. Its selected state is step `303`, UID `20`, arrival `15`, SOC `0.6404768921709051`, `ring_inner`, actor mask true, retry-free, and no committed plan. Cell 3 has `C=1.0`, `p_before=0.15514136850833893`, `p_after=0.14121808111667633`, Δp=`-0.013923287391662598`, `D=+6.777866457872946`, and `local_m=-0.09437018259527523`; wait and linear-residual fields are unavailable. Per-seed D is `[12.306142033410083, 4.107853647160425, 11.396477786987532, -0.6990076360662556]`; q-weighted recomputation matched within `5.7e-14` per seed and mean `+6.777866457872975`. This is state-local evidence; C4 D/local_m remain unknown.
>
> The proof status label is `RUNNING_CELL04`, but the observed progress was still cell 3 `104/104` (seed `9303404`, proposal `24`); no C4 first output was observed. Full/error/partial and C4 summary artifacts were absent, while the immutable pilot and C1/C2/C3 results remained unchanged. `PPO_ALLOWED=false`; the active monitor continued.
>
> **历史快照：LATEST FRESH VALIDATION STATUS（2026-10-01）：** Profile 16-arm FULL 已完成，保留 commit `903c083` 与 `25` 项 focused tests 证据；该 profile 只回放既有 tapes。fresh validation V1 的 caller-RNG check 在 restore 前执行导致技术停止，V2 的 replay-factory 计数检查也导致技术停止；两次均保留，V2 的真实 `778`-step 首臂 actions、requests、outcomes 与业务字段与 V1 一致。计数修复 commit `0556a123` 与 `10` 项定向 tests 已通过。
>
> V3 no-env preflight 的真实序列化校验已 `PASS`，preflight SHA=`9e8602a3c3a4c005b9559981654b88082efd8da908e31ec917cbb093ce40c451`；runner=`14d95...`，test=`705b...`，shared protocol=`6571...`。ROOT 已准入 V3 pilot：PID `2164825`，cwd `/home/user/projects/terminal`，phase=`pilot`，输出根为 `/home/user/training_outputs/terminal/p3_followup_20260930/prefix_fallback_fresh_validation_20261001_v3`，startup snapshot 为 step `150`、`21` generated、`16` active，三个 env flags 为 `1`，proof SHA 前缀=`6E625...`。目前只有首 scene/arm 的 startup，V1/V2 outputs 保留；必须先对 V3 首 `before_legacy` 与 V1 的 actions、requests、outcomes、wait/elapsed/energy、raw reward 和 rejection penalty 做 exact 复核，ROOT review 后才决定其余 `20` arms。monitor `p3-prefix-repair-validation` ACTIVE、间隔 `3` 分钟。`PPO_ALLOWED=false`，无 training/optimizer/teacher/held-out/default promotion，P3 未解决。
>
> 旧 `b5c017...` mixed-newline suffix 的原始 bytes 未在当前 `.tmp` 候选中恢复；当前工作树 suffix 保持其现状（SHA=`a2f8b65c37cfc5dd80c3e8f1f0564917eed4755b5ccaf3fc18bd768cb5edf667`），ROOT 已异步询问原始副本路径。本次文档更新只修改该文件 marker 前的 prefix。

> **历史快照：LATEST PROFILE STATUS（2026-10-01）：** V2 passive profile FULL 已 `FULL_OK`，16 arms 全部 natural、0 censor/error，exact 24-request、binding、replay、model/caller RNG、reward、ledger、safety provenance checks 全通过；full SHA=`d0434e370199a7ce990be4b5b6305d07f383a3c8cea591e42ce463b252d0b1db`，pilot SHA=`ddbe49caa1121f14e47f404b01c32fd409056ba1face3ec14215d77e01b72d7f`。不可变四-cell evidence 的 legacy 为 BEFORE `80N/16E`、AFTER `74N/22E`，prefix 为 BEFORE `84N/12E`、AFTER `88N/8E`；profile 只回放既有 tapes，不是新 performance population。四组 proposal/suffix/late approval 为 BL `2576/0/0`、BP `17970/15485/85`、AL `2618/0/0`、AP `17682/15250/83`；正确 suffix count 使用 per-release delta，历史 raw suffix 字段 `15` 保留为 erratum。FULL 的 instrumented timing 与 reason counters 已记录；observer overhead 未测得（记录 count=`0` 不等于零开销）、actor outputs 未执行且 RNG 已恢复，最大 env-step 为 `3472.175239 ms`；均为 offline simulation measurements，不作 hardware deadline claim。fresh six-scene validation 的 cell 1 四 arms 已获准但未完成：PID `2160913` 已于 `2026-10-01 14:21:26+0800` 退出，终态为 `PILOT_TECHNICAL_ERROR`（`caller_rng_unchanged is not true`），仅 `before_legacy` 首臂完成；该臂 `778` steps、`24` requests、`22N/2E/0F/0D/0U`，request identity、ledger、replay、safety 和业务字段已观测，但 fresh pilot 证书未接受，`FULL` 不存在，heartbeat 已删除且当前无后台任务。ROOT 已选择独立输出根 `prefix_fallback_fresh_validation_20261001_v2` 的技术恢复校验：固定同一 `6` scenes×`4` arms、原 seeds 和 cap=`1800`，保留 V1 error/首臂，不选择 seed/权重、不扩展 population、不调整业务 acceptance。native RNG restore 后 check 的最小修复由 commit `e73a221` 接受，定向 tests=`10`；V2 actual preflight SHA=`3a48b81fc7a77dbe7aec34e67b4c4b711206ec78349871c7cc465e3d39ac2b78` 已通过，ROOT 已准入 pilot。PID `2163291` 当前按预注册 module/cwd/argv 运行，cell 1 `before_legacy` 为 step `150`、`21` generated、`16` active，三个 env flags 为 `1`，输出根为 `.../prefix_fallback_fresh_validation_20261001_v2`；monitor `p3-prefix-repair-validation` 已 ACTIVE、间隔 `3` 分钟。首臂终态必须先与 V1 的 actions、requests、outcomes、wait/elapsed/energy、raw reward 和 rejection penalty 精确复核；FULL 仅在 ROOT review 后处理其余 `20` arms，不自动训练。`PPO_ALLOWED=false`，P3 未解决，不作 formal/PPO promotion 或 unique root-cause claim。

> **历史 PROFILE 运行快照（原 LATEST PROFILE STATUS，2026-10-01）：** profile V1 在首个 arm 后以 `PILOT_FAILED` 终止，未运行第二个 arm 或 FULL；该状态与 earlier startup snapshot 分开记录。安全 provenance fix 已由 commit `b5be014` 接受，profile focused tests 最新为 `20` 项（替代此前 `10` 项记录，不与旧测试累计）；原始 `uncertified_execution=None` 保留，derived certificate 只在 immutable source、exact config、full replay、无 blocked、known primitive 全为零及 input-hash conditions 全通过时标记，任何 positive value 不被覆盖。V2 technical-recovery server no-env preflight SHA=`e0022ca5b1648eacb96fded39a515d9b2c6128dd92ae2f1d1a078210ff884506`，previous replay with original-source summary 的 derivation conditions 全通过，proof=`.tmp/p3_prefix_profile_v2_server_preflight_proof.json`。V2 pilot PID `2157928` 已 `PILOT_OK` 退出，原始 `2` 个 existing tapes 的两臂均通过 bindings、metrics、reward、safety provenance、model RNG 与 exact 24-request checks；proof=`.tmp/p3_prefix_profile_v2_pilot_snapshot_proof.json`。BEFORE-legacy 与 BEFORE-prefix 均为 `19N/5E/0F/0D/0U`，steps=`761/723`、wait=`3327.4996213/3123.9868664`、elapsed=`9050/8878`、energy=`7757.9171088/7513.1647880`；timing 为 instrumented offline simulation wall，不是 hardware deadline proof。counter proof=`.tmp/p3_prefix_profile_v2_pilot_counter_proof.json` 给出 total proposal=`4865`、approvals=`15`、正确 distinct suffix count=`4206`，由 `sum(max(delta-3,0))` 与 distinct `reason_rank>3` 两种聚合一致；raw `suffix_proposal_checks=15` 保留但标为 semantic erratum，不作 accepted count。FULL 已准入并运行 PID `2158963`，当前 cell 1 `after_legacy` 快照为 step `250`、`24` generated、`17` active、`250` native probes、`3692` release calls；immutable pilot 两臂复用，另运行剩余 `14` tapes，无新 trajectories，FULL metrics unknown。heartbeat `p3-prefix-repair-validation` 已更新，间隔 `3` 分钟；V1 保留且不计为 accepted profile，`PPO_ALLOWED=false`，不作 performance、physics root-cause 或 P3 solved claim。

> **历史汇总（原 LATEST ROLLING STATUS，2026-10-01）：** 当前 trained regression 已复现；uniform-q floor candidate 已拒绝；matched-state gate-shift causal diagnostic 已完成 `2918` steps；causal tool tests `18` 项通过。trained gate-value V2 FULL 已 `FULL_OK`：4 states、416 branches 全部 natural，0 censor/error/hardsafety，pilot reused，full SHA=`056bbc9d4f4949c7c932b1b21c49204b2533d1fe1f7b1fe44b48a2e845ee5a5a`。C1/C2/C3/C4 full `D` 均值分别为 `-0.049999999999926104`、`+0.2713038294623118`、`-0.049999999999954525` 与 `-1.2160576541882904`；该机制诊断不支持 universal increased-RELEASE 或 causal training blame；prefix fallback V2 FULL 已完成并通过 development screen only，历史 gate-value 结论不变。BEFORE-prefix FULL 已 `FULL_OK` 退出：4 scenes natural、0 censor/error/new failed/diverted/unresolved IDs，exact 24-request identity、model/RNG/config/dependency/alias/source-q/replay/ledger/energy/safety checks 全通过；aggregate candidate/reference 为 `84/12/0/0/0` 对 `80/16/0/0/0`，wait=`11026.483499761933`、elapsed=`34876`、energy=`28895.644581442906`，full SHA=`b4ee3a191c26911edf94fd24e1da4077bd2a0962e10cd5507174b52a8af73028`。该结果为 fixed development screen only；heartbeat 已删除，无后台任务，下一步记录为测量累计 fallback checks 与实际 inference latency，并单独复核 opt-in executor/admission 后再考虑 PPO/default promotion。`PPO_ALLOWED=false`。

> **Prefix fallback rolling status（2026-10-01）：** V1 no-env preflight 为 `PREFLIGHT_OK`（SHA=`c82a5a749bc4afa92468a9f7c6e2ddbf28748afa6568e3766a80501ca44cab46`），pilot PID `2136806` 以 `technical_error` 退出；保留的后续 pilot artifact SHA=`1cf74aa3ee3369ad0da4604d7a9e8b64e04e31b465f25ef8aca8b1b8ab554b10`，V1 full metrics unknown，未计性能。singleton helper 修复已通过 `21` 项定向测试、compile 和 diff check。V2 pilot PID `2138749` 已 `PILOT_OK/PAIR_OK` 退出；24 requests 全部 generated/admitted/resolved/successful，candidate/reference 为 `21/3/0/0/0` 对 `18/6/0/0/0`，技术、source/RNG、q、replay、ledger、energy 和 identity checks 全通过。V2 FULL PID `2139795` 已退出 `FULL_OK`，4 cells natural、0 censor/error，candidate/reference aggregate 为 `88/8/0/0/0` 对 `74/22/0/0/0`，wait/elapsed/energy 均下降但 cell4 局部 elapsed/energy 上升；该结果为 development screen only，不是 qualification/default/PPO。BEFORE-prefix pilot 已 `PILOT_OK/PAIR_OK` 退出；BEFORE-prefix FULL PID `2145833` 已 `FULL_OK` 退出，4 scenes natural、0 censor/error，aggregate candidate/reference 为 `84/12/0/0/0` 对 `80/16/0/0/0`，wait/elapsed/energy 均低于 native BEFORE。`p3-prefix-fallback` 与 `p3-before-prefix-fallback` heartbeat 均已删除，无后台任务。实验 controller 为 public K=3 prefix + max25 fallback，native 初始 policy seed 保持原 Torch RNG 序列，非 perfect CRN；all-energy exhaustion 扩展到 25 可能改变 induced emergency 行为。`PPO_ALLOWED=false`，P3 未解决，不自动 relaunch、不重采样、不使用 PPO/held-out/teacher/optimizer。

> **本轮 preregistration / preflight 状态（2026-10-01）：** passive profile protocol 与 runner 已分别由 commit `2ba8f83`、`5fd7f5a` 固化，profile focused tests 最新为 `20` 项；fresh validation runner commit `0ddf8b7` 的 selected suite 为 `38` 个 distinct tests，相关 suites 含既有回归，不累计历史重复测试。profile server no-env preflight SHA=`7d2745abaef5c06ca9eb45583913884d5ed7f67057b2d7dd685ea150eac9db4d`，serialized validator 通过 `16` tapes、`8` sidecars、`3` full artifacts；fresh no-env preflight SHA=`e25cde37af2da6ac1988e7c36dd1097d8bebb956be807441a8837c68046e2b92`，通过 `6` scenes×`4` arms（env `886101..886106`、policy `886201..886206`、cap `1800`）。V1 profile pilot 已在首个 arm 后以 `PILOT_FAILED` 终止；其旧 startup snapshot 保留为历史，V2 technical-recovery 另用输出根 `prefix_fallback_profile_20261001_v2`，原 V1 output 保留。V2 no-env server preflight SHA=`e0022ca5b1648eacb96fded39a515d9b2c6128dd92ae2f1d1a078210ff884506`，previous replay with original-source summary 的 derivation conditions 全通过，proof=`.tmp/p3_prefix_profile_v2_server_preflight_proof.json`；V2 pilot PID `2157928` 已 `PILOT_OK` 退出，2/2 arms 的 bindings、metrics、reward、safety provenance、model RNG 与 exact 24-request checks 通过；counter proof 给出 total proposal=`4865`、approvals=`15`、correct distinct suffix count=`4206`，raw field `15` 保留为 semantic erratum。FULL PID `2158963` 已启动，immutable pilot 两臂复用并运行剩余 `14` tapes，当前 metrics unknown，无 new trajectories；`PPO_ALLOWED=false`，held-out 未使用，不做 default promotion。

## 2026-10-01 P3 prefix fallback V1 技术恢复与 V2 准入

本项是已授权的实现调试记录，不是科学重采样或性能评估。V1 与 V2 的尝试、artifact 和状态分开保存；V1 不覆盖、不重跑，也不计为性能成功。

V1 preflight 为 `PREFLIGHT_OK`，SHA=`c82a5a749bc4afa92468a9f7c6e2ddbf28748afa6568e3766a80501ca44cab46`。PID `2136806` 的 pilot 终态为 `technical_error`；较早的“缺少 pilot”只对应早期快照，后续 pilot artifact SHA=`1cf74aa3ee3369ad0da4604d7a9e8b64e04e31b465f25ef8aca8b1b8ab554b10` 已保留。V1 观察到 `31` native forwards、`33` registrations、`generated/admitted/observed_unresolved=14/13/14`，`N/E/F/D=0/0/0/0`，full metrics unknown。model-caller RNG、alias、distribution checks 通过；`technical/hard_safety_clear=false` 仅表示 incomplete certificate，不能据此宣称 actual unsafe 或 verified violation 为零。

当前最小 helper 修复使 native canonical singleton RELEASE 直接委托原始函数，不请求 order provider、不采样、不扩展；已注册的 native K=3 保持 prefix checks，malformed lengths fail closed。该修复处理了包括 policy map 未观察到的新 admission（UID `28`、`calls=0`）的路径；21 项定向测试、compile 和 diff check 已通过。

V2 预注册输出为 `p3_followup_20260930/prefix_fallback_20261001_v2`：保持原四 cells、checkpoint、policy seeds、1800-step cap、thresholds 和 protocol SHA=`29aa458a81e50aba4d0249cb1fe3b856829a1261feac3a3ad507c68fcfbdbc0c`，新 preflight 绑定 helper 修复。V2 preflight 与 fresh JSON phase validation 已通过；pilot PID `2138749` 已 `PILOT_OK/PAIR_OK`，24 requests 的 candidate/reference 为 `21/3/0/0/0` 对 `18/6/0/0/0`，wait/elapsed/energy 均有完整值且全部 checks 通过。pilot 每 UID 的记录为 singleton `24`、extended `0`，这是每 UID 分类而非累计 fallback 次数，累计 fallback 总量未暴露，不能据此声称 never fallback。V2 FULL PID `2139795` 已 `FULL_OK` 退出，4 cells natural、0 censor/error，aggregate candidate/reference 为 `88/8/0/0/0` 对 `74/22/0/0/0`，wait/elapsed/energy 总体下降但 cell4 局部 elapsed/energy 上升；full SHA=`020312813fc2c90a445bd3fe30e94e6739797151fe46efb65f760809a9361c50`，corrected proof SHA=`daf83439a3d007f0cf50bcc94fb2781be794be51e0cafe300b2e937ef4066ca7`。该结果为 development screen only，不是 qualification/default/PPO；baseline walltime 与累计 fallback 检查次数未暴露。输出使用 native 初始 policy seed，不作 perfect keyed-CRN claim；experimental public K=3/max25 controller 的 all-energy exhaustion 可能改变 emergency 行为。下一项只准备 BEFORE-prefix panel 对照，无自动 relaunch、seed resample、population expansion、PPO、held-out、teacher、optimizer 或 default promotion，`PPO_ALLOWED=false`。

V2 FULL 的逐 cell delta（candidate 减 native reference）为：cell1 wait/elapsed/energy=`-232.6590365534803/-106/-149.25083407896545`，cell2=`-951.1555657216227/-765/-880.4334896140444`，cell3=`-678.3272158045793/-675/-707.5583376474897`，cell4=`-2.8214014478126046/+103/+35.67359789283546`。candidate walltime=`721.0561337517574 s`（含 replay），baseline walltime 与累计 fallback 检查次数 unknown；raw reward 与 rejection penalty 分列记录，不能将 penalty 变化单独当作业务改善。AFTER-prefix 仅通过 development screen，下一项为同 controller 的 BEFORE-prefix panel，不把 panel 差异解释为 exact-CRN difference-in-differences 或总体训练根因。


## 2026-09-30 P3 follow-up 修复

ALL24 causal paired diagnostic 已完成：24 个 selected pre-action states × 4 continuation seeds × 26 branches = 2,496 个 branch，均为 `natural_terminal`，0 error/censor；model、environment、caller RNG unchanged checks 均通过。前置诊断中有 14 个 retry-guaranteed states。首状态 4 个 continuation seed 的 D 符号混合；不据单个 seed 的 D<0 作归因。

已选择 source `3fe7+a82`、88D/K3、同一 `dense24` 物理配置的 `legacy` vs `pre_action_forced_wait` 对照，范围为 24 个 complete-scene seed pairs。固定 pilot pair `env877101/keyed9303101` 已 `PILOT_OK/PAIR_OK`；完整运行已 `FULL_OK/pilot_reused`，PID `2063491` 已退出，24 pairs/48 runs 自然终止且 0 censor/error，安全、source/model/caller-RNG、配置/runtime、请求 identity、ledger/energy/wait checks 均通过。forced-WAIT mechanical/probability semantics 已验证，但不提升为 formal/default；P3 仍未解决，后台长任务已结束，`PPO_ALLOWED=false`、heldout 未用。详见 [follow-up 修复与结果](evidence/v4_2/p3_followup_results_20260930.md)、[预注册诊断协议](evidence/v4_2/p3_followup_protocol_20260930.md) 和 [forced-WAIT frozen comparison protocol](evidence/v4_2/p3_forced_wait_frozen_protocol_20260930.md)。

## 2026-09-30 P3 executable-opportunity 预注册准备

已登记 9 个参考 arms：`1/5/9/13/17/21` 对应 `env877101..877106 + keyed9303101`，以及 `14/15/16` 对应 `env877104 + keyed9303102/9303103/9303104`。保持 source `3fe7+a82`、88D、K=3、legacy mask 与同一物理配置；每个场景在 `100–149`、`300–349`、`500–549` 首个合法窗口最多取一个 state，最多 `27×26` 个 one-step `WAIT + 25 proposal` 分支。诊断只使用真实 `q` 计算 `C/pC`，不使用 `D/global value`，不取 best、不删除失败、不重抽样。详见 [executable-opportunity coverage protocol](evidence/v4_2/p3_opportunity_coverage_protocol_20260930.md)。

本轮 `newly_run` local preflight 为 `PREFLIGHT_OK`，compile/diffcheck exit `0`，机会 runner focused suite 为 `11 passed`，combined recheck 为 `37 passed in 8.15s`；服务器只读快照 exit `0`，115-file runtime manifest mismatch `0`，source/warmup checkpoint SHA 与预注册值匹配。纠正 cwd 后 V1 server pilot `PID 2085294` 为 `PILOT_OK/PAIR_OK`：env `877101`、keyed policy `9303101`、`694` ticks、`24` requests、`23 normal/1 emergency`、3 states/78 branches、0 error/censor，source/model/caller-RNG、ledger、energy、wait 和 safety checks 通过；这个新 legacy replay 与原 legacy reference 的 reward、rejection penalty 和排除 penalty 的 raw reward 差均为 `0`。首个 launcher 在 phase 前的 `NameError: name 'subproces' is not defined` 保留于 `.tmp/p3_opportunity_launcher_v1.stderr.txt`，不属于该 pilot phase；修正后的启动 guard 完成了 pilot。paired execution digest 为 `20afdf4b60babc63a0a64eba6702d6f9db3891b537168a5e5346639f148b1d23`，canonical source-binding digest 为 `a8a133863bc03627452814535aa938ed062fe0496b35678d0fe9333524503f07`，二者为同一加载模型的不同摘要算法，不表示权重变化。`preexisting_artifact` 为既有 forced-WAIT `full.json`/24 pairs；V2 preflight、V1 pilot 只读导入 guard 和 full 均已完成：9/9 pair `PAIR_OK`、25/27 state 有结果、650 branches 全部 `ok`，pair14/15 的 window03 未选出并保留 unknown。9 个 arm 的 source/model/caller-RNG、配置、request identity、ledger、energy、wait 和 hard-safety checks 均通过。详细 state 表和 env877104 arrival13 trace 摘要见 [机会诊断终态报告](evidence/v4_2/p3_followup_results_20260930.md)。该诊断仍不是性能或因果评估；`C=0` 不推出 full-cycle 无作用，full `D/m` 未产生，`PPO_ALLOWED=false`、held-out 未使用。

其中 pair9/window02/tick300 的 `C=2.0926044087687933e-18` 是 q-starvation witness；pair15/window01/tick134 的原始 `C=1.0000000000000053`、`p=0.0003375054511707276` 是即时可执行但低 RELEASE 概率、当步未请求的 witness。这不证明等待有害；当前 WAIT 可能有未来收益，是否有害要看完整周期 gate control。多数 selected states 被 native safety/capacity/priority/energy guards 拒绝，未作 PPO 根因归因。历史 frozen-WAIT 对照的 `+2.5` 仍保留在其独立结果段，不归入新 legacy replay。随后完成了同 9 个 pairs、同 source/88D/K=3/legacy eligibility/真实 q/物理配置的 frozen always-RELEASE gate control；candidate review 因 pair `15` 的新 failed arrival `21` 被拒绝。该 control 未训练、未 promote，`PPO_ALLOWED=false`、held-out 未使用。

## 2026-09-30 P3 frozen gate-control 终态

冻结的 off-policy always-RELEASE control 已完成 `FULL_OK`：`9/9 PAIR_OK`、9/9 natural、technical/safety/source-RNG checks 通过。它使用 `6` 个环境 profile 上的 `9` 个固定 keyed pairs，复用既有 9 个 reference scenes，不把 `216` 个请求当作独立实验；source `3fe7+a82`、88D/K=3、legacy eligibility、真实 proposal `q` 和物理配置不变，仅固定 legacy `actor_mask=true` 行为 RELEASE，其余行 canonical WAIT。artifact proof 为 `.tmp/p3_gate_control_full_proof.json`，full SHA256 为 `fa856f3980be848c8cb2fd5971115188a9e2f5303c37d8699b44631e57a58403`；`PPO_ALLOWED=false`、heldout 未使用，commit `3a3e1ba` 未推送，p3-gate 已删除且无后台任务。

Aggregate control/reference 为 `N 190/180`、`E 25/34`、`F 1/2`、`D 0/0`、`U 0/0`，admitted `216/216`，successful completed `215/214`，resolved `216/216`；wait `25124.700815134504/29230.017251347945 s`，request elapsed `79035/82700 s`，energy `64612.22804454819/68515.23095364538`。raw reward excluding penalty 为 `4315.902567907697/4132.633739495724`，rejection penalty 为 `-1140.55/-165.8`，total reward 为 `3175.3525679076965/3966.8337394957234`。候选被拒绝，因为 pair `15` 新增 failed arrival `21`：UID `108`、cargo、初始 SOC `0.5758906587575819`、request `150`、terminal `1013`、`ring_outer`、`failed_energy`、wait `807.8147075862395 s`。pair-level N/E/F/D/U 及 wait/elapsed/energy delta 见[终态结果](evidence/v4_2/p3_followup_results_20260930.md)。

随后已完成 root 批准的 passive exact replay：pair `15` 的 control 与 legacy 各执行一次，source、seeds、config 和 1800-step cap 保持不变，target arrival `21` 的 trace exact checks 通过，结果为 `REPLAY_OK`。其后 proposal-support control full 已 `FULL_OK`：9/9 fixed scenes natural、0 error/censor，技术、source-RNG、request、ledger、energy、wait 和 hard-safety checks 全通过；aggregate control N/E/F/D/U=`183/33/0/0/0`，reference=`180/34/2/0/0`，wait/elapsed/energy 分别下降 `4.7158647%/1.0617%/1.3591%`。随后 balanced proposal-support full 也已 `FULL_OK`：24/24 pair natural、0 error/censor，aggregate control/reference normal=`491/491`、emergency=`85/83`、failed=`0/2`，wait/elapsed/energy 均满足 development review。两项结果均为 development candidate only，不是 qualification、PPO/default promotion 或训练根因结论；`PPO_ALLOWED=false`。

## 2026-10-01 P3 pair-15 passive replay 终态

服务器 PID `2097784` 于 `2026-09-30T16:54:53Z` 终止。固定 pair `15`（`env877104`、keyed `9303103`）的 legacy/control 分别为 `724/1077` ticks；request identity、outcomes、terminal rows、energy、wait、reward、model/RNG 和 hard-safety exact checks 均通过。target arrival `21` 为 cargo：legacy 在 `606` 终止为 `emergency_landed`，control 在 `1013`、`ring_outer` 终止为 `failed_energy`；legacy 为 `ring_outer→ring_middle→ring_inner`，control 为 `ring_outer` 等待 `777` 帧。legacy 有一次成功 emergency attempt（tick `438`、SOC `0.39723546555721995`）；control 有 `120` 个实际 attempt-positive emergency calls，首个 safety-block 记录为 tick `414`、SOC `0.39966843653535883`，首个 energy-block 记录为 tick `664`、SOC `0.23300176986869217`。`58` safety-block 与 `69` energy-block calls 存在重叠，不相加；native call totals 为 `336/899`，含 early-return/retry。首轮 launch proof 的 postprocessing parser 误读 release 字段路径，因而错误报告字段缺失；collector/replay 本身未变。按 `trace.steps[*].targets[str(arrival_id)].after.release_decision` 的 corrected raw field check，control target `21` 的 `ring_outer` `777` 个 release frames 为 `51` `safety_reject` + `661` `safety_wait` + `64` `emergency_energy_guard` + `1` terminal-null；legacy `15` 个 release frames 为 `10` `safety_reject` + `4` `safety_wait` + `1` `approved`，approved 为 step `312`、proposal `10`、shortlist `[10,11,12]`。旧 proof v1 保留；由于没有新的 live server hash 核验，这次修正来源于同步 mirror proof，不是新的 server hash check。`last_safety_decision` labels 保留为 stale labels，不用于推断即时 failure reason。上述均为该 pair 的 causal trace observations，不构成总体 root-cause 结论。artifact status 为 `REPLAY_OK`，proof 为 `.tmp/p3_gate_failure_replay_launch_proof.json`，`PPO_ALLOWED=false`。
随后 server 直读 cross-check 通过：更新后的 proof SHA256 为 `914a4b235916a64f165ebf8073ea5f74828b18cf2af12f1b45e00bb414e651ac`；replay summary 为 `REPLAY_OK`、mismatch `0`、model/RNG/safety checks 为 `true`，server trace SHA 为 legacy `d1b71ef6abfe32feb84c009961b974e369935377eef85db1dd01c10dcd607745`、control `1b185b39f7e26cd0f8d534e3573001bf70bbbd04afb760516f1bf7af8364ec1b`。首次 parser 修正的 mirror 来源属于直读前历史 provenance 记录，与此次 server 直读复核不矛盾。

## 2026-10-01 P3 proposal-support control development 终态

冻结 proposal-support control 的 full 比较已完成。服务器 PID `2102330` 已退出，full mtime 为 `2026-10-01 01:49:36`（Asia/Shanghai）；artifact proof 为 `.tmp/p3_proposal_support_full_proof.json`，full SHA256=`0178f6cef9083babab207d1fbd0e71cc50b4ebd90ffc2e4601caaeb07a8668b5`。9/9 fixed scenes natural、0 error/censor，24 requests、source/model-caller RNG、request identity、ledger/energy/wait 和 hard-safety checks 全通过。该运行只评估冻结 off-policy proposal-support control，无训练、optimizer、teacher、PPO 或 heldout；9 个场景来自 6 个固定 profiles 和 9 个 keyed pairs，不能视作 216 个独立样本。

| pair | scene | ticks | control N/E/F/D/U | reference N/E/F/D/U | Δwait | Δelapsed | Δenergy |
|---:|---|---:|---|---|---:|---:|---:|
| 1 | 877101/9303101 | 722 | 22/2/0/0/0 | 23/1/0/0/0 | +252.53942549349722 | +282 | +324.9373252078476 |
| 5 | 877102/9303101 | 799 | 20/4/0/0/0 | 20/4/0/0/0 | +261.4727465143619 | +363 | +322.5319456248826 |
| 9 | 877103/9303101 | 711 | 20/4/0/0/0 | 20/4/0/0/0 | +367.11554965350433 | +351 | +397.4807547823866 |
| 13 | 877104/9303101 | 758 | 19/5/0/0/0 | 20/3/1/0/0 | -578.3562047072567 | -374 | -359.987016850677 |
| 17 | 877105/9303101 | 685 | 21/3/0/0/0 | 21/3/0/0/0 | -504.171103904288 | -497 | -542.724601236105 |
| 21 | 877106/9303101 | 836 | 19/5/0/0/0 | 20/4/0/0/0 | -176.074608448065 | -154 | -150.762815305935 |
| 14 | 877104/9303102 | 742 | 19/5/0/0/0 | 19/5/0/0/0 | -246.58688277428746 | -274 | -276.163126016308 |
| 15 | 877104/9303103 | 685 | 21/3/0/0/0 | 21/3/0/0/0 | -66.14638229183583 | -64 | -61.602128748916584 |
| 16 | 877104/9303104 | 738 | 22/2/0/0/0 | 16/7/1/0/0 | -688.2406166132155 | -511 | -584.9103552780334 |

聚合 control N/E/F/D/U=`183/33/0/0/0`，admitted/completed/resolved=`216/216/216`；reference N/E/F/D/U=`180/34/2/0/0`，successful completed=`214`。wait 为 `27851.56917427036` 对 `29230.017251347945`，Δ=`-1378.448077077585`（`-4.7158647%`）；elapsed 为 `81822` 对 `82700`，Δ=`-878`（`-1.0617%`）；energy 为 `67584.03093562453` 对 `68515.23095364538`，Δ=`-931.20001802085`（`-1.3591%`）。raw reward（排除 rejection penalty）为 control `4250.50923752765`、reference `4132.63373949572`；rejection penalty 分别为 `-156` 和 `-165.8`，两者分开记录。

按预注册规则，该结果是 development candidate PASS only；不是 qualification、PPO 准入、默认配置晋升，也不能单独证明训练退化根因或 P3 已解决。`PPO_ALLOWED=false`，heldout 未使用。

## 2026-10-01 Balanced proposal-support full 终态 (`FULL_OK`; development candidate)

balanced proposal-support control full 已完成。服务器 PID `2105971` 已退出，full mtime 为 `2026-10-01 02:37:12`（Asia/Shanghai）；artifact proof 为 `.tmp/p3_proposal_support_balanced_full_proof.json`，同步 full JSON 为 `D:/7_codespace/training_outputs/terminal/p3_followup_20260930/proposal_support_balanced_20261001/full.json`，SHA256=`f5cd9041f14c86f3b2e6ebde504a4e98f0eafcc4ee4c404db0939db38b2bb48d`。独立核对 proof 为 `.tmp/p3_proposal_support_balanced_full_proof.json`（SHA256=`02ad0f032a369fc412121e887830da71618092150ffbd08481333a7c05ae8634`），per-pair CSV 为 `.tmp/p3_proposal_support_balanced_full_perpair.csv`（SHA256=`dfdfe17476b01eda3ae3c8c96000fcb60572fabf35fc038878937e01ca3fae8f`）；ALL24 与六 profile 汇总均在 `1e-8` 内，24 control/reference 匹配、reference hashes、reuse9/new15 IDs 和 all checks 均通过。固定 24 个 pair 全部自然终态，0 censor/error；source/model RNG、request identity、ledger、energy、hard-safety checks 全通过，未出现 new failed/unresolved/diverted arrival IDs。范围为复用 9 个既有 control/reference 场景并新增 15 个 control 场景，不能视作 576 个独立样本。

| pair | scene | ticks C/R | C N/E/F/D/U | R N/E/F/D/U | Δwait | Δelapsed | Δenergy |
|---:|---|---:|---|---|---:|---:|---:|
| 1 | 877101/9303101 | 722/694 | 22/2/0/0/0 | 23/1/0/0/0 | 252.53942549349722 | 282 | 324.9373252078476 |
| 2 | 877101/9303102 | 713/713 | 23/1/0/0/0 | 23/1/0/0/0 | 0 | 0 | 9.094947017729282E-13 |
| 3 | 877101/9303103 | 768/655 | 20/4/0/0/0 | 23/1/0/0/0 | 685.5275409802998 | 642 | 646.8438997043531 |
| 4 | 877101/9303104 | 698/725 | 21/3/0/0/0 | 23/1/0/0/0 | 260.5767619747626 | 302 | 339.0913124511708 |
| 5 | 877102/9303101 | 799/795 | 20/4/0/0/0 | 20/4/0/0/0 | 261.4727465143619 | 363 | 322.5319456248826 |
| 6 | 877102/9303102 | 801/791 | 18/6/0/0/0 | 19/5/0/0/0 | -202.24211122579618 | -251 | -247.63182639344632 |
| 7 | 877102/9303103 | 735/786 | 20/4/0/0/0 | 20/4/0/0/0 | -169.61955018254366 | -121 | -173.83825824859105 |
| 8 | 877102/9303104 | 809/824 | 22/2/0/0/0 | 19/5/0/0/0 | -465.464473863537 | -397 | -443.5336065149195 |
| 9 | 877103/9303101 | 711/668 | 20/4/0/0/0 | 20/4/0/0/0 | 367.11554965350433 | 351 | 397.4807547823866 |
| 10 | 877103/9303102 | 831/713 | 18/6/0/0/0 | 21/3/0/0/0 | 241.61708810662867 | 113 | 197.95249503380546 |
| 11 | 877103/9303103 | 702/770 | 21/3/0/0/0 | 20/4/0/0/0 | -428.18800897632354 | -440 | -399.874432841023 |
| 12 | 877103/9303104 | 751/797 | 19/5/0/0/0 | 19/5/0/0/0 | -478.6270830262347 | -570 | -532.4805304688753 |
| 13 | 877104/9303101 | 758/863 | 19/5/0/0/0 | 20/3/1/0/0 | -578.3562047072569 | -374 | -359.9870170506765 |
| 14 | 877104/9303102 | 742/742 | 19/5/0/0/0 | 19/5/0/0/0 | -246.58688277428746 | -274 | -276.1631260163076 |
| 15 | 877104/9303103 | 685/724 | 21/3/0/0/0 | 21/3/0/0/0 | -66.14638229183583 | -64 | -61.602128748916584 |
| 16 | 877104/9303104 | 738/863 | 22/2/0/0/0 | 16/7/1/0/0 | -688.2406166132155 | -511 | -584.9103552780334 |
| 17 | 877105/9303101 | 685/828 | 21/3/0/0/0 | 21/3/0/0/0 | -504.171103904288 | -497 | -542.7246012361056 |
| 18 | 877105/9303102 | 706/706 | 22/2/0/0/0 | 22/2/0/0/0 | 0 | 0 | -9.094947017729282E-13 |
| 19 | 877105/9303103 | 742/865 | 22/2/0/0/0 | 19/5/0/0/0 | -944.2728093343212 | -802 | -886.4363900023491 |
| 20 | 877105/9303104 | 818/776 | 22/2/0/0/0 | 22/2/0/0/0 | -148.2594080823701 | -223 | -245.13615725234376 |
| 21 | 877106/9303101 | 836/752 | 19/5/0/0/0 | 20/4/0/0/0 | -176.0746084480652 | -154 | -150.76281530593496 |
| 22 | 877106/9303102 | 748/748 | 20/4/0/0/0 | 20/4/0/0/0 | 0 | 0 | 9.094947017729282E-13 |
| 23 | 877106/9303103 | 759/755 | 19/5/0/0/0 | 20/4/0/0/0 | 476.3792891837884 | 485 | 469.0832526346667 |
| 24 | 877106/9303104 | 799/757 | 21/3/0/0/0 | 21/3/0/0/0 | 276.4580103747421 | 151 | 211.86047995051922 |

| profile | pair IDs | C normal | R normal | C wait (s) | R wait (s) | Δwait (s) |
|---:|---|---:|---:|---:|---:|---:|
| 877101 | 1,2,3,4 | 86 | 92 | 12260.308761386932 | 11061.665032938374 | 1198.6437284485582 |
| 877102 | 5,6,7,8 | 80 | 78 | 13760.424366494517 | 14336.277755252031 | -575.853388757514 |
| 877103 | 9,10,11,12 | 78 | 80 | 13243.684865547717 | 13541.767319790142 | -298.08245424242523 |
| 877104 | 13,14,15,16 | 81 | 76 | 12144.420913491927 | 13723.750999878524 | -1579.330086386597 |
| 877105 | 17,18,19,20 | 87 | 84 | 12382.405353223297 | 13979.108674544275 | -1596.703321320978 |
| 877106 | 21,22,23,24 | 79 | 81 | 13516.843074574452 | 12940.080383463985 | 576.7626911104671 |

聚合 control N/E/F/D/U=`491/85/0/0/0`，reference=`491/83/2/0/0`；generated/admitted/resolved 均为 `576/576/576`，completed 为 control/reference=`576/574`。wait 为 `77308.08733471885` 对 `79582.65016586734`，Δ=`-2274.5628311484907`；request elapsed 为 `221361` 对 `223350`，Δ=`-1989`；energy 为 `183268.3495482137` 对 `185263.6493281816`，Δ=`-1995.2997799678997`。raw reward（排除 rejection penalty）为 control `11371.746812675045`、reference `11251.448813678655`；rejection penalty 分别为 `-403.90000000000003` 和 `-426.05`。

root review criterion 为 development candidate PASS only：technical/source/model-RNG/request identity/ledger/energy/hard-safety checks 全通过，normal 不低于 reference、wait 不高于 reference、successful completed 不低于 reference，strict business gain 为 true。该结果不是 formal qualification、默认配置晋升、PPO 准入或训练 remedy；`PPO_ALLOWED=false`，未使用 heldout。六个 profile 中 `877101` 的 normal 为 `86/92` 且 wait 增加 `1198.6437284485582 s`，`877106` 的 normal 为 `79/81` 且 wait 增加 `576.7626911104671 s`，因此不记录为普遍收益。

balanced full 终态后 heartbeat `p3-balanced-proposal-support` 已删除，无 active background job。下一步由 root 选择 actual trained checkpoint validation；本轮不启动训练。

## 2026-10-01 P3 trained proposal-support pilot 终态（full running）

这是新的 frozen trained proposal-support development comparison，不是训练、qualification、default promotion 或 PPO。`preexisting_artifact` 包含原始 2-second/3-arm fixture 的 raw proof；该 fixture 的 evaluator classification 不再作为生产终态，新的 runner 只将 nonterminal exact-replay `IncompleteLifecycle` 保留为 lifecycle qualification metadata，并生成 safety-snapshot fallback summary。修正后的 3-arm contract 为 valid/censored，`generated=12`、`admitted=12`、`pending=12`，full metrics 保持 null；没有 physical rerun。

`newly_run` 代码 commit 为 `9b66049`，28 个定向测试和 compile 通过。server no-env preflight 为 `PREFLIGHT_OK`，proof SHA 前缀为 `0911e9...fe38`；范围是原始 4 cells（`875101..875104` / `875201..875204`），每 cell 3 arms，native stochastic epsilon=`0.10` only support。该 preflight 不作 perfect keyed-CRN claim；PPO、teacher、heldout 和 default promotion 均关闭。

pilot 服务器 PID `2112900` 已自然退出；server pilot SHA=`b9133fdbe1c94b7e2a2d5225c052c103339bb55ca2d59d6df3e885a871a55cda`，与 preflight SHA `0911e9...fe38` 不同。3 arms 均 natural，0 censor/error，safety、source/model、caller-RNG、24-request identity、replay、ledger 和 energy checks 全通过。pilot cell 1 的 observed metrics 如下：

| arm | N/E/F | wait (s) | energy | raw reward excl. rejection penalty | rejection penalty |
|---|---|---:|---:|---:|---:|
| before_legacy | 19/5/0 | 3327.4996213135246 | 7757.917108755452 | 451.32887937331253 | -19.55 |
| after_legacy | 18/6/0 | 3046.1997149195568 | 7457.839771374826 | 450.31157694902555 | -20.950000000000003 |
| after_support | 18/6/0 | 3749.379758593257 | 8176.11847746592 | 434.348464181481 | -24.700000000000003 |

该 pilot 首 cell 的 before normal=`19`、after legacy/support normal=`18`，与历史首 cell 计数一致；support arm 的首 cell wait 为 `3749.379758593257 s`。这些是该 pilot cell 的 observed metrics，不形成 global acceptance 或 root-cause 结论。root 已授权 fixed full：复用 pilot cell 1，加上 3 cells/9 arms。以下启动记录属于历史状态：PID `2113754`，cwd `/home/user/projects/terminal`，module phase 为 `full`，server env/executable checks 通过；首个新 arm `cell_02_before_legacy.json` 已生成，大小 `71978` bytes，server mtime ns=`1790799508572507832`。full 随后已完成，终态见下方；启动时的 heartbeat `p3-trained-proposal-support` 已在终态后删除。

## 2026-10-01 P3 trained proposal-support full 终态（development candidate rejected）

trained proposal-support full 已完成。服务器 PID `2113754` 已退出，full mtime ns=`1790799958139098485`；full artifact SHA=`3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919`，independent proof `.tmp/p3_trained_support_full_proof.json` SHA=`6e3cd91c472cdaf522020f39d7f8180e2802dfd035332187c21b4204a79f15b8`。4 cells × 3 arms 全部 natural、0 censor/error、hard-safety clear；每个 arm generated/admitted/resolved=`96/96/96`，source/model/caller-RNG、24-request identity、replay、ledger 和 energy checks 全通过。

| cell | env | arm | N/E/F/D/U | simulation s | request elapsed s | wait s | energy | raw reward excl penalty | rejection penalty |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 875101 | before_legacy | 19/5/0/0/0 | 761 | 9050 | 3327.4996213135246 | 7757.917108755452 | 451.32887937331253 | -19.55 |
| 1 | 875101 | after_legacy | 18/6/0/0/0 | 689 | 8737 | 3046.1997149195568 | 7457.839771374826 | 450.31157694902555 | -20.950000000000003 |
| 1 | 875101 | after_support | 18/6/0/0/0 | 755 | 9504 | 3749.379758593257 | 8176.11847746592 | 434.348464181481 | -24.700000000000003 |
| 2 | 875102 | before_legacy | 20/4/0/0/0 | 726 | 9133 | 3234.6369068496506 | 7677.921091787695 | 465.58665490460396 | -16.75 |
| 2 | 875102 | after_legacy | 18/6/0/0/0 | 727 | 9232 | 3292.5856932730585 | 7750.015446572095 | 443.8305713428898 | -20.650000000000002 |
| 2 | 875102 | after_support | 20/4/0/0/0 | 726 | 9159 | 3189.316530514645 | 7679.415958554219 | 463.8215531471326 | -19.450000000000003 |
| 3 | 875103 | before_legacy | 22/2/0/0/0 | 707 | 8369 | 2667.140927166041 | 7098.867749409129 | 497.63654049060835 | -14.3 |
| 3 | 875103 | after_legacy | 21/3/0/0/0 | 776 | 9194 | 3521.746147520452 | 7950.49929366206 | 471.0605408998054 | -17.55 |
| 3 | 875103 | after_support | 20/4/0/0/0 | 833 | 9186 | 3573.5243270983137 | 8005.473731383905 | 462.88896723234376 | -15.350000000000001 |
| 4 | 875104 | before_legacy | 19/5/0/0/0 | 720 | 8712 | 2731.326453726862 | 7034.847039351723 | 460.9677443897178 | -14.4 |
| 4 | 875104 | after_legacy | 17/7/0/0/0 | 726 | 8795 | 2833.9475529832466 | 7139.991726676687 | 442.2643044437435 | -15.200000000000001 |
| 4 | 875104 | after_support | 17/7/0/0/0 | 757 | 9443 | 3345.4609609007143 | 7798.724737635735 | 438.3307463573676 | -14.100000000000001 |

聚合 before_legacy/after_legacy/after_support 分别为 normal=`80/74/75`、emergency=`16/22/21`、failed/diverted/unresolved=`0/0/0`。wait 分别为 `11960.603909056077`、`12694.479108696314`、`13857.68157710693` s；request elapsed 分别为 `35264`、`35958`、`37292` s；energy 分别为 `29569.552989304`、`30298.346238285667`、`31659.73290503978`。raw reward（排除 rejection penalty）分别为 `1875.5198191582426`、`1807.4669936354644`、`1799.3897309183249`；rejection penalty 分别为 `-65.0`、`-74.35000000000001`、`-73.60000000000001`。

before→after_legacy 的 observed delta 为 normal `-6`、wait `+733.875199640237 s`、request elapsed `+694 s`、energy `+728.7932489816667`。after_support→after_legacy 的 observed delta 为 normal `+1`，但 wait `+1163.2024684106163 s`（`+9.16305788091587%`）、request elapsed `+1334 s`、energy `+1361.3866667541115`（`+4.493270543703249%`）。按既有 aggregate criterion，trained fixed-epsilon support 被拒绝为 development candidate；不进行 epsilon retune 或 default promotion。

证据边界：该 full reproduces the four-cell trained regression；proposal encoder/head、gate head、temperature bitwise frozen，允许变化范围仅为 gate adapter/critic，因此不把本结果写成 newly trained proposal-head collapse。coverage 结果不足以单独建立 fix；本轮不作 team-credit、entropy、clipping root-cause 或 P3 solved 结论。原 balanced24 的 `491/85/0` 对 `491/83/2` 结果仍是 separate preexisting cohort/source/sampler，不能与本 full 合并。

full heartbeat `p3-trained-proposal-support` 已删除，无 active job。随后 root 已授权下方 matched-state gate-shift diagnostic；该诊断仍未训练，`PPO_ALLOWED=false`，teacher/optimizer/heldout 均未使用。

## 2026-10-01 P3 matched-state gate-shift diagnostic（pilot 与 full 终态）

这是当前代码 `6f4acd0` 下的 matched-state gate-shift 只读诊断，不是训练、PPO、optimizer、Teacher、held-out 或 qualification。定向测试为 `13` 项；pilot 复用固定 cell 1，full 仅复用该 cell 加上其余 3 条既有 tape，不生成新的 policy trajectory。

pilot 服务器 PID `2117546` 已自然退出，状态为 `PILOT_OK`。cell 1 既有轨迹精确回放 `689` 步；同状态分别计算 before/after 分布；eligible actor rows=`2655`，全部 observation rows=`7175`，proposal equal=`0`；raw team reward=`429.36157694902556`。NPZ SHA256=`b3121d0019233e97450ec6899d35768694acaf56e3d3c625e0062e1dcac0f424`。重复 state-row 的描述性均值为：inner 的 `p=.3382572144344204→.32877570660476363`、Δ=`-.009481507829656819`、KL=`.0014337947496490945`；middle Δ=`+.0010723269267593717`、KL=`.00019261295864622603`；outer Δ=`+.0001329123`、KL=`.0003650273`。这些是重复 state-row 的描述性统计，不是独立试验或 causal return。

root 已复核 fail-fast LP validation（代码行 `625–634`）会执行 eligible gates 和 RELEASE proposals；eligible gate/RELEASE proposal logprob 逐行 fail-fast 核验已执行，结果 JSON 未单独序列化 LP 核验计数。full 已完成，full JSON SHA256=`237bc7393e0dc831204b59aaf29aa99ea55dc142ce7ab3cc9fbc4effe8c4ec3e`，PID `2118294` 已退出且无 error，heartbeat `p3-matched-gate-shift` 已删除。ALL4 的 steps=`689/727/776/726`（合计 `2918`），native output reward totals=`429.36157694902556/423.18057134288983/453.5105408998054/427.06430444374354`；model before/after、caller RNG、observer RNG 均 unchanged，instance-step hook restored，extra steps=`0`，policy sampling/optimizer 均为 false。每个 cell 的 proposal q/logit equality max abs 为 `0`，argmax proposal difference 也为 `0`。

| cell | steps | matched / eligible rows | raw team reward | inner Δp / KL | middle Δp / KL | outer Δp / KL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 689 | 7175 / 2655 | 429.36157694902556 | -0.009481507829656819 / 0.0014337947496490945 | +0.0010723269267593717 / 0.00019261295864622603 | +0.00013291226652663724 / 0.00036502730924742956 |
| 2 | 727 | 7535 / 2920 | 423.18057134288983 | -0.014564387732508299 / 0.002288819788640633 | +0.0015276760721562412 / 0.0003872494991814025 | +0.00021801173142277483 / 0.0005416399008456102 |
| 3 | 776 | 7764 / 3175 | 453.5105408998054 | -0.01141738188580179 / 0.001978376961868185 | +0.0007357496051711796 / 0.00019696224443501164 | +0.0019944499217231073 / 0.0005377803990094331 |
| 4 | 726 | 7117 / 2670 | 427.06430444374354 | -0.007014479502420132 / 0.0009183402729311774 | +0.0007393442887577024 / 0.00016364977011383615 | +0.0009027146738585873 / 0.0003357394796345426 |

`actual_outcomes` counts are repeated observation rows, not unique flights: cell 1 `4681 completed / 2494 emergency_landed`, cell 2 `4701/2834`, cell 3 `6443/1321`, cell 4 `4010/3107`. Inner eligible mean Δp is negative in all four tapes; middle and outer changes are small positive values in these descriptive rows. These statistics are not independent trials, action credit `D`, full value, or a global gain; no outcome-selected state claim is made.

## Trained gate-value matched-state diagnostic：V1、V2 pilot 终态与 FULL 运行

协议已写入 [trained gate-value protocol](evidence/v4_2/p3_trained_gate_value_protocol_20261001.md)。代码先后为 `d849b2` 与 serialization fix `c5d3d900970dc036a05fb2dd8f4b1cd552cc0564`；18 个 focused tests、compile/diffcheck 和 local no-env preflight 通过。估计器固定四个 states、policy seeds=`9303201..04`、cap=`1800`，使用 common native K=3 q，并沿 `arrival_id|tick|stream` 复用同一 frozen BEFORE `3fe7` continuation。C1/C2/C3/C4 的 target-only exact Δp 与 full `D` 均已取得；local `m` 不求和为 global gain。counterfactual suffix 是新的模拟轨迹，不能表述为“没有 simulation”。

server V1 preflight SHA=`fc05db44044615c329cfa34147c49df4e5cf50630002f137427d3c95c505ccc2`，状态为 `PREFLIGHT_OK`、no-env；wrapper malformed invocation 在 AFTER module 后返回 `1`，module exit unknown。V1 pilot 的真实 PID=`2121108`、argv/cwd 已核对，但在 environment/branch 之前因 `matched/native full binding changed` 退出；`pilot_v1.log` 为 `115` bytes，没有 pilot/error/branch outputs，preflight progress 保持不变。root 已核对 cause JSON 中保存的 string paths 与 runtime paths 的 raw comparison；inputs SHA checks 未变化。serialization fix 统一 canonical paths/SHA，并保留 prevalidation error artifact；V1 failed output root 保留为 `/home/user/training_outputs/terminal/p3_followup_20260930/trained_gate_value_20261001`。

V2 output root 为 `/home/user/training_outputs/terminal/p3_followup_20260930/trained_gate_value_20261001_v2`；server V2 no-env preflight proof SHA=`9dd8b463d61086c0c76e02db582b6a3b105f2bb4c4352b4da95358493eb05a0f`，`PREFLIGHT_OK exit0`；actual readJSON→`_validate_preflight` phase validation=`PHASE_VALIDATION_OK exit0`，no-env/replay/model-update checks 通过，stored source/runtime/runner/protocol bindings 与四个 locked-state bindings 匹配。V2 pilot `PID=2123288` 已 `PILOT_OK`：`104` branches natural、0 censor/error/hardsafety；C1 为 step `168`、UID `3`/arrival `3`，four-seed full `D` 为 `[-0.04999999999989768,-0.049999999999954525,-0.04999999999989768,-0.049999999999954525]`，mean=`-0.049999999999926104`，helper mean=`-0.04999999999989768`，C=`0`，4×25 RELEASE branches 均 safety-reject。gate p 从 `3.072129038628191e-5` 到 `2.416560164419934e-5`，Δ=`-6.555688742082566e-6`；local m=`+3.2778443710364384e-7`，只作为 local 值，不求和为 global。该状态的 p 下降没有显示有害证据，不能据单一状态归因。104 alternate-world outcome counts `N=2002/E=494/F=0/D=0/U=0` 是分支总计，不是 `2496` 个独立人口绩效；source/model/environment/RNG、外生 CRN、non-target first action 和 replay `689` 检查均通过。pilot proof SHA=`e213fbaedd2d700736bd70b2b1787ed103914e21b356ca62b9708642f512ac77`，state SHA=`0a51d689f575bcee9e2a64fdd440ce1a339bd3cb76f08f02bf14e53a5620d7d6`。独立 readonly business 核验中，four-seed q-weighted RELEASE 与 WAIT 的 N/E/F/D/U、elapsed、energy 相同（abs noise≤`1.82e-12`），wait 两臂均为 null；首步 WAIT reward=`-0.3106620065699539`，全部 RELEASE reward=`-0.36066200656995395`，complete D 差与首步 reward 差一致。first-step rejection-penalty 分量未单独暴露，保持 unknown；local m 不代表业务改善。V1 artifact 的 stage string 保持 method stage 语义，不是 V2 folder mismatch。

C2 state proof `state_02.json` SHA=`03d5ee476b286b65f149475f2be2c110e72a60566fb5c3a32b668abd2b6cd375`（step `202`、UID `2`/arrival `2`）显示：`C=0`，四个 repeats 的 25 个 RELEASE branches 全为 `safety_reject`；full `D` 为 `[-8.109680200730281,-22.35170231001115,-8.058118334540154,39.60471616313083]`，mean=`+0.2713038294623118`，helper=`+0.2713038294623402`；`p=.9102736115455627→.9061629772186279`，Δp=`-.0041106343269348145`，local m=`-.0011152308344166478`，linear residual=`-.00014419527724385262`。三个 seed 的 D 为负、一个为正且均值很小，不能据此稳定归因，也不因 local m 为负宣布 p 下降为根因；C=`0` 不证明 full-cycle 无效。alternate-world outcome counts 为 `N=1898/E=598/F=0/D=0/U=0`，是分支总计而非独立人口或绩效。四个 continuation seed 的 WAIT/q-weighted RELEASE 实际值如下；wait_seconds 均为 `null`，未知 rejection-penalty 分量不填。独立按 `group.paired.proposal_q` 与各 seed 的 `release_returns` 逐项重算 full `D`，结果为 `-8.109680200730224/-22.351702310011092/-8.058118334540154/+39.60471616313089`，与记录最大绝对差≤`5.7e-14`，PASS。

| continuation seed | WAIT N/E/F/D/U | RELEASE N/E/F/D/U | WAIT elapsed | RELEASE elapsed | WAIT energy | RELEASE energy | WAIT return | RELEASE return |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 9303201 | 20/4/0/0/0 | 19/5/0/0/0 | 5752 | 5763 | 7581.128143600825 | 7596.189746597886 | 333.64271582118243 | 325.53303562045215 |
| 9303202 | 17/7/0/0/0 | 16/8/0/0/0 | 5753 | 5935 | 7877.507344213232 | 8166.286571543861 | 299.0213197882069 | 276.6696174781958 |
| 9303203 | 20/4/0/0/0 | 19/5/0/0/0 | 5669 | 5708 | 7550.753971850365 | 7580.31106287849 | 336.5217532135773 | 328.46363487903716 |
| 9303204 | 16/8/0/0/0 | 19/5/0/0/0 | 6175 | 5963 | 8188.862611350151 | 7905.125660514603 | 279.0753708983965 | 318.68008706152733 |

C3 state proof `state_03.json` SHA=`0a84549c5313d0149aef845ceb1636f0c73c78896730bba3f1913074bdf54699`（step `198`、UID `2`/arrival `2`、SOC=`0.49960816800385516`）显示：`104` branches natural、invalid=`0`、censor/error/hardsafety=`0`；`C=0`，四个 repeats 的 25 个 RELEASE branches 全为 `safety_reject`；full `D` 为 `[-0.049999999999954525,-0.04999999999989768,-0.049999999999954525,-0.05000000000001137]`，mean=`-0.049999999999954525`，helper=`-0.04999999999989768`；`p=.0030303376261144876→.002411351539194584`，Δp=`-.0006189860869199038`，local m=`+.00003094930434596704`，linear residual=`+.00007521093357354403`。source/model/env/RNG、外生 CRN、non-target first action 和 exact replay `776` 检查均通过。local m 不代表 business gain 或 global gain；该单状态不支持把 p 下降归因于有害机制。alternate-world outcome counts 为 `N=2210/E=286/F=0/D=0/U=0`，是分支总计而非独立人口或绩效。四个 continuation seed 的 WAIT/real-q RELEASE 实际值如下；wait_seconds 均为 `null`，未读取的首步 reward 与 rejection-penalty 分量不填。

| continuation seed | WAIT N/E/F/D/U | RELEASE N/E/F/D/U | WAIT elapsed | RELEASE elapsed | WAIT energy | RELEASE energy | WAIT return | RELEASE return |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 9303201 | 21/3/0/0/0 | 21/3/0/0/0 | 5995 | 5995.000000000001 | 7990.542702321978 | 7990.542702321979 | 342.07787631805616 | 342.02787631805614 |
| 9303202 | 22/2/0/0/0 | 22/2/0/0/0 | 5822 | 5822 | 7737.156762648392 | 7737.156762648393 | 355.72659281137635 | 355.6765928113764 |
| 9303203 | 20/4/0/0/0 | 20/4/0/0/0 | 6005 | 6005.000000000001 | 8015.881079010222 | 8015.8810790102225 | 334.21445937453444 | 334.16445937453443 |
| 9303204 | 22/2/0/0/0 | 22/2/0/0/0 | 6341 | 6341.000000000001 | 8292.364322081408 | 8292.364322081408 | 344.0289732407352 | 343.9789732407352 |

FULL terminal evidence：server `08:25:35`（Asia/Shanghai）PID=`2127202` 已退出，`FULL_OK`；exact module/cwd/envflags 匹配，pilot immutable reused，`4` mechanism states、`416` branches 全部 natural，invalid/censor/error/hardsafety 均为 `0`。source/model/environment/caller RNG、外生 CRN、non-target first actions、exact replay 与 model/env invariants 全部通过；artifact proof 为 `.tmp/p3_trained_gate_value_v2_full_proof.json`，full SHA=`056bbc9d4f4949c7c932b1b21c49204b2533d1fe1f7b1fe44b48a2e845ee5a5a`。该运行是 mechanism-state diagnostic，不是 performance population、qualification 或 PPO 结果；alternate-world outcome counts 不视为独立人口绩效。

| state | C | per-seed D | mean D | Δp | local m |
|---|---:|---|---:|---:|---:|
| C1 / step168 | 0 | `[-0.04999999999989768,-0.049999999999954525,-0.04999999999989768,-0.049999999999954525]` | `-0.049999999999926104` | `-6.555688742082566e-6` | `+3.2778443710364384e-7` |
| C2 / step202 | 0 | `[-8.109680200730281,-22.35170231001115,-8.058118334540154,39.60471616313083]` | `+0.2713038294623118` | `-0.0041106343269348145` | `-0.0011152308344166478` |
| C3 / step198 | 0 | `[-0.049999999999954525,-0.04999999999989768,-0.049999999999954525,-0.05000000000001137]` | `-0.049999999999954525` | `-0.0006189860869199038` | `+0.00003094930434596704` |
| C4 / step187 | 1 | `[5.684341886080802e-14,-4.864230616753275,5.684341886080802e-14,0.0]` | `-1.2160576541882904` | `+5.9545040130615234e-5` | `-7.241020181978357e-5` |

C4 state proof `state_04.json` SHA=`b1ca9eba9d05dde6487f941e4d503737d5530176a307ec309e23d45d6cdae6e6`（step `187`、UID `3`/arrival `3`）为 `C=1`，四 repeats 各 `25` 个 RELEASE branches 均 `approved`；104 branches natural、invalid/censor/error/hardsafety=`0`。C4 的 wait_seconds 四 seed 均为 `null`。其中 seed `9303202` WAIT/RELEASE outcome 均 `20/4/0/0/0`，elapsed=`5410→5670`，energy=`6737.448645086157→6975.869534481303`，D=`-4.864230616753275`；其余 seed 的 D 为 `5.684341886080802e-14/5.684341886080802e-14/0.0`，对应 return 差为数值近零。不要把 local m 求和为 global gain。

ALL4 结果不支持 universal increased RELEASE，也不支持把训练退化归因于 causal mechanism；entropy、team-credit、gradient clipping 等仍未由本诊断证明。只读运行契约显示：`parallel_env.py:880–883` 的 safety reject 会设置 `safety_retry_after`，`student_rollout_v4_2.py:111–153` 的 WAIT/not_requested 不设置，因此即时 `C=0` 不意味着 future trajectory 相同。实际 cfg 为 `v4_2_rejected_release_hold_mode=canonical_wait`、`emergency_priority_guard_enabled=true`、`rule_flow_coordination_enabled=false`；本运行不按 `legacy_requested` 解释，也未修改 retry 规则。

ROOT 选择的下一项仍是 experimental candidate 的 implementation/test：保持 original sampled-first K3 的 prefix，再仅在需要时加入 remaining raw-source-logit ranked-25；gate/q/first3/checkpoint/safety/retry 保持不变。选择依据是 source16 step445 中 proposals22/23 的 q-starved 但 complete G-WAIT 对四 seed 为正的局部 witness；single failure witness 不作为总体 root cause。该 candidate 尚未 launch/admit 为 performance experiment，也未进入 default/PPO；`PPO_ALLOWED=false`。
文档 erratum：现有 protocol 文本仍残留句子 “keyed by UID and absolute tick”。实际冻结源码的 scoped hook 按 `arrival_id|absolute_tick|stream` 续跑，action-map 才使用 UID；该实现与 source SHA 前缀 `6e06f837` 及 preflight 绑定一致。本轮只记录该文字差异，不修改 protocol 文件或其 digest。

V2 固定四条既有 tape 中 first actor-eligible INNER、`0.4 < SOC <= 0.5`、retry-free、native-order 的 states：C1 step `168` UID `3`/arrival `3`，C2 step `202` UID `2`/arrival `2`，C3 step `198` UID `2`/arrival `2`，C4 step `187` UID `3`/arrival `3`。这些 states 按 eligibility 选择，不按 outcome 或 p-delta 选择，不替换原有 24-state 结果，也不重采样；exact prestate 恢复、其余 first requests 保持 native AFTER tape，首步比较 WAIT 与全部 25 个 RELEASE actual-q/native-K3 branches，full `D` 包含首步 reward。gate probability 的下降或上升仍需 `D` 验证；当前不作 root-cause 或 solution 结论，`PPO_ALLOWED=false`。

## 2026-09-30 研究复核与修复更新

本轮完成“评估研究与训练”对话中 17 项不足的逐项复核，并完成已确认实现缺陷的代码修复与文档同步；新增实验性 Observation V2.4、`student_decision_mask_mode=pre_action_forced_wait` 和高层计量修正。相关实现回归、修正后的 paired helper 与 real-artifact smoke 已通过；distinct pytest cases 按 `154 = 126 + 18 + 1 + 9` 记录，重跑不重复计数。real-artifact smoke 为 `COMPLETED exit0`，因 `censored=true` 未产生 `qualified_delta_Q`，不形成 causal/performance conclusion；详见 [研究复核与修复登记](evidence/v4_2/research_review_repairs_20260930.md)。方法性贡献、qualification 与 P3 性能退化仍未证解决；`PPO_ALLOWED=false`，正式 P3 继续使用 88D，新 schema 需独立 qualification，heldout 未消费。

> **状态日期**：2026-09-29
>
> **本轮用户范围修订（当前，2026-09-28）：** 当前主线是 dynamic terminal airspace、multi-UAV
> landing-flow coordination 和 safety-filtered executable intents。低层新主课程采用
> `task_priority_enabled=false`：目标为正常降落、吞吐、完整周期等待和等待尾部 starvation
> 监测，保留随机到达、SOC 差异、能量/安全约束、ring/path feasibility 和 resource reservation。
> ArrivalProcess 仍消耗历史 base 抽样的 RNG，但只存储有效 `base_priority=1`；raw `mission`
> 保留为物理服务/兼容性 metadata，policy mission one-hot 固定 `STANDARD`，88D/mission/base
> 接口语义兼容。service time `30/120/10/60 s` 保留以维持负荷。能量应急 takeover 始终保留，
> 与后续独立 emergency mission 鲁棒性场景分开；raw `mission=emergency` 不表示本课程已启用
> mission-level emergency preference。
>
> `check_only_v2` 已 `CHECK_ONLY_OK exit0`：6 个 cells、84 个 requests 的 arrival time、gate、SOC、
> raw mission、service 等物理属性与 legacy 同 seed 一致，task-neutral effective base 全为 `1`；
> changed fields 仅为 `task_priority_enabled: true→false`、`low_level_priority_time_weight: 1→0`，
> `low_level_soc_urgency_time_weight=2` 不变；该检查为 zero env steps/records。代码 8 files 已提交 `91b1dd4`；新测试 `19 passed`
>（core `10`、evaluation `9`），直接 core regression 为 `100 passed / 1 skipped`，phase 检查为
> `35 passed / 2 failed`，两项失败在干净旧 HEAD `8a4e1b2` 上精确重现。首轮 server check-only
> wrapper 在 `ROOT/check_only/check_only.log` 预创建 leaf，导致 fresh output directory refusal
>（exit `2`）；未进入程序预检，也未执行 env reset/step；该失败按 wrapper 问题保留。smoke 仅作
> 启动/输出链路检查，首个 arrival 为 `10 s`、actor actions 为 `0`，不据此验证真实 actor 动作
>（88D/reward 已有单测）。full-cycle report 已由主审核对：`COMPLETED`、`12/12` paired records、
> `technical_ok=true`、server `full.exit=0`，两臂各 `84` requests；no Teacher/update/heldout，
> 两臂 `collisions`、`hard_separation`、`NMAC`、`SafetyInvariantError`、`uncertified_execution`
> 均为 `0`，weights/model unchanged 为 `true`。完整指标见 [task-neutral baseline result](evidence/v4_2/p3_task_neutral_baseline_result_20260928.md)。
> report overall 为 source `72 normal / 11 emergency / 1 failed_energy / 0 unresolved`、candidate
> `72 / 12 / 0 / 0`；mean wait `111.47949744714144→96.37119998288188 s`，p95
> `428.211635460218→329.0057350762542 s`，max `804.6897032491129→503.6264147256603 s`，
> low-SOC wait `148.3647493634526→119.71345607558123 s`，new reward
> `1530.4372197172681→1623.028105994644`。其余5个 cells 指标相同，唯一变化为
> `dense24/873106`：source `14/9/1` 到 candidate `14/10/0`（normal/emergency/failed_energy），
> mean wait `238.71178555903364→185.8327444341252 s`。
> 唯一 source failed-energy 为 arrival `4`：`standard`、`base_priority=1`、initial SOC `.60258461984`
> 到 `0`，在 `ring_inner` 因 `energy_depleted` 终止，wait `804.6897 s`；candidate 同 arrival 为
> `emergency_landed`，touchdown `659`、final SOC `.1546030`、wait `503.6264 s`。其它 outcome 变化为
> arrival `2/13` emergency→normal、`5/18` normal→emergency；normal 总数不增，candidate 减少1个
> energy failure 但仍有 `12/84` emergency outcomes。
> heartbeat `task-neutral-landing-baseline` 已删除。
> 执行顺序为：最小实现/单测 → 同课程 source/candidate 配对 baseline →
> 有界 PPO repair pilot → independent validation → static qualification → dynamic airspace。
> 等待尾部只作 starvation 监测，不增加每个子组无退化硬 gate；新 reward 与旧课程 reward 不
> 直接比较。source `03c0` 与已知 candidate 只作初始化/对照，candidate_retained、稳定收益、
> formal/heldout 和 `PPO_ALLOWED` 均不因本次文档修订自动改变。full 结果中 candidate 仅作为后续
> task-neutral 开发修复起点，source `03c0` 保留对照；旧 `candidate_retained=false` 与历史 outputs
> 不改写。协议见
> [task-neutral baseline protocol](evidence/v4_2/p3_task_neutral_baseline_protocol_20260928.md)。

> **上一阶段 joint-batch full 终态（2026-09-28）：** 代码提交 `c38071f` 对应的服务器
> check-only 已完成，`status=CHECK_ONLY_OK`、`exit=0`；相关 tests 为 `28`、joint-batch tests
> 为 `6`；`training_rollouts=0`、`ppo_updates=0`、`evaluation_rollouts=0`。training 的六个 dense cells 各为 `1800` horizon，
> 合计 `10800`；固定 evaluation 每臂为四个 dense cells 各 `1800`、flow6 `1210`、flow12
> `1570`，合计每臂 `9980`、两臂 `19960`。主审已核对 report，core `exit=0`，已批准唯一
> full。full PID 为 `1132703`；cmdline 模块 `run_p3_task_neutral_joint_batch` 与 cwd 已核对。
> full report 为 `COMPLETED`、`technical_ok=true`、`full.exit=0`；六条 training rollouts 实际合计
> `4635` steps，`1` 次 PPO update，`290` accepted、`0` rejected，KL mean/final/max 为
> `.00053613/.00039126/.00407790`，q、freeze、roundtrip 检查正常。`joint_batch.pt` SHA256 为
> `C961512CD5EDD5A4AE191A8F29D9D799AB971B66C8F44E53C333889F10D6EA0E`。配对 evaluation 共
> `12` rollouts、两臂各 `114` requests，安全计数为 `0`；before `98/16/0`、after `97/17/0`
> （normal/emergency/failed），mean/p95/max wait `107.7779/273.7358/361.1838→113.1974/306.1524/414.6886 s`，
> low-SOC wait `98.6126→91.7405`，reward `2202.5621→2178.6727`，normal deltas
> `[1,0,0,-2,0,0]`。heartbeat `ppo` 已删除；该阶段随后进行了只读核对奖励/折扣时间跨度与学习信号；
> 后续当前授权见
> [full-episode credit protocol](evidence/v4_2/p3_task_neutral_full_episode_credit_protocol_20260928.md)。
> 协议见
> [joint-batch protocol](evidence/v4_2/p3_task_neutral_joint_batch_protocol_20260928.md)。

> **历史阶段：full-episode credit repair protocol 终态（2026-09-28）：** 输出根为
> `/home/user/training_outputs/terminal/p3_task_neutral_full_episode_credit_20260928`。恢复
> `3fe7` actor 与 `b136` critic，作为同一 task-neutral baseline control，不从 `c961` 继续；
> 复用 joint-batch 六个已消费 development training cells，完整 terminal、收集期间不更新，
> `gamma=1`、`lambda=1`，return 为 episode 实际后缀 reward 和，不跨 episode bootstrap。
> 其余 lr/epochs/minibatch/clip/value/entropy/target-KL、q 与 KL reference 保持原值，只训练
> gate_adapter+critic，执行一次 joint update。代码提交 `f8a8b53`，定向 tests `45`（joint `16`）；
> check-only 为 `CHECK_ONLY_OK exit0`，验证 reuse6、new6 evaluation 上限 `9980`、training `10800`，
> `env/update/runtime=0`。首次直接执行不可执行 launcher 因权限不足未启动且无产物，随后绝对 Python
> launcher 成功，非实验重跑。唯一 full child PID `1530563`，cwd `/home/user/projects/terminal`，
> cmdline 含 `--full-episode-credit`；最终 `exit=0`、report 为 `COMPLETED`，六个 training cells
> 合计 `4635` steps，执行 `1` 次 update，`290` accepted/`0` rejected，KL mean/final/max 为
> `.0004373/.0005048/.0014622`，MC match 为 `true`、最大误差 `3.0517578125e-05`，q/freeze/
> roundtrip 通过；full checkpoint SHA256 为
> `13a98ce2b6289126a5831f6cda5e26e857e1a49d33ab3e606edb5051542f191e`。heartbeat `automation` 已删除。
> 配对 evaluation 为六条复用 before 与六条新 after，两臂各 `114` requests，安全与 unresolved 均为
> `0`：before `98/16/0`、after `93/21/0`（normal/emergency/failed），mean/p95/max wait
> `107.778/273.736/361.184→118.833/307.999/377.930 s`，low-SOC wait `98.613→111.458 s`，
> reward `2202.562→2121.598`，normal deltas `[-1,-1,-1,-2,0,0]`。主审不保留该 checkpoint，
> 不把 `gamma=1`、`lambda=1` 视为既定修复，继续以 `3fe7` actor 为开发起点。时间跨度依据与完整边界见
> [full-episode credit protocol](evidence/v4_2/p3_task_neutral_full_episode_credit_protocol_20260928.md)。
> 当前结论仍为退化未解决；reward、安全、proposal、task-neutral、heldout `829xxx` 和
> `PPO_ALLOWED=false` 边界不变。下一步仅做本地可枚举 toy 验证，检查 actor 是否按每 world 人数取均值会改变
> team reward 梯度方向，未授权下一次环境训练。

> **历史阶段：joint-team actor aggregation protocol 终态（2026-09-29）：** 输出根为
> `/home/user/training_outputs/terminal/p3_task_neutral_joint_team_actor_20260928`。恢复 `3fe7` actor
> 与 `b136` critic，复用 joint-batch 六个 task-neutral development training cells，预算为每 cell
> `24` requests、六个 cells 合计 `144` requests、最多 `10800` steps，原 `gamma=.99`、`lambda=.95`，
> 一次 joint PPO update；
> 只训练 gate_adapter+critic，其余 proposal/actor_encoder/gate_head/temperature 冻结，q 与 KL
> reference 保持原值。六条完整 episode 合并为一个 update batch，所有 minibatch 共用整批唯一常数
> `C=totalactor/T`（`T` 为六条 episode 的全部 steps 总数，禁止按 episode 分别重归一），actor loss 为
> `sum(all actor clipped losses)/(minibatch world_count*C)`，`C=0` 时 actor loss 为 `0`；critic、
> advantage normalization、entropy、KL 与 per-agent clipped ratio 定义不变。已有 toy 证据显示
> objective weighting 方向不一致，但不证明实际退化的全部根因。代码提交
> `fb1a9a61d0dbb524a60928f598a0ad847297a080`，定向 tests `57`；server check-only 为
> `CHECK_ONLY_OK exit0`，源码与两个 tests 的四 SHA 已同步核对，joint-team、`gamma=.99`、
> `lambda=.95`、reuse6、new6 evaluation 上限 `9980`、training 上限 `10800`，
> `env/update/runtime=0`。唯一 full launcher PID `1798690`、child PID `1798697`，cwd
> `/home/user/projects/terminal`，cmdline 含 `--joint-team-actor`；首个 rollout 曾为 seed `873103`、
> `375` steps、`10` completed、`actor=2120`、`terminal=false`。最终 `exit=0`、report 为
> `COMPLETED`、`technical_ok=true`，六个 training cells 合计 `4635` steps、`1` 次 update、
> `290` accepted/`0` rejected，整批 `C=3.844012944983819`，KL mean/final/max 为
> `.0004372743/.0001949033/.0026966119`，q/freeze/roundtrip 通过；full checkpoint SHA256 为
> `ac7fbb5b3b506e693c6a3073c99b85cc3c0d395644755a7496cff8d48c5fd05a`。配对 evaluation 两臂各
> `114` requests，安全/unresolved 均为 `0`：before `98/16/0`、after `97/17/0`
> （normal/emergency/failed），mean/p95/max wait `107.778/273.736/361.184→107.613/262.942/393.551 s`，
> low-SOC wait `98.613→100.942 s`，reward `2202.562→2193.303`，normal deltas
> `[+1,0,0,-2,0,0]`。heartbeat `automation` 已删除；主审保留 actor aggregation 数学修正，
> 但 `ac7` 仅待重复验证、不晋级，不宣称退化已解决。详情见
> [joint-team actor aggregation protocol](evidence/v4_2/p3_task_neutral_joint_team_actor_protocol_20260928.md)。
> 六条 training cells 仅为 development 证据而非 fresh cohort；问题仍未解决。

> **历史阶段：joint-team fixed-model sampling repeats 终态（2026-09-29）：** 输出根为
> `/home/user/training_outputs/terminal/p3_task_neutral_joint_team_repeats_20260929`。before 固定
> `3fe7` actor，after 固定 `ac7fbb5b3b506e693c6a3073c99b85cc3c0d395644755a7496cff8d48c5fd05a`，
> 两臂不训练、不使用 optimizer/Teacher。四个 dense24 env seeds `875101..875104`，每个三条
> policy sampling；每臂 `12` records/`288` requests，合计两臂 `24` records，其中每臂每场景一条
> 复用、两条新 sampling；新 sampling 两臂共 `16` rollouts、总步数上限 `28800`。原 joint-team
> 单 seed 仅作配对背景，不作为新的独立证据。代码提交 `8d67f578a32a858735712c926c98bf6b38fc0a0f`，
> 定向 tests `17`；server check-only `CHECK_ONLY_OK exit0`，`24` unique/`8` reuse validated/
> `16` new planned/`0` runtime，真实 horizon `1800`、new-rollout 上限 `28800`，before=`3fe7`、
> after=`ac7`。唯一 launcher PID `2065459`、full child PID `2065466`，cwd `/home/user/projects/terminal`，
> cmdline 含 `--joint-team-cohort`。最终 `exit=0`、report 为 `COMPLETED`、`technical_ok=true`；
> 两臂各 `288` requests：before `236/52/0`、after `230/58/0`（normal/emergency/failed），
> 安全与 unresolved 均为 `0`；mean wait `131.8504766→135.6220302`，p95
> `315.1479841→327.7887054`，max `692.8147076→692.8147076`，low-SOC wait
> `115.4893036→125.5779790`，reward `5359.1727272→5276.9198028`。new-only 两臂各 `192`
> requests：before `156/36/0`、after `151/41/0`，mean `135.4809029→141.2362062`，p95
> `331.1184447→333.4887312`，low-SOC `113.5369283→127.2227828`，reward
> `3548.6529080→3475.6094504`。所有 freeze 与 safety/unresolved 检查通过，heartbeat `automation`
> 已删除；主审拒绝 `ac7` 性能候选，但保留 joint-team 数学修正，不宣称退化已解决。复用根、seed
> 映射和报告范围见
> [joint-team repeats protocol](evidence/v4_2/p3_task_neutral_joint_team_repeats_protocol_20260929.md)。
> 两臂各 `288` requests 的全部结果与 new-only `192` requests 结果均保留；问题仍未解决。

> **当前授权：joint-team learning-curve development protocol（待定向 tests/check-only，2026-09-29）：** 输出根为
> `/home/user/training_outputs/terminal/p3_task_neutral_learning_curve_20260929`。恢复 `3fe7` actor
> 与 `b136` critic，fresh Adam 在同一 run 的六次 update 间持久；不沿用 `ac7`、不回到 full-MC。
> 六批各六条完整 on-policy rollouts，共 `36` rollouts/`864` requests，training 上限 `64800` steps；
> env `877101..877136`、policy `878201..878236`、permutation `878001..878006` 依序分批。
> 第 `2/4/6` 次 update 后评估四个 dense24 cells，before 复用既有四条 records，after 共 `12` 新
> rollouts、最多 `21600` steps、每截面 `96` requests。无 Teacher/heldout，`PPO_ALLOWED=false`；
> 先定向 tests/check-only，主审准入后才运行唯一 full，当前尚未启动。
> 详情见 [learning-curve protocol](evidence/v4_2/p3_task_neutral_learning_curve_protocol_20260929.md)。

> **历史阶段：Scoped task-neutral PPO pilot 终态（2026-09-28）：** pilot report 为 `COMPLETED`、
> `technical_ok=true`，服务器 `full.exit=0`；3 次 `update()` 调用记录 `148` accepted attempts
> （`40/62/46`）、`0` reject，roundtrip/profile/freeze/固定 `q` 均为 `true`，安全计数为 `0`。
> 结果见 [pilot result](evidence/v4_2/p3_task_neutral_ppo_pilot_result_20260928.md)，协议边界见
> [pilot protocol](evidence/v4_2/p3_task_neutral_ppo_pilot_protocol_20260928.md)。pilot checkpoint
> `1f7ca8ad1fa4a041b3e62017c56dc47a7c418865613a6241c86fa18763d0fb81` 不保留，继续以
> `3fe7...d4ab` 作为 task-neutral 开发起点，source `03c0` 保留对照；不声称稳定收益已解决，
> 也不把任务优先级归为唯一原因。该阶段的“无更多训练”仅约束该 pilot 阶段，不是本轮 replay
> 活动的停止条件；`PPO_ALLOWED=false`、formal/heldout/P4/static flags 保持 false，heartbeat
> `task-neutral-ppo-pilot` 已删除。

> **历史阶段：task-neutral update diagnosis replay 终态（2026-09-28）：** 服务器 `b23966f`
> 对 `p3_task_neutral_ppo_pilot_20260928/full/training/episode_03` 既有 tape 完成精确 replay：
> 同一 task-neutral `dense24/873103`，`before=episode_02/pilot_after_update.pt`、
> `after=full/pilot.pt`；`729/729` exact、`24` requests、`621` actor steps、`2508` actor rows，
> 前后模型不变且无 optimizer。复用既有 capture/GAE，核对同状态概率变化、原 GAE 与 MC、以及
> arrival `20/21` 身份。此前本地失配事实保留；本地 reset 数值差最大为
> `1.136868e-13`，不能称本地 exact 通过，服务器 exact 已通过。该 replay 不预断
> gamma/reward/critic 根因，不增加普通 priority 保护；安全与物理负荷保持不变，
> formal/global flags 不自动改变。输出根为
> `/home/user/training_outputs/terminal/p3_task_neutral_update_diagnosis_20260928`。

> **历史阶段：先决采样诊断终态（2026-09-28）：** 代码提交为 `12f7873`，定向测试 `37` 项；
> check-only 为 `CHECK_ONLY_OK`、`16`-matrix、`0` records，full report 已完成。该 sampling
> repeats 对两臂各 `192` requests（总计 `384`）：旧 PPO 重复评估 `before` 为 `140 normal / 51 emergency /
> 1 failed`，`after` 为 `136 / 56 / 0`，安全计数为 `0`。输出根为
> `/home/user/training_outputs/terminal/p3_task_neutral_sampling_repeats_20260928`。
> `16` rollouts 的 sampling-repeats 诊断，只评估已有两权重：`before=3fe7`、`after=1f7`。
> 使用相同两个 dense env seeds `873103`、`873106`，每个各使用预定策略采样 seeds
> `874301..874304`；只分离 torch policy sampling seed，env 的物理 seed、arrival、SOC 和 service
> 不变，不更新模型。输出根为
> `/home/user/training_outputs/terminal/p3_task_neutral_sampling_repeats_20260928`。
> 目的仅是区分单条采样轨迹波动与期望退化，不等于独立的新环境泛化；首请求差异在 step `56`
> 执行相同，但 conditional proposal draws 改变后续采样链，不能仅凭相同 env seed 作强因果解释。
> 上述 full 为 `2` arms × `2` dense env seeds × `4` policy seeds 的 `16` rollouts，不自动重复。
> mean wait `160.633→172.684 s`，p95 `369.113→445.294 s`，low-SOC wait
> `187.337→197.268 s`，reward `3219.130→3166.920`；8 个 paired normal deltas 为
> `[-2,0,1,0,3,-3,-3,0]`，其中 `7/8` 个 mean wait 上升。两既有 env 的 action repeats 支持
> 继续修复，但不能归因仅一次采样，也不是独立环境泛化或显著性证明；本轮安全与 heldout 边界不变。
> 结果见 [critic repair result](evidence/v4_2/p3_task_neutral_critic_repair_result_20260928.md)。

> **历史阶段：critic warmup 终态（2026-09-28）：** 用户继续授权仅此 task-neutral
> 开发修复；代码提交为 `92baea9`，定向测试 `37` 项、check-only 通过，已从 `6` 个既有 tapes
> 收集 `4775` steps。第一次启动 wrapper 返回 launcher `262040` 后立即消失，未创建
> warmup/log/pid/exit；第二次尝试在外层 compile (`launcher_code`) 暴露 `SyntaxError: unterminated
> string literal`，原因记录为 `pid_path.write_text` 拼接换行转义错误，未 `Popen`、无 leaf/log/fit，
> 两次均不计作实验重跑。随后唯一真实 runner PID `397550` 完成 `exit0`：`6` exact、`84`
> requests、`4775` steps、`20` epochs；MSE `733.8008877569→104.6766084265`，MAE
> `20.2631550239→5.6295053403`，EV `.25466145→.76812814`，actor/noncritic bitwise unchanged
> 为 `true`，无 PPO、sampling 或 Teacher。`warmup.pt` SHA256 为
> `b136a57a8a977b60d5ae71fe3925c8cf133433c0c5fc4e6bef9872295353844d`。这些是数据内拟合结果，
> 不代表泛化或问题已解决。
> 主审已通过 warmup；`pilot_check_only` report 为 `CHECK_ONLY_OK`，随后唯一 `pilot/` report
> 为 `COMPLETED`、`technical_ok=true`。6-cell、84-request 配对为 normal `72→74`、emergency
> `12→10`、failed/unresolved `0/0`、安全 `0`；mean wait `96.3711999828819→94.8296761036019 s`，
> p95 `329.005735076254→349.561162932274 s`，max `503.62641472566→492.246915376754 s`，
> low-SOC wait `119.713456075581→110.00386056458 s`，reward `1623.02810599464→1637.18852268814`。
> checkpoint SHA256 为 `fa0e4218893f51934772e9629e507b86d51024d9f7b4664e6831e31ac135c289`。
> 主审判定为开发 candidate：单次 6-cell 有收益与尾部 tradeoff，不称稳定修复；已有 6-cell 已被
> critic warmup 使用，`validation` split 名称为历史保留，本轮不作独立验证。包装修复要求为
> `write_text(str(child.pid))` 与 `str(returncode)` 不带换行，并在启动前完成本地与远端 compile 预检。
> 输出根拟为 `/home/user/training_outputs/terminal/p3_task_neutral_critic_warmup_20260928`。
> 固定 `3fe7...d4ab` actor，重放已有 neutral baseline 的 `6` 个 tapes 收集 world/reward；
> critic target 使用完整终态 MC、`gamma=.99`，不使用 GAE bootstrap。只优化
> `critic_encoder`/`critic_head`，fresh Adam、`lr=3e-4`、`20` epochs、minibatch `64`、seed
> `874501`、`gradclip=1`；actor、proposal、temperature 逐位冻结。拟合结果只说明数据内误差。
> 通过冻结、finite、checkpoint 检查后，主审才可启动一次与旧 pilot 相同的 `3` episodes PPO
> 和 `6`-cell evaluation，用于验证 warmup 效果；不改变 reward、gamma、lambda、安全或
> task-neutral 配置。repeats check-only 已 `exit0`：`8` new / `8` reuse、`16`-matrix、
> `fa0e` checkpoint 校验正确、`0` env steps；唯一 repeats runner PID `779974` 已结束，`exit0`，
> report 为 `COMPLETED`、`technical_ok=true`、weights frozen、安全计数为 `0`。`before` 共
> `192` requests 为 `140 normal / 51 emergency / 1 failed`，`fa0e after` 为 `142 / 50 / 0`；
> mean wait `160.633→157.797 s`，p95 `369.113→395.619 s`，max `725.530→496.725 s`，
> low-SOC wait `187.337→176.742 s`，reward `3219.130→3304.646`；8 个 normal deltas 为
> `[-1,0,2,1,2,0,-2,0]`。旧未经 warmup 的 PPO `136 normal` 对比另列，不能与本次 repeats
> diff 当作独立样本。主审判定采样开发总体有改善且部分 seed 退化、p95 上升，支持继续新物理场景
> 验证，但不称已稳定解决。全局 formal/heldout 仍为 `false`。

> **历史阶段：物理场景验证终态（候选路线关闭，2026-09-28）：** 原计划固定 candidate `fa0e`，使用
> dense24 env seeds `874601..874604`、policy seeds `874701..874704`，两臂 `3fe7/fa0e` 共
> `8` rollouts、每臂 `96` requests。fresh runner 代码提交为 `3470e5a`，定向测试 `7` 项；
> fresh check-only 为 `CHECK_ONLY_OK exit0`，`8`-matrix、每臂 `96` requests、`0` records、
> `0` runtime，profile 与 neutral 检查均为 `true`。fresh report 为 `COMPLETED`、
> `technical_ok=true`、`8/8`；before `3fe7` 为 `80 normal / 16 emergency / 0 failed`，after
> `fa0e` 为 `73 / 23 / 0`，安全与 weights 检查正常。mean wait `129.233681726712→139.866715305331 s`，
> p95 `308.647078911221→325.725321758148 s`，max `480.280648536374→513.415812504217 s`，
> low-SOC wait `116.631497236334→145.221409523741 s`，reward `1798.49415586202→1708.07438156263`；
> normal deltas 为 `[0,-3,0,-4]`。执行偏差为 PID `960446` 已结束且完整 report/log 存在，但
> `ROOT/fresh.exit` 缺失；`fresh.launcher.log` 记录 `AttributeError: 'int' object has no attribute
> 'returncode'`，wrapper 终态未知，不写 `exit0`，不重跑。结果见
> [critic repair result](evidence/v4_2/p3_task_neutral_critic_repair_result_20260928.md)。
> 主审拒绝 `fa0e`，保留 `3fe7` 作为 actor 起点；warmup 数据内改善不足以解决泛化退化，不称
> root cause 已证。四个新 env seeds 已评估并消费，不再称 untouched。该历史阶段当时的下一步仅为
> 只读检查多条同策略完整 episode 拼接后的统一 PPO update 边界，且尚未授权训练；现行后续授权
> 与边界由 [joint-batch protocol](evidence/v4_2/p3_task_neutral_joint_batch_protocol_20260928.md)
> 覆盖。全局 formal/heldout/`PPO_ALLOWED` 不变。
>
> **上一轮普通 soft-priority 课程段落（历史，仅保留事实）**：
> **当前语义澄清（2026-09-28）：** 普通 `base_priority=1/3/5` 是可权衡的偏好；只有 `mission_type=emergency` 的紧急任务，在安全间隔、能量/轨迹可执行性及既有不可中断计划约束下，对同时争用同一可分配资源的可执行普通请求优先安排降落。这不是无条件最先落地，也不允许突破安全。`mission_type=emergency`、`base_priority=5` 与 `outcome=EMERGENCY_LANDED` 是不同语义。普通 priority wait 和 low-SOC wait 是连续优化/分析指标，不独立要求每组或每个 cell 均无退化；low-SOC wait 作为软统计不撤销既有 energy/admission、route-energy/slack、`min_landing_soc` 等硬边界，也不撤销既有危急 SOC 的应急接管语义；`failed`/安全的既有要求保持不变。本轮未选择新的 reward 权重或修改 scheduler，不声称代码已实现紧急优先强保证。“下一步添加普通优先级硬保护”已撤回；上一轮 `candidate_retained=false` 是旧协议历史结果并保持，候选仅作为需进一步验证的 normal 收益信号，当前仍使用 source `03c0`，不自动晋级 PPO/formal/heldout。主审按上述边界只读核对 reward→GAE→PPO actor 链路及既有 counterfactual 回报、按 mission 统计；诊断结果见下方，旧准入判定不变。
>
> **PPO credit signal 诊断终态（2026-09-28）：** 按 [PPO credit signal protocol](evidence/v4_2/p3_ppo_credit_signal_protocol_20260928.md) 完成一次只读 replay，结果见 [PPO credit signal result](evidence/v4_2/p3_ppo_credit_signal_result_20260928.md)。代码提交为 `77a549bbc12c2dfdc7abb12f9361e325eb72b2e4`，定向测试 `32 passed`（`9` diagnostic + `11` lifecycle + `12` evaluation）；修正一次启动前远端 cwd 遗漏导致的 `ModuleNotFoundError` 后，唯一 replay 为 `COMPLETED exit0`，`702/702` exact、`24/24` ledger/all-requests、`512` actor steps、terminal step `701`，source weights unchanged、RNG restored，collision/hard-separation/NMAC 均为 `0`，无 active run/heartbeat。原预检未执行 reset/step，不计为第二次实验。
> 诊断 scope 明确无 training/optimizer/sampling/Teacher/heldout；source raw reward 为 `448.98876382803303` 且组件吻合。关键 step 的 `gamma=.99` 与 fixed-critic `gamma=1.0` 对照均记录在结果页；局部 normalized 信号为负不能宣称 step102 必然鼓励 WAIT，也不把 actor-step 标准化写成 bug，固定 critic 下仅切换 gamma 未改变三步信号符号；V 与单轨迹 MC 的差异不证明 V7 全部退化根因或 critic 统计失准。
> mission 配对只覆盖 dense24/873103 的 `24` requests；source 新 trace 记录 mission，candidate 仅在内存分析副本中按 source exact replay 的 immutable request 属性 join，结果见结果页，不写候选 JSON sidecar；`request_time`/`base_priority` 完全一致，SOC 重建 `maxdiff=3.885780586188048e-15`。cargo 与 standard 分组存在 normal/emergency outcome 变化；surveillance 四架均为 normal，仅等待变化；唯一 emergency mission 为 arrival20，两边均 normal，wait `+11.176075422`、touchdown `545→567`，未完成资源争用层检查，不据此宣称 mission 优先保证。candidate 的补齐 reward 为 six-cell 总体 `1606.1318631737224→1621.4070350399447`，normal `76→78`、emergency `8→6`、PW `+0.95309s`、low-SOC `−2.67283s`；dense873103 单 cell 为 `448.9887638280331→464.2639356942554`、normal `21→23`、emergency `3→1`。这是待进一步验证的 normal 收益信号，旧 `candidate_retained=false` 历史保留，不自动晋级或切换 source。当前仍 `PPO_ALLOWED=false`，未修改 gamma/reward/scheduler，也未启动新的 PPO。
>
> **代码快照**：state-only 收集实现为 `768b191`，gate-adapter 训练器为 `b9f1077`，本轮 SOC gate 修复为 `a8ea93bf0e41f316fc161c1c9d1d7576395a8850`；V7 adaptive-KL 实现与定向测试提交为 `7b85d79`，evaluation runner/tests 为 `5292945`；runtime rank anchor 修复与证据整合为 `2097990`，指定定向套件 `61 tests passed`。V7 YAML SHA256 保持 `d2bbd3a3361abb0589099fd2096b8f404f61c34bf94db10a958de99fc3f14a53`。Git 无 remote，未 push；本轮 terminal credit/reward audit/pairing correction 已提交 `e0615df`、`44ec2d5`、`d44fde1`；gate-only bounded-backtracking 实现提交为 `9346ee3`，trainer/evaluator 定向测试 `15 passed`。
>
> **当前 normal-advantage gate repair 终态（2026-09-28）：** 实现提交为 `adfe0f2a255e2d5c34268777624d228edeac1935`，原五点实现定向测试记录为 `39 passed`（`11` local-window + `28` trainer/eval），三点修订 commit 为 `a1c249e4e667337ccc09a476dee658869d05497b`，修订后定向测试仍为 `39 passed`。活动 V2 credit `COMPLETED exit0`：固定 steps `102,103,104`，`153` runtime branches，`27 logical / 18 actual / 9 reuse`，`9/9` acceptance，`27/27` terminal/full24/ledger/model/RNG/safety checks 通过，failed=`0`；normal deltas 为 `[2,1,1]`、`[3,-3,0]`、`[0,0,0]`，A=`[4/3,0,0]`。原 `100..104` credit FAILED 与 fit_smoke 命令偏差均保留。
> 原定 fit_smoke 单 logical update 命令实际遗漏 `--smoke`，于 `16:06:04–16:06:07 (+0800)` 在 `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke` 运行 `smoke=false/full120`，metadata 为 `smoke=false`；不是训练器忽略参数，不声称单步预检通过。该 full fit `COMPLETED exit0`：`120 accepted / 120 logical / 123 optimizer`，final KL mean/max=`1.0704449096010649e-6/.0016782581806182861`，freeze/proposal/roundtrip 全 true，`11592` anchors；候选 SHA256=`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`。预算已消耗，不再 fit。
> 六 cell evaluation `COMPLETED exit0`、`technical_ok=true`、`candidate_retained=false`，保留 source `03c0`；84 requests 为 normal `76→78`、emergency `8→6`，failed/unresolved/safety 均为0，PW wait `86.66905761611285→87.62214654891294`，lowSOC wait `89.77926924393867→87.10643757570104`，mean wait `81.38983429949938→78.89363393551653`。仅 train dense24/873103 变化：normal `21→23`、emergency `3→1`，PW `99.77872368158535→102.66602250742093`，lowSOC `123.60082027309466→110.23666193190647`，其余5 cell不变。by_priority：priority5（16）wait `90.83635980074234→100.83653713035764`、normal/emergency `14/2→13/3`；priority1（39）wait `71.32276779715836→64.84362501084031`、`36/3→37/2`；priority3（29）wait `89.71642690541016→85.68204417499636`、`26/3→28/1`。
> check-only 为 `CHECK_ONLY_OK exit0`；六 cell 无更新/Teacher/heldout，proposal/nonadapter 冻结，checkpoint roundtrip exact。开发集存在 normal 收益，但最高优先级子组退化且 aggregate PW wait 保留标准失败；不写成稳定收益或训练退化已解决，不归因于 KL 或单个状态因果。下一方向记录为终态收益增加优先级保护，尚未预注册/启动。heartbeat `normal-advantage-repair-v2` 已删除，评估完成无需 monitor；不新训练、不启动 PPO、不消费 untouched heldout。协议见 [normal advantage gate repair protocol](evidence/v4_2/p3_normal_advantage_gate_repair_protocol_20260928.md)，结果见 [result](evidence/v4_2/p3_normal_advantage_gate_repair_result_20260928.md)，credit_v2 见 [report](../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/credit_v2/report.json) 与 [summary](../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/credit_v2/summary.json)，fit 见 [report](../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/report.json)，evaluation 见 [manifest](../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/evaluation/manifest.json)。
> **上一轮终态动作机会诊断（历史背景，2026-09-28）：** Full `COMPLETED exit0`，共 `120` runtime rows 和 `6120` 单步 branches；3 个 selected states × 3 repeats，`27 logical / 24 actual / 3 reuse`。source replay/outcomes/model/RNG checks 均为 true。selected：state0 arrival12 uid42 step102，`p=.000710682827`、`qK3=1`、`C=24`；state1 arrival12 uid42 step139，`p=.998769342899`、`qK3=9.7754e-15`、`C=9`；state2 arrival21 uid309 step366，`p=.759813606739`、`qK3=1`、`C=13`。Release-vs-WAIT normal gains 分别为 `[2,1,1]`、`[1,1,1]`、`[1,0,0]`；target_normal gains 分别为 `[1,1,1]`、`[0,0,0]`、`[1,0,0]`；failed deltas 全为 `0`，policy gaps 为 `1.3323857562 / .0012306571 / .0800621311`。proposal deltas 分别为 `[0,0,0]`、`[0,-1,1]`、`[0,0,0]`，均不满足机会条件。state0 PW wait 变化为 `[+16.54458,-11.51764,-16.29416]s`，lowSOC wait 均下降 `[-12.63103,-24.98760,-19.92814]s`。heartbeat `landing-outcome-credit-diagnosis` 已终态删除。state0 是当前优先研究的局部 gate 机会：source 放行约 `0.0711%`，目标 UAV 三次均由 emergency 转为 normal；state1 source 放行 `99.8769%`，gate 改进空间小；state2 证据较弱。不把三个 state 收益相加（同场景有重叠）；MC3/global RNG 后续 actor 路径可能分叉，单一 development 场景且未训练，尚不能声称稳定收益或训练退化已解决。保留 source `03c0`、no-heldout，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`；旧 V2 结果不变。协议见 [landing outcome credit protocol](evidence/v4_2/p3_landing_outcome_credit_protocol_20260928.md)，结果见 [landing outcome credit result](evidence/v4_2/p3_landing_outcome_credit_result_20260928.md)，full 见 [report](../../training_outputs/terminal/p3_landing_outcome_credit_20260928/full/report.json)、[summary](../../training_outputs/terminal/p3_landing_outcome_credit_20260928/full/summary.json) 和 [selection](../../training_outputs/terminal/p3_landing_outcome_credit_20260928/full/selection.json)。arrival12/13/21 各40 runtime rows，C 非空数为 `2/0/2`（非空 steps：arrival12 `[102,139]`、arrival21 `[366,369]`；369 距366仅3步，按 `<10` 排除，最终3 states）；全部9个 state/repeat 的 acceptance、arrival signature、base/model/parent RNG 均为 true，27 branches 的 terminal、full-requests24、ledger、weights 均为 true，collision/hard-interval/NMAC/execution-plan-hash mismatch 增量均为0，technical checks 全部通过。无独立 exogenous 命名字段，未声称检查未提供字段。该段保留上一轮诊断事实；normal-advantage gate repair 的已完成结果见顶部当前状态，不直接重启 PPO 或扩大旧 hard CE fit。
> **最新摘要：terminal gate repair V2（2026-09-28）：** V2 smoke `COMPLETED exit0`，`1 accepted / 2 optimizer attempts`，accepted trial scale `.5`，max KL `0.0034920275`；V2 full `5 accepted / 6 logical / 18 optimizer`，accepted scales `[.5,1,.5,.25,.0625]`，第 6 个 logical update 的全部 trial 拒绝并完整回滚，最终 mean KL `0.000055444827`、max KL `0.0096337497`，freeze/roundtrip invariants 全部通过。六个 development cells 的 matched evaluation `COMPLETED` 且 `technical_ok=true`，candidate 不保留、选择 source `03c0`；source `76 normal / 8 emergency`，candidate `74 / 10`，failed/unresolved/safety 均为 `0`。有界更新技术有效，但稀疏五标签未转化为完整收益；当前不声称训练退化已解决，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`。结果见 [V2 smoke report](../../training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/smoke/report.json)、[V2 fit report](../../training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/fit/report.json) 与 [V2 evaluation manifest](../../training_outputs/terminal/p3_terminal_gate_repair_v2_20260928/evaluation/manifest.json)。

> **当前 gate-only 终态（2026-09-28）：** V1 smoke 的机械零更新历史保留；V2 smoke 为 `COMPLETED exit0`、`1 accepted / 2 optimizer attempts`、max KL `0.0034920275`。V2 full 为 `5 accepted / 6 logical / 18 optimizer`，第 6 个 logical update 全部 trial 拒绝并完整回滚，最终 mean/max KL 为 `0.000055444827 / 0.0096337497`，freeze/roundtrip invariants 全部通过。evaluation 为 `COMPLETED`、`technical_ok=true`、`6/6` cells、`84` requests、全 ledger 通过；candidate 不保留，继续使用 `03c0` source。四个 flow6/flow12 cells 不变；dense24/873103 的 normal `21→21`，priority wait `99.77872368→104.41590090`，low-SOC wait `123.60082027→125.77824302`；dense24/873106 的 normal `19→17`，priority wait `159.20141913→157.05353879`，low-SOC wait `145.30236533→176.32696493`。主审拒绝候选并保留 source；首处分歧及反向改善已列事实，未作完整因果判断。`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`。结果链接见顶部摘要。

> **主审解释与下一步（2026-09-28）：** source weightedCE `0.174300178885` 降至 candidate `0.168798521161`；state2 WAIT 的权重为 `0.00905526`，source CE `17.531902`、weightedCE `0.158755898476`（`91.1%`），完整后续收益仅 priority wait `0.129199155313 s` 且 normal 不变；state6 权重 `0.95162936`，source RELEASE `p=0.999998689`、source weightedCE `0.0000012479`，candidate `p=0.999998093`；state11 RELEASE `p=0.148311973→0.161266223`，最小收益 `0.044117647059 s` 且 normal 不变。主审结论为稀疏硬 CE 目标受微小收益与高置信反标签主导；拒绝继续扩大五标签 fit 或重启 PPO，不声称训练退化已解决。该句属于历史下一步记录；normal-advantage gate repair 的完成结果见顶部当前状态，以上诊断边界仍按历史事实保留。完整分歧事实见 [terminal credit result](evidence/v4_2/p3_terminal_credit_result_20260928.md)。

## V7 formal campaign 终态（2026-09-26，全请求 lifecycle 回放已完成）

formal 输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/`。三 run 均已退出：

| run | status | processed / UPDATED / SKIPPED | accepted minibatches | training requests（normal / emergency_landed / failed_energy / other / unresolved） |
|---|---|---:|---:|---:|
| run0 | `FAILED exit1` | `80 / 80 / 0` | `3954` | `1320 / 1185 / 135 / 0 / 0` |
| run1 | `COMPLETED exit0` | `240 / 240 / 0` | `11737` | `3960 / 3518 / 442 / 0 / 0` |
| run2 | `COMPLETED exit0` | `240 / 240 / 0` | `11808` | `3960 / 3508 / 452 / 0 / 0` |
| 合计 | — | `560 / 560 / 0` | — | `9240 / 8211 / 1029 / 0 / 0` |

上表合计 accepted minibatches 不做求和展示；各 run 数值以对应 manifest 为准。所有 `560` 个 update 的 gate delta 均大于零，训练 safety 计数为 `0`，restore/RNG false 为 `0`。reference KL 最大值（run0/run1/run2）为 `.0243685003/.0255418718/.0265666898`，rollout KL 最大值为 `.0099292190/.0099943196/.0099836820`。

final development 各为 `84` requests：run0 在 episode `80` 为 `76 normal + 7 emergency_landed + 1 failed_energy`，run1 在 episode `240` 为 `76 normal + 8 emergency_landed + 0 failed_energy`，run2 在 episode `240` 为 `72 normal + 12 emergency_landed + 0 failed_energy`；baseline 为 `76 normal + 8 emergency_landed`。run0 的 development failed-energy 单独计入 development，不混入上表训练请求的零 failed-energy 统计。

对应 checkpoint SHA256 为：run0 episode `0080` `ad42e9a72cac4943834041482e93adc34f123c851c4297f7462381fe0d52ff4e`；run1 episode `0240` `db74245759653bb61f8d17b0071cf9a04c2e3334e019a63cd99dff988cc23b3a`；run2 episode `0240` `4797ae2eea5da0e60a4b24e2fd0a60a6860c37b8efe1c77c0366a33a2d8fb5f9`。checkpoint 均位于上述 formal root 的对应 run 目录。

主审结论为可学习性成立，但三 seed 全通过不成立；目前没有性能全面提升证据，`P3_QUALIFIED=false`、`P4_ALLOWED=false`，不自动恢复 run0。原 `64` episode final evaluation 的前置条件未满足，未消费 `880xxx` 或既有 heldout。全请求 lifecycle 回放已完成：manifest `/home/user/training_outputs/terminal/p3_adaptive_kl_terminal_dev_metrics_20260926/manifest.json`，SHA256 `7188f7160511446a91cb873ed5cd607378eede3bc00d567467cf25f26496f3a1`；`24/24 COMPLETED`、exit `0`、error `0`、validation error `0`，四组均为 `84/84` terminal rows、missing `0`、exact/ledger 一致。`smoke_v2` 为 `1/1 COMPLETED`，`6` requests/`6` terminal rows、missing `0`、ledger/replay/outcome 全通过、exit `0`；首次 smoke 输出不作为本次指标证据。本轮仅重放既有 tape，不是新 policy rollout。source baseline 与三个 final development 共 `24` 个既有 tape 的全请求指标已完成。

> **历史 V6 与受限恢复结论（V7 之前，2026-09-26）**：V6 formal run1 完成 `240/240`，run0/run2 分别在 `79/80`、`48/49` records 后停止；合计 `6078` requests、`0 failed_energy`，P3 未通过，该轮结束时三 run 均已终止。随后针对两个停止点完成了受限 KL recovery probe：两组各处理 `4` 个诊断回合、各 `2` 次 UPDATED 与 `2` 次安全 `SKIPPED_KL`，无 HARD_FAILED；该流程验证了回滚/缩步恢复路径，但有效更新不足以恢复长期 formal 训练。

> **V7 adaptive probe 与 formal 终态（2026-09-26）：** adaptive probe run0/run2 各 `8` 个回合均为 `8 UPDATED / 0 SKIPPED_KL / 0 HARD_FAILED`、`8/8` restore exact；accepted minibatches 为 `298/332`，最终 reference KL 为 `.0077864/.0133243`。formal 三 run 已终态：run0 `FAILED exit1`，`80 processed / 80 UPDATED / 0 skipped`、`3954 accepted minibatches`；run1 `COMPLETED exit0`，`240 processed / 240 UPDATED / 0 skipped`、`11737 accepted minibatches`；run2 `COMPLETED exit0`，`240 processed / 240 UPDATED / 0 skipped`、`11808 accepted minibatches`。训练请求合计 `9240`（`8211 normal + 1029 emergency_landed`），训练 `failed_energy/other/unresolved=0`；开发失败单独计入 development，不与训练请求混合。所有 `560` 个 update 的 gate delta 均大于零，训练 safety 计数为零，restore/RNG false 为零。主审结论为可学习性成立但三 seed 全通过不成立；无性能全面提升证据，不自动 P3/P4 晋级，也不恢复 run0。
> **历史 V7 用户目标（2026-09-26，已终态）：** 完成有效的 `3×240` formal PPO，并在训练后对同一场景集合执行冻结策略评估；adaptive probe 已完成并通过 primary 的可学习性审查，但 dense24 性能存在退化，formal 三 run 已终态但 run0 失败、run1/run2 完成；全请求 lifecycle 回放已完成，`64` episode final evaluation 的前置条件仍未满足且尚未启动。候选 reference regularization 为 `0.1*K + 400*relu(K-0.02)^2`，reference weighted-KL hard cap `.05`、rollout-KL target `.01`、03c0 source、proposal 子图冻结、安全边界保持不变；不增加 probability floor、强制 RELEASE、UID/SOC override 或 SOC 专用规则。正式计划保留三组独立 `240` episode 与同场景 development/evaluation；dense30 与同负荷随机密度评估只在正式训练完成并冻结策略后执行，当前不消费 heldout。random layout opt-in 默认 `false`，仅供后续 evaluation。V7 YAML 参数已冻结。
>
> **历史摘要：runtime rank repair 三候选终态（2026-09-27）：** 已完成 `164` 个 runtime-label rows、`8,364` 个 branches 与 `6/6` exact records；运行时标签失配、非目标 proposal head 漂移和报告接口已修复。source `03c0` 保持参考；V1 runtime rank、sparse rows `[9,20]` 与 anchored rows `[9,20]` 三个候选均不采用，训练退化尚未解决。`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`，未消费 heldout。下一步针对终态 `normal/emergency` 与长程等待做 action-credit 目标诊断；继续调 coverage/MSE 或扩大 formal 不作为已验证解决方案。详见 [runtime rank repair result](evidence/v4_2/p3_runtime_rank_repair_result_20260926.md) 与 [anchored protocol](evidence/v4_2/p3_runtime_rank_anchored_20260927.md)。

| policy | overall normal / emergency | train normal / emergency | validation normal / emergency | overall priority / low-SOC wait |
|---|---:|---:|---:|---:|
| source `03c0` | 76 / 8 | 39 / 3 | 37 / 5 | 86.6690576161 / 89.7792692439 |
| V1 runtime rank | 73 / 11 | 39 / 3 | 34 / 8 | 94.5130544827 / 85.3075803003 |
| sparse rows `[9,20]` | 74 / 10 | 39 / 3 | 35 / 7 | 83.4356842607 / 90.9519380786 |
| anchored rows `[9,20]` | 75 / 9 | 40 / 2 | 35 / 7 | 89.7123814381 / 103.6657082349 |



四组全请求 lifecycle 指标如下（等待单位为秒；low-SOC 为 25 个样本均值）：

| policy / checkpoint | normal | emergency_landed | failed_energy | priority-weighted wait | low-SOC wait |
|---|---:|---:|---:|---:|---:|
| baseline | 76 | 8 | 0 | 86.6690576161 | 89.7792692439 |
| run0@80 | 76 | 7 | 1 | 110.881727846 | 115.732325433 |
| run1@240 | 76 | 8 | 0 | 90.1429896281 | 93.538322835 |
| run2@240 | 72 | 12 | 0 | 95.8325596837 | 92.876852833 |

主审结论为：run1 的 normal 数量持平但等待退化；run2 的 normal 数量减少 `4` 且等待退化；run0 保留 `1` 个 development `failed_energy`。可学习性成立，但性能全面提升不成立，P3 目标未完成。下一步先冻结与实际 K3 执行一致的可执行标签覆盖诊断/小 probe，不直接启动大规模训练，不开启 late flag 晋级；原 heldout 不消费。

> **2026-09-26 bounded KL recovery probe（已完成，非 formal 续训）：** 针对上述两个 `reference_kl_exceeds_max`、零 accepted update 现场，在新根 `/home/user/training_outputs/terminal/p3_kl_recovery_probe_v1_20260926/` 完成 run0/run2 各 `4` 个 training episodes。两组均为 `4 processed / 2 UPDATED / 2 SKIPPED_KL / 0 HARD_FAILED`，每组各有 `2` 个真实 accepted minibatch，合计 `4`；共 `8` 个 checkpoint restore exact。固定原始 `03c0` reference、reference cap `.05`、rollout cap `.01` 以及安全/能量 hard gates 未改变。run0 的 `860080` 在原始 LR 与 `.5` 重试拒绝、`.25` 接受，`860081/860082` 四档均拒绝并安全 skip，`860083` 在 `.125` 接受；run2 的 `862049` 在 `.5` 接受、`862050` 在 `.25` 接受，`862051/862052` 安全 skip。最终 reference KL 为 run0 `.0499624982`、run2 `.0499344878`，均仍不超过 `.05`，但接近上限。两组训练诊断合计 `132` requests：`117 normal`、`15 emergency_landed`、`0 failed_energy`、`0 unresolved`；run0 为 `58/8`、run2 为 `59/7`（normal/emergency_landed）。before/after development 均 full-resolved；run0 的三个 development 场景指标相同（dense24 emergency `1→1`），run2 的 flow6/flow12 指标相同，dense24 emergency `3→3`，priority wait `-3.97625s`、low-SOC wait `+2.51198s`。本诊断已消费所列 seeds，旧 formal 输出保持 immutable；它验证了受限回滚/缩步恢复流程，不能据此恢复剩余长期 formal、晋级 checkpoint、通过 P3/P4 或改变 source/reference cap/SafetyFilter。下一步仅保留小规模验证能改变更新方向的 reference 正则/优化器处理（例如 adaptive KL regularization），并比较有效更新率及 priority/SOC wait；不以继续缩步或增加回合替代该验证。未启动新的大规模训练、heldout 或 dense30。详见 [V6 bounded KL recovery probe](evidence/v4_2/p3_gate_anchored_v6_20260925.md#8-bounded-kl-recovery-probe-completed-review-required)。
>
> **历史阶段背景**：Phase A / A5。fresh fixed-rule、SOC `.75` 的 density six-cell 已完成，`failed_energy=0` 且命名 incident counters 为零，但不建立普适容量或零失败保证；fresh `.55–.90` late-slot pairs 仍未通过 `zero_energy_failure`，变体默认关闭且未晋级。singleton target-gap 修复已完成 `35` 个定向测试及四-cell paired evidence：fixed 与 legacy 的聚合分别为 `32 normal + 17 emergency + 11 failed-energy` 与 `26 normal + 23 emergency + 11 failed-energy`；该结果保留几何修复，但不宣称 zero-energy 已解决或已 qualification。此前固定 SOC `.75` density baseline 不作为该修复的验证。同版本性能链语义保持一致、wall-time 下降 `19.4%/18.6%`，但 100-step prefix 不是 qualification。fallback future route 尚未作未来 SafetyFilter 认证；`P3_QUALIFIED=false`、`checkpoint_promotable=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`。
>
**历史 V6 目标与当时授权（已被 V7 持续跟进目标替代，2026-09-25）：** V6 runner 已实现，外部 V6/PPO/GateReference 测试共 `34 passed`，冻结 YAML SHA256 为 `043ae7296d838e0970aadf78667621c16596c100e92992ce4e62fc2f1e7c0cf8`。服务器 smoke `859000` 已完成：两步、无 PPO，`SMOKE_COMPLETED PASS`，checkpoint restore exact、weights unchanged、安全计数为零；`unresolved=24` 属于局部 smoke，未执行 full-scenario 终态核算。首次 pilot 尝试 PID `2588122` 已 `EXIT1`；其余 V6 smoke/pilot/formal 事实保留在下方历史段。该段当时的 no-heartbeat 与不持续跟进要求仅适用于已结束的 V6 记录，不覆盖当前 V7。

**V6 pilot_restart1 终态（2026-09-25）：** manifest 为 `/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/pilot_restart1/manifest.json`，SHA256 `2545c9cb88dd86828018bc36eeee3401d894250e41278a5f40b7a86bab63634f`。该次 pilot `COMPLETED`、`PILOT_TECHNICAL_PASS=true`，12 个 episode、198 requests（181 normal、17 emergency_landed、0 failed_energy、0 unresolved）、622 accepted minibatches、0 rejected、12 次 restore exact；final reference KL 为 `.0166467186`，第 11 个 episode 的最大值为 `.0170518`，mean `Release source` 为 `.35982→.34847`。paired development 的 flow6 无变化；flow12 priority wait `+1.08689s`、low-SOC wait `+1.5s`；dense24 的 emergency `4→7`、normal `20→17`，priority wait `114.998896→103.393995s`（`-11.6049s`），low-SOC wait `133.919561→135.778840s`（`+1.8593s`）。全部 development 回合 full-resolved 且 `failed_energy=0`；technical pass 不等于性能指标全部改善或 P3 qualification。旧辅助 `priority_outcomes` 表对流入 UID 缺少完整映射，不能作为完整 priority 分母；本轮 priority wait 使用 lifecycle 全 `24` rows 且 replay exact。

**V6 formal campaign 启动验证（2026-09-25）：** 三组 check-only 通过后，于 `23:21:36/36/37 +08:00` 启动 formal run0/run1/run2。decision 为 `/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/formal_admission_decision.json`，SHA256 `a57dcba2c98ebbd573d905feec36518728fa72459f3b65a48921ee97c17d489e`；child PID 分别为 `2592623/2592625/2592627`，supervisor PID 分别为 `2592622/2592624/2592626`，输出根为 `/home/user/training_outputs/terminal/p3_gate_anchored_v6_20260925/formal/runN`。三组均已生成首个 `checkpoint_episode_0001.pt`，大小分别为 `1,039,047/1,038,983/1,038,983` bytes，并进入 episode 2；启动核验时 CPU 约 `99.9%`、stderr 为空。完成启动核验后按用户要求停止观察，不创建 heartbeat 或持续监控。按 pilot 每 episode 平均 `71.353s` 估计，240 episodes 约 `4.76h`，另加约 `42` 个 development episodes `0.49h`，含余量的参考窗口为 `5.5–8h`，预计 `2026-09-26 04:50–07:20`；该时间不代表完成保证。formal 启动记录之后的终态见下文；P3/P4 仍未通过。

**V6 formal campaign 终态复核（2026-09-26，服务器进程与退出码已复核）：** run0 为 `FAILED`，`79 accepted / 80 records`，exit code `1`；run1 为 `COMPLETED`，`240/240`，exit code `0`；run2 为 `FAILED`，`48 accepted / 49 records`，exit code `1`。run0 的失败 seed `860080`、run2 的失败 seed `862049` 均在首个 minibatch 遇到 `reference_kl_exceeds_max` 并 rollback；两次均只有 `accepted_updates` 为 false。三 run 的 last reference KL（run0/run1/run2）为 `.048804771/.008465138/.049026713`。全部训练 records 的 full-request 分母合计 `6078`：`5374 normal`、`704 emergency_landed`、`0 failed_energy`、`0 unresolved`；accepted episodes 合计 `367/720`，records 合计 `369`（含两条 rollout 已完成但 update 未接受的 records），`367/367` checkpoint restore exact。11 个 development 阶段均 `hard_failure=false` 且 `full_resolution=true`。run0/run2 的 development 仅为 `0/40`，不把 episode 40 的指标称为最终 `79/48` checkpoint 性能。run1 的 final development 两个 dense 回合分别为 normal `19→21`、`19→19`，emergency `5→3`、`5→5`；priority wait `114.606→118.235s`、`127.983→141.995s`，low-SOC wait `152.235→101.874s`、`105.807→109.148s`。修复后本批未出现能量耗尽；三 run 计划未完成，不能宣称跨 seed 稳定、普遍优于 source 或 P3 通过。保留 run1 候选，不自动晋级 P4、不恢复训练，未启动 heldout 或 dense30 新实验。主审下一步记录为：保持安全与 reference cap，先小规模验证回退步长及允许完整回滚的零更新回合恢复语义；不直接放宽 cap，也不从 I0 重训。

服务器完整复核还确认三 run `otherfailures=0`、stderr 均为 `0 bytes`、development 安全非零记录均为 `0`；run1 的 `11` 次拒绝均为 rollout KL，未发生 reference 拒绝，accepted reference KL 最大值为 `.0444951<.05`，因此下一步聚焦两次提前停止的更新恢复语义，不直接判断 reference cap 普遍过紧。

新的准入边界见[受控应急 P3 V5 协议](evidence/v4_2/p3_controlled_emergency_admission_v5_20260925.md)，正式规格见[formal V5 plan](evidence/v4_2/p3_controlled_emergency_formal_plan_v5_20260925.md)。旧 P3 formal、Q3、canonical comparison 和 heldout 记录保持历史含义，不追溯替换为新语义；`829001–829006` 继续 untouched。V5 entry 与 bounded pilot 已完成：6 个 registered PPO episode 的 `296` 个 accepted minibatches 全部无拒绝，checkpoint restore 全部 exact，安全、能量、完整性 gate 通过；paired dense24 的 `emergency_landed` 为 `4→3`，low-SOC wait 改善 `44.036s`，priority-weighted wait 增加 `2.763s`，该权衡只作性能审查报告，不构成 P3 qualification。该过程不重训 I0/I1/P1/P2；使用 mixed old anchors 做回归保护，避免从 I0 重新开始。

> 本文是滚动状态页。规范语义以 [README.md](README.md) 中列出的主规范为准。

**此前暂停点（2026-09-25，已解除）：** 当前代码迭代已完成：canonical strict loader 为 `29a4dba`，其 `32` 个定向测试通过；evaluator 的 `27` 个定向测试通过；只读服务器 checkpoint/hash/profile-load preflight 通过。暂停时 `848000/848101/848102` 尚未 reset 或 rollout；该句只记录暂停时状态。恢复后 `848000` smoke 与 `848101/848102` full pair 均已完成，结果仍只用于历史基线与修复候选的开发比较，不自动放宽新 V5 准入，也不消费 heldout。`P3_QUALIFIED=false`、`PPO_ENTRY_ALLOWED=false`、`FORMAL_P3_ALLOWED=false`、`P4_ALLOWED=false`。

**canonical pair 冻结结果（2026-09-25）：** `848101/848102` 的 canonical 与 repair 四个 arm 均 exit `0`、full-resolved、zero-energy 且安全硬计数为零；aggregate 为 canonical `31 normal + 17 emergency`、repair `40 normal + 8 emergency`。repair 在两组 priority-weighted wait 与 low-SOC wait 上均更优，因此 V5 entry/pilot 的冻结初始化来源选择 repair candidate E04：SHA256 `e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`，路径 `/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt`。旧 canonical `no_failed_outcomes` gate 的历史 false 保持不变；该开发选择不构成 P3 qualification。

**P3 V5 当前路线（entry/pilot 已完成，restart1 FAILED）：** 固定 `80/40/10 m`、`energy_capacity=1500`、`min_landing_soc=.10`、`soc_reserve=.25`、`emergency_trigger=.40`、`late-slot OFF`、`K=3`、`Observation V2.1=88D`、`Teacher=false`、`downstream_priority`。`849000` smoke、stochastic no-update entry `849101/849102/849103`、6 个 registered pilot arm `849201–849206` 和 paired development `849301/849302/849303` 均已完成。6 个 pilot arm 共记录 `296` accepted minibatches、`0` rejected，全部 update/restore/safety/energy/integrity gates 通过；paired before/after 均 full-resolution。首轮正式 run0/1/2（PID `2488257/2488326/2488392`）在首次 checkpoint 前因 `_save_and_restore` 未创建 checkpoints 父目录而停止：首轮 rollout、PPO update、策略/奖励/安全检查通过，但 `torch.save` 抛出 `RuntimeError`，无持久 checkpoint、accepted episode 为 `0`。保存修复提交 `b79d7a2` 已通过服务器 30 项含真实保存恢复测试；restart1 已按同一 E04、fresh Adam 和原 240-seed 计划启动，PID 为 run0 `2497811`、run1 `2497879`、run2 `2497946`，新输出根为 `/home/user/training_outputs/terminal/p3_controlled_emergency_formal_v5_restart1_20260925/`，各 run 的 `dev_before_training/*/progress.json` 已生成，首个持久 checkpoint 已验证。heartbeat `p3-formal-v5-restart1-monitor` 已删除；正式 campaign 终态为 `FAILED`，不自动重跑。本轮训练不增加随机化或高拥堵场景，不消费 heldout，也不启动额外 stress。正式训练完成后，再独立评估 dense30 以及同负荷下随机环数量/安全非均匀角度；具体评估 seeds/config 待训练结果后确定。

**P3 V5 restart1 首回合验证（2026-09-25）：** 三个 restart1 run 的首回合均 `accepted_episodes=1`、`records[0].checkpoint_restore_exact=true`、`stopped=false`。首 checkpoint 已验证：run0 `checkpoints/checkpoint_episode_0001.pt` 为 `1,039,047` bytes，run1 为 `1,039,111` bytes，run2 为 `1,039,111` bytes；当前状态为 `FAILED`，不宣称训练完成。

**P3 V5 formal campaign 终态（2026-09-25）：** restart1 三个正式 run 均已终止并记录为 `FAILED`；汇总与逐 run manifest 见 [formal V5 restart1 result](evidence/v4_2/p3_controlled_emergency_formal_v5_result_20260925.md)。合计 `337` accepted episodes、`340` records、`5,604` requests，其中 `4,726` normal、`875` `emergency_landed`、`3` `failed_energy`、`0` other-failed、`0` unresolved；三条失败均为 dense24，所有 records 的安全非零计数均为 `0`。P3 formal training 未完成，`P3_QUALIFIED=false`、`P4_ALLOWED=false`；heartbeat 已删除，不自动恢复或重跑，未消费 heldout，也未启动额外 stress 评估。

**P3 holding 诊断与修复候选（2026-09-25，coverage fit 与 anchor closed-loop 已完成）：** [诊断与候选修复记录](evidence/v4_2/p3_holding_diagnosis_repair_20260925.md) 的 12 个 state-only probe 显示 gate 在 E04 后进一步偏向 WAIT（E04 平均 `pRelease=.0368057613`，final 为 `.0010184070`，`1/12` certified RELEASE、`11/12` WAIT）；安全/资源/能量执行链仍是独立约束，UID108 在 `t=850` 的能量 `138` 小于约 `286` 的 required energy，因此不放宽 energy guard。三个 `trained_release_only` 反事实分别为 `21/3/0`、`20/4/0`、`21/3/0`（normal/emergency_landed/failed_energy），均无 unresolved 或 safety error；该结果不是训练成绩。

bounded `.05` + canonical WAIT development replay 已完成，三个 run 均为 `24/24` resolved 且 safety hard failures 为 `0`：原始与 repaired 的 `normal / emergency_landed / failed_energy` 分别为 run0 `16/7/1 → 16/8/0`、run1 `15/8/1 → 15/9/0`、run2 `18/5/1 → 21/3/0`，steps 为 `802/825/754`。目标 arrival outcome 为 run0 arrival4 `emergency_landed`、run1 arrival18 `normal` 与 arrival20 `emergency_landed`、run2 arrival16 `emergency_landed`。原始 lifecycle wait 分母为 `23`、repaired 为 `24`，不能直接按同一分母比较。该开发诊断仅消除了已知三例能量耗尽，未解决目标长期等待，不作为最终策略，deterministic argmax 不变。结果根为 `/home/user/training_outputs/terminal/p3_bounded_gate_diagnostic_v1_20260925/runN/`。

coverage repair 已以 E04 为起点完成一次固定 20-epoch adapter-only fit：coverage `1062` 行、anchor `1529` 行、dev `1335` 行，checkpoint SHA256 为 `03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`。训练后同一 checkpoint 在三个已知 dense24 failure seed 上为 `22/2/0`、`19/5/0`、`19/5/0`（normal/emergency_landed/failed_energy），steps `810/739/781`，每回合 `24/24` resolved、安全 hard counters 为零；仅 gate_adapter 训练改变，非 adapter 参数与 proposal logits 逐 tensor 不变，rollout 前后整个 checkpoint 不变；该在线 evaluation 为 Teacher=false，未调用 PPO/heldout。相对原失败现场三回合合计 `49 normal + 20 emergency_landed + 3 failed_energy`，coverage 结果为 `60 normal + 12 emergency_landed + 0 failed_energy`；这只表示已知三例 energy failure 在该复现集合中消失，emergency 仍是可接受的受控降级结果，不表示所有盘旋已消失或正式资格已通过。

同一 canonical 配置下，E04 与 coverage 在既有 flow6/flow12 development seed 的 closed-loop 回放均完成：flow6 为 `6/0/0`、627 steps，flow12 为 `12/0/0`、976 steps（normal/emergency_landed/failed_energy），四条回合安全 hard counters 均为零，checkpoint SHA 前后不变。该回放与三个 failure seed 复现均不是独立 qualification cohort。coverage fit 的 dev/old-flow 分类指标退化与改善均保留在证据页，不以单一 pooled 指标替代分组结果。run2 bad-CR 软件现场与 formal failure roots 保持 immutable；服务器 prefix exact 检查通过，本地镜像 hash 差异不作为服务器产物校验失败。本轮 target/lifecycle/replay 事实已补齐；dense30 与同负荷随机环数量/安全非均匀角度评估尚未启动，heldout 未消费。

coverage 三条 replay 的 exact 可视化已生成：run0/run1/run2 分别为 `810/810`、`739/739`、`781/781` steps，均为 `24/24`。`step_trace.emergency_departures` 的三 run 计数为 `2/5/5`，与各 run 的 `emergency_landed` outcome 完全对应；全 72 个 development targets 的 `before_ring_role=outer` emergency-origin 计数为 `0`。此前依据 stride-2 frame 的 run2 arrival16 UID134 “末帧为 outer”及 emergency origin `1/0/1` 的说法已撤回：该抽样漏掉了瞬时 middle 状态，不能作为实际 origin 结论，也不构成 outer→inner 直接跳环的证据；备份文件 `replay_report_frame_stride2.json` 保留，最终统计以 full-step 事件为准。四个重点 target 的 full-step 事实为：run0 arrival4 UID4 在 `t=205` 完成 outer→middle、`t=266` 完成 middle→inner，最终 normal；run1 arrival18 UID286 在 `t=448` 完成 outer→middle、`t=558` 完成 middle→inner，最终 normal；run1 arrival20 UID272 在 `t=353` 完成 outer→middle、`t=420` 完成 middle→inner、`t=487` 在 inner 受控 emergency；run2 arrival16 UID134 的 outer→middle 到达发生在 `t=421`，并在 middle 于 `t=421` 受控 emergency，release 发起时间为 `t=357`。全 step exact 诊断 [run2_remaining_outer_diagnostic.json](D:/7_codespace/training_outputs/terminal/p3_gate_coverage_repair_v2_20260925/run2_remaining_outer_diagnostic.json) 同时显示该目标在 `t=357`、SOC `.41377` 获准 O2M，`t=421` pre-step 为 `ring_middle`、SOC `.37669`，终态 SOC `.30926`。六个诊断采样点 `173/213/253/293/333/421` 的 state-only Teacher 标签均为 WAIT 且无 certified RELEASE；这是定点采样，不能推广为整条 trajectory，实际 ring 阶段记录为 `11` 次 safety reject 与 `1` 次 approved，`t=357` 未额外查询 Teacher。priority-weighted wait 为 `131.1688/157.2021/147.9678`，low-SOC wait 样本数为 `10/10/8`、均值为 `111.0077/85.1909/99.9963 s`；原 formal 分母 `23` 与 coverage 的 `24` 不按同一分母比较。HTML 入口为 [coverage run0](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/coverage_run0/recorded_stochastic_policy_replay.html)、[run1](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/coverage_run1/recorded_stochastic_policy_replay.html)、[run2](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/coverage_run2/recorded_stochastic_policy_replay.html)。该 run2 target 实际切入 middle 后完成受控 emergency，不是原先所写的持续外环盘旋耗尽问题；后续 formal PPO 仍需 short pilot。

相关检查为合并 `33 PASS`、既有 runtime `38 PASS`；energy 相关 `12 pass / 4 fail` 可在修复前当前 HEAD worktree 复现。上述 replay 与 coverage fit 均为开发诊断，不是 PPO 或 heldout；定向 gate 修复已完成，本轮不重训全部既有课程、不放宽 SafetyFilter、不增加 SOC 专用规则。`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`。

**P3 V5 原始策略回放可视化（2026-09-25，已完成）：** 三个 restart1 run 已通过入口 `python -m terminal_marl.visualization.p3_formal_replay --run-dir .../runN --output-dir .../runN` 完成原始 stochastic action tape 回放，三个任务均 `exit 0` 且 `exact=true`。输出根为 `/home/user/training_outputs/terminal/p3_formal_policy_visualizations_20260925/`，HTML 与 `replay_report.json` 已同步到本地：

| run | checkpoint / seed | dense24 记录 | steps | HTML 大小 | 本地产物 |
|---|---|---|---:|---:|---|
| run0 | `217` / `850218` | `16 / 7 / 1` | `919/919` | `8,595,877` bytes | [HTML](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/run0/recorded_stochastic_policy_replay.html) · [report](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/run0/replay_report.json) |
| run1 | `31` / `851032` | `15 / 8 / 1` | `1018/1018` | `9,028,121` bytes | [HTML](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/run1/recorded_stochastic_policy_replay.html) · [report](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/run1/replay_report.json) |
| run2 | `89` / `852090` | `18 / 5 / 1` | `989/989` | `9,206,441` bytes | [HTML](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/run2/recorded_stochastic_policy_replay.html) · [report](D:/7_codespace/training_outputs/terminal/p3_formal_policy_visualizations_20260925/run2/replay_report.json) |

表中 dense24 记录顺序为 `normal / emergency_landed / failed_energy`，三个回放各含 `24` requests；不同 seed 不作公平性能比较。该过程使用原 stochastic 轨迹，不重新采样、不训练、不消费新 seed 或 heldout。

**P3 V5 首轮保存失败与一次性恢复例外（2026-09-25）：** 首轮三个 run 的失败属于软件 checkpoint 路径错误，不是性能筛选；首轮 `850001/851001/852001` 训练 seed 和 `853` development seed 已被该失败尝试消费，restart1 对相同 seed 的重复不构成新的独立样本，不追加或隐藏 seed。原 run0/1/2 目录、manifest 和 tapes 保持 immutable；由于没有持久 checkpoint，restart1 不续训首轮更新，也不声称存在可恢复模型。其余未消费 training seeds 与 heldout 不替换。本例外只针对该具体保存故障，不建立未来自动重跑权限；冻结 YAML、admission decision、场景、策略、奖励和安全边界不变。

**P3 V5 admission evidence（2026-09-25）：** entry manifest SHA256 为
`7c3536ac2d3d977866993c2a072a251c6eac480c38955f9cd7b9da6c770ceec5c`，pilot manifest SHA256 为
`4fc0dba78e19a8d34dd78db7ce641ec11f496d4ab87ecbfc059646467b5a651b`；两者位于服务器输出根
`/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/`。
Formal runner 与定向测试实现提交为 `be50cde`；冻结 formal YAML SHA256 为
`e3f88e52a327c64621226631c1575cd48ffec4d93b98d505c561fe55b33b22b1`。当前 decision binding
要求 admission JSON 阶段 `P3_FORMAL_ADMISSION_DECISION_V5`，并同时绑定 E04 initial SHA、pilot
manifest/protocol SHA、actual config snapshots 和 optimizer boundary。实际 formal check-only
preflight 已在服务器通过：decision JSON 为
`/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_admission_decision.json`
（SHA256 `d4f34955bd2230e5f531ab71f451d7eb538fbd6f16cc6bf05387ff0a673d2d8d`），结果为
`/home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_check_only_run0.json`，
CLI 使用 `/home/user/envs/terminal_env/bin/python`、cwd `/home/user/projects/terminal`，exit `0`、
status `CHECK_ONLY_OK`，且 run0 目录未创建；这是首轮正式启动前的 preflight 记录。首轮 run0/1/2 已因 checkpoint 保存路径错误停止；当前 restart1 launch status 为 `FAILED`：run0 PID `2497811`、run1 PID `2497879`、run2 PID `2497946`，各 run `dev_before_training/*/progress.json` 已生成。

学生 rollout summary 已记录 `reward_component_totals` 与 `reward_total`，并验证组件和一致；
累计实际 `time/progress/release/touchdown/failure/teacher/shield/urgency/emergency` 字段，
排除 per-UAV `total` 避免重复计算。`urgency` 仍是 priority 与 SOC 的合并项，不虚构独立
分解；实现提交为 `4d34672`，定向测试 `9 passed`。

**当前 trained dense30 V2 bounded diagnostic（2026-09-24）：** 已完成首个有界训练策略诊断
`842101`：`1173` simulated steps、wall `64.53536491142586 s`，`8 normal / 18 emergency /
4 failed_energy / 0 unresolved`；actor `14877 WAIT / 1055 RELEASE`，`24` approved、`1031`
rejected。最小实际/扫掠间隔为 `80.00345665736643/80.0 m`，`safety_error=null`；collision、NMAC、
hard separation、shield/joint fail-closed、executed-plan hash mismatch 均为 `0`。记录了 `9` 次
shield intervention（correction，不是 failure）和 `1315` 次 safety intervention，不能概括为
“all safety counters=0”。weights unchanged、无 Teacher/optimizer、无训练/PPO；该诊断不是 PASS 或
qualification。`842101` 的 read-only trace aggregation 已完成：4 个 failed-energy UAV 的 threshold 前
RELEASE 为 `3/3/3/9` 且全部 rejected，WAIT 为 `411/305/493/238`；inner RELEASE 的主要观察
blocker 为 `primary_blocker=LZ_RESOURCE=593/615`（`510 emergency_lz_priority + 83 lz_unavailable`），
approved `8`。没有 independent feasible RELEASE witness 支持将 WAIT 全部判错或批准
imitation-only fix。详见[诊断结果](evidence/v4_2/trained_dense30_v2_diagnostic_result_20260924.md)。

**当前 random-SOC P3 slot-repair 结果（2026-09-25）：** 详见[结果页](evidence/v4_2/random_soc_p3_slot_repair_result_20260925.md)。
同一 seed `845101` 的 dense24 三 arm 为：late-OFF forced `23/1/0`、late-ON forced `19/5/0`、
late-ON trained `18/6/0`（normal/emergency/failed-energy）；三者均 `unresolved=0`、terminal、
exit `0`。三 arm 均无 `SafetyInvariantError`，collision/NMAC/hard-separation/fail-closed/
hash-mismatch 均为零，weights unchanged、Teacher/PPO off。`845000` late-ON 两步 smoke 有效；
server `29` tests PASS，local diagnostic `17` 与 slot `12` PASS，Teacher `12` PASS 单独统计。
旧 protocol、2026-09-24 日期与 raw root 保持不变；`P3_QUALIFIED=false`、`PPO_ALLOWED=false`、
`P4_ALLOWED=false`，旧 heldout 保持 untouched。

**当前授权/工作状态（2026-09-25）：** 不采纳 late-ON，默认 late-slot OFF 保持。外置
state-only labeler 已由 `768b191` 实现，gate_adapter-only 监督训练器已由 `b9f1077` 实现，
对应 data protocol 为 `56b4c3`；proposal head/encoder 冻结。服务器 smoke `846000` 已以 exit `0`
完成，`18` 条 actor labels 为 `7 WAIT + 11 RELEASE`，全部为 finite 88D 且 schema 映射正确，
weights unchanged；服务器指定的 `5` 个模块测试为 `39 PASS`，本地 `44` 个测试属于另一组，
不合并为同一测试总数。完整 `7` 回合 collection 已完成，输出根为
`/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925`；详见
[collection/fit result](evidence/v4_2/random_soc_gate_repair_v1_result_20260925.md)。启动 wrapper 曾因
reset 前预建 output 被拒绝，归档为 `launch_failure_train_flow6_846101`；修正独立 `_control` 日志后，
首个实际 `846101` 回合已运行，不称为重复 seed 实验。授权的 gate-adapter-only fit 已完成 `35/35`，
artifact SHA256 为 `e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`，train/dev
rows 为 `1529/812`；dev RELEASE/WAIT recall 为 `.6131687243→.7860082305` /
`.8119507909→.7768014060`，macro-F1 `.7092638611→.7562677855`，NLL
`1.265022278→1.031340837`，train NLL `1.259755254→.755264521`。仅 adapter 改变，
non-adapter/proposal logits bitwise unchanged；strict load passed、无 optimizer state、无 dev gradient/
early stop/PPO。wrapper 最后有 CRLF shell exit error，但 Python 已完成且 control 保存 Python exit `0`，
未重跑。`846301/846302/846303` paired rollout 已完成且三者 exit `0`；三场合计 `42` requests，
source 为 `36 normal + 6 emergency + 0 failed_energy`，candidate 为 `41 normal + 1 emergency +
0 failed_energy`。详见[collection/fit/pair result](evidence/v4_2/random_soc_gate_repair_v1_result_20260925.md)。
六个 arm 的 replay、safety、Teacher off、no updates、weights unchanged 均为 true；dense24
唯一 false arm gate 为 `no_failed_outcomes=false`，因此不称 P3 通过。整体 lifecycle aggregate 为
source priority-weighted wait `150.6971849500`、low-SOC wait `53.7159174128`（count `12`），
candidate 分别为 `80.1562018040`、`21.4881016766`（count `12`）；selection gates 全部 true。
exact emergency replay 与当前 `32` 个独立单步 shadow checks 已记录：均绑定 exact `795`-step
replay、state unchanged、无 safety error、approved `0`，其中 `safety_wait=16`、
`final_capacity=10`、`safety_reject=6`。这只说明抽查状态未获批，不宣称所有 WAIT 错误或 SOC
排序有效。`847101/847102` arbitration paired development 已完成，baseline 与 soc-priority
各自为 `41 normal + 7 emergency + 0 failed_energy + 0 unresolved`（48 requests）。两 pair 的
安全/完整性检查均通过，唯一 false arm gate 为 `no_failed_outcomes=false`；soc-priority 两 pair
均降低 low-SOC wait 但提高 priority-weighted wait，因此不采纳，默认 `downstream_priority`
保持。详见[arbitration result](evidence/v4_2/random_soc_arbitration_repair_v1_result_20260925.md)。
本轮 repair 来源是 `714d141f...`；历史 P3 canonical initial 是独立的
`4b8be66c...`，不静默替换基线。canonical strict loader、只读 preflight 与 `848000/848101/848102`
smoke/full rollout 已完成；pair 结果与冻结 E04 初始化来源见[受控应急 P3 V5 协议](evidence/v4_2/p3_controlled_emergency_admission_v5_20260925.md)。旧
`829001–829006` 保持 untouched。
`P3_QUALIFIED=false`、`PPO_ALLOWED=false`、`P4_ALLOWED=false`，不放宽既有安全与 P3 gate。

**P3 random-SOC scope（沿用现有 P3 V4 语义）：** initial SOC 为 `.55–.90` uniform，priority 为
`1/3/5`，场景覆盖 `flow6`、`flow12` 和 `dense24`。dense30 pressure 诊断不能单独构成 P3
qualification；后续资格证据仍需 train/dev/heldout 独立性。现有 P3 hard gates 保持原协议口径：
dev/eval completion `1.0`、safety、exact replay、K3 shortlist、low-SOC wait non-worse 和
priority-weighted wait improvement；本轮不新增 completion threshold。预留 `829001–829006` heldout
保持 untouched，不改作本轮开发集。

> **当前阶段更新（2026-09-24）**：固定规则协调流已完成三组 development
> regression（`841301/841501/841502`），均为 `30/30` resolved、`failed_energy=0`，
> normal/emergency 分别为 `29/1`、`29/1`、`28/2`；最小 touchdown SOC 为
> `.301878/.285780/.278703`，最小实际/扫掠间隔均达到 `80 m`，命名安全计数为零。
> 这是 opt-in fixed-rule development evidence，默认旧 route 保持不变，不构成 I1 或 PPO
> qualification。deadline-fallback bounded fresh check 的 `841701/841702` 均已完成并通过硬门槛，
> 两个 fresh case 合计 `60` UAV、`57 normal / 3 emergency / 0 failed-energy`；与三份 development
> regression 合计五场、`150` UAV、`143 normal / 7 emergency / 0 failed-energy`。此前 V7 fresh gate
> 的 FAILED 历史仍保留。`PPO_ALLOWED=false`。

**控制器身份澄清（2026-09-24）：** 上述 `fixed_rule_dense30_coordinated` 回放以及五场、
`150` UAV、`143 normal / 7 emergency / 0 failed-energy` 证据均使用规则控制器，不使用训练的
神经低层 checkpoint；启动时会打印所选模式、低层控制器类型和实际使用的 checkpoint。训练版
V4.2 回放位于 `ppo_checkpoint_viewer`，受冻结 profile/hash 约束；本文不宣称
`dense30_coordinated` 已通过训练策略回放验证。上述历史回放的 `min_landing_soc=.25` 是工程假设，
其文献依据是条件性的，尚未核验为 EH216-S AFM 的限制；见
[landing reserve basis](evidence/v4_2/landing_reserve_basis_20260924.md)。

**能量口径更新（2026-09-24，后续诊断适用）：** 归一化容量保持 `1500`，车辆 profile
profile 已更新为 `eh216s_research_v2`，`min_landing_soc=.10`；`soc_reserve=.25` 单独保留为
queue/admission reserve。其它安全、电量和运行阈值不变。已有 `.25` floor 的实验与 checkpoint
均保留为历史证据，不对 `.10` floor 作性能或 qualification 声明。本轮仅运行
profile/energy/checkpoint metadata 边界相关定向单测，全部通过；没有 fixed-rule 回合复跑或训练，
旧统计不更新为 `.10`。

**最新集成 fixed-rule development 回归（2026-09-24）：** 使用
`fixed_rule_dense30_coordinated` CLI，配置为 30 架 warm-start、初始 SOC `.55–.90`、route
guard on、`continuous_inward_v1`、`downstream_priority`、late-slot off，并启用 opt-in
`rule_flow_coordination_enabled=true`。三份 raw record 均 `30/30` resolved、
`failed_energy=0`、`safety_error=null`；collision、NMAC、hard separation、shield/joint
safety fail-closed、executed-plan hash mismatch 等命名计数均为零。

| seed | sim s | wall s | normal / emergency / failed-energy | min touchdown SOC | min actual / swept m |
|---:|---:|---:|---:|---:|---:|
| 841301 | 757 | 264.76303469995037 | 29 / 1 / 0 | 0.3018781902911374 | 80.1532184002751 / 80.1532184002751 |
| 841501 | 783 | 217.04460870008916 | 29 / 1 / 0 | 0.2857804017484627 | 80.0 / 80.0 |
| 841502 | 840 | 255.41621090006083 | 28 / 2 / 0 | 0.2787034160630745 | 80.0 / 80.0 |

原始 `841301` baseline 为 `18 normal / 8 emergency / 4 failed-energy`；协调回归为
`29 / 1 / 0`。在协调回归中，UID3 的 M→I commit/进入 I 环/normal touchdown 为
`140/236/299 s`，UID4 为 `168/252/327 s`；commit 时刻与进入 I 环时刻不同。
此前 baseline 的三个观察点保持为：早期 O 环候选主要出现
`dynamics_unreachable_merge_time`，552 s 时 UID7/23 在两个空闲 pad 下仍得到
`emergency_lz_priority` 且 `runtime_shortlist_candidates_checked=0`，以及旧的
emergency-before-normal 与 forced energy guard 可在 SOC `.451690/.474427` 触发应急。
后续 inner-queue guard 属于独立的新观察，由 deadline fallback 处理，不改写上述 baseline
事实。

`flow_repair_regression_delivery.txt` 记录 `241 passed`、`2` 个未改变的 frozen-config
source-binding failures（`terminal_marl/core/config.py`）；focused repair subset 的 `82`
项不并入该计数。`validate_plan` 的 `100`-step 等价测量为 scalar `45.319964 s` 到
vectorized `36.037378 s`，仅代表函数/prefix，不能外推为完整 GUI 或整体吞吐 wall-time
加速；三次 development wall time 也只作绝对记录。

deadline-fallback fresh check 已完成 bounded hard gates：`841701` 为
`30 normal / 0 emergency / 0 failed-energy`、`30/30` resolved、863 s、最低 touchdown
SOC `.3451866507009634`、最小实际/扫掠 `80/80 m`；`841702` 为
`27 normal / 3 emergency / 0 failed-energy`、`30/30` resolved、789 s、最低 touchdown
SOC `.2919970856931216`、最小实际/扫掠 `80/80 m`。两份 fresh 记录合计 `60` UAV、
`57 normal / 3 emergency / 0 failed-energy`，最低 fresh touchdown SOC
`.2919970856931216`；连同三份 development regression 的五场总计为 `150` UAV、
`143 normal / 7 emergency / 0 failed-energy`，整体最低 touchdown SOC
`.2787034160630745`，命名物理/安全错误计数均为零。该 bounded check 为 **PASS**，但此前
fresh V7 paired gate 的 **FAILED** 事实保留，当前不进行正式 I1/PPO qualification。

协调 30-UAV 可视化的 PowerShell 命令为：

```powershell
.\venv\Scripts\python.exe -m terminal_marl.visualization.rollout_viewer --mode fixed_rule --scenario fixed_rule_dense30_coordinated --seed 841301 --gui --labels --trajectories --playback-speed 1 --max-simulation-seconds 1800
```

**历史 baseline 可视化回放观测（2026-09-24）：** current-code 精确重放 seed `841301`、30 架全 warm、初始
SOC `.55–.90`、`continuous_inward_v1`、route guard on、`downstream_priority`、late-slot off，使用
`env.step()` 得到 `1364` steps、`18 completed + 8 emergency + 4 failed-energy`，与此前结果一致。
`t=813 s` 时 6 个 LZ 均为 free；UID `4/11/12` 均处于 inner `R=200, z=40`、speed `16` 且
`committed_plan=None`。该回放没有返环让出降落点轨迹证据；`emergency_priority` 只作用于尚未放行的
request，不撤销已经 committed 的 path。完整本地 trace 为 `results/viewer_trace_841301_full.json`
（约 149 MB，未提交）。813 s 的局部只读 probe 进一步显示 UID12 的 6 条 emergency candidate
全部被 `predicted_holder_reachable_tube` 拒绝；即使其他 holder 固定为 nominal 速度，UID4 nominal
swept 最小间隔按 pad `3/4/5/0/1/2` 仍为 `15.040/17.248/17.210/63.639/39.236/20.469 m`，
全部低于 `80 m`。该 probe 未执行候选、未放宽阈值，不能把该时刻归结为 tube 约束过严；当前候选
本身与后方 UID4 的环飞存在冲突。probe 文件为 `results/viewer_trace_841301_probe813.json`。

同一 trace 的 failed-energy 记录为 `UID3@883 outer`、`UID4@991 inner`、`UID11@992 inner`、
`UID12@1363 inner`，reason 均为 `energy_depleted`；`_resolve` 的 `self.now` 是处理 tick 起点，
因此下一 snapshot 时间为 `+1 s`，不把记录解释为 substep 精确瞬间。full vertical execution
trace 显示 `26/26` 个成功 landing plan 均实际 touchdown、`z=0`，pad XY 最大误差 `0`，每架
有 `15–36` 个真实竖降控制步样本且高度单调下降；vertical duration 为 `14.696–34.834 s`，
内环正常 final 约从 `40 m` staging 下降，外/中环 emergency 可从 `120 m` staging 下降。4 架
failed-energy 仍在环飞阶段耗尽，不能与成功落地后退出 active 混淆。UID12 在 `t=991/993` 的
剩余能量为 `372.1257/370.1257`、SOC 为 `24.808%/24.675%`，已低于 `375` reserve，按同一
约束不能通过降落 energy gate。viewer 修复范围已覆盖低 SOC 橙色、应急红色、failed-energy 持久
事件/标记、LZ 状态、顺序 source frame、长间隔禁用插值和正常结束保留最终画面；headless 不等待。
UI 联合回归为 `67 passed`，但 failed-energy 本身尚未修复，`PPO_ALLOWED=false` 保持不变。详见
[viewer flight diagnosis](evidence/v4_2/viewer_flight_diagnosis_20260924.md)。

本轮 PyVista/VTK 离屏 viewer 检查使用 PyVista `0.48.4`、VTK `9.6.2`：30 点云保持
30 个 vertex cells，真实 picker 可选中非零 UID `4/11/29` 并在收缩后保持映射，标签 actor
不可拾取；轨迹线宽为普通 `1.5`、选中 `3.0`，轨迹头跟随插值位置。30 Hz 的 position-only
更新复用了标签缓存和轨迹 mesh，单次中间帧只调用一次 plotter update。八个既有/专项测试文件
本轮合计 `35 passed`。临时 30 UAV 离屏渲染基准（预热 `10` 帧、计时 `60` 帧）记录 before
`3.245240299962461 s`、after `0.6678959000855684 s`；这是 offscreen render-only 测量，
不是仿真或 runtime FPS，也不据此宣称完整 GUI 或整体性能提升。
本轮未改变仿真控制器、能量阈值或轨迹执行语义；rollout 仅新增启动时 controller identity 输出。

**2026-09-24 综合结论（密度、late-slot 与性能）：** fresh fixed-rule、SOC `.75` 的六个密度
cell 已完整结束：12 机为 `24/24 normal`，20 机为 `37 normal + 3 emergency / 40`，30 机为
`46 normal + 14 emergency / 60`；六个 cell 均 `failed_energy=0`，并且
`safety_error`、`hard_separation_violations`、NMAC、collision、`shield_fail_closed`、
`joint_safety_fail_closed` 均为零。这些是 fixed-rule exploratory 结果，不建立普适空域容量上限，
也不建立零失败保证。当前证据显示静态环容量不等于 deadline service capacity；在该混合需求设置中，
输入间隔 `8 s` 小于 final headway `12 s`，若持续则会形成持续过载；此前 mixed 场景的有限 `18` 架 burst 只观察到
transient accumulation，不构成有限批次不可行定理。已有 future fallback energy budget
不是 future trajectory safety certificate；在没有经过完整认证的可行替代路径前，不能据此宣称
SafetyFilter 安全阈值过严。详见 [density efficiency evidence](evidence/v4_2/density_efficiency_20260924.md)。

原始 `.55–.90` fresh late-slot 配对的 off/on 聚合分别为 `36 normal + 19 emergency + 5 failed-energy`
与 `32 normal + 24 emergency + 4 failed-energy`；on 模式的 normal completion 下降、emergency 上升，
且 `zero_energy_failure` 仍未通过，因此 late-slot 变体未晋级，默认保持
`reconcile_late_merge_slots_enabled=false`。性能链四个 100-step prefix 的同语义比较为
`28.607→23.053 s` 与 `28.875→23.501 s`，即 `19.4%/18.6%` wall-time 降低；它们仍是未完成
`29` 架请求的 prefix diagnostic，不能作为 qualification 或容量结论，实际 GUI 也尚未视觉验收。
详见 [trajectory-chain performance evidence](evidence/v4_2/trajectory_chain_performance_20260924.md)。

singleton target-gap 修复已将同 holder point 的零 gap 改为 antipode gap；`35` 个 singleton
定向测试和本轮 combined regression 的 `103 passed` 均已完成。fresh `.55–.90` paired full cells
中，seed `841301` 的 fixed/legacy 分别为 `18 normal + 8 emergency + 4 failed-energy` 与
`12/13/5`，seed `841302` 分别为 `14/9/7` 与 `14/10/6`；聚合为 fixed
`32/17/11`、legacy `26/23/11`。该结果是描述性 paired evidence：保留几何正确性修复，aggregate
failed-energy 未减少且一个 seed 结果更差，不能宣称 zero-energy 已解决或进入 qualification。
该四-cell 证据使用 SOC `.55–.90`，不把固定 SOC `.75` density baseline 作为 singleton 修复验证。
详见 [single-holder gap evidence](evidence/v4_2/single_holder_gap_20260924.md)。保持
`PPO_ALLOWED=false`。

**最新前置观测（2026-09-24，runtime timing reconciliation）：** 只读 timing audit 显示，all-warm
seed `840626` 前 `200` steps 有 `21` 次 approved ring transfer，其中 `20/21` 的 `|end-slot|` 超过
`1 s`，median `-8.379 s`、minimum `-49.229 s`；mixed seed `840627` 前 `200` steps 为 `20/20`
mismatch，median `-8.661 s`、minimum `-41.292 s`。mixed actual arrival adjacent gaps 中，
ring_middle 最小 `1.97 s`（`5` 个低于 `9 s`），ring_inner 最小 `8.29 s`（`4` 个低于 `9 s`）；
prefix safety events 为 `0`。对应 timing audit 仍属于前置观测，不是 qualification。

product viewer 20-step offscreen benchmark 为 headless `2.918 steps/s`、process `2.820 steps/s`，
排除 `1.63 s` startup 后 slowdown `3.37%`，`12.01 fps`，p95 gap `0.192 s`，测试 `22 passed`；
该结果不能替代 GUI 主观验收。CPU profile 的 20-step 总耗时为 `13.18 s`，`ring_transfer` cumulative
`5.912 s`、runtime protect `3.086 s`、SafetyFilter `1.890 s`；profile 调用嵌套，数值不可相加。
timing reconciliation fix 已完成并通过 focused integration 与 actual slot audit；`PPO_ALLOWED=false`。

**最新 merge-slot reconciliation 回归（2026-09-24）：** `MergeScheduler` 现将 ring-transfer slot
对齐到 `plan.end_time`，将 final slot 对齐到已验证的 `scheduled_departure`；integration tests 为
`158 passed, 1 skipped`。mixed30 前 `200` steps 的 actual slot audit 为 `29/29` exact delta `0`，
o2m/m2i/i2f 最小事件间隔 `9/9/12 s`，`0` 个低于 headway，prefix safety events 为 `0`。
all-warm `840626` full-cycle 为 `17 normal + 10 emergency + 3 failed-energy`（step `1067`，
最小实际/扫掠 `81.578/81.550 m`，壁钟 `64.925 s`，`47` approvals）；mixed `840627` 为
`18 normal + 12 emergency + 0 failed-energy`（step `910`，最小实际/扫掠 `80.00062/80.00062 m`，
壁钟 `42.690 s`，`61` approvals）；两者 safety events 均为 `0`。相较此前 all-warm `20/8/2`、
mixed `23/6/1`，该修正改善了 slot timing 的安全与一致性观测，但 throughput/emergency gate 仍未
通过，不能作为 qualification 或训练授权；`PPO_ALLOWED=false`。SOC arbitration pilot 已完成，
但 `zero_energy_failure` 未通过，未晋级默认模式。

**最新 SOC-priority opt-in pilot（2026-09-24）：** `downstream_soc_priority` 保持 downstream ring
precedence，并在同一优先级内按自身 SOC 升序排序；不改变 observation 维度，默认仍为
`downstream_priority`。前 `200` steps 的 all-warm `840626` approvals `23→19`、resolved `5→4`、
墙钟 `26.423→27.693 s`；mixed `840627` approvals `29→28`、resolved `5→5`、墙钟
`13.841→13.420 s`。mixed full-cycle 中 default 为 `18 normal/12 emergency/0 failed`（step `910`，
`42.690 s`），SOC priority 为 `20/9/1`（step `1043`，`43.70 s`）；失败 arrival `17` 为
`UID71`、`ring_middle`、`t=1042`，两种模式 safety events 均为 `0`。该 pilot 因
`zero_energy_failure` hard gate FAIL，未晋级且不改变默认模式；没有 full all-warm SOC pilot 结论，
`PPO_ALLOWED=false`。counterfactual emergency contract probe 已完成，单状态结果见下方。

**最新 nominal-holder counterfactual（2026-09-24）：** 只读 current-code seed `840626` 的 `UID13`
在 `t=200 s`、`ring_outer`、SOC `0.4595651` 状态中，6 条 emergency routes energy `6/6`、
SafetyFilter `0/6`；唯一 free compatible `LZ4` 为 energy `1/1`、SafetyFilter `0/1`，原因计数为
`predicted_holder_reachable_tube=3`、`predicted_separation=3`。将 in-memory nominal-holder exact
trajectory 同时用于 lower/upper tube bounds 后，全部结果和原因计数保持相同。该结果只说明单状态下
nominal speed promise 单独不足以改变 verdict，不构成一般不可能性证明；未放宽运行时安全阈值、未修改
source，`PPO_ALLOWED=false`。

**最新诊断（2026-09-24，dense30 continuous final/LZ exploratory）：** 为检查 30 机潜在冲突场景，新增
opt-in `final_approach_geometry_mode=continuous_inward_v1`。该模式使用连续切向脱环的共享几何，保留
正常 final-approach 语义，并由中心扫描按兼容性排序的空闲 LZ；SafetyFilter `80/40/10 m` 保持不变。
默认 `brake_bezier_v1` 未改变。集成定向测试为 `121 passed`。在 seed `840626`、30 架全 warm、
high-SOC 的 100 s 探索中，该模式获得 `13` 次 approval、完成 `1` 架 normal，emergency、failed-energy
和 safety 均为 `0`；最小实际间隔 `82.03085 m`，服务器墙钟时间 `13.4638 s`（旧模式同条件约
`16.47 s`），但仍未完成完整回合。延长同一模式至终态后，运行在 `t=1117` 结束（服务器墙钟
`51.9834 s`）：`20` 架 normal、`8` 架 emergency、`2` 架 failed-energy、`0` 架 unresolved；失败目标
为 outer UAV `UID13@889 s` 与 `UID14@1116 s`。请求记账为真，累计 `51` 次 approval、`304` 次
emergency attempt/`8` 次 commit；最小实际/扫掠间隔为 `80.0331/80.0226 m`，hard-separation、
NMAC、collision 和 shield fail-closed 均为 `0`。该 full-cycle 结果未通过 `zero_energy_failure`
hard gate，不能进入课程或 PPO，`PPO_ALLOWED=false`。

本轮 Windows 早期 benchmark 的 `2.21 steps/s` 与服务器完整回合的 `21.49 steps/s` 不可直接比较：
后者后段 active UAV 已减少，状态计算量较轻，不能作为 30 架并发实时速率。实际 GUI 尚未完成视觉
验收。UID 级 emergency/release trace 已补齐；下一步前置是量化 merge-slot shortfall，完成安全规则/
应急路线修正，再冻结 fresh normal/stress protocol；在这些前置条件完成前不推进低层课程或 PPO。
详见 [dense30 低层预训练前置诊断](evidence/v4_2/dense30_pretraining_20260924.md)。

**最新归因补充（2026-09-24，dense30 UID trace）：** 同一 seed、同一 continuous 配置的完整回放与原始运行
保持 exact outcome parity：`1117` steps、`20` normal、`8` emergency、`2` failed-energy。`UID13` 首次
emergency 为 `t=290, SOC=.399565`，有 `120` 次实际尝试和 `14` 次 emergency plan commit 尝试，
其中 `8` 次被 holder tube、`6` 次被 predicted separation 拒绝；`UID14` 首次 emergency 为
`t=495, SOC=.414041`，有 `125` 次实际尝试和 `11` 次计划尝试，其中 `9` 次被 holder tube、`2` 次被
predicted separation 拒绝。两条失败终点均有 `6` 个空闲 LZ；UID13 的 `t=200/290` 和 UID14 的
`t=400/495` 窗口中，6 条 emergency routes 均 energy `6/6` 通过但 SafetyFilter `0/6` 通过，
能量 slack 为正（`25.556–177.007`）。因此当前静态 dense30 的 zero-failure gate 明确失败，
不能归结为 LZ 数量不足，也不支持放宽安全阈值。正常 25-proposal 扫描在 `t=100/200` 为
`dynamics_unreachable_merge_time` 全拒，在 `t=500/750` 为 `emergency_energy_guard` 全拒。
相关 forensic artifacts 见 [dense30 证据页](evidence/v4_2/dense30_pretraining_20260924.md)。

**最新 merge-slot 诊断（2026-09-24）：** seed `840626` 的 outer `UID13/UID14` 在 `t=100/200`
各扫描 `63` 个候选（speed residual `-1/0/1`、merge residual `-1/0/1`、delay `0/10/20/40/80/120/180 s`）。
`t=100` 两架均为 `0/63` preliminary certified，`t=200` 分别为 `13/63` 和 `21/63`；但最佳
planned endpoints 比请求 merge slots 提前约 `95–97 s`，因此不能作为可执行安全合流时隙证书。
严格 `<=1 s` alignment search 的 synthetic candidate 通过，但 12 机回归 `840620` 为
`1 normal + 8 emergency + 3 failed_energy`，`840622` 为 `1 normal + 10 emergency + 1 failed_energy`；
该改动已完全回滚，energy guard 定向测试 `8 passed`。详情见 [dense30 证据页](evidence/v4_2/dense30_pretraining_20260924.md)。

**最新 mixed30 探索（2026-09-24）：** seed `840627` 使用 `12` 架 warm-start 加 `18` 架每 `8 s`
到达、SOC `.55–.90`、continuous geometry 和 route guard，运行 `1032` terminal steps、服务器墙钟
`43.3962 s`。结果为 `23 normal + 6 emergency + 1 failed_energy`，`30/30` resolved，`68` 次 approval，
`174/6` 次 emergency attempt/commit，route checks `615`（其中 negative `143`），最小实际/扫掠间隔
`81.2511/81.2478 m`，safety events `0`；唯一 failure 为 arrival `17` 的 outer `UID44@t=1031`。
该回合 `20%` emergency 和 `1` 条 energy failure 使训练 gate 失败，不构成 qualification 或问题已解决
的证据。第二次 20-step 离屏测量为 headless `3.026 steps/s`、async viewer `2.883 steps/s`，开销约
`4.8%`，插值帧 `24`（前次 `3`）；GUI subjective QA 仍不可用。详见 [dense30 证据页](evidence/v4_2/dense30_pretraining_20260924.md)。

**最新 mixed30 UID44 归因（2026-09-24）：** UID44 exact replay 与 mixed30 原始回合一致：`1032`
terminal steps，`23 normal + 6 emergency + 1 failed-energy`，peak active `21`，peak ring
outer/middle/inner `9/4/4`；arrival `17` 的 outer UID44 在 `t=1031` 失败。首个 tracked inbound 为
`t=70, SOC=.63691`，outer 状态为 `t=130, SOC=.600799`，首次 emergency 为 `t=145, SOC=.590799`。
在 `t=130/145`，6 条路线均 energy `6/6` 通过但 SafetyFilter `0/6` 通过，energy slack 约
`+313–386`，normal proposals `25/25` 全拒；`t=750` 为 SafetyFilter `3/6`、energy `0/6`，
`t=900` 为 SafetyFilter `6/6`、energy `0/6`。`t=70`/`t=100` 仍是 inbound，未进行环上 emergency
scan，不能据此推断已经执行 admission check。早期 future-exit certification 与 coordinated
holder/pad traffic plan 仍是 proposed prerequisites，尚未实现或证明；`PPO_ALLOWED=false`。

**最新诊断（2026-09-24，dense30 低层前置与性能）：** 为覆盖 30 机潜在冲突场景，fresh
high-SOC all-warm seed `840626` 在 route guard、ring `600/400/200 m`、SOC `.55–.90`、
K1 `[1,0,0]` 下运行 `100 s`，`0` approved、`0` resolved、`30` unresolved，`3000` 次
release rejection；观察到 `0 failed_energy` 和 `0` hard-separation/NMAC/collision，但
该 horizon 未完成完整回合。25-proposal catalogue 在 `t=0/t=50` 均为 `750/750` rejected；
扩大到 `800/550/300 m` 后 100 s 仍为 `0` approvals。Windows 每步耗时已由 `1.06462` 降至
`0.451737 s`（约 `2.21 steps/s`）；异步离屏 stride1 检查为 headless `2.3685`、viewer
`2.297 steps/s`，约 `3%` 开销且无 queue backlog，实际 GUI 流畅性尚未由此证明。12 机
same-seed 回归 outcome/accounting 不变，高 SOC `12/12` normal，近下限 `6` normal+`6`
emergency，均无 failure/safety event。详见 [dense30 低层预训练前置诊断](evidence/v4_2/dense30_pretraining_20260924.md)。

**历史 exploratory 状态（2026-09-24，emergency reachability guard）：** fresh stress seeds
`840607/840611/840612/840613` 的 old/route 配对每对终态相同；两种模式分别聚合为
`13 completed + 27 emergency_landed + 8 upstream diverted_low_soc = 48`，均
`failed_energy=0`、安全事件为0。route-aware checks 非零，但没有改善这些终态。正常高 SOC
`.55–.90` 的 `840620/840621` 合计为 `23 completed + 1 emergency_landed`；近下限
`.41–.55` 的 `840622/840623` 合计为 `10 completed + 13 emergency_landed + 1 upstream
diverted_low_soc`；两组均 `failed_energy=0`、安全事件为0。全部记录为 exploratory，未通过
exact replay/full qualification，不能据此启动 PPO。V2.3 旧 source-bound observation-only
证据独立保留。低层 `12` 机课程与 `30` 机离线回放的先后顺序，以及是否先重构 `30` 机安全层
实时性能，当时仍待决定；现由上方最新的 [dense30 低层预训练前置诊断](evidence/v4_2/dense30_pretraining_20260924.md) 结论覆盖；当前不宣称可正式 PPO。

**最新实现补充（2026-09-23，source-bound Observation V2.3）：** 已加入保持 88D 宽度的
opt-in `phaseA_low_obs_v2_3_source_bound_88d`。自身状态、有限
`neighbor_comm_range_m` 内的 5 个邻机槽，以及由起降场中心广播的环/队列状态、紧急未分配
数量和兼容降落点释放 ETA 已接入；不增加观测维度，V2.1/V2.2 保持原 schema。当前实现采用
零延迟、无丢失的理想通信假设，后续仍需鲁棒性验证。新旧 observation/audit 定向测试共
`28 passed`；该实现说明不改变既有 evidence，`PPO_ALLOWED=false` 保持不变。

**最新实现补充（2026-09-23，warm-start 能量前置检查）：** 已加入默认关闭的
`warm_start_energy_reachability_check_enabled`。启用后，warm-start reset 会在每架 UAV
进入 `active` 前计算其当前 ring 到兼容 LZ 的最低单机可执行应急路线 SOC 要求；采样 SOC
低于该要求时直接报出 uid、ring、初始 SOC 和要求 SOC，不裁剪、不重采样，也不把 UAV
分流为上游请求。`.35` OUTER 样本被拒绝，`.41` 的三环小样本通过；默认关闭时历史
warm-start cohort 语义保持不变。这只是单机 reset 前置条件，不提供拥堵、等待或时间耦合
保证。独立 candidate energy/LZ budget 模块已覆盖 candidate 能耗、LZ release ETA、ring
hold 等待、应急 fallback route、landing reserve 与 margin；截至 2026-09-23，该模块尚未
接入 admission/release gate；截至 2026-09-24，admission pad ETA 与 candidate
route/等待/reserve/margin budget 已在 opt-in exploratory 链中接入。fallback future route
仍未作未来 SafetyFilter 认证。warm-start、budget、emergency 定向测试共
`24 passed`；`PPO_ALLOWED=false` 保持不变。

**最新结论（2026-09-23，emergency-priority guard V2）：** fresh `840501/840502` × rule/original K3 × guard off/on 的 `8/8` 条 formal records 均通过终态、请求记账、exact replay、安全计数为零和 Teacher-free；聚合为 `33 normal + 33 emergency + 24 upstream-divert + 6 failed_energy`，guard-on/off 分别为 `4/2` 次 failed energy，hard gate 失败，`PPO_ALLOWED=false`。只读 evidence 中，guard-on 的 K3 失败目标多次触发 `emergency_energy_guard`，但最后窗口 6 个 LZ 均 free；guard-off 的一个失败目标因长时间 `safety_wait`。V2 不晋级，guard 默认关闭。该 formal 运行使用的历史 `_warm_start()` 直接注入 `.35–.55` SOC；当前已加入默认关闭的单机可达性前置检查，详见上方实现补充。下一步是待实现的三段修复：计划承诺前的可达能量/时间预算与上游分流；有 abort/hover/返环证书、足够 SOC、原子切换和释放 LZ 条件时才允许安全抢占；规则/安全层通过后扩展共享 policy 的 deadline、pad 时隙和自身可达能量观测并加入非法动作 mask，区分 normal/stress、保持 `failed_energy=0` 硬门槛和 `PPO_ALLOWED=false`。详见 [Emergency-priority guard V2](evidence/v4_2/emergency_priority_guard_pilot_v2_20260923.md)。

**最新结论（2026-09-23，emergency-priority guard V1）：** opt-in guard V1 默认关闭。fresh `840401/840402` × rule/original K3 × guard off/on 的 `8/8` 条 formal records 均通过终态、请求记账、exact replay、安全、Teacher-free、无 labels/updates 和 K=3 检查；聚合为 `28 normal + 51 emergency + 16 upstream-divert + 1 failed_energy`，guard-on `zero_failed_energy` gate 失败，`PPO_ALLOWED=false`。rule `840401` 的 witness 显示固定 `.40` 能量投影切环硬阻断可能造成低 SOC UAV 长时间等待：guard-on 的 `o2m` release 连续 `647` 次被 `emergency_energy_guard` 拒绝并在 `t=647 s` 电量耗尽，而 guard-off 同一 UAV 在 `t=0` 获准切环并于 `t=274 s` 应急降落。Guard V1 不晋级，不宣称预留 LZ 或 PPO 能保证零失败；下一步只允许移除该硬阻断，设计带可达能量预算的应急替代路径，并以全新 seeds 验证，不能重跑 `840401/840402`。详见 [Emergency-priority guard V1](evidence/v4_2/emergency_priority_guard_pilot_v1_20260923.md)。

## 历史一句话状态（旧课程记录）

**最新实现状态（2026-09-23，应急路线 repair）：** 应急路线已由源环有限刹停改为从真实初始位置/速度连续切向脱环的 cubic 路径；OUTER/MIDDLE 进入 `120 m` staging，INNER 保持当前高度，再经 LZ 上方垂直下降，仍使用原 SafetyFilter 和 `80/40/10 m` 契约。第一 repair 协议（`e99cb...`）在 fresh `840101/840102` 上为 `7 normal + 14 emergency + 2 upstream-divert + 1 failed_energy`，安全事件为0，因此 FAIL；同 seed forensic replay 已使目标 UAV 在 `t=118 s` 应急提交并落地。route-repair 协议（`8c13dc...`）在 fresh `840201/840202` 上为 `9 normal + 11 emergency + 4 upstream-divert`，`failed_energy=0` 且安全事件为0，但结果仍是 rule-only 诊断证据，不构成 qualification 或真实 EH216-S 标定；当前 phase_motion_v2 未显式设置初始 SOC 范围时 default uniform floor 为 `.4106`，显式 `.35–.55` 仍保留为 stress；`PPO_ALLOWED=false`。详见[应急迫降与分阶段能耗 V2](evidence/v4_2/emergency_energy_v2_20260923.md)。

**最新结论（2026-09-23，SOC 优先级共享策略配对诊断）：** 在同一 `phase_motion_v2`、应急阈值 `.40`、`80/40/10 m` 安全契约和显式初始 SOC `.35–.55` 下，fresh seeds `840301/840302` 的 rule/original K3 四条记录均完成终态、请求记账、exact replay、K=3 shortlist、Teacher-free 和安全检查；安全事件总数为 `0`。合计为 `7 normal + 32 emergency + 6 upstream low-SOC diverted + 3 failed_energy`，因此唯一失败 hard gate 为 `zero_failed_energy`，aggregate `PASS=false`。seed `840301` 的全请求延误为 rule/original K3 `426.25/332.33 s`，seed `840302` 为 `326.17/328.92 s`；两颗种子仅作描述性配对证据，不能据此宣称 SOC 优先级已闭合或判断 failed-energy 根因。详细结果见 [SOC 优先级配对诊断](evidence/v4_2/emergency_soc_priority_paired_v1_20260923.md)。`PPO_ALLOWED=false`。

**最新归因（2026-09-23，immutable replay）：** `failure_trace_evidence.json` 的三个失败目标回放均 exact：rule `840301` arrival `2` 为 `609/609` steps，rule `840302` arrival `11` 为 `742/742`，original K3 `840302` arrival `9` 为 `751/751`；文件 SHA256 为 `a17820aa59bbbece91cc96d066981c20e4b42ba57e2edc317be23bec24da1a57`，witness 文件 SHA256 为 `1a93a063db4982946532c26516629e98ff0224c04bcfee51c112290642a03844`，且未新增 seeds。三者第一次 `SOC<=.40` 时刻为 `t=1/142/143`，均在 committed plan 中；当前 `_try_emergency_landing` 只考虑无 committed plan 的 ring holder。其后首个无 committed plan 状态分别是 rule `840301` 的 `t=70, SOC=.35922`（LZ `5/1` free，候选 energy margin `+37.1625/+20.6904`，下一步 `predicted_separation`、两候选均 rejected、`0` commit）、rule `840302` 的 `t=148, SOC=.39562`（occupied `0/3/5`、reserved `1/2/4`，下一步 `blocked_lz`）和 original K3 `840302` 的 `t=181, SOC=.37948`（occupied `0/2/5`、reserved `1/3/4`，下一步 `blocked_lz`）。三条 target tape 的 exact/reset/base/config/recipe 与 pre/post state hashes 均匹配。该代码观察不证明提前抢占长计划必然安全；全局计数也不能一般性归属于目标 UAV。formal 的 `39` 次成功 touchdown 中有 `32` 次 emergency，因此该 stress diagnostic 不足以证明稳健 SOC sorting。下一步改为独立预注册 low-SOC commitment gate/有证书的长计划安全 preemption 与 emergency pad/corridor capacity 的最小 repair，保持现有 SafetyFilter。

**最新结论（2026-09-22，P3 当前 MDP formal baseline）：** high-SOC qualification 的 6 条记录已完成落盘，但 rule seed `835001` 与 original K3 seed `835003` 各有 2 条 `diverted_low_soc`，因此 all-complete gate 失败；双方 controller 汇总均为 `70/72` 完成，安全、exact replay、Teacher-free、labels/updates 与 runtime shortlist gates 均通过。独立 low-SOC stress 的 6 条记录也已完成：rule 为 `71/72`、1 条低 SOC 分流，original K3 为 `62/72`、10 条低 SOC 分流；其完整性、安全与 replay gates 通过，完成率和分流按协议仅作报告。完整结果见 [P3 high-SOC 正式结果](evidence/v4_2/p3_current_mdp_baseline_result_20260922.md) 与 [P3 low-SOC stress 正式结果](evidence/v4_2/p3_current_mdp_low_soc_stress_result_20260922.md)。不授权 P3 PPO 或 checkpoint promotion；`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

**此前诊断结论（2026-09-22，高 SOC paired pilot）：** dense24 pilot seed `831002` 中，rule 与 original K3 均完成 `24/24`，diversion 与安全计数均为零，action tape exact replay；完成时间为 `686/740 s`，平均等待为 `99.5899/114.6562 s`。该回合仅作诊断，不是 qualification。它与后续 formal high-SOC seeds `835001–835003`、low-SOC stress seeds `835101–835103` 分开，不能替代 formal gate。

**最新结论（2026-09-22，P1/P2 当前 MDP 复评）：** 复评协议已冻结于 [`p12_current_mdp_requalification_v1.yaml`](../terminal_marl/configs/low_v4_2/p12_current_mdp_requalification_v1.yaml)，协议与 formal 结果见[当前 MDP P1/P2 复评协议](evidence/v4_2/p12_current_mdp_requalification_protocol_20260922.md)和[复评结果](evidence/v4_2/p12_current_mdp_requalification_result_20260922.md)。服务器 formal 已完成 `20/20` records：P1 seeds `834101–834104`、P2 seeds `834201–834206` 的 candidate/rule 配对记录均完成，safety、exact replay、terminal 与 hard gates 全部通过，shield actor fraction 为 `0`。P1 candidate/rule mean wait 为 `59.21658969142434/8.351137293207188 s`，paired delta `+50.86545239821715 s`；P2 为 `5.734572452355265/7.794466418817502 s`，paired delta `-2.059893966462237 s`。这些 paired 指标只作描述性报告，协议没有预设效率改善阈值；硬门槛通过不等于 PPO 授权。manifest SHA256 为 `57c50b926fc03a907a21c83738626a41b0c65077a7b4547d7f9ac43aaada80af`，`PPO_ALLOWED=false`、`P3_ALLOWED=false`。首轮 wrapper 因提前创建 output 目录在 reset 前失败，其 stub 与 log 保留；第二次正式运行未重复 seeds。

**最新结论（2026-09-22，P3 当前 MDP baseline evaluator）：** 新增 `terminal_marl/training/evaluate_v4_2_p3_current_mdp_baseline_v1.py` 与 `tests/test_p3_current_mdp_baseline_v1.py`；10 个定向测试和 compile 检查均通过。使用真实 original K3 SHA `4b8be...b8c8c` 的 high-SOC diagnostic seed `835000` 做 CLI 20 步 smoke，得到 rule/K3 两条 records；safety、exact replay、Teacher-free、labels/updates 与 K=3 shortlist gates 通过，`24` 架未全部完成属于 smoke 的预期，formal seeds 消耗为 `0`。正式 high-SOC qualification 与 low-SOC stress 均已按协议完成并逐 episode 落盘：high-SOC all-complete gate 失败，low-SOC `PASS=true` 仅表示其 integrity gates 通过，完成率和分流是报告结果。SSH 地址已修复，源码、协议和 checkpoint 哈希已核对一致，服务器 10 个定向测试通过。详见 [P3 当前 MDP 基线协议](evidence/v4_2/p3_current_mdp_baseline_protocol_20260922.md)。

**最新结论（2026-09-22，SOC 下限核验）：** 在当前 `80/40/10 m`、P3R 资源语义和 rule controller 下，完整单机 `gate(4) x mission(4)` flow-from-empty 矩阵 16/16 完成，touchdown 最大能量消耗为 `240.9`；按 `SOC=.25` reserve 与 `1500` 容量计算，单机名义最低初始 SOC 为 `.4106`。因此当前随机 `.55--.90` 下限高出 `.1394` SOC（`209.1` 个能量单位）；固定 `.55` replay 仍 16/16 完成并保持请求绑定 exact。历史 legacy_phase 的 `ArrivalProcess` 默认 fallback 下限为 `.35`；现行 phase_motion_v2 在 uniform 且未显式设置 `initial_soc_min/max` 时已改为 `.4106`，显式 `.35–.55` 仍仅用于低 SOC stress。该证据只覆盖无拥堵单机名义流程，不保证密集等待/交通可达性；本地命令退出码 `0`，不作为服务器 qualification。该单机证据本身不授权 PPO；当前动作以本页最新 formal baseline 与当前允许的下一步为准；`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。详见 [P3 单机 SOC 下限核验](evidence/v4_2/p3_solo_soc_floor_20260922.md)。

**最新结论（2026-09-22，覆盖下列历史时间线的下一步表述）：** 已实现目标方向的速度可行性修正；运行时首选 80 m，必要时在 swept + recursive 证明满足 40 m 保护下限的条件下由 shield 接管并继续执行，接管进入 reward 惩罚；collision event 门槛改为 `<10 m`。EH216-S 归一化能量参考改为 1500 s-equivalent，touchdown 要求 SOC≥.25。fresh dense24 seeds `830102/830103` × 间隔 `60/80/100 m` × rule/original K3 的 12/12 诊断回合中，288 请求、269 完成、19 低 SOC 分流；0 安全错误、0 `<40 m`、0 NMAC/collision、生成时的服务器同环境 12/12 exact replay（不宣称跨平台字节级一致）。80 m 的原始 K3 仅 41/48 完成，rule 47/48；本结果是诊断而非 qualification，不能启动旧 V4 PPO。P1/P2 旧结果不重写，但新 MDP 下需先 fresh 资格/效果复评，不能凭旧 gate 晋级；P3 需重新冻结能源可达的 dense24 资格场景与低 SOC 压力场景，再决定训练。详见 [P3 安全、能量与间距敏感性诊断](evidence/v4_2/p3_safety_energy_recalibration_20260922.md)。

**最新结论（2026-09-21，优先于下列历史时间线）：** P3 V3 urgency-shaped PPO 正式训练 12/12 回合、全部安全与更新 gate PASS，但 final checkpoint 12 在一次性 heldout dense24 seed `825006` 第684秒触发 `SafetyInvariantError`；heldout 停在 15/18，不能晋级。对这个已暴露 seed 的 13 个检查点只作诊断：original K3 与 checkpoint 1–11 安全，只有 12 失败；诊断 manifest SHA256 `e2be6b10f9cad30262bf68c3b9b0d57e73d9d5408f08cb7fe77f9e9f0ed40dfa`。事后选定 checkpoint 5，用完全新鲜的 Q3 seeds `826001–826006` 做一次配对确认：18/18 完整性与安全 gate PASS，但 seed `826006` 的 trained dense24 仅 `13/24` 完成（initial `24/24`），dense 聚合完成率 `.771<1.0`，故 Q3 `PASS=false`。manifest SHA256 `2ad3c6e52d52a2431f86284cd604911aed99329bbc9af897fbbb723548309e8c`。等待改善只在已完成 UAV 中聚合，不得遮蔽未完成；全部 825xxx/826xxx seeds 已消费，不能用于训练、参数选择或再次资格评估。详见 [P3 V3 与 Q3 证据](evidence/v4_2/p3_v3_early_stop_q3_20260921.md)。

将同一 Q3 dense24 回合延至 1800 秒后，trained 仍 `13/24`、末次 touchdown 第 380 秒，原始 K3 在第 911 秒 `24/24`，因而单纯延长上限不能解决停滞。新 V4 保持安全阈值和 1 秒决策步长，dense24 上限改为 1800 秒并采用“全部完成即停、到上限仍未完成则失败”；正式随机回合完成率仅作学习反馈，两个 fresh 确定性 dense24 开发回合须均 24/24 且安全、延误门槛通过，才可选择首个合格检查点。预留 heldout `829001–829006` 不在训练中评估。启动时一次 JSON 序列化错误发生于预约标记落盘后、首次 reset 前；严格核对标记与仅有扫描进度文件后已恢复，entry 2/2 PASS。但 dev baseline 的首个 fresh dense24 回合在第 533 秒、完成 12/24 时出现无可认证联合备份动作，进程于 exact-likelihood replay 阶段退出；formal PPO 更新 0 次、无检查点、无 heldout。见 [P3 V4 证据](evidence/v4_2/p3_v4_completion_guard_20260921.md)。

下文 2026-09-17–20 的“当前/下一步”句子是当时的历史状态，不覆盖以上终态；以后续本页末尾的最新允许动作与证据边界为准。

历史 P2 门槛：低层 V4.2 / Observation V2.1（88D）完成 R4 → I1 V2/Q2 → Safe DAgger → I3 → 初始 PPO 准入，并在 target-KL 保护下完成 P2 V4 与一次性确定性heldout。其后 V2→V5 on-policy support、learnability audit V2、anchored repair V3 与 fresh student-only Qualification V2 均 PASS；但当时 fresh paired effectiveness V1 仅关闭 baseline–rule 等待差距的 `46.01%`，低于预注册 `50%`，故该独立协议 **FAIL**、不授权从它直接启动 PPO/P3。后续 K3 独立 admission 和 P3 V3 训练已有新证据，不能用这条历史停止语句描述现在。已消费的 Qualification V1/V2 与 effectiveness V1 seeds 均不得重跑、reset 或用于调参。

初始 N3–N6 的 `PPO_ALLOWED=true` 是对应冻结版本及协议的历史准入。P2 V1/V2 累计 28 次合格更新；V3 在入口 KL 超限后保持历史 FAIL。独立预注册的 V4 重新资格与入口通过，并补足剩余 8 回合；这不改写 V1–V3 的历史终态。

2026-09-19 V4正式训练、确定性heldout、WAIT/credit诊断、support successor chain、Qualification V2、effectiveness V1、feasibility attribution、exact-proposal 与 set-valued proposal target chain 均已终止，当前没有遗留服务器进程。历史 PASS/FAIL 均保留；effectiveness 终态见[配对效果评估](evidence/v4_2/ppo_p2_gate_effectiveness_v1_summary.md)，fresh gate/proposal 归因见[可行性归因](evidence/v4_2/ppo_p2_gate_feasibility_attribution_v1_summary.md)，proposal target 闭环见[条件提案目标链](evidence/v4_2/ppo_p2_proposal_target_chain_summary.md)。

2026-09-21 完成 P2R 密集降落流资源修复：历史协议继续使用 lowest-ID、插入顺序和 final
容量1；`p2_dense24` 显式改用 nearest-available、downstream-first 和 final容量3。冻结 original
K3 candidate 在 seeds `809006–809009` 的300秒配对诊断中，touchdowns `23→30`、service
completions `17→22`，每个 seed 最大并发 final `1→2`，新语义 LZ assignment regret 全为0，
全部 collision/NMAC/hard-separation/uncertified-execution 计数为0。seed `809010` 反事实扫描中，
108个内环资源机会时刻仅4个存在“已单机认证但策略未请求”的 RELEASE，故当前问题主要属于资源
语义而非 gate 全面失效；未据此启动新 PPO。证据见
[P2R dense landing-flow V1](evidence/v4_2/p2r_dense_landing_flow_v1_summary.md)。

同日完成 `p2_dense24` 仿真与可视化性能修复。无GUI基准证明主要瓶颈位于递归安全预测，
而非单纯PyVista绘制：20步 `env.step` 的cProfile累计时间由 `15.726 s` 降至 `8.052 s`
（`-48.8%`）。修改只消除递归投影对不可变轨迹数组的深拷贝、复用单步内完全相同的已提交
航迹，并为仅需位置的碰撞审查使用等价的轻量插值；安全阈值、预测时域、采样间隔与动作语义
均未改变。同检查点/seed的放行计数、最小间隔与全部安全遥测逐值一致。viewer现在只显示“当前
真实位置 + 尚未执行的计划点”，按UAV稳定着色，选中航迹加粗，并在plan/topology变化或结束时
正确重建/删除actor；P2 viewer默认 `frame_stride=2`、每条航迹最多12点。完整证据见
[P2 dense visualization/performance](evidence/v4_2/p2_dense_visualization_performance_20260921.md)。

2026-09-20 冻结 Observation V2.2 candidate：保持 V2.1 的 88D byte-identical prefix，追加 6 个
scheduler/reservation/同一步竞争上下文特征，形成 94D schema；runtime 仍为单一 intent + 单次
SafetyFilter validation，不引入 catalogue search。V1 fresh pilot 完成 6/6 episodes，但发现
`interface_committed_transfers_norm` 因 reservation stage 与 trajectory phase key 错配而恒为零，故 V1
只作诊断。修复后的 V2 使用全新 seeds；前三个 train episodes 完成，seed `800004` 在 1570 s
仅完成 10/12，末态两机均停留 `ring_middle/m2i`，安全计数为零且 exact replay PASS。V2 保持
immutable FAIL，未运行 dev/probe、未复用 rows。该失败证明 complete-cycle hard gate 混入了 frozen
behavior 的 reachability 要求，不能归因于尚未参与执行的 V2.2 feature。已预注册 V3：fresh
`801001–801006`，固定 1570 s/terminal stop，12/12 arrivals generated 仍为 hard gate，completion
仅报告；其用途只限 88D+proposal 与 94D+proposal 的配对表示诊断，不能授权 repair/qualification。

V3 formal 随后 6/6 episodes、72/72 requests 全部完成，安全与 exact replay hard gates 全 PASS；
train/dev 为 247/106 rows，manifest SHA256
`20cb9d6ac2377aa5bb3e20c87707405f4aa553eecadae0581ac64a0631cc8211`。94D 相对
88D+proposal shared MLP 的 ROC-AUC 三个 seed 分别提高 `.0255/.0340/.0444`，但 infeasible
recall 为 `+.0456/+.0743/-.0695`，macro-F1 为 `+.0161/+.0326/-.0010`，state Top-1 为
`-.0094/+.0094/+.0849`，不构成一致的 observation-only closure。新增特征中
active-reservation 与 committed-transfer 完全同值，future-reservation 恒零，predecessor 仅 6/353
rows 非零，进一步说明 6 个全局 scalar 未充分表达 proposal-specific joint feasibility。下一步仅允许
探索性比较固定 94D shared scorer 与 `94D→25 independent feasibility logits`；后者仍只做一次前向、
选择一个 proposal 并交 SafetyFilter 验证一次，不改变 runtime 边界。

25-logit exploratory audit 已在冻结 V3 artifacts 上完成，manifest SHA256
`9d07f8cb2de43c0edf1f5875aedd9966d72b83df911d31a21996ba5566a43d06`。相对 94D
conditioned shared scorer，infeasible recall 三次提高 `.0552/.0456/.1295`，macro-F1 三次提高
`.0142/.0052/.0361`；但 ROC-AUC 三次下降 `.0054/.0278/.0205`，state Top-1 为
`+.0283/-.0377/-.0566`，因此不允许 architecture freeze。为避免继续使用同一 dev 调结构，V4
一次性冻结 fresh `802001–802006` 与 probe seeds `802101–802103`，使用
`0.5 balanced BCE + 0.5 feasible-set NLL`；每个 probe 必须同时满足 feasible recall≥`.85`，且
infeasible recall、macro-F1、AUC、state Top-1 相对 fresh 94D conditioned baseline 均不下降。
只有 3/3 probes 全 gate PASS 才允许冻结架构；无论结果如何都不直接授权 policy repair。

V4 formal 已完成，6/6 episodes、72/72 requests、安全与 replay 全 PASS，train/dev 236/145 rows；
manifest SHA256 `1983d5dcbb4cc7ceea28bd02a47ccb7829c33d3e8514a2c53bc86f8f1416270d`。
Hybrid probe seed `802101` 全 gate PASS，但 `802102` 的四项相对指标失败，`802103` 的 AUC 与
state Top-1 失败，aggregate `all_three_probes_pass=false`，architecture freeze 保持 false。此结果
触发预注册停止条件：不再使用 V4 dev 调整 learned proposal head/loss。下一步改测 proposal source
边界：gate 仍取 anchored repair V3 checkpoint 的 deterministic argmax，RELEASE 时固定使用 catalogue
index 12（speed/merge residual 均为 0，minimum-intervention 首选），随后只调用一次 SafetyFilter；
该路径无 Teacher、无候选枚举、无训练更新。fresh paired evaluation 使用 flow12 seeds
`803001–803006`，与原 candidate 及 nominal rule 对比。

fixed-neutral V1 虽完成 18/18 records 且表面 decision gate PASS，但 action-tape 复核显示 candidate 与
fixed controller 的 requested gate/proposal 序列在六个 seed 上逐步完全相同；candidate 中仍出现
非 neutral proposals。这证明 `student_rollout_v4_2.student_request()` 直接调用
`policy.model.distributions()`，绕过了 V1 对 `act_details()` 的覆写。V1 manifest
`15bd56cf439be13796a292ee1a745a683e4c8c9632466667676f25c597b3cf35` 保留为
**INVALID_IMPLEMENTATION**，其 PASS 不得使用。V2 已把 proposal distribution 固定逻辑移到 model
boundary，保持原 gate logits/critic 不变，并新增 action-tape hard gates：全部 fixed RELEASE 必须为
proposal 12，且 candidate/fixed 请求序列必须存在实际差异。V2 使用全新 `804001–804006`，不重跑
或复用 V1 seeds/metrics。

Corrected fixed-neutral V2 随后使用 fresh `804001–804006` 完成 18/18 paired records；实现 hard
gates 证明六个 seed 的 candidate/fixed 请求序列均有差异，且 fixed controller 的全部 RELEASE 均为
proposal 12。三种 controller 均 72/72 requests 完成、安全/回放/likelihood 全 PASS、shield 为零。
fixed-neutral 相对 candidate 的 request-to-touchdown 改善 `4.3611 s`，但 weighted mean wait 从
`13.0303 s` 增至 `17.5845 s`（delta `+4.5542 s`），违反预注册 non-worse gate，故 aggregate
FAIL、architecture freeze=false。manifest SHA256
`df7ab01ef8a828189300d2e79e4f2073b2de8fe0937ae4f0b47faa010cbedac0`。
按停止规则，不再用本轮 dev 调固定 proposal index，也不再追加 learned head/loss。该结果把下一步
限定为显式改变 runtime boundary 或回退 rule-only；用户已在随后选择并授权前者，历史 K=1 停止
结论仍保留，不因新架构候选而改写。

用户随后显式授权 K=3 bounded shortlist certification。实现保持一次 policy forward：确定性
RELEASE 的原请求为 proposal Top-1，再附带两个最高分且不重复的候选；runtime 按序最多认证三项，
只执行首个通过的 canonical trajectory。fallback 记为 runtime safety selection，不记 Teacher、
不扣 Teacher penalty，requested likelihood 仍绑定 rank 1；ActionTape 已覆盖 shortlist/rank/check count
并纳入 exact replay。fresh protocol
`ppo_p2_gate_runtime_shortlist_evaluation_v1.yaml` 冻结 smoke seed `805000`、formal seeds
`805001–805006`，比较 candidate K=1 / candidate K=3 / rule。正式 hard gates 包括 safety、completion、
replay、likelihood、每次最多三候选、fallback 非零，以及 K=3 wait/RTT 均不劣且至少一项严格改善。
V1 smoke seed `805000` 完成预定 160 s：服务器 43 tests PASS，三 controller 的 safety、exact
environment replay、requested likelihood 与 shortlist contract 均通过；短 horizon 未完成 12 架，按
smoke 非授权语义保持 `INCOMPLETE`。V1 formal 在 seed `805001` 的 K=1 与 K=3 后 fail-closed 停止：
两者均 12/12 完成且安全、trajectory replay 通过，K=3 提供 38 个 shortlist、实际选择 4 次 fallback、
每次最多检查 3 项；但旧 `_policy_tape_logprob_exact` 在复核时只回放 requested action，没有把 tape
中的 runtime shortlist 交回环境，导致第一次 fallback 后复核世界错误并把 likelihood 标成 false。
因此 V1 是 **INVALID_EVALUATOR**，不是 policy/runtime hard-gate FAIL；`805001–805006` 已消费，
不重跑、不复用其指标。

复核器现与 exact environment replay 一样恢复 `runtime_proposal_shortlists`。fresh V2 successor 已
预注册 protocol SHA `67e61165ef5c056d1b1ebd0a789cafd17f5e0554d07b29128617df3aa8bf6d5d`，
使用 smoke `806000`、formal `806001–806006`，其他 controller、checkpoint、边界和 hard gates 均不变。
V2 formal 已完成 18/18 records、三 controller 各 72/72 requests；全部 safety、completion、exact
replay、requested likelihood、Teacher isolation 与 shortlist contract hard gates PASS。K=3 六个 seed
均实际选择 fallback，每次最多检查 3 项；相对 K=1 的 weighted wait 为 `-9.519080 s`，
request-to-touchdown 为 `-2.055556 s`，两项均改善；K=3 的 wait `11.303291 s` 也优于 rule
`19.883828 s`。manifest SHA256
`6050088d3c28d1531c7e8dde5b828403390bfdeb51dde8bde54e0f2759323905`；正式结论见
[K=3 runtime shortlist V2 summary](evidence/v4_2/ppo_p2_gate_runtime_shortlist_v2_summary.md)。
`RUNTIME_SHORTLIST_FREEZE=true`；按预注册 `PPO_ALLOWED=false`，下一步只允许独立 admission review。

独立只读 admission review 随后全 gate PASS，review SHA256
`a549ffc60c35cb5ac1403ba0c4b59412096b2936d4a209ddb1a25219e09e8783`；它只授权 fresh entry
protocol 预注册，未授权直接训练。fresh entry V1 已冻结 seeds `807001/807002`、flow6/flow12、
160/256 steps 与 K=3 runtime。为保持 PPO requested-action ratio 与实际 shortlist 的一致性，entry
只允许更新 `gate_adapter + critic`，冻结 `actor_encoder/gate_head/proposal_head/temperature`，并把
更新后 shortlist 排序逐状态不变设为 hard gate；任一 update/排序/恢复 gate 失败即事务回滚并保持
`PPO_ALLOWED=false`。协议 SHA256
`d3400bf47d0f7f0d86ce4ed16c5b547b6ab30e0220691eb1f33eef320af932be`。entry 随后
2/2 records 全 gate PASS：26/26 optimizer steps 接受、0拒绝；proposal 子图 byte-exact 不变，
shortlist 排序稳定，安全/replay/likelihood/restore 全 PASS。`admission.json` SHA256
`8d425da16345dd0546851d83bfe706877dd4e6298437b8ef9bd91da324175db3`，因此
`PPO_ALLOWED=true`；entry checkpoints 仍禁止作为 formal 初始化。详见
[K=3 PPO entry V1](evidence/v4_2/ppo_p2_runtime_shortlist_ppo_entry_v1_summary.md)。
formal V1 已独立预注册：fresh training seeds `808001–808006`，按 flow6/8/12 重复两轮并运行
完整 cycle；heldout `809001–809006` 已冻结且训练阶段不可访问。formal 仍从原始 candidate 以
fresh optimizer 启动，只训练 gate adapter + critic；协议 SHA256
`7e6cf765af93925ffe960ff2ab8a4f141e13c1101e8b2f933fd64a2296bf3048`。formal 已完成
6/6 episodes、52/52 requests，354/354 updates 接受、0拒绝；全部 safety/geometry/replay/
likelihood/frozen-shortlist/restore gates PASS。manifest SHA256
`3c1bc69a518f7e5d99bb606d98b0356397e86f97b493947a0f14b0c8b49b010b`，final checkpoint SHA256
`714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e`。详见
[K=3 formal PPO V1](evidence/v4_2/ppo_p2_runtime_shortlist_formal_v1_summary.md)。heldout V1 已按
trained K3 / initial K3 / rule 三控制器预注册，协议 SHA256
`eee2543918bf80610e80ccf11efdac602369d3bb46dd2226ce53ed87d57950e6`。heldout 18/18 records
全部 completion/safety/replay/likelihood/shortlist gates PASS；trained K3 相对 initial K3 的 wait
改善 `0.033386 s`，但 RTT 恶化 `0.019231 s`，违反预注册双指标 non-worse gate，故 immutable
`PASS=false`、checkpoint 不可晋级。两种 K3 均显著优于 rule，说明失败是 PPO 无一致增益，而非
K=3 runtime 边界失效。manifest SHA256
`f44acf5e5c78925b71202e0bb81ce4c2c56d302a51fd207025b2228714a666ce`；详见
[K=3 paired heldout V1](evidence/v4_2/ppo_p2_runtime_shortlist_heldout_v1_summary.md)。
独立只读 selection review 的 9 项 gate 全 PASS，review SHA256
`5617c96255c040797abf2d2f44ed81ce53be955ed94d3a8447a5c943b2f59969`。结论为：保持 heldout
FAIL，拒绝 formal checkpoint，停止追加 P2 PPO，选择 original K3 candidate；仅
`P3_PROTOCOL_PREREGISTRATION_ALLOWED=true`，`P3_ALLOWED=false`、`checkpoint_promotable=false`。
据此已冻结 P3 V1：保持固定几何与 N6/N8/N12、60 s 到达间隔及 K=3 runtime，只把 priority 从
固定 1 恢复为随机 `1/3/5`，SOC 从固定 `.8` 改为可完成区间 `[.55,.90]`；qualification seeds
`810001–810006`，entry `811001/811002`，formal `812001–812006`，并保留 untouched heldout
`813001–813006`。entry 更新不得作为 formal 初始化，formal 仍从 original K3 + fresh optimizer
启动。协议 SHA256 `ea5efc064f763b39cd18f07f55e84f07754aebbe40d8ded588b9b61998559b25`。

fresh P2 gate support V1 因 seed772009 出现 3/959 次 shield 而保持 FAIL；其 rows 不得训练。随后
V2→V5 successor support chain 在不混入旧 support/qualification rows 的前提下完成：V3 为 15/15
安全 episodes 但 train/dev 仅 1,150/908 rows；V4 追加后为 27 个安全 episodes、combined
3,786/1,787 rows，仅 dev cardinality 少 13；V5 dev-only 追加 3 个安全 episodes 后冻结
train/dev 为 3,786/2,415 rows，全部 support gates PASS。V5 manifest SHA256 为
`100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960`，train/dev rows SHA256 为
`40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4` /
`da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b`。其后 learnability audit V2
PASS、anchored repair V3 PASS、Qualification V2 PASS；完整链条见[汇总证据](evidence/v4_2/ppo_p2_gate_support_repair_qualification_summary.md)。

## 2026-09-17 切环轨迹局部修复

checkpoint12 演示确认 m2i 中段逆环向并进入目标环内侧，根因为旧 cubic 仅约束端点切向。已完成局部修复及[固定回归验证](evidence/v4_2/transfer_geometry_repair_20260917.md)：3场景12/12完成，实际切环无逆环向、安全0、回放一致；UID6最小切向速度16.02m/s。历史 PASS/FAIL 不变，但不外推到新生成器；**修复后资格与入口复核已通过，第二个 PPO block 已完成**：24回合96/96架完成，安全/逆环向/shield全0。详见[续训结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。

## 当前冻结的主线语义（基础 schema；课程语义见顶部本轮修订）

| 项目 | 当前值 |
|---|---|
| 研究范围 | Phase A，landing-only；Phase B/C 仅为未来路线 |
| 代码主线 | `terminal_marl/`；`src/` 只读参考 |
| 空域 | `three_ring_v1`：OUTER / MIDDLE / INNER |
| 研究机型 | 当前为 `eh216s_research_v2`；`eh216s_research_v1` 仅保留为历史 checkpoint/provenance |
| 动力学 | `performance_constrained_v2`；`physics_dt = 0.2 s` |
| 低层观测 | 当前合格策略仍为 `phaseA_low_obs_v2_1_retry_state` 88D；`phaseA_low_obs_v2_2_scheduler_context_94d` 为未晋级 candidate |
| 低层动作 | `low_action_schema_v4_2_hierarchical` |
| 低层分布 | `WAIT/RELEASE` 二分类 gate + RELEASE 条件下 25-way proposal |
| 执行解码 | 1 个 canonical WAIT + 25 个 `speed × merge-time` RELEASE 意图 |
| 高层动作 | `high_airspace_v1_10d`，当前训练关闭 |
| 安全权威 | TrajectoryGenerator → SafetyFilter → PredictiveSafetyShield |
| Teacher 角色 | 训练期 expert labeler / auditor；不是部署控制器或 PPO behavior action |
| 当前低层课程 | `task_priority_enabled=false`；正常降落、吞吐、完整周期等待和 tail starvation 监测 |
| 课程兼容 | raw mission 保留为物理服务/兼容 metadata；base 抽样只消耗 RNG，新课程有效值为 `1`；policy mission one-hot=`STANDARD` |
| 服务负荷 | mission service time `30/120/10/60 s` 保留；不表示 mission-level preference |
| 能量应急 | energy emergency takeover 始终保留，与 raw `mission=emergency` 独立 |

低层完整语义见 [reference/low_level_policy.md](reference/low_level_policy.md)。

## 里程碑与证据

| 里程碑 | 状态 | 说明 / 证据 |
|---|---|---|
| Phase A 核心环境与 rule pipeline | 已进入后续训练使用 | 具体接口见主规范；历史迁移过程不再作为状态源 |
| D0–D7 动力学 qualification | PASS | [正式报告](evidence/dynamics/dynamics_qualification_report_2026-08-28.md) |
| P0 仿真性能 qualification | PASS | 结果摘要保留在 [历史升级计划](history/plans/CRAZYFLIE_DYNAMICS_LAYER_UPGRADE_PLAN.md) 末节 |
| C0 capacity calibration | PASS | 作为 user-provided frozen evidence 记录于 [V4 migration summary](evidence/v4/v4_new_mainline_migration_summary.md) |
| V4.2 hierarchical S0 | PASS | [S0 qualification](evidence/v4_2/v4_2_s0_qualification_summary.md) |
| Observation V2.1 final freeze | PASS | [final freeze v2](evidence/v4_2/v4_2_v2_1_final_freeze_decision_v2.md) |
| Fresh V2.1 I0-R | PASS | [I0-R qualification](evidence/v4_2/v4_2_v2_1_formal_i0_r_summary.md) |
| Fresh multi-seed I1 | FAIL | RELEASE manifold/context generalization gap；[qualification](evidence/v4_2/v4_2_v2_1_i1_formal_qualification_summary.md) 与 [attribution](evidence/v4_2/v4_2_v2_1_i1_failure_attribution.md) |
| I0-R2 repair collection | FAIL | 未闭合所需 support；协议和结果保存在 `evidence/v4_2/` |
| I0-R3 repair collection | FAIL（历史） | [R3 result](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md) |
| I0-R4 repair collection | PASS | [正式结果](evidence/v4_2/v4_2_v2_1_i0_r4_collection_summary.md) |
| I1 Qualification Protocol V2 / Q2 | PASS | [正式结果](evidence/v4_2/v4_2_v2_1_i1_v2_summary.md)，Q2 1,787 rows，三 seed 全通过 |
| I2 Safe DAgger | R1 PASS / I3 PASS | [I3 协议与 R1 结果](evidence/v4_2/v4_2_v2_1_student_qualification_protocol.md)，976 新事件、35 epochs、全部 gate PASS |
| 初始 PPO 历史准入 | **PASS（冻结 N3–N6 配置）** | [准入结果与训练命令](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，正式 train 模式已接受最新 admission |
| P2 自然流 V3 | **FAIL（entry KL）** | 资格 6/6 PASS；entry seed763001 的 `approx_kl=0.11257042959332467>0.05`，未执行 formal train；[持续课程终态与诊断](evidence/v4_2/continuous_curriculum_20260917.md) |
| P2 自然流 V4 | **训练 PASS / 效果审查 BLOCKED** | target-KL 资格6/6、入口2/2、训练8/8 PASS；heldout 18/18 PASS，但等待55.11s仍高于rule 16.18s；[正式终态](evidence/v4_2/ppo_p2_v4_training_evaluation_summary.md) |
| P2 WAIT/credit诊断 | **FAIL** | fresh stochastic 12/12 records；candidate50/52、baseline48/52；false-WAIT 83.21%/80.76%，安全与shield全0；[诊断终态](evidence/v4_2/ppo_p2_v4_wait_credit_diagnostic_summary.md) |
| Gate forgetting离线审计 | **COMPLETE / 不支持单纯PPO遗忘** | I3 false-WAIT 82.85%，checkpoint24 83.11%，V4 80.37%；I3并未解决P2状态支持缺口 |
| P2 gate support V1 | **FAIL（shield gate）** | 15/15 records、train/dev覆盖通过、安全0；seed772009有3/959次shield，V1零容忍gate失败；rows不得训练 |
| P2 gate support V2 | **PASS** | train 78/78、dev 52/52；5,627/2,009 rows；manifest SHA `09402508bc02155c1369de477dc320d5ca596768d4de6461f4a0526f2c7d9304`；仅进入后续离线审计 |
| P2 gate support V3 / V4 / V5 successor | **V5 PASS** | V3 15/15 安全但 train/dev 1,150/908；V4 后 combined 3,786/1,787（仅 dev cardinality 少13）；V5 dev-only 追加3个安全 episodes，最终 train/dev 3,786/2,415，全部 gates PASS；V5 manifest `100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960` |
| P2 gate learnability audit V2 | **PASS（离线）** | 3/3 fresh probes；manifest SHA `5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679`；support/dev 通过，unanchored adapter probe 在 D1 降级；无 environment/Teacher/formal policy training |
| P2 gate repair V1 / V2 | **FAIL（历史）** | V1 non-finite loss；V2 仅 support-dev NLL `.969>.50` 失败，其他主要 metric 与 D1 anchor gates 通过 |
| P2 gate repair V3 anchored | **PASS** | protocol `f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8`；manifest `e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410`；仅 gate_adapter 六参数变化，T/proposal/critic/common state invariant |
| P2 gate calibration V3 | **PASS（离线）** | fixed `T=2.5`；无 training/optimizer/environment/Teacher；仅授权 fresh student-only qualification |
| P2 student-only qualification V1 | **FAIL（历史终态）** | formal seeds `774001–774006`；原 false-WAIT / RELEASE recall gates失败；seeds 不得重跑或调参 |
| P2 student-only qualification V2 | **PASS（当前终态）** | formal seeds `782001–782006`；6/6、52/52、621 actor rows；sampled false-WAIT `18/206=.0873786`、false-RELEASE `16/415=.0385542`；仅新授权 P2 effectiveness evaluation |
| P2 paired effectiveness V1 | **FAIL（该协议历史终态）** | formal seeds `788001–788006`；18/18 controller episodes、各52/52请求；全部 hard gates PASS；等待 gap closure `.4600799557<.50`，其余效果 gates PASS；该协议本身不授权 admission review/PPO/P3，后续独立 K3 路线另计 |
| P2 flow12 feasibility attribution V1 | **COMPLETE（诊断）** | formal seeds `789001–789006`；6/6、72/72请求、1,050 actor rows；89次 rejected RELEASE 中66次为 Teacher也判RELEASE但proposal不同，74次发生于后半到达；仅支持转入fresh proposal support/learnability设计 |
| P2 exact-proposal support V1 | **PASS（数据）** | train/dev seeds `790001–790009` / `791001–791006`；540/455 rows；15/15 episodes、180/180请求；仅授权离线exact-label learnability audit |
| P2 exact-proposal learnability V1 | **FAIL** | anchored residual probes fresh-dev Top-1约`.433`、Top-3约`.577`，D1 Top-1约`.962`；未授权repair |
| P2 exact-proposal failure attribution V1 | **COMPLETE** | standalone MLP Top-1 `.580–.600`、Top-3 `.712–.732`；分类为`observation_or_label_learnability_limit_supported` |
| P2 proposal feasible-set diagnostic V1 | **COMPLETE** | fresh seeds `794001–794003`；36/36请求；166个requested-RELEASE状态，Top-1/Top-3可行107/123，27个状态无可行RELEASE |
| P2 proposal-set support V1 | **PASS（数据）** | train/dev seeds `795001–795009` / `796001–796006`；463/372 rows；15/15、180/180请求；全部episode/split/catalogue gates PASS |
| P2 proposal-set learnability V1 | **FAIL** | 3个D1-anchored probes整体Top-1约`.777–.780`，partial-set Top-1约`.478–.484`；Top-3/D1通过，Top-1与o2m失败 |
| P2 proposal-set failure attribution V1 | **COMPLETE** | standalone linear仅1/3全gate PASS，MLP 0/3；分类为`observation_or_set_target_learnability_limit_supported`；不授权repair |
| P2 proposal-conditioned feasibility V1 | **FAIL** | 88D+proposal参数的binary MLP feasible recall约`.90`，但 infeasible recall `.454–.462`、AUC `.793–.810`，o2m与全部综合gate失败；不授权factorized architecture或repair |
| Observation V2.2 closure V3 | **COMPLETE（诊断，不闭合）** | 6/6 episodes、247/106 train/dev rows；94D 的 AUC 3/3 提高，但 negative recall、F1、Top-1 不一致；observation-only repair 不授权 |
| V2.2 25-logit feasibility head exploratory | **COMPLETE（不闭合）** | negative recall/F1 3/3 改善，但 AUC 3/3 下降、Top-1 仅1/3改善；architecture freeze 不授权 |
| Observation V2.2 hybrid confirmation V4 | **FAIL（架构）** | collection 6/6 PASS；hybrid 仅1/3 probes 全 gate PASS；学习式 proposal 分支停止，不再基于 V4 dev 调参 |
| Learned gate + fixed neutral proposal V1 | **INVALID_IMPLEMENTATION** | rollout 绕过 `act_details()` adapter；candidate/fixed action tapes 完全相同；表面 PASS 作废，803xxx 不重跑 |
| Learned gate + fixed neutral proposal V2 | **FAIL（效果）** | 18/18 records 安全完成且实现 hard gates PASS；RTT改善4.36s，但平均等待恶化4.55s，aggregate gate FAIL |
| K=3 bounded runtime shortlist V2 | **PASS / FROZEN** | fresh `806001–806006`；18/18 records、216/216 requests；全部 safety/replay/likelihood/contract gates PASS；K3 相对 K1 等待改善9.52s、RTT改善2.06s |
| K=3 PPO admission review V1 | **PASS（只读）** | review SHA `a549ffc60c35cb5ac1403ba0c4b59412096b2936d4a209ddb1a25219e09e8783`；仅授权 fresh entry protocol 预注册，不直接授权 PPO |
| K=3 PPO entry V1 | **PASS / PPO_ALLOWED** | fresh `807001/807002`；26/26 updates 接受、0拒绝；全部 safety/replay/likelihood/frozen-shortlist/restore gates PASS；entry checkpoints 不得作为 formal 初始化 |
| K=3 formal PPO V1 | **PASS / HELDOUT ALLOWED** | fresh `808001–808006`；6/6、52/52完成；354/354 updates 接受、0拒绝；全部 hard gates PASS |
| K=3 paired heldout V1 | **FAIL（效果）** | 18/18 records hard gates PASS；trained vs initial wait `-0.033386s`、RTT `+0.019231s`；formal checkpoint 不可晋级，seeds 不重用 |
| P2 candidate selection review V1 | **PASS（只读）** | original K3 被选为下一阶段候选；formal checkpoint 淘汰；`P2_PPO_STOP=true`；仅授权 P3 protocol 预注册 |
| P2R dense landing-flow semantics V1 | **PASS（诊断/资源语义）** | seeds `809006–809009`：300秒 touchdowns `23→30`、completed `17→22`、并发final `1→2`、LZ regret归零且安全计数全0；seed `809010` confirmed gate miss `4/108`；不授权新 PPO 或 checkpoint promotion |
| P3 priority/SOC + dense24 PPO | **V3 formal PASS，final heldout FAIL；早停 Q3 FAIL** | V3 12/12 训练 gate PASS，但 checkpoint12 heldout 安全不变量失败；checkpoint5 事后选择，fresh Q3 18/18 安全/实现完整性通过而 dense 完成率 `.771<1.0`；全部检查点不晋级。见 [P3 证据](evidence/v4_2/p3_v3_early_stop_q3_20260921.md) |
| A6 高层动态空域训练 | BLOCKED | 需要低层 static-geometry qualification 后冻结低层策略 |
| Crazyflie 实物验证 | FUTURE | 仅在主仿真策略冻结后进入 small-N 缩尺验证 |

## 保留的历史失败：I0-R3

I0-R3 的 provenance、Teacher determinism、exact replay、Safety、partition independence、
aliasing 与 nominal anchors 均通过，但正式结果仍为 FAIL：

- complete-cycle 停止产生 4,541 个训练 rows 和 1,511 个内部审计 rows，超过冻结上限
  4,200 / 1,400；
- proposal-14 的 N=5 支持只有 12 个独立 lifecycle，低于最低 15；
- shared gate false-negative、shared proposal error，以及 post-WAIT、transition/retry、
  post-conflict、proposal-11、proposal-14 的 radius coverage gate 全部未达预注册阈值。

R3 仍为 FAIL；后续通过的是独立预注册的 R4，不能追溯升级 R3。完整数值见
[I0-R3 collection summary](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md)。

## 历史进度与结果（旧课程记录）

| 阶段 | 已核验进度 | 结论 |
|---|---|---|
| 初始 PPO block1 | 12回合，48/48架完成；checkpoint12独立评估24/24架完成 | 安全与闭环 gate PASS |
| 轨迹修复后 block2 | 24回合，96/96架完成；checkpoint24配对评估双方各24/24架完成 | 全 gate PASS，但候选相对 checkpoint12 的完成时间差为0/0/+12/0/+9/-2秒，没有明确改善 |
| P2 可达性 pilot | N6/N8/N12，rule/student各52/52架完成 | 可达性通过；N12 student等待65.66/73.91秒，rule为19.07/26.88秒，效率差距仍明显 |
| P2 V1 | 前23次更新合格，第24回合未更新 | initial ratio 检查误差超限，历史 FAIL |
| P2 V2 | 从 checkpoint23 恢复后5次更新合格，第6回合未更新 | 连续轨迹内侧偏差超限，历史 FAIL |
| P2 V3 | 新生成器资格6/6通过；entry仅执行1/2 | seed763001 的 KL 超限，formal train 未启动，历史 FAIL |
| KL 双种子诊断 | fresh 765001/765002；恢复 Adam 与清零 Adam 配对 | 仅诊断；未复现原失败，不支持清零 Adam 作为修复 |
| KL 更新保护 | 已提交并接入独立 V4 协议；资格6/6、入口2/2、训练8/8 | 468/468 formal optimizer steps 接受，0拒绝；最大 accepted rollout KL 0.006534<0.01 |
| P2 V4 heldout | 755001–755006 已一次性消费；candidate/baseline/rule 各52/52完成 | candidate 等待55.11s，优于baseline 65.02s但落后rule 16.18s；效果审查不通过 |
| P2 WAIT/credit fresh diagnostic | 771001–771006 已消费；12/12 records生成 | `DIAGNOSTIC_COMPLETE=false`；candidate50/52、baseline48/52；false-WAIT和team credit失配 |
| P2 gate support V2 | 775001–775009 / 776001–776006 已消费；78/78、52/52请求完成 | train/dev 5,627/2,009 rows；support V2 PASS，仅进入离线 learnability |
| P2 gate support V3 → V4 → V5 | successor seeds 已按协议消费；V3 15/15 安全，V4 27 个安全 episodes，V5 dev-only 3 个安全 episodes | 最终冻结 train/dev 3,786/2,415，V5 全部 support gates PASS |
| P2 gate learnability V2 → repair V3 | support V5 后已完成；learnability V2 与 anchored repair V3 均 PASS | learnability 3/3 fresh probes；repair 仅 gate_adapter 六参数更新，D1 anchor 与非 adapter state 保持不变 |
| P2 student-only qualification V1 | 774001–774006 已消费；6/6 episodes、52/52 UAVs、983 actor rows；recovery仅补写manifest | **FAIL（immutable）**：sampled false-WAIT 0.333333、RELEASE recall 0.625786；不得重跑/调参 |
| P2 student-only qualification V2 | 782001–782006 已消费；6/6 episodes、52/52 UAVs、621 actor rows；formal manifest/marker 已落盘 | **PASS**：sampled false-WAIT 0.0873786、false-RELEASE 0.0385542；概率 macro-F1 0.9532325、WAIT recall 0.9590361、RELEASE recall 0.9563107、NLL 0.1034486、Brier 0.0314997；仅 effectiveness evaluation 新获允许 |
| P2 paired effectiveness V1 | 788001–788006 已消费；candidate/baseline/rule 各6 episodes、52/52请求 | **FAIL**：candidate等待15.9619s、baseline17.3097s、rule14.3801s；等待 gap closure 46.01% 未达50%；request-to-touchdown与全部hard gates通过 |
| P2 flow12 feasibility attribution V1 | 789001–789006 已消费；6/6 episodes、72/72请求、1,050 actor rows | **COMPLETE**：66/89 rejected RELEASE 为 Teacher-RELEASE + proposal mismatch；late half 占74/89 rejections；o2m/m2i分别贡献55/11个该类事件；无训练授权 |
| P2 exact-proposal target chain | support 15/15 episodes PASS；learnability 3个probe FAIL；failure attribution COMPLETE | exact canonical Teacher proposal 在fresh dev上不可稳定学习；未保存或修改策略 |
| P2 proposal feasible-set / support | diagnostic 3/3 episodes COMPLETE；support 15/15 episodes、180/180请求 PASS | train/dev 463/372集合标签rows；Top-1可行率351/463与265/372；仅进入离线audit |
| P2 proposal-set learnability / attribution | anchored audit 3/3 probe FAIL；standalone linear/MLP各3 seeds attribution COMPLETE | Top-3、D1保持与接口aggregate可过，但partial-set Top-1未闭合；repair/PPO仍禁止 |
| P2 proposal-conditioned feasibility | linear/MLP各3个probe，11,575/9,300 train/dev state-proposal pairs | **FAIL**：MLP 0/3；o2m、negative recall、macro-F1、AUC与state Top-1未闭合；旧closure-bank可分性未泛化到fresh flow12 |
| P3–P5 / static qualification | P3 尝试后未合格 | Q3 dense24 吞吐失败；P4/P5 与 A6 高层动态空域均未获授权 |

## 历史问题与处理状态（旧课程记录）

1. **奖励与等待目标冲突已修正，但效率改善仍不足。** 旧 progress 奖励计入绕环距离：外环22m/s约每秒+0.022，高于时间惩罚-0.01。P2配置将该系数置0，物理、安全及其他奖励保持不变。V4 heldout 已验证 candidate 等待相对 checkpoint24 baseline 下降9.91s，但仍比rule高38.93s；修正有效果，不足以认定收敛。
2. **初始动作概率检查存在批次浮点差异，已局部修复。** V1 seed754024 的误差为1.1444091796875e-5，超过1e-5。原action tape只读复放表明完整world批次误差0，actor子批次复现误差。检查器现保留原采样批次再mask，阈值和PPO行为不变；后续V2/V3初始ratio检查通过。
3. **轨迹节点检查漏掉连续线段内切，已局部修复并通过V3资格复核。** V2 seed757006实际内侵0.0592194m，超过0.05m。UID39见证曲线节点半径均合格，但线段最近点半径199.9275204m。修复使用每段投影最近点，仅超限旧曲线进入已有forward-annular fallback，保持有效旧曲线和原安全容限；此前38项定向测试通过。V3资格最大内侵0.0186652m，安全/shield/逆环向均0。
4. **V3 的 PPO 更新漂移已由独立 V4 协议闭合，但历史 FAIL 不变。** V4 每个 optimizer step 后检查完整 rollout requested-action KL，超限或非有限值时事务回滚模型与 Adam；原0.05 gate不放宽。资格、入口和8回合训练均通过，formal 468次更新全部接受，最大 accepted rollout KL 0.006534<0.01。该结果恢复了独立 V4 的训练准入，不追溯升级 V3。
5. **P2 gate support 的 distribution successor 与 Qualification V2 已闭合。** support V3/V4/V5 逐步补足 calibrated candidate 自身状态分布，最终 train/dev 3,786/2,415 且全部 support gates PASS；learnability audit V2 3/3 PASS，anchored repair V3 PASS。Qualification V1 的 `774001–774006` FAIL 保持 immutable；Qualification V2 的 `782001–782006` 以 6/6、52/52、621 rows 和全部 episode/safety/provenance gates PASS 终态 PASS。两组 seeds 均不得重跑或用于调参。
6. **fresh attribution 将主要剩余瓶颈定位到 proposal/执行可行性。** 6个全新 flow12 episodes 全部安全完成；1,050 actor events 中 false-WAIT/false-RELEASE 为43/23，而89次 rejected RELEASE 中66次发生在 Teacher同意RELEASE但student proposal不同的事件，且这66次无一proposal match；其中55次在o2m、11次在m2i。late arrival half 占74/89 rejections与52/66上述proposal mismatch rejects。i2f 的6次reject则全部对应false-RELEASE。
7. **单标签、可行集合和动作条件化三条proposal监督路线均未闭合。** exact-label audit 与独立failure attribution先后失败；完整25-proposal catalogue显示166个requested-RELEASE状态中107个Top-1可行、123个Top-3可行，说明单一Teacher tie-break并非正确监督目标。fresh set support 463/372 rows全部gate PASS，但D1-anchored set probes的overall Top-1仅`.777–.780`、partial-set Top-1仅`.478–.484`，o2m仍失败。去掉D1 anchor和旧logits后，standalone linear仅1/3全gate通过，MLP 0/3。再将每个状态展开为25个`88D+proposal` binary样本，MLP feasible recall虽约`.90`，infeasible recall仍仅`.454–.462`，AUC`.793–.810`且o2m失败。因此当前证据支持88D信息/多UAV同一步竞争或target factorization限制，不支持直接proposal repair。
8. **课程覆盖仍不完整。** 历史 P2 N6/N8/N12是总请求数，N12 pilot在空峰值仅6–7架，不能称12架并发或高负载验证。P3 已加入 priority `1/3/5`、SOC `.55–.90` 和 dense24 warm+flow，但 Q3 吞吐失败；仍未合格覆盖间隔扰动、20/40/60总请求、持续/突发流和合法静态几何泛化。离最终终端区场景仍有明确跨度。
9. **文档参数及运行交接需保持一致。** 需求文档仍有旧高度/速度数字；当前验证配置为三环高度100/70/40m、速度22/20/16m/s，不能静默切换。后续长任务按用户要求由`luna_worker`跟踪、主agent静默等待终态。Git无remote，代码仅经Syncthing同步，不能声称已推送。
10. **密集可视化中的三个资源问题已闭合。** nearest-available 消除了可避免的 LZ 距离后悔；downstream-first 防止上游插入顺序抢占同一步认证；final容量3允许安全过滤后的并发进近。环空闲不再直接解释为低效：seed809010 的95个“有名义资源但未请求”时刻中只有4个存在 individually-certified RELEASE，另外91个受航迹/headway/安全约束。环容量是上限而非占满目标。

## 模型、评估与证据边界

- 历史 P2 V4 候选：`p2_resume_v4_20260918/train/checkpoint_8.pt`，SHA256 `5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`。其训练与 heldout hard gates PASS，但该协议因 rule 效率差距未晋级；后续 P3 从另一条 original K3/anchored repair 路线开展，不能混淆模型血缘。
- 配对 baseline：`ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt`，SHA256 `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`。
- V4恢复起点：`p2_resume_v2_20260917/train/checkpoint_5.pt`，SHA256 `8a8466f61119f7f5b5f780249d26f218ee9b8ae3e4aededc5c7f5fe2e51b201b`。它是V2失败训练块中的最后合格中间模型，现仅作为V4 provenance，不是当前最终候选。
- V3 root/entry manifest SHA256：`2098af1c3109c17fc489b7c9f20a9db11eb67aa62b77b5a03beb2de66909cd7d`；KL诊断manifest：`41017e10f45105a6125213ac51a630bdcfda9c35bac196608e2affc891955f69`。结果位于输出根目录下 `p2_resume_v3_20260918` 与 `p2_v3_kl_diagnostic_20260918`。
- checkpoint24评估seeds710001–710006与P2最终评估755001–755006均 **已消费**，不得重跑或用于调参。V4 heldout manifest SHA256为 `167d22cb29d8ef6d08275cb0ff4cbe270c0d9005611a176b85d81fd64730f104`。
- WAIT/credit diagnostic seeds771001–771006亦 **已消费**；manifest SHA256为 `b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93`，终态FAIL不得重跑或升级。
- P2 gate support→learnability→repair→qualification 链的完整证据见[汇总](evidence/v4_2/ppo_p2_gate_support_repair_qualification_summary.md)：最终 V5 support manifest SHA256 为
  `100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960`，train/dev rows SHA256 为
  `40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4` /
  `da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b`；learnability audit V2 manifest SHA256 为
  `5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679`。
- Anchored repair V3 protocol/manifest/checkpoint SHA256 分别为
  `f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8` /
  `e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410` /
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`；仅 gate_adapter 六参数改变，T/proposal/critic/common state invariant。
- Qualification V2 protocol SHA256 为 `7ab9e2a84b9aea8da2ea975d98b4867de4a45bd5725bc72019b2e95ddbb75094`，formal
  `782001–782006` 已消费；manifest SHA256 为
  `7cd4f02c88dc953fe1545c613424e8c961c295d4f8c099456d6697f300088feb`，marker SHA256 为
  `43fafcd99e22d48bffb6c2389767a265c2376b05f95a070ae3d823f4066949d9`。形式化 rollout 的 source runner
  hash 为 pre-fix `b91e0d3794495bb768cf80ab266f1584b0af01addd2b0998c9c07e30af8d4a82`；CLI serialization 修复提交
  `d486bab` 仅修复终端打印，没有 rerun/reset，不能声称 formal rollout 使用了 patched source。
- Effectiveness V1 protocol SHA256 为 `5882361adcabaff4c0017cae73a8b95dc8029e204942bf4a83879981bd321d84`；formal
  `788001–788006` 已消费。manifest/log/marker SHA256 分别为
  `6aaebeac4dc8dcf425ce250df6c6abcd991921263605b3dd71d3d717f0783e88` /
  `df9aebf8860212a72ede78f0078e22d0a8d8df90535d20798a436ba4f28c3acd` /
  `b9ca4306835b19756c070eb7917d3e01de429209329e59f397257a5e4c54fd8c`。终态 FAIL，不得重跑或以其 seeds 训练/调参。
- Feasibility attribution V1 protocol SHA256 为 `50b2cb7dc40160a54ecfe150122d7bcc3407b638a40e887b20045120e220b1fd`；formal
  `789001–789006` 已消费。manifest/rows/log/marker SHA256 分别为
  `f2382b1a8c5ee486c4246e716fc27b97a34d2948bfe90633e33d0f428a56f35c` /
  `b07f51b0354d92943864232b7698a0d53061de22e6593911b10e40a8192c412f` /
  `371fd775e823fe347a593fec4ba79ca03217bdfb44ffc38ddbceb382bcbc5161` /
  `89e7af446d312c06019d88f2c4b1e8b4424bde8711dae4cba9294550d1ddfa2e`。rows 仅用于归因，不得训练。
- Exact-proposal support/audit/failure-attribution manifest SHA256 分别为
  `48c0b1cd0b92bd746bd52f18090bbffd8facf59457eb82d503802eb4fc2c512b` /
  `87bdf9b5ce0b9d9cbf967d858669a6f17679c0cf271d83c8c39e7f4fa4e52e95` /
  `3c8e0980f4f970bb66ca207ceb3305dc3fbe9e3eb79e9e7a827a1b086acf11f5`。support rows冻结；audit FAIL，不能进入exact-label repair。
- Proposal feasible-set diagnostic protocol/manifest/catalogue SHA256 分别为
  `dbe387cc134741f87cde8caa2b8910410e42e5ad756dcb197bc2c04c208eabca` /
  `957bc2535fbef767a90fbf6cd299ea5aa38cd2f680099a3c5f79aff97fe4821f` /
  `c7dbbf203084d1b90db8662f42514b8c3f45a5e94b2d7e9e706ad80143784242`；仅诊断，不可训练。
- Proposal-set support protocol/manifest/train/dev SHA256 分别为
  `fa2d764604e05526262549559c718291caeef3ca23202bce38d20d3e038d2972` /
  `d057d0f39a3a99f6641c331eecd1a127a2b639b5046bf78d39b55d1b8caf7f37` /
  `f5511da554f21614e399b6384609bac93a41e9fdef94b8735eb2ecd6cb5d3f14` /
  `dd07e8f2b877bd7f08db113aa0fc8988ed8d106223d799284b3e1f8b6324d17a`。support PASS，但后续learnability FAIL，rows不得用于repair。
- Proposal-set learnability audit 与 failure attribution manifest SHA256 分别为
  `a1b76fa16427cecdc0c70e0d38c6247d8e6e6d943063aeeddd00a8cede28e937` /
  `4a1d6d9d9b594c1d680203e8c8b728e08101ec1fdfc26aa64028be9f800c5e10`。二者均未保存probe checkpoint，candidate state digest始终为
  `fa8740a8a4cb54fffd3bb669537133ede8f231edd754e4b10ebfc0f982a0ed61`。
- Proposal-conditioned feasibility audit protocol/manifest SHA256 分别为
  `68694cefa2e8907b3d4beb77ca3abc2daf569bc9b5ffab6b46b1d70b9888a896` /
  `4b37952e8b624284f1d45280043f89aaad9ef97be4e1f33e690644c03827067f`。终态FAIL；未保存probe checkpoint，未调用environment/Teacher/policy optimizer。
- 2026-09-19 该批任务终止时没有遗留 server process；2026-09-21 已另行启动 P3 V4。Qualification V1 的 recovery manifest/marker 只属于旧 `774001–774006` FAIL 证据，不改写新 V2 PASS。
- 入口 `entry/checkpoint_1.pt` 和Adam诊断快照均不能晋级正式模型；历史失败manifest不得重写。
- 服务器输出根为 `~/training_outputs/terminal`，本地镜像为 `D:/7_codespace/training_outputs/terminal`。源码只在本地修改，经Syncthing同步，服务器仅运行。

## 当前允许的下一步

本轮 task-neutral baseline full-cycle 已完成，主审已核对 report；后续进入受限开发决策：

1. 保留 `ROOT/check_only` 的 wrapper 失败记录，并以 `ROOT/check_only_v2` 的 `CHECK_ONLY_OK exit0`
   作为当前 check-only 事实；
2. 本轮有界 task-neutral joint-batch protocol 已完成：继续以 `3fe7...d4ab` actor 为起点，
   `fa0e` 与新 `c961` checkpoint 均不保留，`03c0` 保留为对照；完整终态与失败证据见
   [joint-batch protocol](evidence/v4_2/p3_task_neutral_joint_batch_protocol_20260928.md)。当前
   [full-episode credit protocol](evidence/v4_2/p3_task_neutral_full_episode_credit_protocol_20260928.md)
   已完成：代码 `f8a8b53`、定向 tests `45`（joint `16`）、check-only `CHECK_ONLY_OK exit0`，
   child PID `1530563`，六个 training cells 合计 `4635` steps、`1` 次 update、`290` accepted/`0`
   rejected；配对 evaluation 两臂各 `114` requests，安全/unresolved 均为 `0`。checkpoint 不保留，
   `3fe7` actor 继续作为开发起点；`gamma=1`、`lambda=1` 未成为既定修复。full-episode 失败
   结果保留。当前授权见
   [joint-team actor aggregation protocol](evidence/v4_2/p3_task_neutral_joint_team_actor_protocol_20260928.md)：
   已完成 `COMPLETED exit0 technical_ok=true`，代码 `fb1a9a61d0dbb524a60928f598a0ad847297a080`、
   定向 tests `57`，`4635` steps/`1` update/`290` accepted/`0` rejected，checkpoint `ac7` 仅待重复
   验证、不晋级；full-episode 失败与本终态结果均保留。当前授权见
   [joint-team repeats protocol](evidence/v4_2/p3_task_neutral_joint_team_repeats_protocol_20260929.md)：
   已完成 `COMPLETED exit0 technical_ok=true`，两臂各 `288` requests，before `236/52/0`→after
   `230/58/0`，new-only before `156/36/0`→after `151/41/0`，安全/unresolved 均为 `0`；ac7
   性能候选拒绝，joint-team 数学修正保留。当前授权见
   [learning-curve protocol](evidence/v4_2/p3_task_neutral_learning_curve_protocol_20260929.md)：
   先定向 tests/check-only，主审准入后才启动唯一 full；当前未启动。
3. 保留 full-cycle 的完整 ledger、normal/emergency/failed/unresolved、mean/p95/max wait、
   low-SOC wait 和 reward 记录。新 reward 不与旧课程 reward 直接比较；不增加普通 priority-weighted
   wait 或每 cell/子组无退化硬 gate；等待尾部只作 starvation 监测；
4. `PPO_ALLOWED=false`，formal/heldout、P4、全局 source 晋级和 qualification 均未因本轮 pilot
   自动放行；pilot technical completion 不等于稳定收益或全局 PPO 放行。

## 历史允许的下一步（已由本轮 task-neutral protocol 取代）

**2026-09-24 当前顺序：** merge-slot timing reconciliation、fixed-rule dense30 development 与
viewer bounded checks 已完成；历史结果、已消费 seeds 和 `.25` floor 记录保持 immutable。当前按
staged authorization 推进，不把历史 fixed-rule evidence 写成 learned-policy qualification。

1. 保持已冻结的 `eh216s_research_v2` profile、checkpoint/hash contract 和无 Teacher/optimizer
   语义；首个 bounded diagnostic 的协议与结果分别见
   [protocol](evidence/v4_2/trained_dense30_v2_diagnostic_protocol_20260924.md) 与
   [result](evidence/v4_2/trained_dense30_v2_diagnostic_result_20260924.md)。global loader/core
   保持不变，不自动授予 qualification。
2. `842101` 的 read-only trace aggregation 已完成；failed-energy trace 与 inner `LZ_RESOURCE`
   统计见结果页。该证据用于区分 controller request 与 execution feasibility，不能将 WAIT 全部判错，
   也不能写入 imitation-only fix 或新的 PASS。`842101` 历史证据保持 immutable，`842102` 保持
   unused。
3. fresh random-SOC development comparison 已完成，详见
   [protocol](evidence/v4_2/random_soc_repair_development_20260924.md) 与
   [result](evidence/v4_2/random_soc_repair_development_result_20260924.md)。`843000` smoke 与
   `843101` trained/nominal_rule 配对均已记录；trained `9/18/3/0`、nominal `17/13/0/0`
   （normal/emergency/failed-energy/unresolved），两者均 terminal 且无 collision/NMAC/
   hard-separation/fail-closed。该结果包含 19 条 trained probe、4 个 approved witness，不能写成
   gate-only 归因或 qualification。
4. 当前目标转为准备 fresh P3 `flow6`/`flow12`/`dense24` development 与 dense24 forced-gate/
   frozen-proposal 消融；代码与新 protocol pending。配对结果不授权 targeted repair 训练或
   qualification，`PPO_ALLOWED=false`，保持 `80/40/10 m`、`min_landing_soc=.10` 与
   `soc_reserve=.25`；低层 static qualification 完成后才进入高层 A6。
5. P3 random-SOC 范围沿用现有 P3 V4：initial SOC `.55–.90` uniform、priority `1/3/5`、场景
   为 `flow6`/`flow12`/`dense24`。dense30 pressure 诊断不能单独构成 qualification；后续资格证据
   仍需 train/dev/heldout 独立性。现有 P3 hard gates 保持 `dev/eval completion=1.0`、safety、
   exact replay、K3 shortlist、low-SOC wait non-worse 和 priority-weighted wait improvement；本轮不
   新增 completion threshold。预留 `829001–829006` heldout 保持 untouched，不改作本轮开发集。

**2026-09-25 正式训练运行状态：** 用户已授权保持冻结 formal YAML、E04 source SHA、3×240
schedule、optimizer/trainable scope 和安全边界，按 `[flow6, dense24, flow12, dense24] × 60`
首次消费注册的 `850001..850240`、`851001..851240`、`852001..852240`。首轮 run0/1/2 已在首次
checkpoint 前因软件保存路径错误停止；当前 restart1 launch status 为 `FAILED`：run0
PID `2497811`、run1 PID `2497879`、run2 PID `2497946`，输出根为
`/home/user/training_outputs/terminal/p3_controlled_emergency_formal_v5_restart1_20260925/`，各 run 的
`dev_before_training/*/progress.json` 已生成，首个持久 checkpoint 已验证。heartbeat
`p3-formal-v5-restart1-monitor` 已删除；正式 campaign 终态为 `FAILED`，硬失败保留现场，不自动重跑。本轮训练不增加
随机化或高拥堵场景，不消费 heldout，不启动额外 stress；训练完成后另行独立评估 dense30 及同负荷
随机环数量/安全非均匀角度，具体评估 seeds/config 待训练结果后确定。正式训练仍不自动设置
`P3_QUALIFIED` 或 `P4_ALLOWED`，中断后不得自动重跑或替换 seed。

**历史顺序（2026-09-23，已由上方 2026-09-24 当前顺序取代）：** SOC-priority paired diagnostic 保持 FAIL，原因是 `3` 条 `FAILED_ENERGY`；immutable replay/witness attribution 已完成。下一步独立预注册 fresh-seed 的最小 repair，覆盖 low-SOC commitment gate 或有证书的长计划安全 preemption，以及 emergency pad/corridor capacity；保持现有 SafetyFilter 和安全阈值，先评估最小修复，再决定后续课程，repair 证据完成前不启动 PPO，`PPO_ALLOWED=false`。P3 high-SOC 与 low-SOC formal seeds 仍不得 formal 重用、reset 或用于调参。

1. 保留原始失败 manifest/tape、`failure_trace_evidence.json` 与 witness 文件为 immutable evidence；不 rerun、reset 或调参。
2. 单独冻结 fresh-seed 的最小 repair protocol，覆盖 low-SOC commitment gate/有证书的长计划安全 preemption，以及 emergency pad/corridor capacity；保持现有 SafetyFilter 和安全阈值，该诊断不授权 PPO。
3. 保留 support、learnability、repair、Qualification V1/V2、effectiveness、exact-proposal 与 proposal-set 全链条为 immutable evidence；已消费环境seeds不得 rerun/reset，也不得用于训练、阈值或模型调优。
4. 暂停 proposal repair、learned proposal head/loss 与 fixed proposal index 调整；V1/V2/V3/V4 及 fixed-neutral V1/V2 seeds、rows、metrics 均为 immutable evidence，不得重跑或作为新阈值调优集。
5. K=3 runtime boundary 已冻结；P2R 只改变 scheduler-owned LZ/同一步仲裁/final资源语义，不得隐式加入 runtime catalogue search 或让策略直接输出 `lz_id`。
6. P2R 当前不需要因4/108个单种子漏放行立即重训。若后续 fresh 多种子确认稳定的 individually-certified-but-not-requested 缺口，先独立预注册 train/dev/heldout seeds 和 warm+flow mixed curriculum，再决定 gate support/训练；当前诊断不得直接授权 PPO。
7. **最新 P3 停止规则：** checkpoint12 的安全 FAIL 与 checkpoint5 的 fresh Q3 完成率 FAIL 均不可改写；不得按 825006/826006 的表现继续挑选旧检查点，或将低等待（只统计完成者）当作高吞吐证据。Q3 dense24 的1080步 exact replay 显示 380秒后放行批准持续为0，后续累计 `NO_MERGE_GAP=305`、`PREDICTED_SEPARATION=344`，末态11架积压、6个LZ空闲。1800秒对照中 trained 仍停在13/24，而初始 K3 在911秒24/24；V4 fresh dev 则证明原始 K3 对所有 dense24 seeds 并非安全保证。828001 失败前60秒 509 个 actor 状态仅17次请求 RELEASE、1次获批；失败当步两架内环 UAV 均未请求 RELEASE，触发的是 `backup_existence` 递归可恢复性门槛。V4 在正式更新前已停，不得因基线失败而继续同协议或放松 SafetyFilter；下轮应先核验该备份认证失效是可行性边界还是求解器漏证，再预注册渐进密度课程和独立安全/吞吐开发集。P4/P5 与 A6 仍禁止。

完整协议、旧失败与诊断详情见[持续课程证据](evidence/v4_2/continuous_curriculum_20260917.md)；初始准入见[准入结果](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，轨迹修复与第二训练块见[续训结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。

## 不应恢复的旧路线

- V3 factorized 三头策略；
- V4 30-way 与 V4.1 flat-26 actor；
- FULL-Teacher PPO 或用 Teacher corrected action 计算 requested-action PPO ratio；
- 87D V2 checkpoint 直接续训 88D V2.1；
- 两环、8D high-level、自由 `ψ/γ/V` 或姿态/电机 action；
- 把日期型 handoff 或单次 diagnostic 当作新的 canonical specification。

## V7 run0 dev80 6-state witness（2026-09-26，离线只读）

诊断文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_state_witness_20260926/state_witness.json`。原 tape `897/897` exact 且 `outcomes_match`；模型、checkpoint、live environment 与 RNG 未改变。以下同一既有 tape 的 6 个状态均为 `certified RELEASE=0`：

| time | source P(RELEASE) | episode80 P(RELEASE) | reason_counts |
|---:|---:|---:|---|
| 136 | 0.0008189225 | 0.0005967535 | 25 `dynamics_unreachable_merge_time` |
| 152 | 0.9314221 | 0.8835936 | 25 `dynamics_unreachable_merge_time` |
| 200 | 0.0873282 | 0.1432162 | 18 `dynamics_unreachable_merge_time` + 7 `predicted_separation` |
| 250 | 0.000151486 | 0.000217653 | 25 `dynamics_unreachable_merge_time` |
| 297 | 1.622394e-10 | 1.049770e-10 | 25 `dynamics_unreachable_merge_time` |
| 500 | 0.001674179 | 0.001753280 | 18 `dynamics_unreachable_merge_time` + 7 `emergency_energy_guard` |

`filter_events=0` 是诊断插桩在 reset 重建 SafetyFilter 前仍挂接旧实例造成的，不能解释为没有 safety reject；后续仅修正临时 helper 的真实 emergency reason 读取。主审结论限定为：这些状态不能只靠提高 release 概率解决，也不能由 6 个状态证明整条轨迹不存在可行窗口或 PPO 没有责任。state witness 记录时 run1/run2 为 `109 UPDATED / 0 skipped / 0 hard`；formal 已终态为 run1 `COMPLETED 240/240`（`11737 accepted minibatches`）、run2 `COMPLETED 240/240`（`11808 accepted minibatches`），CPU `99.9%`、restore/RNG 无 false；dev120 两组均完整，各为 `84` requests（`75 normal + 9 emergency_landed + 0 failed_energy + 0 other + 0 unresolved`），`hard_failure=false`；dev200 两组均完整，各为 `84` requests：run1 `76 normal + 8 emergency_landed`、run2 `74 normal + 10 emergency_landed`，能量/other/unresolved 均为零，安全检查全通过。

## V7 run0 dev80 filter witness（2026-09-26，离线只读）

filter witness 文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_filter_witness_20260926/filter_witness.json`。原 tape `897/897` exact 且 `outcomes_match`；13 条 target filter 记录（其中 10 条为 emergency）均为拒绝。`t=152` 的 3 条均为 `dynamics_unreachable_merge_time`；`t=297`、`t=302` 各为 `predicted_separation + predicted_holder_reachable_tube`；`t=497` 的 3 条及 `t=502` 的 2 条均为 `predicted_holder_reachable_tube`。主审限定结论为：已确认正常合流预约时间不可执行与应急轨迹受交通安全约束阻挡，随后连接到能量不足；正常 safety reject 不作为 SafetyFilter bug，也不据此声称 6 个状态覆盖整条轨迹。随后已完成原失败轨迹状态副本的有界 `slot-reconciliation` off/on 候选可行性对照（同 6 点、25 网格），结果见下节；保持能量与 SafetyFilter 不变。不改冻结 formal、不重启、不把 state-only 结果当 closed-loop 效果。历史 late-slot 已有 mixed 结果，不能直接开启正式训练。formal 目标未完成，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、final evaluation 尚未启动。

## V7 run0 dev80 slot-state compare（2026-09-26，离线只读）

对照文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_state_compare_20260926/slot_state_compare.json`。prefix `2` exact；full `897` exact 且 `outcomes_match`（prefix 未全终态）；`liveconfig=false`，live environment 与 RNG 未改变。

| time | off certified RELEASE | on certified RELEASE |
|---:|---:|---:|
| 136 | 0 | 0 |
| 152 | 0 | 0 |
| 200 | 0 | 0 |
| 250 | 0 | 12 |
| 297 | 0 | 6 |
| 500 | 0 | 0 |

`on` 的 `t=250` slack 为 `195.644..236.227`，`t=297` slack 为 `165.852..192.081`；其余拒绝仍来自同一 SafetyFilter/energy 检查。主审限定结论为：旧 slot 截止时间挡住了部分经实际 endpoint 重新预约后可安全执行的候选；现有 gate 在这些状态仍保持低 `P_RELEASE`，不能只打开 flag 就宣称修复或晋级。该对照未放宽安全阈值，也不是闭环效果。随后已完成已消费 `873106` dev seed、source `03c0` 与 run0 episode80 上的有界 inference-only late-slot off/on 闭环配对，结果见下节；该对照不使用新 seed/heldout、不做 PPO、不覆盖旧 outputs，也不改变 formal。`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

## V7 run0 dev80 late-slot closed-loop pair（2026-09-26，技术完成）

输出根为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_closed_loop_20260926`。对照根状态为 `FULL_COMPLETED`，四个 cell 技术检查通过；各 `24` requests；weights unchanged、`Teacher=false`、`PPO=false`、安全 counter 为 `0`、各 tape exact。source off 为 `19 normal + 5 emergency_landed + 0 failed_energy`，source on 为 `15 normal + 9 emergency_landed + 0 failed_energy`；episode80 off 为 `18 normal + 5 emergency_landed + 1 failed_energy`，episode80 on 为 `14 normal + 10 emergency_landed + 0 failed_energy`，其余 `other/unresolved` 均为 `0`。

| cell | steps | outcomes (normal / emergency_landed / failed_energy) |
|---|---:|---|
| source off | 761 | 19 / 5 / 0 |
| source on | 881 | 15 / 9 / 0 |
| episode80 off | 897 | 18 / 5 / 1 |
| episode80 on | 825 | 14 / 10 / 0 |

四个 cell 的 full tape 均 exact；同算法 model digest 一致，之前不同算法摘要不作直接比较。该对照消除了已知单例的 energy failure，但 source 与 episode80 两组的 normal 均下降 `4`，emergency 分别增加 `4` 与 `5`，不构成晋级依据，也不改变 formal 或触发重启。旧 lifecycle 只统计 `completed_uavs`；episode80 off 漏计 `1` 个 failed（`23 rows / lowSOC=9`，on 为 `24 / 10`），旧等待指标仅保留为成功样本条件统计；同分母结论使用下文 v3 全请求数据。v3 lifecycle observer 已完成四个 tape 的 full-request terminal wait 核对，结果见文档末尾。下一步先将已验证的 failure-inclusive 逻辑固化到独立离线评估指标路径并做针对性测试，不改正在运行的 formal 依赖；随后由 primary 准备并冻结与 proposal 随机 K3 一致的外部 gate 标签/小 probe 协议，保留旧 flow/priority 表现。当前不宣称已适配或开启新训练。formal 目标未完成，`P3_QUALIFIED=false`、`P4_ALLOWED=false`、final evaluation 尚未启动。

## V7 shortlist 与冻结 K3 语义核对（2026-09-26）

`terminal_marl/training/train_p3_gate_coverage_repair_v2.py` 的报告明确：`state_only_gate_feasibility_over_all_26_candidates` 不是 neural K3 success guarantee；collector 的 shortlist 只用于 replay，不作为 label 判据。既有 witness/slot compare 中，`t=250` 的 frozen proposal top3 为 `[11, 10, 13]`，late-on certified 集合与其相交于 `11/10`；`t=297` 的 top3 为 `[10, 14, 13]`，late-on certified 为 `[15, 22, 21, 23, 20, 24]`，无交集。runtime 使用 sampled proposal 排第一，再加最高的其余两个 proposal（`student_rollout_v4_2.py` lines `94–143`）；因此 deterministic top3 仅是诊断参考，不等同所有 stochastic shortlist。

formal run1/run2 继续使用原协议。若后续研究外部 gate 适配，标签需对齐冻结 proposal 的实际 K3 执行能力，并先进行有界采样/输出预检和小 probe；当前未启动该适配，不解冻 proposal、不增加强制 RELEASE、probability floor 或在线 Teacher，也不把 catalogue 可行性等同于闭环成功。

## V7 run0 dev80 slot lifecycle v3（2026-09-26，full-request 核验完成）

结果文件为 `/home/user/training_outputs/terminal/p3_adaptive_kl_run0_slot_lifecycle_v3_20260926/manifest.json`。四个 tape 均 exact；每个有 `24` 个 unique terminal rows，完整 wait、priority、initial SOC、ledger、outcome 均匹配，独立算术复核吻合。v1 漏记 completed 路径、v2 使用错误 key；两份失败产物保留为历史记录，不作为模型失败证据。

| cell | priority-weighted wait（24 rows） | low-SOC wait（10 rows） |
|---|---:|---:|
| source off | 159.201419 | 145.302365 |
| source on | 158.035714 | 158.798314 |
| episode80 off | 213.842109 | 210.452604 |
| episode80 on | 174.378916 | 137.549010 |

原 episode80 off 的成功样本条件统计为 priority `165.793583`、low-SOC `147.079037`（low-SOC 9 rows），不再与完整 24-row/10-row 结果作同分母比较。arrival15 初始 SOC 为 `.5566427132`：off wait `780.8147076`，`t=896` `failed_energy`、touchdown null；on wait `208.8147076`，`t=561` terminal `emergency_landed`。late-on 对该已知单例消除了 energy failure 并降低全请求等待，但 source 与 episode80 的 normal 均减少 `4`，emergency 比例分别为 `5/24→9/24` 与 `5/24→10/24`；不晋级、不修改正式配置、不重启 run0。formal 终态为 run1 `COMPLETED 240/240`、`11737 accepted minibatches`，run2 `COMPLETED 240/240`、`11808 accepted minibatches`，restore/RNG 无 false；dev200 两组均完整，各为 `84` requests：run1 `76 normal + 8 emergency_landed`、run2 `74 normal + 10 emergency_landed`，能量/other/unresolved 均为零、安全检查全通过；dev120 两组均为 `75 normal + 9 emergency_landed`、共 `84` requests、无 energy/other/unresolved、`hard_failure=false`。

下一步先将已验证的 failure-inclusive 逻辑固化到独立离线评估指标路径并做针对性测试，不改正在运行的 formal 依赖；随后才登记并由 primary 审查、冻结与 proposal 随机 K3 一致的外部 gate 标签/小 probe 协议，先做小样本输出检查。保留旧 flow/priority 表现，不解冻 proposal、不增加强制规则、不启动新 formal 或恢复 run0，final heldout 仍未消费。以上只记录计划，不表示公共旧 lifecycle 统计已修复或外部 gate 已适配。
## V7 lifecycle metrics 与 K3 feasibility helper 状态（2026-09-26，最终核验）

独立 `terminal_lifecycle_metrics` 模块已接入全请求 terminal wait 指标路径；final evaluator 尚未启动，legacy 成功样本统计保留，formal 依赖未修改。run0 episode80 off 的真实 tape 为 `897/897` exact，`24` requests：`18 normal + 5 emergency_landed + 1 failed_energy`；全请求 priority-weighted wait 为 `213.8421085607388`，low-SOC `10` 个样本均值为 `210.45260434162256`。该 `897` exact 核验仅由 stdout 记录，未生成新的持久化文件。

新的 `k3_gate_feasibility` 纯 helper 按实际 sampled proposal `j` 加其余最高的两个 proposal 工作；三组 metrics/eval/K3 定向测试最终 `28 PASS`，py_compile 通过。六个既有 witness 的质量值为 `0, 0, 0, 1, 8.6933744925e-10, 0`，该值基于保存概率的近似 rank，仅用于诊断，不是训练结果，也不是闭环成功概率。

聚合规则已在独立模块与测试中固化：所有 `record.requests`（含 legacy、missing、zero-row）均计入分母并保留 missing；`landingSOC` 只统计成功且 touchdown 可用的记录，failed SOC 单列为 `terminalSOC`。这不表示公共旧 lifecycle 函数已修改。

正式 run 终态为 run0 `FAILED`（`80 UPDATED`、`3954 accepted minibatches`）、run1 `COMPLETED 240/240`（`11737 accepted minibatches`）、run2 `COMPLETED 240/240`（`11808 accepted minibatches`）；restore/RNG 无 false。dev200 已完成，各为 `84` requests：run1 `76 normal + 8 emergency_landed`，run2 `74 normal + 10 emergency_landed`，能量/other/unresolved 均为零，安全检查全通过。final evaluation 未启动，不宣称 P3 结业，不重启 formal。下一步由 primary 冻结与实际 K3 随机执行一致的外部标签/小 probe 协议；fit 尚未启动。

## V7 reward 与 actor credit accounting（2026-09-26，代码事实）

`student_rollout_v4_2.py:349` 将每个 UAV 的环境 reward 汇总为 `sum(rewards.values())` 后写入 team scalar；`StudentStepV42` 与 GAE 使用 `gamma=.99`、`lambda=.95`（`student_ppo_v4_2.py:20–33,67–75`）。每个 world step 的 advantage 复制给该步的各 actor；没有 actor step 的 UAV 不进入 actor loss，但仍保留在 GAE/critic（`student_ppo_v4_2.py:197–221`），没有额外按 actor 数加权。

当前 reward 组件事实为：urgency `=-.01*dt*(clip((priority-1)/4)+2*clip((.65-currentSOC)/(.65-.25)))`，time `=-.01*dt`，approved release `+.1`、reject `-.05`，touchdown `+25`，emergency 额外 `-8`（净 `+17`），energy `-20`。后处理的 priority-weighted wait 为 `(external+internalwait)*basepriority` 的加权指标，low-SOC 按 `initialSOC<=.65` 分组；两者与 reward 不是同一目标。本段只记录实现事实，reward 本轮保持冻结，不据此单独归因性能变化。

## V7 K3 proposal coverage diagnostic（2026-09-26，完成）

完整诊断根为 `/home/user/training_outputs/terminal/p3_k3_coverage_diagnostic_v1_20260926`，manifest SHA256 为 `0a1d329a929bdaf0131cf052fddd918ad0695d0db7245fedf3eb61b798006690`。任务 `COMPLETED exit0`，包含 `6` 个 source report 与 `1` 个 witness report：`36` 个 source states、`6` 个 witness states，missing `0`；7 份报告均 exact、outcomes/RNG/model 一致，`policy_update=false`、`teacher_online=false`。source 的 off/on 各有 `22` 个 catalogue 可行状态；`q` 分布为 `q=0:14`、`0<q<.01:3`、`.01<=q<.5:0`、`q>=.5:19`。source 中没有同时满足 `q>=.5` 且 `P_RELEASE<.1` 的样本，因此不能据此表述为广泛 gate 抑制。

三个低 q 但 catalogue 可行的 source 状态为：flow6/seed873101/step187/arrival2/outer/SOC `.719587`、priority 1、`q=1.09908e-25`；dense24/seed873103/step0/arrival0/outer/SOC `.821394`、priority 1、`q=6.3502e-19`；dense24/seed873106/step0/arrival0/outer/SOC `.832284`、priority 3、`q=5.32975e-19`。witness 单列为 off `0/6`、on `2/6` 可行；`t=250` 的 on 状态 `q=1`，四个 checkpoint 的 `P_RELEASE` 为 `.000151486/.000217653/.000386301/.000515108`；`t=297` 的 on 状态 `q=8.69337449e-10`，即使 gate 为 1 也几乎选不到 certified proposal。`42` 个状态在 `4` 个 checkpoint 上共 `168` 组 proposal logits 与 source 逐组一致。`q` 是单状态 certified 尝试概率，不是最优 RELEASE 标签或闭环成功概率。

smoke_v3 位于 `/home/user/training_outputs/terminal/p3_k3_coverage_diagnostic_v1_20260926_smoke_v3`，`SMOKE_COMPLETED`、exact `69/69`，首步观测 `88D`，off/on certified 候选为 `24/25`，proposal map `0..24`，RNG/live/weights unchanged；新模块两项定向测试和编译通过。更早 v1 的 `StopAfterFirstSample` RuntimeError 被 SafetyTrace 捕获而未 exact，v2 因 proposal mapping incomplete，均保留为软件预检失败，不作为模型失败。

primary 对该诊断的限定解释是：slot 机制、少数 gate 压制和冻结 proposal 覆盖共同出现在这些观测中，不能由单独训练 gate 或 probability floor 整体解决；观测状态不代表总体泛化。reward、安全/能量边界保持不变，不重启 formal、不消费 heldout，未启动 fit；受限 proposal 适配范围待用户确认，`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

## V7 run0 dev_after_0080 outcome totals erratum（2026-09-26）

服务器 manifest：`/home/user/training_outputs/terminal/p3_adaptive_kl_formal_v7_20260926/run0/dev_after_0080/manifest.json`。逐 record 复核的六个 dev seed 请求终态按 manifest 顺序为：

| dev record | normal | emergency_landed | failed_energy |
|---:|---:|---:|---:|
| 1 | 6 | 0 | 0 |
| 2 | 12 | 0 | 0 |
| 3 | 22 | 2 | 0 |
| 4 | 6 | 0 | 0 |
| 5 | 12 | 0 | 0 |
| 6 | 18 | 5 | 1 |
| 合计 | 76 | 7 | 1 |

因此该组 `84 = 76 + 7 + 1`，`other/unresolved=0`。文档此前的 `74 normal + 9 emergency_landed + 1 failed_energy` 汇总为勘误后的错误数字，现以 manifest 六条明细为准；不据此推断来源，run0 的退出/失败定性保持为开发能量 hard gate 触发。

## V7 proposal-head probe V1（2026-09-26，fit 终态；后续评估见下节）

用户已授权受限 proposal-only 适配。注册协议、collector、fit runner 的提交分别为
`350dc1e`、`170c184`、`299d3d1`；注册协议文件未修改。实际 collection manifest 为
`COMPLETED`，六个 source records 均 exact、outcomes/RNG/model invariant 通过且无 errors，
共保存 `164` 行（train `82`、validation `82`；相对最多 `192` 行缺失 `28`），rows SHA256 为
`595f644e5d7376a44fe1be98ee8bfe997d9ee4856c102050df4b8323f1004ba8`。

唯一一次 fit 使用固定 03c0 source（SHA256
`03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`），只更新
`proposal_head.weight/bias` 共 `2425` 个参数；gate、encoder、critic、temperature、SafetyFilter、
energy/reward、late-slot OFF 与 K=3 语义保持冻结。fit 在第 `12/40` 个 epoch 因 train retention
KL cap 停止：`35` accepted、`1` rejected，proposal parameter delta norm 为 `0.65643138`，
train KL 为 `0.04770959`，validation KL 为 `0.10888803`。状态为
`STOPPED_KL_CAP`，没有重试或自动调参。

proposal-head checkpoint 为
`/home/user/training_outputs/terminal/p3_proposal_head_probe_v1_20260926/fit/p3_proposal_head_probe_v1.pt`，
SHA256 为 `d7a0908b30395ddaa90cb1ffe6f2f5c0d49176e946927ea9fb5a6b44bdb0bade`。fit 的 train qK3
均值由 `0.5121951158` 变为 `0.5121951176`，validation 由 `0.4999999951` 变为
`0.4999999959`，未见有意义的整体 coverage 改善；只读复核中 source `qK3<.01` 且 C 非空的
4 条状态仍全部 top3 miss，164 条中 top3 改变 `30` 条，source-covered→candidate-uncovered 为
`0` 条。该段记录 fit 完成时的状态；六个 development seed 的 corrected frozen evaluation 见下节。

本结果只记录 proposal-only 开发试验，不能写成 formal training 完成、P3/P4 通过或 heldout 结果。
旧 p3-v7 automation 已经工具核实不可用/删除，当前不声称存在 active automation；正式训练、heldout
与后续晋级仍保持未启动/未消费状态。

## V7 proposal-head probe V1 corrected development evaluation（2026-09-26，终态）

校正后的离线评估产物为
`/home/user/training_outputs/terminal/p3_proposal_head_probe_v1_20260926/eval_summary_corrected/manifest.json`，
SHA256 为
`f1db01557c112817b1e84501236242dfec077324702d9b4b335e366fd7e78354`，状态为
`REPORTING_ERROR_CORRECTED`，对应六个 development records。原始评估 manifest 的 SHA256 为
`4c0b48d1503ea70c621a9595049c73d79e1e41b6152f7ab90f1e5e57f6f2ffaa`，source 03c0 SHA256
仍为 `03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba`；原始记录已保留。
校正使用 evaluator 修复提交 `46c27db` 的离线汇总，未重新 rollout、fit 或调用 training/update
（`policy_forward_executed=false`、`training_or_update_called=false`、`environment_replayed=false`）。

| split | baseline normal / emergency / failed_energy / other / unresolved | proposal-head normal / emergency / failed_energy / other / unresolved | priority-weighted wait baseline → candidate | low-SOC wait baseline → candidate |
|---|---:|---:|---:|---:|
| overall (84) | 76 / 8 / 0 / 0 / 0 | 75 / 9 / 0 / 0 / 0 | 86.6690576161 → 92.0527228621 | 89.7792692439 → 89.6788480393 |
| train (42) | 39 / 3 / 0 / 0 / 0 | 40 / 2 / 0 / 0 / 0 | 70.6967099479 → 81.1172039290 | 62.4444731092 → 47.3267282907 |
| validation (42) | 37 / 5 / 0 / 0 / 0 | 35 / 7 / 0 / 0 / 0 | 102.9545885720 → 103.2026637351 | 115.0113887529 → 128.7731124226 |

六个 record 均 exact/aligned，weights unchanged；SafetyInvariantError、collision、
hard-separation、NMAC、uncertified-execution 等安全计数均为 `0`。校正报告明确
`admission_gates_reconstructed=false`、`checkpoint_promotable=false`、`development_screen=false`、
`P3_QUALIFIED=false`、`P4_ALLOWED=false`，且不包含 heldout。该结果将本次 proposal-only 用户授权试验记为已完成但未通过 development screen；它不表示 formal training 完成或 P3 晋级。

原评估中的 `KeyError: 'hard_gates'` 属于报告阶段错误；校正报告复用了已保存的六个原始 tape/record，未重跑环境。下一步由 primary 在再次训练前安排有界的现有状态 counterfactual 检查，分别核对 WAIT/RELEASE utility/credit 与 proposal coverage；在该检查前不自动重复 fit、改变 KL 或启动新的 PPO，且不新增 protocol/seed。

## V7 runtime action-credit / K3 branch labels（2026-09-26，partial diagnostic；路径已验证）

已验证 runtime 输出根 `/home/user/training_outputs/terminal/p3_action_credit_runtime_labels_20260926`：`16/16 COMPLETED`，每个状态 `51` 个 branches、共 `816` 条；exact、outcomes、state、model、RNG 均为 `true`，命名 safety counters 无非零。新 `51` branches 是 branch-level singleton labels，但本轮仍是 `16` 状态 partial diagnostic。旧 `/home/user/training_outputs/terminal/p3_action_credit_20260926/manifest.json` 的 `16` 状态只含 WAIT+`25`K3，同样是 partial diagnostic，不称 singleton，也不是完整训练集。

原 report 的 FP/FN 两个字段名称相反，但原始 `C` 与 branch 内容正确。代码修复后的离线更正 summary 状态为 `REPORTING_LABEL_NAMES_CORRECTED`，路径为 `/home/user/training_outputs/terminal/p3_action_credit_runtime_labels_20260926/reporting_correction/summary.json`；同组检查 `34 tests PASS`，原始报告保留，不 rerollout。

两条 catalogue false positive 为：`873103/0` 与 `873106/0` 的原始 `C=[2]` 更正为 `C=[]`，对应实际 `25` 个 K3 proposal 全部 `safety_reject`。`873101/188` 的 `C=[20,21,22]` 与 `873102/311` 的 `C=[2,3,4,9]` 保持不变，其余 `12` 个 `C` 也保持不变。由 singleton `C` 推导的 runtime K3 概率与真实 `25`K3 weighted approved 的最大误差为 `5.029141902923584e-08`。详细记录见 [runtime action-credit / K3 labels](evidence/v4_2/p3_action_credit_runtime_labels_20260926.md)。

本轮修复诊断与执行上下文标签路径已验证；这不表示训练完成或性能改善。静态 catalogue 可行性不等于实际 step 执行上下文可行性，两个 step0 false positive 不作为 proposal 漏覆盖训练样本。rank loss 候选尚未 fit 或接入 formal；训练退化与 P3 qualification 尚未解决。WAIT/RELEASE utility 属于分支比较的 one-step critic proxy，不是长期 advantage；它不定义 rank surrogate 的训练目标。reward、安全边界与 heldout 不变。四条 [source logit counterfactual](../results/p3_k3_logit_counterfactual_20260926.json) 只说明 logit-space 反事实，不等于 runtime 改善。

下一步先从现有 source tapes 补齐完整 runtime labels，再做一次 bounded rank-aware proposal 修复与 `6` 个 development seed evaluation；不直接启动 formal 或消费 heldout。reporting correction 已完成，但训练与 P3 qualification 仍未通过，也不据此作 PPO credit 根因或同时机预约抢占结论。
