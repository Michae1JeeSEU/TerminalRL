# 2026-10-29 selective repair v2：正式结果与有限同状态诊断

两条开发例均完成300秒并通过保存回放自动审计。正常着陆增加，但完整normal计算工作增加，高流量外环等待也增加。本轮保留该版；不自动进入B，不继续扫描权重或lead。人工完整GUI回放仍为 **HUMAN_PENDING**。

## 协议、版本与验证范围

正式实现固定为 `opportunity_cooperative_v2`、`coalesced_restart_v2`，nominal K8/max12、lead20、heartbeat5、preparation1、handoff4；正式offset0在诊断前已固定。没有学习、动态K或提前量扫描，没有改变几何、物理、目标和应急规则。max12允许有界扩展，不能把每轮实际K都称为8。

物理实验冻结提交为 `895bc76fd226cbc4a0ebb4685e73c14f05da7ff5`，source hash为 `ccb391229af7f04c039a5ca723e365e21b52ce0b01ab15dc8ec15d00de6c725b`，清单含1210个文件。两个根从原Stage A相同t0各运行一次：s3r23/seed7324/CPU12、s3r24/seed7325/CPU14，均为300秒、每臂child wall上限1200秒、数值库线程1。旧follow-up baseline20只是**整体版本前后参照，不是全量公平对照**；不能隔离各修改的因果贡献，也不能从两个开发例作统计推广。

报告为后续只读后处理版本，与物理冻结版本分开：报告SHA256为 `0fe76bbcd2e393a1f25e20a9d181926f0652a0c87003c85aa29575e5ec9571a7`，对应测试SHA256为 `73fdf56a1d2b71593c5823222fd91d43c84467232dc1b737bb7aa9d949ad60c5`。后续报告提交不替代物理实验的895bc76源绑定。

核心服务器回归 **225 passed**，最终报告定向检查 **21 passed**，均0 failure/error/skip。旧八臂及新两臂保存回放audit均PASS，`full_a20_covered=true`；覆盖连续间隔CERTIFIED、资源/C2及未发布提案非执行PASS，PVA状态CHECKED。自动audit与人工GUI验收分开。

## 300秒结局与各环等待

下表全部为旧baseline20→新v2；两版均无上游pending、无reroute，计数守恒。总能耗与holding飞机秒是不同量。

| 开发根 | 正常着陆 | 应急/失败 | 空中未完成 | 总能耗MJ | holding飞机秒 |
|---|---:|---|---:|---:|---:|
| s3r23 medium | 10→14 | 0/0→0/0 | 30→26 | 568.791642→552.523207 | 6110.303399→5385.848641 |
| s3r24 high | 14→15 | 0/0→0/0 | 35→34 | 808.276453→799.097823 | 8432.009856→8750.572076 |

各环holding时长采用 `DERIVED_EXECUTED_HOLD_ENERGY_DIV_CANONICAL_POWER`：已执行holding能耗除以保存配置的固定环飞battery power。外/中/内环功率分别为68078.009137、74812.869720、92773.128471 W。四个cell的派生和与记录总holding时间误差绝对值均小于2.24e-9秒，通过1e-6秒核验；没有用总能耗代替环等待。

| 根/环 | holding飞机秒：旧→新 | 已执行holding能耗MJ：旧→新 | 终点该recorded ring空中未完成：旧→新 |
|---|---:|---:|---:|
| medium 外 | 3281.735047→3243.328511 | 223.413989→220.799348 | 20→18 |
| medium 中 | 2436.568351→1408.203987 | 182.286671→105.351781 | 8→6 |
| medium 内 | 392.000000→734.316143 | 36.367066→68.124806 | 1→1 |
| high 外 | 4391.149669→4724.149208 | 298.940727→321.610673 | 17→21 |
| high 中 | 3009.494080→2992.368721 | 225.148889→223.867691 | 11→8 |
| high 内 | 1031.366107→1034.054148 | 95.683060→95.932438 | 6→4 |

终点计数来自保存最后一帧真实aircraft ring/status，转移中飞机仍按recorded ring计入，不换成projected node。四个cell另各有1架ring=None的INBOUND，单列unmapped，因此环内计数加unmapped分别等于30/26/35/34。

高流量外环holding增加约333飞机秒、recorded-ring未完成17→21，HOLD_OUTER本身15→20。保存执行事件中的O→M开始11→6、M→I开始10→9、normal final开始18→17；着陆14→15还取决于动作时间分布，不能用转移数代替服务质量，也不能称为全面改善。

## 证书、最终ready与执行续接

`selective_replanning.first_certified` 旧字段实际在 `scheduler.plan` 返回后记录，现报告命名为 **first_certified_call_return**，包括合法WAIT；它不是首份任意证书生成瞬间。真实首个非空证书取 `search_handoff.first_nonempty_certified`；没有直接first-any事件时，首任意证书瞬间保持null/unknown。call-return晚于非空证书不是时钟异常。

下表首证书为elapsed wall秒；ready、publish、动作开始及下一snapshot为simulation绝对秒，t0任务均从simulation 0开始。

| 根 | t0首个非空证书wall：旧→新 | 最终ready sim：旧→新 | 实际publish/动作开始sim | 下一snapshot sim：旧→新 |
|---|---:|---:|---:|---:|
| medium | 6.803269→4.140097 | 9.836828→12.392826 | 均20/20 | 25→20 |
| high | 7.184440→4.458432 | 12.184067→13.901750 | 均20/20 | 25→20 |

首个非空证书更早，但最终ready更晚，必须同时阅读。所有有直接非空证书的任务中，证书wall中位数medium为6.532→3.544秒（n=16→14），high为6.913→4.370秒（n=14→14）；这是有证书任务的条件统计，任务组成和闭环轨迹不同，不能归因为纯计算优化。

新medium的20次、高流量的15次已结束任务→下一启动gap全部为0，并有明确coalesced记录。旧medium/high分别15/13次gap均非零，中位2.612850/2.380307秒、累计37.239235/30.949650秒，首次均多等5秒。**执行后的空等修复已有直接时间线证据。**

新medium在snapshot80至99.222680处记录TRAFFIC_CHANGED_AFTER_DEADLINE；新high在167.004766至186.035085处记录READY_DEADLINE_MISSED，其ready deadline为186.004766。后者保存normal work7.45414、prepare0.85109及charged public service9.71535秒，支持共同在线服务占用ready窗口的解释，不支持CPU抢占归因。新medium/high的NO_EXECUTABLE_PREFIX次数为7/2，旧为4/0；每个cell终点仍有1个pending normal task，其已消耗工作计入下节。高流量最后一条pending revalidation的REVALIDATION_WORK_DEADLINE不能冒充已完成拒绝。

## 实际K、回退、接入与有效复查

旧high存在3次MANDATORY_OVERFLOW_FULL，实际K为29/32/32；新两根均0次overflow，medium旧版也为0。超龄触发的全量突发在这两例中消除，但这不表示公平欠账已解决。新返回scope实际K分布如下，保留有界扩展行为：

| 根 | 新实际K：callback数 |
|---|---|
| medium | 8:11，9:8，10:1，11:1 |
| high | 8:5，9:4，10:7 |

任务接入以已结束的非空offered任务为分母，排除pending和no-prefix；飞机机会接入以各callback返回selected集合中的飞机机会为分母，包含pending callback；去重飞机接入是另外一个分母。接入计划、飞机和着陆数不混计。

| 根 | 任务接入：旧→新 | 已接入飞机机会/selected机会：旧→新 | 去重接入飞机/去重selected：旧→新 | 实际提交计划：旧→新 |
|---|---|---|---|---:|
| medium | 6/11→12/13 | 17/130→30/181 | 13/34→22/25 | 17→30 |
| high | 11/13→12/13 | 39/181→32/146 | 29/48→20/27 | 39→32 |

v2要求有效完成复查后才更新review证据，定义比旧版严格，不能直接横比旧age。新最后一次snapshot分别位于297.832029/273.651658秒，review age≥60的债务飞机仍有19/29架，最大age为297.832/273.652秒。high的M01/M02分别被fair队列选中10/9次，但评估UNKNOWN分别出现16/15次；这是评估状态，不等于age字段缺失。**反复选中而有效复查未闭合，公平问题仍在。**

非着陆原因标签保存在summary的 `nonlanding` 中，可并存且不构成因果。UNSELECTED依据返回selected集合，不是扩展尝试并集；旧7份扩展后保留较早best的记录不能解释为从未考虑。NO_VALID_CANDIDATE只表示已返回有限目录中未观测有效MOVE，UNKNOWN不补成物理无解；JOINT_CONFLICT只用明确保存的飞机ID证据。CP预测后缀、被拒提案和“已考虑”均不算执行。

## 完整计算工作与独占阶段

完整normal work包含normal、准备、重验证、依赖守卫、发布等实际工作及pending已消耗部分；online service诊断不再重复相加。job wall另含环境恢复、物理推进、记录和I/O，不能当作纯调度耗时。

| 根 | 完整normal work秒：旧→新 | 全job wall秒：旧→新 | process CPU秒：旧→新 | LocalPlanner calls：旧→新 |
|---|---:|---:|---:|---:|
| medium | 145.840646→170.456709 | 410.865480→467.761820 | 409.264330→463.923177 | 24→42 |
| high | 134.044976→165.746352 | 495.841606→568.740195 | 494.861175→564.345717 | 18→32 |

独立CPU12/14、线程1，process CPU接近wall，本轮记录未提示CPU抢占。新每callback两批，scope expansions为medium8→10、high4→11，与验证工作增加同时出现；没有配对消融，不能把增量单独归因为缓存或某一个修改。

按snapshot_time连接完整per-task work，pending部分保留。各组统计如下；UNKNOWN组均无任务，其median/p95为null，不补0：

| 根/版本 | scope组 | 任务数（含pending数） | 总秒 | 中位秒 | p95秒 |
|---|---|---:|---:|---:|---:|
| medium旧 | LOCAL | 16（1） | 145.840646 | 8.346617 | 12.487231 |
| medium新 | LOCAL | 21（1） | 170.456709 | 8.227481 | 12.394293 |
| high旧 | LOCAL | 11（1） | 95.199168 | 8.288278 | 11.641416 |
| high旧 | OVERFLOW_FULL | 3（0） | 38.845808 | 12.879920 | 13.714480 |
| high新 | LOCAL | 16（1） | 165.746352 | 9.803715 | 13.646107 |

下表是全部callback的主要**独占phase wall总和**，不是完整pipeline；嵌套计时已扣除，不再与dispatch或完整normal work相加。phase减少不能直接抵消整个pipeline的其他工作。

| callback独占phase秒 | medium旧 | medium新 | high旧 | high新 |
|---|---:|---:|---:|---:|
| candidate_generation_and_single_vehicle | 63.956709 | 54.673637 | 57.721419 | 32.520041 |
| trajectory_validation | 40.144722 | 68.417484 | 45.048537 | 76.153970 |
| prefix_validation | 14.088768 | 28.796149 | 14.949773 | 36.481043 |
| cp_solve | 21.842075 | 5.166288 | 6.950555 | 2.798079 |

共享global frame另记录medium0.115278→0.142156、高0.145383→0.142113秒；selector为medium0.031011→0.092043、高0.048062→0.132579秒。near-term feature没有独立phase时钟，保持未单独观测，不能补成0或再添加一笔费用。候选/CP变少、轨迹/前缀验证变多，而完整normal work上升，**不能宣称整体提速**。

保存geometry evidence累计hit率约medium2.21%（3770/(166984+3770)）、high1.79%（2999/(164210+2999)）。纯几何cache key不依赖projected UUID；此处不把repr身份问题和该几何缓存直接相连。大量eviction不能单独归因耗时；没有缓存容量/关闭缓存的配对计时，不能声称缓存带来净加速。

## 固定t40的有限同状态比较

对旧baseline20同根 `checkpoint_0040.json` 冷恢复、lead20，只准备、不推进、不发布。每根offset0/1/2的初始S完全相同，**每根只有1种不同的默认S**；不取三组平均，不当作三份独立子集样本。表中默认列仅取offset0，FULL_H为同版本公共过程。

| 固定根/方法 | 初始K | 首个非空证书wall秒 | 不发布准备结果 |
|---|---:|---:|---|
| medium默认K8 | 8 | 3.833652 | PREPARED_COUNTERFACTUAL |
| medium同版FULL_H | 18 | 6.292256 | PREPARED_COUNTERFACTUAL |
| high默认K8 | 8 | 无直接记录 | NO_PROPOSAL |
| high同版FULL_H | 32 | 无直接记录 | NO_PROPOSAL |

准备通过的token均已丢弃；没有非空组不表示物理无解。该有限比较不是FULL_H的300秒闭环公平对照。

生产projected version和frame digest的repr输入包含Section默认内存地址，使同内容生产ID不同，原offset身份驱动去重未能归并；三次实际计算成本全部保留。每根另独立纯重建两遍，projected/frame ID均不同，但递归规范化fullpayload完全相同：

| 根 | 两次canonical内容SHA256 |
|---|---|
| medium | `42ee047e85155800376fb7901ad83a883386090ced46497100b8b3a7ffb98353` |
| high | `0aabbc1e0563398cfecae43f2f91e6538dbfa00ba4e7a7947b2b5499bc67345d` |

规范化仅移除snapshot根version及frame根snapshot_id，保留物理版本及精确数值，**只作内容比较证据**；未改生产hash或缓存授权key，也不比较或声称token ID相等。此前八次固定t40 snapshot没有各自完整canonical内容，不能追溯声称八次都已逐字段canonical相等。另六个初始附加snapshot因父CPU亲和性与参数不符而在set_cpu、求解前失败；原failed ID/费用和之后独立recovery ID均保留，不是为了取得更好结果重跑。

### 手工子集A/B

这里只在隔离进程中临时适配selector并finally恢复，复用冻结callback/LocalPlanner/CP及counterfactual preparation。原high S为 `{I05,I06,I07,I08,I09,M00,M01,M10}`；I/M/O缩写initial_inner/middle/outer。A以O00替换M01，B再以O01替换I07。两组均K8、hard=[]、fair=I05/I06，eligible和真实definite依赖闭包通过，未纳入执行中的I02/I04；正式规则和offset0没有改变。

| 诊断 | 初始S | 首个非空证书wall秒 | 准备/物理执行 |
|---|---|---:|---|
| A | I05/I06/I07/I08/I09/M00/M10/O00 | 8.005776 | PREPARED_COUNTERFACTUAL；未执行、未发布 |
| B | I05/I06/I08/I09/M00/M10/O00/O01 | 10.314349 | PREPARED_COUNTERFACTUAL；未执行、未发布 |

两组source physical before/after不变，canonical内容SHA均与high纯重建相同。保存的可准备单前缀为O00→middle slot1，hash为 `9ecf969e672d3c17cb337a2c04d1c426b09b6020f9f37d4af426d9620bafe392`，token已丢弃，actual submitted hashes为空。它说明默认S**遗漏了可准备的局部动作/组合**。

预先设想的释放链涉及M00的middle slot0，实际找到的是O00→middle slot1，不能把诊断理由当成已实现的机制。这里只有一条未执行前缀，**没有证明I→M→O联合释放链、实际着陆或闭环收益，更不保证阶段B有效**。

## 本轮判断、预算与回放

执行后的等待修复、这两例中的全量突发消除和较早非空认证已有保存证据；完整计算没有节省，成本更多落在轨迹/前缀验证且callback增多。两例正常着陆增加，但high外环积压退化，有效复查/候选覆盖缺口仍在。本轮保留该版，不自动进入B，不继续调权重或lead；后续定位方向是验证复用开销及有效审查、候选覆盖缺口，不预设学习一定能改善。

新额度5400 child-wall秒，最近已记账 **1477.5308457711415秒**，含先前旧回放audit82.8723364593秒。summary生成时嵌入的账本是较早快照，最终报告job等随后入账，以最新独立ledger为准。历史累计7323.47022066447秒单独保留，不重置。早期两次local async test仅有pytest内部9.51/22.04秒，完整child wall未知且未入账，因此该数不声称穷尽本轮所有耗时，也不声称精确剩余额度。

主对话AI已检查新medium/high t70、t300及旧对应关键帧；high新t300外环更集中，与真实保存计数一致。这是**AI关键帧检查完成**，不是用户完整GUI验收，HUMAN_PENDING保持不变。

本地新回放目录为 `D:/7_codespace/terminal/artifacts/rs_selective_repair_review_20261029/new_results/arms/<root>/evaluation/<root>/RULE_LOCAL_K8/replay`。使用既有 `ReplayController`、`RotatingSlotsReplayViewer` 打开对应目录。关键帧：[medium t300](D:/7_codespace/terminal/artifacts/rs_selective_repair_review_20261029/new/s3r23/t300/top_down.png)、[high t300](D:/7_codespace/terminal/artifacts/rs_selective_repair_review_20261029/new/s3r24/t300/top_down.png)。

## 直接证据

数据根为 `D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029`，服务器对应 `/home/user/training_outputs/terminal/rs_selective_repair_20261029`。本稿只读保存数据，不追加实验。

- [最终分析证据包](D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029/analysis_evidence.zip)：`repair_analysis_summary.json`、`REPAIR_REPORT.md`、21项最终XML、两新audit、原始任务/调度记录与报告源SHA；summary的scientific_acceptance是生成时PENDING_PRIMARY_REVIEW，本稿记录主对话本轮研究判断。
- [正式证据包](D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029/formal_evidence.zip)：两臂boundary0300、执行状态及回放审计。
- [服务器预检](D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029/preflight_evidence.zip)、[smoke](D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029/smoke_evidence.zip)、[固定snapshot v2](D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029/snapshot_evidence_v2.zip)及[手工A/B和canonical](D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029/fixed_scope_evidence.zip)。
- [最新共享预算账本](D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029/repair_budget_ledger.json)与[旧八臂audit](D:/7_codespace/training_outputs/terminal/rs_selective_repair_20261029/prior_audits)。非着陆标签、完整时序、phase、实际K/overflow原因及所有null证据均保留在机器可读summary，不用未记录值补齐结论。
