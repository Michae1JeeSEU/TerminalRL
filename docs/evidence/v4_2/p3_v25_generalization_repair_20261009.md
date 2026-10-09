# P3 V2.5 独立评价失败后的泛化诊断与修复

## 已有结论与当前范围

用户已授权持续修复至新P3正式训练就绪并在服务器实际启动；不因本次失败结束。物理/证明/执行监测修复后的完整student与规则Q均无实际安全、hash、耗尽和未决阻塞，详细历史见`p3_v25_nominal_curve_fresh_training_20261008.md`。随机任务优先级全程关闭；保留total-wait/SOC urgency与任务物理service。核心5d75d7e，三Q接受记录49994cb。

一次冻结评价qualification_initial已EVALUATED/exit0，参数更新0，三个Q evaluation_consumed marker存在；所有冻结checkpoint未变。结果SHA a6d0457c18486473fdef5377760a1e870565e04bff3fc0d077603a958b7122ac，两端一致。975标签192R783W下：

| seed | WAIT recall | RELEASE recall | proposal top1/top3 | joint | macro F1 |
|---|---:|---:|---:|---:|---:|
|91901|.892720|.390625|.479167/.630208|.768205|.650823|
|91902|.872286|.505208|.515625/.666667|.765128|.686897|
|91903|.773946|.546875|.453125/.630208|.683077|.632088|

三者均未通过既定recall/F1/proposal/joint门槛；类别支持和非坍缩检查通过，不能概括为纯全WAIT坍缩。canonical91901漏放117/192，同时误放84/783；训练内proposal top1=1而独立仅.479，已经存在显著训练内外落差。只凭这张表不能把原因完全归给样本少、网络容量或观测缺失。Q已消费，不修改门槛、不挑91902、不用这975行训练或继续试模型；critic/PPO/formal保持未启动。

## 先做的两项有界诊断（执行前记录）

1. **只用原严格train三周期的整回合交叉验证曲线。** 每次两周期训练、一周期作开发validation，保持原网络/未加权gate+conditional proposal CE/Adam3e-4/batch128/clip.5；单一诊断初始化91911，观察0/100/500/1000/2000/5000updates，300s/fold。记录训练/validation loss、两类recall、proposal与联合准确率及validation未见proposal支持。只消耗原train，不触碰Q，不保存可部署checkpoint，不据此自动授资格。先2-step synthetic输出测试，再正式三fold。目的为判断原固定5000steps是否只是记忆高度相关的少量回合，而非盲目再加训练步数。
2. **只用train可信snapshot检查剩余资源观测混叠。** 当前105+N50包含全部空中邻机预测，但LZ只提供free比例/earliest ETA等汇总，没有逐pad空间可用状态。从原train的inner eligible状态中选有限个有free与occupied地面pad的状态，交换free/occupied pad身份及对应地面service机位置/assigned_lz，保持空中状态、服务剩余时长和free数量；比较数值观测是否相同、state-only teacher原标签是否复现及反事实可行首选是否变化。此为开发反事实，不混入原train、不重写标签、不消费Q；每次记录操作与查询预算。若未找到有效同观测异标签见证，不宣称已经证明该缺失解释本次全部失败。

根据结果选择最小修复与新的开发数据方案，再在候选真正冻结后登记新Q；不先扩大成无界采集矩阵。真实安全约束和原失败记录始终保留。最终正式训练仍需完整student、独立新评价、真实有界PPO更新及业务审查，不能拿安全闭环完成代替学习有效。

## 已完成的诊断

整回合CV三fold全部COMPLETE（newly_run，约31/28/30s），只读原1298 train，未访问Q/产生部署checkpoint。代码的2项测试通过；首个序列化断言因共享metric的int histogram key在JSON变string而失败，规范化测试比较后通过，生产loss/训练未变。结果`.tmp/p3_train_cv_20261009/result.json` SHA aea23705a8a39eba11385e410fc0cd3a826ffcb0a6cabe5ae102fe497a892342。

| 留出的train回合 | 训练/validation行 | 100step validation loss | 5000step train/validation loss | 5000step validation R recall/proposal top1 |
|---|---:|---:|---:|---:|
|893511 flow|1268/30|2.870|.029/12.449|.739130/.434783|
|893512 fixed|577/721|2.444|.001/11.214|.404255/.489362|
|893513 mixed|751/547|2.664|.001/10.432|.482759/.379310|

5000step三fold train proposal top1均1，validation仍低；fixed/mixed在早期100–500step也未满足R/proposal指标，简单早停并不能取得已需资格。两个dense fold分别缺2/5种validation proposal支持。结论为**回合间泛化不足与过拟合已在train-only诊断复现**，1298行不是1298个独立交通情境；不能继续把训练内拟合成功当成足够I0支持，也不能只加epoch。后续需要更多独立开发交通状态，并处理已证实的观测混叠。

逐LZ探针的第一项小样本（1状态2次teacher、3.031s）交换train893512/tick255/UID0的最近free pad1与occupied pad5，105+N50完全相同但teacher都为joint15，未证明混叠。primary查原teacher最后audit：pad1/3被预测扫掠拒绝，真正可行见证为pad4；于是第二项最小探针只改为交换实际见证pad4与occupied pad5，仍保持空中状态/各pad位置/服务时间/free数量不变，并同步地面服务机的assigned_lz和位置。

第二探针6.844s、2次teacher/0reset-step，原标签精确复现joint15（RELEASE、proposal14），反事实变joint0（WAIT）；**完整105D ego、全部50D邻机和mask逐位完全相同**。快照SHA 87cdf2821eae25381d7d08a26210c259694b5ef953be2d4113b7034b103cf548；新结果`.tmp/p3_train_lz_alias_selected_pad_20261009/result.json` SHA 03d3b1fd0b1ae55816c9c855b0381e1cf359af83753abd6e0ec4e24d6d4ffaa9。探针服务交换单测通过，第一次无见证结果保留。这证明当前观测无法区分至少一类实际重要的空间资源状态；不是teacher随意改标，也不是观测函数应读取teacher结果。它解释了一个真实缺陷，但不量化本次975行失误有多少来自该缺陷。

## 选定的修复方向

先增加显式V2.6资源观测：保留105D自身/调度及全部邻机50D，附加六个固定物理pad的相对位置、free/reserved/occupied、兼容性、ETA是否已知与有限ETA缩放，避免仅free比例/最早释放时间的多对一压缩。这些均来自决策前中心广播/资源状态，不查询teacher、候选安全结果或未来未承诺动作。旧V2.5 schema/checkpoint保持可读取，新schema需重新构建训练观测与fresh拟合，不把159D静默塞入旧105D模型。

先在上述同观测异标签见证及最小单元测试证明新特征确实区分状态，再仅从原train可信snapshot重建新观测、保持所有teacher标签。依据新开发交叉验证决定独立train采集与正则化预算，当前不消费新Q、不动PPO。更多数据用于补独立回合覆盖，而非按已消费Q挑样或追分。

V2.6特征及显式159D actor、CV输出测试共7项通过。真实上述train见证在0teacher/0step下新159D的indices144/146/149/153/155/158有差别，旧105prefix和全部traffic保持一致；结果`.tmp/p3_resource_context_witness.json` SHA df16265dd063611b14a0790ec02186fc075178c3410856bca87486b6eeb56174。模型默认仍105，159模型拒绝旧105权重/输入；尚未接入实际部署。

首次资源CV重建在893512/t0/UID4逐位比较停止，没有发生训练，失败日志保留。primary定位到Windows/Linux矩阵旋转在理论近零分量的数值舍入：4个traffic分量约1e-16，最大绝对差1.4210854e-16，ego完全相同。重建改为数值一致性检查rtol1e-6/atol1e-7，并始终直接保留归档105prefix/traffic原字节，仅追加新54资源特征，不修改原标签或数据。新输出目录重跑一次既定CV；不是放宽物理安全阈值。

资源CV已COMPLETE，三fold33.093/28.438/29.953s，result SHA 7ce2ea001d84100c61dbb5dbbee3984707078652b6a7f87105c0074b7662c1a55，路径`.tmp/p3_resource_cv_roundoff_20261009/result.json`。5000step留出flow/fixed/mixed的R recall分别.826087/.521277/.425287、proposal top1 .217391/.425532/.344828，validation loss11.844/10.372/10.684；仍显著过拟合。补资源消除了一个已证实的不可区分状态，但不能宣称已经改善总体泛化，更不能仅据训练拟合授资格。下一步需要独立回合状态，不能重复记忆三周期或消费旧Q调参。

显式159D runtime/critic/record通路已实现：checkpoint schema选择观测，旧105维保持默认；新critic使用actor对应ego宽度，空场景仍有team value/奖励记录。37项local定向tests通过（包括旧105、新159、K3/likelihood、冻结actor和PPO回滚构件）；只有合成更新，没有新实景PPO或准入。

## 下一项有界独立训练数据（首次reset前登记）

新增协议`p3_v26_spatial_resources_independent_train_20261009_v1`，不改当前物理合同、teacher排序或标签语义。第一cycle三个独立train seed893601/893602/893603，对应flow6 mixed、dense24 fixed.8、dense24 mixed.55–.90；同现有中性课程profile。开发smoke892871为3请求/warm3/2ticks，独立新目录。后续893604–6只保留，不在本cycle自动启动；新Q尚未登记/消费。运行前查重、同步源码SHA；依次完成和primary分析后才下一臂，最多一个环境父进程加一个顺序annotation child。

输出根服务器`/home/user/training_outputs/terminal/p3_v26_resources_20261009`，各臂`train_01`、`train_02`、`train_03`；入口`python -u -m terminal_marl.training.p3_v26_course_data --index 1/2/3 --output ...`，先`--index 1 --smoke --output .../smoke`。完整周期最多1800ticks（flow按既定1210tick课程上限）、环境3600s、全actor按tick/数值UID、最多2048query/51200 RELEASE意图、annotation21600s；多pad plan调用另计。超限/错误/安全/F/D/U/未完整标注均保留失败，不继续下一臂、不扩预算。流程先写不可混淆的protocol，然后原105重建与新159资源重建分别保存，逐行要求ID/mask/teacher label/105prefix/traffic不变。

每臂收尾亲自核对完整N/E/F/D/U、error/censor、checked-executed hash和实际安全、能量与E、raw/actor/selected/label/resource行对应、三接口/低SOC/繁忙LZ/retry与proposal支持。完整首cycle后再决定是否第二cycle；和旧893511–13同合同train共六个独立周期可用于整回合开发验证（旧Q始终隔离）。先判断数据支持/泛化后选定fresh拟合方案，不立即登记新Q、盲目增加epochs或运行PPO。实际formal需新冻结候选、独立资格、完整student、真实有界PPO与业务审查；用户持续授权有效，最终formal实际启动并验初产物后删除heartbeat，不监控formal。

## V2.6实施与独立输出验证

实现提交ea0ed80，local scoped提交、index空、无remote未推送。服务器38项源码/docs/tests SHA已逐项与本地一致，8file定向回归48 passed（23.88s），包含旧105与新159接口、数据入口与输出检查；与之前local测试重叠，不累加为独立总数。

独立892871 server smoke自然SMOKE_OK/exit0：2ticks/6raw/3actor/3完整标签，三接口各一个nominal RELEASE，实际执行3R，均K3第1项；prefix/traffic/ID/原label与新增159行完全对应、无安全/hash异常。原始开发null与forced行仍隔离不训练。环境2.134s/标注3.656s，root SHA8d4ee29788734c8abb21abd2672f350e4f45b13ace6ec4118b0caf31e5ecaf04，资源行SHA28823a951a2b39eaa330a2b042739bd9bdb58f0b6636c471377c31e5c46587b2，两端一致；primary读全量分析`.tmp/p3_v26_smoke_analysis.json`（SHA e019db4d92081a34e705f945a34faa095a40576bf91eef68d45f428acc94589a）后接受train01。短任务结束时真实Python已退出，不能声称观测到live argv/cwd；启动器明确cd且实际产物正常，后续长臂须核对live信息。

另修旧开发runner的protocol记录：先验标为未载入，模型验证载入后、reset前写入真实schema，避免159模型被记录成105。新增本地runtime9tests通过（26.38s，包含原7项）：105/159分别请求与2tick重放；159 actor-critic的12tick空场景→首次到达、存储/加载/完整reward tape形状保持159。只是接口小样本，不是student qualification/实际PPO。

train01/893601已自然COLLECTION_OK/exit0：632ticks、6N/0E/F/D/U、1719raw/18actor/18完整标签和资源行，original prefix/traffic/label完全保留。三接口各6R，proposal12×16/11×1/13×1；低SOC<.55标签3、busyLZ12、高速inner5，无retry恢复/WAIT标签。本低并发情境只补独立RELEASE支持，不能称gate两类充分。环境21.625s/标注21.494s，最小actual/swept110.127264/110.127201m，最低完成SOC.409027、剩余能量613.540；所有实际安全/hash/耗尽/技术/censor均无异常，0应急。初始wrapper/timeout/Python2940176/7/8和steps/protocol已观测，结束后原PID及child均不存在；wrapper cwd正确，Python已结束后未观测到其cwd，不追称已观测。

root SHA a407c3f568415c0f8ba24172c01842910f3048f8c723f45bec900f9f7f4dac5b、资源SHA365d0d54ff25b4757491dd2908a80fedf7023e1c7e91779c4b17f4c0c353b925两端一致。primary完整分析`.tmp/p3_v26_train01_analysis.json`（SHA fd0226e8b4b5c3462b8e1e3ad5d817369d819464b7c63f94eeddb63c27c8192f）后接受下一唯一train02/893602，所有冻结预算/物理/teacher/priority关闭合同不变。train01不重跑；新候选/Q/学习仍待整个cycle分析。

UTC20:32左右train02已启动，wrapper2940965/timeout2940966/Python2940967，三者真实cwd均`/home/user/projects/terminal`，argv为V26 collector/index2/正确新路径，源码SHA和无重复检查通过。初progress1tick/12raw/12selected（每25tick刷新），相邻steps已有18行，root result/exit尚无，属于正常启动前缀。当前唯一环境父进程；运行中核心冻结，dense预计1–2h，继续60分钟低频监控，不把首tick或单条annotation COMPLETED当全臂终态。

等待采集期间仅准备不被collector导入的新`p3_v26_train_dataset`：要求893601–3三完整train、严格profile/partition/sha、保留null/forced隔离与label、不同邻机数补mask；可与旧893511–13可信snapshot重建159的train合并，共六周期。仅新增loader的2项fixture测试已通过，尚未读取未完成cycle拟合。采集运行核心未改。

首cycle无阻塞完整后的下一项开发判断固定为**六个整回合leave-one-cycle-out**（各五回合train/一回合validation），沿用单一诊断初始化91911、原未加权CE、Adam3e-4/batch128/clip.5、marks0/100/500/1000/2000/5000、300s/fold，全部159+邻机50。仅train，无Q、teacher查询或可部署checkpoint。用完整曲线检查补充独立情境后是否仍记忆训练回合，不在每新增一条数据时反复拟合；须先完成整个cycle。该步骤尚未执行；后续fresh拟合预算与新Q仍需依据结果再冻结，不能由新的loader准备推断已资格。

## 独立dense fixed完整结果与下一臂

train02/893602自然COLLECTION_OK/exit0，原Python2940967及query child均退出。1167ticks、24请求18N6E/0F/D/U，12999raw（12428 forced WAIT）与571actor=571selected=571完整OK标签=571资源行；75R496W、13种proposal。原90D/null/mask、teacher与behavior/actual分离、105prefix/traffic及159追加字段全量检查通过，所有actor有效priority=1。三接口o2m/m2i/i2f分别24R328W、24R71W、27R97W；144条低SOC<.55、553条busyLZ、120条高速inner、444条retry结束后标签，same-UID/same-interface WAIT→RELEASE有27次。forced retry本身不入actor loss。

实际/扫掠最小间隔81.783820m，实际安全违规、checked/executed hash、技术error/censor、耗尽及未决均0；6次应急提交全部触地完成，最低终态SOC.153794、能量230.691535。16次应急尝试和36次安全阻塞计数保留，不称无应急代价。worker紧凑报告曾把emergency字典的10个字段误称“emergency count 10”；primary读取原24条终态及6 committed/6 touchdowns确认实际E=6，未采用错误汇总。

teacher R但实际trace W为21条、teacher W但实际trace R为6条；仍是原pre-state单机teacher与服务推进/顺序批准/兜底执行的语义差别，不按实际执行改原标签。55次正常K3批准为rank1/2/3=34/9/12；5条未分rank的RELEASE trace及第6次应急终态不混入proposal准确率。teacher查询12710个意图、12118次实际plan调用，短路资源检查可使plan少于意图，多pad路径反之，分别保留。环境1487.460s、标注9805.817s（约2.72h），未超冻结预算；整臂约3.14h，后续仍60min监控。

9项关键终态文件本地/服务器SHA逐项一致。root SHA1eb8cdb51243a46e8bb36f62af63a3a9c8027da709c756559ce2d4abd686391d、资源SHA5a7b827963a8edb603859013d7aae4e01a0ccfc5455d7b1eec600e69e4bfd009；primary读全量分析`.tmp/p3_v26_train02_analysis.json`（SHA291102dcb0b50c33a4dffe11dae5bea1f329829c36841e122dae512ea50805c5）。首次worker文件检查误用日期/目录层级，已按正确20261009/train_02/seed_893602重新核对；不是同步失败或重跑实验，失败检查原报告保留。

primary接受下一唯一train03/893603 dense24 mixed.55–.90，入口/所有上限不变、新目录`p3_v26_resources_20261009/train_03`，启动前继续Syncthing SHA和重复进程/路径检查。train02不重跑；第三臂完整后才执行上述六回合CV，没有新Q/fit/PPO/formal准入结论。

train02证据提交94e2b3f后，train03已实际启动：wrapper2998136/timeout2998137/Python2998138。UTC2026-10-09 00:29核对Python实际cwd/argv、protocol seed893603/24warm12/mixed.55–.90/task_priority_enabled=false，以及15steps/192raw/18selected；progress每25tick更新，此时仍1tick，不能覆盖更近的steps事实。38项Syncthing SHA及无重复进程/路径/seed检查通过，protocol SHA89aa1107e29278eb152702bcbbc98d9c42a132c35e3e99b38ba6e4ecf506af49。仅Python cwd已实际读取，不追称wrapper/timeout cwd也已观测。当前唯一父进程运行、尚无终态，核心冻结并继续60min静默监控，预计数小时。

train03/893603已自然COLLECTION_OK/exit0，原2998138及query child退出。978ticks、19N5E/0FDU、10357raw/371actor=selected=OKlabel=资源行，75R296W/11proposal；所有priority=1、原null/forced/label/prefix/traffic一致。三接口为24R210W、27R15W、24R71W，低SOC<.55/busyLZ/高速inner/retry结束后标签138/359/90/300；同UID同接口WAIT→RELEASE有33次。teacher R/实际W22条、teacher W/实际R5条，继续保留pre-state单机与顺序执行的语义区别。正常K3批准58次，rank1/2/3=38/11/9；5次应急提交全部完成，78次应急尝试及174次安全拒绝仍为业务代价，不改标掩盖。

实际/扫掠最小间隔80.304940m、最低终态SOC.170930/能量256.395630；实际安全/hash/耗尽/技术/censor/未决异常均0。环境875.914s、标注5034.829s，7685意图/10127多pad plan调用，均未超冻结预算。九关键文件两端SHA一致；root fd0e0e97d790f331585ab6046fec275df09e862b85e79469fcb435a662c65004，资源cdff1875d66c6b78b8260b1aa83b9963b352c7419f969afde04863814742d0cf。执行worker因额度限制未完成分析命令，primary直接接手完成原有逐行核对并读全量`.tmp/p3_v26_train03_analysis.json`（SHA be378e9dd30014f33e95b75bd72b60f631ea6b15bcb7f794f09534f312b44606）。

新三周期共54请求43N11E/0FDU，25075raw、960有效标签168R792W；同合同旧三周期合并2258有效train标签372R1886W。全部采集完成，primary接受原已登记六整回合CV，输出新`.tmp/p3_resource_six_cycle_cv_20261009`。不重跑采集、不消费新Q，按整周期留出避免把同回合相邻行当独立验证；仅诊断模型，无部署checkpoint或PPO。

## 六周期开发结果与候选结构修复试验

上述六fold已COMPLETE/exit0（每fold约28–31s），result SHA73c30407d08d0b795b4ec3ddbff05e342f2b11bf8c113750f960f2d4a58e0198。5000steps各fold训练proposal top1 .980–.997；四个dense留出回合893512/13/602/603的RELEASE recall .521/.644/.573/.507、proposal top1 .372/.368/.387/.467，validation loss5.810/7.447/5.644/5.494。500steps对应dense R recall仅.255/.276/.320/.400，proposal top1 .457/.379/.520/.573；早停仍不能满足要求。新flow18行全R留出表现较好（500step R .944、proposal .889），不能掩盖dense失败，也不能评价其WAIT召回。新增独立回合仍不足以让原两个独立输出头泛化，不继续盲目增加更新或立即新Q。

primary逐条检查六train的原teacher audit，2258行全部符合冻结first-feasible排序：372个已检查正候选、48285个已检查负候选、7793个未查询候选；结果`.tmp/p3_six_train_teacher_prefix.json` SHA9b1a097b4a155df284592e0dec69a2021dc7835f6fa781098a93337de0daf938，0新teacher/step/Q。这些检查前缀可由原first-feasible标签与固定顺序推导，**不是额外独立样本或已查出teacher误标**；未查询候选仍未知，不能当负样本。目前独立gate/proposal头没有显式利用这种共享候选结构。

选择一项有界模型表示试验：保留159+N50输入/原teacher标签/所有安全与执行逻辑，改为共享state encoder加同一候选打分网络（输入speed/timing两个数值），对25候选输出分数。按原固定teacher偏好顺序将分数分解为first-hit质量`h_k * product(1-h_j,j<k)`，WAIT为全拒绝质量；仍输出原WAIT/RELEASE与条件proposal概率，不把分数称安全证明，不在线调用teacher或候选规划器。此结构让gate与proposal共用可行意图表示，降低25分类各自记忆罕见proposal的自由度；是否有效须试验，不能预称解决。

新增`OrderedProposalActor`仅接开发CV，旧actor/runtime/checkpoint没有替换。保持同六整周期划分、91911初始化、原未加权gate CE+仅R的proposal CE、Adam3e-4/batch128/clip.5、0/100/500/1000/2000/5000steps、300s/fold；不增加class权重或搜索超参数，不改变原资格门槛。输出新`.tmp/p3_ordered_six_cycle_cv_20261009`，仅与上述同数据基线比较，0Q/环境step/teacher/部署checkpoint。先验12项测试通过，覆盖概率质量、first-hit单行joint NLL与已查前缀BCE的数学等价、未知后续候选无该joint NLL梯度、极端分数/空邻机/NaN padding、顺序与teacher一致，以及两step输出序列化。注意整体训练保留原分头平均CE，不把它误称全数据简单BCE求和。正式试验尚未执行；执行后据结果继续修复。

候选结构试验已COMPLETE/exit0，每fold32.8–34.6s，result SHAafb4ab6813698fcb397c66d7f261a3f2d8a43e3090592ef78a21f48fa6b9b6ce。dense四fold末步R recall .351/.598/.507/.467、proposal top1 .245/.287/.387/.293，训练内proposal也仅.522/.551/.512/.508。该特定共享标量分数/first-hit表示没有改善，训练拟合能力亦明显下降，**不接入runtime或后续正式训练**。保留实验代码/失败输出作为开发证据，不将它宣传为已采用的修复，也不继续给它加epoch。

下一最小表示诊断回到原两个分类头及原CE，只调整邻机编码的条件：旧`neighbor(traffic50)`先聚合后才拼ego159；新`neighbor(ego159,traffic50)`先逐邻机计算与本机的关系，然后仍用原mean/max/count聚合及96D分类头。当前target/interface/资源等ego上下文可在压缩前影响邻机表示，检验原固定压缩是否妨碍识别特定进近/切环的相关邻机；这是结构假设，不称已证明全部退化由编码器引起。输入、teacher、候选规划和安全层都不变，不加在线oracle。

该`ConditionedSetActor`也仅进入开发CV，15项定向tests通过（包括原12项），覆盖ego在pooling前的依赖、集合置换、NaN padding/空邻机/梯度与两step输出。下一唯一`.tmp/p3_conditioned_six_cycle_cv_20261009`仍为同六fold/91911/原loss/0–5000steps/300s每fold，无Q或部署checkpoint，比较时完整报告各fold而不挑最好seed。未看到结果前不替换生产actor；若仍无改善，依据数据支持与原始状态证据继续判断，不自动铺开调参矩阵。
