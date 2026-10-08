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
