# V2.5 fresh imitation资格协议（2026-10-08）

## 状态与适用范围

**最新执行暂停原因：C_det出现10N14E。** 这不触发原开发F/安全硬失败，但正常完成率不足，需要先完成C_sto及同seed规则控制实验，区分环境限制和策略调度；详见[持续验证最新记录](p3_v25_formal_readiness_20261008.md)。本协议准备与测试保留，真实smoke/Q/fresh fit仍未启动，不能仅因六开发臂无F就自动进入本阶段。primary完成该业务疑点的必要分析/修复后才恢复。

这是用户“持续修复至新P3正式训练启动”授权下的下一项有界协议。primary亲自制定/实现/分析。**此协议尚未启动正式Q采集或fresh fit**；先完成当前V2.5开发A/B/C六臂、定向测试及独立smoke。任一开发臂技术/实际安全/hash/F/D/U/censor阻塞时暂停本协议，先按证据修复。当前只准备新增文件，不修改运行中的观测/物理/runtime核心。

这次资格是静态三环、12/24有限到达、当前direct C²与K3执行范围内的初始模仿判断；不是动态空域、全交通负载、PPO或硬件资格。独立Q的状态由固定规则占用分布产生，因此即使I1通过，仍需要Safe DAgger/student分布与完整student-only周期检查。teacher标签是pre-state单机首可行/最小干预意图，不是当tick联合批准或全局最优；原请求/批量执行拒绝分开保留。

## 初始数据与I0边界

只使用已有三个完整train回合893001、893101、893201及其V2.5原状态重建，1046 actor行=229 RELEASE+817 WAIT。它们完整结束、全部actor有唯一OK标签、原teacher和raw未改，重建105D ego+全邻机50D，不含UID或候选查询结果。o2m/m2i/i2f的R/W分别61/120、102/137、66/560，低SOC188行含50R；另报告busy LZ、高速inner、retry恢复。训练入口重新逐行核对完整性、partition、原标签与新输入join，固定允许seed。开发pilot/smoke、旧失败采集、audit以及所有Q不进入训练。

这是**三个有效完整回合的初始训练资料**；旧24 train+6 audit采集计划没有完成，不追认原大协议PASS，也不把1046行称充分泛化。I0数据完整性成立与覆盖范围已描述；本协议直接用新的独立完整周期Q检验这个有限初始集是否足够进入后续模仿阶段。若不够，保留失败Q作诊断，不并入训练；另用fresh train/合法Safe DAgger补足，再冻结新的资格cohort。I0/student/PPO资格不会仅凭训练集100%拟合自动置true。

## 固定采集与预算

新输出根`/home/user/training_outputs/terminal/p3_v25_i1_20261008`，独立smoke根`p3_v25_i1_smoke_20261008`。Q顺序：

| case | seed | 请求/warm | SOC |
|---|---:|---:|---|
|Q_A|895301|12/6|高SOC（既定profile）|
|Q_B|895302|24/12|高SOC|
|Q_C|895303|24/12|.55–.90|

每臂完整周期1800ticks/环境3600s，离线标注7200s，全部actor按tick/数值UID，最多2048 queries/51200 RELEASE意图；final多LZ plan调用另计。不得超时续标、补样、丢困难状态或截断到满足指标。raw teacher_label维持null；只OK且actor=true可入监督，forced WAIT无actor loss。技术/实际安全/hash/F/D/U/censor/overflow/标注不全均阻塞下一臂。E保留单独报告并要求完整落地，不将应急称正常完成。每臂由primary查看完整结果后才开始下一臂，最多一个环境父及一个顺序query child。

复用已验证的all-actor collector、原state-only teacher和90D label契约，再从可信pickle重建V2.5。新Q保留partition=Q，专用evaluation decoder；禁止把Q标成train绕过原loader。保存context JSONL/hash、完整raw/steps/labels/selected/snapshots/coverage，核对重建不改env、原90D/label不变。采集不读取checkpoint、不调用student，不按label选择样本。

独立smoke seed892805、3请求/warm3高SOC、2ticks、环境/标注各180s，只检验采集→label→Q型重建输出；partition为development_pilot_never_train。另16行合成数据两次更新检查fit/checkpoint，与正式train/Q无交集。smoke不能授予资格。代码/docs初始精确seed搜索91801–3/895301–3/892805未命中，两个服务器输出根未存在；真正Q首次reset前还要检查服务器历史输出身份及现有进程，发生碰撞即停止，不现场换seed。

## Fresh训练与一次评估

三臂Q全部完整后先冻结cohort元数据、行数与字节SHA；训练过程只检查这些元数据，不解析Q行。随后只对上述1046 train从头初始化91801/91802/91803，canonical预定91801（编号顺序，不选最好seed）。沿用当前集合actor59451参数、独立gate/proposal encoder、未加权gate CE+teacher RELEASE条件proposal CE、Adam3e-4、batch128、clip.5、每seed固定5000updates/300s、final-only。参数/预算已由train内曲线制定，不根据Q结果调阈值、重采样、换loss或选epoch。

所有三个final checkpoint写完并hash后才读Q数值行。训练attempt及evaluation attempt分别在Q目录用exclusive marker记录；训练/评估中断也保留已消费状态，不能改输出目录重试同一cohort。先完成、后一次评价；checkpoint仍标为未qualified候选，I1通过不会直接把它变成正式PPO源。

沿用既有I1 per-seed指标，不为新结果降低门槛：WAIT recall≥.90、RELEASE recall≥.70、gate macro-F1≥.80、条件proposal top1≥.85/top3≥.95、decoded joint accuracy≥.85；joint稳定类11–14在Q中出现且模型均预测到；每个预测gate占比≥.05、主导预测proposal占比<.95、所有梯度有限。三seed全部通过才认为此次I1 imitation要求满足。Q若不支持稳定类则报告cohort覆盖不足，不能将其记成模型通过。条件proposal始终在teacher RELEASE上计算，不用模型gate筛掉难样本；记录全组及各episode/interface/低SOC/busyLZ/高速inner/retry恢复指标和三seed均值/标准差/范围。

这些离线准确率度量对原teacher的模仿，不能替代真实安全或联合执行可行性。I1通过仅进入必要Safe DAgger/student-only完整闭环；I1失败先定位gate、proposal、上下文或访问分布的具体问题，再有界修复。最后仍需critic/PPO有界防退化检查与正式配置冻结。用户的持续推进授权有效，不在每个已授权步骤反复请求许可；正式P3启动前不会仅据此协议或smoke宣称就绪。

## 实现与结果

新增`training/p3_v25_imitation_qualification.py`与对应tests。首轮local为28通过/1失败：Windows相对checkpoint清单写成反斜杠而校验为POSIX路径，尚未触及实际数据或服务器正式运行。primary改为`.as_posix()`，并将测试对比明确为JSON序列化形式（指标字典整数键正常变字符串），未放宽模型/标签判据。修后local/server 29项全通过（25.84s/13.41s）；随后新增完整合成Q评估测试，仅重跑新文件15项通过（19.20s/9.57s），本轮合计30个不同用例。检查独立partition解码、完整actor重建、失败保留、固定fresh checkpoint、Q消费纪律、全三seed预测/聚合指标输出，以及原loader/runtime回归。两端module SHA `130831a0dd4c06980f822a28bd53a1382bff3803b4e29cf7addcd713a1b50a76`，test SHA `b4a721f126bcea1d5a26ba2a9c611988fb1ce23924b831979d37d2b4521300aa`。

真实独立smoke/正式Q/fresh fit均尚未启动，当前B_sto开发评估继续，正式资格仍false。
