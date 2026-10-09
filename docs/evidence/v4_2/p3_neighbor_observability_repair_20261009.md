# P3 意图条件邻机关系与可获取观测修复

## 当前问题与授权

用户要求解决“哪个邻机阻挡哪次切环或进近”，允许调整架构，并补充硬约束：新观测必须来自本机传感器、实际通信或终端区管理广播，不能把无限全局信息当作单机可知状态。primary本人设计、诊断和修改核心。旧Q已消费，不进入本轮开发。随机任务priority继续关闭。

此前六回合的原集合模型、first-hit候选模型、ego条件集合模型均未充分泛化；训练拟合接近满分不能支持P3正式训练。补充数据入口8901bfe的服务器preflight为55 passed/1 warning/exit0，但smoke892872与train893604–6尚未reset。根据最新用户方向，先处理邻机关系；补充采集暂缓。

## 第一项真实训练见证（newly_run）

原train893602/tick0/UID6，snapshot SHA `e09f793a92232f36da0492c514a0e9a78f4a77eda24006e34318a048964248b8`，默认joint13切入inner。原teacher WAIT，原单候选拒绝reason精确复现`predicted_holder_reachable_tube`。保持生成出的canonical plan不变，对每一邻机分别运行原SafetyFilter，唯一拒绝对为UID11；其余10对通过。移除UID11的诊断世界中同一plan通过；此反事实不作为新训练场景或改写原标签。

候选时长62.453782s，UID11初距600.749532m，0.05s诊断采样最近距离49.209640m、时刻52.7s，明显小于原80m安全要求。最近距离来自原预测模型的密采样，不冒充实飞测量或连续时间独立证明；原native证明链给出拒绝。旧90D五槽为7/9/10/10/0，没有UID11；V25/V26全邻机集合含UID11。因此旧槽有具体漏选见证，新集合则仍需识别“本意图相关”的邻机，不能仅凭集合包含它就认为网络会正确使用。

已有七个forecast knots若线性插值，在52.7s对UID11的位置误差92.916163m。模型并未显式使用该插值，所以不能把该数值声称为神经网络实际预测误差；它说明稀疏坐标不能未经验证视作精确曲线。已知圆环运动可由当前状态/合同推算，新的表示应利用时间结构而非把21个坐标匿名压缩。

结果`.tmp/p3_v26_blocker_first.json`，0reset/step/Q，1次原候选生成及有限的同plan pair检查。物理/资源state比较通过；临时脚本末尾最初有一个无效自比较断言，已修正为与候选查询后的state比较。原查询前后的独立`state_delta`检查始终实际执行，因此不依赖该末尾断言确认原候选未改物理状态。

## 有界离线邻机监督（执行前登记）

只使用当前六完整train周期893511–13/893601–3的可信快照。在每周期的interface×原teacher gate组内按tick/数值UID排序，均匀取最多4状态，总数最多144。每状态检查默认joint13；若原R且首可行不是13，再检查原首可行；若原W，再检查固定joint1。都在原teacher已查前缀内，未知后缀不当负例。最终进近枚举当前广播资源中可用pad，为每一意图/pad保持生成plan不变逐邻机复核。最多1728 plan calls、3600s，超限保留INCOMPLETE，0环境reset/step。

输出保留原teacher/意图/pad/邻机UID/预测检查阶段/时间/数值及资源、动力学等非邻机原因。隔离pair未通过动态检查时，邻机阻塞标签为null，不能当安全或阻塞；resource-only失败不编造阻塞邻机。它是同一冻结候选的反事实监督，不代表删除邻机会保持调度器重新生成的轨迹，也不代表联合动作全局最优。先三个interface的小样本验证输出，完整数据输出新目录。后续整回合留出必须同时隔离该回合的原BC行和所有辅助监督，不允许辅助标签泄漏到validation。

## 在线可获取信息边界

离线阻塞UID、候选可行性、预测最近间隔及安全证书均只作为监督/诊断目标，不放进在线observation。策略推断不得调用teacher、全候选安全扫描或读取其他策略的未来请求。

| 信息 | 允许的实际来源 | 边界 |
|---|---|---|
| 本机位置、速度、SOC、自身状态/请求 | 本机估计器、电池监测、自身控制/调度接口 | 不使用仿真真值未来状态；估计误差须在部署验证中处理 |
| 邻机当前跟踪状态与已公布轨迹承诺 | 邻机协同通信，或终端区监视服务转发 | 只能使用收到的报告；必须限定服务范围、更新时刻、有效期和失联处理 |
| 邻机预测位置 | 从上述报告、已公布承诺及公共nominal-hold合同推算 | 是模型预测而非真实未来；未知未来放行不能假定已知 |
| 环形空域几何/运行参数、接口调度、逐pad可用状态和服务ETA | 终端区管理方广播 | 只含本终端必要运行资源；ETA是已公布估计，不读取未来随机到达/任务 |
| 哪个邻机阻挡候选、候选是否通过安全检查 | 仅离线teacher/native诊断监督 | 明确禁止作为在线输入 |

当前V26直接从env构建全active记录是仿真实现，不等于已验证通信可达。采用新架构前，须把所用字段落实到有限广播接口/声明的信息合同，限定覆盖、更新频率、消息大小/数量以及过期/缺失处理；不能仅写“都可以广播”即宣称真实可用。范围内已承诺轨迹允许传输或压缩参数，但额外未来动作、真实未来位置、隐藏状态和无限场外交通不允许。尚未选定新网络或新增在线字段，先核对真实监督结果再确定最小表示。

## 已完成离线复核与待验证结构

三个interface小样本SMOKE_OK：3状态、8个意图/pad plan、21个pair检查，2.406s，输出SHA `6d75987cb4f6599cce4abb39a22946b7bb6e6f6ff45939a2996d369f38c03884`。后续完整既定选择为127状态（各seed19/24/24/12/24/24），317 plan calls，408.313s，无error或预算截断，原teacher每项选定意图的可行/拒绝及final首可行pad全部复现。结果`.tmp/p3_neighbor_witness_train_20261009/witnesses.jsonl` SHA `b5e6a9c37efc327782b2c2f853c2fd504064fe19150a2d4c677f8e97cffadcc4`。

共2899个同plan隔离pair：260 blocked/2639 safe/0unknown；172个意图/pad包含至少一个阻塞机，129个意图/pad原生通过。另10次final_capacity与2次lz_unavailable在计划前被拒绝，没有编造邻机原因。pair拒绝123 predicted_separation、75 predicted_swept_separation、55 holder_tube、5 handoff margin、2 post-plan separation。这里的次数是查询/轨迹对，不是新增独立回合或实际碰撞次数。原127条状态的标签、现场行为和执行结果均未修改。

新增开发`IntentNeighborActor`保留159+N50输入，不增任何在线观测；逐“25意图×目的地（切环一个虚拟目标，final六pad）×邻机”预测拒绝分数。全局当前上下文只帮助逐pair编码，邻机身份直到输出标量后才按最大风险汇总；目的地按最低风险汇总，再按既定first-feasible顺序得到gate/proposal分布。独立scene分支表达资源/动力学等非邻机拒绝，防止把每个WAIT都归给某邻机。输出blocker index是可验证的模型预测，不是注意力因果解释或安全证书；真实SafetyFilter保持最终决定。

actor输入最多32邻机，超限直接拒绝接口，不静默按最近距离截断；这个软件边界本身不证明无线链路可达。尚未接runtime、未宣称真实通信实现或部署资格。现有range/staleness/report来源仍须在正式接入前落实，不能因此次不增加维度而绕过用户约束。

9项local测试通过（4.33s，含原6项），覆盖邻机置换等变/最终动作不变、每意图保留邻机身份、辅助查询与实际policy用同一分数、NaN padding/空邻机/梯度、无可用pad/数量超限、原前缀与null、全回合辅助数据隔离，以及两step四fold-arm真实输出序列化。不是学习效果结论。

下一有界比较固定为同六整回合留一、91911初始化、原未加权gate+R proposal CE、Adam3e-4/batch128/clip.5、0/100/500/1000/2000/5000 steps。两臂为同一关系网络不加辅助与增加native pair辅助；辅助batch64，blocked/safe分别BCE取均值后再平均，未知无loss，pair权重1，已知scene BCE权重.25。辅助监督与原BC都严格随parent seed划分，留出周期监督不入训练。每fold-arm最多1200s，超时保留失败不延时；单线程synthetic128×20一次前后向约.142s，12 fold-arm预计约2–3h，按60min监控。所有输入来自train，0新teacher/环境step/Q，无checkpoint保存；先核对完整loader与服务器同步/测试，再启动新输出。判断同时报告原BC泛化和held-out native pair召回/首位阻塞识别，不能只挑辅助指标或训练拟合宣称P3就绪。

六周期2258原行与317辅助queries的完整loader已核对，均max20邻机；原label/159数值/traffic对应不变，2899个已知pair中260blocked，145个已知scene目标中16blocked。其余scene未知不施加loss，不能因先遇到邻机冲突就假定其所有资源/动力学检查一定能过。分析`.tmp/p3_neighbor_witness_analysis.json`。

核心提交3291be0，本地index空、无remote未推送。server源码/三项witness文件SHA一致，16项定向回归1.42s通过（含原9项）。已启动唯一比较：Python3045749/timeout3045748/wrapper3045747，输出`/home/user/training_outputs/terminal/p3_neighbor_relation_cv_20261009/six_cycle`；真实Python argv/cwd已确认。初次快照仅加载输入、尚无protocol/result；第二次已见RUNNING protocol/result和held893511 without_auxiliary100updates/18.429s，属于前缀，不报告任何效果结论。首次进程过滤将自身只读探针也列入，已按真实argv token修正；实际只有上述一个CV。后续60min监控、冻结其导入源码，不重复启动。

## 有限广播接收接口（准备，尚未部署）

在CV运行期间独立新增`terminal_policy_packet.py`，不改CV导入代码。接收器只接受typed本机sample与终端publication，不接env；159数值拆成29个本机/本机控制接口字段和130个中心/由广播推导字段，50D邻机保持原schema，未追加真实未来或teacher信息。local release_eligible包含同一publication的interface freeze状态；需要本地控制器与该广播参考时刻一致，不能把旧相对坐标拼到新的本机位置。

协议限定一个订阅终端当前登记服务范围、最多32邻机、1Hz发布意图、按接收时刻最多2s信息年龄、现有7个预测点/150s模型预测窗。任何缺报、重复/外域报告、中心声明监视不完整、时间不一致、过期、非有限数或超限，均返回`ObservationUnavailable`；接入runtime时必须留在hold/fail-safe而非静默删掉可能的阻塞邻机。UID只用于报文完整性/结果解释，不进神经网络。32邻机加159数值、时间戳/UID及固定header的数值载荷估算7504bytes，小于8192bytes声明上限；这是软件报文预算，不是实测无线吞吐。

10项接收接口测试通过，覆盖逐值重建、拒绝env样对象、缺失/错误scope/接收者/时间/NaN/超容量，以及邻机年龄相对实际接收时间计算，不能把“report→publication”和“publication→receiver”两段各放2s变成4s。接口还没有替换实际policy输入，publisher实际传感误差、链路时延/失联及安全处置闭环尚未验证；不凭这组单测声称已经满足真实通信或部署资格。CV结果若支持采用新模型，下一接入必须经此有限来源边界并验证对应ideal广播假设，后续硬件仍需实测。

CV完成记录及待执行的 max-candidate gate 校准候选见 [P3邻机关系 gate 校准候选](p3_neighbor_gate_calibration_20261009.md)。
