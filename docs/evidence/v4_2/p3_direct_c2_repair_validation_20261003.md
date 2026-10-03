# P3直接连续离环修复与重跑（2026-10-03）

后续审查更正（2026-10-03）：下文A/B/C“无优化器”指90D推理wrapper及未发生optimizer更新；实际旧88D上游loader曾创建空Adam而未使用，不能据migration的`optimizer_created=false`推断全过程未创建。无参数更新/无训练的结论不变。该加载器已在[三状态契约预检](p3_direct_c2_contract_preflight_20261003.md)中改为纯模型加载并回归；下文保留当时记录以区分历史证据与后续更正。此前登记的三状态后续方案现已获得用户授权并开始执行，最新进度见该预检文档。

## 决策与适用边界

用户要求比较“直接进入降落轨迹、途中平滑调速”和“先向内收缩再减速”，由primary选择并实现后重跑。primary读取研究文档、分析、修改核心代码和本记录；worker只执行明确的测试、SSH、监控和Git命令。

选择 **直接C²连续进近至停机坪上方，再接原垂直下降段**。离环前减速不是本项目动力学的必要条件；必要的是位置、速度、加速度连续，整条轨迹满足物理性能与交通间隔。旧Bezier构造因曲率限制先降低入口速度，导致占用内环22–29s减速，后机维持16m/s而追近。这个先决条件属于原轨迹构造，不应被当成所有可执行降落轨迹的物理要求。

“不必先向内收缩”指不再强制飞完一个独立200→100m收缩段，仍需沿进近曲线从环飞向中心停机坪。此前prototype在60s内保持角速度0.08rad/s、半径200→100m，所以切向速度实际已由16→8m/s；“先收缩再减速”只是简写，不能理解为收缩过程中完全不降速。直接曲线同时决定空间运动和速度变化，也不保证任意时刻速度单调下降。

选择根据当前固定状态与研究接口，未声称这是所有流量/几何下的全局最优轨迹；原SafetyFilter及停机坪可用性继续决定是否接纳。若直接曲线仍存在冲突，不切换未记录的隐藏规划器，不放松间隔，也不增加电池来掩盖失败。

## 比较证据（newly_run，同一个已记录状态）

原快照：`p3_integrated_validation_20261003/A_blocker_replay/snapshot_600.pkl`，可信本项目pickle，7个未提交内环holder，速度16m/s；无环境推进、训练、权重更新。原A完整周期及根因报告见 [退出阻塞](p3_inner_exit_blocker_findings_20261003.md)。

| 构造 | 检查范围 | 全世界安全通过 | 时长与说明 |
|---|---|---:|---|
| 原环上制动+Bezier | 7机×6pad×3speed=126 | 0/126 | 全部单机动力学和能量通过，但真实邻机冲突 |
| 固定半径收缩prototype | uid6、pad1一个候选 | 1/1 | 105.823s，最小间隔136.966m；不能按1/1与其他全量比率直接排名 |
| 直接quintic至地面探索 | uid6×6pad | 3/6 | 约38.5–49.7s；未选用，最终方案保留已有垂直降落段 |
| 直接quintic至pad上方+垂直段 | 7机×6pad、目标速度参考10m/s=42 | 11/42 | 全部动力学通过，完整路线52.5166–70.4846s |

最后一行分机通过数：uid0=0、uid6=2、uid7=5、uid8=4、uid9/10/11=0（每机6个pad）。证明该状态有可用退出路线，尚不证明所有飞机能依次完成。原K3对照为126候选，新几何探索仅42候选，不把这两者当成同一候选分布上的成功率增益。

比较输出保存在服务器旧输出根的`direct_quintic_compare.json`和`direct_staging_compare.json`；运行脚本分别为本地`.tmp/p3_direct_quintic_compare_20261003.py`、`.tmp/p3_direct_staging_compare_20261003.py`，全部保留。本轮重跑用下述正式生产实现，不以临时prototype代替执行。

## 实现

1. `SimConfig.v3_landing_path_mode`新增显式`direct_c2`选择；历史默认仍为`brake_then_curve`。仅V3、continuous landing geometry允许直接模式。统一开发profile升为`p3_integrated_repair_direct_c2_v2_20261003`并启用新模式；旧88D/V2冻结source加载时显式还原旧模式，然后按既有规则迁移为90D，不重新训练或自动qualified。
2. 正常与应急降落共用新入口。以实际p/v/a作为五次多项式初边界，pad上方的staging位置、零速度/加速度作为末边界，再连接原垂直参数化下降段。内环保持当前高度；外/中环应急沿用原高空staging高度语义。整个轨迹保持C²连续，并经原解析多项式动力学与Safety检查后才能执行。
3. 有预约延迟时，先按当前名义环速飞行至实际离环时刻，不为入口曲率额外环上制动。正常speed proposal仍由`10+residual`映射到轨迹时长参考；入口真实16m/s不会瞬间重置为目标8/10/12m/s。时长以距离、当前速度与水平/垂直物理限制计算下界参考，再按目标速度缩放；最多20次、每次×1.1寻找物理可行时长。只检查单机动力学，不根据交通安全结果暗中搜索路线。
4. metadata记录mode、requested/actual entry duration和是否物理限幅，保留requested/accepted departure、真实entry speed、required braking=0等可审查字段。Safety、资源预约、能量和checked/executed hash逻辑不变。
5. 修复独立数值错误：`ramp_end`与总时长在1e-9s内相等时，在两种浮点偏差方向上统一为同一末端节点，内部ease/ramp节点也排除端点邻域。消除原约1e-13s段；不放宽动力学边界误差阈值。复现原6个transfer候选，并用正/负时间舍入误差回归。

## 有界验证顺序（运行前确定）

新输出根：`/home/user/training_outputs/terminal/p3_direct_c2_validation_20261003`。保留原A与所有失败，不覆盖输出、checkpoint。测试/评估前核对Syncthing相关文件SHA；只本地改代码，运行时核心冻结。最多两个本轮评估进程。

先运行direct landing、integrated repair、V3 generator/execution/safety/dynamics相关测试；再对规则/source各做2tick smoke，检查输出、error和迁移参数。通过后执行原A完整配对：

```text
python = /home/user/envs/terminal_env/bin/python
module = terminal_marl.scripts.diagnose_p3_integrated_repair
args = --run --requests 12 --warm-start 6 --high-soc
       --seed 890501 --policy-seed 890601 --max-steps 1800
       --trace-blockers --snapshot-ticks 400 600 850
controllers = rule / source
output = <新输出根>/A_rule / <新输出根>/A_source
checkpoint = /home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt
checkpoint SHA256 = 3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab
```

检查完整N/E/F/D/U、error/censor、耗尽和应急原因、实际安全计数、checked/executed一致性及能量消耗。A有技术/安全/F/U阻塞即不升负载、不训练；primary基于trace与可信快照选择最小诊断。A通过才继续已预定B（24请求/warm12、高SOC、seed890502/890602）及C（同B仅SOC.55–.90）。这一轮无训练对照首先验证环境可达性和冻结source；即使A/B/C改善，也不单独证明PPO训练退化已解决。任何学习前仍需student-only资格判断与有界更新方案，`PPO_ALLOWED=false`；无full training、held-out或硬件。

## 执行记录

实施后首先核对Syncthing四个改动源码/测试SHA一致，六个定向测试文件 **66 passed / 48.86s / exit0**，日志`targeted_tests.log`。随后按下列时间顺序记录smoke、完整周期及监控结果；最新状态以末尾阶段及CURRENT_STATUS为准。


两臂smoke完成：各2ticks、预定CENSORED、error=null、exit0；各2行blockers与snapshot_2存在，config确认direct_c2。rule/source checked samples=60/25、hash mismatch=0/0，碰撞/NMAC/硬间隔/joint fail-closed均0；原6个transfer数值失配未再出现。source迁移记录checkpoint_weights_unchanged=true、source_policy_weights_unchanged=true、optimizer_created=false、NOT_QUALIFIED，固定checkpoint SHA与上述一致。两步不能证明完整周期，primary据此仅放行预定A完整对照。


### A完整周期结果（newly_run）

核心提交`a7096fb66efc50401db3db267fa1f22ce55f6137`；无Git remote，未推送。两个实际Python PID为rule2445800/source2445799，argv与协议一致，初始progress与trace确认后创建heartbeat `p3`。本轮运行速度显著快于原失败A（新约2–3分钟墙钟），监控设为5分钟，并在当前任务继续后续阶段。

| 臂 | N/E/F/D/U | ticks / 墙钟s | min swept m | checked samples / hash mismatch | 最低终态SOC |
|---|---|---:|---:|---:|---:|
| A_rule | 12/0/0/0/0 | 660 / 128.290 | 81.4774 | 15506 / 0 | .662702 |
| A_source | 12/0/0/0/0 | 611 / 169.842 | 82.3455 | 13475 / 0 | .691173 |

两臂natural terminal、error=null、censored=false，generated=12，active/queue/service=0；碰撞、NMAC、硬间隔、operational buffer、joint fail-closed、trajectory_infeasible均0。应急attempts/committed与energy negative budgets均0，外部队列累计各20 UAV-s。规则/source全过程已检查plan matches各38；最大同时final=2/3，无overcapacity。正常release仍有36/49次拒绝，不能将需要等待的安全拒绝本身认定为bug。

原同场景5N7F / 4N8F的持续内环耗尽，在新A完整配对中消失；支持本次组合修复对已定位环境阻塞有效。这里同时修复直接进近和transfer近零段，完整周期改善不能被分摊成两项各自的独立因果效应；同状态几何探针与数值复现提供各自的机制证据。此结果仍不是PPO更新前后比较或普遍liveness证明。

primary据完整业务/安全/能量结果批准进入预定B：保持生产commit、checkpoint、max1800、trace和400/600/850快照，改为24请求/warm12、高SOC、env890502/policy890602，独立输出`B_rule`/`B_source`。新脚本已在两步smoke与A验证输出，不重复构造新验证矩阵；运行前仍核对实际进程及Syncthing，最多两臂。B结果返回后才决定C。


A终态核对：两exit=0、两个原PID均退出且无重复Python诊断进程；相邻log各108bytes，仅终态摘要。result SHA本地/服务器一致：rule=`9b5a390bfe971e84cb3d97f4071175dc493f88dd845a83e678e4f8d56054c289`，source=`f72e18c20c504d2baaeb76e59a833c57e75c8ab8853bdb81a9738db2d3802ac0`。400/600快照存在，850因提前自然终态未生成，符合输出条件。progress保留running标签，终态以result+exit+实际进程为准。

规则/source总能耗=3236.017/3056.491（模型能量单位），每请求均值=269.668/254.708；只作当前两臂描述，不把单个seed差异称为学习优势。


B已实际启动，Python PID rule2446805/source2446806，primary已核对完整argv与预定B一致。约运行3分52秒时，两臂progress分别150/175ticks、正常完成2/3、活跃18/18、队列0/1，尚未终态；不能以中间状态作通过结论。heartbeat `p3`已经切换到B及其后续条件C，每5分钟，正常推进静默；预计B比A墙钟更长，维持当前间隔。A证据文档提交`964e6a4`，无remote、未推送。运行期间核心保持`a7096fb`版本。


### B完整周期结果与C准入（newly_run，UTC 2026-10-03 09:50后核对）

| 臂 | N/E/F/D/U | ticks / 墙钟s | min swept m | checked samples / hash mismatch | 最低终态SOC |
|---|---|---:|---:|---:|---:|
| B_rule | 24/0/0/0/0 | 881 / 1287.926 | 81.7378 | 33668 / 0 | .447194 |
| B_source | 24/0/0/0/0 | 857 / 860.268 | 84.5822 | 31128 / 0 | .501018 |

两臂均natural terminal、error=null、censored=false、exit0，generated=24且active/queue/service=0，原PID均退出。碰撞/NMAC/硬间隔/operational buffer/joint fail-closed为0，checked plan matches各76。emergency attempts/committed、energy negative budget均0。规则/source总能耗=9158.089/9168.071，每请求均值381.587/382.003（模型单位）；外部队列累计124/97 UAV-s。source此单seed完成时间较短，但能耗没有同时更低，不据此宣称学习优势。

高密度候选筛选仍会拒绝动作：release rejected=422/231，inbound等提交路径safety_interventions=96/61，trajectory_infeasible_candidates=15/6；这些不是实际安全违规。primary检查完整trace：出现的trajectory_infeasible关联o2m/m2i；未见segment_mismatch/continuity错误，最终所有请求完成，未形成技术终止或持续业务阻塞。规则trace有21条该secondary标签，包含已批准decision保留的此前原因，不能当成15个新增候选失败以外的独立事件。原始RuntimeError详细文本没有被该既有聚合字段保留，不能声称逐个失败原因已完全归因；当前证据足以继续同负载能量测试，不据此放行训练或扩大新矩阵。

result SHA本地/服务器一致：rule=`ffce2cefe9bdbf9f1145c29563a7dcce4bb82ca94dcfb532bb67eefa207b5ff3`；source=`de4dd72e4b88c28063b88b974f591914eaf68411c2a899098e0cfb6c2881abc0`。log各108bytes、blockers行数881/857；progress仍保留running标签，终态以result/exit/进程共同确认。

primary据完整结果继续预定C，生产核心冻结a7096fb：24请求/warm12、env890502/policy890602、max1800ticks，去掉`--high-soc`（config明确initial_soc_min=.55、max=.90），其余与B一致；固定source 3fe7、trace-blockers和400/600/850快照，独立新目录`C_rule`/`C_source`，最多两个进程。沿用已经验证的CLI，不新增脚本、不重跑A/B。C需继续检查完整outcomes、应急提交与能量、实际安全和执行一致性；任何技术/安全/耗尽/未决阻塞先诊断，不训练。预计墙钟约15–25分钟，监控每5分钟；当前PPO_ALLOWED=false。


C启动核对：rule PID2449638、source PID2449639，完整argv与上述一致且无`--high-soc`；四个源码SHA本地/服务器一致、checkpoint仍3fe7。两臂首progress和trace已出现，约60秒内分别到75/50ticks，仍运行且result/exit尚未生成。heartbeat `p3`已切换到C，每5分钟；不重启A/B/C，不改运行中核心。

当前资格边界：A/B通过的是两个固定开发流量的环境业务闭环与冻结source运行检查。按`docs/reference/low_level_policy.md`第6–7节及`p3_integrated_repair_20261003.md`，V3/90D与新proposal物理映射不能继承旧V2/88D资格；本次没有fresh I0/I1/I2/I3，也没有完整requested-action likelihood/replay资格证据。source仍NOT_QUALIFIED；C终态前能量压力闭环待定，当前不建立PPO更新方案、不启动任何学习。C返回后先完成环境和student-only证据缺口判断，再记录与结果匹配的最小后续方案，不能用开发对照代替正式资格。


### C完整周期结果（newly_run，UTC 2026-10-03 10:51后终态核对）

| 臂 | N/E/F/D/U | ticks / 墙钟s | min swept m | checked samples / hash mismatch | 最低终态SOC |
|---|---|---:|---:|---:|---:|
| C_rule | 19/5/0/0/0 | 871 / 1182.369 | 81.0852 | 33734 / 0 | .244965 |
| C_source | 20/4/0/0/0 | 806 / 924.495 | 83.4973 | 32614 / 0 | .301994 |

两臂均COMPLETED、natural terminal、error=null、censored=false、exit0，generated=24且active/queue/service=0。规则/source碰撞、NMAC、硬间隔、operational buffer、shield/joint fail-closed、运行时安全接管均0；checked plan matches=75/76，final overcapacity=0。实际SOC范围配置确认.55–.90，direct_c2与nominal holder契约生效；source迁移记录权重不变、optimizer_created=false、NOT_QUALIFIED。无训练或checkpoint写入。

规则/source总能耗=8972.381/8813.830、每请求均值373.849/367.243（模型单位），外部队列累计124/50 UAV-s。最低终态SOC均高于research_evtol真实降落下限.10；.25是另一个历史queue/admission reserve，不能把规则最小.244965误判为降落能量违规。

应急链闭环：规则25次attempt、5次commit、5次touchdown；source47/4/4。按pad候选累计的safety拒绝52/92，energy/lz/dynamics/slack拒绝均0，energy negative budget均0。规则首次尝试及提交为395s；source首次尝试406s、首次提交471s。最后应急touchdown分别841/683s，随后完成地面服务并记入E类。每次attempt并非不同飞机；某架触发过应急尝试后仍可能及时正常提交，不能由47与4的比值计算独立应急成功率。实际9个已提交应急全部安全完成，没有再次陷入原“应急提交0、最终耗尽”的链条。

候选层trajectory_infeasible计数15/7；完整trace未见mismatch/continuity标签。安全候选拒绝与实际违规继续分别报告，不能隐去失败候选或把开发闭环通过解释成每个proposal都可行。source的正常完成率20/24、规则19/24；本场景仍依赖4/5次应急兜底，不能称低SOC全正常通过或策略业务最优。

两个原PID2449638/2449639已退出，无匹配的本轮诊断Python进程；log各131bytes，trace行数871/806。result SHA本地/服务器一致：rule=`476a2fbae41acf9527764acad2918baa13af3d6cd5f5ec9e7b0887493ed13f12`，source=`0bdd150e8b8af2ba53ef306b96c79380cf4d6dbcb4611e01d4b1a9d0e432f1f6`。实际墙钟约19.7/15.4分钟，在预估15–25分钟内；不再启动评估。

## 本轮结论与qualification判断

1. **已复现的环境阻塞得到修复，限于已测开发场景。** 原A规则/source为5N7F/4N8F，新A两臂均12N；B两臂均24N；C为19N5E/20N4E。共120请求111N9E、0F/D/U，所有臂完整终态且实际安全/hash/error/censor无异常。三种条件来自两个环境seed且B/C共享seed，不是120个独立泛化样本，也没有旧C直接对照。
2. **选直接C²进近有实测支持。** 同状态几何证据解释了为什么占用环上先制动会被后机追近，而完整周期证明新组合可持续疏通。近零段数值修复同时生效；不把所有收益仅归于某一个改动，也不声称任意流量下全局最优。
3. **P3训练退化尚未被验证解决。** 本轮没有优化器或更新前后配对，无法测量“训练之后是否又退化”。当前环境开发闭环通过；冻结source仅完成新MDP上的开发运行检查，V3/90D student-only imitation qualification仍未取得，PPO_ALLOWED=false。旧V2/88D的成功不能替代新schema/物理映射的资格。
4. **剩余工作有明确边界。** 首要缺口转为新轨迹语义下的动作/likelihood/replay契约、fresh数据与student覆盖，再是有界学习前后效果。C中的应急比例是后续减少等待/提升正常降落的目标，但本次证据不能判定这9次应急全部可避免。无需再扩大环境矩阵来重复证明A的旧耗尽现象已消失，也不继续盲调PPO超参数。

## 最小有界后续方案（计划，未执行，不构成学习准入）

下一轮先做无梯度接口预检，固定使用已有可信`A_source/B_source/C_source`各自的`snapshot_400.pkl`，3个状态、不新增人群或seed搜索。源文件分别481119/2401799/2816488bytes，固定3fe7 checkpoint与a7096fb生产映射。旧记录仅作开发诊断，不作为fresh训练或untouched qualification数据。

- 每个状态只构造一次固定source请求及K3 shortlist，覆盖该状态全部实际UID；显式检查90D观测、UID顺序、forced-WAIT mask、requested gate/proposal、rank-1 log-probability与proposal条件分布的绑定。检查零扩展后的88D前缀输出保持性质，但不把数值相等解释成跨MDP业务资格。
- 每个状态最多记录并重放1个决策tick：在两个独立副本上使用同一已记录动作/shortlist，比较下一步观测、reward、ledger、实际rank、checked/executed轨迹与安全；总上限3个tick。不是重建原第400tick的随机动作（原pickle没有保存原policy RNG tape），使用单独固定诊断RNG并如实标记。
- 不反传、不创建optimizer、不改权重、不评价held-out、输出新目录；先单个状态小样本验证字段/序列化，再完成其余两个固定状态。任何契约失配立即停止并用该状态定位；不改阈值、不换状态重试来凑通过。
- 预检通过后，依研究规范另立V3/90D fresh I0/I1/I2/I3协议，冻结数据/seed/上限和正式资格集后才采集/学习；新schema使用fresh actor，旧source保留为对照。数据需覆盖高速内环、busy LZ、低SOC、retry与拥堵恢复，不能把本轮快照直接并入训练或把metadata改名当资格。
- 新student资格完成后才评审一个有限PPO pilot及同场景更新前后完整周期配对；需事先固定可训练参数、proposal冻结状态、rollout/update预算、seed、停止条件和保留规则。当前没有足够依据填写或自动准入这个训练预算，后续不得沿用旧V2 admission直接启动。full training、held-out和硬件均不在本轮执行范围。

本轮已经按原顺序完成A/B/C、必要机制诊断及核心修复回归。以上是后续最小任务的可审查边界，不是已经执行的资格结果。所有失败、旧A结果和可信快照保留；正常/应急业务类别不合并。本轮终态清理不停止其他用户任务，只删除本线程的P3验证heartbeat。


终态清理已完成：automation工具确认heartbeat `p3`删除成功。删除返回的调度快照间隔为40分钟，早先创建/更新请求为5分钟；此处保留实际返回值，不将heartbeat触发时间用作服务器耗时证据。C墙钟时长仍以result中的1182.369/924.495秒为准。本轮无后台评估或新训练任务，后续预检仅登记为未执行计划。
