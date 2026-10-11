# 研究重构计划：可执行协同到达优先

> **版本与状态**：2026-10-11；第一批有界 rollout reference `48/48`、full pilot `9/9` 均已完成，
> M1–M3 已完成，M4 及以后未完成。2-tick smoke 的 `9/9` replay 为 `MATCH`，但不是 full-cycle；
> 4 个 selected reference 均通过 whole-cycle replay `MATCH`。本批所有 seeds 属 development，
> Q untouched、`PPO_ALLOWED=false`；结果见 [第一批证据](evidence/arrival_restructure/first_batch_20261011.md)。
>
> **当前研究入口**：[docs/README.md](README.md)。当前阶段和授权边界仍由
> [CURRENT_STATUS.md](CURRENT_STATUS.md) 决定；业务语义、模块职责和技术接口分别服从
> [requirements.md](requirements.md)、[design_spec.md](design_spec.md) 和
> [technical_spec.md](technical_spec.md)。本文档取代这些文档中“先取得低层 RL 成功、再开始
> 其他研究”的研究排序叙述；旧实验、旧 qualification 和历史 PPO 字段仍按其原文档解释，
> 不被本次计划追溯改写。

## 1. 当前研究决定

研究对象收束为：在随机、突发和资源受限的到达流中，利用原生轨迹生成、安全过滤、能量
可达性、LZ 服务和调度预约，研究多 UAV 的**可执行协同到达与降落管理**。首个可比较的
参考应是一个能够在完整原生仿真中闭合全周期的传统联合调度参考。IL/RL 只有在测量结果
显示它为放行排序、能量余量、等待或资源利用带来增量时才进入后续阶段；没有低层 RL 的
成功结果，也不阻止可靠传统低层支撑后续几何研究。

当前研究仍属于 Phase A landing-only。departure、arrival/departure integrated、网络级
航路、任意自由飞行 RL 和完整旋翼气动模型不进入本批次。

### 1.1 不变的语义与安全边界

- 主课程继续使用 task_priority_enabled=false。mission_type 只表示物理服务/兼容性
  metadata；随机 base-priority 抽样继续消耗兼容 RNG，但有效值统一为 1，policy mission
  one-hot 为 STANDARD。
- 随机到达、SOC 差异、phase-dependent energy、energy slack、30/120/10/60 s 服务时长、
  external queue、LZ reservation、ring/path feasibility、headway、retry 和完整
  N/E/F/D/U ledger 均保留。能量应急 takeover 与 mission-level emergency 语义分开，
  不因本计划取消。
- 低层仍使用现有 V4.2 WAIT/RELEASE 与 conditional proposal、冻结的 K=3 bounded
  shortlist、既有 pad/LZ 分配、canonical trajectory、SafetyFilter 和 PredictiveSafetyShield。
  不新增自由航向、航迹角、姿态或电机 action。
- 80 m desired/planning separation、40 m protected/NMAC floor、<10 m collision
  proxy、trajectory certification、fail-closed、committed plan 和 scheduler slot 语义
  不放宽。旧 qualification 不跨物理 profile、action/schema 或协议自动继承。
- Q/heldout 保持 untouched；第一批所有 seeds 都是 development，PPO_ALLOWED=false。
  第一批不消费 Q，不把 pilot 结果写成 qualification。

## 2. 第一批控制边界：两种 release-order authority

第一批保持同一仿真、同一 action、同一 K=3 shortlist、同一 pad/LZ 资源、同一安全层和同一
应急规则，只比较 release-order authority，并把 policy 请求和实际执行分开记录。

### 2.1 legacy arbitration

这是默认的既有调度仲裁路径。其 release order、headway、reservation、轨迹认证、shield
和 energy emergency 规则保持原语义，作为 rule_nominal_k3 及 legacy 对照记录。它不因新
参考搜索而被静默替换。

### 2.2 external_dispatch_v1（opt-in）

这是第一批新增的可选 release-order authority。它只在同一控制时刻从已有候选请求和已由
原生 scheduler 计算出的可行边界中选择有界 release 顺序；它不得产生自由飞行动作、绕过
reservation/headway/SafetyFilter、取消已 committed 的 flight、扩展 K=3 为全 catalogue，
或改变 emergency takeover 规则。legacy 与 external_dispatch_v1 均使用相同的 action、
K=3、pads/LZ、trajectory、safety、energy 和 emergency 语义。

每个控制 tick 至少记录：requested order、controller-selected order、scheduler-approved
order、actual applied order、候选集合、未执行原因、reservation/slot、SafetyFilter 与
shield 结果。若 actual order 与 requested/selected order 不同，保留两者及其原生拒绝或
等待原因，不把执行层结果改写成 policy 请求。

## 3. 传统可执行联合参考

### 3.1 有界 candidate dispatch policies

参考搜索只覆盖小规模闭合场景 N ∈ {3,4,5,6}。每个 N 最多登记 24 个 priority
permutations；serial 与 all-release variants 的硬上限为每个 N 共 48。Protocol V2 的首批实际登记
每个 N 的 6 个 priority permutations：在完整 n! lexicographic 排列索引中等距选取 6 个
（N=3 覆盖全部 6 个），若初始 energy-slack order 不在其中，则替换最后一个。每个排列
各运行 all-release 与 serial-release，因此每个 N 实际 12 个候选，N=3..6 合计 48 个；
24 permutations / 48 candidates per N 仍是硬上限，不代表本批实际登记数。候选可以包含优先级
排列和 wait/serial 变体，但每个候选都必须通过完整 native simulator、原生轨迹生成、
SafetyFilter、LZ service 和 energy ledger。

Protocol V2 在首个 full pilot / reference rollout 前登记：revision 原因为先形成有界机制与
reference 证据，并控制首批成本；仅使用 dense 2-tick smoke 的运行时间作为性能输入，
未使用 full-rollout outcome 数据选择候选。此范围不构成 exhaustive optimum。

该 small-N reference 使用当前 physics 与 task-neutral service 语义；N=3,4,5,6 均在
t=0 warm-start，初始 SOC 固定为 0.8，不提供尚未到达的未来未知 demand。三场景 full
pilot 保留原有 mixedSOC demand/course；两者的 SOC 与 warm-start 条件单独标注。

在线 lookahead 只使用当前状态、已到达请求和已登记的可见信息，不读取尚未到达的未来
request。每个候选都使用 closed cohort；每架 UAV 必须到达终态或明确记录为 diverted、
failed 或 unresolved。候选选择后保存 whole-cycle tape，并用独立的同输入 replay 复核。

### 3.2 证据边界

完整 resolved rollout 与 joint completion witness 分开标注。仍在有效、完整 resolved
rollout 中按 F/D/U、E、wait、makespan、energy 的顺序选取候选；只有 N+E 等于完整请求
分母，且 F=D=U=not_yet_requested=0，并经 independent replay MATCH whole-cycle 后，
才标注 joint_completion_witness；进一步 E=0 才标注 normal_completion_witness。
有 failure/diversion 的完整 rollout 保留为 replayed resolved rollout，不使用 witness 名称。

本批次记录 bounded rollout 及满足上述条件的 joint completion witness：完整请求分母、normal/emergency/failed/
diverted/unresolved、full-cycle wait、all-request wait、native safety、landing reserve、
committed/reservation 记录和实际执行轨迹。候选的“最好”只表示登记集合中的比较结果。

不把未找到 schedule 写成 global infeasibility、physical infeasibility 或 continuous
optimality；不把有限 candidate search 写成全局最优或完整 recursive feasibility 证明。
本批次只是 bounded rollout reference 与 local reorder，不是已经完成的 joint recursive
feasibility architecture。
预算耗尽、时间上限或未完成 replay 一律记为 unknown/censored，禁止把它重新标注为
failure 或 PASS。

## 4. 原生 energy-slack 与 bounded rolling local reorder

第一批同时实现一条 normal-route energy-slack heuristic 和 bounded rolling local reorder，
只在既有需求定义不变的三个场景运行：

~~~text
flow6_mixed
dense24_fixed
dense24_mixed
~~~

现有 rule_nominal_k3 作为基线。energy-slack heuristic 只改变 normal-route 候选的有界
排序，energy emergency takeover、轨迹认证、资源预约和安全拒绝保持原生语义。

energy_slack 与 rolling_reorder 的对照保持同一请求集合和同一 nominal K3 requests；
physics、task-neutral service、trajectory/SafetyFilter、emergency/safety authority 均不变，
只改变 release ordering 或 replan ordering。刻意的 WAIT/serial variants 仅属于第 3 节
small-N reference search，单独命名和报告，不将其变化归因于 order-only benefit。

rolling local reorder 的初始冻结边界为：horizon 4 ticks、每次最多 4 个 candidate
orders、每 10 ticks replan。搜索必须尊重 committed plans、已有 scheduler slots、
headway 和 reservation expiry；不得取消、回写或重排已经 committed 的 flight。所有 knobs、
candidate count、replan decision、native planning cost 和实际 applied order 都写入输出。

实现和 controller 可以在首次完整 development pilot 前做有界 refinement；每次 refinement 必须记录
版本、原因、输入协议、旧/新 knobs 和是否重新执行 smoke。本批实现冻结于 `59e3f24652c7480087b8110a832105bbd010bea1`；后续修订另行登记，不改写既有结果。

## 5. 第一批冻结的协议、seed 和预算

第一批只使用 development seeds，永远不作为 Q：

| 用途 | 场景/规模 | seed |
|---|---|---|
| 2-tick smoke | 三个场景/控制器 | 896091, 896092, 896093 |
| full pilot | flow6_mixed | 896101 |
| full pilot | dense24_fixed | 896102 |
| full pilot | dense24_mixed | 896103 |
| candidate search | N=3,4,5,6 | 896113, 896114, 896115, 896116 |

每个 scene/controller 的 full bound 为 native config time limit、2400 ticks 或 7200 s wall
time 三者中先到者。环境自然终止时照常记录真实终态；若任一上限先到且 cycle 未完整
闭合，状态写为 CENSORED/unknown。runner 不得静默把自然终止改成新的 horizon，也不得
因达到任一上限而伪造完成。每个候选搜索和 independent replay 都记录 native checks 在内
的完整 timing。

执行顺序固定为：

1. focused tests，覆盖 controller boundary、order provenance、K=3、committed-plan/slot
   protection、energy/slack、N/E/F/D/U ledger 和 serialization；
2. 三个场景与所有 controller 的 2-tick smoke，核对输入、字段、索引、hash、序列化和
   replay 输出；
3. CPU-parallel full pilot，按 scene/controller 分开运行；完成 smoke 后，9 个 pilot arms
   与第 4 项独立 reference candidates 允许并发，pilot 使用 9 个 worker；
4. N=3..6 bounded candidate search（首批合计 48 个 rollouts）与独立 whole-cycle tape replay；
   与 pilot 并发时 reference 使用 15 个 worker，总并发上限 24，禁止 nested pools；
5. 逐 scene/controller 汇总 native safety、landing reserve、total wait、all-request wait、
   planning cost、实际 order、N/E/F/D/U 和 censor/unknown 计数。

本批次是 development pilot，不是 qualification、admission、模型选择或 superiority
结论。启动前若发现 protocol、controller 或输出路径需要修订，先记录 revision，再重新跑
focused tests 和 smoke；不消费 Q、不启动 PPO、不改写旧结果。

## 6. 证据闭环与后续学习选择

第一阶段先获得可重放的传统 executable joint reference，再用同一 closed cohort、同一
trajectory/safety/resource authority 做配对比较。信息研究至少包括：

- order authority 与 actual execution 的差异、scheduler slot/committed-plan blocking、
  energy slack 和 LZ service 的相互作用；
- small-N candidate coverage、whole-cycle replay、censor/unknown 分母和 full-cycle wait
  tail；
- fixed/dynamic geometry、long sustained/bursty demand 和 geometry dwell time 的信息研究
  留到第一批之后；它们不是第一批 bounded reference/local reorder 的前置条件；
- common random numbers、独立 train/evaluation seeds、strong backpressure/min-slack、
  rolling MILP/MPC 或小规模可行解/下界的可比范围；
- native trajectory cost、80/40/<10 m 安全事件、reserve/diversion 和 LZ occupancy。

传统参考先闭合；随后按测量结果决定学习路线：

1. 若传统规则已达到稳定、可解释且可执行的参考，优先研究其在静态/动态几何下的作用，
   不要求低层 RL 成功；
2. 若标签显示 imitation 能在相同 authority 下减少等待、错误 release 或 planning cost，
   可登记 IL/DAgger；其准入仍需 fresh protocol、独立 replay 和 heldout 隔离；
3. 若 traditional finite-horizon reference 暴露出可重复的 short-sightedness，可另行登记
   bounded pure-simulation RL research pilot。该 pilot 只允许在预注册 paired comparison、
   现有 native safety/resource authority、独立 seeds 与 replay 下研究是否值得保留；不要求
   先证明 RL gain 才能开展研究试验，paired comparison 完成后才决定是否保留 RL。这个
   research pilot 不等于 PPO admission；当前 PPO_ALLOWED=false，仍需独立 protocol/admission；
4. 学习策略永远不能替代原生 trajectory/safety/resource authority，也不能把 teacher、
   dispatcher、scheduler 或 shield 的实际动作当作 actor requested action。

complete-suffix reservation/optimization、滚动全局协调和 learned value 都属于参考证据之后
的后续工作；它们必须另行冻结 protocol、预算和 replay 边界，不能在第一批 pilot 中隐含启用。

## 7. 静态几何到动态几何

静态几何研究先固定 reliable traditional lower layer，分别比较 fixed_rule、同几何的
local reorder/dispatch variants 以及后续经证据准入的 IL/RL。动态几何只在静态比较形成
完整、可重放的参考后进入，并用固定三环与动态三环、geometry-only、flow-only、joint
三类变化分开注册。动态实验必须跨越 geometry dwell time 的长峰场景，记录
reconfiguration、transition、temporary capacity-loss、queue、service rate、等待和
安全成本；短小 finite cohort 只能作为机制/响应证据。

高层 geometry action、moving reference、affected-interface freeze、canonical trajectory
和 safety contract 继续服从现有设计规范。可靠传统 lower layer 足以支撑 geometry causal
comparison；低层 RL qualification 不是进入该研究的必要条件。

## 8. 硬件、论文与阶段里程碑

硬件放在仿真 reference 和 static/dynamic causal evidence 之后。Crazyflie/Lighthouse/
Crazyswarm2 只验证同一 TrajectoryPlan 的小规模可执行性、SafetyFilter compatibility、
reservation/coordination 和 adapter 行为，不把 2–3 架缩尺实验写成 dense capacity 证明。

阶段里程碑如下：

| 阶段 | 目标 | 状态 |
|---|---|---|
| M0 | 文档、scope、schema/Q/PPO 边界冻结 | 完成/持续维护 |
| M1 | controller authority、serialization、2-tick smoke | 完成 |
| M2 | 三场景 full pilot 与 small-N candidate search | 完成：9/9 pilot、48/48 reference（[证据](evidence/arrival_restructure/first_batch_20261011.md)） |
| M3 | whole-cycle tape、independent replay、joint completion witness | 完成：small reference 四个 selected whole-cycle replay MATCH（[证据](evidence/arrival_restructure/first_batch_20261011.md)） |
| M4 | measured-value gate；决定传统、IL 或 bounded RL 是否继续 | 待 M3 |
| M5 | reliable lower layer 下的 static geometry causal comparison | 待 M4 或传统参考闭合 |
| M6 | dynamic three-ring geometry 与 reconfiguration cost | 待 M5 |
| M7 | optional hardware validation 与论文材料 | 待 M6 |

第一批本身不建立 paper novelty；第一篇只有在 baseline assembly 之外出现独立、可复核的
方法或机制 evidence 后才进入写作/投稿。论文可按证据分成一篇或两篇：第一篇聚焦
energy/resource-constrained executable coordinated arrival、bounded traditional joint reference、
native trajectory/safety/resource accounting；
第二篇只有在 M5–M6 的 geometry causal comparison 形成独立增量时才单独成文。若动态几何
没有独立可复核增量，保留为同一论文中的机制与消融；不把未完成的 IL/RL 或硬件结果写成
贡献。

## 9. 冻结、同步与输出边界

- 原始 checkout 为 D:/7_codespace/terminal，原分支为
  codex/v4-2-v2-1-formal-i1，冻结 HEAD 为
  99c4ffb5af5c106a1aefedb99dc1fe72ed884c2b，tag 为
  codex/frozen-pre-restructure-20261011。tag 记录 committed HEAD，不包含 working-tree
  deletions；原 checkout 中 36 个 .tmp tracked deletions 仅存在于原工作目录并保持不动。
- 当前文档工作区为 D:/7_codespace/terminal/.worktrees/arrival-restructure，分支为
  codex/arrival-restructure，初始 HEAD 同上。代码、脚本和文档修改只在本地该 worktree
  完成，随后由 Syncthing 同步到服务器 ~/projects/terminal/.worktrees/arrival-restructure；
  不使用手工 scp、rsync 或压缩包覆盖源码。
- 运行输出统一写入服务器 $HOME/training_outputs/terminal，同步到本地
  D:/7_codespace/training_outputs/terminal；Git 只在本地操作。新分支目标上游为
  origin/codex/arrival-restructure。
- 第一批结果、protocol、manifest、tape、replay 和 logs 必须带 config snapshot、source
  commit、controller/version、seed、scenario、预算、censor 状态和 native timing。任何
  后续文档只能引用已存在 artifact，不以目录存在或进程启动替代结果。
