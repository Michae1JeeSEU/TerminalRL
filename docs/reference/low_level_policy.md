# 低层共享策略规范

> **当前已资格版本**：V4.2 hierarchical action + Observation V2.1 + K=3 bounded runtime shortlist
>
> **冻结候选**：Observation V2.2 scheduler context（待 fresh closure/learnability qualification）
>
> **适用范围**：Phase A / A5 fixed-airspace low-level learning
>
> **状态**：schema 已冻结；新的 I1、DAgger、PPO 当前仍受
> [项目状态页](../CURRENT_STATUS.md) 的 gate 约束。本轮低层主课程采用 task-neutral
> 语义，完整评估待运行，状态以 [项目状态页](../CURRENT_STATUS.md) 为准；最小实现与单测已完成，
> 配对 baseline 仍待协议执行，再决定有界 PPO repair pilot。
> `PPO_ALLOWED`、formal/heldout 和 static qualification 不因本次文档修订自动放行。

本文件给出低层策略的当前唯一语义。旧 V3/V4/V4.1 设计、迁移原因和逐次实验记录分别保留在
`../history/` 与 `../evidence/`，不再作为实现依据。

当前 V2 车辆/能量边界为 `research_evtol/eh216s_research_v2`：
`min_landing_soc=.10` 是用户选定的研究 floor，`soc_reserve=.25` 仍是独立的
queue/admission compatibility threshold，`SOC_warning=.35`、`SOC_emergency=.20`
和 legacy `emergency_trigger_soc=.35` 保持原语义。`phase_motion_v2` 的无显式
SOC 边界 uniform arrival 默认从 `.35` 开始；这不是实测续航保证。现有 trained
checkpoint 尚未在 dense30 或 V2 vehicle/energy profile 上完成 qualification，
fixed-rule flow 诊断结果也不属于 learned performance。

## 1. 职责边界

共享低层策略 `π_L` 做的是每架 UAV 的调度意图决策：等待，或请求一次带速度/时序残差的
阶段推进。它不直接输出航向角、航迹倾角、位置、姿态或电机命令。

```text
Observation V2.1
→ shared actor π_L
→ requested WAIT / ranked RELEASE shortlist（冻结 K=3）
→ ExecutionScheduler（P2R 同一步按 INNER→MIDDLE→OUTER，再按 priority/wait/UID；
  每架 shortlist 仍按序认证并只执行首个通过项）
→ CanonicalTrajectoryGenerator
→ SafetyFilter
→ PredictiveSafetyShield
→ executable trajectory
```

TeacherScheduler 只在训练和审计中提供 state-conditioned expert label。它不是最终部署组件，
也不得把 policy 请求替换后仍用原请求的 log-probability 做 PPO 更新。

当 `rule_flow_coordination_enabled=true` 时，fixed-rule coordinator 的 normal release
按 `INNER → MIDDLE → OUTER`、remaining energy slack、dynamic priority、UID 处理，
全部 normal 尝试完成后才进入 route-energy/slack emergency pass；这条 opt-in 规则路径
不改变 learned policy 的 action schema，也不把规则诊断当作 checkpoint 结果。

LZ assignment 同样不属于策略动作。历史回放使用 `lowest_id`；P2R 使用
`nearest_available`，且 final 容量从历史 1 显式提升为 3。上述变化只扩大可被认证的资源候选，
不允许绕过 headway、reservation、SafetyFilter 或 shield。

## 2. Observation V2.1

```text
observation_schema_version = phaseA_low_obs_v2_1_retry_state
dimension = 88
```

前 87 维严格继承 topology-aware Observation V2：

| 组 | 维度 | 内容 |
|---|---:|---|
| self / flight | 18 | phase、ring role/progress、speed、SOC、wait、energy/wait urgency、release eligibility |
| mission / resource identity | 12 | 兼容保留的 mission/base 字段、gate、assigned LZ、final/LZ availability |
| semantic topology slots | 45 | 5 个语义邻机槽 × 9D |
| system / flow | 12 | ring occupancy、headway、queue/resource、interface/reconfiguration context |

中性主课程使用 `task_priority_enabled=false`：ArrivalProcess 仍保留 raw mission；base priority
仍按历史随机流抽样但只存储有效值 `1`，以保持 seed 与接口兼容，policy 的 mission one-hot 固定为 `STANDARD`；
88D 宽度、字段位置和 schema hash 不变。raw `emergency` 仅是服务类别 metadata，不能解释为
已启用 emergency mission preference。能量应急 takeover 仍由执行/安全层独立保留。

5 个邻机槽固定为：same-ring leader、same-ring follower、target-ring predecessor、
target-ring successor、nearest predicted conflict。V2.1 仅在索引 87 追加：

```python
retry_remaining = max(0.0, uav.safety_retry_after - env.now)
safety_retry_remaining_norm = clip(retry_remaining / 30.0, 0.0, 1.0)
```

该特征只表示当前 retry mask，不包含未来动作、候选扫描、trajectory certificate 或 Teacher
私有信息。V2 的 87D checkpoint 不能作为 V2.1 的 88D checkpoint 续训。

### 2.1 Observation V2.2 冻结候选

V2.2 使用 schema `phaseA_low_obs_v2_2_scheduler_context_94d`。索引 `0–87` 必须与
V2.1 逐字节一致，只在末尾追加 6 个当前状态字段：

| 索引 | 字段 | 语义 |
|---:|---|---|
| 88 | `interface_active_reservations_norm` | 当前接口 active reservation 数量 |
| 89 | `interface_future_reservations_norm` | 当前接口已登记 future reservation 数量 |
| 90 | `interface_committed_transfers_norm` | 以当前接口目标为终点的 committed transfer 数量 |
| 91 | `interface_reservation_block_remaining_norm` | 最近预留槽与当前 headway 对接口形成的剩余阻塞余量 |
| 92 | `same_step_interface_contenders_norm` | 当前同接口、可实际进入 release 检查的 holder 数量 |
| 93 | `same_step_interface_predecessors_norm` | 确定性执行顺序中位于本机之前的上述 contender 数量 |

计数按系统或源环容量归一化；阻塞余量使用单调 `tanh` 归一化。所有字段均在 policy
动作前从当前环境状态重算。V2.2 不扫描 25 个 proposal，不生成候选轨迹，不读取 SafetyFilter
certificate，也不预测其他策略动作。该 observation 结论不授权 runtime catalogue search；
V2.1 的 88D checkpoint 不能直接加载为 94D policy；closure
pilot 只允许冻结旧策略读取前 88 维作为行为策略，不构成 94D checkpoint 迁移或训练授权。

### 2.2 Observation V2.3 source-bound opt-in

```text
observation_schema_version = phaseA_low_obs_v2_3_source_bound_88d
dimension = 88
```

V2.3 是保持 88D 宽度的 opt-in schema。每架 UAV 使用自身状态，以及由有限通信范围
筛选出的 5 个语义邻机槽；`neighbor_comm_range_m` 必须配置为有限正值。`320 m` 仅是
仿真实验示例，不代表无线电硬件通信半径验证。邻机槽按当前距离过滤，观测构造不扫描
其他全局 active UAV 的轨迹或动作。

起降场中心向每架 UAV 广播环占用、headway、队列、紧急且尚未分配降落点的数量，以及与
本机任务兼容的降落点释放 ETA。该中心广播复用原有 `assigned_lz_usable` 与
`global_free_lz_fraction` 的冗余位置，因此不增加维度；已知释放时间编码剩余等待，未知
的 RESERVED ETA 使用有界不可用值。旧 V2.1/V2.2 schema 保持原有语义。V2.3 使用独立
的字段名和 schema hash，旧 K=3 checkpoint 不能按位置加载到这些变更语义中。

当前实现采用邻机与中心广播零延迟、无丢失的理想通信假设；通信延迟、丢失和鲁棒性仍需
后续单独验证。

## 3. V4.2 hierarchical action

```text
action_schema_version = low_action_schema_v4_2_hierarchical
gate = Categorical(2): WAIT / RELEASE
proposal = Categorical(25), only when gate = RELEASE
```

- `WAIT` 没有 proposal，执行解码固定为 `[0, 0, 0]`；holder speed 由执行/安全层负责；
- `RELEASE` 的 proposal 是 5 个 speed residual 与 5 个 merge-time residual 的笛卡尔积；
- residual levels 均为 `{-1.0, -0.5, 0.0, 0.5, 1.0}`；
- proposal `p = speed_index * 5 + merge_index`，执行 joint index 为 `p + 1`；
- 执行适配器的唯一顺序是 `[release_request, speed_residual, merge_time_residual]`。

物理映射由 vehicle/profile 与 scheduler 配置完成：

```text
delta_v = speed_residual * max_speed_residual_mps
delta_t = merge_time_residual * max_merge_time_residual_seconds
```

V4.2 只改变 policy factorization，不改变 1 WAIT + 25 RELEASE 的物理解码集合。

## 4. 概率、损失与 provenance

BC 使用未加权条件损失：

```text
CE(gate) + CE(proposal | Teacher RELEASE)
```

未来 PPO 若获准，请求动作的 log-probability 必须为：

```text
WAIT:     log P(WAIT)
RELEASE:  log P(RELEASE) + log P(proposal | RELEASE)
```

条件 entropy 为 `H(gate) + P(RELEASE) H(proposal)`。rollout buffer、old log-prob、ratio
和 entropy 只能对应 `requested` action。必须分别记录：

```text
requested gate / proposal / component log-probs
Teacher gate / proposal
execution-applied action
shield-applied action
observation hash
post-state hash
```

Teacher、ExecutionScheduler 或 Shield 的 action 都不能替代 requested action 进入 PPO ratio。

### 4.1 Qualified bounded runtime shortlist

`runtime_proposal_shortlist_k ∈ {1, 3}`，当前正式边界冻结为 K=3；K=1 保留为配对消融。
K=3 不增加网络
前向：gate 与 25-way proposal logits 由同一次前向产生；确定性 RELEASE 请求取 proposal Top-1，
并附带其后两个最高分且不重复的 proposal。运行时按 rank 最多检查三个 canonical candidate，
执行第一个通过 trajectory/SafetyFilter 认证的 candidate；全失败则 RELEASE 被拒绝并进入既有
backoff。该选择属于 runtime safety selection，不是 Teacher repair，不产生 Teacher penalty，且
PPO likelihood 仍只对应 rank-1 requested action。ActionTape 必须保存完整 shortlist、实际 rank、
检查数和 applied action，以支持精确回放。fresh V2 paired qualification 已以 18/18 records、
216/216 requests、安全与 replay/likelihood 全 PASS 冻结此边界；该结论仍不直接授权 PPO、
checkpoint promotion 或 25 项 catalogue scan。

## 5. 执行与安全不变量

1. K=1 只检查 actor 请求的 proposal；冻结 K=3 只可检查同一次 policy forward 给出的
   有序 Top-3，不得扩展为 25 项 catalogue search 或 delayed repair；
2. RELEASE 必须通过 headway、reservation、resource、vehicle dynamics、trajectory 和 safety 检查；
3. SafetyFilter 检查的 trajectory 必须与实际执行的 trajectory 相同；
4. PredictiveSafetyShield 是部署安全层，不是 Teacher；
5. 运行时以 `80 m` desired/planning separation 为首选认证目标；若 bounded
   `PredictiveSafetyShield` 对实际执行轨迹取得与 executor 等价的 swept + recursive
   证明且全程 `>=40 m`，允许接管并继续 episode。`40--80 m` buffer incursion
   单独记录；takeover/intervention（包括零数值修正）计入 shield reward cost；
   预测/实际 `<40 m`、collision event `<10 m`、缺少证明或 uncertified execution
   才在物理执行前 fail-closed；
6. committed trajectory 和 reconfiguration takeover step 不作为普通 actor behavior action。

这里的 `80/40/<10 m` 是当前 EH216-S 仿真的 desired separation、protected/NMAC
floor 和 collision event 研究代理阈值，不是机型适航认证间隔。安全 takeover 不会把
shield 动作伪装成 actor action；其代价和原始 requested action 的 likelihood 必须分开。

此前 qualification 将任何实际低于 `80 m` 的 hard-separation 事件判为失败；
`SafetyInvariantError` 则阻止物理执行并使该回合失败。已归档结果保持不变，
不追溯应用当前的 `40 m` protected-floor 语义。

## 6. 正式训练顺序

```text
S0  schema / semantics qualification
→ I0  fresh structured expert dataset qualification
→ I1  fresh multi-seed hierarchical BC + untouched cohort qualification
→ I2  Safe DAgger preflight and bounded aggregation
→ I3  student-only imitation qualification
→ paired task-neutral baseline on the new course
→ bounded PPO repair pilot (only after admission review)
→ independent validation
→ static-geometry qualification
→ freeze low-level policy
→ A6 dynamic-airspace learning
```

主课程的最小实现必须先通过单测，再以相同 `task_priority_enabled=false` 运行 source/candidate
配对 baseline。课程目标是正常降落、吞吐、完整周期等待和尾部 starvation 监测；保留随机到达、
SOC 差异、能量、安全、环/路径可执行性和资源预约。任务服务时长分布保持 `30/120/10/60 s`，
但不表示服务等级偏好。等待尾部用于监测，不增加每个子组必须无退化的硬 gate。

新 baseline 按协议记录 schema/replay/safety、完整 ledger、full-cycle accounting、等待尾部和
checkpoint lineage 技术核对；普通 shield intervention 可非零，既有 actor takeover fraction
`<=.05` 与 shield/joint fail-closed 为零的安全 gate 保持不变。PPO 仍只在既有 admission
范围内决定；本节不授权 formal/heldout 或 PPO，`PPO_ALLOWED` 仍由独立 admission 决定。

### 6.1 Task-neutral course semantics

`task_priority_enabled=true` 是历史兼容模式；新主课程固定为 `false`。ArrivalProcess 仍保留 raw
mission；base priority 仍抽样以消耗相同随机流，但新课程只存储有效值 `1`，包括 raw `emergency`
mission，policy mission one-hot 固定为 `STANDARD`，Observation V2.1 仍为 88D。
PriorityEngine 保留等待与 SOC/能量公式，但不执行任务等级偏好；执行排序仍使用 dynamic
priority、waiting 和 UID，ring/path feasibility、resource reservation 与 SafetyFilter 作为独立
约束，不能退化为 SOC 单排序。

能量应急接管始终保留在执行/安全层，与后续独立 emergency mission 鲁棒性场景分开。raw
`emergency` 任务标签、`EMERGENCY_LANDED` 结果和能量应急接管不能互换。新课程对照与旧课程
报表不直接跨任务比较；旧 source、candidate_retained、formal/heldout 和失败记录只按历史协议解释。

## 7. 数据与 qualification 纪律

- 每次 schema 或 protocol 修改都使用新版本和 fresh actor；
- qualification cohort 在训练前冻结，只评估一次；
- 已消费 cohort 只能作为 diagnostic reference；
- internal audit 数据不得训练或选模；
- 禁止复制 rows、按 label 丢弃 episode、用旧 Q 训练、未声明 oversampling/class weighting；
- protocol、seed registry、停止上限和 hard gates 必须在 collection 前冻结；
- 一次失败不能靠事后改阈值追溯升级为 PASS。

当前完成度与最新阻塞见 [CURRENT_STATUS.md](../CURRENT_STATUS.md)。
