# 低层共享策略规范

> **当前已资格版本**：V4.2 hierarchical action + Observation V2.1
>
> **冻结候选**：Observation V2.2 scheduler context（待 fresh closure/learnability qualification）
>
> **适用范围**：Phase A / A5 fixed-airspace low-level learning
>
> **状态**：schema 已冻结；新的 I1、DAgger、PPO 当前仍受
> [项目状态页](../CURRENT_STATUS.md) 的 gate 约束。

本文件给出低层策略的当前唯一语义。旧 V3/V4/V4.1 设计、迁移原因和逐次实验记录分别保留在
`../history/` 与 `../evidence/`，不再作为实现依据。

## 1. 职责边界

共享低层策略 `π_L` 做的是每架 UAV 的调度意图决策：等待，或请求一次带速度/时序残差的
阶段推进。它不直接输出航向角、航迹倾角、位置、姿态或电机命令。

```text
Observation V2.1
→ shared actor π_L
→ requested WAIT / ranked RELEASE shortlist（K=1 默认；K=3 候选）
→ ExecutionScheduler（按序认证，执行首个通过项）
→ CanonicalTrajectoryGenerator
→ SafetyFilter
→ PredictiveSafetyShield
→ executable trajectory
```

TeacherScheduler 只在训练和审计中提供 state-conditioned expert label。它不是最终部署组件，
也不得把 policy 请求替换后仍用原请求的 log-probability 做 PPO 更新。

## 2. Observation V2.1

```text
observation_schema_version = phaseA_low_obs_v2_1_retry_state
dimension = 88
```

前 87 维严格继承 topology-aware Observation V2：

| 组 | 维度 | 内容 |
|---|---:|---|
| self / flight | 18 | phase、ring role/progress、speed、SOC、wait、priority、release eligibility |
| mission / resource identity | 12 | mission、gate、assigned LZ、final/LZ availability |
| semantic topology slots | 45 | 5 个语义邻机槽 × 9D |
| system / flow | 12 | ring occupancy、headway、queue/resource、interface/reconfiguration context |

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

### 4.1 Bounded runtime shortlist candidate

经用户显式授权，当前仅允许验证 `runtime_proposal_shortlist_k ∈ {1, 3}`。K=3 不增加网络
前向：gate 与 25-way proposal logits 由同一次前向产生；确定性 RELEASE 请求取 proposal Top-1，
并附带其后两个最高分且不重复的 proposal。运行时按 rank 最多检查三个 canonical candidate，
执行第一个通过 trajectory/SafetyFilter 认证的 candidate；全失败则 RELEASE 被拒绝并进入既有
backoff。该选择属于 runtime safety selection，不是 Teacher repair，不产生 Teacher penalty，且
PPO likelihood 仍只对应 rank-1 requested action。ActionTape 必须保存完整 shortlist、实际 rank、
检查数和 applied action，以支持精确回放。K=3 通过 fresh paired qualification 前仍是 candidate，
不能据此授权 PPO、checkpoint promotion 或 25 项 catalogue scan。

## 5. 执行与安全不变量

1. 默认 K=1 只检查 actor 请求的 proposal；候选 K=3 只可检查同一次 policy forward 给出的
   有序 Top-3，不得扩展为 25 项 catalogue search 或 delayed repair；
2. RELEASE 必须通过 headway、reservation、resource、vehicle dynamics、trajectory 和 safety 检查；
3. SafetyFilter 检查的 trajectory 必须与实际执行的 trajectory 相同；
4. PredictiveSafetyShield 是部署安全层，不是 Teacher；
5. 任一 collision、NMAC、hard separation violation、`SafetyInvariantError` 或 uncertified
   execution 都使对应 qualification fail-closed；
6. committed trajectory 和 reconfiguration takeover step 不作为普通 actor behavior action。

## 6. 正式训练顺序

```text
S0  schema / semantics qualification
→ I0  fresh structured expert dataset qualification
→ I1  fresh multi-seed hierarchical BC + untouched cohort qualification
→ I2  Safe DAgger preflight and bounded aggregation
→ I3  student-only imitation qualification
→ P0/P1  PPO admission and student-controlled PPO
→ environment / geometry curriculum
→ static-geometry qualification
→ freeze low-level policy
```

PPO 之前至少必须证明：schema/replay/safety PASS、WAIT 与 RELEASE 均可学习、student-only
闭环不崩溃、queue 不发散、throughput/completion 健康、shield burden 可接受，以及数据和
checkpoint lineage 可复现。任一核心 gate 为 FAIL 时 `PPO_ALLOWED = false`。

## 7. 数据与 qualification 纪律

- 每次 schema 或 protocol 修改都使用新版本和 fresh actor；
- qualification cohort 在训练前冻结，只评估一次；
- 已消费 cohort 只能作为 diagnostic reference；
- internal audit 数据不得训练或选模；
- 禁止复制 rows、按 label 丢弃 episode、用旧 Q 训练、未声明 oversampling/class weighting；
- protocol、seed registry、停止上限和 hard gates 必须在 collection 前冻结；
- 一次失败不能靠事后改阈值追溯升级为 PASS。

当前完成度与最新阻塞见 [CURRENT_STATUS.md](../CURRENT_STATUS.md)。
