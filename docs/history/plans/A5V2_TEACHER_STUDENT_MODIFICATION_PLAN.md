# A5-v2 Teacher–Student Low-Level Revision：详细代码修改与训练方案

> **版本**：2026-08-26 / `a5v2_teacher_student_plan_v1`  
> **用途**：作为 Codex 后续代码修改的逐步执行规格。若与 2026-08-26 之前的 A5/A6 “冻结/禁止训练”指令冲突，以本方案和同步修订后的六份 canonical docs 为准。  
> **核心原则**：不改变 Phase-A 三环/LZ/3D low action/10D high action/hard-safety thresholds；改变的是 **信息输入、规则模块的职责边界、teacher→student 训练路径与正式部署的安全干预强度**。

---

## 1. 为什么需要这次 revision

当前实现有三个结构性问题：

1. **Actor information gap**：`A5_OBSERVATION_SCHEMA` 为 41D，主要是 self/task + occupancy/headway/resource summary，没有实现原设计中的 same-ring leader/follower、target-ring predecessor/successor，也没有 CPA/TTC；因此 policy 很难自己学习局部冲突协调。
2. **Rule dominance**：A6.12 `repair_candidates()` 会在 ±2 m/s、±6 s 及 future delay 内主动搜索安全 correction；SafetyFilter/runtime controller 又继续修复。系统 hard-safe 并不自动等于 low policy 学到了 hard-safe coordination。
3. **Variable-N evidence incomplete**：当前 shared actor + masked mean/max critic 能处理 active-agent set size，但 actor 并未编码 variable surrounding traffic。需要把“网络能跑不同 N”升级为“在 unseen N 下仍能利用局部 topology 自主协调”。

因此目标不是删除所有 deterministic safety，而是：

> **把正常交通决策权交给 learned policy；把 repair-capable rules 变成训练老师；部署时只保留资源/执行约束、exact trajectory verifier 和少量预测式安全兜底。**

---

## 2. 目标架构与模块职责

### 2.1 正式 deployment path

```text
Observation V2
→ Shared π_L
→ requested 3D intention
→ ExecutionScheduler
→ one requested MergeProposal / reservation
→ CanonicalTrajectoryGenerator
→ SafetyFilter verification
→ PredictiveSafetyShield (only if future hard risk)
→ atomic commit
→ execute checked plan
→ RuntimeSeparationMonitor
```

### 2.2 Training-only path

```text
π_L requested action
→ TeacherScheduler can search bounded correction
→ teacher-corrected action may be executed
→ correction stored as label
→ PPO still attributes sampled action to requested action
→ reward penalizes teacher dependence
→ optional BC/auxiliary loss learns correction
```

### 2.3 模块不能做什么

- `ExecutionScheduler`：不能枚举 alternative `Δv/Δt/future delay`；
- `SafetyFilter`：不能决定哪一个 policy action 更优；
- `PredictiveSafetyShield`：不能频繁承担普通 sequencing；
- `TeacherScheduler`：不能在 student qualification / paper proposed deployment preset 中启用；
- `π_L`：不直接输出任意 heading/pitch/altitude；
- 任一安全模块：不能降低 `safe_distance=80 m` 或 silent rewrite committed plan。

---

## 3. Low Observation V2：exact schema

Schema name：`phaseA_low_obs_v2_topology_87d`。

### 3.1 Block A — self / flight state：18D

| Feature | Dim | Normalization / semantics |
|---|---:|---|
| phase one-hot | 8 | canonical active phases |
| ring role one-hot | 3 | Outer/Middle/Inner; non-ring all zero |
| ring progress | 2 | `sin(theta), cos(theta)` |
| speed | 1 | `/ max_speed` |
| SOC | 1 | `[0,1]` |
| total wait | 1 | clipped by priority wait reference |
| dynamic priority | 1 | versioned scale |
| release eligibility | 1 | hard state eligibility only, not rule recommendation |

### 3.2 Block B — mission / resource identity：12D

| Feature | Dim |
|---|---:|
| mission one-hot | 4 |
| assigned gate one-hot | 4 |
| assigned-LZ-present | 1 |
| assigned-LZ-currently-usable | 1 |
| final-resource-available | 1 |
| free-LZ fraction | 1 |

### 3.3 Block C — semantic traffic slots：45D

固定槽位：

1. `same_ring_leader`
2. `same_ring_follower`
3. `target_ring_predecessor`
4. `target_ring_successor`
5. `nearest_predicted_conflict`

每槽 9D：

| Feature | Meaning |
|---|---|
| presence | no neighbor => entire slot zero |
| path/gap distance | signed/role-aware normalized separation |
| Euclidean distance | normalized by safe-distance/reference range |
| relative speed | neighbor minus ego along relevant path |
| closing rate | positive means closing |
| `d_CPA` | closest-point predicted distance |
| `t_CPA` | time to CPA, clipped to shield horizon |
| priority delta | neighbor priority - ego priority |
| reserved/committed | any relevant reservation/committed-plan flag |

### 3.4 Block D — system / flow：12D

- three ring occupancy/capacity ratios：3；
- three headways：3；
- external queue ratio：1；
- final queue pressure：1；
- global free-LZ fraction：1；
- current interface frozen：1；
- current interface headway slack：1；
- reconfiguration active/pending：1。

### 3.5 Neighbor selection rules

**Leader/follower**：按同环 directed ring progress，而不是 Euclidean distance。  
**Target predecessor/successor**：按 ego requested/baseline merge time 预测 target-ring phase，在 prospective merge angle 的前后选择。  
**Nearest predicted conflict**：按未来 horizon 内 predicted risk 排序，优先最小 swept separation / CPA margin；不是 nearest-k。  
**No leakage**：不能读取 future arrivals、future reward、post-action result、teacher correction、outcome label。

---

## 4. Variable-N 设计与 Waltz et al. comparator

参考：Martin Waltz, Ostap Okhrin, Michael Schultz, *Self-organized free-flight arrival for urban air mobility*, Transportation Research Part C 167 (2024) 104806, DOI `10.1016/j.trc.2024.104806`。

该工作使用 shared decentralized policy，并使用 LSTM-TD3 / spatial-temporal recurrent processing 来处理 partial observability 和 variable number of surrounding vehicles。我们的研究不应直接复制算法，而应把问题拆成：

1. **representation**：all-neighbor sequence vs structured topology slots；
2. **temporal memory**：feed-forward vs GRU/LSTM；
3. **RL optimizer**：PPO vs TD3 family。

推荐实验顺序：

```text
A. legacy 41D PPO
B. topology-87D PPO                 ← 主候选
C. topology-87D PPO + GRU           ← memory ablation
D. all-neighbor recurrent/attention PPO
E. only if needed: LSTM-TD3 replication/comparator
```

不要一次同时改 representation + recurrence + optimizer，否则无法归因。

---

## 5. Scheduler split：具体代码责任

### 5.1 `MergeScheduler` common primitives

保留/抽取：

- interface key；
- baseline merge time；
- target-ring phase prediction；
- gap geometry；
- `MergeProposal` / `MergeReservation`；
- reserve/cancel/complete；
- committed/executed headway bookkeeping。

### 5.2 `ExecutionScheduler`

推荐新文件：`terminal_marl/core/execution_scheduler.py`。

接口示例：

```python
class ExecutionScheduler:
    def propose_requested(self, uav, intent, snapshot, target) -> MergeProposal:
        ...

    def check_requested(self, proposal, uav, snapshot) -> SchedulerDecision:
        ...

    def reserve_requested(self, proposal, uav, snapshot) -> MergeReservation:
        ...
```

重要：只有一个 requested proposal。可以拒绝/等待，但不能自动搜 `speed_repair/merge_time_repair/future_delay`。

### 5.3 `TeacherScheduler`

推荐新文件：`terminal_marl/core/teacher_scheduler.py`。

迁移当前 `repair_candidates()`：

```python
class TeacherScheduler:
    def candidates(self, requested, ...): ...
    def choose_verified(self, ...): ...
```

Teacher correction 的排序可继续用“minimum intervention first”。必须输出：candidate kind、normalized/physical correction、delay、plan id、SafetyFilter witness、predicted min separation before/after。

### 5.4 Config migration

替换模糊 bool：

```yaml
low_level:
  scheduler_mode: execution
low_level_teacher:
  enabled: false
  mode: teacher_full
```

formal evaluation config 必须校验 `teacher.enabled == false`。

---

## 6. Predictive Minimal Safety Shield

### 6.1 不再使用“距离已经小于安全距离才救”的逻辑

触发依据：

```text
predicted swept minimum separation < safe_distance
OR d_CPA < safe_distance with t_CPA in shield horizon
```

可增加 buffer/hysteresis 避免临界抖动，但 buffer 不得改变正式 hard threshold 定义。

### 6.2 Intervention ladder

默认：

1. hold/block an **uncommitted release**；
2. bounded `speed_residual` correction on **uncommitted ring holder**；
3. bounded delay on **uncommitted merge/release**；
4. fail closed if no verified correction。

当前不默认加入 hover。若后来确实需要 hover：单独增加 flight phase/energy/kinematic semantics + tests，不要把 hover 作为临时 hack。

### 6.3 Committed traffic

- committed trajectory 绝不 rewrite；
- shield 只能调整未 committed traffic；
- correction 后生成新的 tracks；
- `SafetyFilter.verify_predicted_tracks()` final approve；
- checked/executed plan id 必须一致。

### 6.4 Telemetry

```text
shield_intervention_id
trigger_time
trigger_reason
conflict_pair
predicted_min_sep_before
d_cpa / t_cpa
intervention_type
requested_action
shield_action
correction_magnitude
predicted_min_sep_after
verified
```

---

## 7. PPO 与 Teacher correction：必须避免的实现错误

### 7.1 正确语义

Policy sample：

\[
a_t^{req} \sim \pi_{\theta_{old}}(\cdot|o_t)
\]

Teacher/shield 可以成为 environment mapping：

\[
a_t^{exec}=F(o_t,a_t^{req})
\]

PPO transition 仍记录 `a_req` 与 `log π_old(a_req|o)`，reward/next-state 来自 `a_exec` 造成的环境结果。

### 7.2 禁止语义

```text
sample a_req + logprob(a_req)
teacher changes to a_teacher
store a_teacher in PPO action field
still use logprob(a_req)
```

这会破坏 PPO ratio 的行为动作对应关系。

### 7.3 Optional teacher loss

```python
loss = ppo_loss + bc_weight * teacher_loss
```

Teacher loss 只在 `teacher_used` sample 上计算；建议 weighted Huber。必须单独统计 `ppo_actor_loss / value_loss / entropy / teacher_bc_loss`。

---

## 8. Low-level reward V2

核心目标：鼓励 progress + efficiency，同时让 policy 学会减少 teacher/shield 依赖，而不是学会完全不行动。

建议：

```text
+ stage progress
+ completed merge
+ final entry
+ touchdown
- elapsed wait
- energy
- requested residual magnitude
- teacher intervention
- teacher correction magnitude
- shield intervention
- shield correction magnitude
- predictive risk margin
- starvation / failure
```

需要额外记录：

- release-request rate；
- approved release rate；
- no-release duration；
- intervention vs progress correlation。

若 intervention penalty 上升后 release-request collapse，先修 reward，不把它当“更安全”。

---

## 9. Teacher annealing protocol

不通过随机关闭 hard safety 来制造 curriculum。退火的是 routine teacher authority。

### Phase L0 — teacher dataset

Teacher full，采集 correction/no-correction balanced data。

### Phase L1 — full teacher PPO

允许 current bounded full repair search。目标：policy 学会基本 progress，并开始减少 correction magnitude。

### Phase L2 — immediate-only teacher

禁用 future-delay grid，只保留 requested/immediate bounded repair。

### Phase L3 — label-only teacher

Teacher 仍计算“如果修会怎么修”，用于 BC/diagnostics，但 environment 不采用 routine teacher correction；只有 hard shield 可以改动作。

### Phase L4 — teacher OFF

完全 student execution + predictive shield。只有 L4 的结果可以进入 low-level formal qualification。

退火条件建议基于 rolling metrics 而不是固定 steps 单一条件：teacher intervention rate、correction norm、progress、hard-safety predictor margin。可以设置最大步数防止无限停留。

---

## 10. Low-level qualification protocol

### 10.1 Hard gate

所有 protected episodes：

```text
collision = 0
unique NMAC = 0
hard separation violation = 0
teacher_enabled = false
```

### 10.2 Independence gate

在正式 test 之前预注册。默认 engineering gate：

- ID：≤ 10 shield interventions / 100 release intents；
- OOD stress：≤ 20 / 100 release intents。

还要报告 intervention magnitude 和 per-UAV distribution，不允许只靠平均值隐藏少数 UAV 被频繁救场。

### 10.3 Traffic matrix

至少：

```text
N: 1,5,10,20,40,60
ID N range
OOD lower N
OOD higher N
medium/high/bursty/overload
unseen seeds
```

训练时必须记录 `training_N_range`，避免 test-N 不小心泄漏进入 curriculum。

### 10.4 Success is not only safety

同时要求：

- no unresolved lifecycle bug；
- completion / touchdown rate 合理；
- external/internal wait 不出现极端保守退化；
- release-request distribution 非塌缩；
- LZ/reservation consistency 0 violation。

---

## 11. High-level A6-v2 路径

### 11.1 Entry gate

只有 low-v2 L4 qualification 通过后才能冻结 low checkpoint并训练 high。

### 11.2 High-level safety role

`ReconfigurationSafetyCoordinator` 需要像 low scheduler 一样区分：

**Deployment feasibility supervisor**：
- engineering bounds/order；
- event gate/deadband；
- occupancy-aware shrink；
- interface freeze；
- committed traffic protection；
- necessary transition delay。

**Training/diagnostic teacher repair**：
- holder speed re-spacing；
- merge/release retiming；
- safe gap reservation；
- aggressive candidate repair search。

Formal proposed condition 不应依赖后者频繁替 high/low policy优化。

### 11.3 Low actor must see reconfiguration context

V2 已加入 `interface_frozen` 与 `reconfiguration_active/pending`。如果后续发现高层 desired transition 需要更具体 coordination，可在 V2.1 revision 增加**有限、online、无泄漏**的 pending-interface/time-window features；不要把 future outcome 作为 observation。

---

## 12. 逐文件修改清单

### `terminal_marl/core/config.py`

- schema version；
- teacher/scheduler/shield config；
- qualification thresholds；
- reward coefficients；
- fail-closed validation：formal deployment + teacher enabled => error。

### `terminal_marl/core/observations.py`

- 保留 V1 legacy builder；
- 新增 V2 schema/builder；
- 不把 neighbor-selection 逻辑全挤在同一文件，建议抽模块。

### `terminal_marl/core/topology_neighbors.py`（建议新增）

- ring leader/follower；
- target pred/succ；
- CPA utilities / conflict ranking；
- deterministic tie-breaking by uid。

### `terminal_marl/core/merge_scheduler.py`

- 保留 common reservation/gap primitives；
- 迁走 repair search；
- 明确 `propose_requested()`。

### `terminal_marl/core/execution_scheduler.py`（建议新增）

- single candidate；
- no action optimization。

### `terminal_marl/core/teacher_scheduler.py`（建议新增）

- current repair candidate logic；
- teacher witness/correction object。

### `terminal_marl/core/safety_filter.py`

- 保持 verifier；
- 增加/标准化 risk result fields；
- 不加入 action search。

### `terminal_marl/core/runtime_traffic_safety.py`

- 重命名/重构为 predictive shield semantics；
- CPA/swept pre-breach trigger；
- intervention ladder；
- telemetry。

### `terminal_marl/envs/parallel_env.py`

- action trace provenance；
- teacher path only in training preset；
- step ordering；
- PPO requested action exposure；
- separate `last_teacher_decision`, `last_shield_decision`。

### `terminal_marl/algo/shared_ppo.py`

- V2 dimensions/network namespace；
- Transition 增加 requested/applied metadata；
- update 只读 requested action；
- optional BC loss；
- v1 checkpoint loader isolation。

### `terminal_marl/training/train_low_level_ppo.py`

- teacher phase curriculum；
- curriculum state saved in checkpoint；
- ID/OOD-N split；
- intervention/progress diagnostics；
- no hidden fallback to rule/teacher in L4。

### New training scripts

建议：

```text
training/build_low_teacher_dataset.py
training/train_low_v2_bc.py
training/eval_low_v2_student.py
training/compare_low_representations.py
```

### High-level files

在 low-v2 gate 前只做 compatibility changes，不启动长训练。Gate 后再修改/训练 `reconfiguration_safety.py`、high wrapper、high PPO configs。

---

## 13. Test plan

### Unit

- V2 dimension/slices/norm；
- leader/follower wrap-around；
- target pred/succ at merge time；
- CPA numerical sanity；
- neighbor absence masks；
- agent-order permutation invariance；
- ExecutionScheduler exactly one candidate；
- TeacherScheduler reproduces known repair witness；
- shield safe no-op；
- shield future conflict intervention before breach；
- committed plan immutability；
- PPO requested action/logprob consistency。

### Integration

- single UAV complete landing；
- N=10/20 fixed rule regression；
- teacher full vs execution scheduler CRN；
- teacher-off low-v2 smoke；
- burst scenario；
- dynamic N join/terminate；
- LZ service/reservation no regression。

### Regression

保留 A6.12 frozen unsafe/repaired states，作用改变为：

- TeacherScheduler 应能复现这些 repair witnesses；
- ExecutionScheduler 不应偷偷做这些 search；
- trained student 应尽量自己请求接近 witness 的安全 action；
- shield 只在 student request 仍预测危险时最小救场。

---

## 14. Experiment / paper evidence matrix

### Representation

| Condition | Observation | Memory | RL | Teacher in eval |
|---|---|---|---|---|
| A5 legacy | 41D summary | none | PPO | existing backend |
| topology student | 87D semantic | none | PPO | OFF |
| topology recurrent | 87D semantic | GRU/LSTM | PPO | OFF |
| all-neighbor comparator | variable set/sequence | recurrent/attention | PPO first | OFF |
| optional paper replication | all-neighbor recurrent | LSTM | TD3 | OFF |

### Teacher dependence

- full teacher eval（diagnostic upper bound）；
- annealed checkpoint；
- student teacher-OFF；
- shield-OFF simulation only。

### Generalization

- unseen N；
- unseen demand distribution；
- sensor noise / delay optional；
- computational scaling vs N。

---

## 15. Codex commit sequence

建议每一批独立提交，避免一次改太多无法定位 regression：

1. `docs: adopt A5-v2 teacher-student architecture`  
2. `feat: add topology observation v2 schema`  
3. `test: cover semantic neighbor selection and variable-N`  
4. `refactor: split execution and teacher schedulers`  
5. `feat: add predictive minimal safety shield`  
6. `feat: track requested teacher execution shield actions`  
7. `fix: keep PPO behavior action aligned with requested action`  
8. `feat: add teacher dataset and optional imitation warm start`  
9. `feat: add teacher-annealed low-v2 training`  
10. `eval: add student-only variable-N qualification`  
11. `eval: compare topology and all-neighbor representations`  
12. `feat: enable A6-v2 high-level training after low gate`

每个 commit 前后跑其对应最小 tests；不要等所有代码都写完才跑一次总测试。

---

## 16. Stop / rollback conditions

出现以下任一情况，不要直接增加训练步数：

- V2 information audit 仍无法区分 harmful/safe release；
- teacher intervention rate 不随训练下降；
- policy 通过“几乎永不 release”降低 intervention；
- teacher-off 后 hard-safety 立刻大量失败；
- shield intervention 高于预注册 gate；
- topology model OOD-N 明显差于 all-neighbor comparator；
- PPO loss 正常但 behavior/action provenance 不一致；
- SafetyFilter checked plan 与 executed plan id 不一致。

先做 causal/observation/reward audit，再决定是否引入 GRU/LSTM、attention 或 TD3。

---

## 17. Definition of Done for this revision

只有同时满足以下条件，才能宣布 A5-v2 完成并进入正式 high-level：

```text
[ ] 87D topology observation canonicalized
[ ] action remains exactly 3D
[ ] ExecutionScheduler contains no routine repair search
[ ] TeacherScheduler is training-only and reproducible
[ ] PredictiveShield acts before hard breach and rarely intervenes
[ ] PPO buffer/logprob tracks requested action correctly
[ ] teacher annealing reaches teacher OFF
[ ] student-only ID/OOD-N safety gate passes
[ ] intervention independence gate passes
[ ] no trivial throughput collapse
[ ] low-v2 checkpoint/schema/hash frozen
[ ] representation comparison protocol ready
```

完成后再进入 A6-v2，而不是回到旧的“高层先继续调、低层永远冻结”的路线。
