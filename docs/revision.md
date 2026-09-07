# Phase A Landing-only 重构与对齐说明

> **用途**：这是 Codex 从当前仓库迁移到新研究方案的执行对照表。  
> 五份主规范 `research_positioning.md`、`requirements.md`、`design_spec.md`、`technical_spec.md`、`execution_plan.md` 描述的是**目标状态**；本文档描述“当前实现哪里不一致、应该如何迁移、什么时候算对齐完成”。

---

## 1. Canonical document set

采用本 revision 后，`docs/` 中作为当前实现依据的文件只应是：

```text
docs/research_positioning.md
docs/requirements.md
docs/design_spec.md
docs/technical_spec.md
docs/execution_plan.md
docs/revision.md
```

当前仓库中的阶段性历史说明，例如：

```text
docs/CODEX_NEXT_STEPS.md
docs/dynamic_rule_reward_fix_report.md
docs/event_gated_high_level_control.md
```

应从 canonical `docs/` 工作目录删除，或移到不参与 Codex 上下文检索的历史归档位置。最推荐直接依赖 git history，不在当前 working tree 保留第二套设计说明。

`CLAUDE.md`、`AGENTS.md` 等若含具体动作维度、环数量、旧训练阶段，也必须同步更新为“引用上述六份文档”，不得继续重复旧语义。

---

## 2. 当前代码经核对后的主要不一致

### 2.1 当前 `terminal_marl` 是两环

当前 `SimConfig.rings`：

```text
inner: radius 200, height 200
outer: radius 400, height 400
```

`DynamicAirspace.from_ring_specs()` 还显式要求 `len(rings) == 2`。

#### 目标

三环：

```text
outer  600/500
middle 400/350
inner  200/200
```

并改用 `RingRole`。

---

### 2.2 当前高层 action 是 8 维

当前语义：

```text
outer R
inner R
outer H
inner H
outer V
inner V
ring headway
final headway
```

#### 目标

10 维：

```text
dR_outer
dR_middle
dR_inner
dH_outer
dH_middle
dH_inner
dV_profile
dHeadway_O2M
dHeadway_M2I
dHeadway_I2F
```

capacity 继续由 geometry 派生。

---

### 2.3 当前低层 intention 是 4 维且语义未真正实现

当前 `SafetyFilter` 读取的动作近似为：

```text
release
speed
slot
timing
```

而部分旧说明中顺序不同。

更重要的是当前环境中：

- `slot_residual` 实际再次修改 speed；
- `timing_residual` 实际修改 release threshold；
- 并未形成真正的 target slot/time reservation。

#### 目标

删除 `slot_residual`，统一 3 维：

```text
[release_request, speed_residual, merge_time_residual]
```

`merge_time_residual` 必须真正进入 MergeScheduler。

---

### 2.4 当前 UAVState 缺失场景核心任务语义

当前主 `UAVState` 没有完整：

```text
mission_type
base_priority
dynamic_priority
LZ assignment/service
phase-dependent energy
```

只保留简化的 endurance/time bookkeeping。

#### 目标

恢复并重新设计为：

```text
mission
base/dynamic priority
energy/SOC
LZ reservation/service
full lifecycle timestamps
outcome ledger
```

---

### 2.5 当前 energy 是统一 endurance-seconds

当前主要按 elapsed time 减少 `remaining_endurance_seconds`，无法区分 holding、transfer、final、external wait。

#### 目标

phase-dependent energy model + SOC thresholds。

`battery_endurance_seconds` 如保留为 diagnostic helper，也不能成为 canonical reward、priority 或 outcome 依据。

---

### 2.6 当前没有真正的多 LZ service model

旧 `src/` 中存在：

```text
6 discrete LZ
reservation
30/120/10/60 s mission service
```

但没有完整迁入 `terminal_marl` 主线。

#### 目标

新增 `LandingZoneManager`：

```text
FREE → RESERVED → OCCUPIED → FREE
```

touchdown 不是 mission completion，service completion 才是默认任务完成。

---

### 2.7 当前 4 gate 并非 4 条真实 inbound corridor

虽然有 `gate_count = 4`，但生成逻辑仍主要依赖单一 `inbound_bearing_deg` 后再做队形/横向偏移。

#### 目标

4 个独立 radial gate/corridor，从对应上游 spawn point 进入。

---

### 2.8 当前 trajectory generator 与实际 transfer execution 不完全统一

当前：

- `TrajectoryGenerator.ring_transfer()` 使用 smooth radial/vertical transition；
- VTAC 中又存在 cubic Bézier transition curve。

这会造成 `checked trajectory != executed trajectory` 的风险。

#### 目标

只有一个 canonical `TrajectoryPlan`：

```text
schedule
→ generate
→ safety check
→ commit
→ execute same plan
```

VTAC 若保留，只做 executor/guidance。

---

### 2.9 当前 merge point 主要依赖固定 offset

当前存在 `outer_to_inner_release_offset_deg`、`inner_to_final_release_offset_deg` 等固定几何设置。

#### 目标

由目标环 predicted gap 和 merge time 动态确定 merge angle。

---

### 2.10 当前 observation 以 nearest-k 欧氏邻机为主要上下文

#### 目标

使用交通拓扑：

```text
same-ring leader/follower
target-ring predecessor/successor
nearest predicted conflict
```

actor 对 N 不敏感，critic 使用 set/attention pooling。

---

### 2.11 当前 high-level PPO 训练配置主要仍使用 rule-only low level

说明当前并没有真正完成两个 learned policy 的 HRL 闭环。

#### 目标

按：

```text
rule validation
→ train low
→ freeze low
→ train high
→ compose
```

推进。

---

## 3. 不再保留的历史设计

以下不是 optional legacy mode，而是从主研究设计中删除：

1. 两环作为主场景——只保留为 formal ablation；
2. 直接自由飞行动作——不恢复任意 heading / flight-path-angle / arbitrary speed 的主线动作；
3. `slot_residual`；
4. 固定 ring release angle 作为正式 merge 核心；
5. 统一 endurance countdown；
6. touchdown 后立即释放 landing resource；
7. 单 bearing 伪四入口；
8. 多套 experiment mode 同义词。

正式模式收敛为：

```text
fixed_rule
fixed_low_rl
dynamic_rule
dynamic_hrl
```

---

## 4. 需要保留并升级的当前实现资产

此次 revision 不是推倒重来。以下思想/模块值得保留：

### 4.1 `terminal_marl/` 作为唯一主线

继续。`src/` 只作为参考旧实现抽取 mission/LZ/priority 语义，不作为第二运行环境维护。

最终建议删除 `src/` 运行入口或明确标记为 non-canonical legacy reference，避免 Codex 同时修改两套环境。

### 4.2 PettingZoo-style per-agent action
保留。

### 4.3 Shared policy batching
保留并扩展为 3D intention。

### 4.4 Arrival request 与 admission 分离思想
保留。

### 4.5 Finite-cohort outcome accounting
保留，并加入 `DIVERTED` 与 service completion。

### 4.6 Dynamic capacity from actual radius
保留，扩到三环。

### 4.7 Event-gated geometry
保留，扩到三环 structural dimensions。

### 4.8 Geometry interpolation
保留，升级 committed transfer/interface-freeze 语义。

### 4.9 Trajectory generator / Bézier 思路
保留，但统一成唯一 trajectory backend。

### 4.10 Safety filter
保留，但从动作偏转器升级为 intent/resource + trajectory verifier。

---

## 5. Schema migration

### 5.1 Config

旧：多个 ad-hoc/legacy preset。  
新：

```text
schema_version: phaseA_landing_v1
```

### 5.2 Action

旧 low：4D。  
新：

```text
low_intent_v3_factorized_discrete
```

旧 high：8D。  
新：

```text
high_airspace_v1_10d
```

### 5.3 Observation

旧 checkpoint observation 不兼容。新：

```text
phaseA_obs_v1
```

### 5.4 Airspace

旧：two-ring。  
新：

```text
three_ring_v1
```

---

## 6. Checkpoint policy

所有修改前 checkpoint：

```text
diagnostic-only
```

不得：

```text
resume training
claim formal comparison
mix old and new reward curves
```

loader 检查 metadata schema，不匹配直接报错。

---

## 7. 推荐代码迁移顺序

1. **Revision Step 1 — enums/config**：让仓库能表达新模型；
2. **Step 2 — DynamicAirspace 3 ring**；
3. **Step 3 — UAV lifecycle/energy/priority**；
4. **Step 4 — LZ service**；
5. **Step 5 — arrival/admission/four gates**；
6. **Step 6 — rule pipeline**：先跑通状态机；
7. **Step 7 — merge scheduler / canonical trajectory**；
8. **Step 8 — safety**：保证同一 plan 被检查/执行；
9. **Step 9 — observation/action**：正式 3D shared intent；
10. **Step 10 — low-level PPO**；
11. **Step 11 — high-level PPO**；
12. **Step 12 — experiments**。

---

## 8. 必须先从旧 `src/` 抽取再停止依赖的语义

旧 `src/` 中有当前主线丢失但需要恢复的设计，可作为迁移参考：

```text
num_landing_zones = 6
mission type probabilities
MISSION_TYPE_LZ_OCCUPANCY
base priority distribution
priority formula coefficients
battery critical threshold
LZ reservation concept
warm-start ring distribution
```

抽取后，目标实现必须落到 `terminal_marl/` 新模块中，不允许运行时 import `src.*`。

---

## 9. 数值参数的治理原则

### 9.1 Canonical semantics

不能随意改：

```text
3 rings
4 entry gates
6 default LZ
mission probabilities
service times
3D low action
10D high action
180s high-level interval
event-gated structural geometry
```

### 9.2 Engineering defaults

可以经实验调整，但必须有版本记录：

```text
600/400/200 m radii
500/350/200 m heights
28/24/20 m/s speeds
80 m safe distance
headways
geometry bounds
reward weights
energy multipliers
```

### 9.3 Calibration parameters

不应在论文中宣称普适：exact power multipliers、specific geometry bounds、exact demand rates、reward coefficients。

---

## 10. Formal outcome changes

新建议：

```python
class TerminalOutcome(str, Enum):
    COMPLETED = "completed"
    DIVERTED_LOW_SOC = "diverted_low_soc"
    DIVERTED_CAPACITY = "diverted_capacity"
    FAILED_COLLISION = "failed_collision"
    FAILED_ENERGY = "failed_energy"
    INVALID = "invalid"
    UNRESOLVED = "unresolved"
```

finite cohort：

```text
requested =
completed
+ diverted_low_soc
+ diverted_capacity
+ failed_collision
+ failed_energy
+ invalid
+ unresolved
```

正式 run：

```text
invalid = 0
unresolved = 0
```

---

## 11. Reward migration

不要复用旧 reward 数量级。原因：

- mission completion 从 touchdown 改为 service complete；
- external wait 加入；
- energy 模型变化；
- ring 数量变化；
- action 维度变化；
- final/LZ 资源变化。

因此旧 reward weights 与新模型不兼容。新 low/high reward 应在 rule pipeline 稳定后重新做 diagnostics。

---

## 12. Experiment migration

旧 PPO 实验只作为开发历史。

新 paper evidence 从：

```text
fixed_rule
fixed_low_rl
dynamic_rule
dynamic_hrl
```

开始，然后做：

```text
two_ring_dynamic_hrl
periodic_geometry_hrl
homogeneous_service
no_dynamic_priority
fixed_headway
no_safety_filter
```

不得把旧 `high-level-ppo` 与新 `dynamic_hrl` 混为同一条件。

---

## 13. Main-paper scenario reset

新正式主场景：

```text
t=0 terminal empty
arrival requests start according to demand process
```

旧 ring prepopulation：

```text
60% outer
25% middle
15% inner
```

保留为 `warm_start_rings` debug/curriculum，不作为主论文 traffic scenario。

---

## 14. Geometry transition migration

当前已有 interpolation 思路，新增统一语义：

```text
actual geometry != target geometry during transition
capacity uses actual geometry
ring holders follow actual geometry
new transfers across affected interface frozen
committed transfers finish
```

如果 high-level target 会导致 committed transfer 无法安全落到 target ring，则 action 在 commit 前被拒绝，而不是执行中强改轨迹。

---

## 15. LZ migration details

### 不要

```text
final 无飞机 = pad 可用
```

### 必须

分别维护：

```text
final corridor resource
LZ reservation
LZ occupied/service
```

一个 UAV touchdown 后，final corridor 可以释放；但其 LZ 仍可能继续占用 10/30/60/120 s。两者是不同资源。

---

## 16. Low-level variable-N migration

保留当前 per-agent `action_spec(agent)` 思路，不回到 `max_n × action_dim` 的整体动作。

训练：

```text
active observations
→ stack
→ shared actor
→ unstack dict actions
```

critic：

```text
masked set pooling
```

需要测试 N=1、5、20，以及 episode 内 N 动态变化。

---

## 17. Observation migration

第一轮不要追求“把旧 91 维全部搬过去”。按语义重新构建：

```text
self
same-ring topology
target-ring merge context
resource context
compact global state
```

每个 feature 有名字、slice、normalization、presence mask。

旧 nearest-k raw 3D state 与旧 safety `psi/gamma offset` feature 不作为 canonical observation。

---

## 18. Safety migration

当前 primary safety action 若仍有 `psi_offset/gamma_offset`，应从主路线移除。

新 safety response：

```text
approve
modify timing
reduce residual
delay
hold
reject
```

若极端 emergency 需要 emergency maneuver，必须作为独立 deterministic emergency controller 并记录为异常 fallback，不能作为 RL 正常运行的一部分。

---

## 19. Codex 首轮修改完成标准

第一轮**不是 PPO 能跑就算完成**。必须满足：

```text
[ ] docs only have one canonical design
[ ] 3-ring config and DynamicAirspace
[ ] 10D high-level action
[ ] 3D low-level action
[ ] mission/base priority/dynamic priority
[ ] phase-dependent energy/SOC
[ ] 6 LZ + reservation/service
[ ] 4 real inbound corridors
[ ] external queue
[ ] complete lifecycle
[ ] rule-only one UAV completes service
[ ] rule-only multi-UAV no deadlock
[ ] no LZ double booking
[ ] one canonical transfer trajectory
[ ] safety checks executed trajectory
[ ] outcome ledger closes
```

做到这里才进入低层训练。

---

## 20. Repository clean-up checklist

Codex 完成迁移后执行全文搜索。

### 应只在 revision/history comment 中出现

```text
two-ring
ACTION_SIZE = 8
slot_residual
battery_endurance_seconds
release_offset_deg
psi_offset
gamma_offset
```

### 主代码不得出现逻辑依赖

```text
len(rings) != 2
reshape(4) for low intention
np.zeros(8) for high-level action
```

### 配置不得继续使用

```text
legacy_residual
old 2-ring preset
old low-level placeholder
```

---

## 21. 与未来 Phase B/C 的隔离

本轮 revision 完成后，不新增：

```text
takeoff queue
departure ring
departure corridor
arrival/departure conflict reward
```

当前必须先回答 landing-only 的研究问题并建立可解释基线。Phase B/C 只在 Phase A 核心实验稳定后另开 revision。

---

## 22. 给 Codex 的最终执行指令摘要

> 以 `terminal_marl/` 为唯一运行主线；按六份 canonical docs 把当前两环/简化资源模型重构为 landing-only 三环 terminal management。先恢复任务、电量、priority、离散 LZ service、真实四入口和 external queue，再统一 3D low-level intention、merge scheduler、canonical trajectory 和 trajectory-level safety；rule-only 全链路通过后训练低层；低层冻结后再训练 10D high-level dynamic-airspace policy。不要为兼容旧 checkpoint、旧 action schema 或旧文档保留第二套运行语义。
