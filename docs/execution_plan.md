# 执行计划

> **当前工程目标**：只完成 Phase A — Landing-only。  
> Phase B（departure-only）和 Phase C（arrival + departure）只保留在研究路线中，不进入本轮 Codex 修改。  
> 本执行计划按“先修正环境语义，再训练低层，再训练高层”的依赖顺序推进。

---

## 1. 总原则

本轮不是对当前两环代码继续局部调参，而是进行一次 **core model alignment**。

开发顺序必须遵守：

```text
domain semantics
→ deterministic rule pipeline
→ trajectory/safety correctness
→ low-level learning
→ high-level dynamic airspace
→ hierarchical evaluation
```

禁止倒序：

```text
继续训练旧 PPO
→ 发现场景缺机制
→ 再改 observation/action/reward
```

旧 checkpoint 不作为新模型训练起点。

---

## 2. Phase A 内部里程碑

为避免与未来 Research Phase B/C 混淆，当前 landing-only 使用 `A0–A8` 标号。

| Stage | 目标 | 是否训练 RL |
|---|---|---|
| A0 | 文档与 schema 锁定 | 否 |
| A1 | 三环、生命周期、任务、电量、LZ 恢复 | 否 |
| A2 | 真实四入口、external queue、admission | 否 |
| A3 | merge scheduler + canonical trajectories + safety | 否 |
| A4 | rule-only 全链路稳定 | 否 |
| A5 | 固定三环低层共享策略 | 低层 |
| A6 | 三环动态高层 | 高层 |
| A7 | 完整 HRL 与消融 | 两个已训练策略 |
| A8 | 可选缩尺/硬件接口 | 不作为本轮前置 |

---

## 3. A0 — 文档与 schema 锁定

### 目标

让代码修改前只有一个目标版本。

### 任务

1. 用本次六份文档覆盖/替代旧主文档；
2. 删除或移出工作文档目录的旧阶段性说明；
3. 配置 schema 改为 `phaseA_landing_v1`；
4. 定义：
   - `airspace_schema_version = three_ring_v1`
   - `action_schema_version = low_intent_v1_3d`
   - `high_action_schema_version = high_airspace_v1_10d`
   - `observation_schema_version = phaseA_obs_v1`
5. 禁止加载旧 checkpoint。

### 完成标准

主路径中不存在：

```text
"exactly two rings"
slot_residual
battery_endurance_seconds  # 作为主能量状态
direct free-flight low-level action
```

历史 git commit 中可以存在，但当前 working tree 的 canonical code/doc 不依赖它们。

---

## 4. A1 — Domain model rebuild

### 4.1 三环 DynamicAirspace

修改：

```text
terminal_marl/core/config.py
terminal_marl/core/dynamic_airspace.py
terminal_marl/core/ring_capacity.py
```

任务：

- 2 ring → 3 ring；
- 8D → 10D high-level action；
- `R_O/R_M/R_I`、`H_O/H_M/H_I`；
- shared speed-profile bias；
- three headways；
- capacity from actual radius；
- occupancy-aware shrink projection；
- geometry transition 支持三环；
- affected-interface calculation。

### 4.2 UAVState / lifecycle

修改：

```text
terminal_marl/core/uav.py
terminal_marl/core/lifecycle.py
```

任务：新 FlightPhase、request/mission/priority/energy/LZ fields、lifecycle timestamps、outcome fields，移除通过多个 bool 推断 phase 的逻辑。

### 4.3 Energy

新增/重写：

```text
terminal_marl/core/energy.py
```

实现 phase-based energy、SOC thresholds、energy-to-complete estimator。

### 4.4 Priority

新增：

```text
terminal_marl/core/priority.py
```

实现 `requirements.md` 统一公式。

### 4.5 LandingZoneManager

新增：

```text
terminal_marl/core/landing_zone.py
```

实现 6 个固定地面服务点：

```text
FREE → RESERVED → OCCUPIED → FREE
```

四类任务服务时间：30/120/10/60 s。

### A1 验收

必须通过：

```text
three-ring ordering
capacity monotonic
mission/service sampling
priority monotonic with wait
priority increases when SOC low
LZ mutual exclusion
energy decreases by phase
lifecycle legal transitions
```

此阶段不训练。

---

## 5. A2 — Arrival, gate and admission semantics

### 5.1 ArrivalProcess

保留已有 request-generation 思路，但改成 canonical API。

必须：

- demand request 不被 active cap censor；
- deterministic/poisson/bursty/fixed schedule/finite cohort；
- request 创建时固定 mission/base priority/energy/gate。

### 5.2 Four real entry corridors

修改当前“同一 bearing + lateral offset”方式。4 个 gate 需要：

```text
独立 angle
独立 spawn waypoint
独立 gate waypoint
独立 admission spacing
```

### 5.3 ExternalQueue

新增：

```text
terminal_marl/core/admission.py
```

request 无法 admission 时继续等待。

### 5.4 Overflow

实现：

```text
WAIT
DIVERTED_LOW_SOC
DIVERTED_CAPACITY
```

不能停止产生 demand。

### A2 验收

- 同 seed request stream 完全复现；
- 不同 policy 得到相同 request stream；
- 4 gate 物理位置不同；
- overload 时 queue 增长；
- max_active_uavs 不改变 requested count；
- external waiting 消耗能量；
- 低 SOC 可 diversion。

---

## 6. A3 — Merge, trajectory and safety unification

这是本轮最重要的工程阶段之一。

### 6.1 MergeScheduler

新增：

```text
terminal_marl/core/merge_scheduler.py
```

替代：

```text
fixed release angle offset
slot_residual
timing-as-threshold-bias
```

实现：

```text
baseline merge time
+ bounded Δt
→ predicted target-ring gap
→ merge reservation
→ merge angle
```

### 6.2 Canonical trajectory

重写/统一：

```text
terminal_marl/core/trajectory_generator.py
terminal_marl/core/vtac.py
```

最终只保留一套 trajectory plan。

优先方案：

```text
ring hold: moving circular reference
ring transfer: tangent cubic Bézier + quintic z
final: spline / waypoint composite to assigned LZ
landing: continuous touchdown
```

`vtac.py` 若保留，应降级为 executor/guidance adapter，不再独立生成另一条 path。

### 6.3 SafetyFilter

重写：

```text
terminal_marl/core/safety_filter.py
```

两级检查：

```text
intent/resource feasibility
trajectory feasibility/separation
```

移除以 arbitrary heading/pitch offset 为核心的安全规则。

### 6.4 Reconfiguration protocol

实现：

- ring holders follow moving reference；
- committed transfer continues；
- affected interface freezes new release；
- target geometry projection considers committed traffic。

### A3 验收

必须有 trajectory unit tests：

```text
source tangency
target tangency
continuous altitude
kinematic limits
same trajectory checked/executed
merge gap safe
reservation conflict prevented
reconfiguration interface freeze
```

---

## 7. A4 — Rule-only full landing pipeline

在任何 RL 前完成。

### 7.1 RuleLowLevelPolicy

输出 canonical 3D action：

```text
release
speed residual
merge-time residual
```

所有后端与 RL 共用。

### 7.2 单机 gate

场景：

```text
1 request
→ gate
→ outer
→ middle
→ inner
→ final
→ LZ
→ service complete
```

要求：

```text
outcome = COMPLETED
zero unresolved
no teleport
correct energy
correct LZ occupation duration
```

### 7.3 多机 nominal gate

建议 `N=10,20`、medium demand、fixed airspace。

要求：

```text
0 collision
0 LZ double booking
0 illegal phase transition
0 unresolved in finite cohort
```

### 7.4 High/bursty stress

不要求全部立即 landing，但要求：

```text
queue grows instead of demand disappearing
system remains numerically stable
eventual resolution in finite-cohort cooldown
```

### A4 完成定义

如果 rule-only 还不能稳定完成全部可行 landing：**不进入 PPO**。

---

## 8. A5 — Fixed-airspace low-level learning

### 8.1 环境

```text
airspace geometry fixed
headways fixed
high-level policy disabled
```

### 8.2 Actor

所有 UAV shared actor：

```text
o_i → π_L → [release, Δv, Δt]
```

### 8.3 Critic

第一版可先采用：

```text
agent encoder + masked mean/max pooling
```

稳定后升级 attention。

### 8.4 Curriculum

- **A5.1**：warm-start ring scenes，小规模 5–10 UAV，学习 release/speed；
- **A5.2**：flow-from-empty，10–20 UAV；
- **A5.3**：variable N + medium/high/bursty。

训练期间逐步减少 curriculum shortcut，正式 eval 只用真实 arrival flow。

### 8.5 Baseline comparison

对比：

```text
fixed_rule
fixed_low_rl
```

如果 learned low-level 在安全前提下不能优于或至少不劣于 rule baseline，不进入完整 HRL。

### A5 gate

至少验证：

- action 3D 全部真正影响语义；
- policy 可处理不同 N；
- no NaN；
- no hidden direct flight action；
- safety intervention 频率不会随训练恶化；
- delay/throughput 至少一个主要指标有稳定改善。

---

## 9. A6 — Dynamic three-ring high-level learning

### 9.1 先用 rule low-level

第一轮高层训练：

```text
π_H + stable RuleLowLevelPolicy
```

目的：单独验证 dynamic airspace 是否有价值。

### 9.2 Dynamic-rule baseline

实现 `dynamic_rule`，根据 external queue、three-ring occupancy、LZ utilization、final queue 启发式调整 geometry/headway。

### 9.3 High-level PPO

- action = 10D；
- decision interval = 180 s；
- structural action = event-gated；
- geometry minimum dwell = 600 s。

### 9.4 再换训练好的 low-level

高层 PPO 第二轮：

```text
π_H + frozen trained π_L
```

禁止直接让两个策略同步漂移。

### A6 gate

必须比较：

```text
fixed_rule
dynamic_rule
learned high-level + rule
learned high-level + frozen low-level
```

并证明高层 action 确实改变 actual geometry、derived capacity、headway 和 terminal performance。

---

## 10. A7 — Full hierarchical method and paper experiments

### 10.1 Proposed

```text
dynamic_hrl
```

= trained high-level + trained shared low-level + scheduler + trajectory + safety。

### 10.2 不默认 joint training

首先直接组合已训练策略做 evaluation。只有组合性能显著下降时，进入 alternating fine-tuning：

```text
freeze L → update H
freeze H → update L
```

不使用 simultaneous from-scratch PPO。

### 10.3 Formal baselines

```text
fixed_rule
fixed_low_rl
dynamic_rule
dynamic_hrl
```

### 10.4 Ablations

优先级：

1. two-ring；
2. event gate vs periodic geometry；
3. homogeneous service；
4. no dynamic priority；
5. fixed headway；
6. no safety filter。

### 10.5 Formal protocols

- Finite cohort：N=20/40/60 起步，multi-seed；
- Continuous flow：low/medium/high/bursty/overload；
- 所有 policy 使用 common random numbers。

---

## 11. A8 — 可选 scaled/hardware validation

A8 不是本轮算法修改前置。当 simulation core 稳定后再做：

```text
simulation geometry scaling
→ same TrajectoryPlan abstraction
→ hardware adapter
→ Crazyflie/Crazyswarm2
```

硬件只验证 trajectory execution、safety-filter compatibility、small-N coordination，不通过 2–3 架小型无人机实验直接证明高密度 capacity。

---

## 12. 文件级修改顺序

### 第一批：core semantics

```text
terminal_marl/core/config.py
terminal_marl/core/uav.py
terminal_marl/core/lifecycle.py
terminal_marl/core/dynamic_airspace.py
terminal_marl/core/ring_capacity.py
terminal_marl/core/energy.py
terminal_marl/core/priority.py
terminal_marl/core/landing_zone.py
```

### 第二批：traffic flow

```text
terminal_marl/core/arrival_process.py
terminal_marl/core/admission.py
terminal_marl/core/scenario.py
terminal_marl/envs/parallel_env.py
```

### 第三批：intent/trajectory/safety

```text
terminal_marl/core/merge_scheduler.py
terminal_marl/core/trajectory_generator.py
terminal_marl/core/vtac.py
terminal_marl/core/safety_filter.py
terminal_marl/core/observations.py
```

### 第四批：policy/training

```text
terminal_marl/policies/rule_low_level.py
terminal_marl/policies/dynamic_rule.py
terminal_marl/algo/*
terminal_marl/envs/high_level_wrapper.py
terminal_marl/training/*
terminal_marl/configs/*
```

### 第五批：experiments

```text
terminal_marl/experiments/*
terminal_marl/metrics/*
tests/*
```

---

## 13. 需要删除或重写的测试思想

不能为了让旧测试继续通过而保留旧模型。以下测试应替换：

```text
assert exactly 2 rings
assert high-level action shape == 8
assert low-level intention shape == 4
assert slot residual modifies speed
assert endurance seconds linearly decreases
assert all arrivals share one inbound bearing
```

测试应服务于新规范，而不是锁死历史行为。

---

## 14. 训练前 readiness checklist

```text
[ ] 3 rings end-to-end
[ ] 4 physical gates
[ ] request != admission
[ ] external queue exists
[ ] mission type restored
[ ] base/dynamic priority restored
[ ] phase-dependent SOC restored
[ ] 6 LZ reservation/service restored
[ ] rule pipeline completes finite cohort
[ ] one canonical transfer trajectory
[ ] same trajectory checked and executed
[ ] low-level action exactly 3D
[ ] high-level action exactly 10D
[ ] variable-N observation/policy works
[ ] geometry transition safe
[ ] outcome accounting closes
[ ] old checkpoints rejected
```

任何一项未完成，都不开始正式 PPO。

---

## 15. Codex 提交节奏建议

```text
Commit 1: schema/enums/config
Commit 2: 3-ring dynamic airspace
Commit 3: UAV energy/priority/lifecycle
Commit 4: LZ manager
Commit 5: arrival/admission/four gates
Commit 6: merge scheduler
Commit 7: canonical trajectory + safety
Commit 8: rule full pipeline
Commit 9: observation/action v1
Commit 10: low-level training
Commit 11: high-level wrapper/training
Commit 12: experiment matrix
```

每个 commit 必须带对应 pytest。

---

## 16. Phase B / C 后续路线

### Research Phase B

在 Phase A 核心完成后单独设计 departure：

```text
LZ
→ departure reservation
→ climb-out corridor
→ departure fix
→ exit
```

### Research Phase C

再做 arrival/departure integrated resource management。

**当前任何 A0–A8 任务都不要求实现 departure。**
