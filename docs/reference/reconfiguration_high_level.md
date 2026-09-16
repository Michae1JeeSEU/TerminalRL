# Dynamic Reconfiguration 与 High-Level Integration 规格说明

> **文档定位**
>
> 本文件独立定义 terminal geometry 动态重构、ReconfigurationCoordinator、受影响 UAV 的控制权切换、重构成本建模，以及后续 high-level policy 与重构机制的接口。
>
> **本文件不负责定义 low-level policy。** 当前 V4.2 action、Observation V2.1、Teacher/Execution/Safety 分工、训练阶段与 qualification 见 [low_level_policy.md](low_level_policy.md)。high-level 是否获准启动以 [项目状态页](../CURRENT_STATUS.md) 为准。
>
> **建议版本号**：`RECONFIG-HIGHLEVEL-SPEC-v1`

---

# 1. 研究问题与设计目标

Dynamic reconfiguration 的目标不是让几何可以“免费、瞬时、无限频繁”变化，而是构造一个具有真实运行成本、同时又不会把 high-level action 彻底压制掉的系统级决策问题。

需要同时避免两个退化极端：

```text
A. reconfiguration cost too high
→ high-level never changes geometry

B. large geometry has no persistent cost
→ high-level expands once and never shrinks
```

因此正式设计应满足：

1. geometry change 有明确物理与运行成本；
2. 成本主要来自真实 transition dynamics，而不是巨大人为固定 penalty；
3. 受影响交通尽量局部化，不默认整个 terminal 全局停摆；
4. committed trajectory 得到保护；
5. transition 过程中职责边界明确；
6. high-level 无法高频 chatter；
7. low/high demand 下不同 geometry 具有不同相对价值；
8. high-level 在训练前就能看到一个非平凡、可学习的 reward landscape。

---

# 2. 与 Low-level 的职责边界

Low-level policy：

```text
given current legal/stable geometry
→ coordinate UAV intentions
```

High-level policy：

```text
given terminal/system state
→ decide whether / how geometry should change
```

ReconfigurationCoordinator：

```text
accepted geometry target
→ execute a legal transition
→ protect committed traffic
→ define affected set and temporary control authority
→ return system to stable geometry
```

关键原则：

> low-level 不负责学习“如何把一个 ring 从半径 A 拉到半径 B”。

如果某 UAV 在 transition 期间由 coordinator 接管，则其 low-level action authority 被 mask；transition 完成后 low-level 再重新接管正常交通协调。

---

# 3. Reconfiguration 期间的 UAV 分类

推荐把 UAV 分成三类。

## 3.1 Committed-transfer UAV

已经进入 ring-to-ring / final committed trajectory 的 UAV：

```text
continue protected committed trajectory
```

要求：
- 不因 geometry change 改写已提交轨迹；
- trajectory / velocity continuity 保持；
- reservation / target interface 一致；
- 不允许 reconfiguration 把它们重新吸附回 ring；
- 若目标 geometry 与 committed trajectory 冲突，应拒绝/延迟此次 reconfiguration，而不是破坏已提交轨迹。

---

## 3.2 Affected on-ring UAV

位于真正受此次 geometry change 影响的 ring / altitude band / interface 上的 UAV：

```text
low-level authority = masked
ReconfigurationController = authoritative
```

这些 UAV 随 reference ring / altitude profile 完成受约束的几何迁移。

要求：
- position continuous；
- velocity continuous；
- altitude continuous；
- radial / vertical speed 受限；
- acceleration 合法；
- ring progress remapping 连续；
- leader/follower order 不出现非法翻转；
- hard safety invariants 始终满足。

---

## 3.3 Unaffected UAV

完全不受此次 geometry transition 影响的 UAV：

> **默认继续正常 low-level operation。**

例如只改变 Outer ring 时，如果 Middle→Inner、Inner→Final、Final→LZ 流程与此次 transition 无关，则不应机械暂停这些 operation。

---

# 4. Affected-set Freeze vs Global Freeze

## 4.1 Global Freeze

```text
reconfiguration active
→ almost all normal intentions disabled
```

优点：
- 逻辑简单；
- 保守；
- 初期 debugging 容易。

缺点：
- operational cost 可能过高；
- queue / delay spike 强；
- cost 随 active-N 增长；
- 可能人为诱导 high-level “never reconfigure”。

因此建议将 Global Freeze 保留为：

> **baseline / ablation / fallback safety mode**

而不是默认正式机制。

---

## 4.2 Affected-set Freeze

定义：

`A(g_old, g_new) = affected UAV / ring / interface set`

只对真正受影响的部分进行 gating。

正式方法优先采用该机制，前提是 safety qualification 证明它与 Global Freeze 同样满足 hard-safety invariants。

至少比较：
- collision / NMAC / hard violation；
- transition duration；
- queue growth；
- throughput loss；
- energy overhead；
- affected UAV count；
- post-transition recovery time；
- high-level reconfiguration frequency；
- geometry diversity。

---

# 5. Reconfiguration 状态机

推荐明确使用：

```text
STABLE
  ↓
RECONFIGURING
  ↓
COOLDOWN
  ↓
STABLE
```

## 5.1 STABLE
- geometry 固定；
- low-level 正常运行；
- high-level 可在决策时刻提出 geometry action。

## 5.2 RECONFIGURING
- geometry target 锁定；
- 不接受新的 geometry target；
- affected UAV low-level authority 被 mask；
- committed trajectory 被保护；
- unaffected traffic 尽可能继续运行；
- coordinator 执行合法 geometry transition。

## 5.3 COOLDOWN
- target geometry 已达到；
- low-level authority 恢复；
- 暂时禁止再次 geometry change；
- 防止 expand/shrink 高频振荡。

---

# 6. Reconfiguration Duration 与物理 Rate Limit

不要对所有 geometry change 设同一个固定 transition duration。

建议：

`tau_reconf = f(|Δr_O|, |Δr_M|, |Δr_I|, |Δh_O|, |Δh_M|, |Δh_I|)`

并受：
- maximum radial speed；
- maximum vertical speed；
- maximum acceleration；
- separation constraint；
- committed-trajectory constraint；
- interface geometry constraint；
限制。

基本原则：

```text
small adjustment → short transition
large adjustment → long transition
```

这样 transition cost 来自真实变化幅度。

---

# 7. Low-level Action Mask 与数据语义

当 affected UAV 由 coordinator 接管：

```text
policy_action_valid = 0
actor_loss_mask = 0
```

系统仍可记录：
- state；
- trajectory；
- shield event；
- geometry progress；
- affected-set membership。

但：

> coordinator 运动不能进入 low-level actor PPO loss，不能被当成 `π_L` 自己采样的 behavior action。

transition 完成后：

```text
policy_action_valid = 1
```

low-level 正常恢复。

---

# 8. Reconfiguration 的成本来源

## 8.1 Endogenous operational cost（主成本）

优先让成本自然来自 simulator：
- affected operation temporary gating；
- queue growth；
- delay；
- extra flight distance；
- energy；
- temporary capacity loss；
- throughput loss；
- recovery time。

这些成本应自然进入 high-level return。

---

## 8.2 Explicit switching regularization（小成本）

可加入较小：

`C_switch = c0 * I_change + c_delta * ||Δg||`

用途仅是：

> 抑制无意义高频 chatter。

不应该依靠巨大 `I_change` penalty 人为让 reconfiguration 变得极其昂贵。

---

# 9. Anti-chattering 结构性约束

比巨大 penalty 更推荐：

## 9.1 Minimum dwell time
一次 reconfiguration 完成后必须维持至少 `T_dwell`。

## 9.2 Cooldown
显式保留 `COOLDOWN` 状态。

## 9.3 Action rate limit
限制单次：
- `|Δr|`；
- `|Δh|`；
- 可选 geometry vector norm。

## 9.4 High-level decision interval
High-level 不应每 1 s 都重新选 geometry。

采用明显慢于 low-level 的时间尺度，例如几十秒级或更长；具体数值由后续 economics calibration 确定。

---

# 10. “只扩不缩”与 Geometry State Cost

如果 high-level reward 只包含：

```text
operational reward - switching cost
```

而 expanded geometry 本身没有持续代价，则一旦已经支付 expand cost，high-level 可能没有 shrink 动机。

这不是 RL bug，而是 reward / dynamics 定义的自然结果。

因此必须检查 expanded geometry 是否存在 persistent operating cost，例如：
- circumference / path length；
- flight time；
- energy；
- holding cost；
- airspace footprint；
- noise / externality（若研究范围涉及）。

优先让 simulator 中已有 flight distance / energy / delay 自然体现该代价。

只有物理模型不足以体现研究关心的空域占用成本时，再加入显式：

`C_geometry_state(g_t)`

例如 normalized footprint term。

---

# 11. High-level Reward 推荐结构

概念上：

`R_H = R_operation - C_geometry_state - C_switch`

其中：

## 11.1 `R_operation`
可包含：
- throughput；
- delay；
- external queue；
- energy；
- safety；
- LZ/resource utilization；
- fairness / priority（如果研究目标包含）。

## 11.2 `C_geometry_state`
持续存在，体现当前 geometry 的运行/占用代价。

## 11.3 `C_switch`
只在 geometry 改变时存在，幅度较小，用于防 chatter。

理想 high-level 行为：

```text
low demand
→ compact / nominal more attractive

sustained high demand
→ expanded may become worthwhile

short peak
→ expand only if benefit exceeds transition cost

frequent switching
→ unattractive because of transition + cooldown
```

---

# 12. `N_affected` 与 Reconfiguration Context

同一个 geometry change 在低密度与高密度交通下成本不同。

建议计算：

`N_affected(g, g')`

以及：
- affected interfaces；
- expected transition duration；
- expected temporary blocked capacity。

这些量既用于 logging，也可作为 high-level observation 候选。

---

# 13. High-level Observation 的后续候选信息

等 low-level 与 reconfiguration semantics 稳定后再最终设计 high-level observation。

建议候选包括：
- current geometry；
- current occupancy / ring pressure；
- external queue；
- estimated arrival rate；
- arrival trend；
- gate imbalance；
- LZ utilization；
- current reconfiguration state；
- cooldown remaining；
- time since last change；
- estimated `N_affected` for candidate change；
- estimated transition duration；
- optional demand forecast。

这部分只作为未来 high-level observation 设计依据，不反向改变当前 low-level PPO 文档。

---

# 14. ReconfigurationCoordinator Qualification

在 high-level PPO 前必须先单独验证 coordinator。

使用 scripted geometry transitions：

```text
g0 → g1
g1 → g2
g2 → g0
```

重点检查：

## 14.1 Safety
- collision = 0；
- NMAC = 0；
- hard separation violation = 0；
- minimum actual separation；
- minimum predicted separation。

## 14.2 State continuity
- no teleport；
- position / velocity / altitude continuous；
- ring progress remapping correct；
- leader/follower order legal；
- predecessor/successor consistent；
- reservation preserved；
- committed trajectory preserved；
- LZ/final lifecycle preserved。

## 14.3 Operational impact
- transition duration；
- affected UAV count；
- blocked interfaces；
- queue growth；
- throughput loss；
- energy overhead；
- shield spike；
- post-transition recovery time。

---

# 15. Scripted Reconfiguration Test Matrix

至少覆盖：

## Geometry magnitude
- small；
- medium；
- near-boundary feasible。

## Geometry direction
- expand；
- contract；
- mixed ring changes；
- altitude-only；
- radius + altitude（若正式 action space 支持）。

## Traffic state
- low；
- medium；
- high；
- burst rising；
- burst falling；
- asymmetric gate demand。

## Active density
- sparse；
- nominal；
- dense。

## Resource state
- all LZ free；
- reduced LZ availability；
- busy final/LZ；
- priority / low-SOC traffic。

---

# 16. Global Freeze vs Affected-set Freeze Ablation

在正式 high-level 训练前至少做一次工程级 ablation。

比较同一 scripted demand + geometry transition：

```text
A. Global Freeze
B. Affected-set Freeze
```

使用 CRN / identical arrival schedule。

主要指标：
- safety；
- transition time；
- queue area-under-curve；
- throughput loss；
- energy；
- shield intervention；
- affected UAV count；
- recovery time。

若 B 在 safety 不下降的前提下显著降低 operational cost，则将 B 作为正式机制。

---

# 17. H0 / CH0：Reconfiguration Economics Calibration

在 high-level PPO 前增加一个**不训练 high-level policy**的校准阶段。

目的：

> 检查当前 transition mechanism + reward 定义是否真的形成非平凡的 geometry decision problem。

---

## 17.1 Demand profile

建议使用：

```text
low
→ ramp-up
→ sustained high
→ ramp-down
→ low
```

同时保留：
- short peak；
- long peak；
- repeated peak；
- asymmetric peak。

---

## 17.2 Scripted strategy baseline

比较：

```text
never reconfigure
always nominal
always compact
always expanded
expand early / shrink early
expand only after congestion
frequent switching
```

---

## 17.3 理想 economics landscape

### Low demand
compact / nominal 应更优或至少不劣于 expanded。

### Sustained high demand
expanded geometry 应存在可以覆盖 transition cost 的运行收益。

### Transient peak
应存在“是否值得临时 expand”的强度/持续时间阈值，而不是永远一个答案。

### Frequent switching
应明显劣于有节制的 switching。

---

## 17.4 失败诊断

若：

```text
never reconfigure
```

在几乎所有合理 demand profile 下都最好，则检查：
- Global Freeze 是否太重；
- transition duration 是否过长；
- explicit switching penalty 是否重复计算；
- high-level action step 是否太大；
- `N_affected` 是否过高。

若：

```text
always expanded
```

几乎永远最好，则检查：
- larger geometry 的 energy / path-length cost 是否真实进入 simulator；
- geometry-state cost 是否缺失；
- compact geometry 是否被不合理限制。

---

# 18. Geometry Operational Domain 与 High-level Action Domain

Low-level static geometry qualification 应先得到：

`G_operational`

High-level action domain 原则上满足：

`G_H ⊆ G_operational`

最好不要让 high-level 在训练初期选择 low-level 未 qualification 的 geometry。

对于 physical-feasible 但 low-level 运行质量很差的 geometry，可先排除出正式 action domain。

后续如果 alternating fine-tuning 证明低层能力扩大，再扩展 `G_H`。

---

# 19. High-level Training 前的 Gate

正式启动 high-level PPO 前，至少满足：

1. low-level `L5_geo_qualified` checkpoint 已冻结；
2. `G_operational` 已明确；
3. ReconfigurationCoordinator scripted qualification 通过；
4. action-mask / committed-trajectory semantics 通过测试；
5. Global Freeze vs Affected-set Freeze 已有选择依据；
6. H0/CH0 economics calibration 表明“never change”和“always large”都不是对所有 demand profile 的绝对统治解；
7. high-level action rate / dwell / cooldown 已冻结第一版；
8. reward version 已冻结第一版；
9. reconfiguration metrics logging 完整。

---

# 20. High-level Training 后的可选 Alternating Fine-tuning

High-level 训练后会诱导一个实际 geometry occupancy distribution：

`P_piH(g)`

若 low-level 在 high-level 常用 geometry / transition end-state 上存在系统性效率下降，可考虑：

```text
freeze H → fine-tune L
freeze L → update H
```

而不是从一开始 simultaneous joint training。

目的区分：

```text
L5 static geometry generalization = robustness
post-high-level fine-tuning = specialization
```

---

# 21. 推荐配置文件

```text
configs/
  reconfiguration/
    common.yaml
    coordinator_global_freeze.yaml
    coordinator_affected_set.yaml
    transition_small.yaml
    transition_medium.yaml
    transition_boundary.yaml
    repeated_transition_stress.yaml
    h0_economics_low_high_low.yaml
    h0_economics_short_peak.yaml
    h0_economics_long_peak.yaml

  high_level/
    common.yaml
    obs_v1.yaml
    reward_v1.yaml
    train_v1.yaml
```

Low-level configs 不应混入上述 high-level/reconfiguration 参数。

---

# 22. 标准日志

每次 reconfiguration 至少记录：

```text
transition_id
start_time
end_time
old_geometry
target_geometry
actual_geometry_trace
Delta_geometry
affected_uav_ids
N_affected
affected_interfaces
committed_uav_ids
blocked_operation_time
queue_before
queue_peak
queue_after
throughput_before
throughput_after
energy_overhead
shield_interventions
min_separation
recovery_time
cooldown_end
```

High-level 训练时另外记录：

```text
high_level_observation
high_level_action
accepted/rejected
rejection_reason
R_operation
C_geometry_state
C_switch
high_level_return
```

---

# 23. 必须避免的错误

1. 把 reconfiguration 期间 coordinator 运动当成 low-level PPO behavior action；
2. 默认全局冻结却不评估其人为成本；
3. 通过巨大 fixed penalty 让 high-level 几乎永远不改变；
4. 只有 switching cost，没有 persistent geometry-state cost；
5. high-level 每秒改变 geometry；
6. 允许 transition 过程中再次提交相反 target；
7. geometry action 超出 low-level `G_operational`；
8. committed trajectory 被 geometry change 改写；
9. 用同一组 test demand profile 反复调 reward，造成 qualification leakage；
10. economics landscape 未校准就直接跑长时间 high-level PPO。

---

# 24. 推荐实际执行顺序

```text
1. Freeze low-level L5_geo_qualified checkpoint
2. Define G_operational
3. Implement / verify coordinator state machine
4. Verify action-mask semantics
5. Verify committed-trajectory protection
6. Run scripted reconfiguration qualification
7. Compare Global Freeze vs Affected-set Freeze
8. Freeze first formal reconfiguration mechanism
9. Run H0/CH0 economics calibration
10. Tune only if landscape is degenerate
11. Freeze high-level action interval / dwell / cooldown / reward v1
12. Finalize high-level observation
13. Start high-level PPO
14. Run hierarchical qualification
15. Optional alternating fine-tuning
```

---

# 25. 一句话原则

> **Reconfiguration 应当有真实但不过度的人为成本：受影响交通由 coordinator 安全接管，未受影响交通尽可能继续运行；成本主要来自 transition duration、delay、energy 和 temporary capacity loss，并用 cooldown / dwell / rate limit 防止 chatter。只有在低需求时保持小/nominal、高需求时扩张、短峰值时存在权衡、频繁 switching 明显不划算时，high-level geometry decision 才是一个真正非平凡且值得学习的问题。**
