# 执行计划

> **当前工程目标**：只完成 Phase A — Landing-only。
> Phase B（departure-only）和 Phase C（arrival + departure）只保留在研究路线中，不进入本轮 Codex 修改。
> 本执行计划按“先修正环境语义，再训练低层，再训练高层”的依赖顺序推进。

## 0.1 2026-09-28 当前课程与执行顺序

本轮主线为 dynamic terminal airspace、multi-UAV landing flow 和 safety-filtered executable
intents。低层先采用 `task_priority_enabled=false` 的中性任务课程：正常降落、吞吐、完整周期
等待和尾部 starvation 监测；保留随机到达、SOC 差异、能量、安全、环/路径可执行性和资源预约。
ArrivalProcess 仍消耗历史 base 抽样的 RNG，但新课程只存储有效 `base_priority=1`；raw mission
保留为物理服务/兼容性 metadata，policy mission one-hot 固定 `STANDARD`，Observation V2.1
和 88D/schema 保持不变。`30/120/10/60 s` service time 分布继续保留以维持负荷。

能量应急接管始终保留在执行/安全层，与后续独立 emergency mission 鲁棒性场景分开。等待尾部
只用于 starvation 监测，不增加每个子组必须无退化的硬门槛；新课程对照不得与旧课程报表直接
跨任务比较。旧 source `03c0`、candidate 和 `candidate_retained` 结果只作为初始化/对照历史，
不宣称稳定收益或已解决训练退化。

当前执行顺序冻结为：

```text
最小新课程实现/单测
→ 同课程 source/candidate 配对 baseline
→ 有界 PPO repair pilot（独立 admission 后）
→ independent validation
→ static qualification
→ A6 dynamic airspace
```

本轮协议见 [task-neutral baseline protocol](evidence/v4_2/p3_task_neutral_baseline_protocol_20260928.md)。
协议已完成 check-only_v2，smoke 正在执行，完整评估待运行；状态以 [CURRENT_STATUS](CURRENT_STATUS.md)
为准。
formal/heldout 与 `PPO_ALLOWED` 不因文档修订自动放行。

## 历史记录：2026-09-24 至 2026-09-25

以下条目保留当时的实验事实和失败边界，不定义当前课程或下一步授权。

## 历史 0.1 2026-09-24 集成与顺序

`fixed_rule_dense30_coordinated` 是 opt-in 的 rule-flow coordinator，包含 downstream/deadline
放行、canonical trajectory 和完整动力学/SafetyFilter 检查；最近五场 `150` UAV、
`143 normal / 7 emergency / 0 failed-energy` 仍是规则控制器的历史 `.25` floor 证据，不是 learned
policy qualification。`validate_plan` 的批量等价校验和 viewer 的 position-only/cache 路径已有
受限性能记录，但 prefix/offscreen 测量不作为整体吞吐或 runtime FPS 结论。viewer 的拾取、标签和
30 Hz 中间帧回归已通过专项测试。

后续能量口径采用 `eh216s_research_v2`、`energy_capacity=1500`、`min_landing_soc=.10`；
`soc_reserve=.25` 继续作为独立 queue/admission reserve，其它阈值保持不变。已有 `.25` 实验、
checkpoint 和 seed 消费记录保留为历史事实。当前按 staged authorization 推进：

1. 已完成既有 V4.2 learned-policy diagnostic replay 到当前 30-UAV / v2 profile 的
   checkpoint/config/schema/hash preflight，以及首个 bounded diagnostic `842101`；协议与结果见
   [protocol](evidence/v4_2/trained_dense30_v2_diagnostic_protocol_20260924.md) 和
   [result](evidence/v4_2/trained_dense30_v2_diagnostic_result_20260924.md)。保持无
   Teacher/optimizer，global loader/core 不变，不自动授予 qualification。
2. `842101` 的 read-only trace aggregation 已完成：4 个 failed-energy UAV 的 threshold 前
   RELEASE 为 `3/3/3/9` 且全部 rejected，WAIT 为 `411/305/493/238`；inner RELEASE 的主要观察
   `primary_blocker=LZ_RESOURCE=593/615`（`510 emergency_lz_priority + 83 lz_unavailable`），approved
   `8`。这些是 controller 与 execution feasibility 诊断的既有观察，不写入新的 PASS 或结果结论。
3. fresh random-SOC P3 slot-repair comparison 已完成，详见
   [protocol](evidence/v4_2/random_soc_p3_slot_repair_20260924.md) 与
   [result](evidence/v4_2/random_soc_p3_slot_repair_result_20260925.md)。seed `845101` 的
   late-OFF forced、late-ON forced、late-ON trained 分别为 `23/1/0`、`19/5/0`、`18/6/0`；
   三 arm 均 terminal、exit `0`、unresolved `0`，命名安全/失败计数均为零，weights unchanged、
   Teacher/PPO off。`845000` smoke 有效；server `29`、local diagnostic `17`、slot `12`、
   Teacher `12` 分组报告，不合并为单一测试总数。
4. late-ON 不采纳，default late-slot OFF 保持；forced arms 不作为 trained policy 成果。外置
   state-only labeler 已由 `768b191` 实现，gate_adapter-only 监督训练器已由 `b9f1077` 实现，
   对应 data protocol 为 `56b4c3`；proposal head/encoder 冻结。服务器 smoke `846000` exit `0`，
   已接受 `18` 条 actor labels（`7 WAIT + 11 RELEASE`），全部 finite 88D 且 schema 映射正确、
   weights unchanged；服务器指定 `5` 个模块测试为 `39 PASS`，本地 `44` 个测试属于另一组，
   不合并为同一测试总数。完整 `7` 回合 collection 已完成，结果见
   [collection/fit result](evidence/v4_2/random_soc_gate_repair_v1_result_20260925.md)。启动 wrapper 曾因
   reset 前预建 output 被拒绝，归档为 `launch_failure_train_flow6_846101`；修正独立 `_control` 日志后，
   首个实际 `846101` 回合已运行，不称为重复 seed 实验。授权的 gate-adapter-only fit 已完成 `35/35`，
   fit artifact SHA256 为 `e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`，
   train/dev rows 为 `1529/812`；dev RELEASE/WAIT recall 为 `.6131687243→.7860082305` /
   `.8119507909→.7768014060`，macro-F1 `.7092638611→.7562677855`，dev NLL
   `1.265022278→1.031340837`，train NLL `1.259755254→.755264521`。仅 adapter 改变，
   non-adapter/proposal logits bitwise unchanged；strict load passed、无 optimizer state、无 dev gradient/
   early stop/PPO。wrapper 最后有 CRLF shell exit error，但 Python 已完成且 control 保存 Python exit `0`，
   未重跑。`846301/846302/846303` paired rollout 已完成且三者 exit `0`；三场合计 `42` requests，
   source 为 `36 normal + 6 emergency + 0 failed_energy`，candidate 为 `41 normal + 1 emergency +
   0 failed_energy`。六个 arm 的 replay、safety、Teacher off、no updates、weights unchanged 均为
   true；dense24 唯一 false arm gate 为 `no_failed_outcomes=false`，不称 P3 通过。整体 lifecycle
   aggregate 与自动生成的 `development_pair_aggregate.json` 一致：source priority-weighted wait
   `150.6971849500`、low-SOC wait `53.7159174128`（count `12`），candidate 为 `80.1562018040`、
   `21.4881016766`（count `12`）；selection gates 全部 true。exact replay 与 `32` 个 shadow checks
   均为 exact `795`-step、state unchanged、无 safety error、approved `0`，其中 `safety_wait=16`、
   `final_capacity=10`、`safety_reject=6`。这只记录抽查状态未获批，不宣称所有 WAIT 错误或 SOC
   排序有效。`847101/847102` arbitration pair 已完成：baseline 与 soc-priority 各为
   `41 normal + 7 emergency + 0 failed_energy + 0 unresolved`，两 pair 均降低 low-SOC wait 但提高
   priority-weighted wait；因此不采纳 SOC 排序，默认 `downstream_priority` 保持。详见
   [arbitration result](evidence/v4_2/random_soc_arbitration_repair_v1_result_20260925.md)。本轮
   repair 来源为 `714d141f...`，历史 P3 canonical initial 为独立的 `4b8be66c...`，不静默替换
   基线。下一步转入[canonical reference protocol](evidence/v4_2/random_soc_canonical_reference_v1_20260925.md)。
   canonical strict loader 为 `29a4dba` 且 `32` 个定向测试通过，evaluator 的 `27` 个定向测试通过，
   只读服务器 checkpoint/hash/profile-load preflight 通过；暂停时 `848000/848101/848102` 尚未消费，
   heldout 未消费。暂停点已于 2026-09-25 解除；恢复后 smoke/full pair 均已完成，四个 arm 均
   exit `0`、full-resolved、zero-energy 且安全硬计数为零，canonical aggregate 为 `31 normal +
   17 emergency`、repair aggregate 为 `40 normal + 8 emergency`。repair 两组 priority-weighted
   wait 与 low-SOC wait 均更优，V5 entry/pilot 冻结从 repair candidate E04 开始：SHA256
   `e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`，路径见[受控应急协议](evidence/v4_2/p3_controlled_emergency_admission_v5_20260925.md)。该开发比较不自动授权 PPO；
   该开发比较在当时不自动授权 PPO；当时的 `P3_QUALIFIED=false`、`PPO_ENTRY_ALLOWED=false`、
   `FORMAL_P3_ALLOWED=false`、`P4_ALLOWED=false` 状态由后续 V5 entry/pilot admission 更新，
   旧 formal/heldout 保持 untouched。
5. P3 random-SOC 范围沿用现有 P3 V4：initial SOC `.55–.90` uniform、priority `1/3/5`、
    `flow6`/`flow12`/`dense24`。dense30 pressure 仅作诊断，不能单独构成 qualification；后续资格
    证据仍需 train/dev/heldout 独立性。旧 P3 formal 与 canonical comparison 继续使用其冻结的
    all-complete/`no_failed_outcomes` 历史 gate；新的 V5 PPO entry/学习路线允许安全认证且能量未耗尽
    的 `emergency_landed` 作为带代价结果，但 `failed_energy`、unsafe、entry/development/evaluation
    `unresolved`、replay/likelihood 和 finite/restore/frozen-proposal 仍是硬失败；pilot 训练中的
    自然 horizon 截断单独报告。预留 `829001–829006` heldout 保持 untouched，
    不改作本轮开发集。详见[受控应急 P3 V5 协议](evidence/v4_2/p3_controlled_emergency_admission_v5_20260925.md)。

### 历史 0.2 2026-09-25 P3 V5 formal admission 状态

`849000` smoke、`849101/849102/849103` stochastic no-update entry、`849201–849206` 六个
registered PPO arm 和 `849301/849302/849303` paired development 均已完成。六个 pilot arm
共 `296` accepted minibatches、`0` rejected，update/restore/safety/energy/integrity gates
全部通过；paired dense24 的 emergency `4→3`、low-SOC wait `-44.036s`、priority-weighted
wait `+2.763s`，作为 report-only tradeoff 保留，不构成 qualification 或自动晋级。

当前 formal YAML 已冻结为 `READY_FOR_FORMAL_TRAINING`，SHA256 为
`e3f88e52a327c64621226631c1575cd48ffec4d93b98d505c561fe55b33b22b1`；其 admission 状态为
`PPO_ENTRY_ALLOWED=true`、`FORMAL_P3_ALLOWED=true`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。
Formal runner 与定向测试实现为 `be50cde`。`P3_FORMAL_ADMISSION_DECISION_V5` 已绑定 E04
source SHA、pilot manifest/protocol SHA、实际 config snapshots 和 exact optimizer boundary；
服务器 admission-bound `--check-only` 已通过，exit `0`、status `CHECK_ONLY_OK`，且 run0 目录
未创建。正式 campaign 尚未启动；通过后按注册顺序首次消费 `850001..850240`、`851001..851240`、
`852001..852240`，不自动重跑或替换失败 seed。

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
| A1 | 三环、生命周期、服务元数据、电量、LZ 恢复 | 否 |
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
   - `action_schema_version = low_intent_v3_factorized_discrete`
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
- request 创建时固定 raw mission/effective base/energy/gate；中性课程仍消耗 base 抽样 RNG，但只存储 effective `base_priority=1`。

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
88D Observation V2.1
→ shared encoder
→ WAIT/RELEASE gate
→ conditional 25-way RELEASE proposal
→ [release, Δv, Δt] execution intent
```

### 8.3 Critic

第一版可先采用：

```text
agent encoder + masked mean/max pooling
```

稳定后升级 attention。

### 8.4 Curriculum

- **A5.0 / S0**：action、observation、checkpoint、replay 与 safety schema qualification；
- **A5.1 / I0**：fresh structured expert dataset qualification；
- **A5.2 / I1–I3**：hierarchical BC、Safe DAgger、student-only imitation qualification；
- **A5.3 / P1+**：通过 admission 后才进行 student-controlled PPO、variable N 与 traffic curriculum；
- **A5.4 / L5**：static geometry generalization，完成后冻结低层供 A6 使用。

每个阶段必须先通过独立 gate。训练期间逐步减少 curriculum shortcut，正式 eval 只用真实
arrival flow；详细顺序与数据纪律见 `reference/low_level_policy.md`。

### 8.5 当前 task-neutral baseline sequence

新主课程固定 `task_priority_enabled=false`。最小实现和单测通过后，先以相同新课程配置运行
source/candidate 配对 baseline，再由主审决定有界 PPO repair pilot 的目标与预算。ArrivalProcess
仍消耗历史 base 抽样的 RNG，但只存储有效 `base_priority=1`；raw mission 保留为物理服务/
兼容性 metadata，policy mission one-hot 固定 `STANDARD`，88D schema 不变。service time
`30/120/10/60 s` 保留以维持负荷。能量应急接管始终保留，和后续独立 emergency mission
鲁棒性场景分开。等待尾部仅作 starvation 监测，不增加每个子组无退化硬 gate。

新 baseline 报告 normal/emergency/failed/unresolved/full-cycle mean/p95/max wait、low-SOC
wait、reward 和安全/ledger 技术核对；新 reward 与旧 reward 不直接比较。完整课程评估待运行，
状态以 [CURRENT_STATUS](CURRENT_STATUS.md) 为准，
不能写成 candidate retained、稳定收益或训练退化已解决。formal/heldout 与 `PPO_ALLOWED`
仍须独立授权。

### 8.5 历史 baseline comparison（不定义当前课程）

#### 2026-09-17：PPO 场景课程审查与阶段边界

#### 2026-09-25：受控应急语义下的 P3 PPO 准入边界

此前 V3/Q3 的全完成与 `no_failed_outcomes` 判据继续作为历史 checkpoint 选择和
qualification 证据，不能被新规则追溯改写。恢复后的 P3 V5 将 `emergency_landed` 定义为
安全认证、能量未耗尽且完成应急着陆的降级结果：它进入性能和学习反馈，并单独报告事件数、
priority、low-SOC 与全请求等待；`failed_energy`、未认证/安全失败和 entry/development/evaluation
`unresolved` 仍是硬失败。pilot 训练中的自然 horizon 截断单独报告，不自动写成普遍安全
失败。V5 不用事后比例门槛制造历史 PASS。

后续独立 qualification 可把 dense `<=10%`、sparse `<=5%` 作为 provisional 研究目标；
这些目标尚未验证，也不是当前 PPO admission threshold。P3 训练仍保留 priority `1/3/5`、
SOC `.55–.90`、mission/资源混合和 `downstream_priority`，避免把 SOC 变成唯一决策因素。
学生 rollout 已在 summary 中记录 `reward_component_totals` 与 `reward_total`，并验证组件和
一致；使用实际 `time/progress/release/touchdown/failure/teacher/shield/urgency/emergency`
字段，排除 per-UAV `total`。`urgency` 仍是 priority 与 SOC 的合并项，不虚构独立分解；实现
提交 `4d34672`，对应定向测试 `9 passed`。

当前只有下表 P1 配置已经实现并获得 88D 准入。P2–P5 是后续执行顺序，
不是可直接切换的已授权配置；每阶段先冻结实际参数、seed、预算并作小样本验证。
旧 `low_v4/`、87D 课程文件不作为当前训练入口。

| 阶段 | 数量与到达流 | 任务/资源差异 | 阶段目的与限制 |
|---|---|---|---|
| P1 初始 PPO（已准入） | 12 episodes 内循环 warm3、flow3、flow6，各4次；自然流10秒首到达、60秒间隔，900秒上限 | SOC=.8；常规 priority=1，emergency仍提升至5；保留原四种 mission/service分布、6个LZ、final容量1 | 先验证真实 PPO 更新下低层意图与闭环稳定，不宣称容量或收敛；warm-start是训练辅助 |
| P2 自然流与 variable-N | 以flow3/6为锚点，先扩到总请求8/12；逐步降低warm-start占比；先保持到达强度，再单独引入间隔扰动 | 先保留P1属性，避免同时改变负荷与任务属性 | 900秒不能照搬：按末次到达与rule基线清空时间预设观察/排空窗口；正式评估只用自然流 |
| P2R 密集资源桥接 | warm12（6/4/2）+ 12 架按15秒继续到达；保留旧自然流回归，后续训练应混合 warm 与 flow-from-empty，而非只用 warm-start | 6个LZ；`nearest_available`；downstream-first；final容量3；每条轨迹仍逐条认证 | 暴露跨环同一步竞争、LZ分配和并发final；报告吞吐、touchdown间隔、并发final、LZ距离后悔、反事实安全机会与全部hard-safety计数；不得改写旧P2结果 |
| P3 任务与资源异质性 | 保持已验证负荷和入口分布 | 保留 priority 1/3/5、SOC `.55–.90`、mission 混合、长 service、reservation/retry 压力；允许安全认证且能量未耗尽的 `emergency_landed` 作为带代价结果 | 目标是学习能耗、优先级、资源占用的权衡，避免 SOC 单因素排序；`failed_energy`、unsafe、entry/development/evaluation `unresolved` 仍为硬失败。pilot 训练截断单独报告；高密度应急比例只作后续独立资格报告，不追溯改写历史 gate |
| P4 终端区交通压力 | finite cohort总请求20/40/60；continuous low/medium/high/bursty，overload单列；根据本机型/布局rule基线容量选择强度 | 保持真实任务/资源分布，逐步加入入口偏斜和突发流 | 总请求数不等于同时在空数；报告active峰值、队列、请求完成/未完成/转降与延误。过载不要求队列稳定或全部成功，不能套用P1全完成gate |
| P5 静态几何泛化 | 在已经验证的流量与N范围内，每episode冻结一种合法布局 | 分阶段改变合法半径/高度/速度与headway组合；仍执行完整可执行性检查 | 先静态泛化并冻结低层，再进入A6高层动态空域；不同时引入高负荷、几何变化和新任务分布 |

数量判断：P1的N=3/6是**每episode总请求数**；warm3同时初始化3架。
`max_active_uavs=60`只是上限，并不表示当前训练有60架并发。N3–N6作为PPO起点合理，
但与论文最终20/40/60请求及随机持续到达场景之间尚有课程跨度，不能直接认定完成终端区训练。
60秒间隔对应到达时段60请求/小时，不等于实测落地吞吐；多入口和随机mission已存在，
时间到达目前仍是确定性的。六个LZ在当前低流量下未必形成足够排队压力。

P2R 是旧 P2 与 P3/P4 之间的资源语义桥接，不是把“环占满”作为奖励。warm-start 用于主动覆盖
三环同时有机和瞬时竞争，flow-from-empty 用于检查真实到达链；二者必须混合，heldout 至少包含
未见 seed 的 dense24 压力场景。只有当 individually-certified-but-not-requested 比例显示稳定的
gate 漏放行时，才新增 fresh support/训练；资源空闲但轨迹不可认证不得作为错误标签。

2026-09-21 P3 执行更新：priority/SOC urgency-shaped mixed PPO 的 V3 formal 12/12 gate PASS，
但 final heldout 在 dense24 安全不变量失败；暴露种子诊断发现只有最后一个检查点失败。
事后选取 checkpoint 5 的全新 Q3 尽管 18/18 安全与完整性通过，第二个 dense24 仅完成
13/24，对照 initial 24/24，故 P3 仍 FAIL。详情见
[P3 V3/Q3 证据](evidence/v4_2/p3_v3_early_stop_q3_20260921.md)。以后 P3 的
formal/heldout 必须同时报告完整请求分母的完成率与已完成者等待；不得用后者代替前者。
出现长时无 touchdown、空闲 LZ、shield/shortlist 激增时先审查状态与可认证动作，
再预注册新训练修复。P4 流量压力不得因 V3 formal PASS 而提前启动。

所有阶段从第一步起保留：EH216-S性能约束、80m安全间隔、NMAC/collision阈值、
轨迹可执行性、SafetyFilter/shield、切环headway、final/LZ容量、预约与retry、能耗和外部admission。
课程只增加负荷/属性/布局覆盖，不能先关闭安全、能耗、资源约束，再在后期突然打开。
真实PPO只用student requested gate/proposal计算ratio，Teacher始终禁用。

当前主要余量问题：I3 deterministic flow6用时867/900秒。首个正式block沿用原协议，
不因启动训练而临时放宽截止时间；若失败保留结果并定位。后续扩N时另行设定合理的
末次到达后排空窗口。检查优化前后的完成、延误、shield与rule对比后再升级课程，
首个12-episode block不是完整PPO收敛训练。

对比：

```text
fixed_rule
fixed_low_rl
```

在相同 task-neutral 课程、相同负荷和安全前提下，learned low-level 若不能优于或至少不劣于
rule baseline，不进入完整 HRL。当前 source/candidate 配对 baseline 只完成下一步核对，不能
替代这项研究比较；旧跨任务报表不转成当前 gate。

### A5 gate

至少验证：

- V4.2 gate/proposal 语义、单请求执行边界与 ActionTape replay 通过；
- WAIT/RELEASE 和有效 RELEASE proposal 均有独立支持，不发生保守性 collapse；
- policy 可处理不同 N；
- no NaN；
- no hidden direct flight action；
- safety intervention 频率不会随训练恶化；
- full-cycle ledger、完整请求等待 mean/p95/max 和 tail starvation 监测可复核；在同课程主指标上
  必须观察稳定的 delay/throughput 改善后才进入完整 HRL；不增加每个子组无退化硬 gate，也不把
  当前未运行结果写成稳定收益已解决。

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
[ ] low-level action is V4.2 gate + conditional proposal
[ ] execution decoder remains one canonical 3-value intention
[ ] observation is versioned 88D V2.1
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
