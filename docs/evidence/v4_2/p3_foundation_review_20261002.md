# P3 基础实现复核与修复审核方案（2026-10-02）

本轮核查P3动作—预约—轨迹—安全—动力学—能量—观测—reward/PPO链路。重大生产修改待用户审核；没有新训练或checkpoint变化。

本文件是主审结论的证据草稿，保留既有 artifact 的 cohort 边界与失败记录。`PPO_ALLOWED=false`；本轮没有启动完整训练、held-out、new fit 或 checkpoint 写入。

## 本轮状态（2026-10-02）

用户已批准首阶段；核心实现、88→90D migration 与预检已完成。全面复核新增轨迹运动学一致性 blocker，修复方案待本次用户审核，详见 [轨迹运动学一致性修复审核方案](trajectory_kinematic_consistency_repair_review_20261002.md)。完整八臂业务矩阵尚未运行；`PPO_ALLOWED=false`。

## 当前终态锚点

最新 `CURRENT_STATUS` 顶部记录的 fresh V4 six-scene screen 为 `FAIL`，与后续 normal-inner C1–C4 属于不同 cohort。四个旧 trained gate-value states 均为 `C=0`；AP=`132N/12E`、BP=`134N/10E`，两者 `F/D/U=0`。AP 相对 BP 的三项成本差分别为 `+2.9843324913833236%`、`+0.5281622267627961%`、`+0.7351975729166904%`。same-prefix 扣除 rejection 后 raw return 为 AP `2967.1651355245745`、BP `2987.9587313050633`。来源：`docs/CURRENT_STATUS.md`（历史块：`LATEST FRESH VALIDATION STATUS（2026-10-01）`）、`docs/evidence/v4_2/p3_followup_results_20260930.md:22`。

normal-inner FULL 是另一组四状态描述性证据：四个状态均 `C=1`，四个 continuation seed 的 `D` 符号均混合；`wait` 与 `linear_residual` 未提供，`local_m` 未汇总为 global gain。来源：`docs/CURRENT_STATUS.md`（历史块：`LATEST P3 STATUS — 2026-10-01 normal-inner gate-value FULL terminal`）、`docs/evidence/v4_2/p3_followup_results_20260930.md:7-22`。

最后一次有限 gate fit 为 `0 accepted / 1 rejected`；candidate KL=`0.03208379613660517`，target KL=`0.011689337256158998`，cap=`0.001`，随后 rollback，weights unchanged。四状态目标与回滚结果属于可复现的 local artifact，不能当作一般化修复。来源：`docs/CURRENT_STATUS.md`（历史块：`LATEST P3 STATUS — 2026-10-01 normal-inner gate-value FULL terminal`）、`docs/evidence/v4_2/p3_followup_results_20260930.md:22`。

## Newly-run foundation contract probe

`terminal_marl/scripts/diagnose_p3_foundation_contracts_20261002.py` 此前已完成 help/import/compile；v3 本次正常 CLI/full/readback 在 Windows 11、CPython `3.12.14`、NumPy `2.3.5` 上状态为 `CONTRACTS_READY`、exit `0`、`technical_errors=[]`。该 runner 使用 zero-arrival environment 加 deterministic synthetic ring holders；没有 policy、rollout、Teacher、tape、held-out、training 或 optimizer。最新本地 artifact 为 `.tmp/p3_foundation_review_20261002_contracts_v3_local.json`；v1/v2 artifacts 保留为历史验证记录，证据类型 `newly_run`。

该 isolated fixture 的 effective probe configuration 为：phase-motion energy、emergency trigger SOC `.40`、minimum landing SOC `.10`、SOC reserve `.25`、speed residual authority `±2 m/s`、merge-time residual `±6 s`、final capacity `3`、nearest-compatible-LZ、downstream-priority、88D/90D schema。它是合同探针配置，不是 frozen formal P3 配置或 closed-loop performance result。

| final geometry | rows | unique trajectory hash | canonical departure | duration | requested slots | initial errors | dynamics / SafetyFilter |
|---|---:|---|---:|---:|---|---|---|
| `brake_bezier_v1` | 25 | 1 | `10.666666667` | `47.370792661` | `0,3,6,9,12` | position/velocity `0/0` for all rows | all `[]` / all single-UAV rows approved |
| `continuous_inward_v1` | 25 | 1 | `0` | `60.620754272` | `0,3,6,9,12` | position/velocity `0/0` for all rows | all `[]` / all single-UAV rows approved |

这 50 个 isolated rows 显示 final proposal 的 speed/time residual 在当前 generator fixture 中没有形成不同的 physical trajectory/hash；它确认了 isolated generator 的 proposal physical dimensions 被忽略，不能直接外推为 closed-loop performance 或 P3 root cause。该结果补充了代码级观察，不等于完整执行链已解决。

## 既有结果（均为 `preexisting_artifact`）

| 证据 | 既有结果 | 边界 | 来源 |
|---|---|---|---|
| 已知 failure coverage replay | 原 formal failure scenes `49N/20E/3F`；coverage replay `60N/12E/0F` | 仅 local known-scene coverage，不是 fresh qualification | `preexisting_artifact` — `docs/evidence/v4_2/p3_holding_diagnosis_repair_20260925.md:213-214` |
| task-neutral baseline | source `72N/11E/1F`；candidate `72N/12E/0F`，normal 数未增加 | development paired result；不构成全局 promotion | `preexisting_artifact` — `docs/evidence/v4_2/p3_task_neutral_baseline_result_20260928.md:17-49` |
| critic predictive repair | critic 指标改善，但 fresh `fa0e` physical candidate `80/16/0→73/23/0` | mean wait `129.233681726712→139.866715305331 s`；`fa0e` 被拒，保留 `3fe7` | `preexisting_artifact` — `docs/evidence/v4_2/p3_task_neutral_critic_repair_result_20260928.md:31-44,76-96` |
| larger batch + full MC | `gamma=lambda=1`、`full_episode_mc`；`80/16/0→74/22/0` | mean/p95/max wait `124.5896/281.3078/361.1838→132.2342/333.6133/374.5735 s` | `preexisting_artifact` — `docs/evidence/v4_2/p3_task_neutral_larger_batch_protocol_20260929.md:10-13,96-99` |
| forced-WAIT 24 pairs | 两臂均 `491N/83E/2F`；物理 wait/elapsed/energy/outcome delta 均 `0`；raw reward `+65.6` | reward 差异全部来自 rejection penalty；扣除 penalty 后两臂 reward 相同 | `preexisting_artifact` — `docs/evidence/v4_2/p3_followup_results_20260930.md:724-815` |
| prefix development 与 fresh screen | development prefix `74/22→88/8`；最新 fresh AP=`132N/12E`、BP=`134N/10E`，screen `FAIL` | development executor comparison 不能替代 fresh six-scene screen 或 qualification | `preexisting_artifact` — `docs/CURRENT_STATUS.md`（历史块：`LATEST FRESH VALIDATION STATUS（2026-10-01）`）、`docs/evidence/v4_2/p3_followup_results_20260930.md:22` |

这些结果支持保留旧结果与 cohort 边界：静态或 known-scene qualification 不能直接证明 dense-flow 下的 energy/resource feasibility；单一 repair 结果也没有形成 P3 全局 PASS。

## 实现核查记录（主审当前假设；已完成部分与未闭合边界分开记录）

### 动作、预约与最终进近执行

- `TrajectoryGenerator.final_approach` 接收 `reservation`，但 continuous inward 分支调用 `_continuous_inward_landing_points` 时只传 `now` 与 `scheduled_delay`；helper 以 `departure_start=now` 起步，正的 `scheduled_delay` 才通过 ring hold 形成 departure。默认 brake-bezier 分支以 `now + max(scheduled_delay, braking_delay)` 作为 `start`。对应代码：`terminal_marl/core/trajectory_generator.py:409-425,432-451,498-560`。
- 最终进近的参数化目标水平速度在 continuous helper 中固定为 `10.0`，默认分支为 `min(10.0, max_speed)`；`speed_residual_mps` 被写入 metadata，但当前 final-approach geometry 没有把它作为目标速度改变轨迹。对应代码：`terminal_marl/core/trajectory_generator.py:475-480,507-512,552-558`。
- `MergeReservation.merge_time` 由 scheduler 保存；`canonical_plan_slot_time` 对 final approach 读取 `scheduled_departure` metadata，并在环境侧用 `_retime_reservation_to_plan` 做预约对齐。isolated contract probe 已显示 25 个不同 requested slots 在两种 final geometry 下各自只生成一个 trajectory hash；full closed-loop reservation-to-execution 一致性仍未 qualification。对应代码：`terminal_marl/core/merge_scheduler.py:17-20,30-60`、`terminal_marl/envs/parallel_env.py:560-575,885-949,1020-1064,1285-1339`；probe：`.tmp/p3_foundation_review_20261002_contracts_v3_local.json`。
- 首轮修复草案要求显式 opt-in 的 final speed/timing execution 版本：有限加速度 prefix、canonical reservation exact match、clamp trace；legacy path 保留用于 replay。该草案未实施，等待用户审核。

首阶段草案的验收条件为：可行且未 clamp 的 final speed proposal 必须改变执行速度或时长；timing proposal 必须改变真实 departure；受动力学约束不能实现时必须显式 clamp 或 reject，并记录 requested/accepted/actual；initial state 连续且全轨迹动力学与安全检查通过；reservation 必须与被检查和执行的 canonical plan 一致。物理约束允许多个 proposal 合法地 clamp 为同一结果，不要求 25 组全部不同。

### 观测 schema 与任务语义

- 88D V2.1/V2.3 中 mission 是四维 token；在最近 task-neutral 路线的 `task_priority_enabled=false` 配置下，`_mission_observation` 直接返回 `[1,0,0,0]`，因此该路线的 `STANDARD` token 不携带 raw service duration 或 pad compatibility。priority-enabled 的 88D raw mission token 可以区分 mission；此处不能泛化为所有 88D 都不携带任务信息，也不能回溯解释最早 priority-aware P3 的全部退化。对应代码：`terminal_marl/core/observations.py:39-51,78-98,137-155`。
- raw mission 的 service duration 为 STANDARD/CARGO/EMERGENCY/SURVEILLANCE=`30/120/10/60 s`。对应代码：`terminal_marl/core/uav.py:13-26`。
- V2.4/90D 已实现并在 88D prefix 后追加 `self_service_duration_norm` 与 `center_earliest_compatible_lz_release_wait_norm`，且构造函数使用 raw mission 与 compatible pad ETA；正式 P3 仍使用 88D，90D 尚未 qualification。对应代码：`terminal_marl/core/observations.py:100-107,346-378`；历史边界：`docs/evidence/v4_2/research_review_repairs_20260930.md:27,102,195`。
- newly-run cross-mission probe 确认在同一 physical fixture、最近 task-neutral `task_priority_enabled=false` 路线上，STANDARD 与 CARGO/EMERGENCY/SURVEILLANCE 的 88D vectors byte-equal，三组 `max_abs_delta=0`；对应 90D appended service values 为 `0.25/1.0/0.0833333358/0.5`，分别对应 `30/120/10/60 s`，且 88D prefix byte-preserved。priority-enabled 88D raw mission token 的可区分性不被这项 alias 结果否定。该 alias 仅确认同一 fixture 的 observation alias，不构成完整 90D Markov sufficiency proof，也不测定其对历史 P3 退化的因果份额。来源：`.tmp/p3_foundation_review_20261002_contracts_v3_local.json`，证据类型 `newly_run`。
- 首轮修复草案要求在新 curriculum 中显式采用既有 90D service context，并对 schema/weight migration 写明绑定关系；不覆盖旧 88D replay。
- 90D 迁移验收还要求追加字段确实区分 service、保留 88D prefix；旧 88D checkpoint 不得按 90D 直接加载，迁移必须有显式 schema 标识，并检查 zero-new-input 时初始行为一致。

### 安全 tube、动力学与轨迹证书

- 默认 action authority 为 `max_speed_residual_mps=2.0`，shield horizon 为 `30 s`；rule-flow opt-in 才把 residual authority 设为 `0.0`。对应代码：`terminal_marl/core/config.py:202-221,457-462`。
- SafetyFilter 对未提交 holder 使用 bounded future reachable tube；candidate 与 tube 的距离小于 `safe_distance` 时返回 `predicted_holder_reachable_tube`，同 ring handoff 后还会检查 post-plan tube。对应代码：`terminal_marl/core/safety_filter.py:405-417,439-448`。历史 paired probe 记录 `±2 m/s` filter `0/16`、nominal zero-residual filter `4/16`；这是 preexisting partial audit，不是 distance proof，也不能据此删除合法 ±2 authority：`docs/evidence/v4_2/landing_flow_repair_20260924.md:125-134`。
- 当前 profile 源码约束为：水平速度上限 `25 m/s`；纵向加/减速上限与横向加速度上限均为 `1.5 m/s²`；爬升/下降速度上限为 `4/6 m/s`；垂向加速度上限 `1 m/s²`。执行 physics substep 为 `0.2 s`；validator 检查轨迹节点，以及由位移/时间推导的分段速度和速度差。上述 50 条 generated plans 的起点位置/速度误差均为 `0`，但不构成任意 external plan 初态或 jerk 检查证明。
- 因此 holder future reachable tube 比名义 WAIT 路径更保守；shield 仍拥有真实 ±2 authority。这两项需要分开测量。isolated contract probe 已完成，但不提供 closed-loop distance proof；不能把 holder tube 拒绝标为 SafetyFilter bug，也不能移除 ±2 自由度。
- 轨迹自身有动力学校验与 `dynamic_unreachable_merge_time` fail-closed；TrajectoryPlan 生成 trajectory hash，commit/advance 路径核对被 safety-check 的 plan/hash。对应代码：`terminal_marl/core/trajectory_generator.py:48-85,581-585`、`terminal_marl/core/safety_filter.py:326-336`、`terminal_marl/envs/parallel_env.py:534-558,1175-1189`。
- canonical plan/hash 机制已实现；isolated foundation-contract probe 已取得上述结果。初始 server targeted suite 的 4 个失败已由 stale fixture 复核并修正，修复后 targeted 与 9-file suite 均 exit `0`；当前仍不把 isolated contract 结果写成整体 P3 PASS。

### 能量、应急接管与候选检查边界

- `compute_candidate_energy_budget` 明确计算 candidate route、reported pad release ETA 下的 fallback route 与 reserve；其 docstring 明确这是 executable energy lower bound，不是 SafetyFilter certification 或 future-feasibility guarantee。对应代码：`terminal_marl/core/energy_reachability.py:211-239`。
- normal RELEASE candidate 经过 `_candidate_energy_guard_reason` 后再进 SafetyFilter；WAIT path 没有对应的 candidate energy-guard candidate enumeration。对应代码：`terminal_marl/envs/parallel_env.py:619-640,885-949,1285-1345`。
- `SimConfig` 默认 `emergency_trigger_soc=.35`，但当前 V5-derived P3 physical protocol 覆盖为 `.40`（`terminal_marl/configs/low_v4_2/p3_controlled_emergency_formal_v5.yaml:85`），本次 isolated probe 也使用 `.40`。`_try_emergency_landing` 在每个 tick 的 actor release 处理前执行，因此 WAIT 不会关闭 SOC-triggered automatic takeover；缺少的是 WAIT 对 future pad/route deadline 的同等 candidate assessment。对应代码：`terminal_marl/core/config.py:163-176`、`terminal_marl/envs/parallel_env.py:590-603,1578-1583`。
- newly-run 12-row solo energy-only probe 覆盖 O/M/I、theta `0/pi`、nominal/zero speeds；required SOC 范围为 `0.1433464983–0.2060807011`，均低于 `.40` trigger。该结果只说明这 12 个 solo fixtures 不支持“固定 trigger 在 solo 情况下必然过晚”，不覆盖 congestion 或 safety。来源：`.tmp/p3_foundation_review_20261002_contracts_v3_local.json`，证据类型 `newly_run`。
- 首轮草案保留当前 LZ/route budget 与 fail-closed safety；不降低 SOC 或 separation threshold。应急 energy guard 的 lower bound 不应被表述为 future safety certificate。

### 静态 source-chain 与 reward/PPO 核查

- 最新 task-neutral dense24 的 `_build_neutral_inputs()` builder-only runtime verification 已在服务器 `/home/user/envs/terminal_env/bin/python` 执行，exit `0`；无 environment reset/step、模型、训练或 optimizer。实际 `neutral[dense24]` 确认的十个字段为：`energy_reachability_guard=true`、`emergency_priority_guard=true`、trigger SOC `.40`、`continuous_inward_v1`、`final_queue_capacity=3`、`nearest_available`、`downstream_priority`、`v4_2_rejected_release_hold_mode=canonical_wait`、`student_decision_mask_mode=legacy`、`low_level_soc_urgency_time_weight=2`；`observation_schema_version=phaseA_low_obs_v2_1_retry_state`（88D）。对应源码/协议来源：`docs/evidence/v4_2/p3_followup_results_20260930.md:490`、`docs/evidence/v4_2/p3_task_neutral_baseline_protocol_20260928.md:83-88`、`terminal_marl/configs/low_v4_2/p3_current_mdp_baseline_v1.yaml:48,61-63`。本次 builder 直接确认两个 guard 已打开；不能把“guard 未打开”写作该验证的事实。
- V4.2 action gate/proposal decoding 与 WAIT zeroing 的实现链已核对；K=3 使用 sampled requested proposal 加 fallback 流程，PPO 保存 requested likelihood，不以执行 fallback 替换该 likelihood。scalar team reward、actor-mask 与 complete-cycle aggregation 均保留；历史 `gamma=lambda=1`/`full_episode_mc` 证据排除了已记录的折扣参数误用，但不能据此确立 team-credit、entropy 或 clipping 根因。
- 当前 low-level scalar reward 未完整表示 queue/service business cost；raw return 与业务 cost 的错位可能贡献退化，但其量化尚未完成。来源：`docs/evidence/v4_2/p3_task_neutral_episodic_learning_protocol_20260929.md:9-13,51-57`、`docs/evidence/v4_2/p3_task_neutral_larger_batch_protocol_20260929.md:10-13`。

## 主审结论与审阅边界

当前证据把最新路线的动作/预约/轨迹执行合同和 task-neutral 88D 观测表达列为必须先修复并 qualification 的两个 confirmed implementation issues，再讨论进一步超参数调节；它们对历史 P3 退化的因果份额尚未测定。priority-enabled 88D mission 区分能力不被 task-neutral alias 结果否定。已有结果支持多个可能贡献项共同存在，但没有形成唯一的 P3 global cause。历史静态 qualification、known-scene coverage、forced-WAIT physical tie 或单次 critic fit 均不能单独证明 dense-flow 的 energy/resource feasibility。

## 首轮修复草案（DRAFT，等待用户审核）

1. 新增显式 opt-in 的 final speed/timing execution 版本：有限加速度 prefix、canonical reservation exact match、clamp/trace；legacy 版本继续用于 replay。
2. 新 curriculum 使用已实现的 90D service-context schema，并显式完成 schema/weight migration；保留 `pre_action_forced_wait` 的 opt-in 语义，不改变旧 cohort。
3. 独立监测 actor energy deadline、当前 LZ/route budget 与 fail-closed safety；不降低 SOC、separation 或 SafetyFilter 阈值。
4. 先量化 safety tube 的 conservative rejection；保留现有 ±2 shield authority，不先删除该自由度。
5. 验证链固定为：isolated contracts → fixed development inputs 上的 zero-training rule/source paired complete-cycle → fresh independent development → 仅在准入后进行 bounded learning。每个改动分别报告动作、预约、轨迹、安全、动力学、能量、观测与 reward/PPO 链路影响。

本草案不授权 full training、held-out、new fit、checkpoint 写入或 default promotion。任何重大生产修改须先经用户审核。

## Newly-run server test status

在服务器 `/home/user/envs/terminal_env/bin/python`（Python `3.12.3`、torch `2.13+cu130`、NumPy `2.5.1`、pytest `9.1.1`）上，Syncthing 后 core/test SHA 一致；定向 suite 为 `67 passed, 4 failed`，耗时 `1.32 s`，exit `1`。日志：`/home/user/training_outputs/terminal/p3_foundation_review_20261002/core_tests_python312.log`。初次默认 Python 的 collection 日志保留为 `core_tests.log`，原因是缺少 torch。

4 个失败及直接观察如下，尚不归因于 production 根因：

| 测试 | 直接观察 |
|---|---|
| `test_route_aware_guard_rejects_negative_budget_without_resource_side_effects` | 期望 `emergency_energy_guard`，实际 `None`。 |
| `test_negative_transfer_budget_redirects_to_certified_emergency_route` | `_runtime_shortlist_release` 在 `merge_scheduler.py:264` 对 `object` 读取 `.target`，抛出 `AttributeError`。 |
| `test_low_soc_ring_transfer_guard_rejects_below_landing_plus_margin_reserve` | 期望 `emergency_energy_guard`，实际 `approved`。 |
| `test_final_guard_keeps_touchdown_floor` | 期望 `emergency_energy_guard`，实际 `None`。 |

初始 4 个失败均来自旧 fixture/测试契约：主审复核 numeric fixture 后确认旧 SOC `.37` 时 remaining `555` 大于 route required `440.8952`（slack `114.1048`），原 negative 断言不成立；新 SOC `.25` 时 remaining `375` 小于 required `400.8745`（slack `-25.8745`），而立即应急 required SOC `.2060807` 仍可行。priority 旧 SOC `.259` 时 transfer projected `.174738>.11`、final projected `.228085>.10`，均应允许。

修复仅修改 `tests/test_energy_reachability_guard.py` 与 `tests/test_emergency_priority_guard.py`：priority margin fixture 按当前 `min_landing_soc + .9*margin_seconds*power_reference/capacity` 取 SOC `.109`；negative route fixture 按 `min_landing_soc+.15` 取 `.25`；mock reservation 改为真实 `scheduler.propose`。生产规则未改。修复后 targeted 两文件为 `16 passed`、`.30 s`；原 9 文件 suite 为 `71 passed`、`1.32 s`、exit `0`，SHA parity 已核验。日志为 `/home/user/training_outputs/terminal/p3_foundation_review_20261002/core_tests_repaired_targeted.log` 与 `/home/user/training_outputs/terminal/p3_foundation_review_20261002/core_tests_repaired.log`；numeric artifact `energy_fixture_probe.json` SHA=`31220c3bae6cd8d438e42dbe52f5f8d7059b5ffeb84950a78adb8099ab748d33`。

初始失败表仍保留为历史 artifact；上述 stale fixture 结论已取代此前的待复核状态。这组测试修复不构成 production rule change、P3 PASS 或全局因果结论。

## 首阶段无训练闭环验收边界（DRAFT，等待用户审核）

首阶段只采用 opt-in final speed/timing execution、既有 90D 显式 schema/权重迁移，以及小样本无训练 complete-cycle 对照。对照必须使用相同 action authority、safety、energy 与 arrival 场景，避免旧 rule-flow `±0` 与 RL `±2` 的不公平比较。energy deadline/resource 与 safety tube 先做量化诊断，再决定是否提出 production 修改；当前不把任何一项写成已确认根因，也不承诺立即改变规则。

新 curriculum 的低并发随机 SOC、dense 固定 SOC 异构 service、组合难度应分开控制以隔离贡献。本阶段不授权 training 或 held-out。

## 2026-10-02 轨迹运动学一致性复核结果

用户已批准首阶段接口与迁移工作；本轮新增的统一轨迹/动力学修复仍待用户审核，尚未实施。原“首轮修复草案（DRAFT，等待用户审核）”段保留为历史记录，其等待同一首阶段审批的表述已由本段及[轨迹运动学一致性修复审核方案](trajectory_kinematic_consistency_repair_review_20261002.md)取代。新 mode 仍为 opt-in，未 promotion；P3 尚未解决；`PPO_ALLOWED=false`。

四个固定 fixture 的 position 采样差分结果如下。`duration` 来源于 `end_time-start_time`；只有最大实际水平/垂直加速度来自按 `physics_dt=.2` 的 position 采样差分，末段短 interval 使用实际 `dt`。旧 `validate_kinematics=[]` 并不等于下表的执行曲线满足 profile 上限。

| mode | slot | duration | max actual horizontal accel | max actual vertical accel |
|---|---:|---:|---:|---:|
| `legacy` | `0` | `65.4561997898406` | `28.0235052431024` | `6.51781141164899` |
| `legacy` | `30` | `65.4561997898406` | `28.0235052431024` | `6.51781141164899` |
| `speed_timing_v1` | `0` | `70.1783200237381` | `22.3042676892213` | `5.85048370097321` |
| `speed_timing_v1` | `30` | `86.5132317122911` | `23.3341765134967` | `6.01783603351056` |

现有 profile 的水平与垂直加速度上限分别为 `1.5` 与 `1.0`。legacy 首段中点 position 差分速度与 reported velocity 差的 norm 为 `3.887695799217562`；新模式曲线段的最大 chord-vavg residual 为 `1.0742706573` 与 `0.7061464605`。这些是统一轨迹采样复核的直接数值，尚未测定其对历史 P3 学习退化的因果份额。

首阶段证据锚点：core commit `aff605e4f74a560853ef111d6f8c7c4900d12c1c`，14 文件 `116 tests PASS`；runner/protocol commit `848b2d8570f8ae9b680b98273a45e4e34fd93bec`，`10 tests PASS`；`preflight.json` SHA256 `751baf2e44e8ccc050ed99804ae07a2f16bedb09b84e9bc143c5a859abb61e85`，状态 `CHECK_ONLY_OK`、耗时 `2.065 s`，只验证 metadata/schema/config 与八臂登记；`smoke.json` SHA256 `e0c9d23984c158f29ba8dc36151f118fd88e5c7aba2ca93fc3e09da345c5edac`，状态 `SMOKE_OK`、三 source 各 `2 ticks`、耗时 `3.701 s`。仅 `smoke.json` 记录 88/88/90D energy coverage valid、repaired88/90 requested action tape 最大 diff `0`、metrics/hash 一致与 frozen flags false；smoke 只验证接口与输出，不代表性能提升或可执行性 qualification。foundation repair protocol 绑定的 source checkpoint SHA256 为 `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`。

诊断脚本及 4 项 diagnostic tests 已 PASS（commit `d67597aec27b7f703d952759647b9a974e0f7aa1`；server/local 指定文件 SHA 一致），日志 SHA256 为 `ff9f6c40d73d36f40389303ff64d302f2ef8f04bb289ec72ce3cb10439f18a78`。compact CLI 状态为 `KINEMATIC_CONSISTENCY_FAIL`、`technical_errors=0`、`finding_count=20`；`finding_count` 是重复检查项数量，不是 20 个独立缺陷。compact artifact `.tmp/p3_trajectory_kinematic_consistency_compact_20261002.json` 的 SHA256 为 `FC4FC02E98600174EBD9EC1DA18F725EEAC3C57D31478F1CC2AFA754E95111CA`。相关入口：[基础修复协议](p3_foundation_repair_protocol_20261002.md)、[运动学一致性审核方案](trajectory_kinematic_consistency_repair_review_20261002.md)、[运动学诊断脚本](../../../terminal_marl/scripts/diagnose_trajectory_kinematic_consistency_20261002.py)。完整八臂零训练对照尚未运行；本轮未启动 training、held-out、optimizer step、checkpoint 写入或 default promotion。

## 本轮交付文件清单（16 distinct files）

- **core**：`terminal_marl/core/config.py`、`terminal_marl/core/trajectory_generator.py`（`aff605e`）。
- **runtime**：`terminal_marl/algo/service_context_runtime_v4_2.py`（`aff605e`）。
- **scripts**：`terminal_marl/scripts/evaluate_p3_foundation_repair_20261002.py`（`848b2d8`）、`terminal_marl/scripts/diagnose_trajectory_kinematic_consistency_20261002.py`（`d67597a`）。
- **tests**：`tests/test_final_approach_proposal_execution.py`、`tests/test_phase_a_landing.py`、`tests/test_service_context_runtime_v4_2.py`（`aff605e`）；`tests/test_p3_foundation_repair_evaluator.py`（`848b2d8`）；`tests/test_trajectory_kinematic_consistency_diagnostic.py`（`d67597a`）。
- **docs**：`docs/design_spec.md`、`docs/technical_spec.md`（`aff605e`）；`docs/evidence/v4_2/p3_foundation_repair_protocol_20261002.md`（`848b2d8`）；本轮文档 `docs/evidence/v4_2/p3_foundation_review_20261002.md`、`docs/evidence/v4_2/trajectory_kinematic_consistency_repair_review_20261002.md`、`docs/CURRENT_STATUS.md`。
