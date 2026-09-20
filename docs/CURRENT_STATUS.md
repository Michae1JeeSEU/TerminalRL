# 项目当前状态

> **状态日期**：2026-09-21
>
> **代码快照**：support/audit 历史提交 `18663b3`、`8ed0669`、`b7eaf85`、`c28efd5`、`dfa727f`；repair V3 `8a59312`；qualification V2 `de8047b`；effectiveness/feasibility `81d6c2b`、`13325ff`；exact-proposal chain `4aaea99`、`0e974ad`、`c38aedd`；set-valued chain `6048ccf`、`36aa2e5`、`a25e46f`、`4cdc29b`；proposal-conditioned audit `1ac334c`；Observation V2.2 schema/pilot chain `839dbb1`–`688016f`；runtime shortlist entry/formal/heldout/review `a8424cb`、`96e6329`、`8255e66`、`6129875`、`ccd9999`；P2R dense resource semantics `c1efba1`；dense safety/viewer performance `3157b5c`。Git 无 remote，未 push。
>
> **当前阶段**：Phase A / A5；original K3 candidate 的 P2R 密集降落流资源语义 V1 已完成实现与多种子诊断，未改写旧 P2 证据、未启动新 PPO。P3 priority/SOC pipeline V1 保持已预注册但未运行；P4 与 checkpoint promotion 仍阻塞
>
> 本文是滚动状态页。规范语义以 [README.md](README.md) 中列出的主规范为准。

## 一句话状态

低层 V4.2 / Observation V2.1（88D）已完成 R4 → I1 V2/Q2 → Safe DAgger → I3 → 初始 PPO 准入，并在 target-KL 保护下完成 P2 V4 与一次性确定性heldout。随后完成 V2→V5 on-policy support、learnability audit V2、anchored repair V3 与 fresh student-only Qualification V2，四者均 PASS；但 fresh paired effectiveness V1 仅关闭 baseline–rule 等待差距的 `46.01%`，低于预注册 `50%`，因此终态 **FAIL**。全部 safety/completion/replay/provenance hard gates 通过，失败只在效果阈值；**PPO、P3、课程升级和 checkpoint promotion 仍禁止。** 已消费的 Qualification V1/V2 与 effectiveness V1 seeds 均不得重跑、reset 或用于调参。

初始 N3–N6 的 `PPO_ALLOWED=true` 是对应冻结版本及协议的历史准入。P2 V1/V2 累计 28 次合格更新；V3 在入口 KL 超限后保持历史 FAIL。独立预注册的 V4 重新资格与入口通过，并补足剩余 8 回合；这不改写 V1–V3 的历史终态。

2026-09-19 V4正式训练、确定性heldout、WAIT/credit诊断、support successor chain、Qualification V2、effectiveness V1、feasibility attribution、exact-proposal 与 set-valued proposal target chain 均已终止，当前没有遗留服务器进程。历史 PASS/FAIL 均保留；effectiveness 终态见[配对效果评估](evidence/v4_2/ppo_p2_gate_effectiveness_v1_summary.md)，fresh gate/proposal 归因见[可行性归因](evidence/v4_2/ppo_p2_gate_feasibility_attribution_v1_summary.md)，proposal target 闭环见[条件提案目标链](evidence/v4_2/ppo_p2_proposal_target_chain_summary.md)。

2026-09-21 完成 P2R 密集降落流资源修复：历史协议继续使用 lowest-ID、插入顺序和 final
容量1；`p2_dense24` 显式改用 nearest-available、downstream-first 和 final容量3。冻结 original
K3 candidate 在 seeds `809006–809009` 的300秒配对诊断中，touchdowns `23→30`、service
completions `17→22`，每个 seed 最大并发 final `1→2`，新语义 LZ assignment regret 全为0，
全部 collision/NMAC/hard-separation/uncertified-execution 计数为0。seed `809010` 反事实扫描中，
108个内环资源机会时刻仅4个存在“已单机认证但策略未请求”的 RELEASE，故当前问题主要属于资源
语义而非 gate 全面失效；未据此启动新 PPO。证据见
[P2R dense landing-flow V1](evidence/v4_2/p2r_dense_landing_flow_v1_summary.md)。

同日完成 `p2_dense24` 仿真与可视化性能修复。无GUI基准证明主要瓶颈位于递归安全预测，
而非单纯PyVista绘制：20步 `env.step` 的cProfile累计时间由 `15.726 s` 降至 `8.052 s`
（`-48.8%`）。修改只消除递归投影对不可变轨迹数组的深拷贝、复用单步内完全相同的已提交
航迹，并为仅需位置的碰撞审查使用等价的轻量插值；安全阈值、预测时域、采样间隔与动作语义
均未改变。同检查点/seed的放行计数、最小间隔与全部安全遥测逐值一致。viewer现在只显示“当前
真实位置 + 尚未执行的计划点”，按UAV稳定着色，选中航迹加粗，并在plan/topology变化或结束时
正确重建/删除actor；P2 viewer默认 `frame_stride=2`、每条航迹最多12点。完整证据见
[P2 dense visualization/performance](evidence/v4_2/p2_dense_visualization_performance_20260921.md)。

2026-09-20 冻结 Observation V2.2 candidate：保持 V2.1 的 88D byte-identical prefix，追加 6 个
scheduler/reservation/同一步竞争上下文特征，形成 94D schema；runtime 仍为单一 intent + 单次
SafetyFilter validation，不引入 catalogue search。V1 fresh pilot 完成 6/6 episodes，但发现
`interface_committed_transfers_norm` 因 reservation stage 与 trajectory phase key 错配而恒为零，故 V1
只作诊断。修复后的 V2 使用全新 seeds；前三个 train episodes 完成，seed `800004` 在 1570 s
仅完成 10/12，末态两机均停留 `ring_middle/m2i`，安全计数为零且 exact replay PASS。V2 保持
immutable FAIL，未运行 dev/probe、未复用 rows。该失败证明 complete-cycle hard gate 混入了 frozen
behavior 的 reachability 要求，不能归因于尚未参与执行的 V2.2 feature。已预注册 V3：fresh
`801001–801006`，固定 1570 s/terminal stop，12/12 arrivals generated 仍为 hard gate，completion
仅报告；其用途只限 88D+proposal 与 94D+proposal 的配对表示诊断，不能授权 repair/qualification。

V3 formal 随后 6/6 episodes、72/72 requests 全部完成，安全与 exact replay hard gates 全 PASS；
train/dev 为 247/106 rows，manifest SHA256
`20cb9d6ac2377aa5bb3e20c87707405f4aa553eecadae0581ac64a0631cc8211`。94D 相对
88D+proposal shared MLP 的 ROC-AUC 三个 seed 分别提高 `.0255/.0340/.0444`，但 infeasible
recall 为 `+.0456/+.0743/-.0695`，macro-F1 为 `+.0161/+.0326/-.0010`，state Top-1 为
`-.0094/+.0094/+.0849`，不构成一致的 observation-only closure。新增特征中
active-reservation 与 committed-transfer 完全同值，future-reservation 恒零，predecessor 仅 6/353
rows 非零，进一步说明 6 个全局 scalar 未充分表达 proposal-specific joint feasibility。下一步仅允许
探索性比较固定 94D shared scorer 与 `94D→25 independent feasibility logits`；后者仍只做一次前向、
选择一个 proposal 并交 SafetyFilter 验证一次，不改变 runtime 边界。

25-logit exploratory audit 已在冻结 V3 artifacts 上完成，manifest SHA256
`9d07f8cb2de43c0edf1f5875aedd9966d72b83df911d31a21996ba5566a43d06`。相对 94D
conditioned shared scorer，infeasible recall 三次提高 `.0552/.0456/.1295`，macro-F1 三次提高
`.0142/.0052/.0361`；但 ROC-AUC 三次下降 `.0054/.0278/.0205`，state Top-1 为
`+.0283/-.0377/-.0566`，因此不允许 architecture freeze。为避免继续使用同一 dev 调结构，V4
一次性冻结 fresh `802001–802006` 与 probe seeds `802101–802103`，使用
`0.5 balanced BCE + 0.5 feasible-set NLL`；每个 probe 必须同时满足 feasible recall≥`.85`，且
infeasible recall、macro-F1、AUC、state Top-1 相对 fresh 94D conditioned baseline 均不下降。
只有 3/3 probes 全 gate PASS 才允许冻结架构；无论结果如何都不直接授权 policy repair。

V4 formal 已完成，6/6 episodes、72/72 requests、安全与 replay 全 PASS，train/dev 236/145 rows；
manifest SHA256 `1983d5dcbb4cc7ceea28bd02a47ccb7829c33d3e8514a2c53bc86f8f1416270d`。
Hybrid probe seed `802101` 全 gate PASS，但 `802102` 的四项相对指标失败，`802103` 的 AUC 与
state Top-1 失败，aggregate `all_three_probes_pass=false`，architecture freeze 保持 false。此结果
触发预注册停止条件：不再使用 V4 dev 调整 learned proposal head/loss。下一步改测 proposal source
边界：gate 仍取 anchored repair V3 checkpoint 的 deterministic argmax，RELEASE 时固定使用 catalogue
index 12（speed/merge residual 均为 0，minimum-intervention 首选），随后只调用一次 SafetyFilter；
该路径无 Teacher、无候选枚举、无训练更新。fresh paired evaluation 使用 flow12 seeds
`803001–803006`，与原 candidate 及 nominal rule 对比。

fixed-neutral V1 虽完成 18/18 records 且表面 decision gate PASS，但 action-tape 复核显示 candidate 与
fixed controller 的 requested gate/proposal 序列在六个 seed 上逐步完全相同；candidate 中仍出现
非 neutral proposals。这证明 `student_rollout_v4_2.student_request()` 直接调用
`policy.model.distributions()`，绕过了 V1 对 `act_details()` 的覆写。V1 manifest
`15bd56cf439be13796a292ee1a745a683e4c8c9632466667676f25c597b3cf35` 保留为
**INVALID_IMPLEMENTATION**，其 PASS 不得使用。V2 已把 proposal distribution 固定逻辑移到 model
boundary，保持原 gate logits/critic 不变，并新增 action-tape hard gates：全部 fixed RELEASE 必须为
proposal 12，且 candidate/fixed 请求序列必须存在实际差异。V2 使用全新 `804001–804006`，不重跑
或复用 V1 seeds/metrics。

Corrected fixed-neutral V2 随后使用 fresh `804001–804006` 完成 18/18 paired records；实现 hard
gates 证明六个 seed 的 candidate/fixed 请求序列均有差异，且 fixed controller 的全部 RELEASE 均为
proposal 12。三种 controller 均 72/72 requests 完成、安全/回放/likelihood 全 PASS、shield 为零。
fixed-neutral 相对 candidate 的 request-to-touchdown 改善 `4.3611 s`，但 weighted mean wait 从
`13.0303 s` 增至 `17.5845 s`（delta `+4.5542 s`），违反预注册 non-worse gate，故 aggregate
FAIL、architecture freeze=false。manifest SHA256
`df7ab01ef8a828189300d2e79e4f2073b2de8fe0937ae4f0b47faa010cbedac0`。
按停止规则，不再用本轮 dev 调固定 proposal index，也不再追加 learned head/loss。该结果把下一步
限定为显式改变 runtime boundary 或回退 rule-only；用户已在随后选择并授权前者，历史 K=1 停止
结论仍保留，不因新架构候选而改写。

用户随后显式授权 K=3 bounded shortlist certification。实现保持一次 policy forward：确定性
RELEASE 的原请求为 proposal Top-1，再附带两个最高分且不重复的候选；runtime 按序最多认证三项，
只执行首个通过的 canonical trajectory。fallback 记为 runtime safety selection，不记 Teacher、
不扣 Teacher penalty，requested likelihood 仍绑定 rank 1；ActionTape 已覆盖 shortlist/rank/check count
并纳入 exact replay。fresh protocol
`ppo_p2_gate_runtime_shortlist_evaluation_v1.yaml` 冻结 smoke seed `805000`、formal seeds
`805001–805006`，比较 candidate K=1 / candidate K=3 / rule。正式 hard gates 包括 safety、completion、
replay、likelihood、每次最多三候选、fallback 非零，以及 K=3 wait/RTT 均不劣且至少一项严格改善。
V1 smoke seed `805000` 完成预定 160 s：服务器 43 tests PASS，三 controller 的 safety、exact
environment replay、requested likelihood 与 shortlist contract 均通过；短 horizon 未完成 12 架，按
smoke 非授权语义保持 `INCOMPLETE`。V1 formal 在 seed `805001` 的 K=1 与 K=3 后 fail-closed 停止：
两者均 12/12 完成且安全、trajectory replay 通过，K=3 提供 38 个 shortlist、实际选择 4 次 fallback、
每次最多检查 3 项；但旧 `_policy_tape_logprob_exact` 在复核时只回放 requested action，没有把 tape
中的 runtime shortlist 交回环境，导致第一次 fallback 后复核世界错误并把 likelihood 标成 false。
因此 V1 是 **INVALID_EVALUATOR**，不是 policy/runtime hard-gate FAIL；`805001–805006` 已消费，
不重跑、不复用其指标。

复核器现与 exact environment replay 一样恢复 `runtime_proposal_shortlists`。fresh V2 successor 已
预注册 protocol SHA `67e61165ef5c056d1b1ebd0a789cafd17f5e0554d07b29128617df3aa8bf6d5d`，
使用 smoke `806000`、formal `806001–806006`，其他 controller、checkpoint、边界和 hard gates 均不变。
V2 formal 已完成 18/18 records、三 controller 各 72/72 requests；全部 safety、completion、exact
replay、requested likelihood、Teacher isolation 与 shortlist contract hard gates PASS。K=3 六个 seed
均实际选择 fallback，每次最多检查 3 项；相对 K=1 的 weighted wait 为 `-9.519080 s`，
request-to-touchdown 为 `-2.055556 s`，两项均改善；K=3 的 wait `11.303291 s` 也优于 rule
`19.883828 s`。manifest SHA256
`6050088d3c28d1531c7e8dde5b828403390bfdeb51dde8bde54e0f2759323905`；正式结论见
[K=3 runtime shortlist V2 summary](evidence/v4_2/ppo_p2_gate_runtime_shortlist_v2_summary.md)。
`RUNTIME_SHORTLIST_FREEZE=true`；按预注册 `PPO_ALLOWED=false`，下一步只允许独立 admission review。

独立只读 admission review 随后全 gate PASS，review SHA256
`a549ffc60c35cb5ac1403ba0c4b59412096b2936d4a209ddb1a25219e09e8783`；它只授权 fresh entry
protocol 预注册，未授权直接训练。fresh entry V1 已冻结 seeds `807001/807002`、flow6/flow12、
160/256 steps 与 K=3 runtime。为保持 PPO requested-action ratio 与实际 shortlist 的一致性，entry
只允许更新 `gate_adapter + critic`，冻结 `actor_encoder/gate_head/proposal_head/temperature`，并把
更新后 shortlist 排序逐状态不变设为 hard gate；任一 update/排序/恢复 gate 失败即事务回滚并保持
`PPO_ALLOWED=false`。协议 SHA256
`d3400bf47d0f7f0d86ce4ed16c5b547b6ab30e0220691eb1f33eef320af932be`。entry 随后
2/2 records 全 gate PASS：26/26 optimizer steps 接受、0拒绝；proposal 子图 byte-exact 不变，
shortlist 排序稳定，安全/replay/likelihood/restore 全 PASS。`admission.json` SHA256
`8d425da16345dd0546851d83bfe706877dd4e6298437b8ef9bd91da324175db3`，因此
`PPO_ALLOWED=true`；entry checkpoints 仍禁止作为 formal 初始化。详见
[K=3 PPO entry V1](evidence/v4_2/ppo_p2_runtime_shortlist_ppo_entry_v1_summary.md)。
formal V1 已独立预注册：fresh training seeds `808001–808006`，按 flow6/8/12 重复两轮并运行
完整 cycle；heldout `809001–809006` 已冻结且训练阶段不可访问。formal 仍从原始 candidate 以
fresh optimizer 启动，只训练 gate adapter + critic；协议 SHA256
`7e6cf765af93925ffe960ff2ab8a4f141e13c1101e8b2f933fd64a2296bf3048`。formal 已完成
6/6 episodes、52/52 requests，354/354 updates 接受、0拒绝；全部 safety/geometry/replay/
likelihood/frozen-shortlist/restore gates PASS。manifest SHA256
`3c1bc69a518f7e5d99bb606d98b0356397e86f97b493947a0f14b0c8b49b010b`，final checkpoint SHA256
`714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e`。详见
[K=3 formal PPO V1](evidence/v4_2/ppo_p2_runtime_shortlist_formal_v1_summary.md)。heldout V1 已按
trained K3 / initial K3 / rule 三控制器预注册，协议 SHA256
`eee2543918bf80610e80ccf11efdac602369d3bb46dd2226ce53ed87d57950e6`。heldout 18/18 records
全部 completion/safety/replay/likelihood/shortlist gates PASS；trained K3 相对 initial K3 的 wait
改善 `0.033386 s`，但 RTT 恶化 `0.019231 s`，违反预注册双指标 non-worse gate，故 immutable
`PASS=false`、checkpoint 不可晋级。两种 K3 均显著优于 rule，说明失败是 PPO 无一致增益，而非
K=3 runtime 边界失效。manifest SHA256
`f44acf5e5c78925b71202e0bb81ce4c2c56d302a51fd207025b2228714a666ce`；详见
[K=3 paired heldout V1](evidence/v4_2/ppo_p2_runtime_shortlist_heldout_v1_summary.md)。
独立只读 selection review 的 9 项 gate 全 PASS，review SHA256
`5617c96255c040797abf2d2f44ed81ce53be955ed94d3a8447a5c943b2f59969`。结论为：保持 heldout
FAIL，拒绝 formal checkpoint，停止追加 P2 PPO，选择 original K3 candidate；仅
`P3_PROTOCOL_PREREGISTRATION_ALLOWED=true`，`P3_ALLOWED=false`、`checkpoint_promotable=false`。
据此已冻结 P3 V1：保持固定几何与 N6/N8/N12、60 s 到达间隔及 K=3 runtime，只把 priority 从
固定 1 恢复为随机 `1/3/5`，SOC 从固定 `.8` 改为可完成区间 `[.55,.90]`；qualification seeds
`810001–810006`，entry `811001/811002`，formal `812001–812006`，并保留 untouched heldout
`813001–813006`。entry 更新不得作为 formal 初始化，formal 仍从 original K3 + fresh optimizer
启动。协议 SHA256 `ea5efc064f763b39cd18f07f55e84f07754aebbe40d8ded588b9b61998559b25`。

fresh P2 gate support V1 因 seed772009 出现 3/959 次 shield 而保持 FAIL；其 rows 不得训练。随后
V2→V5 successor support chain 在不混入旧 support/qualification rows 的前提下完成：V3 为 15/15
安全 episodes 但 train/dev 仅 1,150/908 rows；V4 追加后为 27 个安全 episodes、combined
3,786/1,787 rows，仅 dev cardinality 少 13；V5 dev-only 追加 3 个安全 episodes 后冻结
train/dev 为 3,786/2,415 rows，全部 support gates PASS。V5 manifest SHA256 为
`100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960`，train/dev rows SHA256 为
`40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4` /
`da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b`。其后 learnability audit V2
PASS、anchored repair V3 PASS、Qualification V2 PASS；完整链条见[汇总证据](evidence/v4_2/ppo_p2_gate_support_repair_qualification_summary.md)。

## 2026-09-17 切环轨迹局部修复

checkpoint12 演示确认 m2i 中段逆环向并进入目标环内侧，根因为旧 cubic 仅约束端点切向。已完成局部修复及[固定回归验证](evidence/v4_2/transfer_geometry_repair_20260917.md)：3场景12/12完成，实际切环无逆环向、安全0、回放一致；UID6最小切向速度16.02m/s。历史 PASS/FAIL 不变，但不外推到新生成器；**修复后资格与入口复核已通过，第二个 PPO block 已完成**：24回合96/96架完成，安全/逆环向/shield全0。详见[续训结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。

## 当前冻结的主线语义

| 项目 | 当前值 |
|---|---|
| 研究范围 | Phase A，landing-only；Phase B/C 仅为未来路线 |
| 代码主线 | `terminal_marl/`；`src/` 只读参考 |
| 空域 | `three_ring_v1`：OUTER / MIDDLE / INNER |
| 研究机型 | `eh216s_research_v1` |
| 动力学 | `performance_constrained_v2`；`physics_dt = 0.2 s` |
| 低层观测 | 当前合格策略仍为 `phaseA_low_obs_v2_1_retry_state` 88D；`phaseA_low_obs_v2_2_scheduler_context_94d` 为未晋级 candidate |
| 低层动作 | `low_action_schema_v4_2_hierarchical` |
| 低层分布 | `WAIT/RELEASE` 二分类 gate + RELEASE 条件下 25-way proposal |
| 执行解码 | 1 个 canonical WAIT + 25 个 `speed × merge-time` RELEASE 意图 |
| 高层动作 | `high_airspace_v1_10d`，当前训练关闭 |
| 安全权威 | TrajectoryGenerator → SafetyFilter → PredictiveSafetyShield |
| Teacher 角色 | 训练期 expert labeler / auditor；不是部署控制器或 PPO behavior action |

低层完整语义见 [reference/low_level_policy.md](reference/low_level_policy.md)。

## 里程碑与证据

| 里程碑 | 状态 | 说明 / 证据 |
|---|---|---|
| Phase A 核心环境与 rule pipeline | 已进入后续训练使用 | 具体接口见主规范；历史迁移过程不再作为状态源 |
| D0–D7 动力学 qualification | PASS | [正式报告](evidence/dynamics/dynamics_qualification_report_2026-08-28.md) |
| P0 仿真性能 qualification | PASS | 结果摘要保留在 [历史升级计划](history/plans/CRAZYFLIE_DYNAMICS_LAYER_UPGRADE_PLAN.md) 末节 |
| C0 capacity calibration | PASS | 作为 user-provided frozen evidence 记录于 [V4 migration summary](evidence/v4/v4_new_mainline_migration_summary.md) |
| V4.2 hierarchical S0 | PASS | [S0 qualification](evidence/v4_2/v4_2_s0_qualification_summary.md) |
| Observation V2.1 final freeze | PASS | [final freeze v2](evidence/v4_2/v4_2_v2_1_final_freeze_decision_v2.md) |
| Fresh V2.1 I0-R | PASS | [I0-R qualification](evidence/v4_2/v4_2_v2_1_formal_i0_r_summary.md) |
| Fresh multi-seed I1 | FAIL | RELEASE manifold/context generalization gap；[qualification](evidence/v4_2/v4_2_v2_1_i1_formal_qualification_summary.md) 与 [attribution](evidence/v4_2/v4_2_v2_1_i1_failure_attribution.md) |
| I0-R2 repair collection | FAIL | 未闭合所需 support；协议和结果保存在 `evidence/v4_2/` |
| I0-R3 repair collection | FAIL（历史） | [R3 result](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md) |
| I0-R4 repair collection | PASS | [正式结果](evidence/v4_2/v4_2_v2_1_i0_r4_collection_summary.md) |
| I1 Qualification Protocol V2 / Q2 | PASS | [正式结果](evidence/v4_2/v4_2_v2_1_i1_v2_summary.md)，Q2 1,787 rows，三 seed 全通过 |
| I2 Safe DAgger | R1 PASS / I3 PASS | [I3 协议与 R1 结果](evidence/v4_2/v4_2_v2_1_student_qualification_protocol.md)，976 新事件、35 epochs、全部 gate PASS |
| 初始 PPO 历史准入 | **PASS（冻结 N3–N6 配置）** | [准入结果与训练命令](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，正式 train 模式已接受最新 admission |
| P2 自然流 V3 | **FAIL（entry KL）** | 资格 6/6 PASS；entry seed763001 的 `approx_kl=0.11257042959332467>0.05`，未执行 formal train；[持续课程终态与诊断](evidence/v4_2/continuous_curriculum_20260917.md) |
| P2 自然流 V4 | **训练 PASS / 效果审查 BLOCKED** | target-KL 资格6/6、入口2/2、训练8/8 PASS；heldout 18/18 PASS，但等待55.11s仍高于rule 16.18s；[正式终态](evidence/v4_2/ppo_p2_v4_training_evaluation_summary.md) |
| P2 WAIT/credit诊断 | **FAIL** | fresh stochastic 12/12 records；candidate50/52、baseline48/52；false-WAIT 83.21%/80.76%，安全与shield全0；[诊断终态](evidence/v4_2/ppo_p2_v4_wait_credit_diagnostic_summary.md) |
| Gate forgetting离线审计 | **COMPLETE / 不支持单纯PPO遗忘** | I3 false-WAIT 82.85%，checkpoint24 83.11%，V4 80.37%；I3并未解决P2状态支持缺口 |
| P2 gate support V1 | **FAIL（shield gate）** | 15/15 records、train/dev覆盖通过、安全0；seed772009有3/959次shield，V1零容忍gate失败；rows不得训练 |
| P2 gate support V2 | **PASS** | train 78/78、dev 52/52；5,627/2,009 rows；manifest SHA `09402508bc02155c1369de477dc320d5ca596768d4de6461f4a0526f2c7d9304`；仅进入后续离线审计 |
| P2 gate support V3 / V4 / V5 successor | **V5 PASS** | V3 15/15 安全但 train/dev 1,150/908；V4 后 combined 3,786/1,787（仅 dev cardinality 少13）；V5 dev-only 追加3个安全 episodes，最终 train/dev 3,786/2,415，全部 gates PASS；V5 manifest `100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960` |
| P2 gate learnability audit V2 | **PASS（离线）** | 3/3 fresh probes；manifest SHA `5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679`；support/dev 通过，unanchored adapter probe 在 D1 降级；无 environment/Teacher/formal policy training |
| P2 gate repair V1 / V2 | **FAIL（历史）** | V1 non-finite loss；V2 仅 support-dev NLL `.969>.50` 失败，其他主要 metric 与 D1 anchor gates 通过 |
| P2 gate repair V3 anchored | **PASS** | protocol `f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8`；manifest `e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410`；仅 gate_adapter 六参数变化，T/proposal/critic/common state invariant |
| P2 gate calibration V3 | **PASS（离线）** | fixed `T=2.5`；无 training/optimizer/environment/Teacher；仅授权 fresh student-only qualification |
| P2 student-only qualification V1 | **FAIL（历史终态）** | formal seeds `774001–774006`；原 false-WAIT / RELEASE recall gates失败；seeds 不得重跑或调参 |
| P2 student-only qualification V2 | **PASS（当前终态）** | formal seeds `782001–782006`；6/6、52/52、621 actor rows；sampled false-WAIT `18/206=.0873786`、false-RELEASE `16/415=.0385542`；仅新授权 P2 effectiveness evaluation |
| P2 paired effectiveness V1 | **FAIL（当前终态）** | formal seeds `788001–788006`；18/18 controller episodes、各52/52请求；全部 hard gates PASS；等待 gap closure `.4600799557<.50`，其余效果 gates PASS；不授权 admission review/PPO/P3 |
| P2 flow12 feasibility attribution V1 | **COMPLETE（诊断）** | formal seeds `789001–789006`；6/6、72/72请求、1,050 actor rows；89次 rejected RELEASE 中66次为 Teacher也判RELEASE但proposal不同，74次发生于后半到达；仅支持转入fresh proposal support/learnability设计 |
| P2 exact-proposal support V1 | **PASS（数据）** | train/dev seeds `790001–790009` / `791001–791006`；540/455 rows；15/15 episodes、180/180请求；仅授权离线exact-label learnability audit |
| P2 exact-proposal learnability V1 | **FAIL** | anchored residual probes fresh-dev Top-1约`.433`、Top-3约`.577`，D1 Top-1约`.962`；未授权repair |
| P2 exact-proposal failure attribution V1 | **COMPLETE** | standalone MLP Top-1 `.580–.600`、Top-3 `.712–.732`；分类为`observation_or_label_learnability_limit_supported` |
| P2 proposal feasible-set diagnostic V1 | **COMPLETE** | fresh seeds `794001–794003`；36/36请求；166个requested-RELEASE状态，Top-1/Top-3可行107/123，27个状态无可行RELEASE |
| P2 proposal-set support V1 | **PASS（数据）** | train/dev seeds `795001–795009` / `796001–796006`；463/372 rows；15/15、180/180请求；全部episode/split/catalogue gates PASS |
| P2 proposal-set learnability V1 | **FAIL** | 3个D1-anchored probes整体Top-1约`.777–.780`，partial-set Top-1约`.478–.484`；Top-3/D1通过，Top-1与o2m失败 |
| P2 proposal-set failure attribution V1 | **COMPLETE** | standalone linear仅1/3全gate PASS，MLP 0/3；分类为`observation_or_set_target_learnability_limit_supported`；不授权repair |
| P2 proposal-conditioned feasibility V1 | **FAIL** | 88D+proposal参数的binary MLP feasible recall约`.90`，但 infeasible recall `.454–.462`、AUC `.793–.810`，o2m与全部综合gate失败；不授权factorized architecture或repair |
| Observation V2.2 closure V3 | **COMPLETE（诊断，不闭合）** | 6/6 episodes、247/106 train/dev rows；94D 的 AUC 3/3 提高，但 negative recall、F1、Top-1 不一致；observation-only repair 不授权 |
| V2.2 25-logit feasibility head exploratory | **COMPLETE（不闭合）** | negative recall/F1 3/3 改善，但 AUC 3/3 下降、Top-1 仅1/3改善；architecture freeze 不授权 |
| Observation V2.2 hybrid confirmation V4 | **FAIL（架构）** | collection 6/6 PASS；hybrid 仅1/3 probes 全 gate PASS；学习式 proposal 分支停止，不再基于 V4 dev 调参 |
| Learned gate + fixed neutral proposal V1 | **INVALID_IMPLEMENTATION** | rollout 绕过 `act_details()` adapter；candidate/fixed action tapes 完全相同；表面 PASS 作废，803xxx 不重跑 |
| Learned gate + fixed neutral proposal V2 | **FAIL（效果）** | 18/18 records 安全完成且实现 hard gates PASS；RTT改善4.36s，但平均等待恶化4.55s，aggregate gate FAIL |
| K=3 bounded runtime shortlist V2 | **PASS / FROZEN** | fresh `806001–806006`；18/18 records、216/216 requests；全部 safety/replay/likelihood/contract gates PASS；K3 相对 K1 等待改善9.52s、RTT改善2.06s |
| K=3 PPO admission review V1 | **PASS（只读）** | review SHA `a549ffc60c35cb5ac1403ba0c4b59412096b2936d4a209ddb1a25219e09e8783`；仅授权 fresh entry protocol 预注册，不直接授权 PPO |
| K=3 PPO entry V1 | **PASS / PPO_ALLOWED** | fresh `807001/807002`；26/26 updates 接受、0拒绝；全部 safety/replay/likelihood/frozen-shortlist/restore gates PASS；entry checkpoints 不得作为 formal 初始化 |
| K=3 formal PPO V1 | **PASS / HELDOUT ALLOWED** | fresh `808001–808006`；6/6、52/52完成；354/354 updates 接受、0拒绝；全部 hard gates PASS |
| K=3 paired heldout V1 | **FAIL（效果）** | 18/18 records hard gates PASS；trained vs initial wait `-0.033386s`、RTT `+0.019231s`；formal checkpoint 不可晋级，seeds 不重用 |
| P2 candidate selection review V1 | **PASS（只读）** | original K3 被选为下一阶段候选；formal checkpoint 淘汰；`P2_PPO_STOP=true`；仅授权 P3 protocol 预注册 |
| P2R dense landing-flow semantics V1 | **PASS（诊断/资源语义）** | seeds `809006–809009`：300秒 touchdowns `23→30`、completed `17→22`、并发final `1→2`、LZ regret归零且安全计数全0；seed `809010` confirmed gate miss `4/108`；不授权新 PPO 或 checkpoint promotion |
| P3 priority/SOC curriculum V1 | **PREREGISTERED / 未运行** | 固定几何和流量课程；priority `1/3/5`、SOC `[.55,.90]`；qualification→entry→formal fail-closed pipeline |
| A6 高层动态空域训练 | BLOCKED | 需要低层 static-geometry qualification 后冻结低层策略 |
| Crazyflie 实物验证 | FUTURE | 仅在主仿真策略冻结后进入 small-N 缩尺验证 |

## 保留的历史失败：I0-R3

I0-R3 的 provenance、Teacher determinism、exact replay、Safety、partition independence、
aliasing 与 nominal anchors 均通过，但正式结果仍为 FAIL：

- complete-cycle 停止产生 4,541 个训练 rows 和 1,511 个内部审计 rows，超过冻结上限
  4,200 / 1,400；
- proposal-14 的 N=5 支持只有 12 个独立 lifecycle，低于最低 15；
- shared gate false-negative、shared proposal error，以及 post-WAIT、transition/retry、
  post-conflict、proposal-11、proposal-14 的 radius coverage gate 全部未达预注册阈值。

R3 仍为 FAIL；后续通过的是独立预注册的 R4，不能追溯升级 R3。完整数值见
[I0-R3 collection summary](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md)。

## 当前进度与结果

| 阶段 | 已核验进度 | 结论 |
|---|---|---|
| 初始 PPO block1 | 12回合，48/48架完成；checkpoint12独立评估24/24架完成 | 安全与闭环 gate PASS |
| 轨迹修复后 block2 | 24回合，96/96架完成；checkpoint24配对评估双方各24/24架完成 | 全 gate PASS，但候选相对 checkpoint12 的完成时间差为0/0/+12/0/+9/-2秒，没有明确改善 |
| P2 可达性 pilot | N6/N8/N12，rule/student各52/52架完成 | 可达性通过；N12 student等待65.66/73.91秒，rule为19.07/26.88秒，效率差距仍明显 |
| P2 V1 | 前23次更新合格，第24回合未更新 | initial ratio 检查误差超限，历史 FAIL |
| P2 V2 | 从 checkpoint23 恢复后5次更新合格，第6回合未更新 | 连续轨迹内侧偏差超限，历史 FAIL |
| P2 V3 | 新生成器资格6/6通过；entry仅执行1/2 | seed763001 的 KL 超限，formal train 未启动，历史 FAIL |
| KL 双种子诊断 | fresh 765001/765002；恢复 Adam 与清零 Adam 配对 | 仅诊断；未复现原失败，不支持清零 Adam 作为修复 |
| KL 更新保护 | 已提交并接入独立 V4 协议；资格6/6、入口2/2、训练8/8 | 468/468 formal optimizer steps 接受，0拒绝；最大 accepted rollout KL 0.006534<0.01 |
| P2 V4 heldout | 755001–755006 已一次性消费；candidate/baseline/rule 各52/52完成 | candidate 等待55.11s，优于baseline 65.02s但落后rule 16.18s；效果审查不通过 |
| P2 WAIT/credit fresh diagnostic | 771001–771006 已消费；12/12 records生成 | `DIAGNOSTIC_COMPLETE=false`；candidate50/52、baseline48/52；false-WAIT和team credit失配 |
| P2 gate support V2 | 775001–775009 / 776001–776006 已消费；78/78、52/52请求完成 | train/dev 5,627/2,009 rows；support V2 PASS，仅进入离线 learnability |
| P2 gate support V3 → V4 → V5 | successor seeds 已按协议消费；V3 15/15 安全，V4 27 个安全 episodes，V5 dev-only 3 个安全 episodes | 最终冻结 train/dev 3,786/2,415，V5 全部 support gates PASS |
| P2 gate learnability V2 → repair V3 | support V5 后已完成；learnability V2 与 anchored repair V3 均 PASS | learnability 3/3 fresh probes；repair 仅 gate_adapter 六参数更新，D1 anchor 与非 adapter state 保持不变 |
| P2 student-only qualification V1 | 774001–774006 已消费；6/6 episodes、52/52 UAVs、983 actor rows；recovery仅补写manifest | **FAIL（immutable）**：sampled false-WAIT 0.333333、RELEASE recall 0.625786；不得重跑/调参 |
| P2 student-only qualification V2 | 782001–782006 已消费；6/6 episodes、52/52 UAVs、621 actor rows；formal manifest/marker 已落盘 | **PASS**：sampled false-WAIT 0.0873786、false-RELEASE 0.0385542；概率 macro-F1 0.9532325、WAIT recall 0.9590361、RELEASE recall 0.9563107、NLL 0.1034486、Brier 0.0314997；仅 effectiveness evaluation 新获允许 |
| P2 paired effectiveness V1 | 788001–788006 已消费；candidate/baseline/rule 各6 episodes、52/52请求 | **FAIL**：candidate等待15.9619s、baseline17.3097s、rule14.3801s；等待 gap closure 46.01% 未达50%；request-to-touchdown与全部hard gates通过 |
| P2 flow12 feasibility attribution V1 | 789001–789006 已消费；6/6 episodes、72/72请求、1,050 actor rows | **COMPLETE**：66/89 rejected RELEASE 为 Teacher-RELEASE + proposal mismatch；late half 占74/89 rejections；o2m/m2i分别贡献55/11个该类事件；无训练授权 |
| P2 exact-proposal target chain | support 15/15 episodes PASS；learnability 3个probe FAIL；failure attribution COMPLETE | exact canonical Teacher proposal 在fresh dev上不可稳定学习；未保存或修改策略 |
| P2 proposal feasible-set / support | diagnostic 3/3 episodes COMPLETE；support 15/15 episodes、180/180请求 PASS | train/dev 463/372集合标签rows；Top-1可行率351/463与265/372；仅进入离线audit |
| P2 proposal-set learnability / attribution | anchored audit 3/3 probe FAIL；standalone linear/MLP各3 seeds attribution COMPLETE | Top-3、D1保持与接口aggregate可过，但partial-set Top-1未闭合；repair/PPO仍禁止 |
| P2 proposal-conditioned feasibility | linear/MLP各3个probe，11,575/9,300 train/dev state-proposal pairs | **FAIL**：MLP 0/3；o2m、negative recall、macro-F1、AUC与state Top-1未闭合；旧closure-bank可分性未泛化到fresh flow12 |
| P3–P5 / static qualification | 尚未执行 | 满意策略与 A6 高层动态空域均未达成 |

## 已识别的问题与处理状态

1. **奖励与等待目标冲突已修正，但效率改善仍不足。** 旧 progress 奖励计入绕环距离：外环22m/s约每秒+0.022，高于时间惩罚-0.01。P2配置将该系数置0，物理、安全及其他奖励保持不变。V4 heldout 已验证 candidate 等待相对 checkpoint24 baseline 下降9.91s，但仍比rule高38.93s；修正有效果，不足以认定收敛。
2. **初始动作概率检查存在批次浮点差异，已局部修复。** V1 seed754024 的误差为1.1444091796875e-5，超过1e-5。原action tape只读复放表明完整world批次误差0，actor子批次复现误差。检查器现保留原采样批次再mask，阈值和PPO行为不变；后续V2/V3初始ratio检查通过。
3. **轨迹节点检查漏掉连续线段内切，已局部修复并通过V3资格复核。** V2 seed757006实际内侵0.0592194m，超过0.05m。UID39见证曲线节点半径均合格，但线段最近点半径199.9275204m。修复使用每段投影最近点，仅超限旧曲线进入已有forward-annular fallback，保持有效旧曲线和原安全容限；此前38项定向测试通过。V3资格最大内侵0.0186652m，安全/shield/逆环向均0。
4. **V3 的 PPO 更新漂移已由独立 V4 协议闭合，但历史 FAIL 不变。** V4 每个 optimizer step 后检查完整 rollout requested-action KL，超限或非有限值时事务回滚模型与 Adam；原0.05 gate不放宽。资格、入口和8回合训练均通过，formal 468次更新全部接受，最大 accepted rollout KL 0.006534<0.01。该结果恢复了独立 V4 的训练准入，不追溯升级 V3。
5. **P2 gate support 的 distribution successor 与 Qualification V2 已闭合。** support V3/V4/V5 逐步补足 calibrated candidate 自身状态分布，最终 train/dev 3,786/2,415 且全部 support gates PASS；learnability audit V2 3/3 PASS，anchored repair V3 PASS。Qualification V1 的 `774001–774006` FAIL 保持 immutable；Qualification V2 的 `782001–782006` 以 6/6、52/52、621 rows 和全部 episode/safety/provenance gates PASS 终态 PASS。两组 seeds 均不得重跑或用于调参。
6. **fresh attribution 将主要剩余瓶颈定位到 proposal/执行可行性。** 6个全新 flow12 episodes 全部安全完成；1,050 actor events 中 false-WAIT/false-RELEASE 为43/23，而89次 rejected RELEASE 中66次发生在 Teacher同意RELEASE但student proposal不同的事件，且这66次无一proposal match；其中55次在o2m、11次在m2i。late arrival half 占74/89 rejections与52/66上述proposal mismatch rejects。i2f 的6次reject则全部对应false-RELEASE。
7. **单标签、可行集合和动作条件化三条proposal监督路线均未闭合。** exact-label audit 与独立failure attribution先后失败；完整25-proposal catalogue显示166个requested-RELEASE状态中107个Top-1可行、123个Top-3可行，说明单一Teacher tie-break并非正确监督目标。fresh set support 463/372 rows全部gate PASS，但D1-anchored set probes的overall Top-1仅`.777–.780`、partial-set Top-1仅`.478–.484`，o2m仍失败。去掉D1 anchor和旧logits后，standalone linear仅1/3全gate通过，MLP 0/3。再将每个状态展开为25个`88D+proposal` binary样本，MLP feasible recall虽约`.90`，infeasible recall仍仅`.454–.462`，AUC`.793–.810`且o2m失败。因此当前证据支持88D信息/多UAV同一步竞争或target factorization限制，不支持直接proposal repair。
8. **当前课程覆盖仍较窄。** P2 N6/N8/N12是总请求数，N12 pilot在空峰值仅6–7架，不能称12架并发或高负载验证。自然流首到达10秒、间隔固定60秒；SOC=.8、常规priority=1，尚未覆盖间隔扰动、priority/SOC异质性、20/40/60总请求、持续/突发流和合法静态几何泛化。当前课程按单项增难的主线合理，但离最终终端区场景仍有明确跨度。
9. **文档参数及运行交接需保持一致。** 需求文档仍有旧高度/速度数字；当前验证配置为三环高度100/70/40m、速度22/20/16m/s，不能静默切换。后续长任务按用户要求由`luna_worker`跟踪、主agent静默等待终态。Git无remote，代码仅经Syncthing同步，不能声称已推送。
10. **密集可视化中的三个资源问题已闭合。** nearest-available 消除了可避免的 LZ 距离后悔；downstream-first 防止上游插入顺序抢占同一步认证；final容量3允许安全过滤后的并发进近。环空闲不再直接解释为低效：seed809010 的95个“有名义资源但未请求”时刻中只有4个存在 individually-certified RELEASE，另外91个受航迹/headway/安全约束。环容量是上限而非占满目标。

## 模型、评估与证据边界

- 当前 P2 V4 最终候选：`p2_resume_v4_20260918/train/checkpoint_8.pt`，SHA256 `5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`。训练与 heldout hard gates PASS，但因 rule 效率差距尚未晋级 P3。
- 配对 baseline：`ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt`，SHA256 `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`。
- V4恢复起点：`p2_resume_v2_20260917/train/checkpoint_5.pt`，SHA256 `8a8466f61119f7f5b5f780249d26f218ee9b8ae3e4aededc5c7f5fe2e51b201b`。它是V2失败训练块中的最后合格中间模型，现仅作为V4 provenance，不是当前最终候选。
- V3 root/entry manifest SHA256：`2098af1c3109c17fc489b7c9f20a9db11eb67aa62b77b5a03beb2de66909cd7d`；KL诊断manifest：`41017e10f45105a6125213ac51a630bdcfda9c35bac196608e2affc891955f69`。结果位于输出根目录下 `p2_resume_v3_20260918` 与 `p2_v3_kl_diagnostic_20260918`。
- checkpoint24评估seeds710001–710006与P2最终评估755001–755006均 **已消费**，不得重跑或用于调参。V4 heldout manifest SHA256为 `167d22cb29d8ef6d08275cb0ff4cbe270c0d9005611a176b85d81fd64730f104`。
- WAIT/credit diagnostic seeds771001–771006亦 **已消费**；manifest SHA256为 `b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93`，终态FAIL不得重跑或升级。
- P2 gate support→learnability→repair→qualification 链的完整证据见[汇总](evidence/v4_2/ppo_p2_gate_support_repair_qualification_summary.md)：最终 V5 support manifest SHA256 为
  `100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960`，train/dev rows SHA256 为
  `40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4` /
  `da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b`；learnability audit V2 manifest SHA256 为
  `5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679`。
- Anchored repair V3 protocol/manifest/checkpoint SHA256 分别为
  `f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8` /
  `e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410` /
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`；仅 gate_adapter 六参数改变，T/proposal/critic/common state invariant。
- Qualification V2 protocol SHA256 为 `7ab9e2a84b9aea8da2ea975d98b4867de4a45bd5725bc72019b2e95ddbb75094`，formal
  `782001–782006` 已消费；manifest SHA256 为
  `7cd4f02c88dc953fe1545c613424e8c961c295d4f8c099456d6697f300088feb`，marker SHA256 为
  `43fafcd99e22d48bffb6c2389767a265c2376b05f95a070ae3d823f4066949d9`。形式化 rollout 的 source runner
  hash 为 pre-fix `b91e0d3794495bb768cf80ab266f1584b0af01addd2b0998c9c07e30af8d4a82`；CLI serialization 修复提交
  `d486bab` 仅修复终端打印，没有 rerun/reset，不能声称 formal rollout 使用了 patched source。
- Effectiveness V1 protocol SHA256 为 `5882361adcabaff4c0017cae73a8b95dc8029e204942bf4a83879981bd321d84`；formal
  `788001–788006` 已消费。manifest/log/marker SHA256 分别为
  `6aaebeac4dc8dcf425ce250df6c6abcd991921263605b3dd71d3d717f0783e88` /
  `df9aebf8860212a72ede78f0078e22d0a8d8df90535d20798a436ba4f28c3acd` /
  `b9ca4306835b19756c070eb7917d3e01de429209329e59f397257a5e4c54fd8c`。终态 FAIL，不得重跑或以其 seeds 训练/调参。
- Feasibility attribution V1 protocol SHA256 为 `50b2cb7dc40160a54ecfe150122d7bcc3407b638a40e887b20045120e220b1fd`；formal
  `789001–789006` 已消费。manifest/rows/log/marker SHA256 分别为
  `f2382b1a8c5ee486c4246e716fc27b97a34d2948bfe90633e33d0f428a56f35c` /
  `b07f51b0354d92943864232b7698a0d53061de22e6593911b10e40a8192c412f` /
  `371fd775e823fe347a593fec4ba79ca03217bdfb44ffc38ddbceb382bcbc5161` /
  `89e7af446d312c06019d88f2c4b1e8b4424bde8711dae4cba9294550d1ddfa2e`。rows 仅用于归因，不得训练。
- Exact-proposal support/audit/failure-attribution manifest SHA256 分别为
  `48c0b1cd0b92bd746bd52f18090bbffd8facf59457eb82d503802eb4fc2c512b` /
  `87bdf9b5ce0b9d9cbf967d858669a6f17679c0cf271d83c8c39e7f4fa4e52e95` /
  `3c8e0980f4f970bb66ca207ceb3305dc3fbe9e3eb79e9e7a827a1b086acf11f5`。support rows冻结；audit FAIL，不能进入exact-label repair。
- Proposal feasible-set diagnostic protocol/manifest/catalogue SHA256 分别为
  `dbe387cc134741f87cde8caa2b8910410e42e5ad756dcb197bc2c04c208eabca` /
  `957bc2535fbef767a90fbf6cd299ea5aa38cd2f680099a3c5f79aff97fe4821f` /
  `c7dbbf203084d1b90db8662f42514b8c3f45a5e94b2d7e9e706ad80143784242`；仅诊断，不可训练。
- Proposal-set support protocol/manifest/train/dev SHA256 分别为
  `fa2d764604e05526262549559c718291caeef3ca23202bce38d20d3e038d2972` /
  `d057d0f39a3a99f6641c331eecd1a127a2b639b5046bf78d39b55d1b8caf7f37` /
  `f5511da554f21614e399b6384609bac93a41e9fdef94b8735eb2ecd6cb5d3f14` /
  `dd07e8f2b877bd7f08db113aa0fc8988ed8d106223d799284b3e1f8b6324d17a`。support PASS，但后续learnability FAIL，rows不得用于repair。
- Proposal-set learnability audit 与 failure attribution manifest SHA256 分别为
  `a1b76fa16427cecdc0c70e0d38c6247d8e6e6d943063aeeddd00a8cede28e937` /
  `4a1d6d9d9b594c1d680203e8c8b728e08101ec1fdfc26aa64028be9f800c5e10`。二者均未保存probe checkpoint，candidate state digest始终为
  `fa8740a8a4cb54fffd3bb669537133ede8f231edd754e4b10ebfc0f982a0ed61`。
- Proposal-conditioned feasibility audit protocol/manifest SHA256 分别为
  `68694cefa2e8907b3d4beb77ca3abc2daf569bc9b5ffab6b46b1d70b9888a896` /
  `4b37952e8b624284f1d45280043f89aaad9ef97be4e1f33e690644c03827067f`。终态FAIL；未保存probe checkpoint，未调用environment/Teacher/policy optimizer。
- 当前没有遗留 server process；Qualification V1 的 recovery manifest/marker 只属于旧 `774001–774006` FAIL 证据，不改写新 V2 PASS。
- 入口 `entry/checkpoint_1.pt` 和Adam诊断快照均不能晋级正式模型；历史失败manifest不得重写。
- 服务器输出根为 `~/training_outputs/terminal`，本地镜像为 `D:/7_codespace/training_outputs/terminal`。源码只在本地修改，经Syncthing同步，服务器仅运行。

## 当前允许的下一步

1. 保留 support、learnability、repair、Qualification V1/V2、effectiveness、exact-proposal 与 proposal-set 全链条为 immutable evidence；已消费环境seeds不得 rerun/reset，也不得用于训练、阈值或模型调优。
2. 暂停 proposal repair、learned proposal head/loss 与 fixed proposal index 调整；V1/V2/V3/V4 及 fixed-neutral V1/V2 seeds、rows、metrics 均为 immutable evidence，不得重跑或作为新阈值调优集。
3. K=3 runtime boundary 已冻结；P2R 只改变 scheduler-owned LZ/同一步仲裁/final资源语义，不得隐式加入 runtime catalogue search 或让策略直接输出 `lz_id`。
4. P2R 当前不需要因4/108个单种子漏放行立即重训。若后续 fresh 多种子确认稳定的 individually-certified-but-not-requested 缺口，先独立预注册 train/dev/heldout seeds 和 warm+flow mixed curriculum，再决定 gate support/训练；当前诊断不得直接授权 PPO。
5. P3 若继续执行，必须保留 P2R dense24 回归和旧自然流回归；再按[执行计划8.5节](execution_plan.md)逐步引入间隔扰动、P3任务/资源异质性、P4交通压力、P5静态几何泛化。完成 static qualification 后才进入 A6。后续长训练或评估优先由 `luna_worker` 跟踪；若 worker 不可用，主agent可接管已定义执行任务并报告原始证据。

完整协议、旧失败与诊断详情见[持续课程证据](evidence/v4_2/continuous_curriculum_20260917.md)；初始准入见[准入结果](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，轨迹修复与第二训练块见[续训结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。

## 不应恢复的旧路线

- V3 factorized 三头策略；
- V4 30-way 与 V4.1 flat-26 actor；
- FULL-Teacher PPO 或用 Teacher corrected action 计算 requested-action PPO ratio；
- 87D V2 checkpoint 直接续训 88D V2.1；
- 两环、8D high-level、自由 `ψ/γ/V` 或姿态/电机 action；
- 把日期型 handoff 或单次 diagnostic 当作新的 canonical specification。
