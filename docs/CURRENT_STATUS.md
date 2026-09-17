# 项目当前状态

> **状态日期**：2026-09-18
>
> **代码快照**：分支 `codex/v4-2-v2-1-formal-i1`，V3 几何修复提交 `114a5e0`
>
> **当前阶段**：Phase A / A5 固定空域低层策略，初始课程 PPO 正式训练准入完成
>
> 本文是滚动状态页。规范语义以 [README.md](README.md) 中列出的主规范为准。

## 一句话状态

低层 V4.2 / Observation V2.1（88D）已通过 R4、I1 V2/Q2、88D Safe DAgger、
I3 student-only qualification、PPO admission 和入口 V2 验证。
**PPO_ALLOWED = true**，仅限固定空域 N3–N6 初始课程；首个正式 PPO block 已完成（12 episodes、48架全部完成，全部 gate PASS）。P2 自然流 V3 入口未通过，未获得扩大训练范围的准入。
完整证据及可执行命令见 [准入结果](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)。
历史 R3、旧 I1 和入口 V1 的 FAIL 保持不变。
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
| 低层观测 | `phaseA_low_obs_v2_1_retry_state`，88D |
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
| PPO | **PPO_ALLOWED = true（初始 N3–N6）** | [准入结果与训练命令](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，正式 train 模式已接受最新 admission |
| P2 自然流 V3 | **FAIL（entry KL）** | 资格 6/6 PASS；entry seed763001 的 `approx_kl=0.11257042959332467>0.05`，未执行 formal train；[持续课程终态与诊断](evidence/v4_2/continuous_curriculum_20260917.md) |
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

## 持续课程推进（2026-09-17）

用户已授权持续推进至满意低层策略，仍逐级验证，不跳安全gate。checkpoint24独立评估PASS但未见明确性能增益；P2 N6/N8/N12 rule/student 12次可达性pilot全部PASS，student等待明显高于rule。发现旧距离奖励也奖励绕环，P2仅将该奖励系数设0，物理/安全链不变。P2 V1第24回合因initial_ratio检查停止：前23次更新成功，失败回合未更新且安全/完成通过。只读复放证明full-world likelihood误差0，actor子批次浮点误差1.1444e-5；已修正验证器批次一致性，阈值不变，V2从checkpoint23恢复后5次更新成功，第6回合因legacy曲线线段内切0.0592m超0.05m停止。已定位并补连续线段最近点检查，保持原门槛；V3 从 V2 checkpoint5 恢复后资格 6/6 PASS，但 entry seed763001 因 `approx_kl=0.11257042959332467` 超过 0.05 FAIL，未执行 formal train、未生成 checkpoint8。独立 fresh Adam 诊断未支持清零 Adam 为修复，当前候选是显式 target-KL 逐步检查与超限回滚，须先审查和预注册；最终P3–P5与独立效果验收尚未完成，不能称满意策略。Luna high额度不足时主任务接管监控。

## 当前允许的下一步

- 已完成已注册的 PPO 初始 block；P2 V3 entry FAIL 后没有可供独立评估的 checkpoint8，不消费 heldout 755001–755006，先完成 KL 机制审查和新协议注册。
- 正式入口：`terminal_marl.training.run_v4_2_v2_1_ppo --mode train`；参数及证据哈希见准入结果。不能用 entry-test checkpoint 初始化正式训练。
- Admission：`~/training_outputs/terminal/v4_2_ppo_entry_verify_v2_20260917/admission.json`，SHA256 `959bf9300fb44f4eec02783136dcfda11826dbf125e36a85e60550e241511c62`。
- I3 六个 episode 全部完成 24 架 UAV，队列清空，Safety 与 actor shield 均为零；最慢 flow6 为 867/900 秒，扩大交通或缩短间隔前仍需验证。
- 最新 16 项 admission/PPO/rollout/I3 定向测试通过；入口 V2 两个真实 rollout 的 ratio、更新、replay、checkpoint 恢复通过。正式 PPO 收敛效果尚未评估。
- 首个正式 PPO PID `2610770` 已退出，12/12 episodes 全通过；[正式结果](evidence/v4_2/v4_2_v2_1_ppo_initial_block_summary.md)。Luna high 已完成监控。
- checkpoint_12.pt 已通过六个 fresh episode 独立评估，24架全部完成；[结果与本地可视化命令](evidence/v4_2/checkpoint12_evaluation_summary.md)。与DAgger配对结果不足以证明普遍优势，暂保持N3–N6。
- 修复后第二block已完成：六回合student-only（24/24架）→两回合入口→24回合正式续训（96/96架），全部gate PASS；[结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。最终 `ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt`，未扩大N3–N6范围。下一步使用保留710001–710006独立评估该checkpoint，完成前不声称策略普遍改善、不自动升课程。
- P2 V3 终态与 paired KL 诊断已记录于[持续课程证据](evidence/v4_2/continuous_curriculum_20260917.md)：seed763001 entry 唯一失败 gate 为 bounded KL；诊断 seeds765001/765002 仅为 non-formal evidence，不能覆盖原 FAIL 或直接放宽门槛。
- 各阶段数量、到达流、属性与约束审查已更新到 execution_plan.md 的8.5节；后续P2–P5尚未准入，不自动扩大当前训练范围。
- 本地 Git 无 remote；代码仅经 Syncthing 同步，提交未推送。保留用户未提交文件。

```text
R4 PASS → I1 V2/Q2 PASS → 88D Safe DAgger PASS → I3 PASS
→ PPO admission + entry V2 PASS → bounded initial PPO training
→ load/geometry curriculum → full static qualification → A6
```
## 不应恢复的旧路线

- V3 factorized 三头策略；
- V4 30-way 与 V4.1 flat-26 actor；
- FULL-Teacher PPO 或用 Teacher corrected action 计算 requested-action PPO ratio；
- 87D V2 checkpoint 直接续训 88D V2.1；
- 两环、8D high-level、自由 `ψ/γ/V` 或姿态/电机 action；
- 把日期型 handoff 或单次 diagnostic 当作新的 canonical specification。
