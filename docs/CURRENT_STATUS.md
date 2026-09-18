# 项目当前状态

> **状态日期**：2026-09-18
>
> **代码快照**：分支 `codex/v4-2-v2-1-formal-i1`；WAIT/credit终态提交 `4bb400f`，fresh gate support收集协议提交 `4166249`
>
> **当前阶段**：Phase A / A5；P2 V4确定性heldout通过但fresh stochastic WAIT诊断失败，阻塞在P2 gate support/credit修复
>
> 本文是滚动状态页。规范语义以 [README.md](README.md) 中列出的主规范为准。

## 一句话状态

低层 V4.2 / Observation V2.1（88D）已完成 R4 → I1 V2/Q2 → Safe DAgger → I3 → 初始 PPO 准入，并在 target-KL 保护下完成 P2 V4 与一次性确定性heldout。随后fresh stochastic WAIT/credit诊断产生12/12 records但未闭合排空gate：candidate 50/52、baseline 48/52。**当前根因边界已收敛为P2自然流状态上的gate support/calibration缺口与team-step信用分配失配；不能进入P3或继续盲加PPO。**

初始 N3–N6 的 `PPO_ALLOWED=true` 是对应冻结版本及协议的历史准入。P2 V1/V2 累计 28 次合格更新；V3 在入口 KL 超限后保持历史 FAIL。独立预注册的 V4 重新资格与入口通过，并补足剩余 8 回合；这不改写 V1–V3 的历史终态。

2026-09-18 V4正式训练与确定性heldout已终止；WAIT诊断及离线审计也已终止，当前没有需监控的服务器任务。历史PASS/FAIL均保留；V4结果见[训练与评估终态](evidence/v4_2/ppo_p2_v4_training_evaluation_summary.md)，当前阻塞见[WAIT/credit诊断终态](evidence/v4_2/ppo_p2_v4_wait_credit_diagnostic_summary.md)。

fresh P2自然流gate support收集已在任何新reset前预注册：train 772001–772009、dev
773001–773006、未来资格774001–774006严格分离；当前仅允许先做smoke与formal support
collection，不授权训练。见[gate support收集协议](evidence/v4_2/ppo_p2_gate_support_collection_protocol.md)。

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
| 初始 PPO 历史准入 | **PASS（冻结 N3–N6 配置）** | [准入结果与训练命令](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，正式 train 模式已接受最新 admission |
| P2 自然流 V3 | **FAIL（entry KL）** | 资格 6/6 PASS；entry seed763001 的 `approx_kl=0.11257042959332467>0.05`，未执行 formal train；[持续课程终态与诊断](evidence/v4_2/continuous_curriculum_20260917.md) |
| P2 自然流 V4 | **训练 PASS / 效果审查 BLOCKED** | target-KL 资格6/6、入口2/2、训练8/8 PASS；heldout 18/18 PASS，但等待55.11s仍高于rule 16.18s；[正式终态](evidence/v4_2/ppo_p2_v4_training_evaluation_summary.md) |
| P2 WAIT/credit诊断 | **FAIL** | fresh stochastic 12/12 records；candidate50/52、baseline48/52；false-WAIT 83.21%/80.76%，安全与shield全0；[诊断终态](evidence/v4_2/ppo_p2_v4_wait_credit_diagnostic_summary.md) |
| Gate forgetting离线审计 | **COMPLETE / 不支持单纯PPO遗忘** | I3 false-WAIT 82.85%，checkpoint24 83.11%，V4 80.37%；I3并未解决P2状态支持缺口 |
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
| P3–P5 / static qualification | 尚未执行 | 满意策略与 A6 高层动态空域均未达成 |

## 已识别的问题与处理状态

1. **奖励与等待目标冲突已修正，但效率改善仍不足。** 旧 progress 奖励计入绕环距离：外环22m/s约每秒+0.022，高于时间惩罚-0.01。P2配置将该系数置0，物理、安全及其他奖励保持不变。V4 heldout 已验证 candidate 等待相对 checkpoint24 baseline 下降9.91s，但仍比rule高38.93s；修正有效果，不足以认定收敛。
2. **初始动作概率检查存在批次浮点差异，已局部修复。** V1 seed754024 的误差为1.1444091796875e-5，超过1e-5。原action tape只读复放表明完整world批次误差0，actor子批次复现误差。检查器现保留原采样批次再mask，阈值和PPO行为不变；后续V2/V3初始ratio检查通过。
3. **轨迹节点检查漏掉连续线段内切，已局部修复并通过V3资格复核。** V2 seed757006实际内侵0.0592194m，超过0.05m。UID39见证曲线节点半径均合格，但线段最近点半径199.9275204m。修复使用每段投影最近点，仅超限旧曲线进入已有forward-annular fallback，保持有效旧曲线和原安全容限；此前38项定向测试通过。V3资格最大内侵0.0186652m，安全/shield/逆环向均0。
4. **V3 的 PPO 更新漂移已由独立 V4 协议闭合，但历史 FAIL 不变。** V4 每个 optimizer step 后检查完整 rollout requested-action KL，超限或非有限值时事务回滚模型与 Adam；原0.05 gate不放宽。资格、入口和8回合训练均通过，formal 468次更新全部接受，最大 accepted rollout KL 0.006534<0.01。该结果恢复了独立 V4 的训练准入，不追溯升级 V3。
5. **当前主要阻塞是P2 gate support/calibration与team-step credit失配。** fresh诊断中candidate false-WAIT为1,601/1,924=83.21%，三个interface均严重；false-WAIT状态release-eligible、retry=0，RELEASE概率中位数仅6.59e-7。requested WAIT raw team GAE均值+2.17，RELEASE为-1.05；false-WAIT均值+2.60。I3离线false-WAIT已达82.85%，不支持“只是后续PPO遗忘”。因此下一步是fresh自然流label support与seed-disjoint gate repair，而不是继续原team-GAE PPO或放宽SafetyFilter。
6. **当前课程覆盖仍较窄。** P2 N6/N8/N12是总请求数，N12 pilot在空峰值仅6–7架，不能称12架并发或高负载验证。自然流首到达10秒、间隔固定60秒；SOC=.8、常规priority=1，尚未覆盖间隔扰动、priority/SOC异质性、20/40/60总请求、持续/突发流和合法静态几何泛化。当前课程按单项增难的主线合理，但离最终终端区场景仍有明确跨度。
7. **文档参数及运行交接需保持一致。** 需求文档仍有旧高度/速度数字；当前验证配置为三环高度100/70/40m、速度22/20/16m/s，不能静默切换。后续长任务按用户要求由`luna_worker`跟踪、主agent静默等待终态。Git无remote，代码仅经Syncthing同步，不能声称已推送。

## 模型、评估与证据边界

- 当前 P2 V4 最终候选：`p2_resume_v4_20260918/train/checkpoint_8.pt`，SHA256 `5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`。训练与 heldout hard gates PASS，但因 rule 效率差距尚未晋级 P3。
- 配对 baseline：`ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt`，SHA256 `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`。
- V4恢复起点：`p2_resume_v2_20260917/train/checkpoint_5.pt`，SHA256 `8a8466f61119f7f5b5f780249d26f218ee9b8ae3e4aededc5c7f5fe2e51b201b`。它是V2失败训练块中的最后合格中间模型，现仅作为V4 provenance，不是当前最终候选。
- V3 root/entry manifest SHA256：`2098af1c3109c17fc489b7c9f20a9db11eb67aa62b77b5a03beb2de66909cd7d`；KL诊断manifest：`41017e10f45105a6125213ac51a630bdcfda9c35bac196608e2affc891955f69`。结果位于输出根目录下 `p2_resume_v3_20260918` 与 `p2_v3_kl_diagnostic_20260918`。
- checkpoint24评估seeds710001–710006与P2最终评估755001–755006均 **已消费**，不得重跑或用于调参。V4 heldout manifest SHA256为 `167d22cb29d8ef6d08275cb0ff4cbe270c0d9005611a176b85d81fd64730f104`。
- WAIT/credit diagnostic seeds771001–771006亦 **已消费**；manifest SHA256为 `b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93`，终态FAIL不得重跑或升级。
- 入口 `entry/checkpoint_1.pt` 和Adam诊断快照均不能晋级正式模型；历史失败manifest不得重写。
- 服务器输出根为 `~/training_outputs/terminal`，本地镜像为 `D:/7_codespace/training_outputs/terminal`。源码只在本地修改，经Syncthing同步，服务器仅运行。

## 当前允许的下一步

1. 按已冻结协议执行smoke与fresh P2自然流Teacher-label-only support collection；train/dev/future qualification seed分开，旧D1只读绑定，771001–771006只作诊断参考不得训练。
2. 先证明N6/N8/N12自然流状态上的gate learnability、WAIT/RELEASE双类覆盖、三个interface覆盖与校准改善，再做CE/DAgger gate repair和student-only stochastic qualification。
3. qualification必须全部请求排空、安全/shield为0并显著降低false-WAIT；未通过前禁止继续原team-GAE PPO。若后续仍需PPO，必须先预注册local gate credit或监督锚点的单变量修复。
4. 新P2候选在fresh paired效果评估中显著缩小与rule的等待及request-to-touchdown差距后，才允许进入P3。
5. 通过P2效果审查后，按[执行计划8.5节](execution_plan.md)逐步引入间隔扰动、P3任务/资源异质性、P4交通压力、P5静态几何泛化；完成static qualification后才进入A6。
6. 后续长训练或评估由 `luna_worker` 持续跟踪并在终态通知主agent；主agent在运行期间保持静默，不创建heartbeat重复监控。

完整协议、旧失败与诊断详情见[持续课程证据](evidence/v4_2/continuous_curriculum_20260917.md)；初始准入见[准入结果](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，轨迹修复与第二训练块见[续训结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。

## 不应恢复的旧路线

- V3 factorized 三头策略；
- V4 30-way 与 V4.1 flat-26 actor；
- FULL-Teacher PPO 或用 Teacher corrected action 计算 requested-action PPO ratio；
- 87D V2 checkpoint 直接续训 88D V2.1；
- 两环、8D high-level、自由 `ψ/γ/V` 或姿态/电机 action；
- 把日期型 handoff 或单次 diagnostic 当作新的 canonical specification。
