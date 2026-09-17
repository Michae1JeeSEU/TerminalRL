# 项目当前状态

> **状态日期**：2026-09-18
>
> **代码快照**：分支 `codex/v4-2-v2-1-formal-i1`；最新实验诊断提交 `bc2f8ae`，V3 几何修复提交 `114a5e0`；KL 更新保护仍为未提交工作区改动
>
> **当前阶段**：Phase A / A5；初始课程训练及评估完成，P2 自然流续训在入口 KL gate 阻塞
>
> 本文是滚动状态页。规范语义以 [README.md](README.md) 中列出的主规范为准。

## 一句话状态

低层 V4.2 / Observation V2.1（88D）已完成 R4 → I1 V2/Q2 → Safe DAgger → I3 → 初始 PPO 准入，两个初始训练 block 和 checkpoint24 独立评估已完成。**当前尚未得到可认定满意的低层策略，P2 V3 的 `PPO_ALLOWED=false`，不能启动下一轮正式训练。**

初始 N3–N6 的 `PPO_ALLOWED=true` 是对应冻结版本及协议的历史准入，不自动授权修改生成器或 PPO kernel 后的当前工作区。P2 V1/V2 已累计 **28 次合格更新（23+5，原计划36次）**；V3 资格评估通过，但入口更新 KL 超限停止，剩余8次未执行。

2026-09-18 本次通过 SSH 核验：服务器没有运行中的 `python ... terminal_marl` 任务；V3 日志终行为 `{"stage":"entry","PASS":false}`，没有正式 `train/checkpoint_8.pt`。Luna max 子智能体因额度限制退出，**当前没有活跃监控**。历史 PASS/FAIL 均保留，完整结果见[持续课程证据](evidence/v4_2/continuous_curriculum_20260917.md)。

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
| KL 更新保护 | 工作区实现可选 target-KL/逐步回滚，服务器10项单测通过 | 尚未提交、未完成审查、未接入正式入口，不能视为已恢复准入 |
| P3–P5 / static qualification | 尚未执行 | 满意策略与 A6 高层动态空域均未达成 |

## 已识别的问题与处理状态

1. **奖励与等待目标冲突，效率尚未证明改善。** 旧 progress 奖励计入绕环距离：外环22m/s约每秒+0.022，高于时间惩罚-0.01。P2配置将该系数置0，物理、安全及其他奖励保持不变；这消除了已确认的绕环奖励激励，但是否改善等待仍待独立评估，不能仅凭训练完成判断收敛。
2. **初始动作概率检查存在批次浮点差异，已局部修复。** V1 seed754024 的误差为1.1444091796875e-5，超过1e-5。原action tape只读复放表明完整world批次误差0，actor子批次复现误差。检查器现保留原采样批次再mask，阈值和PPO行为不变；后续V2/V3初始ratio检查通过。
3. **轨迹节点检查漏掉连续线段内切，已局部修复并通过V3资格复核。** V2 seed757006实际内侵0.0592194m，超过0.05m。UID39见证曲线节点半径均合格，但线段最近点半径199.9275204m。修复使用每段投影最近点，仅超限旧曲线进入已有forward-annular fallback，保持有效旧曲线和原安全容限；此前38项定向测试通过。V3资格最大内侵0.0186652m，安全/shield/逆环向均0。
4. **当前主要阻塞是PPO更新漂移。** V3 entry seed763001，160步仅8次actor decision，10个minibatch更新后 `approx_kl=0.11257042959332467 > 0.05`，其余入口gate通过。该入口本来就是短窗口，0/6完成不是此次失败原因。原kernel只在完整更新之后由runner判定KL；其返回值是各minibatch更新前KL的均值，不能代表最终策略漂移。fresh诊断seed765002中，清零Adam的该均值为0.0342861，更新后full-rollout sampled KL却为0.0970130；恢复Adam分别为0.00201741/0.00530911。两新seed没有复现原失败，尚不能确定是原seed、稀疏actor样本、优势分布与Adam状态中的哪种交互导致，不能用换seed通过覆盖旧FAIL。
5. **KL保护只完成初步实现，验证尚未闭合。** `terminal_marl/algo/student_ppo_v4_2.py` 与 `tests/test_student_ppo_v4_2.py` 有未提交修改，另有 `ppo_target_kl_guard_diagnostic_v1.yaml`。实现意图是每次optimizer step后检查完整rollout requested-action KL，超过target 0.01时恢复模型与Adam并停止；原0.05 gate不放宽。2026-09-18核对本地/服务器文件SHA一致后，`pytest tests/test_student_ppo_v4_2.py -q` 为 **10 passed**。仍需审查异常处理、已有Adam状态回滚及零接受更新的报告语义，验证真实rollout成本与结果，并在新入口中拒绝零有效更新；当前不能把单测PASS等同于正式训练准入。
6. **当前课程覆盖仍较窄。** P2 N6/N8/N12是总请求数，N12 pilot在空峰值仅6–7架，不能称12架并发或高负载验证。自然流首到达10秒、间隔固定60秒；SOC=.8、常规priority=1，尚未覆盖间隔扰动、priority/SOC异质性、20/40/60总请求、持续/突发流和合法静态几何泛化。当前课程按单项增难的主线合理，但离最终终端区场景仍有明确跨度。
7. **文档参数及运行交接需保持一致。** 需求文档仍有旧高度/速度数字；当前验证配置为三环高度100/70/40m、速度22/20/16m/s，不能静默切换。Luna max已因额度限制退出；恢复长任务前需重新建立可用监控。Git无remote，代码仅经Syncthing同步，不能声称已推送。

## 模型、评估与证据边界

- 最近完整通过训练块及独立评估的模型：`ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt`，SHA256 `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`。这是历史版本证据，不能不经验证直接外推到当前代码。
- P2最近合格更新产物：`p2_resume_v2_20260917/train/checkpoint_5.pt`，SHA256 `8a8466f61119f7f5b5f780249d26f218ee9b8ae3e4aededc5c7f5fe2e51b201b`。它是失败训练块中的最后合格中间模型，可作为新协议候选恢复点，不是P2最终合格策略。
- V3 root/entry manifest SHA256：`2098af1c3109c17fc489b7c9f20a9db11eb67aa62b77b5a03beb2de66909cd7d`；KL诊断manifest：`41017e10f45105a6125213ac51a630bdcfda9c35bac196608e2affc891955f69`。结果位于输出根目录下 `p2_resume_v3_20260918` 与 `p2_v3_kl_diagnostic_20260918`。
- checkpoint24评估seeds710001–710006 **已消费**；P2最终评估755001–755006 **仍未消费**。V3未产出checkpoint8，不能运行当前最终评估入口，也不能按heldout效果挑选中间模型。
- 入口 `entry/checkpoint_1.pt` 和Adam诊断快照均不能晋级正式模型；历史失败manifest不得重写。
- 服务器输出根为 `~/training_outputs/terminal`，本地镜像为 `D:/7_codespace/training_outputs/terminal`。源码只在本地修改，经Syncthing同步，服务器仅运行。

## 当前允许的下一步

1. 审查并完成KL保护的针对性测试及小样本输出验证；保留原approx-KL门槛、student-only requested-action ratio和Adam恢复纪律，不通过清零优化器或换seed绕过失败。
2. 完成代码审查、测试及本地提交后，预注册新修复协议、fresh seeds、预算、源码哈希和有效更新要求；从最后合格P2 checkpoint5恢复，重新做student-only资格与入口验证，全部PASS后才补足剩余8次合格更新。旧V3协议和FAIL不变。
3. 新训练块完整通过后，明确绑定最终checkpoint，再执行保留755001–755006的一次性candidate/baseline/rule评估；报告等待、请求至touchdown、吞吐、安全、shield、在空/队列峰值。硬gate通过但效率未改善时，继续诊断，不能直接升课程。
4. 通过P2效果审查后，按[执行计划8.5节](execution_plan.md)逐步引入间隔扰动、P3任务/资源异质性、P4交通压力、P5静态几何泛化；完成static qualification后才进入A6。
5. 新服务器长任务启动后先检查进程、日志与中间产物，再按用户要求交Luna max监控；当前该模型额度不足，须明确说明监控可用性，不能把已退出的子智能体记为正在监控。

完整协议、旧失败与诊断详情见[持续课程证据](evidence/v4_2/continuous_curriculum_20260917.md)；初始准入见[准入结果](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，轨迹修复与第二训练块见[续训结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。

## 不应恢复的旧路线

- V3 factorized 三头策略；
- V4 30-way 与 V4.1 flat-26 actor；
- FULL-Teacher PPO 或用 Teacher corrected action 计算 requested-action PPO ratio；
- 87D V2 checkpoint 直接续训 88D V2.1；
- 两环、8D high-level、自由 `ψ/γ/V` 或姿态/电机 action；
- 把日期型 handoff 或单次 diagnostic 当作新的 canonical specification。
