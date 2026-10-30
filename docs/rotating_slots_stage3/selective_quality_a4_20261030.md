# A4 质量诊断与共同已运行初态：正式配对、版本和证据

本轮同 common 的两个开发配对中，K8在正常落地、末态pending和能量上优于FULL_H，但medium holding反增138.559 aircraft-s。四臂补链机制均未补成完整链，high K8仍有15个空前缀任务；改善没有消除候选供给与有效推进不足。**本轮阶段A的授权工作已完成，保持阶段A，不进入阶段B。**

更新日期：2026-10-30。四个正式臂均 `COMPLETE`，保存回放审计均 `PASS`，服务器child-wall已结算。证据限于两个冻结根各一次开发配对，不主张泛化，也不直接与旧冷启动结果比较。关键帧已作 `AI_SAVED_FRAME_REVIEW`；连续人工回放仍为 `HUMAN_PENDING`。数值实现 `b80ca3aee7d7ba6495640f4a3b02f72b026fabbb` 与可视化 `61f9ae0ef3b5794ce2275632aff9e934c67e804b` 均已提交并推送；本报告与汇总helpers将纳入本次交付提交，最终hash由主对话交付说明记录。

本轮四项交付入口如下：

1. **明确参考版本**：`opportunity_reference_v4`，三开关均false；[搜索开关代码](../../terminal_marl/core/rotating_slots/selective_replanning/policy.py)、[正式source freeze](../../artifacts/rs_select_quality_a4_20261030/comparison_source_freeze_exportfix1.json)与本文版本约束共同记录。
2. **质量差距诊断**：本文“质量诊断输入与单因素清单”“已审核的R12单状态短续”“仍存在的有界质量差距”记录输入、认证与解释边界；[原诊断输出目录](D:/7_codespace/training_outputs/terminal/rs_selective_quality_a4_20261030/)保留R1/R4/R12/P2、fixed_groups与diagnostic_continuations。
3. **经检查的候选与续行评价修正**：[下游评分](../../terminal_marl/core/rotating_slots/stage2_downstream.py)、[候选callback](../../terminal_marl/core/rotating_slots/stage2_planning.py)、[局部渐进轮次](../../terminal_marl/core/rotating_slots/selective_replanning/local_problem.py)、[针对性测试](../../tests/test_stitched_continuation_v1.py)与本文实现/测试登记。正式补链完整候选数为0，效果改善不归因于补链成功。
4. **可复用共同已运行初态**：[medium checkpoint](D:/7_codespace/training_outputs/terminal/rs_selective_quality_a4_20261030_review/common_initial/s3r23/common.checkpoint.json)、[high checkpoint](D:/7_codespace/training_outputs/terminal/rs_selective_quality_a4_20261030_review/common_initial/s3r24/common.checkpoint.json)、[初始化入口](../../scripts/stage3/run_selective_common_initial_a4.py)及[复用比较入口](../../scripts/stage3/run_selective_common_comparison_a4.py)；SHA、t0与继承FINAL见本文。

附加验证证据：[正式四臂结果与审计镜像](D:/7_codespace/training_outputs/terminal/rs_selective_quality_a4_20261030_review/arms/)、[当前数值分析目录](../../artifacts/rs_select_quality_a4_20261030/analysis_final_timelinefix/)、[补充提取](../../artifacts/rs_select_quality_a4_20261030/analysis_final/supplement.json)、[正式回放图像manifest](../../artifacts/rs_select_quality_a4_20261030/visual_review_formal/manifest.json)。

## 本轮范围与版本约束

本轮范围为阶段 A 的有界质量诊断、必要的评分接口准备，以及两个冻结根上 FULL_H / RULE_LOCAL_K8 的共同已运行初态配对比较。先由质量诊断决定是否及如何修改评分，再固定双方相同的评分和截止条件。没有增加 seed、训练模型、进入阶段 B/C、扫描 lead 或改变几何、能量、安全间隔和着陆物理协议的授权扩展。

`opportunity_reference_v4` 表示 **v2 策略加必要的 v3 工程修复**。它保留 v2 的渐进组织和必要的接口、冻结状态、序列化及边界处置兼容；不默认启用 v3 的三个搜索策略开关。

| 控制项 | 本轮固定值 |
|---|---|
| reference variant | `opportunity_reference_v4` |
| `pad_anchor_priority` | `false` |
| `target_coverage_order` | `false` |
| `ordinary_immediate_filter` | `false` |
| 方法 | `FULL_H`、`RULE_LOCAL_K8` |
| 局部名义 K / max K | 8 / 12；显式 overflow/full fallback 单列 |
| online 组织 / restart | `progressive_v1` / `coalesced_restart_v2` |
| lead / heartbeat | 20 s / 5 s |
| preparation margin / ready offset | 1 s / 19 s |
| search handoff reserve / search offset | 4 s / 15 s |
| 正式 offset | 0；诊断 offset 不自动带入正式臂 |
| 新 score | 显式 `stitched_continuation_v1`；默认仍为 `legacy_window_tail_v8` |

旧 v2/v3 结果、checkpoint、账本和报告均只读。旧结果用于定位诊断输入与说明历史，不直接和新共同初态的增量结果比较效果。正式 comparison 的 `evaluation_profile` 明确、非空且双方相同；四个正式比较臂均显式使用 `stitched_continuation_v1`。

代码入口：[搜索开关](../../terminal_marl/core/rotating_slots/selective_replanning/policy.py)、[normal callback](../../terminal_marl/core/rotating_slots/stage2_planning.py)、[诊断入口](../../scripts/stage3/run_selective_quality_gap_a4.py)、[详细诊断协议](quality_gap_a4_protocol.md)。

## 共同已运行初态的预声明

| 根 | 冻结 seed | 原始场景 |
|---|---:|---|
| `s3r23` | 7324 | `medium24_rate150_v1` |
| `s3r24` | 7325 | `stress36_rate180_v1` |

每个根从原始 t=0 checkpoint 启动一次公共 initializer。initializer 固定使用 RULE_LOCAL_K8、reference_v4、三个开关 false、v2 组织和旧 `downstream_service_v8`。后续对方法评分的修改不追溯改变 initializer；实际计算、准备、公共处置和物理 advance 继续计时、计费，不能暂停物理过程赠送计算时间。

截取切面预先固定为第一次**非空普通 normal task 已 COMMITTED** 的边界：该时刻应启动的动作已经 `_start_due`；资源及必要 public 处置已经完成；当前 dispatch 的 timing、CPU、frame 和 pending bookkeeping 已清算；wrapper `pending=None`；下一普通任务尚未启动。源绝对时间最多推进到 60 s。没有到达该边界就记录 `INIT_NOT_REACHED`，不更换 initializer、不挑选更有利初态、不使用 fallback，也不自动重跑。

取得的 t0 由真实边界决定，可以是 20 s，不能强制归零或截成预设整数。保存完整 world、原 wrapper 与执行 history，以及 initializer records。原已承诺 FINAL 的 UID、trajectory hash、结束时间和 pad 单列记录。

两个方法从同一 SHA 绑定的 common checkpoint 克隆。SOC、peak、累计能量和 executed ledger、时间、active/soft、slots/pads、承诺、历史 frames/events/trajectories 及真实执行 history 均保留。物理 world 内已有缓存不清零；callback/provider/reference/template/validation/global 加速缓存和搜索 blockers 从冷状态开始，彼此不共享算法成果。旧 callback/tasks/services 仅作 provenance，新方法 records 从空数组开始。

先保存 t0 baseline，再于同一绝对 t0 显式 `resume_common` 消费继承的 restart/service 队列，不能推迟到 t0+5。双方运行区间均为 `[t0,t0+300]`，按相对 5 s 边界保存。累计能量不改写；质量计数和能量报告使用 t0 后增量，pending 使用末态快照，同时保留末态完整 quality 检查守恒。由 initializer 已承诺 FINAL 在 t0 后完成的 touchdown 单列为 inherited，不能计作新方法发起的 FINAL 成果。

实现：[共同初态入口](../../scripts/stage3/run_selective_common_initial_a4.py)、[world clean-boundary hook](../../terminal_marl/core/rotating_slots/world.py)、[延迟与外部 predecessor](../../terminal_marl/core/rotating_slots/planning_latency.py)、[单臂比较入口](../../scripts/stage3/run_selective_common_comparison_a4.py)。

### 实际共同初态

两个 initializer 均取得 `COMMON_INITIAL_READY`，真实截面为 **t0=20 s**；各有一个普通任务已 `COMMITTED`，`pending=false`，下一普通任务尚未启动。共同初态来自 `b6590c8a5ede25b5f63e89834dd4cd5ec36a646d`，仍使用旧 `downstream_service_v8`。两方法算法加速缓存从相同冷状态开始，完整物理账本不清零；正式统计窗口为绝对 **20→320 s** 的增量。

| 根 | common checkpoint SHA256 | initializer wall (s) | t0 已承诺 FINAL 数 |
|---|---|---:|---:|
| s3r23 | `0f67e40052c6d80ba35efebaf1e47fb113a1aca7046332d20b0c44f8cc29efc7` | 13.944724339991808 | 5 |
| s3r24 | `255f3ce5a3c45b01d3fb59ce5069ce39dec73b420735759ecdd6ce510f5eda77` | 18.80863841797691 | 3 |

以下 FINAL 均在绝对 **20 s** 开始，来自共同 initializer，后续 touchdown 归为 inherited，不能计作新方法发起的 FINAL。UID 的 `Ixx` 在本表精确表示 `initial_inner_xx`。

| 根 / UID | pad | 结束时间 (s) | trajectory hash |
|---|---:|---:|---|
| s3r23 / I01 | 5 | 68.5 | `44cade18ac611f5309a9c916a80564f27a6ec420440675b5f68b8dd35eeb36f1` |
| s3r23 / I02 | 0 | 68.5 | `1aa7005af7f2be7a39fbc0bf5b11c01375564ca92128c85d9d6d59b924b8e372` |
| s3r23 / I04 | 1 | 73.5 | `f84a73d92b05ddb84ac453a2de1df16e304f2ed207a54f0a5c6b0441f2e5e764` |
| s3r23 / I05 | 2 | 68.5 | `c7a342b07433a119e3a4fc6165a26229f80805933637dc321fb23a9f3c763a88` |
| s3r23 / I07 | 3 | 73.5 | `9e3cd3d302fe893dc418c3af2c174cfe32e2d3e2d7d6848e38631ed624c9f59a` |
| s3r24 / I02 | 5 | 73.5 | `3a7fe26e3cb3923cf305e541f9c6a24724d3bfa4ca7faf6e2ad90a2d192d9a1b` |
| s3r24 / I06 | 2 | 68.5 | `53e5647f64d7a4314cdfc39c1246f3a780e0f8c48ce428be87a91d680204e2c4` |
| s3r24 / I08 | 3 | 68.5 | `e13db4bfc92d1eddf4338b6931e7dc2d499349f10df952fcac2d797bcbbb83b6` |

证据为本轮输出根的 `common_initial/<root>/boundary_summary.json`、`common.checkpoint.json`、`initializer_tasks.json`、`initializer_services.json` 和 `initializer_callback_records.json`；原物理历史及共同初态文件保留。

## 质量诊断输入与单因素清单

四个输入来自旧 v2 高流量根 `s3r24` 的保存状态。下表区分历史 normal source time 和实际恢复边界；后三个是邻态诊断，不能写成原 warm task 输入的精确 replay。

| Case | 历史 source time (s) | 保存/恢复边界 (s) | 新 lead20 effective time (s) | 标签 | 定向对象 |
|---|---:|---:|---:|---|---|
| R1 | 0 | 0，checkpoint 0000 | 20 | `SAME_TIME_BOUNDARY_NOT_ORIGINAL_INPUT` | `initial_inner_02`，P0/P1 |
| R4 | 60 | 65，checkpoint 0065 | 85 | `DIFFERENT_TIME_BOUNDARY`，邻态 | `initial_middle_02` |
| R12 | 206.03508481464814 | 210，checkpoint 0210 | 230 | `DIFFERENT_TIME_BOUNDARY`，邻态 | `initial_middle_01` |
| P2 | 233.65165837993845 | 235，checkpoint 0235 | 255 | `DIFFERENT_TIME_BOUNDARY`，邻态 | 实际选择的 partial 与 frozen virtual pad2 关系 |

共同 baseline 使用恢复后的真实 source、cold provider 和保存的 selector history，冻结 catalog、scope、global frame、实际选择与 callback reference state。离线恢复和候选构建成本单列，不能充作 online 性能证据。

| Factor | Cases | 固定输入 | 唯一变化/用途 | 结果 |
|---|---|---|---|---|
| `baseline` | 四个 case | SHA 绑定旧 checkpoint | 当前旧分数 callback 一次，保存实际有界目录 | 四个 `result.json` 均为 COMPLETE |
| `augment` | 四个 case | baseline archive | 最多 6 s 的定向新列及验证；不改 scope/几何/分数 | R1/R4/R12/P2 新增列数为 0/0/1/0 |
| `catalog_cp8` | 四个 case | 原 catalog/frame | 固定目录 solver8 控制 | R1/R4/R12/P2 prefix 数为 3/1/0/1，预测 L 为 5/7/5/5 |
| `augment_cp8` | 四个 case | augmented catalog、同 frame | 相对控制仅增加候选，solver8 | 已保存 R12 结果：prefix 1，预测 L6；未用无新增列 case 伪造增强效果 |
| `catalog_cp24` | R1、P2 | 与 cp8 相同的原候选 hash | 仅 solver allowance 8→24 s | 两 case 的五维目标均与各自 cp8 完全相同 |
| `offset1`、`offset2` | P2 | 原 projected state/frame、保存 history | 仅 selector offset；不自动扩展到正式臂 | offset1 CPU 前置失败保留；offset2 首/末组与 baseline 相同 |

R1 只尝试目标 UID 的 P0/P1，每 pad 最多保留一列，总计最多两列。R4/R12 保留最小 hash 的既有 immediate partial 前缀并调用既有 reference suffix 生成器补完整链；P2 的 partial 选择及 fallback 路径显式记录，每 case 最多补一条完整链。单列 SAFE 与最终组合/可提交 prefix 的认证分开；有界目录缺失、UNKNOWN 或补全未取得候选，均不能宣称物理不可行。目录比较只在冻结目录内解释，原目录与增强目录使用相同验证/search allowance；不把未认证的 objective 当成改进。

120 s 短续仅允许 R4/R12，先审核同态 baseline choice / forced partial 配对及其新 batch/prefix 认证，再显式启动单个 branch；不扫描 UID/case，不由诊断入口自动启动。短续恢复和离线预验证不并入 online 计算收益。[短续入口](../../scripts/stage3/run_selective_short_continuation_a4.py)的 R12 两 branch 已运行并审核，事实登记如下。R4 已完成 prepare，结果为 `NOT_APPLICABLE / NO_FIXED_UID_IMMEDIATE_PARTIAL`；没有运行 R4 短续，也不会补选其他 UID。

P2 的 offset2 实际首组选组、末组选组均与 baseline 相同，不能算多组比较。后续三个固定组诊断已运行，下面分别登记实际组与认证预测；这些结果没有物理推进或实际提交。

### 已保存的补链与固定目录结果

本节直接读取镜像 `D:/7_codespace/training_outputs/terminal/rs_selective_quality_a4_20261030` 的对应 `result.json`、`augmentation.json` 和 `diagnostic.json`。R12 对 `initial_middle_01` 的既有立即 partial 补出一条完整链；原 prefix 保持不变，新增完整候选 hash 为 `e8d73c1d498fd64d80e66103f2befc0feaa4222d04d63b33b2355094e2fd0537`。R1/P2 在各自有界尝试中未观察到新列；R4 为 `NO_EXISTING_IMMEDIATE_PARTIAL`。这些缺失不作物理不可行结论。

| Case / factor | prefix 数 | 认证预测 L | 五维目标 `(F,-L,risk,E,tau)` | 认证 |
|---|---:|---:|---|---|
| R12 原 catalog CP8 | 0 | 5 | `(0,-5,17076.11472222222,815336563.4000001,14517.5)` | true |
| R12 补链 catalog CP8 | 1 | 6 | `(0,-6,13809.740944444446,782140058.1,13970.0)` | true |
| R1 原 catalog CP8 / CP24 | 3 / 3 | 5 / 5 | 两者均为 `(0,-5,6160.709972222223,1085969989.8,17238.0)` | 两者 true |
| P2 原 catalog CP8 / CP24 | 1 / 1 | 5 / 5 | 两者均为 `(0,-5,24537.67486111111,827662021.3000001,14663.0)` | 两者 true |

R1 的 cp8/cp24 candidate hash 均为 `b4ca04344d06d6ff2ad55a307fa7083d56cf8533d2db8cb3f5f1864604c5cce6`；P2 均为 `f446a173efabbfa156826e4ed3ae0fcb4c7c3152a1d0e3c5d5aefc775e6bdb9e`。这里的 L 是候选组合的认证预测，不是实际落地数；未执行发布。固定目录增加求解 allowance 的这两个观测没有改变目标，不外推为所有状态都无需更多求解时间。

### P2 三个固定组

三组均使用 source 235 s / effective 255 s、同一 frozen frame SHA `6370d3517f7b8bd43361582ffd77ce57dc3f5154ce56104bbcf00a94d9a2f574`、同一旧评分与候选6/CP8/hint3、max32/1024 额度；每组均有两次 solve 和一次 scope expansion。下表 `Mxx` 精确表示 `initial_middle_xx`，`Oxx` 精确表示 `initial_outer_xx`。

| 固定组 | 实际首组（8 UID） | 最终组在首组上增加的 UID / 最终 K | prefix UID / 数 | 认证预测 L | 五维目标 |
|---|---|---|---|---:|---|
| control | M01,M02,O01,O03,O04,O05,O08,O10 | M03,O02 / 10 | O01 / 1 | 5 | `(0,-5,24537.67486111111,827662021.3000001,14663.0)` |
| alternative_a | M01,M02,M03,M05,O01,O03,O05,O10 | O02 / 9 | O01,M05 / 2 | 6 | `(0,-6,25191.843333333334,821551692.5,14456.7)` |
| alternative_b | M01,M02,M07,M08,O01,O03,O05,O10 | O02 / 9 | O01 / 1 | 6 | `(0,-6,22830.005083333337,813049074.8000001,14456.7)` |

三份保存结果均为 `COMPLETE`、`certified=true`、`physical_advance=false`、`actual_submitted=0`、`publication_permitted=false`。三种首组和三种最终组分别不同，说明此开发状态下选组仍有差距；它们是同 frame/额度的认证预测比较，不能写成多完成一次真实落地。证据目录为 `fixed_groups/P2/{control,alternative_a,alternative_b}`，[固定组入口](../../scripts/stage3/run_selective_fixed_groups_a4.py)保持原预算和候选过程。

## 已审核的 R12 单状态短续

此处为开发状态配对，绝对时间 **210→330 s**，两 branch 使用同一恢复边界。下表 landing、energy、holding 是 `terminal.increments` 的 120 s 窗口增量；pending 是绝对 330 s 的末态快照。

| 观测项 | `baseline_choice` | `forced_partial` |
|---|---:|---:|
| branch child wall (s) | 174.60019148804713 | 159.95895879506133 |
| 窗口新增正常落地 | 4 | 5 |
| 窗口新增应急落地 | 0 | 0 |
| 窗口新增 failure | 0 | 0 |
| 窗口能量 (J) | 299338222.96275854 | 300452199.9630679 |
| 窗口 holding (aircraft-s) | 3678.411334311706 | 3509.066657240406 |
| 末态 airborne pending | 34 | 33 |

首个干预的真实交付记录中，baseline 的 `valid_baseline_wait=true`；forced 实际提交 trajectory hash `3ca79306b5ce35cdc9231c2f93eab49844266ef361614d314819a1930af6518f`。forced 新增的正常落地 UID 为 `initial_middle_01`，touchdown 绝对时间为 **326.6068488471304 s**。其余四个落地两边相同，均来自源状态已经承诺的动作，不能归功于本次干预。

| 保存回放审计 | `baseline_choice` | `forced_partial` |
|---|---|---|
| audit status / full A20 | `PASS` / `true` | `PASS` / `true` |
| 连续间隔 | `CERTIFIED` | `CERTIFIED` |
| resource / C2 / PVA / proposal nonexecution | 均通过 | 均通过 |
| audit child wall (s) | 9.926984342979267 | 10.582467736094259 |
| 人工回放 | `HUMAN_PENDING` | `HUMAN_PENDING` |

主对话已审核的限定解释：在这一个开发状态配对中，安全且旧分数更差的推进多完成一次正常落地，同时多耗约 **1.114 MJ**、holding 减少约 **169.345 aircraft-s**；这支持续行评价可能存在误判，不能推出推进总是比等待更好。结合补链和固定组选组诊断，主对话决定重点修正有界补链和连续续行评分，不调权重或增加求解预算。四个正式共同初态比较臂已完成，其同 common 的限定解释见下文。

主对话另外查看了 production ReplayViewer 离线导出的六个关键画面：两臂 t270/t326.7 的 top_down 和 t310 的 side。t310 baseline 的 P2 为 FREE，forced 的 P2 为 RESERVED UID14，显示 flight 278.107 / descent 313.107 / touchdown 326.607；t326.7 forced 的 P2 为 SERVICE UID14，余约29.9 s，而 baseline 仍为 FREE。这与日志中的新增 `initial_middle_01` 落地相符。状态为 **`AI_SAVED_FRAME_REVIEW`（6关键帧）**；已导出24图，见 [图像 manifest](../../artifacts/rs_select_quality_a4_20261030/visual_review_short/manifest.json)。这不是连续整段人工播放，也不是安全证明；`HUMAN_PENDING` 不变，正式四臂的回放审查见下文。

## 新评分与有界补链实现

新评价 profile 为显式 `stitched_continuation_v1`；默认仍为 `legacy_window_tail_v8`。initializer 明确使用旧 profile，历史输出不重算；reference 仍为 `opportunity_reference_v4`，三个开关均 false。新实现见 [下游评分](../../terminal_marl/core/rotating_slots/stage2_downstream.py)、[候选与 callback](../../terminal_marl/core/rotating_slots/stage2_planning.py)、[局部渐进轮次](../../terminal_marl/core/rotating_slots/selective_replanning/local_problem.py)及 [针对性测试](../../tests/test_stitched_continuation_v1.py)。

连续续行估计按同一时间线衔接以下四项，每项计一次：已知 prefix（包括其中真实稳定等待）的能耗；prefix 结束到下一可调整机会的等待；后续运动估计；共享下游队列等待。不会在未完成 partial 后先填满旧240 s窗口、再重复加续行等待。剩余运动来自可用参考或明确标记的估计；共享队列沿冻结 frame 计算。估计不会新增真实 reservation，也不是 trajectory、安全或执行认证；partial 本身不计入 L。

新 profile 的 F 估计依据为已知 prefix 峰值违规，或者**包含共享 queue 等待的估计总能耗大于等于当前 SOC 对应能量**。Python 评价和 CP expressions 使用相同整数能量量化、同一个 `>=` 阈值；F 仍是估计，不是实际物理失败计数。旧 FH 窗口失败估计仅保留在 legacy profile。真实物理评价目标、SOC/peak 执行账本、安全校验与 final authorization 没有改变。

补链在第一次 CP 之后，从既有 immediately executable partial 中最多冻结两架飞机，每架仅一个 partial 并保留其原 prefix。后续 suffix 生成、回放与单列验证都消耗原 candidate **6 s**、CP **8 s**、hint **3 s**、search **15 s** 和 final handoff **4 s** 额度；每机 max32、全局 max1024 attempts 不增加，不另开求解额度。普通 non-progressive 入口只使用新评分，不启用这项补链，并显式记录 `suffix_completion.status=NOT_ENABLED_NON_PROGRESSIVE`。这些实现和测试不替代四个正式臂的评价。

`a2f06f5f385d084607b8ed92fa0b059f8557ae97` 将原来的 per-plane priority 改为 **global bounded completion micro-round**：第一次 CP 后，冻结对象的补链切片先于普通 candidate growth 消费剩余额度，避免轮到目标飞机之前额度已被普通 growth 用尽。最多两架、每架一个原 partial 及全部原额度不变；不足以取得候选的尝试仍如实记录。对应有界 probe 保存在 `quality_verify_priorityfix/R12`，不是正式 rollout。

`b80ca3aee7d7ba6495640f4a3b02f72b026fabbb` 仅把 comparison record export 改为现有 `replay._plain` 视图，避免 live `Section` 经 `deepcopy` 递归；不改变算法、候选、评分、物理状态或执行记录。四份归档 checkpoint 的导出验证均为 `ALL_FOUR_RECORD_EXPORTS_VERIFIED`，`planning_calls=0`、`world_advance_calls=0`，输入 bytes 与 state/history/trajectories digest 不变。归档 record 已被既有保存逻辑规范化，旧 deepcopy 在这些归档上可成功；另一个真实 `GlobalEvaluationFrame.downstream -> JointDownstream.evaluate FOLLOW` fixture 才复现 live `Section` 的 `RecursionError`，并验证新导出路径。证据为 [serialization summary](../../artifacts/rs_select_quality_a4_20261030/serialization_fix_check/summary.json) 和 [live FOLLOW fixture](../../artifacts/rs_select_quality_a4_20261030/serialization_fix_check/live_follow_fixture_verification.json)。

## 预算、输出与四臂配对

本轮输出根为 `~/training_outputs/terminal/rs_selective_quality_a4_20261030`。新 shared child-wall 上限为 **7200 s**，诊断、共同初始化、短续、正式臂、测试/审计等实际子进程成本共同扣除。新账本不重置旧账本，也不抵扣旧消耗；历史 `1374.067592 s` 只读引用。失败和超限尝试保留并计费，不覆盖原输出、不自动 retry。

四臂审计后的 settled child wall 为 **4773.427772170049 s / 7200 s**，账本无 active job。这是服务器 child-wall 计费，不包含全部本地执行或 agent 工时。审计前 4720.513219708926 s、前期 730.7542084540473 s 和 686.3068904069951 s 仅为较早快照，不累计叠加，不把重叠测试数或近似耗时机械相加。

| 项目 | 单 child cap | 来源/成本边界 |
|---|---:|---|
| 单个离线诊断 factor | 300 s | `OFFLINE_DIAGNOSTIC`，不是 online rollout |
| 单根 common initialization | 300 s | 包括恢复、设置、真实计算与物理运行 |
| 单个正式比较臂 | 1200 s | 从 restore 开始计 episode 成本，单列 setup、t0 resume/advance、输出 I/O |
| 单次保存回放审计 | 300 s | 单独 watchdog/账本 job，直接复用原 `audit_job` |
| 单个短续 branch | 300 s | 显式运行，和其离线准备分账 |

诊断使用 `a4_source_hash`；common 使用 `common_initial_source_hash`；正式比较使用 `comparison_source_hash`，三者不能互相覆盖。每次运行验证自身 source freeze、输入 SHA、active parent watchdog、显式 CPU 与原始 launch thread 设置。代码只经本地 Syncthing 同步；服务器输出置于 training_outputs。

| 根 / seed | FULL_H | RULE_LOCAL_K8 | 公共条件 |
|---|---|---|---|
| s3r23 / 7324 | COMPLETE；child wall 880.685717354063 s | COMPLETE；child wall 746.3502275579376 s | 同一根 common checkpoint、同 score/deadline、绝对20→320 s |
| s3r24 / 7325 | COMPLETE；child wall 1049.1584014060209 s | COMPLETE；child wall 1058.5091095649404 s | 同一根 common checkpoint、同 score/deadline、绝对20→320 s |

四臂使用 `b80ca3aee7d7ba6495640f4a3b02f72b026fabbb` 的 export-fix freeze，`comparison_source_hash=1fd5c9a19919891b75f2158b42e291e63d8ef9f9f41400b8e9f2cbd487f7ed03`，见 [comparison freeze](../../artifacts/rs_select_quality_a4_20261030/comparison_source_freeze_exportfix1.json)。结果镜像位于 `D:/7_codespace/training_outputs/terminal/rs_selective_quality_a4_20261030_review`，当前使用的 timeline-fix 汇总为 `COMPLETE`；[summary.json](../../artifacts/rs_select_quality_a4_20261030/analysis_final_timelinefix/summary.json)、[four_arm_summary.csv](../../artifacts/rs_select_quality_a4_20261030/analysis_final_timelinefix/four_arm_summary.csv)、[callback_timings.csv](../../artifacts/rs_select_quality_a4_20261030/analysis_final_timelinefix/callback_timings.csv) 和 [task_timeline.csv](../../artifacts/rs_select_quality_a4_20261030/analysis_final_timelinefix/task_timeline.csv) 保留原精度与逐记录数据。四臂的 common SHA、profile、窗口与终态时间检查均通过，callback/task matching 无缺失或冲突。原 `analysis_final` 保留，不覆盖。

### 20→320 s 正式窗口结果

正常落地、能量与 holding 为 t0 后增量；pending 为绝对320 s末态。四臂 emergency landing、energy failure、failed before landing 均为 **0**。继承 FINAL 与新方法后续发起 FINAL 的实际 touchdown 分列，防止把 initializer 的成果归给方法。

| 观测项 | medium FULL_H | medium K8 | high FULL_H | high K8 |
|---|---:|---:|---:|---:|
| 正常落地 | 10 | 13 | 4 | 10 |
| 其中 inherited FINAL touchdown | 5 | 5 | 3 | 3 |
| 其中后续发起 FINAL touchdown | 5 | 8 | 1 | 7 |
| 末态 airborne pending | 32 | 29 | 45 | 39 |
| 末态 upstream pending | 0 | 0 | 0 | 0 |
| 窗口能量 (MJ) | 627.409 | 557.933 | 905.947 | 811.280 |
| 窗口 holding (aircraft-s) | 5690.182 | 5828.741 | 11374.558 | 10040.716 |
| 全 normal planning work (s，含末个 pending 已发生工作) | 195.687 | 180.065 | 254.268 | 250.300 |

主对话审核的限定解释：在本次同 common、新版本内，K8 在 medium/high 分别多完成3/6次正常落地，末态 airborne pending 分别少3/6架，窗口能量分别较低。medium 的 holding **增加138.559 aircraft-s**，不能写成全部质量指标改善；high holding 较低。两个根各一次开发配对不能直接与旧冷启动结果比较，也不能泛化为 K8 总是更优。

| 根 / 方法 / 环 | holding (aircraft-s) | holding 能量 (MJ) | 按 source ring 记账的执行总能量 (MJ) |
|---|---:|---:|---:|
| s3r23 / FULL_H / inner | 2284.476 | 211.938 | 264.378 |
| s3r23 / FULL_H / middle | 2199.411 | 164.544 | 204.760 |
| s3r23 / FULL_H / outer | 1206.295 | 82.122 | 119.282 |
| s3r23 / K8 / inner | 497.515 | 46.156 | 115.986 |
| s3r23 / K8 / middle | 1515.353 | 113.368 | 134.521 |
| s3r23 / K8 / outer | 3815.873 | 259.777 | 266.544 |
| s3r24 / FULL_H / inner | 1845.075 | 171.173 | 194.566 |
| s3r24 / FULL_H / middle | 4029.850 | 301.485 | 301.485 |
| s3r24 / FULL_H / outer | 5499.633 | 374.404 | 379.346 |
| s3r24 / K8 / inner | 514.386 | 47.721 | 101.135 |
| s3r24 / K8 / middle | 3824.920 | 286.153 | 286.153 |
| s3r24 / K8 / outer | 5701.411 | 388.141 | 393.083 |

分环值沿用执行 ledger 的 source-ring 字段，三环之和不覆盖全部窗口能量；它不是总E的完整分解，不据差额推断具体来源。holding 能量是执行能量的组成部分，不能再与执行总能量相加。

终态 history 在绝对320 s解码后只查询 `ages()`，未调用 `observe()`，未改保存历史。已知年龄与未知值分开；unknown 不填0。

| 终态 history 项 | medium FULL_H | medium K8 | high FULL_H | high K8 |
|---|---:|---:|---:|---:|
| 已知 review_age ≥60 s 数 | 21 | 14 | 12 | 12 |
| review_age unknown 数 | 7 | 10 | 26 | 26 |
| 最大已知 review_age (s) | 264.575076 | 264.575076 | 311.976610 | 311.976610 |
| 已知 progress_age ≥60 s 数 | 24 | 15 | 14 | 14 |
| progress_age unknown 数 | 2 | 6 | 30 | 24 |
| 最大已知 progress_age (s) | 280.000000 | 264.575076 | 311.976610 | 311.976610 |

### 任务交付、分母与计算成本

K8 两臂任务均为 `LOCAL`，medium 实际 K 恒为8，high 为8–10；没有 overflow/full fallback。FULL_H 的实际 K 分别为19–28和33–45。下表 pending 各一个，均未当作完整已结束任务计入完整任务均值。

| 观测项 | medium FULL_H | medium K8 | high FULL_H | high K8 |
|---|---:|---:|---:|---:|
| task 总数 / 已结束 / pending | 13 / 12 / 1 | 15 / 14 / 1 | 17 / 16 / 1 | 22 / 21 / 1 |
| COMMITTED | 9 | 10 | 1 | 4 |
| NO_EXECUTABLE_PREFIX | 0 | 0 | 5 | 15 |
| REJECTED | 3 | 4 | 10 | 2 |
| 已有非空 group 接入率 | 9/12 = 75.00% | 10/14 = 71.43% | 1/5 = 20.00% | 4/5 = 80.00% |
| 全部已结束 task 接入率 | 9/12 = 75.00% | 10/14 = 71.43% | 1/16 = 6.25% | 4/21 = 19.05% |
| 实际提交 plan 数 | 34 | 19 | 3 | 9 |
| READY_DEADLINE_MISSED | 1 | 1 | 7 | 2 |
| TRAFFIC_CHANGED_AFTER_DEADLINE | 2 | 3 | 0 | 0 |
| 已结束 task 完整 work 总和 (s) | 183.205 | 168.467 | 244.207 | 239.072 |
| 已结束 task 完整 work 均值 (s) | 15.267074 | 12.033335 | 15.262918 | 11.384405 |
| 已结束 task 完整 work P95 (s，nearest rank) | 18.991989 | 16.281158 | 25.059494 | 15.549595 |
| 已结束 task 完整 work 最大值 (s) | 18.991989 | 16.281158 | 25.059494 | 15.558113 |

已有 join-rate 的分母是**已结束且提出非空 group**，排除 pending 和空前缀；全部已结束 task 接入率另列，不能混用。high K8 的80%非空接入率同时伴有15个 `NO_EXECUTABLE_PREFIX`，不能据此宣称大部分任务完成交付。完整 work 包含 task 已记账的 history work，未归属 history 为0，不重复加账。单次完整 work 均值由已结束 task 总和除以已结束数取得；全窗口 work 仍保留 pending 已发生的工作。

时间线汇总另修正 restart gap 的方向：原 wrapper 在新记录上写入上一任务到本轮的 `finish_to_next_start_s`，并在已结束的前记录上写 `next_started_time`。汇总以 `raw_restart_gap_s` 保留原值，同时保留 `previous_finished_time` 与 `previous_record_index`；派生 `finish_to_next_start_s` 仅在本轮 `finished_time`、`next_started_time` 均为 finite 时取两者之差，否则为 null。重新提取后四个末尾 `COMPUTING` 的 raw incoming gap 均为0，outgoing gap 均为null；其余63条具有双finite outgoing时间的记录均为0。此项只修正汇总语义，不改原数据，四臂 numeric arm data 与旧分析逐项相同。

补充提取见 [supplement.json](../../artifacts/rs_select_quality_a4_20261030/analysis_final/supplement.json)。它保留旧 summary 路径作为来源，restart 统计直接使用 `next_started_time-finished_time`；当前主分析仍统一使用 `analysis_final_timelinefix`。完整 work 各组成项的**总和s（实际字段记录数）**如下；不同字段缺失数不同，不能以缺失补0。

| 完整 work 组成项 | medium FULL_H | medium K8 | high FULL_H | high K8 |
|---|---:|---:|---:|---:|
| dependency_guard_work_s | 0.008071 (13) | 0.009921 (15) | 0.000726 (17) | 0.003326 (22) |
| normal_evidence_wall_s | 0.031142 (9) | 0.018838 (10) | 0.002508 (1) | 0.012141 (4) |
| normal_work_s | 166.713514 (13) | 161.537088 (15) | 237.774852 (17) | 243.375437 (22) |
| prepare_work_s | 28.318674 (12) | 17.893682 (14) | 16.254918 (4) | 6.576934 (4) |
| publication_bookkeeping_wall_s | 0.016611 (9) | 0.018140 (10) | 0.001636 (1) | 0.007175 (4) |
| ready_registration_bookkeeping_s | 0.002353 (11) | 0.002651 (13) | 0.000228 (1) | 0.000852 (4) |
| revalidation_work_s | 0.140453 (13) | 0.081178 (15) | 0.000000 (17) | 0.000000 (22) |
| selective_history_work_s | 0.456501 (13) | 0.503403 (15) | 0.233208 (17) | 0.324171 (22) |

| 搜索与交付时间项 | medium FULL_H | medium K8 | high FULL_H | high K8 |
|---|---:|---:|---:|---:|
| 首个非空认证 wall 均值 (s) | 12.276348 | 6.266081 | 15.605663 | 7.579443 |
| 首个非空认证有值数 / 缺失数 | 13 / 0 | 15 / 0 | 5 / 12 | 18 / 4 |
| 已COMMITTED ready→effective 均值 (s) | 3.815712 | 5.763279 | 6.607340 | 3.787181 |
| 上项 COMMITTED 数 | 9 | 10 | 1 | 4 |

首个非空认证 wall 是搜索 wall elapsed，不能直接和绝对模拟时间相加，也不代表最终 ready 或实际交付。首任务示例：medium K8 的 snapshot为20 s，first wall约5.041 s，ready为33.084863 s，执行及下一轮均在40 s；high K8 的 first wall为7.201927 s，ready为34.892712 s，执行及下一轮均在40 s。high FULL_H 的 first wall为18.591057 s，prepare后在45.074752 s以 `READY_DEADLINE_MISSED` 拒绝，原effective为40 s，没有执行。63条已完成且有next的任务没有额外5 s等待；4条pending排除。high FULL_H 的期限拒绝为7/16个已结束task，K8为2/21，是实际交付期限瓶颈的证据。

正式 job 的运行成本和规划工作分开。各臂分别绑定CPU16/17/18/19，数值库单线程；下表来自原 `result.json.costs`，full job观察截止于该结果写出前，账本child wall另含最终结果写出。online advance包含t0 resume；dispatch是其中口径，不能把这些行重复相加。

| 成本项 (s) | medium FULL_H | medium K8 | high FULL_H | high K8 |
|---|---:|---:|---:|---:|
| full_job_wall | 878.786 | 744.386 | 1047.304 | 1056.670 |
| process_cpu | 876.591 | 742.453 | 1045.611 | 1055.114 |
| setup wall（含restore I/O） | 0.731 | 0.716 | 1.204 | 1.205 |
| online advance wall（含t0 resume） | 377.457 | 264.648 | 402.772 | 380.018 |
| initial resume wall（已在online内） | 13.196 | 11.488 | 18.597 | 12.823 |
| dispatch wall | 311.179 | 205.893 | 295.431 | 289.776 |
| output I/O wall（含序列化） | 491.303 | 469.933 | 629.944 | 660.357 |
| output I/O CPU | 489.629 | 468.204 | 628.331 | 658.868 |
| other bookkeeping wall | 9.295 | 9.090 | 13.384 | 15.090 |

主对话审核的解释：process CPU与full-job wall接近，现有证据不支持把high表现主要归因于CPU抢占等待。full job的大量耗时来自output I/O口径（含序列化），不能写成单次规划花了十几分钟；CPU占比高的序列化也不能径直归因于磁盘慢。

下表为各**独占 callback 阶段**的 wall 均值，单位s，括号为该字段实际记录数。均值不是每个 primitive 的耗时；缺失字段不填0。特别是 high FULL_H 的 prefix validation 仅13/17有记录，其1.625145 s是这13条的均值。全部原始 wall/CPU 记录见 `callback_timings.csv`。

| 独占阶段 | medium FULL_H | medium K8 | high FULL_H | high K8 |
|---|---:|---:|---:|---:|
| candidate_generation_and_single_vehicle | 5.704804 (13) | 3.694700 (15) | 6.132123 (17) | 2.689917 (22) |
| commit | 0.000007 (13) | 0.000007 (15) | 0.000008 (13) | 0.000007 (22) |
| cp_solve | 0.527807 (13) | 0.370293 (15) | 0.132307 (17) | 0.195345 (22) |
| downstream_explanation | 0.008676 (13) | 0.004952 (15) | 0.030555 (14) | 0.006559 (22) |
| fallback | 0.000007 (11) | 0.000008 (1) | 0.000077 (11) | 0.000044 (4) |
| geometry_template_screen | 0.063134 (13) | 0.032537 (15) | 0.057265 (17) | 0.011223 (22) |
| global_evaluation_frame | 0.015184 (13) | 0.014331 (15) | 0.018997 (17) | 0.017037 (22) |
| hint_construction | 0.025976 (13) | 0.035591 (15) | 0.028063 (17) | 0.052045 (22) |
| model_build | 0.037948 (13) | 0.030332 (15) | 0.049079 (17) | 0.068358 (22) |
| normal_evidence_reporting | 0.000007 (12) | 0.000006 (8) | 0.000007 (17) | 0.000007 (5) |
| prefix_validation | 2.246373 (13) | 2.732343 (15) | 1.625145 (13/17) | 2.000041 (22) |
| resource_conflict_build | 0.003969 (13) | 0.000948 (15) | 0.008745 (17) | 0.000901 (22) |
| selected_validation | 0.036878 (13) | 0.009295 (15) | 0.143264 (17) | 0.009057 (22) |
| selection | 0.028841 (13) | 0.039844 (15) | 0.028317 (17) | 0.045323 (22) |
| selective_selection | 0.008543 (13) | 0.008773 (15) | 0.029603 (17) | 0.018380 (22) |
| snapshot_and_uniform_refinement | 0.000003 (13) | 0.000003 (15) | 0.000003 (17) | 0.000003 (22) |
| trajectory_validation | 4.078436 (13) | 3.764416 (15) | 6.046061 (17) | 5.885422 (22) |

主对话审核的成本解释限于本次同版本FULL_H↔K8：high trajectory validation独占总量从102.783增至129.479 s，而每callback均值从6.046降至5.885 s，callback数17→22是总量增加的主要来源。prefix validation总量从21.127增至44.001 s，观测均值由1.625（13条有效计时）至2.000（22条）；既有更多有效计时callback，也有callback内工作量上升，不能据此宣称每个primitive变慢。

复用确有发生：high K8 callback顶层 `validation_cache` 为pair hits4021/misses16532；selective层跨局部calls聚合的 `pair_checks` 为hits9659/misses31397，后者包括selected-background28693与selected-selected2704次miss。两层口径不能相加。`prefix_reuse` hits89/misses39，另有1个callback缺少hits字段，保留missing；high FULL_H同selective聚合口径pair misses为26659。这说明缓存并非未生效，多scope/background仍需要检查；没有cache-off反事实，不能量化缓存节省，也不把耗时变化归因于候选简化。

### 仍存在的有界质量差距

主对话实读 high K8 的两个 snapshot：40 s 的首个非空认证目标为 `(0,-6,31,494138635.2,9415)`，最终全 KEEP 为 `(0,-6,31,484809632.9,9415)`；215.42753894580528 s 的首个非空认证目标为 `(0,-1,4381.672722222223,603230986,12072.5)`，最终 KEEP 为 `(0,-1,1452.9668333333334,585144651.6,12072.5)`。新评分在这些观测中仍选择后者。这是估计排序证据，不证明 MOVE 的真实后续表现更好；正式四臂没有做对应反事实续行。

high K8 的15个 `NO_EXECUTABLE_PREFIX` 中，11个曾有首个非空认证，4个没有。搜索期间存在安全认证选择、最终按当前目标选择KEEP，不等价于物理安全无解，也不证明未来MOVE一定更好。

| 顶层 suffix completion telemetry | medium FULL_H | medium K8 | high FULL_H | high K8 |
|---|---:|---:|---:|---:|
| attempt数 | 26 | 30 | 34 | 44 |
| 实际priority slice数 | 4 | 30 | 0 | 2 |
| complete_candidate_in_catalog数 | 0 | 0 | 0 | 0 |
| attempt wall合计 (s) | 0.024955750 | 0.289192727 | 0.000000000 | 0.012591026 |

此表只统计顶层telemetry，不与 `selective_replanning.suffix_completion` 的镜像重复相加。high K8中42/44个attempt为 `CANDIDATE_WORK_CUTOFF`、没有执行切片，21/22个callback发生一次scope expansion。补链接口和切片修补已检查，但正式运行没有取得由该机制补成的完整链，**不能把成绩改善归因于补链成功**；候选预算内覆盖与完成度分配仍有缺口。本轮不强制MOVE，不继续调参或重跑，不自动进入阶段B；R12单状态短续与两个正式根各自保留其证据边界。

### 四臂保存回放审计

四臂均调用原 `audit_job`，均为 `status=PASS`、`full_a20_covered=true`；continuous separation 为 `CERTIFIED`，resource、C2、proposal nonexecution 与 independent checks 均 `PASS`，PVA 为 `CHECKED`。这些是保存回放审计结果，连续人工回放仍为 `HUMAN_PENDING`。

| 根 / 方法 | audit child wall (s) | 保存回放审计 |
|---|---:|---|
| s3r23 / FULL_H | 15.677057259017602 | PASS / full A20 |
| s3r23 / RULE_LOCAL_K8 | 13.861587957013398 | PASS / full A20 |
| s3r24 / FULL_H | 10.173829629085958 | PASS / full A20 |
| s3r24 / RULE_LOCAL_K8 | 13.202077616006136 | PASS / full A20 |

### 正式四臂关键帧查看

主对话实际查看四臂全部 t260/top_down 与 t320/side，共 **8图**，状态为 **`AI_SAVED_FRAME_REVIEW`**。已导出的48图与 manifest 位于 `artifacts/rs_select_quality_a4_20261030/visual_review_formal/<root>/<method>/...`；连续整段人工播放仍为 `HUMAN_PENDING`。

保存画面观察：t260 medium K8 的 P0/P1/P2/P4 为 service，P5 的 UID14 为预定进近，而 medium FULL_H 的 P2/P3 为 service；high K8 内环可见空，P4 的 UID3 接近触地、其余 pad 为 FREE，high FULL_H 仍可见多架内环盘旋且仅 P1 为 service。t320 high FULL_H 的6个 pad 为 FREE，high K8 的P4为 service、其余为 FREE。主对话据此保留“改善未消除供给/有效推进不足”的限定解释。关键帧不能证明连续安全；连续安全证据来自上表保存回放审计。

共四臂，不增加 seed，不用旧 t=0 或旧已运行臂替代其中一臂。common 输出位于 `common_initial/<root>`；正式输出位于 `arms/<root>/<method>/evaluation/<root>/<method>`；短续输出位于 `diagnostic_continuations/<case>/<branch>`。每相对 5 s 保存 boundary/checkpoint/progress。`method_*` 文件仅包含新方法计算；供原审计读取的 scheduler/tasks 文件保留 initializer provenance 加新记录，索引仅在导出副本重编号，独立 mapping 解释计算事件，物理 events/frames 不改写。完整回放包含 initializer 与方法阶段，人工验收仍为 `HUMAN_PENDING`。

## 已记录版本与失败尝试

| 提交 | 用途 | 本文允许的结论 |
|---|---|---|
| `3ff1a15518d420bfe2d5cffb85c73decf58237b7` | reference_v4 三开关、有界诊断、共同初态及短续基础实现 | 工程与接口版本；不是新 score 或效果证据 |
| `674bce676db7f924acce5c9070d0567a0d623a57` | 三个 A4 入口在 dependencies 前保存 launch thread 设置 | 修复 preexecution 启动字段绑定 |
| `1414708b04b387a4b3ce002dd6354278b6e474ed` | archive 解码后的 frame identity 重绑定及短续 source compatibility | 修复 prevalidation 归档绑定，保持数值内容与候选不变 |
| `b6590c8a5ede25b5f63e89834dd4cd5ec36a646d` | `stitched_continuation_v1` 评分与有界补链；本轮共同 initializer source | 评分与接口版本；initializer 仍显式使用旧 profile |
| `a2f06f5f385d084607b8ed92fa0b059f8557ae97` | 全局有界 completion micro-round；source hash `74c9225db9af2444d34c82f5e49632790b99707d1e07050aba5aa2088d5e9a5f` | 原额度内将补链切片放在普通 growth 前 |
| `b80ca3aee7d7ba6495640f4a3b02f72b026fabbb` | record export 复用既有 `replay._plain`；正式四臂 source | 仅导出工程修复，不改变搜索/评分/物理算法 |
| `61f9ae0ef3b5794ce2275632aff9e934c67e804b` | 正式四臂可视化工件；已提交并推送 | 50个可视化工件、其中48图；主对话实际查看8图，非连续人工播放 |
| 本报告与汇总helpers的交付提交 | 本次交付提交中登记 | 最终hash由主对话交付说明记录；不在提交内容中自指 |

| 已保留的失败尝试 | 记录的原因 | 边界与处理 |
|---|---|---|
| preexecution 失败 | 入口遗漏 `d.launch_threads`，线程设置无法按原始 launch 值核对 | 发生在该尝试的 world/规划/候选生成之前；记录保留，由 launch 修复版本接续 |
| prevalidation 失败 | 解码 `Section` 后 repr-based frame identity 变化，原 scope/frame 绑定拒绝 | 发生在该尝试的新规划/候选生成之前；先检查归档内容和绑定，再仅刷新 identity link，不复用旧 ACCEPTED 验证缓存 |

两次失败不覆写、不从预算抹除，不作调度效果或物理不可行证据。相关 freeze 文件只读保留于 `artifacts/rs_select_quality_a4_20261030/diagnostic_source_freeze*.json`。

正式入口另有两批工程失败，均保留原账本、日志及回执并计费：launcher 父/子使用同名 log，造成四个 pre-child failures，合计 **0.006357651902362704 s**，见 `receipts/formal_launcher_recovery.json`。首次实际四臂在 checkpoint 后的 live `Section` record export 递归处各自 exit 1，合计 **110.27661685703788 s**；原 arms 完整保存在 `failed_attempts/record_export_failure/arms`，归档映射、hash 和旧 binding 见 `receipts/formal_export_recovery.json`。修复导出后从原 SHA 相同的 common checkpoint 重启四臂属于工程恢复，不按成绩重跑，不用失败片段作正式效果结果；原 common 文件未覆盖。

另行登记两项执行偏差。旧 server lane 遗漏五项线程环境设置，造成 **8 个 preflight 失败**，并在失败后错误继续下一 stage；这些尝试均未进入规划或候选目录创建。8 条 job row 实际保存在本轮输出根的 **`launch_environment_recovery.json`**，对应日志为 `logs/*_launch_environment_failure.log`；不是同名目录。失败和计费记录保留，不能把它们当成目录/质量观测或物理不可行证据。

后续 server 在主对话将该次执行收窄为仅四个 `augment` 后，仍继续执行了此前用户已授权、但不在该次执行范围内的余项。已有数据保留，按其实际输入、预算和执行条件审查；不为获得另一组结果而重跑。P2 offset1 因 CPU 不可用发生前置失败，该失败仍保留。offset2 首组/末组与 baseline 相同的事实如上，不作为不同组选组已经比较的证据。

## 已有服务器测试回执登记

早期六项为主对话转交的 worker receipt；本次直接读取了 fixed_groups/new_profile 的 XML，分别为17/46 tests、0 failure、0 error、0 skipped，账本 job 均 exit 0。下表登记成功回执；原失败回归另列保留。测试集合和版本可重叠，**不将各行通过数相加为独立测试总数**，也不将测试通过视为正式效果结论。服务器 XML 均相对本轮输出根。

| 通过数 | 用途/范围 | XML | child wall (s) |
|---:|---|---|---:|
| 175 | 首次 reference_gap 七文件：reference_v4、executable_v3、repair_selection、repair_progressive、followup、replanning、quality_gap | `tests/reference_gap.xml` | 30.970102772000246 |
| 15 | common_initial 单文件 | `tests/common_initial.xml` | 5.797300022095442 |
| 35 | gap + short 首次两文件；不是 thread-fix 集合 | `tests/short_continuation.xml` | 4.438037261017598 |
| 59 | gap + short + common 三文件，包含 launch 修复的 9 个用例 | `tests/preflight_launch_binding_tests.xml` | 9.524737753905356 |
| 13 | comparison 单文件，含真实 tiny checkpoint/replay 及原 audit | `tests/common_comparison.xml` | 14.897858378011733 |
| 56 | archive fix 后 gap + short 两文件，含 archive rebinding、source compatibility | `tests/archive_binding.xml` | 4.680722963996232 |
| 17 | fixed-group 单文件与三组干预绑定；本次实读 XML | `tests/fixed_groups.xml` | 1.0615829939488322 |
| 46 | stitched continuation + common initial + common comparison 三文件；本次实读 XML | `tests/new_profile.xml` | 23.600472318008542 |
| 101 | global completion priority slice 与相关回归；0 failure/error/skipped | `tests/priority_slice.xml` | 13.563539660 |
| 15 | export fix comparison 回归，含真实 live FOLLOW Section；0 failure/error/skipped | `tests/comparison_export_fix.xml` | 15.004162565 |

本地 launcher helper **16 个 fixture**已通过；comparison summary helper 原4个 fixture加1个 restart-gap fixture，当前 **5 passed，1.48 s**，exit0，XML为 [local_summary_timelinefix_verified.xml](../../artifacts/rs_select_quality_a4_20261030/tests/local_summary_timelinefix_verified.xml)。新增 fixture 验证 pending 的 incoming0/outgoing null、finished 使用 next-finished 而非 incoming，以及非finite时间不产生 gap。它们是工具夹具检查，不是新增实验样本。各行测试范围有重叠，不相加为独立通过数；上述服务器 child wall 与 XML 自身 suite time 分别保留各自含义。首次本地 timeline fixture 调用显示 `EEEE.`，随后因 JUnit 路径 `PermissionError` 退出1；失败 temp目录保留，使用新的 basetemp 并获授权提升执行后得到上述5通过，未覆盖失败目录或实验数据。

原 `new_profile_regression` 服务器回归共154项：151 passed、2 failed、1 skipped，exit 1，child wall **44.44731804705225 s**，原失败收据保留。主对话已读该次 XML：两个失败为 `test_stage2_downstream_campaign_v8.py::test_freeze_profiles_separate_algorithm_and_unchanged_quality_budget` 与 `::test_prepare_run_restore_export_read_frozen_profile`，均在 `campaign.source_provenance` 调用 `git rev-parse HEAD` 时因服务器没有 Git 元数据失败；一个 skipped 为同文件的历史 medium archive Windows 路径未挂载。不能把这三项失效检查记成服务器通过，也不在服务器补建 Git。

主对话随后报告这三个 provenance/历史 archive 项在本地补测 **3 passed、0 skipped，2.54 s**，artifact 为本地 `tests/local_provenance.xml`。本地补测不改写原服务器2 failed/1 skipped。该轮监控 worker 的测试启动另遗漏了显式 `NUMEXPR_NUM_THREADS`；这项执行偏差保留，后续正式入口仍须在 dependencies 改写环境前，对全部五项线程值作前置断言。

## 本轮完成状态

| 工作 | 状态 | 证据与边界 |
|---|---|---|
| 四 case 单因素诊断与选组比较 | 已登记开发诊断 | R12 补链、固定目录CP对照及P2三个实际组见本文；执行偏差保留，不外推泛化 |
| 新评分与有界补链 | 已实现并用于四个正式臂 | 新评分、global completion slice 与 export fix 的版本、freeze 和测试见本文；旧默认保留、预算不增；仍有有界质量差距 |
| R12 120 s 短续 | 两 branch 与保存回放审计已审核；`HUMAN_PENDING` | 210→330 s 观测、首动作及继承落地归属见本文 |
| R4 120 s 短续 | `NOT_APPLICABLE`，已 prepare、未 run | `NO_FIXED_UID_IMMEDIATE_PARTIAL`；不补选 UID |
| 两根共同初始化 | `COMMON_INITIAL_READY` | 两根 t0=20、完整 common SHA 与 inherited FINAL 见本文；物理账本不清零 |
| 四个正式配对臂 | 四臂 `COMPLETE`；已登记主对话限定解释 | 同 profile/deadline，20→320 s；终态、增量、双分母和成本拆分见本文 |
| 四臂保存回放审计 | 四臂 `PASS` / full A20 | continuous CERTIFIED；resource/C2/proposal/independent PASS，PVA CHECKED；非人工回放 |
| 人工回放与最终解释 | R12 6图、正式四臂8图 `AI_SAVED_FRAME_REVIEW`；HUMAN_PENDING | R12已导出24图、正式四臂48图；主对话限定解释见本文，非连续整段人工播放 |
| Git与用户交付 | 数值实现b80、可视化61f9已推送；本报告与helpers纳入本次交付提交 | 最终文件清单、测试、提交hash与push结果由主对话交付说明记录 |

本轮授权研究与工程工作已完成；`HUMAN_PENDING`保持原义，不据关键帧宣称完成连续人工播放。R12单状态配对和两个正式冻结根的观测不外推为总体速度、质量或因果结论；本轮没有继续调参、增加seed、重跑挑选成绩或进入阶段B。
