# Stage2 针对性验证对应表

依据是 `new10.16/rotating_slots_stage2_baseline_v1_0/stage2_targeted_tests_v1_0.json`
的 28 项要求。下表定位当前测试代码及其有限断言，不把“有同名测试”写成整项合同验收通过。
主对话已核对运行前服务器回归日志、XML及其hash：**208 collected，207 passed，1 skipped，0 failure/error**。
该历史验证在所列模块范围通过；本轮随后发生失败恢复异常，旧207项通过不证明该异常路径已覆盖。
此表仍是有限断言映射，不把28项要求称为穷尽验收。
正式实验已 **INVALID_STOP**，最后共同840秒结果及该窗口审计见
[first_campaign_status.md](first_campaign_status.md)；不是完整3600秒或预算截尾结果。
失败恢复及CLI修复已通过统一服务器回归：**219 collected、218 passed、1 skipped、0 fail/error**；
不改写原始失败记录，不将修复验证称为正式3600秒重跑。
PyVista 人工观察保持 **HUMAN_PENDING**。

修复后统一服务器首轮为217 passed、1 skipped、1 failed（pytest48.35秒/wall48.650秒）。
主对话核对唯一失败来自旧upstream测试将占槽起点等同于admission，而非实际目标capture；
对应断言更新仍保留release/admission检查，最终验证结果见下。此验证不替换本次INVALID_STOP的原始证据。

修复后最终12模块验证UTC为 **2026-10-20T19:49:20Z–19:50:09Z**，pytest **48.49秒**、
进程wall **48.767秒**，退出0；核心4、四个数值库线程均1，五个相关源码/测试hash本地与服务器一致。
唯一skip为已安装OR-Tools时不适用的缺依赖负向测试；两个真实CP测试执行通过。
独立输出目录为`D:\7_codespace\training_outputs\terminal\stage2_recovery_validation_20261021`，
最终`regression_final.log/xml/timing.txt`与首轮失败的`regression.log/xml/timing.txt`分开保存。
最终XML SHA256：`73dd80fc156858d39cb94e82cc2950bec14a5d9f44cefa7cd88e75f41fed1d66`。

运行前回归使用已验证的服务器隔离环境，包含4个Stage2模块，以及既有独立audit、replay、world、
upstream、targeted_world、practical_world、performance_checkpoint相关测试。
运行UTC为 **2026-10-20T17:28:26Z–17:29:13Z**；pytest报告 **46.87秒**，进程实测 **47.135秒**。
该历史回归真实CP的T09 **1.696秒**、T15 **0.367秒**，两者均执行并通过，未跳过；
修复后19:49最终回归中的两个真实CP测试也均通过，不混用两轮耗时。
唯一跳过项仍是已安装OR-Tools条件下不适用的缺依赖负向测试；字面3600秒T24两参数已在服务器确认。

产物在正式输出根目录的 `preparation/final_validation.log`、`final_validation.xml`、
`final_validation.timing.txt`；XML SHA256为
`7c8742a85742563c37c186fb899b238f5ea586ae03156dad4303e6a5d4c05402`。

已经由主对话复核的准备阶段服务器证据：隔离环境中的真实CP针对性检查为
**19 passed、1 skipped**；跳过项是“缺少OR-Tools必须明确失败”的负向用例，已安装时按测试条件跳过。
这组较早计数由上述最终回归补充，也不表示28项合同全部验收完成。

繁忙36架夹具的单次正常callback检查中，Energy-slack为 **4.39349 wall秒**，
RH-CP-SAT为 **4.64497 wall秒**；两者均有95列、1286次新构造尝试并记录目录截断，
均认证并实际提交2条正常final。检查未推进物理时间，因此提交不等于已完成接地或吞吐。
CP有6条层级求解记录、3轮按需冲突补充，不安全暂定组合被拒，最终为 **LEX_LIMITED**。
这些是准备阶段的实际集成证据，不能用单次低于5秒声称实时认证、完整质量对比或CP最优。

正式campaign输出位置固定为服务器
`/home/user/training_outputs/terminal/stage2_first_campaign_v1_20261021`，
经Syncthing对应本地
`D:\7_codespace\training_outputs\terminal\stage2_first_campaign_v1_20261021`。
状态、共同截止、检查点、计时、回放和独立审计以该目录实际产物为准；本文不预填提交哈希。

测试模块简写：

- **W**：`tests/test_rotating_slots_stage2_world.py`，执行器及公共调度边界。
- **P**：`tests/test_rotating_slots_stage2_planning.py`，预测列、选择、前缀认证及 CP。
- **C**：`tests/test_rotating_slots_stage2_campaign.py`，初态、预算、恢复和回放适配。
- **A**：`tests/test_rotating_slots_stage2_audit.py`，独立记录检查及内容自洽的篡改反例。

| 合同项 | 具体测试 | 当前代码实际断言及边界 |
|---|---|---|
| T01 初态 | C `test_T01_frozen_busy_fixture_geometry_and_cohorts` | 39 个实体；14/12/10 按槽位公式；SOC 范围及零初始账本；pad0/2/4 的30/60/90秒；初始地面不计成功、36架 admitted、守恒及当前几何≥80m。构造函数调用既有连续认证。没有用此测试证明自然稳态或完整繁忙运行。 |
| T02 正常多步 | P `test_T02_actual_outer_middle_inner_pad_and_only_first_committed` | 实际生成 O→M→I→pad 三段列，选中整列而只提交首个 O→M；active1、soft0。单机夹具。 |
| T03 Slack | P `test_T03_slack_depends_on_remaining_normal_energy_not_soc` | 相同SOC、不同人工给定正常参考导致排序不同；负余量类排在 UNKNOWN 前。不是对全部参考估计准确性的证明。 |
| T04 UNKNOWN | P `test_T04_reference_unknown_is_not_zero_or_failure` | 零尝试预算返回 UNKNOWN、能量/SOC需求为None；保留KEEP/WAIT，尾部键为未知值，世界未失败。 |
| T05 资源 | P `test_T05_wait_retains_slot_and_pair_resource_conflict` | WAIT保留无限槽位区间；进入该槽的移动列与其冲突。该用例直接覆盖槽位；pad 区间另由T08覆盖。 |
| T06 几何交叉 | P `test_T06_actual_geometric_crossing_with_distinct_resources` | 两个不同资源的实际转移轨迹，资源检查无冲突而连续配对检查拒绝，且不是UNKNOWN。 |
| T07 联合离开 | P `test_T07_joint_departures_not_prefiltered_against_other_permanent_hold` | 外环进入与中环离开各自构造；外环单列未被永久holding预筛删除；两列配对及可提交前缀均被接受。 |
| T08 忙pad | P `test_T08_pad_descent_interval_release_not_touchdown` | pad在下降前释放时接受；下降开始=接地前13.5秒；仅在接地前5秒释放时拒绝。 |
| T09 下游改变第一步 | P `test_T09_real_multistage_downstream_pad_changes_selected_first_step` | 使用真实三段列和真实CP；增加下游pad预约后，选中的第一目标改变且仍为三段列。`importorskip("ortools")`；缺依赖时跳过不能记为通过。 |
| T10 前缀依赖 | P `test_T10_safe_full_prediction_cannot_authorize_uncommitted_release` | 完整预测配对可行；6秒后离开不在5秒提交窗，依赖它的首段前缀以RESOURCE_CONFLICT拒绝；真实世界未变。 |
| T11 既有承诺 | P `test_T11_fixed_active_endpoint_uses_successor_version_and_soft_gap_energy`；W `test_atomic_commit_rejects_stale_evaluation_and_keeps_other_plans` | active+soft终点的环、版本、时间正确，soft前5秒holding计能；失配快照提交被拒、有效版本仅提交一次。尚非所有应急/超时/受保护依赖取消分支的穷尽验证。 |
| T12 储备/峰值 | W `test_t12_normal_final_exact_reserve_and_peak_budget_include_descent` | 实际final完整接地，SOC恰25%、累计峰值恰120秒，13.5秒垂降及精化账本包含在内；无能源失败。不是全部跨环临界组合的扫描。 |
| T13 无副作用 | P `test_T13_repeated_reversed_pure_catalog_checks_do_not_mutate_world`；W `test_t13_detached_evaluation_order_leaves_real_ledger_and_resources_unchanged` | 候选正反序重复评估，真实SOC、资源、事件、能量账本、轨迹及计时账本保持；快照不可写且无world/manifest属性。 |
| T14 信息边界 | W `test_t14_planning_snapshot_does_not_depend_on_hidden_manifest_suffix` | 改未释放manifest后缀不改快照；公共应急监测快照不可变、与后续真实字典修改隔离。未直接逐一运行两种调度器比较最终决定。 |
| T15 UNKNOWN回退 | P `test_T15_real_solver_unknown_retains_verified_plan`；`test_cp_dependency_missing_is_explicit` | 真实CP极小限时返回UNKNOWN，保留认证方案、标LEX_LIMITED且不产生物理失败；缺OR-Tools明确报错。前一用例需OR-Tools；后一用例在已安装时主动跳过。 |
| T16 目录上限 | P `test_T16_catalog_budget_truncation_keeps_every_aircraft_and_round_robin`；`test_T16_real_catalog_counts_and_first_target_cap` | 缩小的8/4尝试预算下两机均获机会且保留KEEP/WAIT、标不完整；真实单机目录移动列≤8、每首目标≤2、每机≤64、总≤2048。没有实际耗尽完整2048次的压力用例。 |
| T17 单机失败 | W `test_t17_isolated_battery_loss_other_aircraft_continues_once` | 最早损失时刻在(0,0.2)，归零SOC、终态隔离；另一机继续；再推进不重复耗能/罚款，守恒。公共应急在该物理夹具关闭。 |
| T18 同时失败 | W `test_t18_simultaneous_losses_do_not_repeat_or_stall`；A `test_stage2_simultaneous_isolation_continues_with_independent_energy_checks` | 两机同刻各记一次，时间继续、罚200万；独立重算损失/剩余机、无censor、连续间隔通过且不修改档案。 |
| T19 预约/依赖清理 | W `test_t19_releases_only_failed_resources_and_preserves_other_active`；A篡改参数 `resource` | 失败机自己预约截于失败时刻，active/soft清理；另一active对象/哈希保持，支持依赖清理，失败前active被归档；保留失败资源的篡改被拒。有限的单条依赖用例。 |
| T20 失败分类 | W `test_t20_peak_loss_is_distinct_from_numerical_error`；A `test_genuine_peak_budget_loss_is_audited_separately` | 注入峰值累计与无效能源模型分别导致单机峰值失败和全局无效；后者不收费。另以实际final功率在119.99秒基线触发真实峰值失败并独立审计。 |
| T21 软件/安全无效 | W `test_t21_authoritative_trajectory_mismatch_is_invalid`；`test_t21_executed_80m_violation_stops_invalid_without_energy_penalty` | 权威哈希错误抛出并标无效；故意注入重合实体触发实际间隔无效停止；两者不冒充能源损失或收费。 |
| T22 同失败多落地优先 | C `test_T22_T23_episode_lexicographic_order` | 手工计数中，同0失败时3次落地优于2次，即使能量和应急费用更高。这里验证排名函数，不是实测方法质量结论。 |
| T23 少失败优先 | 同上；A `test_stage2_success_after_loss_uses_touchdown_event_not_service_completion` | 0失败2次落地优于1失败300次落地；实际损失后接地独立统计为1，primary key前两项[1,-1]。 |
| T24 截止与pending | W `test_t24_cutoff_settles_without_new_decisions_and_resume_dispatches_once`；C `test_T24_pending_is_not_success_or_failure`；C `test_T24_literal_3600_actual_final_boundary_and_resume` | 一般测试覆盖1秒服务结算、截止不新调度、恢复仅一次，以及pending非成功/失败。新增实际晚到→入场→两次跨环→含垂降final，分别在3600秒接地计1、3600.1秒接地在截止仍pending且非失败；继续运行才记后者成功，并要求所有决策早于3600−1e−9。两参数及W模块合计14项本地通过11.40秒；字面两参数已包含于服务器最终通过结果。 |
| T25 目标量化 | P `test_T25_quantization_changes_only_solver_coefficients` | 时间/能量只转为求解器系数(0.1秒/1Wh)，原objective元组保持。未独立对所有真实轨迹/SOC字段做量化前后遍历比较。 |
| T26 计时 | W `test_t26_dispatch_measurement_includes_monitor_rescue_admission_and_scheduler`；P `test_T26_exclusive_timing_and_callback_checkpoint`；C `test_T26_timing_quantiles_keep_raw_counts` | 人工短sleep核对公共顺序和完整wall包含关系；真实单机callback包含候选、认证、commit，排他计时不重复；状态可JSON往返；分位数保留样本/超时原值。不是忙36的5秒达标或硬实时认证。 |
| T27 预算 | C `test_T27_budget_survives_restart_and_common_complete_markers`；`test_T27_T28_resident_episode_continues_60_to_120_without_reset` | 仿造时钟越过9.5小时、重启不归零；只取共同完整120秒，不把partial当完成；空世界真实60→120不重建并可恢复中断的summary发布。没有真实等待10小时或父进程强杀事务的长耗时验收。 |
| T28 恢复/观察 | C `test_T28_exact_checkpoint_pending_dispatch_and_absolute_phase`；`test_T28_checkpoint_rejects_identity_corruption_and_unknown_fields`；`test_T28_stage2_replay_separate_predictions_and_failure_ledger`；`test_T28_viewer_isolated_marker_and_actual_dispatch_hud`；上述60→120测试 | 保留绝对时间/位置/槽位/能量账本，恢复后继续旋转；身份及未知字段拒绝；五个Stage2额外成员必需、预测与active分离；失败marker/档案/完整dispatch HUD可查询；导出较早60秒不截断120秒源流。未覆盖所有active+soft+失败混合检查点，也未做真人PyVista窗口观察。 |

独立检查器另以 `test_self_consistent_stage2_corruption_is_rejected` 的9组参数核对：
失败原因、重复失败、罚款、隔离后能量、双重成功、残留资源、末状态档案、初始地面计数和伪造接地。
这些反例先重新计算文件hash再检查，覆盖内容自洽的假记录；不宣称独立实测能源模型正确。

本次故障后新增的对应回归（已包含于218通过/1跳过的最终验证）：

| 对应范围 | 新增测试与有限断言 |
|---|---|
| T19资源复用/隔离 | W `test_t19_isolation_preserves_survivor_intervals_after_source_slot_reuse` 保留幸存者active/soft对象、资源区间及capture起点，真实轨迹状态70→70.1继续；`test_rebuild_preserves_capture_origin_and_still_rejects_real_overlap` 保留capture，人工注入真实重叠仍拒绝。资源夹具不冒称完整历史inbound复现。 |
| T21/T27无效状态交付 | `tests/test_rotating_slots_stage2_cli.py` 的 `test_run_resume_exit_follows_persisted_status_after_output` 用runner替身验证归档返回、JSON输出后读取持久化状态；run/resume×三种状态共6例，INVALID_STOP退出1、正常完成/预算截尾保持正常退出。不是新的正式运行。 |
| CLI兼容边界 | 同模块 `test_invalid_stop_status_and_report_queries_still_succeed`（2例）与 `test_prepare_keeps_existing_output_and_exit_semantics`（1例）保留status/report查询与prepare输出/退出语义，合计9项CLI新增参数实例。 |

可执行服务器针对性命令（无GUI、使用已验证的隔离环境）：

```bash
cd ~/projects/terminal
/home/user/envs/terminal_stage2_v1/bin/python -m pytest tests/test_rotating_slots_stage2_world.py tests/test_rotating_slots_stage2_planning.py tests/test_rotating_slots_stage2_campaign.py tests/test_rotating_slots_stage2_audit.py -q --durations=10
```

若服务器不具备本地PyVista依赖，C中的viewer适配测试应单独在本地执行并报告；不能将缺依赖改称通过。
正式运行、预算截尾、正常/dispatch耗时、独立审计和人工观察应保持各自状态，最终验收由主对话依据实际产物决定。

字面截止测试曾观测到：请求结束3250秒，执行器停于3249.9999999999545，仍为INBOUND且无全局失败。
经主对话确认后，执行器增加正的≤1e−9秒尾段物理积分与安全检查，再精确结算截止，不在该尾段新调度。
这是上述14项本地检查所使用的修复；没有通过直接重设物理时间或伪造接地账本消除差值。
