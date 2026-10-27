# 选择性重优化阶段 A：计算与正常着陆的权衡

阶段 A 的研究状态为 **TRADEOFF_OBSERVED**。RULE_LOCAL_K8 在八个同状态开发快照中减少开放变量规模和观测计算开销；两个 300 秒闭环的正常规划累计工作约下降 20%，正常着陆却分别从 15 降至 10、从 17 降至 13。当前证据体现计算与服务的代价，不能称为无损在线增益。

阶段出口为 **A_TRADEOFF_REVIEW**。本次结果没有批准进入阶段 B，B/C 均未开始。

自动验收状态为 **AUTOMATED_ACCEPTANCE_PASSED_HUMAN_PENDING**。A01–A19、A20 和 X01 已完成对应专项检查、实现及保存回放复核；X02 的标注测试通过，实际人工观看未完成，保持 **HUMAN_PENDING**。只有 FULL_H 与 RULE_LOCAL_K8，没有学习模型、训练、阶段 B/C 或自动推进。

科学源码为 `750cda67a7b2594fc3da740124eb9fcdca7b8c3f`，绑定为 `8945ba891ae3c73af1d223f1ba7315f6bd8912c8444c55a2bfb00912ba969162`。本稿已核对同步摘要内嵌的八对快照及四臂结果、共同终点 CSV 和最终账本。生成器留下的原始摘要仍含 `DATA_INSUFFICIENT_PENDING_PRIMARY_REVIEW`，没有改写该实验输出；最终决定记录在本文件和 [primary_review.json](../../artifacts/rs_select_a_20261027/primary_review.json)。

## 共同 300 秒终点的闭环结果

中流量 s3r23 起始 24 架空中、3 架地面服务飞机，窗口内新增请求 16；高流量 s3r24 为 36 架空中、3 架地面，新增请求 13。初始地面不计入着陆增量。四臂均完整到达 300 秒、实际绑定 CPU12；新增请求着陆均为 0，以下着陆全部来自初始空中 cohort。

| 场景 / 方法 | 正常着陆 | 空中未完成 | 正常规划累计工作 s | 整个作业 wall s | dispatch s |
|---|---:|---:|---:|---:|---:|
| s3r23 / FULL_H | 15 | 25 | 191.242319 | 460.008901 | 199.923687 |
| s3r23 / RULE_LOCAL_K8 | 10 | 30 | 152.685410 | 404.269967 | 163.265132 |
| s3r24 / FULL_H | 17 | 32 | 173.407690 | 584.009808 | 226.834006 |
| s3r24 / RULE_LOCAL_K8 | 13 | 36 | 138.570560 | 478.877527 | 152.763922 |

正常规划工作在中流量下降 20.16%，高流量下降 20.09%。它包含终点 pending 任务已执行的部分工作，selective-history 服务计入一次。dispatch 与正常工作重叠，不可相加。整个作业时间还包括恢复、checkpoint、边界记录和输出，不等同于在线规划或模拟物理时间。

四臂的应急着陆、能源失败、分流、终点 upstream pending、未完成且 SOC<0.25 的飞机数均为 0；每臂终点均有 1 个计算任务 pending。共同终点守恒通过：中流量已准入 40 对应 15+25 或 10+30，高流量 49 对应 17+32 或 13+36。pending 既不算成功，也不算失败。

| 场景 / 方法 | 终端能耗 MJ | 累计 holding s | 最大审查年龄 s | 最大实际进展年龄 s | 普通局部任务 / overflow 全量降级 |
|---|---:|---:|---:|---:|---:|
| s3r23 / FULL_H | 562.743999 | 4555.688583 | 20.546750 | 254.451955 | 0 / 0 |
| s3r23 / RULE_LOCAL_K8 | 568.420284 | 6120.769139 | 81.335804 | 294.863392 | 17 / 0 |
| s3r24 / FULL_H | 830.264069 | 7675.312467 | 42.339095 | 299.247314 | 0 / 0 |
| s3r24 / RULE_LOCAL_K8 | 798.315905 | 9040.631352 | 73.257375 | 282.994080 | 11 / 3 |
中流量局部方法能耗略高；高流量局部能耗较低，同时少完成 4 次正常着陆，不能据此宣称能源效率改善。局部方法在两场景的 holding 均更多。高流量局部臂的 14 个任务中，11 个是普通局部任务，3 个为 mandatory overflow 后的全量降级；后者不能归为局部加速。

### 任务终态延迟与期限

原字段 availability_delay_s 在此称“快照至任务终态延迟”：统计所有已结束任务，包括拒绝和无可执行前缀，不是仅成功交付计划的可用延迟。pending 不进入分布。late rate 分母为所有已结束任务，分子是明确 readiness/expiry/revalidation deadline 拒绝原因计数。

| 场景 / 方法 | 总任务 / 已结束 / pending | late / 已结束 | 终态延迟总计 s | P95 / max s |
|---|---:|---:|---:|---:|
| s3r23 / FULL_H | 15 / 14 / 1 | 1 / 14 (7.14%) | 265.576552 | 20.000000 / 20.000000 |
| s3r23 / RULE_LOCAL_K8 | 17 / 16 / 1 | 1 / 16 (6.25%) | 255.438652 | 20.000000 / 20.000000 |
| s3r24 / FULL_H | 13 / 12 / 1 | 2 / 12 (16.67%) | 269.586494 | 57.226460 / 57.226460 |
| s3r24 / RULE_LOCAL_K8 | 14 / 13 / 1 | 2 / 13 (15.38%) | 253.224110 | 20.000000 / 20.000000 |
高流量 FULL_H 的终态延迟 P95/max 为 57.226460 秒，其他三臂为 20 秒；四臂固定 lead 始终是 20 秒，不能将分布差异写成协议提前量改变。世界 advance 减 dispatch 还包括积分、执行检查和记录，不是独立测量的纯物理开销。离线审计另行计时，不混入在线正常工作。

## 八个同状态开发快照

每对使用同一预注册检查点、全局 frame、20 秒可见投影和冷缓存。16 条结果均为 PREPARED_COUNTERFACTUAL，准备 token 随后丢弃，反事实提议未执行。按每次调用自身开放集合判断，未选飞机的新候选物化次数均为 0。

CP 规模取嵌套 scheduler_records[].selective_replanning.model_scale[] 中每臂决策变量的最大值；字典序阶段会重复记录，不能跨阶段相加。返回 K 与各次开放 ID 的尝试并集 K 分列。

| 场景 / 快照 s | FULL_H K | 局部返回 K / 尝试并集 K | 最大 CP 决策变量 FULL / 局部 | 观测计算 s FULL / 局部 |
|---|---:|---:|---:|---:|
| s3r23 / 15 | 19 | 8 / 8 | 73 / 41 | 12.614239 / 8.147378 |
| s3r23 / 30 | 18 | 8 / 8 | 70 / 35 | 12.473045 / 7.178080 |
| s3r23 / 45 | 17 | 8 / 8 | 62 / 37 | 12.588183 / 7.270992 |
| s3r23 / 60 | 18 | 8 / 8 | 89 / 47 | 12.542052 / 8.544697 |
| s3r24 / 60 | 31 | 8 / 8 | 112 / 29 | 12.997504 / 8.335477 |
| s3r24 / 120 | 35 | 8 / 8 | 129 / 56 | 13.479960 / 8.351293 |
| s3r24 / 180 | 34 | 8 / 11 | 121 / 38 | 13.926421 / 8.612543 |
| s3r24 / 240 | 33 | 8 / 12 | 121 / 31 | 13.278173 / 8.761549 |
观测计算均值为 FULL_H **12.987447283347137 秒**、局部 **8.150251280589146 秒**，按均值之比约下降 **37.24%**。局部五分量预测目标 `(F,-L,risk,Ehat,tau)` 在八对中均字典序更小，仅限这些既有 H 开发状态。

该成本包括公共投影/frame、重置、初始化、规划及一次准备，不是完整在线流水线：未观察未来交通重验、服务排队和真实交付。最后两对虽返回 K8，尝试并集分别为 11/12。较小预测目标未在本次闭环转化为更多正常着陆，不能推出实际质量不劣、学习收益或统计显著改进。

### 节省发生在哪些工作，哪些公共成本仍保留

快照记录中的平均 callback wall 从 12.141398 秒降至 7.696194 秒，一次准备从 0.701692 秒降至 0.304257 秒；这些阶段是本次观测开销下降的主要位置。callback 包含选择、候选构造、联合求解和认证，当前总计不能唯一拆成独立 CP 收益。两方法均计入相同的投影 0.128170 秒和全局 frame 0.015695 秒；重置与初始化也单独计入。完整可见背景、全局需求评价和物理认证保留，闭环还保留公共监测、应急、准入、真实交付服务及执行开销，未因局部集合变小而省略。

同状态开销减少发生在两方法均未实际执行反事实提议的比较中，因此不只是“少完成真实动作”产生的数字。但闭环确实少着陆、holding 和部分审查年龄更高；当前单次配对不能把闭环工作节省与服务量变化独立归因，也不能据快照结构收益抵消实际吞吐损失。

## 保存回放的自动物理复核

四臂审计进程均退出 0，audit_result 均 PASS、full_a20_covered=true；PVA 为 CHECKED，C2 和资源为 PASS，连续间隔为 CERTIFIED，未交付/拒绝提议非执行归因为 PASS。以下非空覆盖量已由服务器原始审计核对，汇总 flags 与本地同步摘要一致。

| 场景 / 方法 | 状态帧 | 累计预约检查条目 | C2 检查 | 缺失前驱 | PVA 样本 | 非执行归因检查 |
|---|---:|---:|---:|---:|---:|---:|
| s3r23 / FULL_H | 491 | 17993 | 47 | 0 | 11794 | 4 |
| s3r23 / RULE_LOCAL_K8 | 448 | 14488 | 19 | 0 | 11352 | 11 |
| s3r24 / FULL_H | 476 | 23165 | 52 | 0 | 16556 | 4 |
| s3r24 / RULE_LOCAL_K8 | 453 | 20257 | 23 | 0 | 16182 | 4 |
最大 P/V/A 误差分别不超过 1.07×10⁻¹² m、1.07×10⁻¹³ m/s、1.51×10⁻¹⁴ m/s²，均通过相应容差。通过结果限定于本次保存执行记录与数值模型，不代表任意未来工况的零失败保证。实际人工回放尚未观看。

## 协议与适用边界

仅阶段 A：继承 downstream_service_v8 目标及 global_preselection_frozen_v1 frame。中流量 seed7324、高流量 seed7325，每方法一次 300 秒，没有 drain。CPU12 串行配对，CP/数值库各一线程；20 秒 lead、5 秒心跳、1 秒准备、4 秒交接；candidate/CP/hint/soft/watchdog 为 6/8/3/30/60 秒，尝试额度 32/1024。原始物理、应急、准入、服务与连续安全机制保持绑定规则。详见 [协议及入口命令](selective_reoptimization_A.md)。

仅有两个开发根、八个已有轨迹快照和单次配对，未建立多种子统计推广、学习收益或无损在线改进。新请求在 300 秒均未着陆，固定窗口与 20 秒接入限制了长期服务解释。目录日期为 20261027，实际服务器执行跨入北京时间 2026-10-28。

## 验收与预算

当前服务器专项入口 pad_fix_all.xml 为 **72 passed、0 skipped、0 failed**，包含真实 CP 的旧重复区间 INFEASIBLE/修复后 OPTIMAL，以及不同 owner 冲突仍保留的检查。历史兼容检查 72 passed/1 skipped 独立归档；缺少依赖分支因服务器 OR-Tools 可用而跳过，不与当前 72 项相加。本地缺少 OR-Tools 的项曾明确跳过，未替代服务器真实 CP 检查。

- A01–A19：PASS_TARGETED_AND_IMPLEMENTATION_REVIEW，测试对应见下表。
- A20：PASS_AUTOMATED_SAVED_REPLAY，四臂具备上述实际覆盖。
- X01：PASS_AT_COMMON_300S，四臂共同终点守恒通过。
- X02：HUMAN_PENDING；只读 S/B/M、阻塞边、倒计时标注测试通过，实际人工观看未完成。

最终账本 paused、active_phase=null：累计 **3277.6263247883435 秒**（约 54.63 分钟），剩余 **3922.3736752116565 秒**，上限仍为 **7200 秒**。采用打包和本地回放验证记账后的最终 budget_ledger.json，不采用摘要内较早的账本快照；保守 elapsed floor 略高于 phase_used。失败、失效实现、修复和交付验证均保留，未重置。科学源码与报告交付分别版本化；科学绑定不因报告更新而改变。

| 条目 | 测试名称或直接对应证据 |
|---|---|
| A01 | `test_A01_off_fixed_work_catalog_and_no_extra_latency_snapshot` |
| A02 | `test_A02_scope_partition_and_decisions_complete_world` |
| A03 | `test_A03_all_candidate_entrypoints_and_cold_reference_scoped`；`test_A03_actual_selected_materialization_unselected_zero` |
| A04 | `test_A04_background_future_crossing_uses_full_pair_check` |
| A05 | `test_A05_resource_blocker_and_expanded_joint_legal_inward_actions`；`test_A05_A11_bounded_expansion_recertifies_with_same_budget` |
| A06 | `test_A06_A07_committed_background_resources_release_and_no_mutation` |
| A07 | `test_A06_A07_committed_background_resources_release_and_no_mutation`；`test_A07_cp_real_reservation_identity_deduplicated`；`test_A07_ground_frame_covered_service_interval_only_once`；`test_A07_ground_frame_keeps_other_owner_and_partly_covered_intervals`；`test_A07_ground_frame_real_cp_old_duplicate_infeasible_normalized_feasible` |
| A08 | `test_A08_background_queue_wait_and_energy_not_constant` |
| A09 | `test_A09_A10_fixed_follow_equivalence_and_frozen_frame`；`test_A09_cp_frozen_background_matches_full_fixed_follow` |
| A10 | `test_A09_A10_fixed_follow_equivalence_and_frozen_frame`；`test_A10_follow_frozen_despite_reference_cache_changes` |
| A11 | `test_A11_scope_cache_and_cut_identity_invalidation`；`test_A05_A11_bounded_expansion_recertifies_with_same_budget` |
| A12 | `test_A12_unknown_not_cached_safe_or_cut` |
| A13 | `test_A13_A19_delivery_ages_and_roundtrip_not_selection_or_execution`；`test_A13_imported_unknown_and_completed_actual_start_recovery`；`test_A13_expected_committed_completion_allows_delivered_review` |
| A14 | `test_A14_mandatory_overflow_explicit_full_shared_scope` |
| A15 | `test_A15_empty_full_single_and_keep` |
| A16 | `test_A16_only_revealed_snapshot_and_selector_soc_sensitive`；`test_A16_X01_revealed_upstream_identity_mask_without_fictitious_energy` |
| A17 | `test_A17_full_long_background_horizon` |
| A18 | `test_A18_frame_budget_exhaustion_is_explicit_unknown`；`test_A18_callback_frame_unknown_returns_no_proposal`；`test_A18_A19_real_latency_delivery_history_and_no_early_progress` |
| A19 | `test_A19_callback_binding_roundtrip_and_phase_limits`；`test_A19_real_selective_checkpoint_restore_all_pending_stages`；`test_A18_A19_real_latency_delivery_history_and_no_early_progress`；`test_A19_multiaircraft_follow_frozen_diagnostics_checkpoint_roundtrip` |

## 可直接使用的本地回放

服务器原始结果完整。原 terminal-results receive-only 同步曾出现删除错误和约 1 小时 4 分钟后的重试，导致嵌套文件迟到。为交付回放，仅将服务器输出打包为 ZIP，经 Syncthing 顶层同步后在新目录解压；没有手工传输源码，也未改同步服务配置。

可用本地目录为 **`D:/7_codespace/training_outputs/terminal/rs_selective_reoptimization_a_20261027_review`**。ZIP 为 25,199,032 字节，SHA256 `8799431d686653dd17af4dda869d1e17c67bf989306b8d2f735e4d51746a19b6`；125 个 payload 哈希全部通过，四个 ReplayBundle.load/validate 均通过，状态帧分别 491/448/476/453。该验证是程序加载，不是人工观看，HUMAN_PENDING 不变。

```python
from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer
controller = ReplayController('D:/7_codespace/training_outputs/terminal/rs_selective_reoptimization_a_20261027_review/evaluation/s3r23/RULE_LOCAL_K8/replay')
RotatingSlotsReplayViewer(controller).run()
```

- [最终结构化评审](../../artifacts/rs_select_a_20261027/primary_review.json)、[本地交付验证收据](D:/7_codespace/training_outputs/terminal/rs_selective_reoptimization_a_20261027_review/review_validation.json)。
- [原始汇总](D:/7_codespace/training_outputs/terminal/rs_selective_reoptimization_a_20261027/analysis_summary.json)、[共同终点 CSV](D:/7_codespace/training_outputs/terminal/rs_selective_reoptimization_a_20261027/system_counts.csv)、[最终预算账本](D:/7_codespace/training_outputs/terminal/rs_selective_reoptimization_a_20261027/budget_ledger.json)。摘要 SHA256 与最终账本 SHA256 记录在评审 JSON，未改写实验输出。
- 服务器根 `/home/user/training_outputs/terminal/rs_selective_reoptimization_a_20261027`；各臂原始边界、任务、checkpoint、ReplayBundle 和五类 audit JSON 位于 `evaluation/<root>/<method>/`，本地 review 目录保存交付副本。

## 附：实现修复与失效记录

源码链为核心 5969da6、运行器 bf81571、JSON 键摘要 cb818f1、终端 Section 输出 e997b93、选择性 checkpoint 序列化 036abfd，最终固定 pad 表示修复 750cda6。早期首对在终端输出失败前已写完，当时保留结果并只接续其余七对，没有因输出失败重跑首对。

早期 t5 checkpoint 的冻结 FOLLOW 诊断序列化故障经 036abfd 修复；真实 27 架、10 秒 checkpoint 恢复状态哈希及 ReplayBundle 验证一致。随后确认旧 frame 将地面服务当前时刻截断区间和原始实际日历重复加入 CP。真实 10 秒诊断原 all-background CP 为 INFEASIBLE，仅移除三条被同 owner 日历完整覆盖的需求区间后为 OPTIMAL，物理状态、日历和 assignment 不变。诊断 JSON SHA256 为 `c66292167bcfa7c142e30d4b37c003f5bd37cee54800e5c904976d10625406be`。

旧八对快照、首个 300 秒闭环和第二臂 155 秒部分结果整体失效，归档至 invalidated_cp_generation/，不计入本稿结果。750cda6 只去掉同 pad、同 owner 且被真实日历完整覆盖的 baseline 重复区间，保留其他 owner 和部分延伸；没有改变物理、资源执行器、选择器、协议或 OFF 路径。这是实现正确性修复，不是质量调参；pad_fix_rebind.json 和同一账本保留归档、旧结果及全部计算成本。

早期辅助命令尾随 `.` 造成无关全仓库收集的 78 errors/1 skipped 也保留并计费，不作为专项通过或生产仿真结果。合同包始终保持原样，原 NOT_RUN 规范标记未改写。
