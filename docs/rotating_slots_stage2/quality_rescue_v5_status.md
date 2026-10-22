# Stage2 quality/rescue v5：诊断与配对实验协议

正式配对状态：**NOT_LAUNCHED**；实现与预检状态：**VERIFIED_FOR_LAUNCH**；人工回放验收：**HUMAN_PENDING**。主对话已验收当前实现，正式运行待提交版本后启动。本文不代表 v5 配对实验已经启动或取得结果。

## 当前诊断

`preexisting_artifact`：[396 秒回放事实](../../artifacts/rotating_slots_stage2_20261022/v5_review_396_facts.json)显示，CP 方法的 UID 15 对应 `initial_middle_02`。其普通 `FINAL_NORMAL` 在仿真 385 秒同时提交并开始，计划结束于 438.5 秒；P5 预约为 `[425, 498.5)`，此前地面服务占用结束于 407.6761 秒。因此 396 秒时该飞机已飞行约 11 秒，不能把它当作尚未启动的预约直接删除。需要检验的策略问题发生在 385 秒的提交与启动决策，而非 396 秒执行中的瞬间撤销。

`preexisting_artifact`：[低 SOC 个体事实](../../artifacts/rotating_slots_stage2_20261022/v5_outer_low_soc_facts.json)中，两方法合计 7 个在 480 秒仍处于外环或中环且 SOC ≤ 0.35 的个体，入场或首帧 SOC 均约为 0.624–0.751。它们并非以低电量入场；但这些轨迹和候选记录不能证明每一次未执行的跨环转移当时都可行，也不能仅据此归因于规划时间不足。

原合同 §12.4 规定正常规划软预算 5 秒、watchdog 30 秒；§10.3 规定 CP 五层目标共享 2 秒 solver 预算。相关设置在仓库首次记录于 `6d7d20d8be060b3cf405190cfdf3b087dd1f3e79`，时间为 `2026-10-21T01:31:09+08:00`。旧合同及旧配置保留；v5 使用显式新预算 profile，5 秒继续作为延迟报告参考线。

## v5 实现与策略取舍

两方法共用应急优先规则。对 HOLD 且没有 active 轨迹的飞机，已声明、已 eligible，或当前已知信息满足 `emergency_threshold(SOC, q_e, 10 × holding_rate)` 时，纳入待救援队列。该纯查询规则也用于规划快照评价与真实提交，防止直接提交时应急标志尚未刷新的漏窗；跨环转移不受普通 final 门禁阻挡。

紧迫值估计为 `(SOC − 已知 q_e) / holding_rate`，其中 `holding_rate` 来自当前位置的既有 holding 能耗函数；随后以 SOC、声明时间和 ID 稳定排序。未知 `q_e` 时明确记录 `q_e_unknown`，退回 `SOC / holding_rate`，阈值判断只使用 SOC 条件。该值用于排序，不是救援轨迹可行性或剩余安全时间的证明；排序本身不搜索完整 reference。

存在待救援飞机时，新的普通 `FINAL_NORMAL` 受到 `EMERGENCY_PRIORITY_WAIT` 门禁，并记录 `blocked_by`。最高紧迫、尚未声明的内环飞机仍可使用既有 normal exception，前提是正常着陆储备与全部安全检查通过。未开始的普通 soft final 通过既有 cancel/submit 事务尝试让位，启动前再次检查。若撤销未获认证，包括 `RESOURCE_DEPENDENCY`、分离检查失败或 `UNKNOWN`，保留原已认证承诺并记录不能让位原因，继续原启动检查；不把撤销失败视为可以安全 holding。已开始的轨迹保留，本轮没有引入安全复飞模型或瞬移式释放资源。

强优先可能使 pad 在紧迫飞机暂时无法救援时闲置。这是本轮明确的效率取舍；不能预先声称全局最优，或保证不再出现外环低 SOC。

## 配对 480 秒协议

比较方法为 `energy_slack` 与 `rh_cp_sat`；沿用同一 fixture、manifest、冻结随机输入及 `service_half_v1`（短/标准/长服务分别为 30/60/120 秒），新建输出根目录 `stage2_quality_rescue_480s_v5_20261022`。旧运行及其耗时单独保留，不覆盖或并入本轮成本。

| `quality_v5` 设置 | 值 |
| --- | ---: |
| normal soft budget | 30 |
| watchdog | 60 |
| candidate budget | 12 |
| hint budget | 3 |
| shared solver budget | 8 |
| solver reserve | 8 |
| hint validation reserve | 0.5 秒 |
| max refinement rounds | 64 轮 |

上述时间属于同一轮规划的阶段约束与保留量，不应相加为每轮允许耗时。64 轮 refinement 共用原始 30 秒 soft 时钟和累计 8 秒 solver 预算，不逐轮重置；hint 留出 0.5 秒供认证。`UNKNOWN` 验证结果不转成排除候选的 cut，只有已证不安全结果可驱动相应排除。实际预算设置含上述字段并进入来源记录。完整规划仍报告真实 wall time、超时及 5 秒参考线命中情况。

对正常剩余耗电未知的候选，slack 采用 `(SOC − 0.25) / holding_rate − remaining_normal_time_lower_bound`。这里的 lower bound 及 UNKNOWN slack 仅是候选排序提示；不能替代真实能耗、储备、资源和分离认证，也不能据正 slack 宣称正常着陆可行。

首次 Slack callback 约 25.6 秒，已超过旧 5 秒参考线。主对话据此在 prepare 前将本轮总 wall time 上限确定为 **6 小时**：计算最多 5.5 小时，归档预留 0.5 小时，取代此前 4 小时草案。预检时间计入本轮计算成本，拟保守扣计 900 秒预检 credit，最终须核对实际成本。旧运行约 365 次决策；若每次约 25–30 秒，再计监控开销，初始总耗时 `T0` 粗估约 3 小时。该外推不能覆盖本轮交通密度、事件数和每次规划成本的变化，须据有效运行进度更新。是否达到双方完整 480 秒、实际停止位置及归档完整性仍须据运行结果记录，预算不是完成保证。预算与应急策略同时改变，配对结果只支持这一组合设置下的比较，不能单独归因于任一改动。

## 已有检查与待填记录

`newly_run`：本地执行 `venv/Scripts/python.exe -m pytest tests/test_stage2_emergency_priority_v5.py tests/test_rotating_slots_stage2_world.py tests/test_rotating_slots_world.py -q`，结果 **84 passed in 45.73s**，含 15 项新优先级测试；`git diff --check` 通过。覆盖紧迫排序、未知 q fallback、纯评价不改 world、普通 final 与 transfer 的门禁差异、soft 安全撤销、protected/active 保留、取消未认证时保留原承诺，以及未刷新 eligibility 的直接提交入口。

`newly_run`：服务器集成检查 **210 passed, 1 skipped, 8 deselected in 54.81s**；跳过的是 OR-Tools 缺失分支的反向测试，因为服务器已安装 OR-Tools。另一本地目标组 **8 passed in 4.51s**。refinement 修复后服务器目标组为 **39 passed, 1 skipped**，进程计时 **9.549 秒**（pytest 报告 9.38 秒）。这些测试组存在重叠，分别记录，不相加为唯一测试总数。

`newly_run`：首次繁忙 fixture callback 预检产物为 `D:/7_codespace/training_outputs/terminal/stage2_quality_rescue_v5_validation_20261022/callback_smoke.json`。Slack 在约 **25.63 秒**提交 26 个动作；CP 在约 **15.92 秒**提交 0 个动作，虽进入 6 次 phase 求解调用，但 3 轮 refinement 的 phase 1 均出现 `HARD_SEPARATION` 不安全结果，固定 3 轮上限使当次搜索提前停止，因此当时暂缓正式配对并修复 refinement。

`newly_run`：修复后 `callback_smoke_refinement.json` 记录如下。callback wall 与外围计时分别保留，不能混用。

| 方法 | callback wall 秒 | 外围计时秒 | 提交动作 |
| --- | ---: | ---: | ---: |
| Slack | 26.532590 | 26.598908 | 26 |
| CP | 31.575586 | 31.636674 | 11 |

CP 进行了 17 轮 refinement、40 次 phase 调用，solver 累计 wall 为 **0.082583327 秒**；其 hint 已认证，目标从 `[0, -5, 93, 30822, 618430825.2220923]` 变为最终 `[0, -7, 87, 29195, 588758213.9691724]`，记录 `strict_improvement_over_hint=true`。这是同一 callback 内相对已认证 hint 的预测目标改进；其中预测着陆项不是已发生 touchdown。该单次观察不能用于宣称某方法优于另一方法，也不能代表 480 秒轨迹表现。

当前没有 v5 正式实验结果；测试通过不能代替配对回放和人工验收。

| 待填项目 | 当前值 |
| --- | --- |
| v5 source commit | `PENDING`（正式运行尚未启动） |
| 当前分支与 3 个代码/4 个测试的本地 SHA-256 | 见 [v5_validation_facts.json](../../artifacts/rotating_slots_stage2_20261022/v5_validation_facts.json)；分支 `codex/arrival-restructure` |
| fixture / manifest hash | `PENDING`（须与配对输入核对） |
| 输出绝对路径 | `$HOME/training_outputs/terminal/stage2_quality_rescue_480s_v5_20261022`（拟定） |
| 首次 callback 预检时间 | `2026-10-22T04:15:35.383385+00:00` 至 `2026-10-22T04:16:16.938426+00:00` |
| 修复后 callback 预检时间 | `2026-10-22T12:26:11.089043+08:00` 至 `2026-10-22T12:27:09.329736+08:00` |
| 本轮预检累计成本 | 拟扣计 900 秒 credit；实际总成本待最终核对 |
| 正式开始、停止时间；命令；PID/job ID | `NOT_LAUNCHED` |
| 两方法实际仿真边界、计算 wall time | `PENDING` |
| 归档时间与完整性 | `PENDING` |
| 人工回放结论与观察时间 | `HUMAN_PENDING` |
