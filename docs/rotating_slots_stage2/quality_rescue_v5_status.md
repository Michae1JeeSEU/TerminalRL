# Stage2 quality/rescue v5：配对 480 秒结果与研究记录

正式配对状态：**COMPLETED_REQUESTED_480S**；程序退出码 **0**；两方法审计均 **PASS**、`issues=[]`；人工回放验收仍为 **HUMAN_PENDING**。本轮从 `2026-10-22T04:35:22.908098+00:00` 运行至 `2026-10-22T06:45:29.750601+00:00`，实际 wall 为 **7806.842503 秒（2 小时 10 分 6.84 秒）**。冻结源码为已推送提交 `77c3d16c4a4cb65467dc6e0e30686906e9d2ec32`，本次只更新实验事实和研究记录。

**计算与安全检查完成不等于质量目标达成。** 在同一 480 秒边界，v5 的正常着陆和总着陆均少于旧 v4；CP 已摆脱未进入求解的状态，并多次改进自身 hint 的预测目标，但完整系统的着陆表现没有改善。该窗口仍有 44/42 架飞机在空中，不能据当前零能量失败推断长期能量安全。结果、来源和字段口径见 [v5 最终事实](../../artifacts/rotating_slots_stage2_20261022/v5_final_0480_facts.json)。

## v5 前的诊断历史

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

首次 Slack callback 约 25.6 秒，已超过旧 5 秒参考线。主对话据此在 prepare 前将本轮总 wall time 上限确定为 **6 小时**：计算最多 5.5 小时，归档预留 0.5 小时，取代此前 4 小时草案。预检时间计入本轮计算成本，保守计入 900 秒预检 credit；这覆盖已知预检成本，不代表恰好测得 900 秒。旧运行约 365 次决策；若每次约 25–30 秒，再计监控开销，初始总耗时 `T0` 粗估约 3 小时。该外推不能覆盖本轮交通密度、事件数和每次规划成本的变化。本轮最终双方均完成 480 秒，预算未触发停止。预算、候选配置与应急策略共同改变，配对结果只支持这一组合设置下的比较，不能单独归因于任一改动。

## 正式运行启动记录

唯一正式输出根目录为 `/home/user/training_outputs/terminal/stage2_quality_rescue_480s_v5_20261022`，同步本地目录为 `D:/7_codespace/training_outputs/terminal/stage2_quality_rescue_480s_v5_20261022`。服务器工作目录为 `/home/user/projects/terminal`，运行命令为：

```bash
/home/user/envs/terminal_stage2_v1/bin/python -u /home/user/projects/terminal/scripts/server/run_rotating_slots_stage2.py run --output /home/user/training_outputs/terminal/stage2_quality_rescue_480s_v5_20261022 --target 480
```

wrapper PID 为 136680，runner PID 为 136682；方法 worker PID 为 136685（CPU 0）和 136686（CPU 2），每进程数值库线程为 1。运行日志为 `campaign.log`；实际终态须核对 `exit_code.txt`、`campaign.timing.txt` 和完整输出。

prepare 的原始进程 wall 为 **2.738124370574951 秒**，退出码 0；预算记录的 `elapsed_floor_s=902.4746096134186` 包含 900 秒 credit，不能读成 prepare 实测耗时 902 秒。总预算为 21600 秒，compute stop 为 19800 秒，归档预留 1800 秒。fixture、manifest 和 environment lock 的字节 SHA-256 均与旧 v4 输入相同；完整值及本轮 source/config hash 见 [验证事实](../../artifacts/rotating_slots_stage2_20261022/v5_validation_facts.json)。

启动复查快照：服务器 UTC 为 `2026-10-22T04:38:03.486070+00:00`，本地 UTC 观测为 `2026-10-22T04:37:12.8979462+00:00`，两时钟观测相差约 50.6 秒；runner 累计约 160 秒。最新状态的仿真时间为 Slack **17 秒**、CP **14.208984375 秒**，两方法 checkpoint 仍为 0 秒，尚无 60 秒 summary。该次复查未见 error、invalid、退出或终态 timing；这些启动证据不代表最终成功。

启动时曾按 `T0=10800` 秒和累计 160 秒粗估剩余 `R=10640` 秒，并成功创建当前对话 heartbeat `stage2-v5-480s`，间隔 177 分钟、计划约 `2026-10-22T07:35Z`（北京时间 15:35）复查。该记录是历史估计；本轮完成核查后，heartbeat 已成功删除，不再处于 ACTIVE 状态。源码保持冻结。

## 预检与实现检查历史

`newly_run`：本地执行 `venv/Scripts/python.exe -m pytest tests/test_stage2_emergency_priority_v5.py tests/test_rotating_slots_stage2_world.py tests/test_rotating_slots_world.py -q`，结果 **84 passed in 45.73s**，含 15 项新优先级测试；`git diff --check` 通过。覆盖紧迫排序、未知 q fallback、纯评价不改 world、普通 final 与 transfer 的门禁差异、soft 安全撤销、protected/active 保留、取消未认证时保留原承诺，以及未刷新 eligibility 的直接提交入口。

`newly_run`：服务器集成检查 **210 passed, 1 skipped, 8 deselected in 54.81s**；跳过的是 OR-Tools 缺失分支的反向测试，因为服务器已安装 OR-Tools。另一本地目标组 **8 passed in 4.51s**。refinement 修复后服务器目标组为 **39 passed, 1 skipped**，进程计时 **9.549 秒**（pytest 报告 9.38 秒）。这些测试组存在重叠，分别记录，不相加为唯一测试总数。

`newly_run`：首次繁忙 fixture callback 预检产物为 `D:/7_codespace/training_outputs/terminal/stage2_quality_rescue_v5_validation_20261022/callback_smoke.json`。Slack 在约 **25.63 秒**提交 26 个动作；CP 在约 **15.92 秒**提交 0 个动作，虽进入 6 次 phase 求解调用，但 3 轮 refinement 的 phase 1 均出现 `HARD_SEPARATION` 不安全结果，固定 3 轮上限使当次搜索提前停止，因此当时暂缓正式配对并修复 refinement。

`newly_run`：修复后 `callback_smoke_refinement.json` 记录如下。callback wall 与外围计时分别保留，不能混用。

| 方法 | callback wall 秒 | 外围计时秒 | 提交动作 |
| --- | ---: | ---: | ---: |
| Slack | 26.532590 | 26.598908 | 26 |
| CP | 31.575586 | 31.636674 | 11 |

CP 进行了 17 轮 refinement、40 次 phase 调用，solver 累计 wall 为 **0.082583327 秒**；其 hint 已认证，目标从 `[0, -5, 93, 30822, 618430825.2220923]` 变为最终 `[0, -7, 87, 29195, 588758213.9691724]`，记录 `strict_improvement_over_hint=true`。这是同一 callback 内相对已认证 hint 的预测目标改进；其中预测着陆项不是已发生 touchdown。该单次观察不能用于宣称某方法优于另一方法，也不能代表 480 秒轨迹表现。

上述检查属于实现及预检历史，测试组不能相加为唯一测试总数。本次终态整理未修改冻结代码，未新增运行测试；使用最终 JSON 聚合、守恒和安全审计、归档完整性及回放加载核查结果。自动检查与主对话查看曲线均不能代替人工回放验收。

| 运行记录项目 | 最终值 |
| --- | --- |
| v5 source commit | `77c3d16c4a4cb65467dc6e0e30686906e9d2ec32`，已推送 |
| 当前分支与 3 个代码/4 个测试的本地 SHA-256 | 见 [v5_validation_facts.json](../../artifacts/rotating_slots_stage2_20261022/v5_validation_facts.json)；分支 `codex/arrival-restructure` |
| fixture / manifest hash | 已核对与旧 v4 字节相同；完整 SHA-256 见验证事实 |
| 输出绝对路径 | `/home/user/training_outputs/terminal/stage2_quality_rescue_480s_v5_20261022` |
| 首次 callback 预检时间 | `2026-10-22T04:15:35.383385+00:00` 至 `2026-10-22T04:16:16.938426+00:00` |
| 修复后 callback 预检时间 | `2026-10-22T12:26:11.089043+08:00` 至 `2026-10-22T12:27:09.329736+08:00` |
| 本轮预检累计成本 | 保守计入 900 秒 credit；prepare 原始 wall 为 2.738124370574951 秒 |
| 正式开始时间 | `2026-10-22T04:35:22.908098+00:00` |
| 停止时间及完整仿真边界 | `2026-10-22T06:45:29.750601+00:00`；两方法共同完成 480 秒；退出码 0 |
| wall 与预算成本 | 实测 wall 7806.842503 秒；`elapsed_floor_s=8741.703433275223` 含 900 秒预检 credit；未触及 19800 秒 compute stop 或 21600 秒总预算 |
| 归档完整性 | 两方法各 28/28 个预期文件，无缺失或空文件；回放时间均覆盖 `[0,480]`；无 invalid 文件；审计均 PASS |
| campaign 原生状态 | `PARTIAL_REQUESTED_WINDOW`：本次请求的 480 秒已完成，原合同 3600 秒未运行；不是本轮 480 秒中途失败 |
| heartbeat | `stage2-v5-480s` 已成功删除 |
| 人工回放结论与观察时间 | `HUMAN_PENDING` |

## 480 秒配对结果

`preexisting_artifact`：以下结果来自本轮已归档输出；`newly_run`：最终事实 JSON 为本地只读聚合，核对 60 秒边界、守恒、事件、决策和回放文件，不重跑仿真。完整字段见 [v5_final_0480_facts.json](../../artifacts/rotating_slots_stage2_20261022/v5_final_0480_facts.json)。

| 480 秒指标 | energy_slack | rh_cp_sat |
| --- | ---: | ---: |
| offered / admitted | 67 / 67 | 67 / 67 |
| 正常 / 应急 / 总着陆 | 17 / 6 / 23 | 17 / 8 / 25 |
| 能量失败 | 0 | 0 |
| 空中 pending / 上游 pending | 44 / 0 | 42 / 0 |
| 初始空中 cohort 成功 / 36 | 22 / 36 | 24 / 36 |
| 新请求 cohort 成功 / 31 | 1 / 31 | 1 / 31 |
| 已声明应急 / 未完成应急 | 23 / 17 | 26 / 18 |
| 低于正常着陆储备的 touchdown | 2 | 5 |
| 360–480 秒新增正常 / 应急着陆 | 0 / 6 | 0 / 7 |
| 审计状态 / issue 数 | PASS / 0 | PASS / 0 |

守恒关系分别为 `67=23+44` 与 `67=25+42`。审计的 `continuous_separation=CERTIFIED`，而 `physical_outcome=IN_PROGRESS`：仍有空中个体，不构成全任务完成或长期能量失败风险消失的证据。低于正常储备的 touchdown 不能改写为能量失败；应急着陆与正常着陆的能量口径须保持区分。

| 同一 480 秒边界 | v4 Slack | v5 Slack | v4 CP | v5 CP |
| --- | ---: | ---: | ---: | ---: |
| 正常着陆 | 26 | 17 | 25 | 17 |
| 应急着陆 | 1 | 6 | 3 | 8 |
| 总着陆 | 27 | 23 | 28 | 25 |
| 空中 pending | 40 | 44 | 39 | 42 |
| 能量失败 | 0 | 0 | 0 | 0 |

因此，本轮未达到提高正常及总着陆效率的质量目标。v5 的正常着陆相对旧 v4 分别减少 9/8 次，总着陆减少 4/3 次；应急着陆增加不能单独抵消上述结果。该比较只有一个冻结 fixture/manifest，且预算、候选生成与优先级同时改变，没有受控消融，不能把全部差异单独归因于应急门禁，也不能作跨场景优越性推断。

## 计算开销与 CP 作用范围

| 最终决策统计 | energy_slack | rh_cp_sat |
| --- | ---: | ---: |
| 正常规划调用数 | 355 | 354 |
| 正常规划 wall 均值 / P95（秒） | 17.4487 / 20.0723 | 19.1182 / 26.2773 |
| dispatch 记录均值 / P95（秒） | 19.0252 / 22.3819 | 21.2108 / 29.2990 |
| 进入 solver 的决策 | 0 | 354 / 354 |
| 已认证 hint | 不适用 | 354 / 354 |
| 严格改进自身 hint 的预测目标 | 不适用 | 264 / 354（74.58%） |

正常规划 wall 包含 dispatch 中的 normal 规划阶段；dispatch 记录是监控中累计的工作量，不应解释为严格连续的单次端到端响应延迟。两方法所有 normal 调用均超过旧 5 秒参考线。CP 的 90 次决策未记录相对 hint 的严格改进；决策状态为 350 次 `LEX_LIMITED`、4 次 `CATALOG_LEX_OPTIMAL`。13135 次 phase 求解均报 `OPTIMAL`，但这是受限候选 catalog 内各 phase 的解状态，不能表述为完整连续联合调度问题的全局最优。CP 未进入求解的饥饿问题在本轮已解决，其优化确实改变自身预测目标；这与完整系统着陆质量改善是不同层次的结论。

CP phase 求解累计 wall 为 13.260924 秒；独立计时的 `cp_solve` 阶段为 13.284436 秒，占 scheduler 总计 6767.094087 秒约 0.20%，两者不能重复相加。候选相关四阶段占约 68.60%，hint 构造与验证约 13.17%，selected validation 约 15.45%。主要耗时仍在候选构造与验证；当前证据不支持仅继续增加 CP 时间预算即可解决系统表现。

## 应急优先门禁与待救援状态

普通 final 最后实际开始时刻分别为 Slack **280 秒**、CP **225 秒**；最后正常 touchdown 分别为 **358.5 秒**、**318.5 秒**。两方法首次应急触发分别为 **291.2 秒**、**227.2 秒**，之后均无新的普通 final 开始，360–480 秒也没有新增正常着陆。

候选生成中 `EMERGENCY_PRIORITY_WAIT` 分别记录 **10759/13499** 次，真实 world 提交入口分别拒绝 **116/103** 次。前者是候选尝试次数，后者也是入口拒绝事件数，均不是不同飞机的数量。结合实现可确认：`emergency_priority_blocker` 发现紧迫 HOLD 队首时，会全局阻挡其他飞机的普通 final，没有按具体 pad 和时间窗区分对救援的影响。既有例外是候选恰为队首本机、位于 inner 且尚未 declared/pending 时，门禁不拦截其普通 final；该候选仍须通过正常着陆能耗、储备及全部安全认证。

本轮没有成功的 soft final 让位事件：两方法 `emergency_priority_yielded` 均为 0；`emergency_priority_yield_blocked` 分别为 42/241 次，原因全部是 `MANEUVER_ALREADY_STARTED`。安全撤销能力已实现，不等于本轮实际发生成功撤销；已开始的动作保持执行，与新的普通 final 门禁生效须分别记录。

主对话据此判断，该规则对联合吞吐过于粗糙。下一步拟议方向是优先保护**具体 pad/时间窗内可行的救援**，允许不会推迟紧迫救援的普通 final；继续保留分离、能量、资源认证及已开始轨迹保护。该方向尚未实现，本记录不授权自动启动 v6 或重启实验，也不引入安全检查绕过或执行中瞬间撤销。

480 秒仍在空中的低 SOC（≤0.35）个体如下。第一组按序列化的 ring 字段统计；active transfer/final 期间该字段可能仍表示来源环，不能一概称为在该环等待。

| 480 秒低 SOC 口径 | Slack 外 / 中 / 内 | CP 外 / 中 / 内 |
| --- | ---: | ---: |
| 全部 airborne，按 ring 字段 | 0 / 1 / 16 | 0 / 6 / 12 |
| 仅 `HOLD_*` 状态 | 0 / 1 / 12 | 0 / 6 / 9 |

记录中的队列 key 排序分别通过 **143/143**、**192/192** 次检查，未见逆序事件。排序紧迫度与实际启动次序可能不同，不能据后启动者排在队首之后就认定排序失效。例如 Slack 在 **324.8 秒**记录队首为 `initial_outer_02`，其紧迫估计 202.91 秒，实际启动的 `initial_middle_11` 排第 4、估计 230.83 秒；启动之前队首已有覆盖 6 个 pad 的 6 次拒绝（4 次 `HARD_SEPARATION`、2 次 `RESOURCE_CONFLICT`）。CP 在 **271.83165 秒**先尝试队首 `initial_middle_09`，记录 5 次 final 拒绝（3 次分离、2 次资源），随后 `initial_outer_00` 开始；同时间另一个 pad 只有候选搜索耗尽记录，不能当作不可行性证明。CP 在 **370 秒**也先记录队首 `initial_inner_09` 的 6 次拒绝，再开始 `req_000000`。这说明队首先尝试与实际先启动、先着陆是不同事件，不能解释成按 ID 排序；上述记录也不证明队首在其他时间或未搜索轨迹上的反事实不可行性。

| 方法 / 个体 | 应急声明（秒） | 应急实际开始（秒） | touchdown（秒） |
| --- | ---: | ---: | ---: |
| Slack `initial_outer_02` | 291.2 | 340.0 | 393.5 |
| Slack `initial_middle_11` | 324.8 | 324.8 | 378.3 |
| CP `initial_middle_09` | 227.2 | 329.5562 | 383.0562 |
| CP `initial_outer_00` | 253.4 | 271.8317 | 320.3317 |
| CP `initial_inner_09` | 299.0 | 380.0 | 458.5 |
| CP `req_000000` | 354.0 | 370.0 | 423.5 |

另一个直接成功救援实例是 Slack `initial_inner_00`：399.4 秒声明并在 P4 开始，447.9 秒应急 touchdown，SOC 为 0.2826608176。

全部个体的声明、实际启动、touchdown、480 秒状态和等待时长保存在最终事实 JSON 的 `emergencies.aircraft`；未启动者的等待截断至 480 秒，不能当成已完成救援延迟。

## 回放与人工验收入口

`newly_run`：[回放加载检查](../../artifacts/rotating_slots_stage2_20261022/v5_replay_load_check.json)对两方法分别 seek 到 `0/270/300/360/396/420/450/480` 秒，合计 **16 次加载均通过**。主对话已查看两方法曲线；Slack/CP 回放分别有 **748/751 帧**，时间均覆盖 `[0,480]`。这些检查证明文件可加载并支持观察，人工动画验收状态仍为 **HUMAN_PENDING**。

曲线中的 pad 占用计数按 `LANDED_SERVICE` 状态表示物理地面占用。空 pad 可能存在未来预约或进近保护，不能直接等同于无预约，更不能由此证明新 final 可行。`pad_busy_seconds` 则累计预约区间内的时间，含 touchdown 之前的进近保护；它不是地面实际占用时长，不能与上述计数混用。

在 PowerShell 中运行 Slack 回放：

```powershell
Set-Location D:/7_codespace/terminal
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; c=ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_quality_rescue_480s_v5_20261022\energy_slack\replay_0480'); c.seek(360); RotatingSlotsReplayViewer(c).run()"
```

CP 回放：

```powershell
Set-Location D:/7_codespace/terminal
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; c=ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_quality_rescue_480s_v5_20261022\rh_cp_sat\replay_0480'); c.seek(360); RotatingSlotsReplayViewer(c).run()"
```
