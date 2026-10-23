# downstream_service_v8：medium 600 s 协议与状态

状态：唯一有效 paired 实验 `_r1` 的两方法均已完成 0→600 s，请求窗口完整共同边界为 600 s，`campaign.exit=0`；campaign 状态为 `PARTIAL_REQUESTED_WINDOW`，不表示全部 manifest 请求已完成。两份终态 audit 均为 PASS、issue_count=0。主对话已复核六张截图和 comparison curves，load/seek/render 自动视觉检查通过，人工验收仍为 `HUMAN_PENDING`。修复代码 `ecd49738f918d861125f4afee3814e2596c33176` 及冻结 provenance 保持不变；首次异常 root、其 provenance 与原始日志继续保留只读。

本轮将正常调度算法显式冻结为 `planning_profile=downstream_service_v8`；默认旧档案仍为 `progress_tail_v7`。candidate policy 与 objective schema 随 profile 保存到 provenance、campaign、checkpoint identity、summary 与 comparison。profile 独立于 `quality_v5` 预算：soft 30 s、solver 8 s、refinement 64 rounds 保持原样。新增下游服务预测与队列启发式不是可行性证明；执行仍需现有连续认证。

两种正常调度方法共享按字典序最小化的目标 `(F, -L, risk, Ehat, tau)`：保留失败优先和正常完成收益，随后比较估计的正常 reserve 缺口与不确定性代价、总完成能量和完成时间。每个 dispatch 固定一次虚拟 pad 分配及 energy-slack / release / id 队列顺序；候选选项改变 ready time、等待率、能耗和实际 pad 区间，CP 显式建模依赖所选选项的队列等待。虚拟服务只用于评价，不能创建实际预约，也不提供安全证明。该固定分配及顺序不搜索所有可能的队列，因此不保证全局最佳队列或 600 s 期末效果。

尾部 motion 记录 `KNOWN`、`ESTIMATED` 或 `UNKNOWN`：与当前状态精确匹配的已缓存数值积分为 KNOWN，跨状态复用剩余运动后缀为 ESTIMATED；找不到可复用后缀时，以不变的正常能量/必要时间下界生成显式标记的非零 proxy。UNKNOWN 不自动计入 F；ESTIMATED 和 UNKNOWN 均有 1 Wh 不确定性罚项，另计估计 reserve 缺口。日志保留估计类型及边界声明，不能将 proxy 解释为可行轨迹、完整剩余能量预测或不可行证明。

候选保留同时照顾 complete / partial 与 immediate / delayed 类别，目录 cap 为每机 10 列，既有生成上限每机 64 次、全局 2048 次保持不变；有限预算内未找到某类候选时，不声称该类不存在。外/中环 delay 从已见资源释放时刻及反推下游可用时间导出，最多取 2 个延后 offset；轨迹仍由原生成器重新生成，并通过原有校验。只提交 `start_time <= now + 5 s` 的首动作；更远期内容是预测，靠后续 dispatch 重评或状态、能量与 epoch 仍有效的链复用接续。

对照为旧 `stage2_rescue_need_v7_medium_600s_20261023` 的两方法从 0 到 600 s 完整结果。修复后正式 root 为 `stage2_downstream_service_v8_medium_600s_20261023_r1`，同 `service_half_v1`、`medium24_rate150_v1`、`rescue_need_v7`、seed 71 及 fixture/manifest/environment，从 0 开始，不允许通过 branch 改变正常调度算法。pair 墙钟 21600 s、compute 19800 s、archive 1800 s 与旧实验一致，已在 prepare 冻结，运行中不修改预算。world、rescue、能源、安全、轨迹与物理配置不在本轮修改范围。

旧运行源码为 `4da4f3`；`2009c0d` 后有 monitor 精确帧日志修复，UI 另线程变更。旧两方法回放审计各有 2 个 warning 缺帧 issue，保留原证据，不计作物理失败，也不能将日志补帧当作调度收益。

分析脚本 `artifacts/rotating_slots_stage2_20261023/v8_analyze_medium.py` 仅读取指定 root，输出 JSON 到 root 外的指定位置。holding 为各环真实稳定 HOLD 时间，含未来预约开始前等待，排除 transfer/final；由实际 maneuver_started 与 inbound/transfer completion 边界积分，按原模型 `stable_power(...).battery_w × 秒` 得到 J，并以 states 左端状态积分、累计 holding counter 和 summary 交叉核对。执行数值 ledger 本来排除 STABLE/GROUND，单独报告非稳定已执行 charge，不能冒充 holding ledger 或总能耗。

每 60 s 及 600 s 终态报告各稳定环 warning、declared waiting、`soc-.25-normal_energy_lower_bound(ring)<=0` 积压和 UNKNOWN。warning onset/declaration 时间采用原 events，避免旧缺帧的偏移；world 没有 warning clear 事件，因此 warning 清除采用后续真实 state，积分仍标 approximate。lower-bound shortage 与 UNKNOWN 的时间积分按真实非均匀 states 左端近似并报告最大帧间隔。lower bound shortage 不是任意 SOC 阈值，也不证明可行。总能耗及 normal/emergency/F/pending/offered 直接取 600 s metrics，不相加累计摘要。另列分环转移、进入 inner 到开始 final/接地的时长与 censored，以及 normal/完整 dispatch 耗时。

旧 medium 基线按上述口径只读提取到 [v8_medium_baseline_facts.json](../../artifacts/rotating_slots_stage2_20261023/v8_medium_baseline_facts.json)，为 preexisting_artifact；本轮实际 600 s 结果为 newly_run，紧凑事实见 [v8_medium_final_summary.json](../../artifacts/rotating_slots_stage2_20261023/v8_medium_final_summary.json)，规划器记录见 [v8_planner_final_facts.json](../../artifacts/rotating_slots_stage2_20261023/v8_planner_final_facts.json)。完整逐 dispatch 分析保留于本地约 38 MB 的 `v8_medium_final_facts.json`，不作为版本交付的大文件。下表 pending 为 airborne pending；四行 upstream pending 均为 0、offered 均为 57。holding 是所有飞机实际稳定 HOLD 时间之和，单位为 aircraft-seconds，包含未来预约实际开始前的等待，排除 transfer/final。

| 档案 / 方法 | normal / emergency / total / F / pending | 总能耗 kWh | O / M / I holding aircraft-s | O / M / I holding kWh | normal wall mean / P95 s |
|---|---|---:|---|---|---|
| 旧 v7 / energy_slack | 27 / 2 / 29 / 0 / 28 | 394.344077 | 342.110798 / 2153.095318 / 6803.744375 | 6.469506 / 44.744233 / 175.334625 | 15.1332 / 16.1196 |
| 新 v8 / energy_slack | 30 / 1 / 31 / 0 / 26 | 356.764083 | 2797.061330 / 2820.114789 / 4362.604979 | 52.893991 / 58.605800 / 112.425698 | 14.0233 / 15.1404 |
| 旧 v7 / rh_cp_sat | 22 / 4 / 26 / 0 / 31 | 406.652004 | 99.958602 / 2516.591876 / 7337.262313 | 1.890273 / 52.298183 / 189.083550 | 17.4118 / 19.8636 |
| 新 v8 / rh_cp_sat | 28 / 3 / 31 / 0 / 26 | 357.924250 | 2085.693772 / 2698.136173 / 4772.667912 | 39.441633 / 56.070919 / 122.993148 | 21.5410 / 26.7566 |

两方法各环事件 holding 合计与 summary.holding_time 差均为 0 s；states 累计 holding counter 差均为 0 s；states 左端按环积分误差最大 1.42e-10 s。holding 能耗均小于总能耗；holding J 加已执行非稳定 ledger charge 与总能耗的差在 3.34e-6 J 内。旧 max state gap 约 1 s；warning 积分状态明确为 `APPROXIMATE_NO_CLEAR_EVENTS`。

本地 bounded 检查：新增 profile/CLI/restore 与 synthetic 分环时间/J/非均匀帧/censored 检查通过；联同 service、legacy branch、CLI 回归共 48 项通过（36.38 s）；加入 failure-isolation 与 no-algorithm-branch fixture 后新增文件 13 项通过（2.19 s）。真实旧 medium provenance 重算 checkpoint identity 与原 campaign.identity 相等。

主对话提供的本轮 profile 实现小测试记录为 10 passed、2 skipped，追加 middle 低能源 case 为 1 passed。本地环境无 OR-Tools，跳过项不计为 CP 实测通过。服务器回归实际为 **95 passed、1 skipped，21.71 s**（JUnit 21.705 s）；随后仅补预计等待解释字段的定向检查为 **14 passed，7.75 s**，wrapper 7.918596 s，UTC 10:06:55.620–10:07:03.541。后一次改动不改变 Ehat、目标、候选或安全规则，未因此重跑 callback。物理、安全、能源、rescue_need_v7 应急规则与 quality_v5 预算保持不变。

服务器验证 root 为 `~/training_outputs/terminal/stage2_downstream_service_v8_validation_20261023`，紧凑事实保存于 `artifacts/rotating_slots_stage2_20261023/v8_validation_facts.json`。两方法使用与旧 medium 相同的真实初始 fixture/manifest，t=0 各执行一次 callback；aircraft/slots/pads 比对通过，baseline 输入文件 before/after 哈希相同，线程为 1，分别使用 CPU 0（core 0）和 CPU 2（core 1，socket 0）。energy_slack callback 16.812474 s、提交首动作 10 个且 ACCEPTED；rh_cp_sat callback 32.686167 s、提交首动作 11 个且 ACCEPTED。正确运行 PID 分别为 172770 / 172771，wrapper 耗时 17.202956 / 33.103907 s，退出码均为 0。没有推进仿真，两个 callback 后实际 touchdown 均为 0。

CP 记录 `refinement_rounds=7`、16 个 phase 记录，solver 合计 1.676627 s。最终目标为 `[0,-6,18,608480650.1,7485.5]`，hint 为 `[0,-5,708.0633055555556,662889430.9,8166]`；其中 -6 表示候选窗口内的预测完成收益，不是已经发生 6 次着陆。早期 L 阶段尚未优化 Ehat/tau，model 的这两层可能大于 canonical 最早排程值，L 本身一致；最后记录的 risk 阶段及 phase 3 的完整 model/canonical 向量一致。phase 3 为 FEASIBLE，scheduler 为 LEX_LIMITED，不是全局最优或 600 s 效果证明。

两份原始 `runtime/*/candidates.json` 均有 136 列：31 complete、57 partial、48 no-move；移动列中 82 immediate、6 delayed，6 个 delayed 均为 complete。原 helper 把同名 NO_MOVE category 与 timing 各加一次，导致原诊断 counts 显示 96，而原始 columns 实际为 48。本轮只修正 helper 统计并增加全局/每机 `COMPLETE+PARTIAL+NO_MOVE==columns_count` 断言，使用原 columns 机械重算紧凑事实，保留原 facts/logs，不重新执行 callback。无真实 callback 的 4 列 synthetic 计数检查通过。

首次两个 t0 验证 wrapper 因工作目录错误尝试加载 `/home/user/artifacts/.../v8_runtime_validation.py`，Python 找不到脚本，PID 172652 / 172651 均以 2 退出，尚未 import 或执行 callback；原日志保留。随后使用正确绝对路径才产生上述各一次真实 callback 记录。上述 t0 验证不代表完整 medium 实验结果，也不作正常着陆率或能耗收益结论。

首次正式 root `stage2_downstream_service_v8_medium_600s_20261023` 已停止并保留只读：`INVALID_STOP`、common completed 0 s、`campaign.exit=1`、GNU real 41.19 s；energy_slack 最后记录 9.388737514630813 s，rh_cp_sat 最后记录 4.0 s，两者均无完整 60 s summary。这是候选生成软件异常，不能当作物理失败结果或可用于比较的部分窗口。紧凑恢复事实见 `artifacts/rotating_slots_stage2_20261023/v8_recovery_facts.json`。

异常发生在 `_delayed_reference_seed`：`97.8887375146308 - 49.38873751463081 - 13.5 = 34.99999999999999`，旧 `int(duration)` 得到 34，而 screen 的实际 duration 键采用 5 s 步长，查询 `(0,34)` 抛出 KeyError。修复只枚举 `screen.items()` 实际存在的键，筛选 `d >= duration - 1e-8` 且 `witness is None`，按 `(d,pad)` 排序；没有放松原生成器、能源或安全校验，也未调目标或预算。修复后 `stage2_planning.py` SHA256 为 `3534521894fdc4d4f2790a81cd89335caba1e7a3ca61a3553018e497f265d3b3`，服务器 `recovery/pytest.*` 记录 **16 passed，10.99 s，exit 0**。

连续 helper 使用原 old-medium fixture/manifest/seed，让 executor 自行开始 t0 并推进至 12 s，各方法一次，不额外手动调用 t0、不自动重试。两方法现已各完成 0→12 s，均有 4 次 callback，时间为 `[0,5,9.388737514630813,10]`，均 `failed=false`、`terminated_invalid=false`、exit 0，profile 及每次 quality_v5 budget 保持原样；baseline 输入与源码 before/after 哈希一致。energy_slack 验证 wall 62.96052694 s、GNU real 63.39 s；rh_cp_sat wall 111.29763754 s、GNU real 111.75 s。该验证覆盖原异常所在的实际非整数 release，没有新增场景或种子。原始事实位于 `~/training_outputs/terminal/stage2_downstream_service_v8_validation_20261023/recovery/continuous/{method}/facts.json`，wrapper 证据为同 recovery 目录的 `continuous_METHOD.exit` / `continuous_METHOD.timing.txt`；紧凑回调列表及准确路径见 `v8_recovery_facts.json`。修复后的 `_r1` 是相同输入的软件错误重启；原 root 无完整窗口可拼接，不构成调参矩阵。

`_r1` 使用 total 21600 s / compute 19800 s / archive 1800 s。旧账目 `elapsed_floor_s=663.6353569030762`、`campaign_wall_s=663.6366401958512` 已包含原 600 s preparation credit；新 preparation 实际承接两者较大值，加恢复验证保守 credit 300 s，即 **963.6366401958512 s**。本次并行连续验证 wall 111.75 s、服务器恢复 pytest wrapper 11.192204 s、本地检查 16.78 s 等由该 300 s 保守额度覆盖；不再追加原 600 s，排除旧退出后人工编码等待，原 budget 文件保持不变。新 prepare 只执行一次，exit 0、GNU real 1.86 s，新持久化 floor 为 965.147096157074 s。运行 elapsed 按 `max(floor, now-started_epoch, monotonic_guard)` 计算，不能把 floor 再加到 epoch 差上。

`_r1` 已绑定 CPU 12 / 14（物理 core 6 / 7），各进程及数值库线程为 1。另一独立 `.worktrees/equal-linear-speed` 实验使用 CPU 0 / 2 / 4 / 6 / 8 / 10，应保持其运行与文件不受影响。物理核心分开仍共享机器内存、缓存与负载，因此 wall-time 比较保留并发环境限制。

以下启动、估时与 heartbeat 段落保留当时的历史快照，终态以本文开头及末尾记录为准。正式启动时间为服务器 UTC **2026-10-23T10:35:10.139Z**，launcher PID 178063、GNU time PID 178073、runner PID 178074、energy_slack spawn PID 178077（CPU 12）、rh_cp_sat spawn PID 178078（CPU 14）。本地与服务器物化 provenance 完全相同，source hash 为 `e9777021491e30aa1e7fbdbd2938f989f0968f4126acdf7c1cb18c741d425c17`；fixture、manifest、environment.lock 与旧 medium 字节相同。启动事实见 `artifacts/rotating_slots_stage2_20261023/v8_launch_facts.json`。

采用 `startup_confirmed.json` 的服务器 UTC **10:39:53.661Z** 快照：runner elapsed 283.522 s，energy_slack 最新真实状态约 29 s、rh_cp_sat 18 s，两者 failed/invalid 均 false，无 error/invalid 文件，进程仍在运行。完整共同 60 s 边界为 **0 s**；18 s 只是 min latest state，不能当作完整共同窗口。本地 SSH 前后时间另记为 UTC 10:38:59.315Z / 10:39:06.035Z，不混作服务器观测时间。原 `startup_snapshot.json` 的错误标签不覆盖，以保留的 `startup_confirmed.json` 纠正。后续日志读取 `campaign.log`，耗时读取 `campaign.timing.txt`，终态退出标记读取 **`campaign.exit`**（不是 `exit_code.txt`）；该确认快照尚无终态退出标记。

初始预计总耗时 T0 为 **12600 s**，依据旧 medium 约 9749 s 及新 CP callback 约 25–32 s；工作不均匀，估时不保证完成时间。确认快照的剩余估计为 12316.478 s。主对话随后在本地 UTC **10:41:25.614Z** 经工具确认当前 chat 的 heartbeat `stage2-v8-600` 为 ACTIVE、间隔 **203 min**：从 SSH 后时刻已过去 139.579 s，故采用 `12316.478-139.579=12176.899 s` 取整。首次调用因缺 destination 被拒、未创建，随后 `destination=thread` 成功，无重复调度。唤醒时检查实际退出状态和完整产物；完成或失败后删除 heartbeat，并继续既定分析交付；非终态且无变化时静默。

正式终态核验：服务器 `campaign.exit=0`，GNU real 为 **9769.01 s**；原 launcher、time、runner 和两方法 PID 均已退出，未发现 error/invalid 文件。两方法 600 s summary、metrics、checkpoint、replay 与 audit 完整，common completed 为 **600 s**，两份 audit 均 PASS、0 issue。最终持久化 `campaign_wall_s=10932.100402692333`，包含冻结预算的历史 credit；不能用任务停止后检查时的 `now-started_epoch≈14143 s` 替代实际终态 cost。heartbeat `stage2-v8-600` 已经工具确认删除，不再安排监控。

本轮 holding 闭合检查中，两方法 event 与 summary 的 holding 差均为 0 s；holding J 加已执行非稳定 ledger charge 减总能耗分别为 energy_slack **−2.98e−7 J**、rh_cp_sat **4.77e−7 J**。warning 清除没有独立事件，warning aircraft-seconds 仍为近似值；lower-bound shortage 与 UNKNOWN 使用真实非均匀帧的左端积分，新旧最大帧间隔约 1 s。旧两方法各有 2 个 warning 缺帧 audit issue，不能称旧 audit 为 PASS，也不能把日志补帧归为算法收益。

相同 seed、输入及物理/应急协议的本次整体对照中，energy_slack 正常着陆由 27 增至 30、应急由 2 降至 1，总能耗下降 **9.5297%**；rh_cp_sat 正常由 22 增至 28、应急由 4 降至 3，总能耗下降 **11.9827%**。inner holding 分别下降 **35.8794% / 34.9530%**。总 holding 时间 ES 为 **9298.950491→9979.781099 aircraft-s（增加 7.3216%）**，CP 为 **9953.812790→9556.497857 aircraft-s（减少 3.9916%）**；inner 等候减少不等于所有等待总量都减少，ES 主要改变等待位置并减少飞行机动能耗。两方法 outer/middle holding 增加，但这两环累计 lower-bound shortage 均为 0 aircraft-s；旧 middle 分别为 38 / 23 aircraft-s。新 CP middle 仍有约 **78.115 aircraft-s warning**。UNKNOWN 仍多，零 shortage 只针对既定下界口径，不能视为完整正常可行性证明。

energy_slack 的 holding 能耗由 **226.548365→223.925489 kWh**，仅减少 **2.622876 kWh**；非稳定执行 charge 由 **167.795712→132.838594 kWh**，减少 **34.957118 kWh**。CP 对应为 holding **243.272006→218.505701 kWh**、非稳定 **163.379998→139.418550 kWh**。因此不能把总节能全部归因于上游较低功率等待。实际完成的 M→I 转移由旧 ES/CP **36/37** 变为 **32/32**，O→M 由 **36/38** 变为 **30/33**；较少跨环而有更多正常着陆，支持减少提前推进的解释，但单次整体对照无法分离每项修改的因果贡献。

新 normal wall 的 mean/P95 见表：ES 较旧略快，CP 由 **17.4118/19.8636 s** 增至 **21.5410/26.7566 s**。新完整 dispatch mean/P95 分别为 **15.6433/19.8988 s** 与 **24.3347/33.0058 s**；完整 dispatch 包含累计 monitor 工作，不能直接视为严格的在线连续响应延迟。旧新运行共享服务器环境，墙钟统计受到并发负载、内存及缓存影响。

规划器原始记录中，CP **388/388** 次进入求解，**125/388（32.2%）** 的最终向量严格改善同一次调用的 hint；该比例不是相对完整 energy_slack 实验的优化增益。共 **1124 rounds**，平均 **2.897**、最大 **46**；3699 个 phase 中有 **3353 OPTIMAL / 305 FEASIBLE / 41 UNKNOWN**，这些有限目录及预算内状态不能宣称全局最优。平均 candidate/validation/solver wall 为 ES **10.6184/3.2521/0 s**、CP **10.1583/5.6435/5.5485 s**；CP solver 合计 **2152.822 s**，约占其 normal wall 的 **25.8%**，不能沿用旧实验约 0.2% 的描述。

DELAYED 被选预测的重复 aircraft-dispatch 观察为 ES **152**、CP **361**，不能当作独立机数或执行动作数。其中 **15 / 20** 条记录与同一次 dispatch、同目标的 normal 提交匹配，提交 hash 有真实启动记录；其余 **137 / 341** 条未预约预测缺少可追踪的执行 signature，后续同目标动作只能作为 same-target proxy，不能宣称全部预测兑现。另有 **32 / 41** 个所有 normal 类型的未来启动预约（start>commit）均按原计划 epoch 实际开始；这一计数不专属于新 DELAYED 候选。预测能量不能跨 dispatch 求和当作物理能耗。

回放证据见 [check.json](../../artifacts/rotating_slots_stage2_20261023/v8_replay_checks/check.json) 与 [check_replays.py](../../artifacts/rotating_slots_stage2_20261023/v8_replay_checks/check_replays.py)。两方法均实际 load/seek/render 300、450、600 s，六张图均为 1600×860 且图像读取通过。ES 450 s 的记录帧为 449.9999999999997 s，controller/snapshot 返回 450 s；其余记录帧等于请求时刻。截图：[ES 300](../../artifacts/rotating_slots_stage2_20261023/v8_replay_checks/energy_slack_0300.png)、[ES 450](../../artifacts/rotating_slots_stage2_20261023/v8_replay_checks/energy_slack_0450.png)、[ES 600](../../artifacts/rotating_slots_stage2_20261023/v8_replay_checks/energy_slack_0600.png)、[CP 300](../../artifacts/rotating_slots_stage2_20261023/v8_replay_checks/rh_cp_sat_0300.png)、[CP 450](../../artifacts/rotating_slots_stage2_20261023/v8_replay_checks/rh_cp_sat_0450.png)、[CP 600](../../artifacts/rotating_slots_stage2_20261023/v8_replay_checks/rh_cp_sat_0600.png)。600 s 显示的 rescue WAIT/RESERVED/ACTIVE 分别为 ES **0/0/3**、CP **0/1/5**；ES 在 450/600 s 的 P4 显示 FREE。单张图不能证明存在安全可执行的着陆机会，不能据此声称不存在 pad 闲置或所有低 SOC 飞机都没有错过最后机会。主对话已查看这些图及 comparison curves；人工验收状态保持 `HUMAN_PENDING`。

在 PowerShell 中先进入仓库，再分别运行以下两条既有平台入口命令；关闭窗口即返回。命令实际调用 `ReplayController` 与 `RotatingSlotsReplayViewer.run()`：

```powershell
Set-Location -LiteralPath 'D:\7_codespace\terminal'
& D:\7_codespace\terminal\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_downstream_service_v8_medium_600s_20261023_r1\energy_slack\replay_0600')).run()"
& D:\7_codespace\terminal\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_downstream_service_v8_medium_600s_20261023_r1\rh_cp_sat\replay_0600')).run()"
```

这次固定 600 s medium 对照显示正常着陆增加、应急次数和总能耗下降，同时 CP 计算时间增加。两种方法终态仍有 **26** 架 airborne pending，较初始 **24** 架多 2，且仍有救援未完成；当前证据不足以证明长期持续稳定、长期无能源失败或所有剩余飞机正常可行。没有增加场景、种子或修改物理/应急协议来扩展这一结论。
