# downstream_service_v8：medium 600 s 协议与状态

状态：修复代码 `ecd49738f918d861125f4afee3814e2596c33176` 已提交并推送，provenance 已冻结；唯一有效 paired 600 s 实验 `_r1` 已正式启动。确认快照的完整共同 60 s 边界仍为 0，尚未完成 600 s，`HUMAN_PENDING` 保持。首次异常 root、其 provenance 与原始日志继续保留只读。

本轮将正常调度算法显式冻结为 `planning_profile=downstream_service_v8`；默认旧档案仍为 `progress_tail_v7`。candidate policy 与 objective schema 随 profile 保存到 provenance、campaign、checkpoint identity、summary 与 comparison。profile 独立于 `quality_v5` 预算：soft 30 s、solver 8 s、refinement 64 rounds 保持原样。新增下游服务预测与队列启发式不是可行性证明；执行仍需现有连续认证。

两种正常调度方法共享按字典序最小化的目标 `(F, -L, risk, Ehat, tau)`：保留失败优先和正常完成收益，随后比较估计的正常 reserve 缺口与不确定性代价、总完成能量和完成时间。每个 dispatch 固定一次虚拟 pad 分配及 energy-slack / release / id 队列顺序；候选选项改变 ready time、等待率、能耗和实际 pad 区间，CP 显式建模依赖所选选项的队列等待。虚拟服务只用于评价，不能创建实际预约，也不提供安全证明。该固定分配及顺序不搜索所有可能的队列，因此不保证全局最佳队列或 600 s 期末效果。

尾部 motion 记录 `KNOWN`、`ESTIMATED` 或 `UNKNOWN`：与当前状态精确匹配的已缓存数值积分为 KNOWN，跨状态复用剩余运动后缀为 ESTIMATED；找不到可复用后缀时，以不变的正常能量/必要时间下界生成显式标记的非零 proxy。UNKNOWN 不自动计入 F；ESTIMATED 和 UNKNOWN 均有 1 Wh 不确定性罚项，另计估计 reserve 缺口。日志保留估计类型及边界声明，不能将 proxy 解释为可行轨迹、完整剩余能量预测或不可行证明。

候选保留同时照顾 complete / partial 与 immediate / delayed 类别，目录 cap 为每机 10 列，既有生成上限每机 64 次、全局 2048 次保持不变；有限预算内未找到某类候选时，不声称该类不存在。外/中环 delay 从已见资源释放时刻及反推下游可用时间导出，最多取 2 个延后 offset；轨迹仍由原生成器重新生成，并通过原有校验。只提交 `start_time <= now + 5 s` 的首动作；更远期内容是预测，靠后续 dispatch 重评或状态、能量与 epoch 仍有效的链复用接续。

对照为旧 `stage2_rescue_need_v7_medium_600s_20261023` 的两方法从 0 到 600 s 完整结果。修复后正式 root 为 `stage2_downstream_service_v8_medium_600s_20261023_r1`，同 `service_half_v1`、`medium24_rate150_v1`、`rescue_need_v7`、seed 71 及 fixture/manifest/environment，从 0 开始，不允许通过 branch 改变正常调度算法。pair 墙钟 21600 s、compute 19800 s、archive 1800 s 与旧实验一致，已在 prepare 冻结，运行中不修改预算。world、rescue、能源、安全、轨迹与物理配置不在本轮修改范围。

旧运行源码为 `4da4f3`；`2009c0d` 后有 monitor 精确帧日志修复，UI 另线程变更。旧两方法回放审计各有 2 个 warning 缺帧 issue，保留原证据，不计作物理失败，也不能将日志补帧当作调度收益。

分析脚本 `artifacts/rotating_slots_stage2_20261023/v8_analyze_medium.py` 仅读取指定 root，输出 JSON 到 root 外的指定位置。holding 为各环真实稳定 HOLD 时间，含未来预约开始前等待，排除 transfer/final；由实际 maneuver_started 与 inbound/transfer completion 边界积分，按原模型 `stable_power(...).battery_w × 秒` 得到 J，并以 states 左端状态积分、累计 holding counter 和 summary 交叉核对。执行数值 ledger 本来排除 STABLE/GROUND，单独报告非稳定已执行 charge，不能冒充 holding ledger 或总能耗。

每 60 s 及 600 s 终态报告各稳定环 warning、declared waiting、`soc-.25-normal_energy_lower_bound(ring)<=0` 积压和 UNKNOWN。warning onset/declaration 时间采用原 events，避免旧缺帧的偏移；world 没有 warning clear 事件，因此 warning 清除采用后续真实 state，积分仍标 approximate。lower-bound shortage 与 UNKNOWN 的时间积分按真实非均匀 states 左端近似并报告最大帧间隔。lower bound shortage 不是任意 SOC 阈值，也不证明可行。总能耗及 normal/emergency/F/pending/offered 直接取 600 s metrics，不相加累计摘要。另列分环转移、进入 inner 到开始 final/接地的时长与 censored，以及 normal/完整 dispatch 耗时。

正式终态结果后再补充结论；当前不作性能收益声明。

旧 medium 基线已按上述口径只读提取到 `artifacts/rotating_slots_stage2_20261023/v8_medium_baseline_facts.json`。以下为 preexisting_artifact，非本轮新实验：

| 方法 | normal / emergency / F / pending | 总能耗 kWh | O / M / I holding 秒 | O / M / I holding kWh |
|---|---|---:|---|---|
| energy_slack | 27 / 2 / 0 / 28 | 394.344077 | 342.110798 / 2153.095318 / 6803.744375 | 6.469506 / 44.744233 / 175.334625 |
| rh_cp_sat | 22 / 4 / 0 / 31 | 406.652004 | 99.958602 / 2516.591876 / 7337.262313 | 1.890273 / 52.298183 / 189.083550 |

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

正式启动时间为服务器 UTC **2026-10-23T10:35:10.139Z**，launcher PID 178063、GNU time PID 178073、runner PID 178074、energy_slack spawn PID 178077（CPU 12）、rh_cp_sat spawn PID 178078（CPU 14）。本地与服务器物化 provenance 完全相同，source hash 为 `e9777021491e30aa1e7fbdbd2938f989f0968f4126acdf7c1cb18c741d425c17`；fixture、manifest、environment.lock 与旧 medium 字节相同。启动事实见 `artifacts/rotating_slots_stage2_20261023/v8_launch_facts.json`。

采用 `startup_confirmed.json` 的服务器 UTC **10:39:53.661Z** 快照：runner elapsed 283.522 s，energy_slack 最新真实状态约 29 s、rh_cp_sat 18 s，两者 failed/invalid 均 false，无 error/invalid 文件，进程仍在运行。完整共同 60 s 边界为 **0 s**；18 s 只是 min latest state，不能当作完整共同窗口。本地 SSH 前后时间另记为 UTC 10:38:59.315Z / 10:39:06.035Z，不混作服务器观测时间。原 `startup_snapshot.json` 的错误标签不覆盖，以保留的 `startup_confirmed.json` 纠正。后续日志读取 `campaign.log`，耗时读取 `campaign.timing.txt`，终态退出标记读取 **`campaign.exit`**（不是 `exit_code.txt`）；该确认快照尚无终态退出标记。

初始预计总耗时 T0 为 **12600 s**，依据旧 medium 约 9749 s 及新 CP callback 约 25–32 s；工作不均匀，估时不保证完成时间。确认快照的剩余估计为 12316.478 s。主对话随后在本地 UTC **10:41:25.614Z** 经工具确认当前 chat 的 heartbeat `stage2-v8-600` 为 ACTIVE、间隔 **203 min**：从 SSH 后时刻已过去 139.579 s，故采用 `12316.478-139.579=12176.899 s` 取整。首次调用因缺 destination 被拒、未创建，随后 `destination=thread` 成功，无重复调度。唤醒时检查实际退出状态和完整产物；完成或失败后删除 heartbeat，并继续既定分析交付；非终态且无变化时静默。
