# 360→480 秒接续与学习加速说明

记录日期：2026-10-22。用户已观看两份 0360 回放，确认总体调度合理，记录为 `HUMAN_CONFIRMED_GENERAL_SCHEDULING`；这项确认不表示所有性能目标均已达到。用户要求再接续 120 秒观察应急处置，0480 人工验收仍为 `HUMAN_PENDING`。本记录只说明已授权的接续、0360 实际计划及耗时证据，以及后续学习方向；不新增实验或修改算法。

## 唯一一次同状态接续

从同一 root 的 0360 checkpoint 继续到 480 秒，不 reset、不重新 prepare，保持原 manifest、source、config 及算法。原 0360 回放与证据保留。root 为 `/home/user/training_outputs/terminal/stage2_progress_service_360s_v4_20261022`，本地同步目录为 `D:/7_codespace/training_outputs/terminal/stage2_progress_service_360s_v4_20261022`。

实际工作目录 `/home/user/projects/terminal`，启动命令：

```bash
/home/user/envs/terminal_stage2_v1/bin/python -u /home/user/projects/terminal/scripts/server/run_rotating_slots_stage2.py resume --output /home/user/training_outputs/terminal/stage2_progress_service_360s_v4_20261022 --target 480
```

服务器启动时间 UTC `2026-10-22T02:06:40.624434Z`；wrapper PID 129810、runner PID 129817，方法 worker PID 129820/129821，绑定 CPU 0/2 不同物理核，线程数及数值库线程环境均为 1。日志为 `resume_0480.log`，计时为 `resume_0480.timing.txt`，终态退出码记录为 `resume_0480.exit_code.txt`。启动后已观察两方法 states 到达 361.4/361.0 秒；这里只确认开始接续，不宣称完成。初始 T0 取 600 秒，依据原 360 秒 runner 1599 秒的三分之一加归档开销作估计，不是完成保证。

接续账目见 root 下 `continuation_0480_accounting.json`，接续前证据保存在 `continuation_0480_pre_run/`。旧持久预算 2502.5826683044434 秒继续保留，已结束后的人工等待不计入新计算，不清零旧账。依据用户新增 120 秒的续跑授权，主对话将本次追加计算与归档上限设为 1200 秒，包含 300 秒归档；前期保守计入 60 秒。

| 记账项 | 秒 |
|---|---:|
| 原已累计预算 | 2502.5826683044434 |
| 接续前开销 credit | 60 |
| 接续初始 live floor | 2562.5826683044434 |
| 新累计 total 上限 | 3702.5826683044434 |
| 新累计 compute stop | 3402.5826683044434 |
| 归档保留 | 300 |

Luna 的四次有界进度观察已结束。最后一次观察时 runner 已运行 470.09 秒，两方法均有完整 `summary_0420.json` 与 `checkpoint_0420.json`，states 均到 420 秒，未见 error/invalid，尚无 resume 终态退出码。持久预算下界为 3094.896069765091 秒，计算截止仍为 3402.5826683044434 秒；当前只确认已完成到 420 秒，不能写成 480 秒完成。

原 T0=600 秒低估了本次耗时。按新增 120 秒中完成 60 秒粗估，总耗时约 940 秒、剩余约 470 秒；计算预算可能先截止，因此不保证到达 480 秒。主对话已通过工具成功创建当前 thread 的 heartbeat `stage2-480`，状态 ACTIVE，间隔 5 分钟，设置时扣除观察后经过时间；不保留睡眠监控 worker，唤醒后继续检查、分析与归档。若最终预算截在 420 秒，先核验终态，再决定如何完成原授权目标剩余的 60 秒；此处不预记任何未执行的追加预算。

## 为什么 CP 只在 24/266 次决策中求解

两种方法共用正常候选生成，候选阶段最多约 3 秒。EnergySlack 按能量 slack 排序，逐机尝试把动作加入联合 batch，并检查整个 batch 的安全性。RH-CP-SAT 先用同样的 EnergySlack 路径建立 hint，再构建资源冲突约束，进行分层离散选择，最后仍需安全验证。CP 只能使用正常调度 5 秒共享软预算中剩余的时间；候选和 hint 已消耗的时间会压缩 CP 机会。

0360 实际记录中，CP 方法的 266 次决策全部为 `LEX_LIMITED`，242 次 solver list 为空；仅 24 次决策包含共 122 次 phase solve，状态均为 OPTIMAL。phase 是当前候选离散模型的一个目标阶段，不是原连续问题的全局解。官方文档中的 OPTIMAL 表示对所提交模型找到了最优可行解，不能据此扩大到完整连续调度问题；UNKNOWN 也不能直接解释为不可行。[OR-Tools CP-SAT 状态说明](https://developers.google.com/optimization/cp/cp_solver)

fallback=0 不表示每轮都运行了 CP。当前日志没有 hint objective 或 strict-improvement 字段，不能量化 CP 相对 hint 的严格改进次数或收益。这次两方法结果接近，与共享候选及 hint 占用主要时间、CP 大部分调用未进入求解相一致；不能写成 CP 与贪心完全等价。

## 正常落地数量相同，实际计划仍不同

按 `maneuver_started` 事件中的 trajectory hash 连接轨迹元数据，逐飞机比较实际动作序列与执行时刻；不按 decision 序号对齐，也不计入尚未执行的预测 suffix。两方法各实际开始 87 个动作。共同有动作的 52 架中，41 架序列或时刻不同，11 架完全相同；共同落地 15 架，其中 9 架落地时刻不同。

| 同机实例 | EnergySlack 实际动作 / 时刻（秒） | CP 方法实际动作 / 时刻（秒） |
|---|---|---|
| initial_middle_09 | M21→I13，15→55 | M21→I13，10→50 |
| initial_middle_03 | 正常落地 275.614 | 正常落地 152.455 |
| initial_middle_06 | 正常落地 203.500 | 正常落地 260.576 |

两方法均正常落地 20 架，不意味着采用相同计划；上述个体结果也不是一方全面占优。差异同时包含轨迹分岔及墙钟预算造成的候选截断，不能把全部差异单独归因于 CP 的优化收益。

## 5 秒花在什么地方

normal 是同步 callback 加 `_start_due` 的实测耗时，包含候选、选择、安全检查及 commit。`dispatch_wall_s` 则为当前应急、准入、normal 等工作，加上此前 `_update_emergency` 累积的 `_monitor_timing_pending`；它是完整调度相关工作量的记账，不严格等同一次传感状态到下发动作的连续端到端延迟，不能用于推导真实部署连续延迟的严格下界。

当前 world 通过同步 callback 等待调度返回，这是仿真执行方式，不表示已实现真实系统的延迟补偿。调度周期参数为 5.0 秒，同时存在事件触发，因此 265/266 次决策并非固定每 5 秒调用。5 秒是软预算，正在进行的评估和最终验证可能越过它；watchdog 为 30 秒。

| 0360 统计（秒） | Slack mean / P95 / max | CP mean / P95 / max |
|---|---:|---:|
| normal | 5.0912 / 5.6498 / 7.9671 | 5.1701 / 5.5778 / 8.0811 |
| 完整 dispatch 工作量记账 | 5.3722 / 6.3127 / 7.9672 | 5.3611 / 6.3262 / 8.0812 |

日志中的阶段计时为 exclusive：父阶段已经扣除嵌套子阶段。下面均值按全部决策统计，不能把父计时 `candidate_generation_and_single_vehicle` 约 0.02 秒的残差当作全部候选耗时。

| exclusive 阶段均值（秒） | Slack | CP |
|---|---:|---:|
| safe progress 生成 | 1.4750 | 1.4827 |
| reference seed 生成 | 1.5207 | 1.5154 |
| 贪心 selection / hint 构建 | 1.8696 | 1.8951 |
| 上述三项占已记录 decision wall | 95.58% | 94.66% |
| 最终 selected validation + commit | 0.1202 | 0.1448 |
| normal 之外的 dispatch 增量 | 0.2810 | 0.1910 |

应急工作量（含累计 pending）P95 为 Slack 1.1135 / CP 1.1286 秒，准入工作量 P95 为 0.5850 / 0.5872 秒。上述 dispatch 与 normal 的均值差是记账口径差，不是一次连续调用的严格延迟下界。

CP 的 122 个 solver phase 合计仅 0.0647187 秒，约 0.530 毫秒/phase，占全部 normal 墙钟约 0.0047%。这是本次候选被截断后的小模型观察，不能泛化为 CP 在任何规模下都很快；也不能只比较 solver 时间而忽略模型构建、安全核及 hint 开销。

## 学习网络应替代哪一部分

主对话建议优先减少候选生成和反复联合 batch 试错。网络可读取 UAV 的 SOC、slack、位置，目标 slot 与未来预约、pad 可用时刻、服务类别和队列，输出优先级，以及目标 slot/pad/启动时刻的 Top-K 或一个小联合动作。slack 等输入特征使用现成记录或廉价估计，不能为了网络输入再次进行完整 reference 搜索，否则原瓶颈仍然保留。保留原安全与能耗的最终核验，并留少量备选回退；不只是替换 solver，也不应在网络之外每轮照常运行完整 3 秒候选搜索。

训练可先用行为模仿 warm start，再用 RL 优化最终正常落地、失败、应急代价与公平等待。进展奖励不能压过真实终局目标。当前日志是受预算约束的策略行为，不是最优 expert；离线使用较充裕搜索构造老师只是后续可选设计，本轮不执行。

正式比较前，应先给 CP 一个确实有优化机会且总时间有界的预算分配，再在同硬件、同安全核和同时间预算下比较学习策略；不能对一个没有充分求解机会的 CP 基线宣称全面胜利。评价须分别记录网络 forward、状态到 commit 的连续时延、完整 CPU 工作量，并报告 P50/P95/P99/max、超时率、安全回退及调度质量，不能只报网络 forward。

第一阶段可设 normal P95 <1 秒；若要求完整系统连续时延 P95 <1 秒甚至 100 毫秒，还须减少监测、准入与安全校验的重复计算。当前累计 dispatch 不能代替该连续时延测量，也不承诺只替换网络就达到目标。这些均为后续设计建议，本轮未修改或实施。

## 证据与待补终态

0360 事实来源为 [计划与耗时比较 JSON](../../artifacts/rotating_slots_stage2_20261022/v4_plan_timing_comparison.json) 与 [360 秒状态文档](progress_service_360s_status.md)。接续账目见 [continuation_0480_accounting.json](../../../training_outputs/terminal/stage2_progress_service_360s_v4_20261022/continuation_0480_accounting.json)。

0480 退出码、共同完成边界、应急声明/落地/未结束数量、能量失败、审计及回放加载：待主对话提供终态后补充。0480 人工验收：`HUMAN_PENDING`。

以下命令仅在对应 `replay_0480` 已生成并同步后有效；先切换目录，再任选一条独立运行。

```powershell
Set-Location "D:/7_codespace/terminal"
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_progress_service_360s_v4_20261022\energy_slack\replay_0480')).run()"
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_progress_service_360s_v4_20261022\rh_cp_sat\replay_0480')).run()"
```
