# 2026-10-16 正式诊断部分结果：benchmark、dt 与部分 traffic/scheduler

本页保留旧 f67 模型的真实诊断结果。2026-10-18 用户授权的准入等待与 pad 到达规则见[修订记录](amendment_20261018.md)，与本页旧测量分开解释。最后一个旧 seed 后补，后台 20 inner workers 的恢复仅补旧矩阵；以下表格不生成缺失项，也不作为新规则验证。

新规则已另有[有效准则 8 项结果](../../artifacts/rotating_slots/upstream_policy_20261018/qualification_changed_amended/results.json)及[受控小包汇总](../../artifacts/rotating_slots/upstream_policy_20261018/small_multi_aircraft_summary.json)。前者为 8 PASS、86 NOT_RUN、10 replay 有效，阶段 `NOT_ACCEPTED`；后者为最多 4 架 airborne 的具名示范，143.7 s 时物理仍 `IN_PROGRESS`，不能与下文旧高密度 traffic 混成同一实验统计。示范实际运行 ignored helper、启动 source hash 未捕获且期间有小修改的边界，以及 62/74/5/1 各轮定向检查的适用范围，均见[修订记录](amendment_20261018.md#最终有效准则执行与受控小回放)。

## 本阶段结论（基于 60 项已完成诊断）

物理、几何、资源、应急、守恒和权威轨迹执行已实现；已有测试与回放支持被测场景的一致性，具备下一阶段基线设计、接口开发和小规模实验的工程基础。结论不扩展为实机认证、全状态安全证明或大规模 RL 吞吐已经合格，符合[合同最终边界](../../new10.16/fixed_terminal_rotating_slots_stage1_operating_contract_v1_0.md#最终执行边界)。

旧[部分合并原件](../../artifacts/rotating_slots/qualification_f67b5c8_partial_20261016/results.json)保留 **92 PASS、G09 `NOT_RUN`、V12 `HUMAN_PENDING`**。2026-10-18 已另有[独立 G09 消费结果](../../../training_outputs/terminal/rotating_slots_g09_f67b5c8_20261018/results.json)：唯一执行行 G09 为 `PASS`，验证旧 40 项 benchmark，不依赖完整 `diagnostics.json` 或最后一个 traffic。未改写旧 partial，未执行原 catalogue fixture，未产生新版本 94 项全通过记录。[合同 §17.3–17.4](../../new10.16/fixed_terminal_rotating_slots_stage1_operating_contract_v1_0.md#173-cpu性能验收)要求计时证据而不设绝对吞吐门槛；G09 不认证新版准入/pad 规则或新吞吐，V12 仍须负责人观察认可，当前不宣布 `STAGE1_ACCEPTED`。

60 项正式诊断由 40 benchmark、3 dt、3 scheduler 和 14 traffic 组成，[对应 60 项独立审计](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_60_report.json)全部 PASS、零问题、连续间隔均 `CERTIFIED`。已审计记录未见批准轨迹执行中的未预测硬违规；独立审计是分离于 World 的记录重算与约束检查，仍共享数值内核，不是独立实测 oracle。

14 项 traffic 中，rate 60 的 seed 23/47/101 完整运行至 5400 s，另 11 项为 `ENERGY_FAILURE` 截尾。11 项均在已完成中→内环转移后，于内环应急等待中 SOC 耗尽，未启动应急 final。当前确定性验收 driver 不能保证全部需求完成或全部应急救回；这不证明场景失效或全局没有可行调度，driver 也不是正式研究 baseline。保留失败、未释放、截尾及成本下界，不以混合完整/截尾成本均值比较方法。[合同 §17.2](../../new10.16/fixed_terminal_rotating_slots_stage1_operating_contract_v1_0.md#172-正向与负向样例的pass含义)区分物理结局与实现违反，后文保留各项原始指标。

dt 固定脚本的相对能源差为 `6.744915891461807e-15`（低于 0.5%），最大事件时差为 `3.410605131648481e-12 s`（低于 0.001 s）；该结果不替代其他流量或状态的验证。scheduler 三周期是单一 seed 的截尾对照，不能据此选择最佳周期。

现有运行成本偏高，会影响后续大量 RL 实验。20 inner workers 的合成 pilot 约 4.33× 总段、5.77× 接续段加速只适用于本次小样本，不保证真实完整 seed 的收益或总体 ETA。下一阶段先测真实在线耗时和基线预算；性能慢不等于物理场景必然失败，当前也不承诺大规模 RL 吞吐足够。

合同固定的 3 个来流率×5 个 seed 对应预定 15 项 traffic；当前 14/15 并非完整预定集合，rate 180 为 4/5。尚缺 `traffic_rate180_seed71` 这一同协议重复，它不是为新增某条核心规则而单独设计的验收样例；但长期运行可能反映困难实例，存在选择偏差边界，不能把已完成项的失败比例当作总体概率。缺失项不记为成功、失败或负样本，不删除冻结 seed，也不修改合同、YAML、目录或结果状态。

### 后补核查

最后一个旧 seed 在后台继续；先前约北京时间 11:09 的 heartbeat 是已过去的历史安排，不作为当前有效调度。取得真实完整结果后核对物理结局，补最后一项审计、形成旧模型完整 15 项统计，保留既有 60 项审计。G09 的旧 benchmark 证据已独立收口，旧矩阵终态和严格合并仍须自身证据；新规则的执行结果不从旧矩阵外推。

## 断电中断、同步恢复与单项重跑

[重启后直接核验](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_snapshot.json)记录服务器于北京时间 18:16:28 启动；10:25 UTC 查询时，旧 PID `3703967`、`3703969`、`3703989` 均不存在，也没有匹配的诊断进程。61 个目录中 60 项结果为 `COMPLETED`，仅 `traffic_rate180_seed71` 缺少 `result.json` 与 `replay/replay.json`，父目录 `diagnostics.exit` 和 `diagnostics/diagnostics.json` 均不存在。这次直接观测取代此前 SSH 不可达时的未知状态；此前默认沙箱的名称解析失败与提权后 SSH 握手关闭是不同阶段的历史观测，不能用其判定仿真的物理结局。

原 job 最后 progress 自身时间为 06:33:37.137329 UTC，记录仿真 1265 s、wall `71462.03031056374 s`。最终保留的 `states.jsonl` 为 **345685317 bytes**、mtime 06:46:58.951994 UTC，最后可见完整状态时间为 `1274.8430578523012 s`，文件尾不是完整 JSON 行；这些不是可恢复的完整 world checkpoint，也不能用 mtime 推断精确断电时刻。[本地重新哈希](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/local_hash_parity.json)与[服务器只读哈希](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_partial_sha.json)一致，states SHA-256 为 `5dea16f8cd568ad9353060e1ac6825534133b41630dbcc19300c248fe4bed025`；该次历史核验时本地 27 个冻结源码/配置文件 SHA 与服务器及 f67 provenance 全部一致，不代表 2026-10-18 规则修订后的磁盘状态。

用户授权下的同步修复已完成：[服务器更名记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/syncthing_repair/server_rename_repair.json)保留 11 个历史路径的内容、mtime 与 SHA，仅将非法名称改为 Windows 兼容名称；[本地忽略配置记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/syncthing_repair/local_ignore_repair.json)在本地 `.stignore` 原 448-byte 前缀之后追加 28 个精确临时测试/cache 排除项。该配置不提交 Git，未删除文件或更改 ACL，原有 receive-only 本地额外文件保留。服务器 10:32 UTC 和本地 10:36:37 UTC 的快照均见 code/results 为 `idle`、errors/pullErrors/needTotalItems 为 0；这不要求两端全目录文件数量相同。

冻结内核没有完整 world checkpoint 恢复功能，原 CLI 不支持 resume，原路径重跑还会与既有 job 目录冲突。恢复编排使用新增的 [wrapper](../../scripts/server/recover_rotating_slots_diagnostics.py)，独立编排提交为 `3068e6ad8a5ffe12c8ac8ba212c1f2f72f6782f0`，脚本 SHA-256 为 `ae62823d50b32a4b540b2fb5c8c9e0d016e8155e26e1954e96e40eb980ba5836`；它在冻结 27 文件 glob 之外，物理数值代码仍为 f67。对应[定向验证记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/wrapper_tests.txt)为 33 PASS，包含真实 0.2 s 独立 pilot 的输出路径 smoke；该 pilot 不是正式矩阵结果或 G09 证据。

恢复协议先验证精确 61 项目录、60 项来源的 provenance/SHA 与 replay，再只将 `traffic_rate180_seed71` 按原冻结参数从仿真 0 重跑，输出到新的 `/home/user/training_outputs/terminal/rotating_slots_recovery_f67b5c8_20261017/diagnostics`。原输出所有字节、60 个结果对象及既有审计保留；单个独立 job 使用 1 worker，数值库线程均为 1。只有新项真实 `COMPLETED` 且 replay 验证通过，才将原 60 项与新 1 项按规范顺序形成完整 `diagnostics.json`，其中显式记录恢复来源、实际命令和两次 attempt 的计时，不将中断前 wall 拼成连续运行耗时。

[服务器只读预检](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_preflight/preflight_observation.json)已于 10:39:59.942–10:40:42.901 UTC 完成：`PREFLIGHT_PASSED`、退出码 0，精确目录 61 项、已验证来源 60 项，唯一重跑项为 `traffic_rate180_seed71`；stderr 为空，预检未创建目标 `diagnostics` 目录。

[实际启动记录](../../../training_outputs/terminal/rotating_slots_recovery_f67b5c8_20261017/parent/launch.json)确认单项恢复于服务器时钟 2026-10-17 10:47:24.664872 UTC 启动，wrapper PID `7697`、Python PID `7698`；工作目录仍为 `/home/user/projects/terminal`。实际命令如下：

```bash
/home/user/envs/terminal_env/bin/python scripts/server/recover_rotating_slots_diagnostics.py --source /home/user/training_outputs/terminal/rotating_slots_formal_f67b5c8_20261016/diagnostics --output /home/user/training_outputs/terminal/rotating_slots_recovery_f67b5c8_20261017/diagnostics --code-commit f67b5c8462433ba701191c493a37e4a8e472334e --recovery-commit 3068e6ad8a5ffe12c8ac8ba212c1f2f72f6782f0 --workers 1
```

新 root 的 `parent/recovery.log`、`parent/recovery.exit` 分别保存日志及退出标记，`parent/launch.json` 保存启动命令与进程记录。[原始启动后核验](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_launch/verification2.stdout)及[本地时间包围记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/recovery_20261017_1022/server_launch/verification2.meta.json)记录当时状态：2026-10-17 服务器快照时钟 10:50:13.097697–10:50:13.118350 UTC，7697/7698 仍运行、cwd/命令正确；`recovery_manifest.json` 为 `RUNNING`，仅重跑 seed 71、复用 60 项、1 worker。progress 自身时间 10:49:16.553638 UTC 记录仿真 `64.55795671792151/5400 s`、wall `70.699175142 s`，failed 0、`truncated=false`、`conservation=true`；states 为 1844750 bytes，最后完整状态时间 `77.00000000000003 s`、`failed=false`，退出标记未出现且日志为空。相同 SSH 调用的本地时间为 10:49:28.677594–10:49:29.051304 UTC，exit 0、stderr 为空；该次服务器时钟约比本地快 44.24 s，不沿用断电前的时钟差。

### 第二次断电与单 seed 并行执行准备

用户确认服务器发生第二次断电并恢复。[重启后原始核验](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/server_after_powerloss/corrected.stdout)的服务器时钟为 2026-10-18 01:41:44 UTC，boot ID 为 `059f1258-c9ef-4254-b9e0-95282ffe1ff6`、uptime 约 300 s；`7697/7698` 均不存在，也没有匹配的 recovery 进程。[实际 parent 路径补证](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/server_after_powerloss/sync_parent.stdout)确认单核恢复目录的 `parent/launch.json` 仍在、`parent/recovery.log` 为 0 bytes、`parent/recovery.exit` 未生成；该 attempt 的 result、replay 和正式报告也均不存在。此次核验的服务器时钟约领先本地 46.55 s，不沿用旧时钟差。原正式目录的 60 项完成包及 60 项既有审计保持，单核恢复 attempt 也按外部断电中断记录，不记作新的物理 `ENERGY_FAILURE` 或已观察到的实现失败。

第二个 attempt 最后 progress 自身时间为 2026-10-18 01:17:19.908905 UTC，仿真 `1265 s`、wall `52154.054037665 s`；states 最后完整状态为 `1268.9574332323343 s`、`failed=false`，文件尾不以 newline 结束。它们不是完整 checkpoint，不将两个 attempt 的 wall 拼接，也不据文件末时刻推断精确断电时刻。此前 10 月 17 日 22:50 的首次复查安排已是历史记录，当前不沿用它作为未来调度。

用户已授权多核重跑。新增 [并行运行层](../../scripts/server/parallel_rotating_slots_runtime.py)、[并行恢复入口](../../scripts/server/recover_rotating_slots_parallel.py)、[有界 probe](../../scripts/server/probe_rotating_slots_parallel.py)及[对应测试](../../tests/test_rotating_slots_parallel_runtime.py)独立执行层提交为 `b936f6ef7b47156f9c3d13ee941580a5da902ebe`，已推送。该执行层启动核验时冻结 27 文件及 f67 指纹、原 wrapper 均未改动，运行身份同时记录 f67 core 和新执行层。2026-10-18 的当前磁盘已发生规则/显示修订，不能沿用历史 parity 作为此刻源码一致的声明。

并行范围仅为同一查询的六个独立 pad 应急 reference 候选；同一步可合并多机查询，使用持久 spawn 进程池且每数值库线程为 1。返回值按原 query/pad 顺序归并、保持原最小值 tie-break；World 时间推进、SOC/资源提交及所有物理事件仍在父进程按冻结顺序执行。独立 job 数仍为 1，`--inner-workers` 表示该 job 内的候选 worker 数；保持 seed 71、rate 180、3600 s 到达窗口与 1800 s drain，从仿真 0 在新目录重跑，不覆盖前两个 attempt。父进程 CPU/profile 与子进程已完成 pad 任务的 CPU/profile 分开标注，worker RSS 是进程生命周期峰值；父 profile 不代表全部进程开销。新 runner 向旧校验返回实际 JSON 对象，统一 pad 整数键与 JSON 字符串键的表示，不改变物理值；该收尾接口处理不是新的计时结论。

[本地定向验证](../../artifacts/rotating_slots/parallel_runtime_20261018_implementation/pytest_v3.log)为 11 PASS、28.37 s；随后修改的九机三环测试在同一 world/pool 内按 0→0.4→0.8 s 推进，[受影响测试](../../artifacts/rotating_slots/parallel_runtime_20261018_implementation/pytest_mixed_9_continuation.log)为 PASS、136.39 s，逐段物理 metrics、状态、事件、拒绝记录与轨迹完全相等，并检查真实 worker 与清理。这是固定合成 fixture，不是实际 seed 71 后段的替身。[服务器 pilot 终态](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/server_probe/query2.stdout)及[读取时间记录](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/server_probe/query2.meta.json)确认：服务器时钟 2026-10-18 01:56:56.289812992 UTC 以退出码 0 结束，`MATCH`、零差异，所有 worker 已退出且池已关闭，观测到的 native 数值库线程均为 1。[原始 probe.json](../../../training_outputs/terminal/rotating_slots_parallel_probe_b936f6e_20261018/probe/probe.json) 的 SHA-256 为 `f171e41081f18698d198d3e268ec6660b5e34eda98db9e4775d9ab2690483ea4`。reference 分支仅执行 1235 s 附近的 9 个查询（3 环×3 slot），cold、exact-cache 和 t+0.2 s 推进三轮在 1/20/27 inner workers 下均与串行结果精确相等；这不是全部 81 个代表查询的覆盖。

world 分支使用上述固定九机、`callback=False` 合成 fixture；同一 world/pool 先推进 0→0.4 s，再继续至 0.8 s，逐段物理 metrics、状态、事件、拒绝记录与轨迹均精确相等。墙钟计时如下，单位为秒；首段包含 fixture 构造与 spawn 冷启动，接续段复用同一 world/pool：

| 执行方式 | 首段 0→0.4 s | 接续 0.4→0.8 s | 两段合计 |
|---|---:|---:|---:|
| 原 World 串行 | 30.485965 | 15.873037 | 46.359002 |
| 1 inner worker | 31.275495 | 15.941934 | 47.217428 |
| 20 inner workers | 7.943274 | 2.752622 | 10.695896 |
| 27 inner workers | 9.221554 | 3.633975 | 12.855529 |

按本次原始计时计算，20 workers 的两段合计相对串行为 **4.3343×**，接续段为 **5.7665×**；27 workers 接续段比 20 workers 多用 **32.02%** 墙钟时间。主对话据该次观测选择正式恢复使用 **20 inner workers**。这是一轮合成小样本观测，未重复统计显著性，不代表实际 seed 71 后段或整个任务的加速比/ETA；exact-cache 命中计时也不作为仿真推进收益。CPU 仍按父进程与子进程已完成 pad 任务分开记录，该和不穷尽所有子进程启动、初始化、序列化等 CPU 开销。完整计时见[派生汇总](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/server_probe/timing_summary.json)。

### 正式单项并行恢复：已启动

[正式预检](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/formal_preflight/preflight.stdout)及[命令/退出记录](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/formal_preflight/preflight.meta.json)确认：服务器时钟 2026-10-18 02:00:54.278692–02:01:37.371005 UTC，`PREFLIGHT_PASSED`、退出码 0，验证原 60 项完成结果，独立 job 数 1、`inner_workers=20`，当时未创建目标 diagnostics。27 个冻结文件及独立执行文件 SHA 均匹配。

[实际启动原始记录](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/formal_launch/launch.stdout)及[本地时间记录](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/formal_launch/launch.meta.json)确认正式并行恢复于服务器时钟 **2026-10-18 02:03:33.423002 UTC** 启动，wrapper PID `6274`、Python PID `6275`，cwd 为 `/home/user/projects/terminal`。本地 SSH 调用区间为 02:02:46.6715999–02:03:37.0964589 UTC，exit 0、stderr 为空。运行元数据记录的 CLI 如下，正式运行未设置 600/900 s 总 timeout：

```bash
/home/user/envs/terminal_env/bin/python /home/user/projects/terminal/scripts/server/recover_rotating_slots_parallel.py --source /home/user/training_outputs/terminal/rotating_slots_formal_f67b5c8_20261016/diagnostics --output /home/user/training_outputs/terminal/rotating_slots_parallel_recovery_b936f6e_20261018/diagnostics --code-commit f67b5c8462433ba701191c493a37e4a8e472334e --recovery-commit b936f6ef7b47156f9c3d13ee941580a5da902ebe --inner-workers 20 --superseded-attempt /home/user/training_outputs/terminal/rotating_slots_recovery_f67b5c8_20261017/diagnostics
```

新 root 为 `/home/user/training_outputs/terminal/rotating_slots_parallel_recovery_b936f6e_20261018`；父目录控制文件为 `parent/launch.json`、`parent/recovery.log`、`parent/recovery.exit`。同步的 [parallel_execution.json](../../../training_outputs/terminal/rotating_slots_parallel_recovery_b936f6e_20261018/diagnostics/parallel_execution.json)实际字段记录 `status=RUNNING`、`supplied_execution_commit=b936f6ef7b47156f9c3d13ee941580a5da902ebe`、`core_commit=f67b5c8462433ba701191c493a37e4a8e472334e`、`inner_workers=20`、`independent_jobs=1`；[recovery_manifest.json](../../../training_outputs/terminal/rotating_slots_parallel_recovery_b936f6e_20261018/diagnostics/recovery_manifest.json)记录 `reused_result_count=60`、`reexecuted_job_ids=[traffic_rate180_seed71]`、`workers=1`、`checkpoint_resume=false`、`restart_from_simulated_time=0`。两种 worker 字段分别描述内部候选池与独立 job 数。`replacement_reason=user_authorized_multicore_replacement` 及 `superseded_attempt` 指向上一个单核恢复目录；单核 attempt 已因第二次断电中断，未人为 kill 它来完成本次替换。

首次启动核验于服务器时钟 02:04:23.437264 UTC 完成，实际观察到 **6 个 spawn 子进程**，不把配置上限 20 当作当时已生成的进程数；池按任务需要扩展。此时 progress 的已写 snapshot 仍为仿真 0，但 states 最后完整记录已到约 `32.8 s`、`failed=false`，因此不能据 progress 0 判断未推进。`recovery.exit` 与正式 `diagnostics.json` 尚未出现；启动与中间产物已成立，最终完成仍待核验。

[最后一次启动补核](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/formal_launch/startup_check2.stdout)与[本地时间包围记录](../../artifacts/rotating_slots/parallel_seed_recovery_20261018/formal_launch/startup_check2.meta.json)的本地结束时间为 2026-10-18 02:09:06.6546405 UTC，SSH exit 0、stderr 为空；服务器快照时钟为 02:09:52.994722–02:09:53.004143 UTC。6274/6275 及当时 6 个 spawn 子进程仍存在。progress 自身于 02:05:18.248286 UTC 写入仿真 `64.5579567 s`、wall `63.559856218 s`，failed 0、`conservation=true`、`truncated=false`；states 最后完整状态已到 `90.8 s`、`failed=false`，文件为 2560677 bytes、mtime 02:09:31.203105 UTC、以完整 newline 结束。wall 属于该 progress 的写入时刻，不能与较晚 states 快照混用。补核时仍无 result/replay、正式报告或退出标记，日志为 0 bytes。

历史 heartbeat 曾设置为 `ACTIVE`，目标为本地 UTC **2026-10-18 03:09:06（北京时间 11:09 左右）**，用于启动后一小时复查；该时刻已过去，不作为当前有效调度或全程 ETA。

2026-10-18 03:32 UTC 的[修订前直接快照](../../artifacts/rotating_slots/user_observations_20261018/live_frozen_pre_edit/snapshot.stdout)确认父进程 `6275` 和 20 个 worker 存续，progress 395 s、较晚 states 约 419.8 s，尚无最后一项完整 result/replay。旧进程/worker 在修订前已载入 f67 模块；以后若 PID 或 worker 改变，须核查载入来源，不能仅用启动 metadata 或当前磁盘冒充新的运行版本核验。旧项完成后只审计新增回放并合并原 60 项索引；独立 G09 已消费旧 40 包，不等待该项，不重复仿真。严格合并和 V12 人工认可仍需相应版本的真实证据。

## 数据与测量条件

结果来自服务器输出的本地同步副本：`D:\7_codespace\training_outputs\terminal\rotating_slots_formal_f67b5c8_20261016\diagnostics\<job>\result.json`。60 项均记录源码提交 `f67b5c8462433ba701191c493a37e4a8e472334e`、源码集合 SHA-256 `110be32211d37e82772d806d86ae599d0a58a32e01af3465be836a58de902750`、配置语义 hash `0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53`，对应 replay 索引均存在。

服务器为 Intel Core i7-14700，28 个逻辑 CPU、20 个物理核；正式矩阵使用 20 workers。Python 3.12.3、NumPy 2.5.1、SciPy 1.18.1、PyYAML 6.0.3；psutil 未安装，RSS 来自 `resource.ru_maxrss`。OMP、OpenBLAS、MKL、NumExpr、VECLIB 线程均为 1，已记录的两个 OpenBLAS 线程池也均为 1。viewer 在计时仿真中关闭；cProfile 全程启用，使用 `process_time` 时钟。

以下是已执行命令记录，工作目录为 `/home/user/projects/terminal`：

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.scripts.run_stage1_diagnostics --output /home/user/training_outputs/terminal/rotating_slots_formal_f67b5c8_20261016/diagnostics --workers 20 --code-commit f67b5c8462433ba701191c493a37e4a8e472334e
```

断电前的有效直接状态见 [此前更正后的服务器快照](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/summary.md)，此前 benchmark/dt 快照见 [21:46 核验记录](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1346/summary.md)。benchmark/dt 算术汇总保存在 [partial_report_aggregates.json](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1346/partial_report_aggregates.json)，仅派生汇总，不替代或创建正式 `diagnostics.json`。

## 1 秒 benchmark

初始稳定占用数为 10/30/60/83，两种模式各重复 5 次（`r0`–`r4`），共 40 个唯一 job。`stable` 保持稳定占用；`inflight_attempt` 包含额外在途准入构造，20/20 次记录 `inflight_admitted=true`，不能把它理解为纯稳定状态计算。每项仿真时长均为 1 s。83 槽位的五次 `inflight_attempt` 在 1 s 时均记录 `actual_airborne_count=84`，展示了本构造中槽位数不等于总空中容量上限。

表中 P95 使用现有 `benchmark_summary` 的 nearest-rank 定义：排序后第 `ceil(0.95×5)=5` 项，即五次中的最大值。吞吐列为每次“仿真秒 / 墙钟秒”的中位数；RSS 列为五次记录的最大值，按 1 MiB = 2²⁰ bytes 换算。

| 初始稳定占用数 | 模式 | 重复 | 墙钟中位数 / s | 墙钟 P95 / s | CPU 中位数 / s | 仿真秒/墙钟秒中位数 | 最大 RSS / MiB |
|---|---|---|---|---|---|---|---|
| 10 | inflight_attempt | 5 | 83.394 | 161.043 | 83.393 | 0.01199126 | 74.363 |
| 10 | stable | 5 | 182.719 | 186.850 | 182.691 | 0.00547287 | 73.723 |
| 30 | inflight_attempt | 5 | 282.765 | 545.233 | 282.764 | 0.00353650 | 76.129 |
| 30 | stable | 5 | 505.556 | 513.896 | 505.342 | 0.00197802 | 75.895 |
| 60 | inflight_attempt | 5 | 549.017 | 1050.870 | 549.012 | 0.00182144 | 78.508 |
| 60 | stable | 5 | 1033.240 | 1046.116 | 1032.870 | 0.00096783 | 77.441 |
| 83 | inflight_attempt | 5 | 1025.844 | 1980.378 | 1025.836 | 0.00097481 | 79.648 |
| 83 | stable | 5 | 1375.521 | 1488.345 | 1375.205 | 0.00072700 | 78.867 |

这些数值包含初始配置、seed 场景、参考候选构造及连续约束检查、cProfile 和回放归档，表示 **1 s fixture 的端到端成本**，不能当作预热后长运行的每仿真秒成本。cProfile 开销尚未单独量化。进程池会复用进程，`ru_maxrss` 是进程生命周期峰值，可能带入此前 job 的内存高水位，并非每项隔离测得的内存峰值。本协议没有绝对吞吐通过阈值。

这些计时来自 20 worker 同时运行的矩阵，并非隔离单进程基准；各模式中位数不能单独用于断言在途场景固有计算量更低。进程池跨 job 复用，未单独控制每项冷/热缓存条件，表格仅说明该协议下的观测。

### CPU 分类分布

下表对 40 项的 `measured_components.exclusive_cpu_seconds` 按类别求和，百分比以互斥分类合计 **25,532.733033 CPU s** 为分母。没有累加嵌套的 `world_cpu_seconds` 计时。

| 类别 | 含义 | exclusive CPU 合计 / s | 占比 |
|---|---|---|---|
| motion_power | 轨迹求值、形状/界限计算与能源函数 | 16170.903191 | 63.3340% |
| other | 其余已采样函数 | 8844.404061 | 34.6395% |
| separation | safety 间隔函数自身 | 375.407491 | 1.4703% |
| serialization | 回放与 JSON 归档函数自身 | 96.510470 | 0.3780% |
| generation | 轨迹生成函数自身 | 33.533340 | 0.1313% |
| state_update | 其他 world 状态更新函数自身 | 10.097483 | 0.0395% |
| emergency | 指定应急入口/参考路由函数自身 | 1.603405 | 0.0063% |
| resources | 资源日历函数自身 | 0.273593 | 0.0011% |

分类规则以函数归属计 exclusive 时间：应急路径调用的能源、轨迹、间隔函数各归自己的类别。因此参考候选形状检查与能源计算的大部分时间进入 `motion_power`，不能将较大的嵌套应急计时解释为应急路由函数自身耗时。以上是观测到的类别分布，不是进一步的原因归属。

## dt 对照

三个 job 使用同一已承诺正常 final 脚本、初始 SOC 0.8、short 服务，执行步长分别为 0.05/0.1/0.2 s，推进至 203.5 s。表中能源、SOC、接地与最终位置均取 `result.json` 顶层字段，连续最小距离取顶层 `minimum`，没有用粗粒度的 `metrics.minimum_separation` 替代证书。

| dt / s | 能源 / J | 最终 SOC | 接地 / s | 最终 ENU / m | 连续最小距离区间 / m |
|---|---|---|---|---|---|
| 0.05 | 13807771.506749442 | 0.6209387448548933 | 143.5 | [100, 0, 1.7763568394002505e-15] | [126.3420680482248, 126.34283293216514] |
| 0.1 | 13807771.506749535 | 0.6209387448548933 | 143.5 | [100, 0, 1.7763568394002505e-15] | [126.3420680482248, 126.34283293216514] |
| 0.2 | 13807771.506749481 | 0.6209387448548933 | 143.5 | [100, 0, 1.7763568394002505e-15] | [126.3420680482248, 126.34283293216514] |

最大能源差为 `9.313225746154785e-8 J`，按 `(max−min)/min` 计算的相对差为 `6.744915891461807e-15`。相对 dt=0.2 s，三个最终位置的欧氏差均为 0 m；五个事件的类型、飞机 ID 与顺序一致，最大事件时差为 `3.410605131648481e-12 s`（dt=0.1 的服务结束事件），接地时差为 0 s。三份最小距离结果均为 `CERTIFIED`，区间宽度 `0.0007648839403344709 m`，小于 0.001 m，见证时刻均为 `58.366943359375 s`。这些 dt 对照只对应此固定脚本，不替代 traffic 与 scheduler 的结果分析。

## 十四个已完成 traffic 结果

下表覆盖截至快照已完成的十四个 traffic job，ID 格式为 `traffic_rate<rate>_seed<seed>`，直接取各 `result.json` 的实际 metrics；`failed` 为 `failed_before_landing`，`simtime` 为实际仿真停止时刻。[本轮 traffic 汇总](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/traffic_metrics.json)和[新增结果](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/new_completed.json)保留来源记录。job 的 `COMPLETED` 表示该次脚本已结束，包含物理失败导致提前终止的运行。

| rate | seed | planned | offered | unreleased | admitted | rerouted | normal | emergency | failed | censored | simtime / s | truncated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 11 | 73 | 33 | 40 | 27 | 6 | 18 | 1 | 1 | 7 | 1391.8134693898965 | true |
| 60 | 23 | 52 | 52 | 0 | 50 | 2 | 47 | 3 | 0 | 0 | 5400 | false |
| 60 | 47 | 66 | 66 | 0 | 58 | 8 | 54 | 4 | 0 | 0 | 5400 | false |
| 60 | 71 | 72 | 28 | 44 | 24 | 4 | 12 | 3 | 1 | 8 | 1207.924968881649 | true |
| 60 | 101 | 74 | 74 | 0 | 64 | 10 | 61 | 3 | 0 | 0 | 5400 | false |
| 120 | 11 | 128 | 52 | 76 | 42 | 10 | 26 | 3 | 1 | 12 | 1242.4315865974008 | true |
| 120 | 23 | 117 | 45 | 72 | 40 | 5 | 26 | 2 | 1 | 11 | 1462.8111138523973 | true |
| 120 | 47 | 117 | 89 | 28 | 60 | 29 | 43 | 9 | 1 | 7 | 2500.3071263697407 | true |
| 120 | 71 | 118 | 36 | 82 | 25 | 11 | 12 | 2 | 1 | 10 | 791.6924626874761 | true |
| 120 | 101 | 138 | 33 | 105 | 26 | 7 | 9 | 4 | 1 | 12 | 784.8752089210166 | true |
| 180 | 11 | 188 | 44 | 144 | 32 | 12 | 10 | 2 | 1 | 19 | 688.1399611573279 | true |
| 180 | 23 | 193 | 38 | 155 | 30 | 8 | 11 | 1 | 1 | 17 | 809.4926152758579 | true |
| 180 | 47 | 184 | 50 | 134 | 34 | 16 | 16 | 3 | 1 | 14 | 1012.3628477209778 | true |
| 180 | 101 | 188 | 45 | 143 | 29 | 16 | 13 | 0 | 1 | 15 | 695.9349873539879 | true |

rate 60 的五个 seed 已齐：23、47、101 完成 3600 s 到达窗口及 1800 s drain，11、71 在能源失败时提前终止；rate 120 的五个 seed 也已齐，全部能源失败截尾，rate 180 仅 seed 71 尚未完成。当前十四项共三项完整、十一项能源失败截尾，截尾项均记录 `cost_is_lower_bound=true`；其累计代价是截尾下界，不能与完整窗口结果直接求均值比较。十四个结果均记录 `conservation=true`，未释放和截尾数量保留如表。

十一个 traffic 失败事件的实际时间线如下，依据对应飞机自身的 events 与序列化轨迹；此前事件见 [首批](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1608/failure_timelines_v2.json)、[第二批](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/failure_timelines.json)和[第三批](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0025/failure_timelines.json)，本轮两项及 scheduler_1.0 事件见 [本轮失败时间线](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/failure_timelines.json)。应急声明时间单列，不假定其总与转移完成同刻。

| rate / seed / 飞机 | middle→inner 转移完成 / s | 应急声明 / s | `ENERGY_FAILURE` / s |
|---|---|---|---|
| 60 / 11 / req_000019 | 1187.2060507520548 | 1187.2060507520548 | 1391.8134693898965 |
| 60 / 71 / req_000015 | 963.485330669024 | 963.485330669024 | 1207.924968881649 |
| 120 / 11 / req_000027 | 831.6699368659772 | 951.5348721919898 | 1242.4315865974008 |
| 120 / 23 / req_000026 | 1128.092429556767 | 1172.0000000000005 | 1462.8111138523973 |
| 120 / 47 / req_000068 | 2145.0 | 2209.403663635252 | 2500.3071263697407 |
| 120 / 71 / req_000005 | 465.2359218932885 | 500.83592189328846 | 791.6924626874761 |
| 120 / 101 / req_000007 | 365.21627249850053 | 493.9999999999998 | 784.8752089210166 |
| 180 / 11 / req_000010 | 510.7753377237965 | 510.7753377237965 | 688.1399611573279 |
| 180 / 23 / req_000002 | 500.2694426498897 | 518.6000000000008 | 809.4926152758579 |
| 180 / 47 / req_000017 | 695.9692601901377 | 721.6000000000004 | 1012.3628477209778 |
| 180 / 101 / req_000006 | 409.2134536844115 | 409.2134536844115 | 695.9349873539879 |

十一架飞机均在完成 middle→inner 转移后、inner 应急等待且尚未启动应急 final 时 SOC 耗尽。应急触发可能发生在转移中，也可能在转移结束后的等待中；声明时间如表。rate 180 / seed 23 与 rate 120 / seed 101 此前分别在 765.7694426498897 s、768.6882216022708 s 记录 `PAD_DEADLINE_MISSED`；本轮 rate 120 / seed 47 与 rate 180 / seed 47 分别在 2474.2993756860465 s、920.9692601901377 s 记录同类事件。这是当时记录的固定 pad 释放条件检查，不据此宣布全局不可行。回放检查及事件记录支持应急等待中的物理能源失败，未见已批准转移中意外失败的证据；共享数值内核边界仍适用。

本次使用的 `DeterministicDriver` 是验收驱动器，不是选定的研究 baseline 或优化器；当前十四个结果不代表全部 15 个 traffic 场景的完整统计结论，最后一个 traffic 仍待完成。

## scheduler 部分结果

`scheduler_1.0`、`scheduler_5.0`、`scheduler_10.0` 的原始结果已齐，均使用 rate 120、seed 11、3600 s 到达窗口与 1800 s drain，周期分别为 1/5/10 s。它们是单一 seed 的周期对照，不是额外的 traffic 随机 seed。三个周期及默认 `traffic_rate120_seed11` 的 128 项请求 manifest 一致。[本轮字段对照](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/scheduler_pair_comparison.json)完整递归比较 36 个 `result.metrics` 字段，无排除项；此前已核对默认 traffic 与同协议独立运行的 5 s job 在这些字段、事件类型/飞机 ID 序列及对应时刻均一致。

| 周期 / s | simtime / s | normal | emergency | failed | censored | 决策次数 | 累计代价下界 |
|---|---|---|---|---|---|---|---|
| 1 | 1242.4315865974004 | 26 | 3 | 1 | 12 | 1573 | 1022582.9083206373 |
| 5 | 1242.4315865974008 | 26 | 3 | 1 | 12 | 583 | 1022579.0339872935 |
| 10 | 1242.4315865974008 | 26 | 3 | 1 | 12 | 459 | 1022596.7258709894 |

三项均为能源失败截尾、`cost_is_lower_bound=true`，均记录 planned 128、offered 52、unreleased 76、admitted 42、rerouted 10 及 `conservation=true`。失败飞机均为 req_000027，831.6699368659772 s 完成 middle→inner 转移；1 s 周期于 951.6000000000001 s 在 inner 触发并声明应急，5/10 s 周期则为 951.5348721919898 s。三者均在 `EMERGENCY_WAITING` 耗尽 SOC、未启动应急 final，失败时刻最大差约 `4.55e-13 s`。本表保留实际停止时刻与代价下界，不将其当作完整窗口代价均值。

该 seed 的 1 s、10 s 分别相对 5 s 的记录差异如下；三项均为截尾结果：

| 记录量 | 1 s − 5 s | 10 s − 5 s |
|---|---|---|
| 累计空中时间 | +3.874333343748731 s | +17.691883695837532 s |
| 累计 holding 时间 | −11.125666656250587 s | +27.691883695837305 s |
| 决策次数 | +990（583→1573） | −124（583→459） |
| 能源 | +566547.5837806463 J（+0.15737432882792746 kWh） | +1845704.2151902914 J（+0.5126956153306423 kWh） |
| 累计代价下界 | +3.8743333438178524 | +17.69188369589392 |

三项各有 604 个事件，但 1 s 与 5 s 的序列在索引 232 开始分歧，5 s 与 10 s 的序列也有分歧，不将全部事件强行逐位配对计算时间差。相近失败时刻不代表整条轨迹或事件过程相同。`metrics.minimum_separation` 是粗粒度记录，不能作为连续距离证书或小于 0.001 m 精度的证明。这些差异只描述本 seed 的截尾对照，不支持选定最优周期、完整窗口均值或性能因果结论。

## 已完成回放检查与剩余边界

[43 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_43_report.json)记录了每项实际命令、退出码、审计文件和日志；43 项退出码均为 0、PASS、零问题、连续间隔 `CERTIFIED`。检查命令形式为本地 `venv\Scripts\python.exe -m terminal_marl.scripts.run_stage1_tests --audit <job的replay目录> --output <对应审计目录>`，完整实际参数以索引为准。

43 项的物理 episode 终态均为 `IN_PROGRESS`：benchmark 在 1 s 结束时仍有 holding/在途飞机；dt 的目标飞机已于 143.5 s 接地、203.5 s 完成服务，但 seed 的 `other_ground` 服务需 300 s，尚未结束。这与 job 已完成、回放检查 PASS 属于不同含义。

最先完成的三个 rate 60 回放的独立检查于 2026-10-16 16:12:50 UTC 在服务器并行启动，每项设置 1800 s 超时，输出位于正式运行父目录的 `independent_audits/<job>`。命令、PID 和日志位置见 [启动核验](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1608/audit_batch/startup.txt)，[16:18:43 UTC 查询](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_1608/audit_batch/query2.txt)确认三个审计均已结束：

| rate 60 seed | 审计退出时刻 / UTC（2026-10-16） | 退出码 | 审计 | 物理结局 |
|---|---|---|---|---|
| 11 | 16:14:38 | 0 | PASS、0 issues、CERTIFIED | ENERGY_FAILURE |
| 23 | 16:16:35 | 0 | PASS、0 issues、CERTIFIED | COMPLETE |
| 71 | 16:15:34 | 0 | PASS、0 issues、CERTIFIED | ENERGY_FAILURE |

三份审计的守恒等式均为 true，`documented_hard_failure_runs` 均为 0。合并后的 [46 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_46_report.json)保留原 43 项记录并加入三个 traffic 的实际产物，不重跑已有检查。审计 PASS 表示记录与约束检查通过，物理能源失败仍保留为能源失败。

此前新增七项回放审计于 2026-10-16 20:29:31 UTC 在服务器启动，输出沿用 `independent_audits/<job>`；既有 46 项不重复执行。[query2 的五个失败结局审计](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/audit_batch/query2.txt)与[服务器 20:37:01 UTC query3 的两个完成结局审计](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261016_2027/audit_batch/query3.txt)共同确认七项均退出码 0、PASS、零问题、连续间隔 `CERTIFIED`，守恒等式均为 true，`documented_hard_failure_runs` 均为 0。rate 60 / seed 47、101 的物理结局为 `COMPLETE`，其余新增五项仍为 `ENERGY_FAILURE`。当时合并的 [53 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_53_report.json)包含 53 个唯一 job、53 PASS、零问题，无重复或意外项，当时距全部 61 项尚余八项。

上一轮新增四项回放审计于 2026-10-17 00:26:20 UTC 在服务器启动，[00:28:54 UTC 查询](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0025/audit_batch/query1.txt)确认全部退出码 0、PASS、零问题、连续间隔 `CERTIFIED`，守恒等式均 true，`documented_hard_failure_runs` 均为 0；四项物理结局均为 `ENERGY_FAILURE`。[57 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_57_report.json)保留旧 53 项记录并加入四项，57 个唯一 job 均 PASS、零问题，无重复或意外项，当时尚余四项。本轮三个新完成回放于服务器 04:39:04 UTC 启动独立审计，输出仍为 `independent_audits/<job>`；见 [启动核验](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/audit_batch/startup.txt)。[04:41:51 UTC 查询](../../artifacts/rotating_slots/server_diagnostics_f67b5c8_20261016/heartbeat_20261017_0429/audit_batch/query1.txt)确认三项均退出码 0、PASS、零问题、连续间隔 `CERTIFIED`，守恒等式均为 true，`documented_hard_failure_runs` 均为 0，物理结局均保留为 `ENERGY_FAILURE`。[60 项检查索引](../../artifacts/rotating_slots/diagnostic_audits_f67b5c8_20261016/batch_60_report.json)保留旧 57 项记录，累计 60 个唯一 job 均 PASS、零问题、连续间隔 `CERTIFIED`、退出码 0，无重复或意外项；61 项目录中仅剩 `traffic_rate180_seed71` 未纳入。

这里的独立检查是分离于 World 的记录重算与约束核对，仍共享冻结的轨迹、能源和连续间隔数值内核，不是独立实测 oracle。点质量能源替代模型及重建参考程序的边界见 [sources](sources.md) 和 [qualification](qualification.md)。独立 G09 已直接消费并验证旧 40 包，详见下节；最后一个 traffic 和完整旧矩阵统计另行收口。旧物理结局及审计不改写为新规则结论，人工 V12 仍未完成。

## 2026-10-18 独立 G09 既有 benchmark 验证

[实际结果](../../../training_outputs/terminal/rotating_slots_g09_f67b5c8_20261018/results.json)为标准 qualification report，94 条目录记录中仅 G09 执行并 `PASS`，其余 93 条为 `NOT_RUN`。对应单行文件为 [G09/result.json](../../../training_outputs/terminal/rotating_slots_g09_f67b5c8_20261018/G09/result.json)，不是新跑整套 catalogue。[服务器控制记录](../../../training_outputs/terminal/rotating_slots_g09_control_20261018/launch.json)记录启动时间 `2026-10-18T03:39:57.094739156Z`，[退出文件](../../../training_outputs/terminal/rotating_slots_g09_control_20261018/exit)记录 `03:39:58.572378291Z`、退出码 0；[本地封存](../../artifacts/rotating_slots/user_observations_20261018/g09_server_validation/summary.md)保留 SSH 观测及同步路径。两端时钟分别保留，不相减当作运行耗时。

本次从原 diagnostics 的 `jobs.json` 选取 N10/30/60/83 × stable/inflight_attempt × r0–r4，40/40 均 `COMPLETED`，无 missing、unexpected、duplicates、failures 或 pending。验证器逐项核对冻结参数、实际 airborne=N+inflight 与 inflight admitted、正 wall/CPU/RSS、分项 profile、viewer 关闭/零 CPU、统一源码/配置身份及 source-hash 聚合，并对 40 个真实 ReplayBundle 验证内容 hash、权威轨迹、metrics、录制状态和准入事件。新增仿真数为 0，`original_catalogue_fixture_executed=false`；没有伪造 `FROZEN_FORMAL_MATRIX` 报告，也没有生成无关 analytic replay。

被测来源仍为 `f67b5c8462433ba701191c493a37e4a8e472334e`，`code_sha256=110be32211d37e82772d806d86ae599d0a58a32e01af3465be836a58de902750`，配置语义 hash 为 `0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53`。独立[验证器](../../scripts/server/qualify_rotating_slots_benchmarks.py)文件 SHA 为 `f61653f6f3beb2e4a6975e3701dbf61699dbde958023e45a7891d666cf61fe0e`；执行环境源码集合 hash 为 `e5684cc43b6d6f602b75f9ec06e93f3426f5315c51ead064aeb1376dd7948746`，`current_source_matches_measured_source=false`，环境 commit 为 `UNAVAILABLE_SET_TERMINAL_CODE_COMMIT`，未借旧 commit 标签冒充当前环境。完整实际命令、环境和来源 mapping 保留在结果的 `verifier` / `source_evidence` 中。

结果汇总保留 8 组各 5 次 wall/CPU/RSS 的 median 与 nearest-rank P95、原 exclusive 分项和 individual inclusive 函数值，不将 inclusive 时间相加。上文历史计时表保持不变。G09 表示这些旧测量记录齐全且与归档一致，没有绝对吞吐通过门槛；它不表明稳态、实时性能、大规模 RL 吞吐或 2026-10-18 新规则的性能合格。旧 partial 原件保持，V12 人工认可与新规则执行验证仍需各自证据。
