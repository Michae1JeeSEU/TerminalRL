# A4 质量诊断与共同已运行初态：协议、版本和运行清单

状态：**协议与进展登记；研究任务未交付完成**。更新日期：2026-10-30。已登记主对话审核的 R12 短续观测与执行偏差；选组诊断、评分选择、正式运行、正式审计、人工回放和最终版本交付仍为 `PENDING`。单状态短续不构成总体效果结论。

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
| 新 score | **PENDING，尚未实现或选定** |

旧 v2/v3 结果、checkpoint、账本和报告均只读。旧结果用于定位诊断输入与说明历史，不直接和新共同初态的增量结果比较效果。新 `evaluation_profile` 需要明确、非空且双方相同；当前 comparison 入口仅完成参数透传准备，不能将接口替身测试当作新评分实现。

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
| `baseline` | 四个 case | SHA 绑定旧 checkpoint | 当前旧分数 callback 一次，保存实际有界目录 | PENDING |
| `augment` | 四个 case | baseline archive | 最多 6 s 的定向新列及验证；不改 scope/几何/分数 | PENDING |
| `catalog_cp8` | 四个 case | 原 catalog/frame | 固定目录 solver8 控制 | PENDING |
| `augment_cp8` | 四个 case | augmented catalog、同 frame | 相对控制仅增加候选，solver8 | PENDING |
| `catalog_cp24` | R1、P2 | 与 cp8 相同的原候选 hash | 仅 solver allowance 8→24 s | PENDING |
| `offset1`、`offset2` | P2 | 原 projected state/frame、保存 history | 仅 selector offset；不自动扩展到正式臂 | PENDING |

R1 只尝试目标 UID 的 P0/P1，每 pad 最多保留一列，总计最多两列。R4/R12 保留最小 hash 的既有 immediate partial 前缀并调用既有 reference suffix 生成器补完整链；P2 的 partial 选择及 fallback 路径显式记录，每 case 最多补一条完整链。单列 SAFE 与最终组合/可提交 prefix 的认证分开；有界目录缺失、UNKNOWN 或补全未取得候选，均不能宣称物理不可行。目录比较只在冻结目录内解释，原目录与增强目录使用相同验证/search allowance；不把未认证的 objective 当成改进。

120 s 短续仅允许 R4/R12，先审核同态 baseline choice / forced partial 配对及其新 batch/prefix 认证，再显式启动单个 branch；不扫描 UID/case，不由诊断入口自动启动。短续恢复和离线预验证不并入 online 计算收益。[短续入口](../../scripts/stage3/run_selective_short_continuation_a4.py)的 R12 两 branch 已运行并审核，事实登记如下。R4 已完成 prepare，结果为 `NOT_APPLICABLE / NO_FIXED_UID_IMMEDIATE_PARTIAL`；没有运行 R4 短续，也不会补选其他 UID。

P2 的 offset2 实际首组选组、末组选组均与 baseline 相同，不能算多组比较。选组差异问题仍未解决；三个固定组的诊断正在实现，**尚未运行**，不能把实现进度写成三组结果。

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

主对话已审核的限定解释：在这一个开发状态配对中，安全且旧分数更差的推进多完成一次正常落地，同时多耗约 **1.11 MJ**；这支持续行评价可能存在误判，不能推出推进总是比等待更好。候选链补全是优先修正方向；最终判断仍等待选组诊断。本节不选定新 score，也不填写四个正式共同初态比较臂的结果。

## 预算、输出与四臂配对

本轮输出根为 `~/training_outputs/terminal/rs_selective_quality_a4_20261030`。新 shared child-wall 上限为 **7200 s**，诊断、共同初始化、短续、正式臂、测试/审计等实际子进程成本共同扣除。新账本不重置旧账本，也不抵扣旧消耗；历史 `1374.067592 s` 只读引用。失败和超限尝试保留并计费，不覆盖原输出、不自动 retry。

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
| s3r23 / 7324 | PENDING | PENDING | 同一根 common checkpoint、同 score/deadline、相对300 s |
| s3r24 / 7325 | PENDING | PENDING | 同一根 common checkpoint、同 score/deadline、相对300 s |

共四臂，不增加 seed，不用旧 t=0 或旧已运行臂替代其中一臂。common 输出位于 `common_initial/<root>`；正式输出位于 `arms/<root>/<method>/evaluation/<root>/<method>`；短续输出位于 `diagnostic_continuations/<case>/<branch>`。每相对 5 s 保存 boundary/checkpoint/progress。`method_*` 文件仅包含新方法计算；供原审计读取的 scheduler/tasks 文件保留 initializer provenance 加新记录，索引仅在导出副本重编号，独立 mapping 解释计算事件，物理 events/frames 不改写。完整回放包含 initializer 与方法阶段，人工验收仍为 `HUMAN_PENDING`。

## 已记录版本与失败尝试

| 提交 | 用途 | 本文允许的结论 |
|---|---|---|
| `3ff1a15518d420bfe2d5cffb85c73decf58237b7` | reference_v4 三开关、有界诊断、共同初态及短续基础实现 | 工程与接口版本；不是新 score 或效果证据 |
| `674bce676db7f924acce5c9070d0567a0d623a57` | 三个 A4 入口在 dependencies 前保存 launch thread 设置 | 修复 preexecution 启动字段绑定 |
| `1414708b04b387a4b3ce002dd6354278b6e474ed` | archive 解码后的 frame identity 重绑定及短续 source compatibility | 修复 prevalidation 归档绑定，保持数值内容与候选不变 |
| 最终 A4 comparison / score / 文档交付提交 | PENDING | 不用骨架文件代替最终交付 |

| 已保留的失败尝试 | 记录的原因 | 边界与处理 |
|---|---|---|
| preexecution 失败 | 入口遗漏 `d.launch_threads`，线程设置无法按原始 launch 值核对 | 发生在该尝试的 world/规划/候选生成之前；记录保留，由 launch 修复版本接续 |
| prevalidation 失败 | 解码 `Section` 后 repr-based frame identity 变化，原 scope/frame 绑定拒绝 | 发生在该尝试的新规划/候选生成之前；先检查归档内容和绑定，再仅刷新 identity link，不复用旧 ACCEPTED 验证缓存 |

两次失败不覆写、不从预算抹除，不作调度效果或物理不可行证据。相关 freeze 文件只读保留于 `artifacts/rs_select_quality_a4_20261030/diagnostic_source_freeze*.json`。

另行登记两项执行偏差。旧 server lane 遗漏五项线程环境设置，造成 **8 个 preflight 失败**，并在失败后错误继续下一 stage；这些尝试均未进入规划或候选目录创建。8 条 job row 及对应日志已经归档到本轮输出根的 `launch_environment_recovery`，保留失败和计费记录，不能把它们当成目录/质量观测或物理不可行证据。

后续 server 在主对话将该次执行收窄为仅四个 `augment` 后，仍继续执行了此前用户已授权、但不在该次执行范围内的余项。已有数据保留，按其实际输入、预算和执行条件审查；不为获得另一组结果而重跑。P2 offset1 因 CPU 不可用发生前置失败，该失败仍保留。offset2 首组/末组与 baseline 相同的事实如上，不作为不同组选组已经比较的证据。

## 已有服务器测试回执登记

以下为主对话转交的 worker receipt 证据；本骨架任务未直接重读服务器 XML。每项回执为 exit 0、0 skipped。测试集合和版本可重叠，**不将各行通过数相加为独立测试总数**，也不将测试通过视为正式效果结论。XML 均相对本轮服务器输出根。

| 通过数 | 用途/范围 | XML | child wall (s) |
|---:|---|---|---:|
| 175 | 首次 reference_gap 七文件：reference_v4、executable_v3、repair_selection、repair_progressive、followup、replanning、quality_gap | `tests/reference_gap.xml` | 30.970102772000246 |
| 15 | common_initial 单文件 | `tests/common_initial.xml` | 5.797300022095442 |
| 35 | gap + short 首次两文件；不是 thread-fix 集合 | `tests/short_continuation.xml` | 4.438037261017598 |
| 59 | gap + short + common 三文件，包含 launch 修复的 9 个用例 | `tests/preflight_launch_binding_tests.xml` | 9.524737753905356 |
| 13 | comparison 单文件，含真实 tiny checkpoint/replay 及原 audit | `tests/common_comparison.xml` | 14.897858378011733 |
| 56 | archive fix 后 gap + short 两文件，含 archive rebinding、source compatibility | `tests/archive_binding.xml` | 4.680722963996232 |

## 后续登记栏

| 工作 | 状态 | 待填证据 |
|---|---|---|
| 四 case 单因素诊断与选组比较 | 部分执行；最终解释 PENDING | 保留执行偏差、offset2 同组事实；三个固定组诊断尚未运行 |
| 新评分是否修改、公式及接口绑定 | PENDING | 质量诊断依据、代码 diff、针对性验证、source freeze |
| R12 120 s 短续 | 两 branch 与保存回放审计已审核；人工回放 PENDING | 210→330 s 观测、首动作及继承落地归属见本文 |
| R4 120 s 短续 | `NOT_APPLICABLE`，已 prepare、未 run | `NO_FIXED_UID_IMMEDIATE_PARTIAL`；不补选 UID |
| 两根共同初始化 | PENDING | 截面、t0、完整 common SHA、inherited FINAL、预算 |
| 四个正式配对臂 | PENDING | 同 profile/deadline 的四份 binding、完整终态和增量 |
| 四臂保存回放审计 | PENDING | 原 `audit_job` 结果与缺失项，不能由 tiny 审计替代 |
| 人工回放与最终解释 | PENDING / HUMAN_PENDING | 实际观察及研究结论 |
| 最终检查、commit、push 与用户交付 | PENDING | 最终文件清单、测试、提交哈希、push 状态 |

进展登记不代表以上工作完成；R12 单状态配对的限定解释不外推为总体速度、质量或因果结论。
