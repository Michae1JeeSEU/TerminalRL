# Stage2 首轮成对运行：60秒中间记录

本记录只汇总双方均完成的 **T_common=60仿真秒**，不代表3600秒实验结束。
数值读取自同步输出目录中的 `campaign.json`、两方法的 `summary_0060.json`、
`initial_references.json`及`independent_audit_0060.json`；启动、准备额度和进程信息由主对话核对。
运行继续接续各自检查点，
未在首次对照后重启t=0。后续终态、完整结果和人工观察仍需更新。

| 状态维度 | 当前记录 |
|---|---|
| implementation | **COMPLETE（本文实现范围内）**；不是穷尽正确性或部署认证 |
| correctness | 已有自动回归 **207 passed、1 skipped**；有限断言与缺口见 [acceptance.md](acceptance.md) |
| experiment | **RUNNING，中间状态**；尚非COMPLETED_3600、BUDGET_LIMITED或INVALID_STOP终态 |
| real_time_observation | **MEASURED_NOT_CERTIFIED** |
| human_review | **HUMAN_PENDING** |

运行源码提交为 `6d7d20d8be060b3cf405190cfdf3b087dd1f3e79`。
服务器输出根目录为 `/home/user/training_outputs/terminal/stage2_first_campaign_v1_20261021`，
本地对应 `D:\7_codespace\training_outputs\terminal\stage2_first_campaign_v1_20261021`。
这里记录的是该提交发起的运行，不把后续研究文档提交追认为已运行的源码。

主对话记录的正式运行启动时间为 **2026-10-20T17:36:59Z**；父进程PID为 **82739**，
wrapper PID为 **82731**。`campaign.json`记录两方法独立进程、物理核心配置 **[0,2]**。
UTC **17:40:34** 的实际进程快照确认：Energy-slack worker **82743** 的affinity为 **0**，
RH-CP-SAT worker **82744** 的affinity为 **2**，tracker PID为 **82742**，数值库线程环境均为1。
原始核验在 `preparation/first60_export_audit_process_snapshot.txt`，因此这里同时区分配置和实际进程证据。
此前一次launcher在Python启动前因工作目录未正确设置而失败；原始证据保留，工作目录已修正。
该失败未运行仿真、未改变场景、未通过重新开始仿真清零预算。

两方法共用一次物化的 **172条** seed71到达manifest，释放窗口为[0,3600)；SHA256为
`10b97225bde7992af59c79857b0c387bde69ec3b1153e36867a9d42406ad4bc6`。
初态仍为36空中+3地面服务；36个初始正常后续参考中 **10 KNOWN、26 UNKNOWN**。
UNKNOWN记录保留空的能耗/余量字段，没有据此重抽SOC、删飞机或改种子；它不等于已发生失败或物理无解。
准备阶段的busy36单callback微检查和本次闭环运行分开记录，不能将其提交的预测动作当成本窗口接地。

| 60秒同窗口指标 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| 能源失败F（电池/峰值） | 0（0/0） | 0（0/0） |
| 成功接地L（正常/应急） | 0（0/0） | 0（0/0） |
| admitted / 空中pending | 41 / 41 | 41 / 41 |
| 上游pending / 分流 | 0 / 0 | 0 / 0 |
| 初始空中cohort pending | 36 | 36 |
| 新请求已释放、已准入、pending | 5 / 5 / 5 | 5 / 5 / 5 |
| 总终端能耗J | 173987480.33721232 | 173987480.33721232 |
| C_legacy | 2270.8648355119203 | 2270.8648355119203 |
| 累计空中+上游时间秒 | 2270.8648355119203 | 2270.8648355119203 |
| 决策 / noop / fallback | 20 / 19 / 0 | 20 / 19 / 0 |
| 正常callback P95 wall秒 | 4.5204356 | 5.0585795 |
| 正常callback >5秒 | 0/20（0%） | 2/20（10%） |
| 完整dispatch P95 wall秒 | 5.1646469 | 5.7447648 |
| 完整dispatch >5秒 | 2/20（10%） | 4/20（20%） |
| 首块advance wall秒 | 92.8790626 | 98.0185357 |
| 首块含检查点wall秒 | 94.2687962 | 99.4494080 |
| summary守恒标记 | true | true |

双方此刻的主词典序键相同：
`(0, 0, 0, 173987480.33721232, 2270.8648355119203)`。
41架空中pending是尚未完成任务，不算失败；初始三架地面服务不计新接地。
本轮策略的 `post_failure_model=abstract_isolation_no_cascade` 保持记录；
该60秒窗口尚无能源损失，没有把失败隔离抽象描述为现实事故后果模型。
本表不据此解释19次noop的原因，也不将相同累计指标推断为所有内部候选或后续决策相同。
只有20次调度样本，P95等分位数为描述值，尤其不能凭少量样本声称已达到5秒实时目标。
完整dispatch包含公共应急/准入，不能只用正常callback或solver时间替代。

RH-CP-SAT首60秒的summary记录 **99条raw OPTIMAL层级状态**。
这表示求解器针对当时有限整数目录的相应层级返回OPTIMAL；它不是99个物理全局最优方案，
也不自动证明全部词典序层级、前缀认证、连续航迹空间或未来需求的全局最优。
应结合每次决策状态、认证记录和实际提交，保留LEX_LIMITED等限制状态的含义。

预算采用主对话保守汇总的 **600秒准备额度**，其中已知准备费用/额度汇总为545.504秒，
其余作为保守余量；这不是对全部人工工作时间的精确测量，编码与人工阅读不计自动实验预算。
`campaign.json`中的准备阶段累计值为 **600.5518747950846秒**。
budget延续原epoch，campaign经过9.5小时停止新增计算，总计最多10小时，余下至多30分钟用于归档/检查。
失败launcher没有清零这一额度；预算后续是否耗尽以持久化budget及最终状态为准。

主对话在 **2026-10-20T17:39:01Z** 的初步观测取运行段t≈122秒、完成比例p=1/60，
得到粗估总运行段 `T≈7320秒`、剩余 `R≈7198秒`。这只是首次60秒吞吐的线性估算，
尚未反映后续交通密度、应急、候选规模或两方法分歧；它不保证完整3600秒能在预算内结束。
准备额度另计入campaign预算，不能从运行段估算中抵消。
主对话已通过应用工具成功创建当前对话的heartbeat，id为 **stage2**、状态 **ACTIVE**，
间隔119分钟；创建UTC为 **17:40:46.963**，下一次约在 **19:39:47Z**，即北京时间
**2026-10-21 03:39:47**。主对话也核对了持久化`automation.toml`。
保存的接续上下文包括运行、命令、PID、600秒准备额度、完成后分析与Git交付；未保留睡眠中的监控worker。
调度创建已确认；届时仍须重新检查实际进度和产物，不能把该时间视为任务完成保证。

首60秒的 `energy_slack/replay_0060/` 和 `rh_cp_sat/replay_0060/` 已导出，可供实际人工观察。
两份 `independent_audit_0060.json` 经主对话复核，并由本文读取同步文件确认：

| 60秒独立审计项 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| status / issue_count | PASS / 0 | PASS / 0 |
| continuous_separation | CERTIFIED | CERTIFIED |
| physical_outcome | IN_PROGRESS | IN_PROGRESS |
| state_frames | 69 | 69 |
| energy_intervals | 2555 | 2555 |
| continuous_pair_runs | 980 | 980 |

这次审计在UTC **17:42:39–17:42:40**、核心4、线程1完成。
其范围是独立事件/状态/能源/资源账本检查，连续安全和能源仍复用冻结数值内核；
它不是对现实物理模型的独立经验oracle。`IN_PROGRESS`保留窗口尚有未完成任务的事实，
`PASS`不表示3600秒实验完成或真人已接受PyVista结果，人工状态仍为 **HUMAN_PENDING**。
60秒且有初始积压的单seed中间窗口不是一小时容量测量，不外推接地数，不给统计显著性或方法普遍优越结论。
