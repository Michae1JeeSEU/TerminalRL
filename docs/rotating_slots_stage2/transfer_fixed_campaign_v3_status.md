# Stage2 v3目标3600秒成对运行：BUDGET_LIMITED，共同1620秒

本轮已结束，状态为 **BUDGET_LIMITED**，最后共同完成窗口 **1620秒（3600目标的45%）**，退出0，原PID均已退出。
运行wall12050.9秒；这是预算截尾，不是软件失败，也不是完成3600秒或达到容量目标。
heartbeat `stage2-3600`已由工具成功删除，没有继续复查安排。
运行源码仍为`bede731fdf871829bf92d792b3c9450a6317167f`，source hash为
`5751ce59d0a828fdbc08a4c1041d0437b62ba774cc763061da3d82fd11448eea`，未改变源码、参数或协议。

## 共同1620秒质量与实际移动

以下数据来自双方`summary_1620.json`及`replay_1620`，不外推3600秒。

| 质量/成本 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| normal / emergency / 总接地 | 13 / 40 / 53 | 14 / 42 / 56 |
| failure；battery / peak | 39；39 / 0 | 37；37 / 0 |
| admitted / 空中pending / 上游pending | 124 / 32 / 0 | 124 / 31 / 0 |
| reroute / 尚未释放请求 | 0 / 84 | 0 / 84 |
| 总终端能量J | 5418169537.261356 | 5292672592.06787 |
| failure / emergency / reroute费用 | 39000000 / 94800 / 0 | 37000000 / 93600 / 0 |
| 累计空中时间 / 上游等待秒 | 72967.780117 / 917.688036 | 70328.982773 / 530.696020 |
| C_legacy | 39168685.46815275 | 37164459.67879288 |
| 守恒 | true | true |

| cohort：admitted / success / failed / air / upstream | Energy-slack | RH-CP-SAT |
|---|---|---|
| initial_airborne | 36 / 23 / 13 / 0 / 0 | 36 / 22 / 14 / 0 / 0 |
| new_requests | 88 / 30 / 26 / 32 / 0 | 88 / 34 / 23 / 31 / 0 |

| 实际开始 / 完成 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| O→M | 77 / 76 | 87 / 83 |
| M→I | 39 / 38 | 53 / 53 |
| 正常final开始 / 接地 | 13 / 13 | 14 / 14 |
| 正常接地原身份：initial O / M / I / new | 3 / 7 / 3 / 0 | 0 / 4 / 8 / 2 |

开始事件按轨迹hash关联`maneuver_started`，完成用`transfer_completed`或`touchdown`；
原始ring来自fixture、new身份来自manifest，不以final时的inner状态推断原身份。
CP在本单seed、共同1620秒内失败少2、总接地多3，不构成普遍优势。
O→M/M→I已实际执行，但新请求正常接地仅0/2，正常落地大多仍来自初始积压，本窗口正常吞吐表现仍弱。

## 正常与完整dispatch耗时

全部记录按`time≤1620`筛选，N及分位数逐项匹配summary；非空提交调用另外计数，未将空提交排除后冒充全调用统计。

| 正常callback wall秒 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| N / mean | 936 / 4.333239 | 968 / 5.062146 |
| P50 / P95 / P99 | 4.259548 / 5.169030 / 5.686554 | 5.115233 / 5.778566 / 6.042736 |
| max / >5秒 | 6.120553 / 84（8.9744%） | 6.170136 / 655（67.6653%） |
| CPU累计秒 | 4052.325626 | 4891.378980 |

| 完整dispatch wall秒 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| N / mean | 936 / 10.549335 | 968 / 10.447307 |
| P50 / P95 / P99 | 10.386952 / 16.965216 / 18.357005 | 10.200791 / 17.023585 / 18.344938 |
| max / >5秒 | 19.675170 / 834（89.1026%） | 19.788957 / 932（96.2810%） |
| CPU累计秒 | 9865.123365 | 10094.947646 |

| dispatch原字段：wall累计 / 均值秒 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| 公共应急监测+救援 | 5293.640231 / 5.655599 | 4839.195340 / 4.999169 |
| 公共准入 | 524.619049 / 0.560490 | 373.634275 / 0.385986 |
| 正常callback | 4055.911746 / 4.333239 | 4900.157259 / 5.062146 |

正常之外的主要已记录成本是公共应急字段，日志未分别记录monitor与rescue，不进一步拆分归因。
正常内部reference种子wall累计2740.975/2764.693秒（Slack/CP），Slack selection1037.049秒，
CP hint1096.844、model401.313、validation315.872、cp_solve仅1.427385秒。
这些阶段有包含关系，正常还包括executor的`_start_due`等边界工作，不能把嵌套阶段重复相加或将完整dispatch归因于CP求解器。
本轮不构成实时保证。

## 候选与决策记录

| 计数 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| 决策 / 非空调用 / noop | 936 / 115 / 821 | 968 / 127 / 841 |
| 提交前缀 / fallback | 129 / 0 | 153 / 0 |
| commit_status ACCEPTED（含noop） | 936 | 968 |
| 目录不完整 / 完整调用 | 936 / 0 | 968 / 0 |
| CANDIDATE_WORK_CUTOFF / SOFT_BUDGET标签 | 936 / 63 | 968 / 626 |
| 决策status | FEASIBLE936 | LEX_LIMITED680；CATALOG_LEX_OPTIMAL288 |
| raw CP阶段status | 不适用 | OPTIMAL3322；UNKNOWN4 |
| CP阶段validation | 不适用 | SAFE3057；UNSAFE265；未记录4 |
| stale / watchdog | 0 / 0 | 0 / 0 |

同次调用可有多个截断标签；所有catalog均有不完整飞机，不能声称全局最优或把未找到称为全局不可行。
raw OPTIMAL/UNKNOWN仅为有限模型阶段状态；265条UNSAFE是被拒绝的阶段候选，不是已执行违规。
commit ACCEPTED包含noop，不等于每次都有实际移动。

## 审计、边界与人工状态

双方[audit Slack](D:/7_codespace/training_outputs/terminal/stage2_transfer_fixed_campaign_v3_20261021/energy_slack/audit_1620.json)、
[audit CP](D:/7_codespace/training_outputs/terminal/stage2_transfer_fixed_campaign_v3_20261021/rh_cp_sat/audit_1620.json)
均 **PASS、issue_count=0、continuous_separation=CERTIFIED**，physical_outcome为ENERGY_LOSSES_CONTINUED。
审计帧2241/2286、能量区间101261/99383、pair runs21924/22838，守恒检查全true。
summary记录的最小间隔下界为80.0208148312702/80.00949917789727米；该字段来自summary，不冒充独立audit最小距离输出。
审计属于使用共享冻结数值kernel的独立账本检查，不是独立现实能源模型oracle。
主对话已核对实际[比较曲线](D:/7_codespace/training_outputs/terminal/stage2_transfer_fixed_campaign_v3_20261021/comparison_curves.png)及审计报告。

两份replay首末帧顶层`time`均0→1620，索引帧数与audit一致；双方`checkpoint_1620.payload.time`均1620。
方法根states末顶层时间为Slack1654.76620219529、CP1653，属于停止前未完成下一60秒块的尾段；
正式对照与审计回放仅用共同1620，不拼接这些尾段。minimum_separation.time不是帧时间。
本窗口人工仍 **HUMAN_PENDING**；[此前120秒用户确认](transfer_fix_120s_status.md)仅限该短窗口切环问题，不替代本窗口人工验收。

在`D:\7_codespace\terminal`的PowerShell可分别打开本轮回放；以下入口沿用已验证API，本次未运行GUI：

```powershell
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_transfer_fixed_campaign_v3_20261021\energy_slack\replay_1620')).run()"
.\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_transfer_fixed_campaign_v3_20261021\rh_cp_sat\replay_1620')).run()"
```

## 预算与描述统计记账

冻结campaign累计 **34309.89667201042秒**（含归档），budget floor34309.895382642746秒；
计算截止34200秒、总额36000秒保持不变。归档属于计算截止后的总预算内工作，不重置预算。
[紧凑事实JSON](../../artifacts/rotating_slots_stage2_20261022/v3_final_1620_facts.json)及
[一次性描述统计脚本](../../artifacts/rotating_slots_stage2_20261022/aggregate_v3_final.py)只在artifacts中保留，不随文档提交，未重跑audit或仿真。
描述统计实际wall1.5647622秒，外层进程wall1.8909992秒；不把人工或排队等待计为计算。
后处理保守记3秒，后续应承接的累计为 **34312.89667201042秒**，仍小于36000；不修改冻结budget或原始产物，
这项记账不启动新实验。

## 历史启动与1440秒中间记录

以下保留各观察时点记录，RUNNING、待终态或heartbeat复查安排均为当时状态；当前以本页前部终态及heartbeat已删除为准。

## 最新RUNNING检查

观察时间为服务器UTC2026-10-21T15:46:12.622288493Z、本地UTC15:45:32.3696263Z。
runner115961的进程elapsed为10675秒；原wrapper、runner、tracker及两方法PID均仍在，命令目标仍为3600秒。
双方1440边界summary/checkpoint及states存在；当时无`exit_code.txt`或`campaign.timing.txt`，无error/invalid文件或exception。
本地已解析双方summary payload，`time=1440`，campaign仍为RUNNING、human_review为HUMAN_PENDING、target为3600。

| 共同1440秒的中间指标 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| normal / emergency / failed | 13 / 33 / 36 | 14 / 37 / 32 |
| F_peak | 0 | 0 |
| 正常callback mean / P95秒 | 4.380140 / 5.206782 | 5.095715 / 5.801093 |
| 完整dispatch mean / P95秒 | 10.596306 / 17.231564 | 10.476852 / 17.089767 |

states尾部为Slack1500秒、CP1494秒，不以这两个不同尾部作质量对照；上表只用双方共同1440秒summary。
最终审计及归档尚未完成，不据此验收本轮。

budget持久化`elapsed_floor_s=32749.12019944191`，`compute_stop_s=34200`、`total_s=36000`、credit60保持不变。
按启动floor22243.141446475754加进程elapsed10675，估计累计32918.1414秒，距计算截止约 **1281.8586秒（21.4分钟）**。
这是检查时估计，实时budget与最终共同边界/终态记录为准。
以已解析的共同1440秒相对3600目标计为40%；初始短窗口外推已明显低估，不能直接把此进度外推为后段速率。
主对话据剩余预算预计本轮将预算截尾，而非完成3600；这是尚待终态确认的预算判断，不是软件失败结论。
本次不改源码或协议、不重跑、不重置预算。

## 源码与场景

冻结且已测试的运行源码为`bede731fdf871829bf92d792b3c9450a6317167f`，source hash为
`5751ce59d0a828fdbc08a4c1041d0437b62ba774cc763061da3d82fd11448eea`，
候选策略`shared_multistep_columns_v3_reference_seed`。
启动时HEAD`2be2093`为后续文档提交；本次没有新增算法或参数变更。沿用已核验的服务器108 passed、1 skipped，
源码未变，不重复运行该测试集。本次源码完整hash已核验，fixture、manifest和environment lock已与v3 pilot逐字节核对一致。

这是独立的新t0运行，不恢复120秒checkpoint，不拼接旧结果；保持原seed、几何、能源、安全阈值、目标函数及算法预算。
服务器输出根目录为`/home/user/training_outputs/terminal/stage2_transfer_fixed_campaign_v3_20261021`，
同步本地为`D:\7_codespace\training_outputs\terminal\stage2_transfer_fixed_campaign_v3_20261021`。
工作目录为`/home/user/projects/terminal`，已授权的运行命令为：

```bash
/home/user/envs/terminal_stage2_v1/bin/python -u /home/user/projects/terminal/scripts/server/run_rotating_slots_stage2.py run --output /home/user/training_outputs/terminal/stage2_transfer_fixed_campaign_v3_20261021 --target 3600
```

`campaign.start_utc.txt`与`runner.start_utc.txt`均确认起点为服务器UTC2026-10-21T12:48:17Z；
最终`campaign.timing.txt`及`exit_code.txt`待运行终态核对。
wrapper115958、runner115961、tracker115970；方法worker115971/115972的`/proc`实际affinity分别为CPU0/CPU2，
均Threads=1，OMP、OPENBLAS、MKL、NUMEXPR四个线程变量均为1。
启动阶段观察记录服务器UTC12:49:51.382、本地UTC12:49:00.855，相差约50.527秒；
runner进程elapsed为93秒，本轮经过时间据此记录，不直接以两地绝对时钟相减。
Slack状态已见middle00/03的TRANSFER_M_I及outer04的TRANSFER_O_M（该outer行时刻为4秒）；
这些是已观察的状态行，不将其当作完成事件或双方共同截止。

## 继承预算与估时

完整继承前轮最终累计 **22180.690032155544秒**，不重置10小时额度。
本轮唯一prepare退出0，campaign preparation为62.453117738245055秒；恢复记账采用准备elapsed62.45141432021046秒，
其中已含新预检保守60秒credit，不重复加。旧累计与本轮准备elapsed相加得到新floor **22243.141446475754秒**，
启动时剩余计算额度为 **11956.858553524246秒**。
计算截止 **34200秒**、总额 **36000秒**保持不变，实际运行费用继续扣减。

初始T0粗估为`345.036 / 120 * 3600 = 10351.08秒`，约2小时52分31秒，依据上一轮相同120秒pilot的wall。
后续密度和决策耗时并不均匀，这不是保证完成的时间，不能据此承诺在继承预算内跑满3600秒。
若达到预算截止，以最后共同完成边界报告，不扩展预算或补拼旧窗口。

首次自动接续由工具成功创建，heartbeat ID为`stage2-3600`，当时设置170分钟后在同一对话复查。
本地创建UTC2026-10-21T12:50:13，初次预计Asia/Shanghai当晚23:40左右；
依据T0=10351.08秒、观察elapsed93秒，观察时R=10258.08秒，创建时扣除经过时间后约10185.935秒。
该估时仍受后续密度变化影响；不保留驻留监控worker或用睡眠替代接续。
同一heartbeat已由工具成功更新为 **21分钟**，替代原170分钟安排，没有创建重复任务。
更新确认时本地UTC2026-10-21T15:46:56，预计Asia/Shanghai **10月22日00:08左右**复查；
依据观察时剩余计算预算1281.8586秒，扣除至设置时已过时间后向上取整分钟。

## 当前证据状态

启动进程、实际affinity/线程、预检一致性及同一heartbeat更新已确认；双方1440秒summary时间和上述指标已读取。
最终timing、exit文件及审计/归档待终态核对。
终态记录将依据实际共同窗口的质量、正常及完整dispatch耗时、候选截断、CP状态、fallback与阶段成本，
并分列独立审计、回放和人工观察；当前不填完整实验结果或审计PASS。
