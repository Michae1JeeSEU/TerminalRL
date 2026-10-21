# Stage2 v3完整3600秒成对运行：RUNNING

用户已授权使用通过120秒切环复查的v3，重新从t0进行3600秒成对运行。当前记录为 **RUNNING**；
最新检查已见双方`summary_1440.json`、`checkpoint_1440.json`及states文件，仍为RUNNING；
双方summary payload已在本地解析，`time=1440`；这不代替最终审计或完整性验收。未观察到invalid、error或exception。
此前[120秒pilot](transfer_fix_120s_status.md)已获用户 **HUMAN_CONFIRMED_TRANSFER_FIX**，仅限所观察的O→M/M→I问题；
本次完整3600秒回放的人工状态仍为 **HUMAN_PENDING**，不预判全局最优性、实时性或长窗口表现。

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
