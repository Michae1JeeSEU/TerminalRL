# Stage2 首轮成对运行：INVALID_STOP，最后共同窗口840秒

**本轮已无效停止；修复已通过统一服务器回归，原实验不改写。** `campaign.json`为`INVALID_STOP`，最后双方完成的共同
检查点为 **840仿真秒**，不是完整3600秒结果，也不是预算耗尽终止。脚本退出码虽为0，
不能覆盖结构化无效状态。本报告只比较840秒前的已保存数据，不把错误后的状态纳入有效结论。

| 状态维度 | 本轮最终记录 |
|---|---|
| implementation | 已交付的有界实现范围为COMPLETE；失败恢复和CLI修复已通过有界验证 |
| correctness | 修复后219 collected、218 passed、1 skipped、0 fail/error；保留旧207通过/1跳过证据及其覆盖缺口 |
| experiment | **INVALID_STOP；T_common=840秒** |
| real_time_observation | **MEASURED_NOT_CERTIFIED**；正常callback和完整dispatch均有超过5秒的调用 |
| human_review | **HUMAN_PENDING**；主对话目视检查了曲线，尚不等于人工PyVista签收 |

运行源码提交为`6d7d20d8be060b3cf405190cfdf3b087dd1f3e79`。
原始证据服务器根目录为`/home/user/training_outputs/terminal/stage2_first_campaign_v1_20261021`，
本地同步根目录为`D:\7_codespace\training_outputs\terminal\stage2_first_campaign_v1_20261021`。
输入为`campaign.json`、两方法的`summary_0840.json`、`audit_0840.json`、各自351行
`decisions.jsonl`、`energy_slack/error.json`，以及初态/manifest和历史60秒记录。
研究文档的后续提交不追认为已运行源码。

正式运行UTC **2026-10-20T17:36:59Z–18:44:12Z**，约 **4033秒**；
`campaign_wall_s=4812.677915491629`，含准备额度等campaign费用。
准备阶段累计600.5518747950846秒；其中主对话保守记入600秒准备额度，已知费用/额度汇总为545.504秒，
不是对全部人工工作时间的精确测量。9.5小时计算截止与10小时总上限未通过重启清零。
主对话确认heartbeat **stage2已成功删除**、无相关进程留存；本轮没有重跑、预算重置或参数变更。
曾经的首60秒粗估和待唤醒时间已失效，本文不保留继续等待的承诺。

## 共同840秒结果

下列物理指标两方法完全相同，主键也相同，**不能把comparison的ranking列表先后当作赢家**。

| 每种方法各自的共同窗口指标 | 数值 |
|---|---:|
| 能源失败F；电池/峰值 | 11；11/0 |
| 实际成功接地L；正常/应急 | 20；8/12 |
| offered；已释放新请求 | 90；54 |
| admitted；空中pending；上游pending | 64；33；3 |
| 分流 | 23 |
| 应急费用+分流费用 | 64200（50400+13800） |
| 终端总能量J | 2756145667.3665805 |
| 累计空中+上游时间秒 | 38646.33424121503 |
| C_legacy | 11102846.334241215 |
| 首次失败仿真时刻秒 | 661.6249818546747 |
| 首次失败前/后接地数 | 15/5 |
| summary守恒 | true |

共同主键为`(11, -20, 64200, 2756145667.3665805, 38646.33424121503)`。

| cohort（两方法相同） | admitted | 接地 | 失败 | 空中pending | 上游pending | 分流 |
|---|---:|---:|---:|---:|---:|---:|
| 初始空中36架 | 36 | 20 | 9 | 7 | 0 | 0 |
| 新请求 | 28 | 0 | 2 | 26 | 3 | 23 |

20次接地全部来自初始空中cohort；新请求在该截止前没有接地，不能把清理初始积压解释为持续来流容量。
三架初始地面服务始终不计入本窗口新接地。剩余空中与上游为未完成，不算失败。
有11次能源损失及失败后5次接地；`post_failure_model=abstract_isolation_no_cascade`仅为任务失败隔离抽象，
不模拟事故连锁，因此不能把该有损失的吞吐称为安全运营容量。单seed、840秒窗口不外推一小时接地数。

## 实际调度时间与有限目录记录

正常callback采用执行器完整边界；dispatch还包含公共应急/准入。表中均为351次调用，
wall单位为秒，未展开raw数组。仿真使用逻辑时钟，没有把观测计算延迟注入物理运动。

| 正常callback wall | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| mean / P50 | 3.471771 / 3.535002 | 3.807527 / 3.826386 |
| P95 / P99 | 4.665974 / 4.946208 | 5.208966 / 5.500819 |
| max | 5.235754 | 5.759527 |
| >5秒 | 3/351（0.85%） | 34/351（9.69%） |

| 完整dispatch wall | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| mean / P50 | 9.717999 / 9.393523 | 9.948977 / 9.457394 |
| P95 / P99 | 18.053508 / 19.460254 | 17.963040 / 19.154020 |
| max | 19.975489 | 19.759017 |
| >5秒 | 246/351（70.09%） | 277/351（78.92%） |
| sum(dispatch wall)/840秒 | 4.060735 | 4.157251 |

这轮不能声称达到5秒实时目标，尤其不能以约3.5–3.8秒正常均值或solver时间代替完整dispatch。
现有记录只支持描述性耗时比较，不构成硬实时或延迟下执行安全认证。

现有`dispatch_timings.jsonl`各351条记录还给出以下阶段wall累计/均值；
其中`emergency_wall_s`合并监测与公共救援，没有可独立统计的monitor/rescue子字段。

| dispatch字段：累计秒 / 每调用均值秒 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| emergency_wall_s | 1971.635489 / 5.617195 | 1938.593940 / 5.523060 |
| admission_wall_s | 220.788575 / 0.629027 | 217.053012 / 0.618385 |
| normal_wall_s | 1218.591547 / 3.471771 | 1336.442051 / 3.807527 |
| dispatch_wall_s | 3411.017645 / 9.717999 | 3492.090958 / 9.948977 |

在已记录的三个阶段中，公共应急合计的wall累计与均值最大；公共应急+准入均值约6.246222/6.141444秒。
这是字段级测量，不进一步拆成未记录的函数耗时，不据此推断异常或调度质量的原因。

以下直接聚合每方法351行`decisions.jsonl`；同一调用可有多个截断标签，标签计数不可相加当调用数。

| 决策记录聚合 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| 正常决策 / noop / fallback | 351 / 343 / 0 | 351 / 343 / 0 |
| 非空提交调用 / 已提交前缀总数 | 8 / 9 | 8 / 9 |
| 至少一机目录不完整的调用 | 351 | 351 |
| 列数 min / max / mean | 10 / 105 / 60.236467 | 10 / 105 / 60.245014 |
| 新轨迹构造尝试总数 | 448083 | 448767 |
| 记录的cache_hits总数 | 44897 | 45108 |
| CANDIDATE_WORK_CUTOFF | 258 | 256 |
| AIRCRAFT_ATTEMPT_LIMIT | 266 | 268 |
| GLOBAL_ATTEMPT_LIMIT | 51 | 51 |
| SOFT_BUDGET | 3 | 34 |
| MOVING_COLUMN_LIMIT | 3 | 3 |
| watchdog / stale | 0 / 0 | 0 / 0 |
| 正常决策status | FEASIBLE 351 | CATALOG_LEX_OPTIMAL 302；LEX_LIMITED 49 |
| raw solver阶段status | 不适用 | OPTIMAL 1777 |
| solver候选认证status | 不适用 | SAFE 1742；UNSAFE 35 |

这些是有限目录和日志计数，不把目录截断称为物理无解，也不凭noop计数推测其原因。
1777条OPTIMAL仅是当前有限整数目录中相应层级的原始solver状态；35条UNSAFE暂定组合被认证拒绝，
不能以solver状态代替认证。CATALOG_LEX_OPTIMAL也不代表连续航迹空间或所有未来需求的物理全局最优。

| 记录阶段累计wall秒 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| candidate_generation_and_single_vehicle | 1047.091356 | 1035.683071 |
| selection | 162.433465 | 0.559578 |
| hint_construction | 不适用 | 162.108163 |
| model_build | 不适用 | 44.462403 |
| resource_conflict_build | 不适用 | 0.282789 |
| cp_solve | 不适用 | 0.737423 |
| selected_validation | 2.881253 | 87.639252 |
| commit | 4.732989 | 3.795511 |

阶段按现有排他计时字段聚合；CP的selection字段不能脱离其hint、建模、求解和认证字段单独当作完整算法时间。
本表不据此提出未验证的因果或性能改进结论。

## 审计范围、曲线与无效停止

两份`audit_0840.json`均为 **PASS、issue_count=0、continuous_separation=CERTIFIED**，
`physical_outcome=ENERGY_LOSSES_CONTINUED`；各有1040个state frames、47239个energy intervals、4427个continuous pair runs。
这只覆盖已完整保存的 **0–840秒**，不覆盖后续报错状态，也不能把整轮INVALID_STOP改成成功。
独立账本检查仍使用冻结轨迹/能源/间隔数值内核，不是对现实能源模型的独立经验oracle。

实际曲线文件为
[comparison_curves.png](D:/7_codespace/training_outputs/terminal/stage2_first_campaign_v1_20261021/comparison_curves.png)，
保留在同步输出目录，没有复制大文件到Git。主对话目视检查：0–840每60秒汇总的接地/失败、空中/上游、
pad忙数与累计能量曲线重合；这不证明所有内部决策相同。图中的耗时为正常planning，**没有画完整dispatch**；
完整dispatch P95约18秒，应以上表单独报告。
`replay_0840/`可供后续真人观察；当前仍为 **HUMAN_PENDING**。

`energy_slack/error.json`保留异常：
`POST_FAILURE_RECOVERY_ERROR: resource ('outer', 16) already reserved`。
traceback经过`_isolate_energy_failures → _calendar_stage → reserve`，即失败隔离后的calendar重建出现异常。
主对话从真实checkpoint核对：outer16旧req11预约为[159.5293,842.73986]，新req49目标预约为
[846.56012,inf]，两者合法不重叠；新机入场轨迹829.60455–859.60455，完成后旧req11仍在离场至917.14993。
重建时将新持有者占槽起点误用admission=829.60455，而非既有预约846.56012，制造了历史区间重叠。
这是主对话已核验的诊断，不把该软件异常归入额外电池失败。

主对话已检查实际diff并核验失败恢复修复：同机同槽保留已有真实预约起点（无已有预约才fallback），
失败隔离仅局部释放，不全量重建幸存者日历；涉及world及回归测试，不改变几何、种子或调度预算。
两项新增回归在旧代码为 **2 failed、2.52秒**，修复后 **2 passed、1.98秒**；
Stage2 world+audit本地为 **27 passed、9.78秒**。资源单元夹具以真实轨迹检查当前状态可继续70→70.1秒，
不宣称完整复现了历史inbound。
CLI修复已由主对话检查diff并新增9项测试：run/resume在归档和完整输出后，依据持久化INVALID_STOP退出1，
status/report仍可正常读取错误实验。本地CLI+campaign **23 passed**；红阶段 **6 failed、3 passed**。
统一服务器首轮验证在独立目录`stage2_recovery_validation_20261021`记录 **217 passed、1 skipped、1 failed**，
pytest48.35秒、wall48.650秒，UTC19:46:49–19:47:38，证据为`regression.log/xml/timing.txt`。
主对话核对唯一失败：旧upstream测试要求重建占槽起点为admission=5，真实inbound目标capture为21.85913；
该旧断言与修复后实际槽占用语义冲突。测试更新保留原capture，并仍核对release=0/admission=5，不回退实现。
主对话已检查该测试diff，本地upstream模块 **14 passed、15.70秒**；
同12模块服务器最终复跑为 **219 collected、218 passed、1 skipped、0 fail/error**，
pytest **48.49秒**、wall **48.767秒**，UTC **2026-10-20T19:49:20Z–19:50:09Z**，退出0。
运行核心4、四个数值库线程环境均1，五个相关源码/测试文件的本地与服务器hash一致。
唯一skip仍是已安装OR-Tools时不适用的缺依赖负向测试，两个真实CP测试均通过。
证据写入独立目录`D:\7_codespace\training_outputs\terminal\stage2_recovery_validation_20261021`的
`regression_final.log`、`regression_final.xml`、`regression_final.timing.txt`；XML SHA256为
`73dd80fc156858d39cb94e82cc2950bec14a5d9f44cefa7cd88e75f41fed1d66`。
首轮失败证据不覆盖；修复验证不替换本次正式运行原始证据，也不是正式场景重跑。
旧实验退出0记录不更改；正式campaign未因此重跑、改变研究参数或重置预算。
旧207通过/1跳过回归保留为运行前证据；本次异常明确暴露失败恢复路径的覆盖缺口，不能说旧测试已覆盖该bug。

## 首60秒历史（非当前状态）

首次共同60秒双方F=0、L=0，admitted与空中pending均41，上游0；能量173987480.33721232J，
C_legacy=2270.8648355119203，20次决策、19次noop、0次fallback。
正常P95为4.5204356/5.0585795秒，完整dispatch P95为5.1646469/5.7447648秒。
两份当时的`independent_audit_0060.json`为PASS、0 issues、CERTIFIED，物理状态IN_PROGRESS。
这是当时窗口的历史事实，不再表示运行仍在进行。

初态正常参考保留10 KNOWN、26 UNKNOWN，无重抽SOC或删飞机；manifest仍为172条，SHA256
`10b97225bde7992af59c79857b0c387bde69ec3b1153e36867a9d42406ad4bc6`。
准备阶段单callback微检查与正式840秒闭环结果分开，不把预测提交当作接地。
历史进程82739/82731及workers82743、82744曾核验在物理核心0、2、线程环境1；当前主对话确认无相关进程留存。
