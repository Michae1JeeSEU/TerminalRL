# 阶段7运行状态

更新日期：2026-10-14。用户单独授权[阶段7Learning to Search计划](../../fixed_terminal_stage7_learning_to_search_plan.md)。阶段6历史验收结论保留。

**pilot已通过流程核验，准备正式搜索数据生成；正式模型、264episodes及15paneljobs尚未运行。** 主对话已读取并接受C、D实际collector，最新观测分别为2026-10-14 10:57:03.524406与11:03:14北京时间。原观测、来源与四个smoke模型文件SHA保存于[pilot执行证据](pilot_execution.json)。pilot只用于流程与估时，不代替正式矩阵或研究功效评价。

冻结输出：服务器`/home/user/training_outputs/terminal/stage7_learning_to_search_20261014`；本地`D:\7_codespace\training_outputs\terminal\stage7_learning_to_search_20261014`。data36jobs(108train/36valcontexts)，单模型一次fit，264episodes，15隔离paneljobs(30states/240rows)，strict summary；具体协议见[冻结协议](../../experiments/fixed_terminal_stage7_protocol.md)。

## 已通过的服务器pilot流程

Pilot源码为`ee6d78cf230f7d632027b766cd134ccb80ebaa91`；接受口径与summary/protocol预注册提交为`b4d3b6e4847fe182d2c7a356f67301fe7fefd705`，已推送。`pilot_launch_logs/launch.json`记录启动前23个本地/服务器源码文件SHA及bytes一致。当前pilot继续使用前者，没有因本地口径修订改变运行。

| 阶段 | 已观测事实 | 当前状态 |
|---|---|---|
| A：数据流程 | seed7001、medium/medium、默认horizon240，t0/5/15/90共4context；每context六个有限JSON、109维features、真实trace widths3/5/8/128；exit0，23.47 s，RSS174548 kB | 流程执行完成 |
| B：smoke模型与分析 | 单独smoke GradientBoostingClassifier fit；train/val使用同一pilot，仅用于流程，smoke_only=true、fit_count1、roundtrip=true；exit0，2.53 s，RSS262916 kB。分析`pilot/analysis/bottleneck.json` exit0，.21 s，RSS77800 kB | 流程执行完成；正式models尚未训练 |
| C：episode流程 | 三任务并行、每进程threads1；beam_adaptive medium/medium PID3469033，hybrid medium/medium PID3469035，full_reference high/high PID3469036，均seed7001、standard | 各exit0、48条decisions、5个预期文件、请求守恒、四模型文件hash与B baseline一致，无残留任务；主对话已验收 |
| D：隔离panel流程 | initial_dense两state（t0/t60）、16方法行，PID3471129；time exit0，wall138.64 s，RSS688280 kB | 24个有限可解析JSON，全部final_check=feasible，原snapshot不变/catalogue一致，四模型hash与B baseline匹配，stderr0且无残留进程；主对话已验收 |

| C方法/场景 | wall秒 / RSS kB | 请求数 | N/E/F/C |
|---|---|---|---|
| beam_adaptive medium/medium | 87.80 / 149484 | 28 | 20/0/0/8 |
| hybrid medium/medium | 89.56 / 239056 | 28 | 20/0/0/8 |
| full_reference high/high | 366.67 / 388056 | 57 | 21/0/0/36 |

这三个pilot case的energy_failure_request_ids均为空、spacing violations/uncertified均0、not_arrived0且请求守恒；这些事实仅限定上述三个case，不泛化为正式实验零能源失败。三任务使用不同profile并且并发运行，计时仅用于估时，不证明方法差异。

D于10:59:09.958310北京时间启动；结束时间11:01:28.598310由开始时间加elapsed138.64 s派生（derived），不是直接wait时间戳。collector另记录完成观测11:02:34，最后汇集观测11:03:14；实际stdout/stderr日志路径保留在JSON。D仍使用pilot的`ee6d78cf230f7d632027b766cd134ccb80ebaa91`版本。

继续沿用当前chat的heartbeat `7-learning-to-search`，不重复创建。已授权下一步依次为正式data36jobs→Step7.1主审→一次正式GradientBoostingClassifier fit并冻结→264episodes（workers20、threads1）→15隔离paneljobs（worker1）→严格summary、文档和结果包。正式run_sourceSHA将在本轮文档提交后设置真实最新SHA；实际launch时间尚未知。这些正式步骤目前均未启动，smoke模型的train/val使用同一pilot仍仅为流程检查，没有新增fit、重训或调参。

正式data初估T0=10 min：medium pilot四context23.47 s×36jobs/20workers约42 s理想耗时，再乘约6倍SOC/load异质系数与约2倍负载/尾部余量，约8.5 min，取10 min。服务器预检为28CPU、约28GB可用内存，data采用workers20、数值库threads1。隔离formal panels初估138.64 s×15jobs×1.5约52 min，后续按实际隔离同状态任务耗时更新，不扩展矩阵。以上是排程预估，不是pilot性能结论。

### 10:41:38运行中历史观测

当时beam_adaptive与hybrid已生成各5个输出文件，最终exit/hash尚待核对；full_reference仍运行，elapsed2:53，输出增长、stderr为空。四次主动查询后Luna结束持续监控，主对话创建heartbeat等待复查。该记录保留为运行中历史状态，已被上面的C、D终态核验接续。

本地检查分别为整合71项通过（13.21 s）、接受口径23项通过（4.16 s）、另一次summary/package14项通过（1.97 s）。这是三次独立运行，不累计成一次套件。以下保留实现、小样本及口径确定的历史记录。

## 实现与小样本证据

- Core worker报告：20项core针对性测试通过，8.19 s；CLI smoke两context累计26闭合rollouts、两context真实partial节点5705/4409，模型只fit一次。context schema v2，109维feature schema保持v1；真实trace widths为3/5/8/128，仅full128保留根闭合label。这些是smoke结果，不属于正式train/val。
- 评价、dispatch、summary、package针对性检查：22项通过，4.25 s，venv Python，绝对basetemp `D:\7_codespace\terminal\tmp_stage7_qa_accepted`。与上面的core检查分别执行，没有伪称为一次合并运行。
- 真实小样本接缝：复用core已生成的smoke模型`tmp_stage7_preflight/core_cli_smoke/models`，没有新增fit；`tmp_stage7_eval_seam`保存一个horizon2的hybrid episode及同snapshot八方法panel。episode累计unique rollouts12、expanded nodes13、candidate checks325，最终验证可行、模型hash不变；raw summary为1episode/1generation/1state/8rows，状态明确为`fixture_or_pilot_incomplete`。
- 检查覆盖独立partial pool、共享root科学值一致而各次runtime不同、请求与J/work重新汇总、common full_reference退化分层、缺失/重复/缓存/最终验证异常、36/264/15任务矩阵、显式cwd/threads1、exit与产物核验、phase互斥、一次fit的context identity/hash绑定及ZIP CRC/entry hash/manifest引用aux。单class diagnostic Dummy只在训练positive/negative计数证实单class时接受。

尚无正式stage7运行PID、正式模型、264episode或15正式panel结果；最终A/B/C研究判断待主对话读取正式证据后给出。

## 正式实验前确定接受口径（2026-10-14）

主对话在正式data、训练及test尚未开始前，依据阶段7真实partial展开/剪枝目标确定主计算条件：共同非退化同状态的实际candidate_checks与完整wall均至少减少10%。旧unique rollout+wall双10%条件继续作为辅助；expanded_nodes、unique_rollouts与candidate_checks分别列ratio、总计和均值，不能宣称全部资源同时降低。质量门槛及模型、搜索、训练、seed、物理和矩阵保持不变。

本次口径确定未读取当前pilot功效，也不依据test。主对话报告当前pilot仍依据`ee6d`源码运行，本地修订不改变该pilot；新的正式来源提交待pilot终态之后记录。本段没有A/B/C研究结论。

本次本地验证：原22项针对性检查与新增“rollouts持平、checks和wall减少可满足primary但旧辅助不成立”回归，共23项通过，4.16 s，exit0；绝对basetemp为`D:\7_codespace\terminal\tmp_stage7_qa_candidate_primary`。未启动新的模拟、训练或正式实验。
