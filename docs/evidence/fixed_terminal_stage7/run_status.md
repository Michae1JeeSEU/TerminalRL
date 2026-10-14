# 阶段7运行状态

更新日期：2026-10-14。用户单独授权[阶段7Learning to Search计划](../../fixed_terminal_stage7_learning_to_search_plan.md)。阶段6历史验收结论保留。

实现及本地核验已完成，服务器pilot流程验证已启动；正式stage7数据、训练、episodes及panels仍未启动。以下服务器事实由主对话提供，最新C阶段观测时间为2026-10-14 10:41:38北京时间；pilot仅用于流程与估时，不代替正式矩阵或研究功效评价。

冻结输出：服务器`/home/user/training_outputs/terminal/stage7_learning_to_search_20261014`；本地`D:\7_codespace\training_outputs\terminal\stage7_learning_to_search_20261014`。data36jobs(108train/36valcontexts)，单模型一次fit，264episodes，15隔离paneljobs(30states/240rows)，strict summary；具体协议见[冻结协议](../../experiments/fixed_terminal_stage7_protocol.md)。

## 当前服务器pilot流程状态

Pilot源码为`ee6d78cf230f7d632027b766cd134ccb80ebaa91`；接受口径与summary/protocol预注册提交为`b4d3b6e4847fe182d2c7a356f67301fe7fefd705`，已推送。`pilot_launch_logs/launch.json`记录启动前23个本地/服务器源码文件SHA及bytes一致。当前pilot继续使用前者，没有因本地口径修订改变运行。

| 阶段 | 已观测事实 | 当前状态 |
|---|---|---|
| A：数据流程 | seed7001、medium/medium、默认horizon240，t0/5/15/90共4context；每context六个有限JSON、109维features、真实trace widths3/5/8/128；exit0，23.47 s，RSS174548 kB | 流程执行完成 |
| B：smoke模型与分析 | 单独smoke GradientBoostingClassifier fit；train/val使用同一pilot，仅用于流程，smoke_only=true、fit_count1、roundtrip=true；exit0，2.53 s，RSS262916 kB。分析`pilot/analysis/bottleneck.json` exit0，.21 s，RSS77800 kB | 流程执行完成；正式models尚未训练 |
| C：episode流程 | 三任务并行、每进程threads1；beam_adaptive medium/medium PID3469033，hybrid medium/medium PID3469035，full_reference high/high PID3469036，均seed7001、standard | 见下方带时间观测；全阶段终态验收尚未完成 |
| D：隔离panel流程 | initial_dense两state、16方法行 | 尚未启动 |

截至10:41:38，beam_adaptive与hybrid已生成各5个输出文件，最终exit、hash及完整验收待核对；full_reference仍在运行，elapsed2:53，decisions/events/frozen输出增长，stderr为空。C全部任务须先核对实际退出、48条decisions、5个预期文件、请求守恒、最终执行检查及模型hash，并确认无残留任务，才启动D隔离panel。

四次主动查询预算已用完，Luna已结束持续监控，没有保留睡眠worker。主对话已成功创建当前chat的heartbeat `7-learning-to-search`，约5分钟后复查。pilot完成后的已授权接续顺序为正式data36jobs与Step7.1主审、唯一正式模型fit并冻结、264episodes、15隔离paneljobs、严格summary及交付；这些正式步骤目前均未启动。

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
