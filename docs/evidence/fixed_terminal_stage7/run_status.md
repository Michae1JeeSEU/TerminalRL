# 阶段7运行状态

更新日期：2026-10-14。用户单独授权[阶段7Learning to Search计划](../../fixed_terminal_stage7_learning_to_search_plan.md)。阶段6历史验收结论保留。

**pilot已通过流程核验；正式data attempt1已结束，33/36成功、3个val任务失败，原件归档已完成，旧source场景验证源码修复已通过主审及33项检查，等待提交推送、同步与attempt2。** 正式模型、264episodes及15paneljobs尚未运行，Step7.1分析尚未启动。下文保留pilot全部事实与本次失败；原pilot观测、来源和四个smoke模型文件SHA保存于[pilot执行证据](pilot_execution.json)。

冻结输出：服务器`/home/user/training_outputs/terminal/stage7_learning_to_search_20261014`；本地`D:\7_codespace\training_outputs\terminal\stage7_learning_to_search_20261014`。data36jobs(108train/36valcontexts)，单模型一次fit，264episodes，15隔离paneljobs(30states/240rows)，strict summary；具体协议见[冻结协议](../../experiments/fixed_terminal_stage7_protocol.md)。

## 正式data attempt1与source验证修复（2026-10-14）

Attempt1实际源码为`c143d7fec71013063ef28091ae67e26be1bc312a`，11:19:01北京时间启动，time wrapper PID3473436、dispatcher PID3473437；workers20、threads1。主对话读取`tmp_stage7_preflight/checks/formal_data_20261014.json`后核验全部任务终态：dispatcher duration149.636576 s（time wall2:29.69），exit1，33成功、3失败，pending/running均0，无残留研究进程。完成132context：train108、val24，缺少12个val context；analyze没有启动。

失败任务为val/high-low/4201、val/high-medium/4201及val/medium-low/4201，均在仿真前的旧source场景比较阶段报告`stage4 frozen scenario differs from the frozen generator`。对全部36场景的对照原件保存于[source场景比较](source_scenario_comparison.json)，逐字节复制自本地`tmp_stage7_preflight/checks/formal_source_scenario_comparison_20261014.json`。全部config、times、类型与离散字段一致；仅上述3场景的4个`requests[*].soc`有限float末位不同，最大绝对差`1.1102230246251565e-16`。

主对话已限定最小修复：仅对旧generator/source一致性验证中的同类型、有限`requests[*].soc` float允许abs≤1e-12、relative0；其他scenario字段与config继续exact。实际执行继续采用stored原SOC/场景，原source字节保留，不以Linux重建场景替换。此项是source artifact兼容性验证规则，非物理容差、非test调参；不改物理、搜索、模型、data种子或矩阵。旧snapshot artifact float比较仍沿用原规则。

原件归档已于11:26:12北京时间完成。主对话已读取并接受`tmp_stage7_preflight/checks/data_attempt1_archive_20261014.json`：输出ROOT的`data_attempt1/`完整保存旧data树、execution_data state、progress、72个job日志及formal_data_launch_logs，共936files逐项before/after SHA256与bytes一致，state/progress原hash保留；33generation/132context和3份各887 bytes的失败stderr均保存，无残留研究进程，pilot未移动。

恢复预案由主对话确定：源码修复已通过主审和data/models33项验证（8.09 s、exit0，日志`tmp_stage7_preflight/checks/source_soc_pytest_20261014_1.log`）；提交推送及Syncthing核对后，只执行一次完整36任务重放同矩阵。须严格核对144context并分析，再经Step7.1主审后进行唯一正式fit。**提交推送、Syncthing核对及attempt2尚待完成；不记录为已恢复或正式训练已启动。**

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

继续沿用当前chat的heartbeat `7-learning-to-search`，不重复创建。已授权下一步依次为正式data36jobs重放→Step7.1主审→一次正式GradientBoostingClassifier fit并冻结→264episodes（workers20、threads1）→15隔离paneljobs（worker1）→严格summary、文档和结果包。此前准备阶段的正式run_sourceSHA随后实际设置为attempt1的`c143d7fec71013063ef28091ae67e26be1bc312a`；attempt1已按上文结束，重放来源与启动时间待提交同步后确定。smoke模型的train/val使用同一pilot仍仅为流程检查，没有新增fit、重训或调参。

正式data初估T0=10 min：medium pilot四context23.47 s×36jobs/20workers约42 s理想耗时，再乘约6倍SOC/load异质系数与约2倍负载/尾部余量，约8.5 min，取10 min。服务器预检为28CPU、约28GB可用内存，data采用workers20、数值库threads1。隔离formal panels初估138.64 s×15jobs×1.5约52 min，后续按实际隔离同状态任务耗时更新，不扩展矩阵。以上是排程预估，不是pilot性能结论。

### 10:41:38运行中历史观测

当时beam_adaptive与hybrid已生成各5个输出文件，最终exit/hash尚待核对；full_reference仍运行，elapsed2:53，输出增长、stderr为空。四次主动查询后Luna结束持续监控，主对话创建heartbeat等待复查。该记录保留为运行中历史状态，已被上面的C、D终态核验接续。

本地检查分别为整合71项通过（13.21 s）、接受口径23项通过（4.16 s）、另一次summary/package14项通过（1.97 s）。这是三次独立运行，不累计成一次套件。以下保留实现、小样本及口径确定的历史记录。

## 实现与小样本证据

- Core worker报告：20项core针对性测试通过，8.19 s；CLI smoke两context累计26闭合rollouts、两context真实partial节点5705/4409，模型只fit一次。context schema v2，109维feature schema保持v1；真实trace widths为3/5/8/128，仅full128保留根闭合label。这些是smoke结果，不属于正式train/val。
- 评价、dispatch、summary、package针对性检查：22项通过，4.25 s，venv Python，绝对basetemp `D:\7_codespace\terminal\tmp_stage7_qa_accepted`。与上面的core检查分别执行，没有伪称为一次合并运行。
- 真实小样本接缝：复用core已生成的smoke模型`tmp_stage7_preflight/core_cli_smoke/models`，没有新增fit；`tmp_stage7_eval_seam`保存一个horizon2的hybrid episode及同snapshot八方法panel。episode累计unique rollouts12、expanded nodes13、candidate checks325，最终验证可行、模型hash不变；raw summary为1episode/1generation/1state/8rows，状态明确为`fixture_or_pilot_incomplete`。
- 检查覆盖独立partial pool、共享root科学值一致而各次runtime不同、请求与J/work重新汇总、common full_reference退化分层、缺失/重复/缓存/最终验证异常、36/264/15任务矩阵、显式cwd/threads1、exit与产物核验、phase互斥、一次fit的context identity/hash绑定及ZIP CRC/entry hash/manifest引用aux。单class diagnostic Dummy只在训练positive/negative计数证实单class时接受。

目前无运行中的stage7任务；失败的正式data attempt1 PID与全部记录已按上文保留。正式模型、264episode和15正式panel尚未运行；最终A/B/C研究判断待主对话读取正式证据后给出。

## 正式实验前确定接受口径（2026-10-14）

主对话在正式data、训练及test尚未开始前，依据阶段7真实partial展开/剪枝目标确定主计算条件：共同非退化同状态的实际candidate_checks与完整wall均至少减少10%。旧unique rollout+wall双10%条件继续作为辅助；expanded_nodes、unique_rollouts与candidate_checks分别列ratio、总计和均值，不能宣称全部资源同时降低。质量门槛及模型、搜索、训练、seed、物理和矩阵保持不变。

本次口径确定未读取当前pilot功效，也不依据test。主对话报告当前pilot仍依据`ee6d`源码运行，本地修订不改变该pilot；新的正式来源提交待pilot终态之后记录。本段没有A/B/C研究结论。

本次本地验证：原22项针对性检查与新增“rollouts持平、checks和wall减少可满足primary但旧辅助不成立”回归，共23项通过，4.16 s，exit0；绝对basetemp为`D:\7_codespace\terminal\tmp_stage7_qa_candidate_primary`。未启动新的模拟、训练或正式实验。
