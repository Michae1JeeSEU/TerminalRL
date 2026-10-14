# 阶段7运行状态

更新日期：2026-10-14。用户单独授权[阶段7Learning to Search计划](../../fixed_terminal_stage7_learning_to_search_plan.md)。阶段6历史验收结论保留。

**正式数据、唯一fit、264episodes、15generation/30states/240rows与strict summary全部完成，主对话研究验收完成；B未成立，按A停止当前配置扩展，C未观察。** Data/model/EP/panel/summary实际source均为`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`，后续docs SHA不改写actual run来源。最终文档提交与ZIP打包待实际交付，未提前记录完成；研究结果见[results.md](results.md)。下文保留全部失败、pilot和运行中历史。

冻结输出：服务器`/home/user/training_outputs/terminal/stage7_learning_to_search_20261014`；本地`D:\7_codespace\training_outputs\terminal\stage7_learning_to_search_20261014`。data36jobs(108train/36valcontexts)，单模型一次fit，264episodes，15隔离paneljobs(30states/240rows)，strict summary；具体协议见[冻结协议](../../experiments/fixed_terminal_stage7_protocol.md)。

## 最终panel、strict summary及交付准备

Panel内部dispatcher于13:36:16.467834北京时间开始，13:47:19.739465结束；duration663.271637 s、time663.32 s（11.1分钟）、exit0，15唯一任务全部成功，30states/240rows完整finite/semantic检查通过、final_check全部feasible，四模型hash未变，无phase lock或残留进程。外层启动13:36:16.418133与内部开始分别记录；原件见[panels_terminal.json](panels_terminal.json)。初始T0=52分钟保留为历史，实际11.1分钟已记录。

唯一strict summary于14:30:02.756629启动，PID3491350；终态观测14:33:02.342107，time exit0、wall179.74 s、RSS1728872 kB、stderr0。Schema fixed-terminal-stage7-summary-v1、status completed、allow_incomplete=false，无missing formal paths；36datajobs、144context SHA/108train36val、fit_count1冻结核验、264episodes及15generation/30states/240rows均通过，模型四文件hash不变，无lock或残留进程。原件见[summary_execution.json](summary_execution.json)。完整summary20,348,580 bytes，SHA256=`ec2bfd0a09c6abad466c6580900f1733f2480ffe263414020874750576f6fdae`，留在结果目录；Git只保存紧凑投影/CSV，不复制全summary或node traces。

本地最终summary/package14项测试通过，2.46 s、exit0，日志`tmp_stage7_preflight/checks/final_summary_package_tests_20261014.log`；这是独立执行验证，不追加新实验。Package CLI dry-run按真实raw重新汇总及portable summary校对，ready、writes=false、392.403 s，检查4463个source files（本轮最终新增docs之前），没有创建delivery ZIP。最终版本提交、实际包内容/CRC/hash与推送状态待后续真实交付记录，不能把dry-run写成打包完成。

主对话已接受正式研究证据并给出最终A/B/C判断；每方法仍有2次能源失败、合计16，0jobfailed不是0能源失败。后续不自动新训练、扩模型/Q/value、重跑EP或下一阶段，运行与数据资产保留。完整质量、计算、回退/C和泛化边界见results.md，所有先前启动进度描述以下保留为历史观察。

## 正式EP终态与隔离panel启动

EP终态原件见[episodes_terminal.json](episodes_terminal.json)。外层启动11:55:32.967143、内部dispatcher启动11:55:33.018977北京时间分别记录；12:34:44.727322结束。264唯一任务全部completed、0failed、exit0，dispatcher duration2351.708353 s、外层time2351.77 s（39.2分钟），20workers/threads1，264个progress唯一记录；各场48decisions，共12,672条，1,320个预期文件齐全、finite、无重复/缺失，final_check全部feasible、请求守恒通过。四个正式模型hash与冻结值相同，EP无残留进程。

全八方法累计请求结局N/E/F/C=4524/197/16/3791。任务0failed是dispatcher运行结果，**不表示0能源失败**；本组实际存在16个failed_energy请求。最终每arm质量、计算指标和A/B/C结论仍待严格summary与主对话审查，不在运行状态中提前解释方法效益。

Panel唯一启动原件见[panels_launch.json](panels_launch.json)。13:36:16.418133北京时间正常启动，time wrapper PID3487053、dispatcher PID3487054、worker PID3487062，worker1、数值库threads1，source仍c892。启动前26个runtime文件本地/服务器SHA及四个模型hash一致，没有其他研究进程；EP在12:34:44已终态，两phase无重叠。

13:36:33.714555观测为0completed、1running、14pending、0failed，共15；首场low/high/7301 t0 snapshot、8arm JSON与full_values非空，stderr0，主对话已接受启动核验。Panel尚未终态，strict summary未生成。T0=52分钟（pilot138.64 s×15×1.5），预计14:28:16.418133北京时间完成；主对话将同一heartbeat更新到估时完成复查，无常驻monitor，不重复创建。日志位于ROOT/formal_panel_launch_logs，state/progress为ROOT/execution_panels.json、progress_panels.jsonl，准确路径保存在collector。

## attempt2、Step7.1、正式模型及EP启动历史（2026-10-14）

正式episodes唯一启动原件见[episode_launch.json](episode_launch.json)。26个运行源码文件本地/服务器SHA匹配，四个正式模型hash仍冻结；actual EPsource=`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`，与纯文档整理并行，后续docs SHA不可改写该来源。2026-10-14 11:55:32.967143北京时间启动，time wrapper PID3478486、dispatcher PID3478487，20workers、数值库threads1。

11:56:11.124682的历史启动观测为9completed、20running、235pending、0failed，共264；stderr0，首批frozen_input、decisions与events非空，主对话接受启动核验。稍后progress13行仅为followup观察，不表示当时全体完成。初始T0=100分钟、目标13:35:32.967143；11:59:40曾设置同heartbeat96分钟间隔，约13:36复查。实际EP于12:34:44.727322结束、39.2分钟，终态见上方；原估时与观察保留为历史，后续panel遵守不重叠约束。

正式data attempt2于11:35:09北京时间启动，wrapper/dispatcher PID3475605/3475606，source c892，workers20、threads1，attempt2初估T0=6分钟（149.6 s×36/33×2约5.5分钟，取6分钟）。36/36任务成功、exit0，dispatcher156.193307 s、外层time156.24 s；144context（train108/val36）完整finite、109维、trace3/5/8/128正确，source hash及stored scenario/SOC/config/times均无不匹配。Generator/source audit为33场exact、3场SOC末位roundoff，共4处、max1.1102230246251565e-16；实际执行使用stored场景。首轮失败原件保留在data_attempt1，不混入attempt2。

Analysis于11:40:20启动，PID3476384，exit0、5.19 s，144context结果finite。主对话已审查并接受Step7.1。[搜索瓶颈记录](search_bottleneck.md)区分candidate调用与batch调用、cached revisits、有限参考标签与类别不平衡；其中312,713 eligible节点、positive4,575（1.463%）是所有train+val，不是仅train。多width进程累计时间不等于隔离online wall，也不证明学习收益。

唯一正式fit于11:43:40.751676启动，wrapper/Python PID3476873/3476874；source c892，ROOT/data/train、ROOT/data/val→ROOT/models，无smoke。Exit0，wall96.33 s、fit86.591769 s、RSS2204180 kB、stderr0、无残留；100 estimators/depth3/lr.05/seed7401、threshold.5、109维、fit_count1、roundtrip=true、test_read=false，val report-only。Train108context/228,516节点（positive3,413/negative225,103），val36/84,197（1,162/83,035）；144个context SHA/split与四模型文件hash均通过主对话核验，完整原件见model_freeze.json。

Train precision/recall/Brier=.983264/.413126/.0067902，val=.928962/.438898/.0073096。Val accuracy=.991793，但全negative类别基线已约98.62%，不能独自证明模型有用；不据val修改权重、threshold、参数或重训。正式fit只有一个顺序boosting模型，没有独立训练副本可并行，threads1；data与episodes仍20进程，panels隔离1进程。

正式模型一次train/low-high/4101 t0 hybrid决策接缝已通过：exit0、time1.10 s、RSS135760 kB、109维API接受、final_check=feasible/joint_normal_batch，模型hash和原snapshot不变、两个JSON finite，无新fit/episode/panel。原collector与产物hash保存于data_analysis_evidence.json。这仅为流程核验。

已推送c892运行源码固定、数据/模型/接缝主审通过，主对话已授权Luna在本轮纯文档整理并行期间启动264episodes，workers20、threads1，初估T0=100分钟（pilot最慢full/high-high366.67 s×264/20约80.7分钟，再乘1.25余量）；随后15隔离paneljobs，worker1，初估约52分钟（138.64 s×15×1.5）。Fit原T0=5分钟依据smoke2.53 s/9,689 rows×全部312,713作为上界约82 s，再留3倍余量；实际96.33 s已记录。Actual EPsource仍c892，唯一启动与启动验证已经主对话接受，具体PID/时间和部分进度见上方。以上仅作排程与监控预估，没有新增矩阵或A/B/C结论。

## 正式data attempt1与source验证修复（2026-10-14）

Attempt1实际源码为`c143d7fec71013063ef28091ae67e26be1bc312a`，11:19:01北京时间启动，time wrapper PID3473436、dispatcher PID3473437；workers20、threads1。主对话读取`tmp_stage7_preflight/checks/formal_data_20261014.json`后核验全部任务终态：dispatcher duration149.636576 s（time wall2:29.69），exit1，33成功、3失败，pending/running均0，无残留研究进程。完成132context：train108、val24，缺少12个val context；analyze没有启动。

失败任务为val/high-low/4201、val/high-medium/4201及val/medium-low/4201，均在仿真前的旧source场景比较阶段报告`stage4 frozen scenario differs from the frozen generator`。对全部36场景的对照原件保存于[source场景比较](source_scenario_comparison.json)，逐字节复制自本地`tmp_stage7_preflight/checks/formal_source_scenario_comparison_20261014.json`。全部config、times、类型与离散字段一致；仅上述3场景的4个`requests[*].soc`有限float末位不同，最大绝对差`1.1102230246251565e-16`。

主对话已限定最小修复：仅对旧generator/source一致性验证中的同类型、有限`requests[*].soc` float允许abs≤1e-12、relative0；其他scenario字段与config继续exact。实际执行继续采用stored原SOC/场景，原source字节保留，不以Linux重建场景替换。此项是source artifact兼容性验证规则，非物理容差、非test调参；不改物理、搜索、模型、data种子或矩阵。旧snapshot artifact float比较仍沿用原规则。

原件归档已于11:26:12北京时间完成。主对话已读取并接受`tmp_stage7_preflight/checks/data_attempt1_archive_20261014.json`：输出ROOT的`data_attempt1/`完整保存旧data树、execution_data state、progress、72个job日志及formal_data_launch_logs，共936files逐项before/after SHA256与bytes一致，state/progress原hash保留；33generation/132context和3份各887 bytes的失败stderr均保存，无残留研究进程，pilot未移动。

恢复由主对话确定并接续：源码修复通过主审和data/models33项验证（8.09 s、exit0，日志`tmp_stage7_preflight/checks/source_soc_pytest_20261014_1.log`），提交推送及Syncthing核对后，以c892执行唯一完整36任务重放；144context及analysis核验、Step7.1审查与唯一正式fit已按上文完成。首次失败及归档保持原样。

## 已通过的服务器pilot流程

Pilot源码为`ee6d78cf230f7d632027b766cd134ccb80ebaa91`；接受口径与summary/protocol预注册提交为`b4d3b6e4847fe182d2c7a356f67301fe7fefd705`，已推送。`pilot_launch_logs/launch.json`记录启动前23个本地/服务器源码文件SHA及bytes一致。当前pilot继续使用前者，没有因本地口径修订改变运行。

| 阶段 | 已观测事实 | 当前状态 |
|---|---|---|
| A：数据流程 | seed7001、medium/medium、默认horizon240，t0/5/15/90共4context；每context六个有限JSON、109维features、真实trace widths3/5/8/128；exit0，23.47 s，RSS174548 kB | 流程执行完成 |
| B：smoke模型与分析 | 单独smoke GradientBoostingClassifier fit；train/val使用同一pilot，仅用于流程，smoke_only=true、fit_count1、roundtrip=true；exit0，2.53 s，RSS262916 kB。分析`pilot/analysis/bottleneck.json` exit0，.21 s，RSS77800 kB | 仅pilot流程证据；正式模型另见上文 |
| C：episode流程 | 三任务并行、每进程threads1；beam_adaptive medium/medium PID3469033，hybrid medium/medium PID3469035，full_reference high/high PID3469036，均seed7001、standard | 各exit0、48条decisions、5个预期文件、请求守恒、四模型文件hash与B baseline一致，无残留任务；主对话已验收 |
| D：隔离panel流程 | initial_dense两state（t0/t60）、16方法行，PID3471129；time exit0，wall138.64 s，RSS688280 kB | 24个有限可解析JSON，全部final_check=feasible，原snapshot不变/catalogue一致，四模型hash与B baseline匹配，stderr0且无残留进程；主对话已验收 |

| C方法/场景 | wall秒 / RSS kB | 请求数 | N/E/F/C |
|---|---|---|---|
| beam_adaptive medium/medium | 87.80 / 149484 | 28 | 20/0/0/8 |
| hybrid medium/medium | 89.56 / 239056 | 28 | 20/0/0/8 |
| full_reference high/high | 366.67 / 388056 | 57 | 21/0/0/36 |

这三个pilot case的energy_failure_request_ids均为空、spacing violations/uncertified均0、not_arrived0且请求守恒；这些事实仅限定上述三个case，不泛化为正式实验零能源失败。三任务使用不同profile并且并发运行，计时仅用于估时，不证明方法差异。

D于10:59:09.958310北京时间启动；结束时间11:01:28.598310由开始时间加elapsed138.64 s派生（derived），不是直接wait时间戳。collector另记录完成观测11:02:34，最后汇集观测11:03:14；实际stdout/stderr日志路径保留在JSON。D仍使用pilot的`ee6d78cf230f7d632027b766cd134ccb80ebaa91`版本。

继续沿用当前chat的heartbeat `7-learning-to-search`，不重复创建。正式data重放→Step7.1主审→一次正式fit/冻结及API接缝→264episodes已完成；15隔离paneljobs运行未终态，之后才进行严格summary、文档与结果包。Attempt1来源c143、attempt2/analysis/fit及EP/panel来源c892分别保留；后续纯文档SHA不可改写actual run来源。Pilot训练val同pilot仅为流程，正式模型只fit一次，无重训、EP重跑或调参。

首次正式data attempt1历史初估T0=10 min：medium pilot四context23.47 s×36jobs/20workers约42 s理想耗时，再乘约6倍SOC/load异质系数与约2倍负载/尾部余量，约8.5 min，取10 min。服务器预检为28CPU、约28GB可用内存，data采用workers20、数值库threads1。隔离formal panels初估138.64 s×15jobs×1.5约52 min，后续按实际隔离同状态任务耗时更新，不扩展矩阵。以上是排程预估，不是pilot性能结论。

### 10:41:38运行中历史观测

当时beam_adaptive与hybrid已生成各5个输出文件，最终exit/hash尚待核对；full_reference仍运行，elapsed2:53，输出增长、stderr为空。四次主动查询后Luna结束持续监控，主对话创建heartbeat等待复查。该记录保留为运行中历史状态，已被上面的C、D终态核验接续。

本地检查分别为整合71项通过（13.21 s）、接受口径23项通过（4.16 s）、另一次summary/package14项通过（1.97 s）。这是三次独立运行，不累计成一次套件。以下保留实现、小样本及口径确定的历史记录。

## 实现与小样本证据

- Core worker报告：20项core针对性测试通过，8.19 s；CLI smoke两context累计26闭合rollouts、两context真实partial节点5705/4409，模型只fit一次。context schema v2，109维feature schema保持v1；真实trace widths为3/5/8/128，仅full128保留根闭合label。这些是smoke结果，不属于正式train/val。
- 评价、dispatch、summary、package针对性检查：22项通过，4.25 s，venv Python，绝对basetemp `D:\7_codespace\terminal\tmp_stage7_qa_accepted`。与上面的core检查分别执行，没有伪称为一次合并运行。
- 真实小样本接缝：复用core已生成的smoke模型`tmp_stage7_preflight/core_cli_smoke/models`，没有新增fit；`tmp_stage7_eval_seam`保存一个horizon2的hybrid episode及同snapshot八方法panel。episode累计unique rollouts12、expanded nodes13、candidate checks325，最终验证可行、模型hash不变；raw summary为1episode/1generation/1state/8rows，状态明确为`fixture_or_pilot_incomplete`。
- 检查覆盖独立partial pool、共享root科学值一致而各次runtime不同、请求与J/work重新汇总、common full_reference退化分层、缺失/重复/缓存/最终验证异常、36/264/15任务矩阵、显式cwd/threads1、exit与产物核验、phase互斥、一次fit的context identity/hash绑定及ZIP CRC/entry hash/manifest引用aux。单class diagnostic Dummy只在训练positive/negative计数证实单class时接受。

数据、分析、唯一fit、API接缝及264episodes均已终态；失败attempt1 PID与全部原件仍保留。15隔离paneljobs已正常唯一启动、运行未终态，两个phase无重叠。Strict summary尚未生成，最终A/B/C研究判断待主对话读取完整正式评价证据后给出。

## 正式实验前确定接受口径（2026-10-14）

主对话在正式data、训练及test尚未开始前，依据阶段7真实partial展开/剪枝目标确定主计算条件：共同非退化同状态的实际candidate_checks与完整wall均至少减少10%。旧unique rollout+wall双10%条件继续作为辅助；expanded_nodes、unique_rollouts与candidate_checks分别列ratio、总计和均值，不能宣称全部资源同时降低。质量门槛及模型、搜索、训练、seed、物理和矩阵保持不变。

本次口径确定未读取当前pilot功效，也不依据test。主对话报告当前pilot仍依据`ee6d`源码运行，本地修订不改变该pilot；新的正式来源提交待pilot终态之后记录。本段没有A/B/C研究结论。

本次本地验证：原22项针对性检查与新增“rollouts持平、checks和wall减少可满足primary但旧辅助不成立”回归，共23项通过，4.16 s，exit0；绝对basetemp为`D:\7_codespace\terminal\tmp_stage7_qa_candidate_primary`。未启动新的模拟、训练或正式实验。
