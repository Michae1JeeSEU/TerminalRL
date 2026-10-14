# 阶段7运行状态

更新日期：2026-10-14。用户单独授权[阶段7Learning to Search计划](../../fixed_terminal_stage7_learning_to_search_plan.md)。阶段6历史验收结论保留。

当前处于实现及针对性小样本核验，尚未启动正式stage7数据、训练、episodes或panels，不填造PID、进度或结果。小样本为fixture/pilot证据，不代替正式矩阵。运行启动、source SHA、PID/log、预计耗时与实际退出由后续执行证据更新。

冻结输出：服务器`/home/user/training_outputs/terminal/stage7_learning_to_search_20261014`；本地`D:\7_codespace\training_outputs\terminal\stage7_learning_to_search_20261014`。data36jobs(108train/36valcontexts)，单模型一次fit，264episodes，15隔离paneljobs(30states/240rows)，strict summary；具体协议见[冻结协议](../../experiments/fixed_terminal_stage7_protocol.md)。

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
