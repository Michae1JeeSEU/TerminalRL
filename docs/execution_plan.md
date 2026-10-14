# 阶段6执行记录：研究验收完成

## 阶段7研究验收完成（2026-10-14）

[阶段7用户计划](fixed_terminal_stage7_learning_to_search_plan.md)的36datajobs（108train/36valcontext）、唯一GB fit、264episodes、15generation/30states/240rows与严格summary全部完成，四模型hash不变。**B未成立，按A停止扩展当前Learning to Search配置；C未观察到新选中值超过有限full参考。** 保留传统联合优化、beam_adaptive的工程参考及full/B3/B5/B8质量成本tradeoff，不泛称所有learning方法普遍无效，不自动新增模型、训练、Q/value路线或下一阶段。

每方法33场/1,066请求，均有2次能源失败、共16次；间隔违反/未认证均0、最终执行检查feasible，0jobfailed不能写成0能源失败。对adaptive三学习全局质量门槛通过，但共同非退化同状态candidate checks与完整wall均减10%的primary门槛都未通过；对full计算通过、在线质量失败，15全局比较均无同时成立。按冻结J精度，pruning/hybrid对adaptive各有2次全J配对loss，不能写成每场全J不退化。完整结论、全部七J分量、加权正常delay/energy、成本、泛化与边界见[阶段7结果](evidence/fixed_terminal_stage7/results.md)。

研究运行源码（data/model/EP/panel/summary）均为`c892d82a77ec4647e0c9f77fe3f73786ae57bab2`，后续文档SHA不改写actual run来源。结论限定单GB seed7401、固定catalogue8/cap8/scan48、partial3/5/8/128、109维、test7301–7303同profile及6初态变体、已到达FCFS闭合有限reference。首次33成功/3失败的SOC source验证问题及936files归档、全部pilot与一次formalfit事实保留于[运行状态](evidence/fixed_terminal_stage7/run_status.md)。**研究验收已完成，最终版本提交与ZIP打包尚待实际交付，未提前宣称推送或包校验完成。**

阶段1–6历史结论、证据与用户计划保持原样；旧文档中“不自动启动下一阶段”属于当时边界。本次阶段7结论不自动授权新研究；后续传统优化或其他路线须另定范围。

## 阶段6研究验收完成（2026-10-14）

[阶段6用户计划](fixed_terminal_stage6_learning_assisted_scheduling_plan.md)的264episode、15个独立panel jobs（30states/240方法行）及严格汇总全部完成，模型hash未变。**B未成立；相对同机制beam_adaptive未见学习增量计算收益，当前学习排序配置按A停止扩展，保留实验资产。** 学习相对full的非退化根数301→202（减少32.9%），完整wall仅减少约6.4%–6.6%，未达预定双10%门槛；相对beam_adaptive三学习根数都相同，wall比约0.9994/1.0015/1.0013。C的大量验证现象存在，但不支持扩大value或长期Q路线，不启动新训练或下一阶段。

每方法33场/1163请求，仍各有6次能源失败；全部间隔违反/未认证为0、最终执行验证通过，不能写成零能源失败。结论限定pool14、已到达FCFS闭合值、单模型seed5401和本profile新seed及初态占用变体，不泛称所有学习无效。主表、15种质量/计算条件、泛化和成本见[阶段6结果](evidence/fixed_terminal_stage6/results.md)；[方法](arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](experiments/fixed_terminal_stage6_protocol.md)、[运行状态](evidence/fixed_terminal_stage6/run_status.md)保留完整范围与执行事实。源码冻结`f1018d8aaeff03836b13475301087f83ab70a155`。交付版本、ZIP条目与校验结果由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，包链接与最终交付状态见交付回复。

阶段1–5历史计划和结论保持原样；下文“下一阶段另定范围”等属于当时历史状态。后续可另定高效传统优化范围，本次不自动开展新实验、actor/PPO、teacher动作模仿、offline RL、GNN/attention或改变几何。


## 阶段5历史执行记录

阶段5冻结实现、评价与研究验收已完成。存在离线排序信号和部分在线收益，但尚未达到不退化系统质量并明显降低计算的B类门槛，也无长期价值C类证据；不能据此笼统宣称学习完全无效或普遍无必要。保留beamB5/B8及full工程参考，rank/GB保留实验性结果；下一阶段另定范围，不自动扩大网络或启动新实验。详细结果见[结果报告](evidence/fixed_terminal_stage5/results.md)，固定参数见[协议](experiments/fixed_terminal_stage5_protocol.md)，原始运行与失败处理见[运行记录](evidence/fixed_terminal_stage5/run_status.md)。

1. **旧证据诊断与冻结已完成。** 保留阶段4诊断及用户计划；固定93维表示、物理、模型seed5401、train/val和新test分组，TOTAL预算含anchors。
2. **实现、pilot与训练已完成。** 四模型各唯一实际训练并冻结；首次cwd启动失败和pilot输出目录失败保留。101项原检查与13项summary/package检查分别通过，不作为一次合并检查。
3. **正式test已完成。** 360jobs全部exit0，330episode、30snapshot jobs、90base contexts、9large contexts、24计时面板/264行，21:17:04.821508北京时间结束，耗时4317.407811 s，四模型hash不变。
4. **continuation修复与完整重放已完成。** 首轮3成功6失败的33files/133416 bytes逐hash保存；仅旧artifact浮点比较修复，26项检查0.79 s通过，提交b7233cf已推送。第二轮9/9成功，21:37:28开始，外层10.20 s、exit0。正式test与模型没有重跑或改变。
5. **严格汇总与研究验收已完成。** summary一次`completed`且`allow_incomplete=false`，完整分母核对通过。学习离线筛选和部分正常完成收益伴随额外能源失败，同状态计时没有明显加速；32上限未实际扩池，continuation全HOLD不能识别排序变化。这些负面与退化证据全部保留。
6. **版本与交付记录。** 版本与交付校验信息由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，交付状态见最终交付回复。

部署边界维持有限池FCFS闭合参考；仅3个初相、同profile family新seed和单一模型seed，20并发耗时不是实时认证。后续研究另定范围。

## 阶段4已完成的执行记录

用户[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)已按[方法](arrival_restructure/stage4_learning_scheduler.md)与[冻结协议](experiments/fixed_terminal_stage4_protocol.md)完成。完整事实和研究结论见[结果报告](evidence/fixed_terminal_stage4/results.md)。rank有离线排序信号，但尚无优于同预算beam的综合证据；学习必要性仍未证明。

1. **准备与冻结已完成。** 原样保存计划，核对运行环境，冻结新seed分组、统一池、简单模型及评价预算。阶段3闭合/实际排序反转限制保留。
2. **实现与tiny预检已完成。** 93维可见特征、train-only scaler、闭合J/Δ标签、两类MLP、top3+anchors及独立复查完成；10项新增、67项合并测试通过。pilot仅验证流程，不计正式集。
3. **train/val与固定训练已完成。** 36个数据任务exit0，train108/val36 contexts；两模型各seed4401、300epoch训练一次。以informative val top3召回、val loss、最早epoch选择value17/rank35后冻结。
4. **首次test已完成。** 27个新场景生成108个离线snapshot，六方法162episode；合计189任务全部exit0。独立offline评价exit0、6.014 s。四模型文件hash未变，无补采样、重跑或追加计时矩阵。
5. **汇总与研究判断已完成。** 保留全部分组、配对J、全请求结局、安全与完整链成本。rank原始排序提高未形成优于beam的最终保优率，在线结果有失败与耗时权衡；不按结果更改预算或删除数据。

阶段4工作已完成。当前优先保留确定性full_reference与beam_topk，rank保留实验性结果；已授权阶段5范围见上文，不扩展GNN/attention、offline RL、actor/PPO、teacher动作模仿或几何变化。阶段1–3计划和证据保持原样，状态见[CURRENT_STATUS](CURRENT_STATUS.md)。
