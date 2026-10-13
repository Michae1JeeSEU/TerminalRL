# 阶段5执行记录：研究验收完成

## 阶段6当前范围（2026-10-13）

用户已授权[阶段6学习辅助联合调度](fixed_terminal_stage6_learning_assisted_scheduling_plan.md)，当前进入implementation/preflight，正式264episode和30同状态面板（15jobs、240方法行）尚未启动。复用阶段5冻结MLP rank/GB，无需重训；固定物理、93维特征、pool14和FCFS闭合J。八方法比较采用TOTAL B3→B5→B8→full及已到达机群正常完成上界证书，证书不保证未来到达或全J最优。质量条件epsilon_normal=0与同状态非退化计算减少10%门槛预先冻结，A/B/C判断等待正式证据。

当前依据为[方法](arrival_restructure/stage6_learning_assisted_scheduling.md)、[协议](experiments/fixed_terminal_stage6_protocol.md)、[运行状态](evidence/fixed_terminal_stage6/run_status.md)。阶段1–5历史事实与结论保持；下文“下一阶段另定范围”等表述属于阶段5结束时的历史状态，本节记录后续明确授权。阶段6仍不开展PPO、actor、teacher动作模仿、offline RL、GNN/attention或改变几何。


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
