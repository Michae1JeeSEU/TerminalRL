# 阶段4学习辅助调度执行记录

用户[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)已按[方法](arrival_restructure/stage4_learning_scheduler.md)与[冻结协议](experiments/fixed_terminal_stage4_protocol.md)完成。完整事实和研究结论见[结果报告](evidence/fixed_terminal_stage4/results.md)。rank有离线排序信号，但尚无优于同预算beam的综合证据；学习必要性仍未证明。

1. **准备与冻结已完成。** 原样保存计划，核对运行环境，冻结新seed分组、统一池、简单模型及评价预算。阶段3闭合/实际排序反转限制保留。
2. **实现与tiny预检已完成。** 93维可见特征、train-only scaler、闭合J/Δ标签、两类MLP、top3+anchors及独立复查完成；10项新增、67项合并测试通过。pilot仅验证流程，不计正式集。
3. **train/val与固定训练已完成。** 36个数据任务exit0，train108/val36 contexts；两模型各seed4401、300epoch训练一次。以informative val top3召回、val loss、最早epoch选择value17/rank35后冻结。
4. **首次test已完成。** 27个新场景生成108个离线snapshot，六方法162episode；合计189任务全部exit0。独立offline评价exit0、6.014 s。四模型文件hash未变，无补采样、重跑或追加计时矩阵。
5. **汇总与研究判断已完成。** 保留全部分组、配对J、全请求结局、安全与完整链成本。rank原始排序提高未形成优于beam的最终保优率，在线结果有失败与耗时权衡；不按结果更改预算或删除数据。

阶段4工作已完成。当前优先保留确定性full_reference与beam_topk，rank保留实验性结果；后续新研究需用户另定范围，不自动开展GNN/attention、offline RL、actor/PPO、teacher动作模仿或几何变化。阶段1–3计划和证据保持原样，状态见[CURRENT_STATUS](CURRENT_STATUS.md)。
