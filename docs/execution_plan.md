# 阶段5学习筛选验证执行计划

按[用户阶段5计划](fixed_terminal_stage5_learning_validation_plan.md)、[方法](arrival_restructure/stage5_learning_validation.md)和[冻结协议](experiments/fixed_terminal_stage5_protocol.md)推进方向A。**当前为实现与预检阶段；正式模型训练、新test及完整评价尚未开始。**

1. **范围与旧证据诊断已记录。** 只读阶段4冻结产物，精确复现旧top3，记录anchors、误差、特征别名及旧online证据缺口；不把旧430x test当作新独立test。[诊断简报](evidence/fixed_terminal_stage5/diagnosis.md)保留结论边界。
2. **实现和小样本验证进行中。** 四类评分器、训练入口与8项模型检查已完成；诊断7项检查通过。pilot5001四模型3epoch/trees仅验证流程。评价链需完成同状态、预算、输出聚合及序列化小样本核对，主对话查看实际diff和输出后才能启动依赖实现的正式任务。
3. **正式训练待执行。** 固定旧train/val、model seed5401，每类一次300epoch或100树，scaler train-only；validation TOTAL B5选checkpoint并锁定。禁止正式超参数覆盖或根据test重新选择模型。
4. **冻结后新test待生成。** test5301/5302/5303共30场景。运行11arms×30完整episode，以及90snapshot质量、24state真实计时、9context大池和9旧context continuation敏感性；范围、配对与预算按协议，不按结果扩矩阵。
5. **汇总与交付待执行。** 保留全请求N/E/F/C、质量预算曲线、同状态与各自轨迹完整成本、计算量和安全证据。主对话依据全部证据解释收益及限制，更新接口文档与结果后按项目规范提交、推送。

服务器使用`lab-server`及`/home/user/envs/terminal_env/bin/python`，正式CPU运行，输出进入`/home/user/training_outputs/terminal`。28 logical CPU下拟20workers、threads1，具体并行与T0由主对话根据pilot核定；正式启动前等待本地提交同步。运行启动验证、有限查询和超过一小时的heartbeat接续遵守AGENTS规则，当前没有正式PID或完成记录。

## 阶段4已完成的执行记录

用户[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)已按[方法](arrival_restructure/stage4_learning_scheduler.md)与[冻结协议](experiments/fixed_terminal_stage4_protocol.md)完成。完整事实和研究结论见[结果报告](evidence/fixed_terminal_stage4/results.md)。rank有离线排序信号，但尚无优于同预算beam的综合证据；学习必要性仍未证明。

1. **准备与冻结已完成。** 原样保存计划，核对运行环境，冻结新seed分组、统一池、简单模型及评价预算。阶段3闭合/实际排序反转限制保留。
2. **实现与tiny预检已完成。** 93维可见特征、train-only scaler、闭合J/Δ标签、两类MLP、top3+anchors及独立复查完成；10项新增、67项合并测试通过。pilot仅验证流程，不计正式集。
3. **train/val与固定训练已完成。** 36个数据任务exit0，train108/val36 contexts；两模型各seed4401、300epoch训练一次。以informative val top3召回、val loss、最早epoch选择value17/rank35后冻结。
4. **首次test已完成。** 27个新场景生成108个离线snapshot，六方法162episode；合计189任务全部exit0。独立offline评价exit0、6.014 s。四模型文件hash未变，无补采样、重跑或追加计时矩阵。
5. **汇总与研究判断已完成。** 保留全部分组、配对J、全请求结局、安全与完整链成本。rank原始排序提高未形成优于beam的最终保优率，在线结果有失败与耗时权衡；不按结果更改预算或删除数据。

阶段4工作已完成。当前优先保留确定性full_reference与beam_topk，rank保留实验性结果；已授权阶段5范围见上文，不扩展GNN/attention、offline RL、actor/PPO、teacher动作模仿或几何变化。阶段1–3计划和证据保持原样，状态见[CURRENT_STATUS](CURRENT_STATUS.md)。
