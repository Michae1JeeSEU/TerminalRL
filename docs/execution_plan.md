# 阶段5学习筛选验证执行计划

按[用户阶段5计划](fixed_terminal_stage5_learning_validation_plan.md)、[方法](arrival_restructure/stage5_learning_validation.md)和[冻结协议](experiments/fixed_terminal_stage5_protocol.md)推进方向A。**正式训练与四模型冻结已完成；服务器评价pilot已通过，正式test运行中，独立continuation检查失败，比较补丁准备中，阶段5尚未完成。** [运行状态](evidence/fixed_terminal_stage5/run_status.md)记录模型hash、日志及真实启动失败。

1. **范围与旧证据诊断已记录。** 只读阶段4冻结产物，精确复现旧top3，记录anchors、误差、特征别名及旧online证据缺口；不把旧430x test当作新独立test。[诊断简报](evidence/fixed_terminal_stage5/diagnosis.md)保留结论边界。
2. **实现、本地验证与服务器pilot已通过。** 源提交`4811abb805ab03997f5ce4186c6b2610f3c6b3b3`已推送，阶段1–5合并101项检查通过（17.21 s、exit0）。四项pilot首次因wrapper预建输出目录在仿真前退出；seed5001于pilot_v2四新路径的检查现已通过，三个episode各48decisions，snapshot保存t0/15/90并含两个11arm计时面板，均exit0，模型hash不变。
3. **正式训练和冻结已完成。** 固定train108/val36、model seed5401，三个神经模型各300epoch、boosting100树，scaler train-only；22.092 s、maxRSS913020 KB、exit0。TOTAL B5选中linear155、MLP161、GB100、value113，validation保留24/25、25/25、24/25、24/25，只用于选checkpoint。首次launcher因cwd缺失在导入前失败；修正后唯一实际训练成功。模型已冻结，后续不重新训练。
4. **冻结后正式test运行中，continuation比较补丁准备中。** 新530x共360jobs（30snapshot场景+330episode），另9个旧4303 continuation jobs；二者于北京时间20:05:07启动。test初验3完成、20running、337pending、0failed；continuation15.93 s后exit1、3成功6失败，原始`ValueError: old visible snapshot mismatch`保留；九场旧4303比较只发现已存plan坐标的浮点末位差异，artifact比较补丁仅准备中，待test终态后应用。 test5301/5302/5303共30场景。运行11arms×30完整episode，以及90snapshot质量、24state真实计时、9context大池和9旧context continuation敏感性；范围、配对与预算按协议，不按结果扩矩阵。
5. **正式汇总与交付待执行，工具已做独立小验证。** summary7+package6共13项通过（1.10 s），与原101项分开报告；真实pilot汇总exit0、fixture_or_pilot_incomplete，41文件hash一致。 保留全请求N/E/F/C、质量预算曲线、同状态与各自轨迹完整成本、计算量和安全证据。主对话依据全部证据解释收益及限制，更新接口文档与结果后按项目规范提交、推送。

服务器使用`lab-server`及`/home/user/envs/terminal_env/bin/python`，正式CPU运行，输出进入`/home/user/training_outputs/terminal`。28 logical CPU下已授权test20workers与continuation4workers、threads1，24并发预计约17 GB，T0约90 min，估时依据见运行状态。训练任务PID3412631已完成；test启动/dispatcher PID3416299/3416301，continuation为3416300/3416302；test继续运行，continuation比较补丁准备中，四模型hash不变，当前线程heartbeat id=5已创建并view确认，20:13:57北京时间设置81 min，约21:35自动复查并接续授权分析/交付，无常驻worker。运行启动验证、有限查询和超过一小时的heartbeat接续遵守AGENTS规则。

## 阶段4已完成的执行记录

用户[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)已按[方法](arrival_restructure/stage4_learning_scheduler.md)与[冻结协议](experiments/fixed_terminal_stage4_protocol.md)完成。完整事实和研究结论见[结果报告](evidence/fixed_terminal_stage4/results.md)。rank有离线排序信号，但尚无优于同预算beam的综合证据；学习必要性仍未证明。

1. **准备与冻结已完成。** 原样保存计划，核对运行环境，冻结新seed分组、统一池、简单模型及评价预算。阶段3闭合/实际排序反转限制保留。
2. **实现与tiny预检已完成。** 93维可见特征、train-only scaler、闭合J/Δ标签、两类MLP、top3+anchors及独立复查完成；10项新增、67项合并测试通过。pilot仅验证流程，不计正式集。
3. **train/val与固定训练已完成。** 36个数据任务exit0，train108/val36 contexts；两模型各seed4401、300epoch训练一次。以informative val top3召回、val loss、最早epoch选择value17/rank35后冻结。
4. **首次test已完成。** 27个新场景生成108个离线snapshot，六方法162episode；合计189任务全部exit0。独立offline评价exit0、6.014 s。四模型文件hash未变，无补采样、重跑或追加计时矩阵。
5. **汇总与研究判断已完成。** 保留全部分组、配对J、全请求结局、安全与完整链成本。rank原始排序提高未形成优于beam的最终保优率，在线结果有失败与耗时权衡；不按结果更改预算或删除数据。

阶段4工作已完成。当前优先保留确定性full_reference与beam_topk，rank保留实验性结果；已授权阶段5范围见上文，不扩展GNN/attention、offline RL、actor/PPO、teacher动作模仿或几何变化。阶段1–3计划和证据保持原样，状态见[CURRENT_STATUS](CURRENT_STATUS.md)。
