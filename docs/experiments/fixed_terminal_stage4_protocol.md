# 固定终端区阶段4学习辅助调度协议

状态：**实现与最终预检已完成；正式train/val数据生成已获主审授权、尚未启动，正式训练和test尚未授权。** 本文落实[用户阶段4计划](../fixed_terminal_stage4_learning_scheduler_plan.md)，方法责任见[方法文档](../arrival_restructure/stage4_learning_scheduler.md)。当前API、93个FEATURE_NAMES与产物schema已按实现代码核对，仍须随预检修订同步；未以接口存在宣称测试或正式运行通过。

## 问题与固定范围

验证学习筛选能否以较低的完整在线成本逼近预算传统闭合rollout参考。只近似当前已到达机群在固定FCFS continuation下至240 s的闭合价值，不学习期望未来Q或长期价值，不实施offline RL。[阶段3](../evidence/fixed_terminal_stage3/results.md)已观察到闭合与实际未来排序反转，这一限制继续成立，不能将本阶段监督标签理解为真实未来最优动作。

[阶段1物理合同](../fixed_terminal_problem_definition.md)、容量、SOC/负载分布、生成器、安全、应急证书与外队列模型不变。模型不替代轨迹生成、能源、连续安全、容量或pad检查，不输出自由动作。不实施actor/PPO、teacher动作模仿、GNN/attention或动态几何。阶段3数据只作历史动机，不用于训练。

## 统一候选池与六方法

每个需要新决策的可观测状态现场生成扩展joint候选：cohort8、scan48、keep8、beam128，保留前12个唯一批次，加HOLD和现场FCFS根，按动作去重后最多14根。无阶段2warm-start、无跨方法缓存。候选生成本身已经包含可行性检查，不能只统计筛选后的3根而隐去生成成本。

|方法|根选择与精确评估|
|---|---|
|fcfs|原FCFS实现|
|rolling_horizon|原滚动实现；其冻结目标与本阶段闭合J不同|
|full_reference|独立评估全部统一池根，closed rollout至240，以真实J/key/tie选最大|
|beam_topk|生成beam序前3根，加FCFS/HOLD anchors，去重最多5根，精确rollout后按真实J选|
|value_topk|七维价值模型预测前3根，加同样anchors，最多5根精确rollout后按真实J选|
|rank_topk|排序模型预测前3根，加同样anchors，最多5根精确rollout后按真实J选|

单根池使用共同安全快路径，不调用rollout或模型推理，仍作最终独立提交复查和共同应急处理。5s决策周期、dt1s、完整预约不撤销；退化与非退化决策分开统计。full_reference只是在此有限池、此continuation和闭合需求假设下的最好值，是scope-limited参考，不是全问题最优或CP-SAT oracle。

## 数据划分和采样顺序

所有seed为同分布新种子，每seed固定九个load×SOC组合。joint_window作为共同behavior，使用其已执行prefix，在0、5、15、90 s决策前取snapshot；保留退化状态，不补换时刻。按seed场景分组，禁止随机拆候选导致同状态跨集合。

|用途|seed|场景数|snapshot数|
|---|---|---:|---:|
|pilot|4001|最多九组合的预检用途，具体tiny命令待主审|不计正式集|
|train|4101、4102、4103|27|108|
|validation|4201|9|36|
|test|4301、4302、4303|27|108|

先生成train/val，分别训练一次两模型并固定checkpoint；之后才首次生成test数据并评估。不得先查看test选择epoch或调整模型、预算、seed。正式运行须在tiny输出预检、针对性测试与主审后启动；train/val数据生成已获主审授权，等待源码提交后启动；正式训练和test仍未授权。

## 特征与标签

部署可见手工特征类别为：当前时刻/剩余时域；已到达state/ring计数；holding/external的SOC和等待统计；ring theta聚合；scheduled剩余完成时间与资源预约占用；候选批次数、起始ring、SOC、wait_cycles、离环/完成偏移、能耗/余量、资源duration，以及未选holding摘要。精确FEATURE_NAMES见本文附录，顺序与实现一致。

禁止未来draw、seed、load/SOC profile标签、method来源、aircraft ID数值和真实未来结局进入模型。输入/输出scaler只用train拟合，val/test仅应用；未来请求可由仿真器推进实际episode，但不进入观察或闭合标签生成。

监督标签是已到达闭合机群的episode累计J及相对HOLD的ΔJ，统一FCFS continuation，不是teacher action。J=`(-F,N,-E,-C,-T,-D,-energy)`：实际失败、正常完成、应急完成、已到达未终态（不含not_arrived）、观察暴露、正常延迟总和、实际消耗。前四计数精确，T/D round8、SOC round10作真实comparison key，tie按stage3 action_key最小；保留raw分量。两世界分母不混用，词典序胜出不等于Pareto改善。

## 两个固定学习模型

两类均为简单CPU MLP，hidden=(64,64)、ReLU、Adam、lr0.001、weight_decay0.0001、batch256、最多300epoch、torch seed4401、threads1。仅一个固定模型种子，两类各训练一次，不宣称训练稳定性，不因结果改参。

A为七维标准化ΔJ的MSE回归；反标准化后，预测前四个计数维用Python `round`取最近整数（恰为.5时ties-to-even）作词典序排序，其余三维保留预测raw值。B为同结构scalar评分，同context内所有真实非tie pair采用logistic rank loss，context等权，真正same-key ties跳过。禁止用动作分类交叉熵替代价值/排序目标。

checkpoint选择只看validation中“超过1根且存在不同真实key”的informative contexts：top3中包含任一真实best-key根的比例最大者优先；同分取validation loss更低者，再同分取最早epoch。test保持未见直至两checkpoint确定。

## 离线test评价

A报告各raw价值分量MAE。两模型均报告非tie pair accuracy（含固定action_key tie-break的最终诱导排序正确率）、top1 best-key命中、top3召回；学习筛选后精确选择相对full pool的真实J赢/平/输及raw regret。raw组件regret为full minus filtered逐分量差，不是保证非负的标量；同池子集精确选择不应优于全池真实key。beam_topk为同样top3+anchors预算的非学习对照。all、informative、degenerate分别报告，明示分母；单根快路径不能掩盖非退化排序或成本表现。

## 完整episode评价与计时

27个fresh test场景（九profiles×三test seeds）对六方法共162episode，物理观察窗240 s。报告N/E/F/C、以所有请求为分母的应急比例、全请求观察暴露、正常完成者条件平均延迟、全请求实际能耗、安全违反/未认证、最低着陆SOC；保留失败与截尾，不将条件均值变化直接称系统改善。

每方法独立生成，无跨方法缓存。计时覆盖generation、feature、inference、exact_rollout、final_validation_and_emergency、total；加载checkpoint另列，I/O排除在线链。候选生成中的全部feasibility工作保留在generation，不能声称整个决策仅检查top3。单根快路径与非退化链分别统计。

最多4个并行CPU进程，每进程torch threads1。真实计算耗时不推进离线交通时钟，本机并发墙钟受竞争影响，不能据服务结局宣称实时可用或固有复杂度。汇总整体/非退化成本，不自行追加计时重复矩阵。

## 执行与产物状态

现有venv Python3.12.4、torch2.5.1+cpu、NumPy2.4.4、SciPy1.18.1；CUDA不可用，sklearn缺失。只读环境记录在`tmp_stage4_preflight/runtime.json`及日志，未安装依赖。公开命令、schema和特征表已按当前实现列于[技术规范](../technical_spec.md)及下文；最终针对性测试与pilot已完成，记录如下；其通过不构成正式性能结果。

保留tiny预检、冻结设置、数据split、train-only scalers、epoch/val/checkpoint选择、offline统计、episode逐请求/事件和分项计时。本次只授权正式train/val数据生成，训练与test/eval仍须另行明确授权。结果不佳时不重调参数、不删数据、不换种子，结论由主线程依据事实决定；本协议不预设学习有效。

## 当前接口与已准备执行器

训练数据CLI为`python -m terminal_marl.scripts.generate_fixed_terminal_stage4 --split train --seed 4101 --load low --soc high --output NEW_DIRECTORY`，`--list-jobs --split train|val|test|pilot`只打印注册任务。test生成必须另给`--models MODEL_DIRECTORY`，检查已完成非smoke训练及value.pt/rank.pt/training.json/protocol.json哈希；train/val不读模型。

训练CLI为`python -m terminal_marl.scripts.train_fixed_terminal_stage4 --train TRAIN_ROOT --val VAL_ROOT --output NEW_MODEL_DIRECTORY`，正式固定300epoch；`--smoke --epochs N`仅用于pilot预检，不能用于正式test。

评价CLI为`python -m terminal_marl.scripts.evaluate_fixed_terminal_stage4 --offline TEST_DATA_ROOT --models MODEL_DIRECTORY --output NEW_OFFLINE.json`；完整episode用`--method METHOD --load LOAD --soc SOC --seed TEST_SEED --models MODEL_DIRECTORY --output NEW_EPISODE_DIRECTORY`。所有正式test方法都核对冻结模型记录，只有学习方法加载scorer。`--aggregate EPISODE_ROOT --output SUMMARY_PREFIX`输出JSON/CSV。以上是接口说明，不是已执行记录。

临时`tmp_stage4_analysis/dispatch_stage4.py`已准备trainval phase，默认只列任务；未使用`--execute`，未启动进程。固定顺序train4101/4102/4103再val4201，每seed按load low/medium/high、SOC high/medium/low填槽，共36任务、144快照。每条命令使用既有venv `-m terminal_marl.scripts.generate_fixed_terminal_stage4 --split SPLIT --seed SEED --load LOAD --soc SOC --output tmp_stage4_results/data/SPLIT/LOAD-SOC-SEED`。最多4并发，OMP/MKL/OPENBLAS线程各1；结果根须尚不存在。保存execution、progress、逐run命令/PID/起止/真实exit/stdout/stderr，首context+frozen_input+实际进程核对后记录startup_check。失败停止新任务调度、保留在运行任务与原始stderr，不自动重跑。该脚本不包含test或online phase；后续另授权。

## 附录：93个固定特征

|序号（从1计）|分组|数量|
|---|---|---:|
|1–2|当前时间/剩余时域|2|
|3–8|六状态计数|6|
|9–26|六状态×三环计数|18|
|27–42|holding/external的SOC与arrival age四统计|min/mean/std/max，共16|
|43–48|三环holding theta的sin/cos均值|6|
|49–52|scheduled完成剩余四统计|4|
|53–61|transfer/approach/pad预约条目、剩余duration和最早结束|9|
|62–65|候选计划数与三起始环计数|4|
|66–69|选中SOC四统计|4|
|70–73|总等待圈与分环等待圈|4|
|74–85|离环/完成偏移、路线能耗、储备余量的min/mean/max|12|
|86–88|三类候选资源预约duration|3|
|89–93|未选holding计数与SOC四统计|5|

空集统计为0，std为总体标准差。arrival age包括准入前等待；scheduled所在环按当前segment的ring，离环段为None；其他状态用记录ring。预约特征统计条目，包括重叠条目，duration求和不是占用区间并集。emergency_corridor归transfer，名称以_approach/_pad结尾分别归类。以下名称与顺序原样来自`learning_features.FEATURE_NAMES`：

```text
time
remaining_horizon
state_holding_count
state_external_count
state_scheduled_count
state_normal_landed_count
state_emergency_landed_count
state_failed_energy_count
state_holding_ring0_count
state_holding_ring1_count
state_holding_ring2_count
state_external_ring0_count
state_external_ring1_count
state_external_ring2_count
state_scheduled_ring0_count
state_scheduled_ring1_count
state_scheduled_ring2_count
state_normal_landed_ring0_count
state_normal_landed_ring1_count
state_normal_landed_ring2_count
state_emergency_landed_ring0_count
state_emergency_landed_ring1_count
state_emergency_landed_ring2_count
state_failed_energy_ring0_count
state_failed_energy_ring1_count
state_failed_energy_ring2_count
holding_soc_min
holding_soc_mean
holding_soc_std
holding_soc_max
holding_arrival_age_min
holding_arrival_age_mean
holding_arrival_age_std
holding_arrival_age_max
external_soc_min
external_soc_mean
external_soc_std
external_soc_max
external_arrival_age_min
external_arrival_age_mean
external_arrival_age_std
external_arrival_age_max
holding_ring0_theta_sin_mean
holding_ring0_theta_cos_mean
holding_ring1_theta_sin_mean
holding_ring1_theta_cos_mean
holding_ring2_theta_sin_mean
holding_ring2_theta_cos_mean
scheduled_completion_remaining_min
scheduled_completion_remaining_mean
scheduled_completion_remaining_std
scheduled_completion_remaining_max
reserved_transfer_entry_count
reserved_transfer_remaining_duration_sum
reserved_transfer_earliest_end_remaining
reserved_approach_entry_count
reserved_approach_remaining_duration_sum
reserved_approach_earliest_end_remaining
reserved_pad_entry_count
reserved_pad_remaining_duration_sum
reserved_pad_earliest_end_remaining
candidate_plan_count
candidate_start_ring0_count
candidate_start_ring1_count
candidate_start_ring2_count
candidate_selected_soc_min
candidate_selected_soc_mean
candidate_selected_soc_std
candidate_selected_soc_max
candidate_wait_cycles_sum
candidate_ring0_wait_cycles_sum
candidate_ring1_wait_cycles_sum
candidate_ring2_wait_cycles_sum
candidate_first_nonhold_remaining_min
candidate_first_nonhold_remaining_mean
candidate_first_nonhold_remaining_max
candidate_completion_remaining_min
candidate_completion_remaining_mean
candidate_completion_remaining_max
candidate_energy_min
candidate_energy_mean
candidate_energy_max
candidate_postroute_reserve_slack_min
candidate_postroute_reserve_slack_mean
candidate_postroute_reserve_slack_max
candidate_transfer_reservation_duration_sum
candidate_approach_reservation_duration_sum
candidate_pad_reservation_duration_sum
unselected_holding_count
unselected_holding_soc_min
unselected_holding_soc_mean
unselected_holding_soc_std
unselected_holding_soc_max
```

## 最终预检记录与本次授权

实现任务最终新增10项测试通过（3.28 s），阶段1–4合并67项通过（12.64 s），均exit0，覆盖最终生产字节；3个CLI help均exit0。原始日志为`tmp_stage4_preflight/targeted_tests.log`、`combined_tests.log`，完整记录为`validation.json`。

pilot4001 low/medium在0/5/15/90 s生成14/9/1/1根，共25候选、39个非tie pair、2个informative context。两模型仅2epoch且train/val复用同一pilot，smoke_only=true，test_read=false；这是流程检查，不是验证集性能或泛化证据，不能用于正式test。模型重载预测一致；offline与value/rank两个240s episode均完成，均11请求、N/E/F/C=10/0/0/1、执行spacing违反/未认证0。59个JSON和8个JSONL（176行）严格finite可解析。预检早期失败测试日志部分被覆盖，validation.json明确记录仅恢复工具输出摘录，不将摘录伪称完整原日志。

主审已接受最终代码和预检，并仅授权36任务train/val数据生成，等待源码提交后执行；当前尚未启动。正式训练、test数据和162episode仍须另行授权。预检不混入正式数据，不因其指标调整冻结参数。
