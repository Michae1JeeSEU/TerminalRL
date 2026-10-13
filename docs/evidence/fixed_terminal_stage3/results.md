# 阶段3机制实验与学习问题定义结果

阶段3定义与冻结机制实验已完成，**学习必要性仍未证明**。较强预算的传统参考在部分案例改善结局，也有负结果和权衡；它不是传统方法上界或动态全局最优。后续可以研究联合批次价值或给定其余批次的单窗口条件价值，但下一阶段模型、训练和新实验仍须用户授权。

实现版本：`5a67b9ac6d8421ac31172a00ce0de851ed0a4c14`。阅读顺序为[冻结物理合同](../../fixed_terminal_problem_definition.md) → [机制协议](../../experiments/fixed_terminal_stage3_protocol.md) → 本报告 → [学习问题定义](../../fixed_terminal_learning_problem_definition.md)。用户原计划保持原样，其开头强断言仅作动机假设。

## 运行与核对

固定15context全部真实exit0，共209唯一root；12个t0各16–18根，三个t60各仅1根。57项阶段1/2/3针对性测试通过（17.58 s、exit0）。全部15个FCFS根的实际重放逐请求记录与原阶段2精确一致，45次选根按冻结comparison key和tie核对正确，原始价值与差分可重算。全部已检查observed、realized和可行remove-one报告的执行间隔违反与未认证区间均0；着陆最低SOC为0.06025582712738723，超过储备0.06。此为合成运动学/能源模型的内部验证，不是硬件安全证明。

15个案例依阶段2结果回顾性选择，且t0/t60可共享场景，不是统计独立或heldout样本。三个t60均无holding：low有3正常已落地+5已预约，medium有4+13，high有6+17及6个external。已到达外部请求尚未准入，不能分配环内普通窗口。六控制与HOLD去重成同一空批次；这三项是真实退化对照，保留原样，不补换时刻。

## 同池窗口选择与实际结局

同一根池在仅已到达的闭合世界、共同FCFS continuation下分别选20/60/until240根；再在完整固定未来路径作一次根干预，后续统一FCFS。两世界各有分母，以下仅比较同一实际未来世界。J为episode累计`(-F,N,-E,-C,-T,-D,-energy)`；计数精确、T/D round8、SOC round10构成比较key。三窗选根不同10/15。

|实际词典序配对|赢|平|输|
|---|---:|---:|---:|
|20 vs FCFS|0|12|3|
|60 vs FCFS|5|7|3|
|until240 vs FCFS|6|8|1|
|until240 vs 20|8|7|0|
|until240 vs 60|3|11|1|

这些计数包括三个退化t60平局，不代表泛化胜率。**词典序win不是Pareto改善**；全部分量必须同时阅读。下表为实际240 s终点的正常/应急/失败/截尾（N/E/F/C）。六控制、HOLD及三参考的150行完整分量、正常条件均值见[summary.csv](summary.csv)，选根labels、原始J与配对Δ见[summary.json](summary.json)。

|context|FCFS|20|60|until240|
|---|---|---|---|---|
|low-high-101-t0|9/0/0/3|9/0/0/3|9/0/0/3|9/0/0/3|
|low-medium-101-t0|9/0/0/3|9/1/0/2|9/0/0/3|9/0/0/3|
|low-low-101-t0|9/1/0/2|8/2/0/2|9/0/0/3|9/0/0/3|
|medium-high-101-t0|18/0/0/6|18/0/0/6|18/0/0/6|18/0/0/6|
|medium-medium-101-t0|18/0/0/6|18/0/0/6|18/0/0/6|18/0/0/6|
|medium-low-101-t0|16/3/1/4|16/3/1/4|17/2/1/4|17/2/1/4|
|high-high-101-t0|20/0/0/36|20/0/0/36|18/0/0/38|20/0/0/36|
|high-medium-101-t0|20/0/0/36|20/0/0/36|20/0/0/36|20/0/0/36|
|high-low-101-t0|19/7/3/27|18/9/4/25|19/7/3/27|17/8/3/28|
|medium-low-303-t0|18/2/1/7|18/2/1/7|20/1/1/6|20/1/1/6|
|high-high-202-t0|20/0/0/33|20/0/0/33|18/0/0/35|20/0/0/33|
|high-high-303-t0|20/0/0/29|20/0/0/29|18/0/0/31|20/0/0/29|
|low-medium-101-t60|9/0/0/3|9/0/0/3|9/0/0/3|9/0/0/3|
|medium-medium-101-t60|18/0/0/6|18/0/0/6|18/0/0/6|18/0/0/6|
|high-medium-101-t60|20/0/0/36|20/0/0/36|20/0/0/36|20/0/0/36|

low/low/101的until240相对FCFS正常同为9，应急1→0，但截尾2→3，观察暴露增加203.3996939 s、实际能耗增加0.1587697551 SOC。固定J先偏好较少应急，因此判为win；不能把它写成所有服务指标改善。正常延迟均值只针对完成者，分母改变时也不能横向称整体提升。

medium/low/101正常16→17、应急3→2；medium/low/303正常18→20、应急2→1。选择根包含uav-0003离环/完成各延后一圈52.3598776 s、uav-0007各提前一圈：前者离环44.1132→96.4731 s、完成51.7134→104.0733 s；后者离环87.6504→35.2906 s、完成99.7228→47.3629 s。但完整批次的承诺数量和后续处理也改变，不能单独归因两机交换。303案例仍保留uav-0013在t0的合法应急。承诺时刻与实际离环时刻分开；共同ID的首非hold段/完成Δ、新增和遗漏预约、后续外部等待与准入拒绝原因均在summary.json。拒绝次数为累计尝试/ticks，不是唯一受阻飞机数。

high/high三个种子的60s根均比FCFS少2正常；until240恢复相同正常数，303还减少约157.08 s暴露与正常延迟总和。high/low/101却是until240比60s退化，因此延长窗口不保证单调收益。

## Priority与旧联合根的同状态对照

以下均为同一t0状态施加一次控制根、随后统一FCFS，区别于阶段2inventory的各方法完整独立rollout。

|根来源|medium/low/101 N/E/F/C|medium/low/303 N/E/F/C|
|---|---|---|
|FCFS|16/3/1/4|18/2/1/7|
|SOC|14/5/1/4|19/2/1/6|
|energy/deadline|16/3/1/4|18/2/1/7|
|旧joint/rolling|15/3/1/5|19/2/2/5|
|新until240参考|17/2/1/4|20/1/1/6|

Priority并非一概失败：SOC有收益也有退化，旧联合根也可能增加能源失败。结合不可行remove-one与完整路线批次差异，排序的系统效果依赖其他预约、保持轨迹和后续入场；这些结果不能唯一归因某一资源或某架飞机，也不先验认定joint/rolling有效。

## 闭合估计与实际未来的排序反转

high/low/101在已到达闭合世界的until240终点，SOC根为17N/4E/2F/0C，FCFS为16N/5E/2F/0C，前者按J较好。但施加同一原始实际未来后，SOC根为17N/8E/3F/28C，FCFS为19N/7E/3F/27C，排序反转。完整未来只用于ex-post执行，没有进入选择；这不是跨世界同分母相减，而是分别在两个世界内部比较两根。

这项负结果限制参考标签的用途，不能盲目蒸馏闭合参考。如未来研究期望未来价值，须另行定义未来需求模型和独立需求路径；当前实验没有完成该任务。两个已知额外失败ID在全部根下的请求记录和重要事件见[个案记录](previous_failure_case_outcomes.json)，不以不同整策略rollout替代单次干预归因。

## 条件价值与不可行的remove-one

对until240所选根中ID最前至多3个计划进行局部remove-one，共33次：13可行，20不可行。20项均因去除后批次的continuous_spacing检查不通过（uav-0001为9、uav-0002为9、uav-0000为2）。未选飞机回到保持轨迹，删除计划不保证剩余批次可行。13项可行边际中4项全部为0，其余分量保留于[条件边际](conditional_margins.json)。

这些是指定根和前3ID的局部样本，不能外推全部候选。批次可行性和价值不能默认可加；`ΔJ(a_i|B)`仅在B与B∪{a_i}均可行时有定义。undefined不得作为0或负奖励。可研究的输出是声明continuation、horizon与需求假设后的结果分量向量或批次比较分数，不是固定单机urgency、自由飞行动作、optimal/expected Q。

## 分项计算成本

正式运行使用16逻辑CPU的本地机器，最多4个并行子进程，总执行墙钟约130 s。下面范围覆盖12个t0案例，参考成本为共享根生成+控制获取+全部根至该终点的累计observed rollout耗时；三个终点共用轨迹，不能把三列再相加为独立运行成本。完整45行分项见[costs.csv](costs.csv)。

|观察终点|t0参考成本范围/s|
|---|---:|
|20 s|4.443–20.418|
|60 s|4.864–22.048|
|until240|6.208–27.389|

12个t0中，20s与60s参考各11/12项累计成本超过现有5s仿真决策周期，until240为12/12。根生成耗时4.123040–8.835548 s，占三窗参考总成本的比例分别为39.1%–94.4%、36.2%–89.8%、29.2%–76.4%；候选生成与多根rollout都占成本，不笼统归因某一模块。此比较受4并发与预计算控制条件限制，不代表固有复杂度或实时部署评测。

控制根在t0从已存阶段2日志读取并独立验证，是预计算warm-start。以下是12个t0案例中各方法原decision.seconds的范围，未将读取时间冒充求解时间，也未并入新参考成本；它们来自阶段2并发执行，不能据两次计时直接比较速度。

|控制方法|原求解时间范围/s|
|---|---:|
|fcfs|0.006229–0.075103|
|soc|0.015606–1.250477|
|energy_slack|0.018446–0.469722|
|deadline_slack|0.019536–0.421343|
|joint_window|1.044833–2.134769|
|rolling_horizon|2.668487–70.964226|

t60控制重新在共同状态求解，三个单HOLD根的until240参考成本约0.109/0.228/0.694 s。ex-post与remove-one属于机制分析开销，单列而不算参考求解成本。并发墙钟受资源竞争影响，离线env.time不随计算等待推进；这些结果不是实时可用证明或算法固有复杂度。它们可支持快速估值的研究动机，仍不能证明学习必要。

## 复现与产物

环境为既有Python 3.12.4；使用上述源码版本、原阶段2episodes和冻结FORMAL_CONTEXTS。单context命令如下，输出目录必须新建且不存在；完整15名单见协议，可用`--list-contexts`核对。

```text
python -m terminal_marl.scripts.analyze_fixed_terminal_stage3 --load low --soc medium --seed 101 --time 0 --stage2-root tmp_stage2_results/episodes --output NEW_DIRECTORY
python -m terminal_marl.scripts.analyze_fixed_terminal_stage3 --list-contexts
```

测试命令：

```text
python -m pytest tests/test_fixed_terminal_domain.py tests/test_fixed_terminal_safety.py tests/test_fixed_terminal_env.py tests/test_fixed_terminal_stage2_joint.py tests/test_fixed_terminal_stage2_baselines.py tests/test_fixed_terminal_stage2_oracle.py tests/test_fixed_terminal_stage2_runner.py tests/test_fixed_terminal_stage3_values.py tests/test_fixed_terminal_stage3_replay.py tests/test_fixed_terminal_stage3_reference.py -q
```

[完整正式产物](../../../tmp_stage3_results)含contexts、逐run命令/PID/起止/exit及stdout/stderr；每context七个入口文件及各root的root/observed/realized.json，具体schema见[技术规范](../../technical_spec.md)。[临时汇总](../../../tmp_stage3_results/summary)含209行完整decision database；[预检记录](../../../tmp_stage3_preflight)含三次分目录smoke和57测试原始日志，预检不计正式结果。紧凑summary含来源路径、全部选择/成本/配对分量；本目录保留五份JSON/CSV与本文，完整raw未复制进Git。

固定物理、安全、应急证书、外部队列、种子及运行预算均未改变，未补选案例或调参。阶段3止于定义和机制结论：学习必要性尚未证明；下一阶段待用户授权，不开展网络、训练、teacher或动态几何。
