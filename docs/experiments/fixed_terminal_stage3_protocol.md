# 固定终端区阶段3机制实验协议

状态：**冻结协议已执行完成：57项测试通过，15context全部exit0，共209唯一根。** 用户已授权第三阶段机制分析与学习问题定义，不授权训练或预设学习必要性。依据[用户计划](../fixed_terminal_stage3_learning_problem_definition_plan.md)及主审明确的边界执行；计划开头关于joint/rolling有效、传统方法长期不足的文字是待检验动机，不能替代[阶段2结果](../evidence/fixed_terminal_stage2/results.md)。

预算和场景在首次预检前确定；最终数值比较口径与输出接口在正式运行前固定。首个low/medium/101、t0序列化smoke在源码审阅期间完成（17root，11.480 s），随后主审在未查看结果时提出comparison key、observed Δ字段与计时命名修正。旧产物保留为preliminary smoke，修正后已在独立目录完成最终smoke；全部预检均不计入正式15context。

## 已知事实与本阶段问题

阶段2每方法831请求：FCFS正常444/失败7、deadline445/7、joint429/9、rolling441/9。rolling相对joint正常+12，但不支配FCFS；当前结果没有证明学习必要。现有162场决策inventory只描述各策略独立演化轨迹的相关差异，不能当作同状态单决策因果对照。

阶段3检验：在共同状态、共同候选池和共同continuation下，候选联合窗口的有限窗系统价值是否不同；延长评估窗是否改变排序，以及这种选择在固定真实未来重放中对应什么结局。所有结论限定在下列已选机制案例，不新增场景筛选、不扩大性能矩阵。

## 固定15个context

继承原阶段1/2物理、安全、容量、SOC分布、场景和指标；不重新抽样，不调难度。全部time=60状态来自FCFS执行共同prefix后的决策前状态。

| 负载 | SOC | 种子 | 根时刻/s |
|---|---|---:|---:|
| low | high | 101 | 0 |
| low | medium | 101 | 0 |
| low | low | 101 | 0 |
| medium | high | 101 | 0 |
| medium | medium | 101 | 0 |
| medium | low | 101 | 0 |
| high | high | 101 | 0 |
| high | medium | 101 | 0 |
| high | low | 101 | 0 |
| medium | low | 303 | 0 |
| high | high | 202 | 0 |
| high | high | 303 | 0 |
| low | medium | 101 | 60 |
| medium | medium | 101 | 60 |
| high | medium | 101 | 60 |

这些案例依阶段2结果选择，属于机制样本，**不是heldout验证集，也不是新的全方法性能矩阵**。结果不支持无偏泛化或独立测试集结论。

## 候选与根池

单机候选`a_i`是给某机指定`wait_cycles`的规范完整路线窗口，覆盖各环等待、跨环、最终进近和pad。联合根`A`是相容候选批次，既有承诺保持不变，未选飞机继续保持；所有批次经独立可行性层验证，不绕过应急资格。

| 项目 | 固定设置 |
|---|---|
| 扩展joint生成 | cohort=8、scan=48、keep=8、beam=128 |
| 扩展候选根 | 扩展joint排序前12个唯一root |
| 附加根 | HOLD、四规则、旧joint、旧rolling，去重保留所有来源labels |
| 总根池上限 | 19个唯一root |
| 公共周期与搜索 | 决策5 s、正常/应急与优先级分析256；物理dt仍1 s |
| 所有价值比较的continuation | FCFS |
| 闭合观测估计终点 | 根时刻+20 s、根时刻+60 s、绝对时刻240 s |
| remove-one | until240参考所选根中按飞机ID最前至多3个plan |

来源labels仅用于审计和去重，不是部署特征。选根与no-worse判断统一使用下述正式运行前冻结comparison key；相同key按`action_key`最小确定性tie。扩窗、扩候选、增beam仅建立预算受限离线参考，不标为全局最优。

t=0的六个控制根读取既有阶段2决策日志并独立验证，属于预计算warm-start；保留原`decision.seconds`，读取耗时不能替代原控制求解时间，也不能据此宣称新reference更快。t=60六根必须在共同FCFS状态重新求解，不能截取各方法自身rollout的t=60动作。

## 无权重价值与条件边际

以字典序最大化：

`J_H = (-F, N, -E, -C, -T, -D, -energy)`。

| 分量 | 实际观测口径 |
|---|---|
| F | 已发生的能源失败数 |
| N | 已实际正常完成数，不计尚在预约中的未来着陆 |
| E | 已实际应急完成数，不用应急激活数替代 |
| C | 已到达且尚未进入终态的请求数；不含`not_arrived` |
| T | 全部请求已观察暴露时长总和，保留失败和截尾 |
| D | 实际正常完成请求的延迟总和 |
| energy | 全部请求实际消耗的SOC总量，不是计划路线能耗或均值 |

每次输出全部原始分量与正常完成条件均值，不拟合权重奖励。这是指定continuation下的有限窗模拟量，不是真实optimal Q或期望Q。

数值比较口径在正式运行前冻结：J前四个计数分量按精确整数比较；T、D各round至8位小数，energy SOC round至10位组成selection comparison key。选根和no-worse判断均使用同一key，raw J与raw ΔJ全部保留。相同key选择`action_key`最小者；`action_key`是按`aircraft_id`及`wait_cycles`排序的元组。不能把约1e-12浮点残差解释成收益，正式运行后不因结果更改此口径。

`ΔJ(A;H)=J_H(A)−J_H(HOLD)`按分量记录，另对FCFS根给同样差分。单机仅定义条件边际：`ΔJ(a_i|B)=J(B∪{a_i})−J(B)`。两批均可行才定义；否则保存`undefined`与各自可行性状态。价值不假定可加，不能以不同策略全程结局差异代替这些对照。

原始J一律从episode起点累计，保留共同FCFS prefix；t=60分支共享的历史项在Δ中抵消，不另定义根后增量J。报告显式保存`root_time`、`evaluation_stop`、`episode_cumulative=true`。Δ比较必须来自同一context和同一评估世界。

## 闭合估计与真实固定未来重放

闭合估计使用`observed_clone`，只包含根时刻已到达请求（包括已终态历史），删除未到达请求的身份、SOC及到达时间。所有root从同一当前状态出发，分别施加一次normal batch及共同应急fallback，下一决策起统一FCFS。三个评估窗共用同一root池、同一continuation，分别记录实际J并按该J选根。旧rolling的目标和continuation不同，只能在上述受控同池比较中讨论时长影响，不能直接把旧整策略差异归于lookahead。

ex-post对照在完整原需求路径副本执行相同的一次root干预和共用fallback，下一决策起全部FCFS至240 s；原future只供离线环境执行，决策仍仅见已到达信息。保存每根的配对结局、events、指标与相对HOLD/FCFS的ΔJ。闭合观测估计与ex-post固定未来标签分开保存，各自记录请求分母与已到达数（value字段为`request_denominator/arrived_request_count`），不得跨两个世界直接作同分母性能比较。后者不得作为决策输入；本阶段不把数据库称作teacher数据或据此许可训练。

对until240参考根按ID取前至多3个plan作remove-one；完整批次与去除后批次都通过独立可行性才比较，否则明确undefined与状态。不得把删除一个plan必然可行当作假设：未选飞机的保持轨迹仍影响约束。

## 输出、计时与核对

每context记录来源场景、共同prefix/根前状态、物理合同、候选生成设置、唯一根及来源labels、可行性、全部J分量、分量差分、选根与tie、闭合估计和ex-post逐请求/events、remove-one结果。schema/API/CLI已按实现任务提供的接口记录于[技术规范](../technical_spec.md)，最终smoke与正式15context均已完成，事实核对保存于本地结果summary，研究结论见[阶段3结果](../evidence/fixed_terminal_stage3/results.md)。

机制描述须区分承诺/预约时刻与实际离环时刻：t=0承诺可以包含额外等待圈，离环按Plan首个非hold段的start确定。正式结果可比较所选root与FCFS根的新增/遗漏预约ID，以及共同ID离环/完成时刻差，再列后续外部等待、admission_rejected次数与原因；拒绝次数是累计尝试/ticks，不是唯一受阻飞机数。这些是配对重放描述，不据单个计数归因。

计时拆分为候选生成、控制根获取、closed-observed rollout、actual ex-post rollout与remove-one分析。每个J stop另存`elapsed_through_stop_seconds`，表示同条closed rollout至20/60/until240终点的累计耗时，不参与value比较。参考本身的成本为生成、控制获取及observed累计；ex-post和remove-one属于额外机制分析成本，分别报告。t0控制的原始求解时间单列。真实计算耗时不推进离线仿真时钟；并发条件另记，不能把计时当作实时部署指标或传统方法固有下限。

执行顺序为共同状态/未来隔离/批次/价值/序列化核对、主审代码与smoke、再运行15context。现已完成stage2日志inventory、最终修正smoke与57项合并测试（原始记录见技术规范），并按FORMAL_CONTEXTS顺序填最多4槽完成正式运行；所有命令/PID/日志/退出码保存于`tmp_stage3_results`。15context均exit0，共209唯一root，事实核对已完成；三个t60状态均holding=0，只有一个去重后的空HOLD根。保留这些预先指定的退化对照，不替换时刻、不按结果改预算或补跑成功单元。原阶段1/2证据保持不变。

## 预先限定的判断范围

若预算参考局部改善且慢，只支持研究候选快速求解的动机，不证明学习唯一必要。若短长窗排序相反且有配对结局收益，只支持这些案例的有限窗敏感性，不证明传统方法无法捕获长期影响或必须学习value。若强传统方法表现好，应如实承认。最终定义见[学习问题定义](../fixed_terminal_learning_problem_definition.md)，已依15context完成核对，学习必要性仍未证明；不训练actor/网络、PPO、IL/teacher，不设计attention/GNN，不调奖励，不自动进入后续训练。
