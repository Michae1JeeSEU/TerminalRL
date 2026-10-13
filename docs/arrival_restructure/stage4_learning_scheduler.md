# 阶段4方法：学习筛选与精确闭合参考

状态：**实现、预检、train/val数据及固定训练已完成；test数据已生成，完整episode评价运行中，结果待证据。** 参数、seed和执行顺序集中在[阶段4协议](../experiments/fixed_terminal_stage4_protocol.md)，原[学习问题定义](../fixed_terminal_learning_problem_definition.md)及[物理合同](../fixed_terminal_problem_definition.md)仍成立。

阶段4执行事实与后续结论见[结果报告](../evidence/fixed_terminal_stage4/results.md)，test未完成前不作最终性能判断。

## 研究对象

本阶段问：简单学习评分能否减少需要精确闭合rollout的根数，从而以更低完整在线成本接近有限预算传统参考。评分对象是完整可行联合窗口批次；单窗口价值只有条件在其他批次时才有明确语义，不建立固定单机urgency，也不让网络决定安全。

学习标签近似当前已到达机群、给定FCFS continuation至240 s的闭合J。阶段3发现闭合与实际未来排序反转，不能将标签称为expected/optimal Q或真实长期价值。训练不使用旧teacher、动作模仿或阶段3案例，offline RL不在本轮实现。

## 在线职责与流程

当前已到达状态 → 确定性候选生成与独立可行性 → 候选池 → beam序或学习分数筛选 → 小池精确闭合rollout → 按真实J选择 → 独立提交复查与共同应急 → 执行。

统一池使用扩展joint的8/48/8/128预算，前12unique批次+HOLD+现场FCFS，去重≤14，不读warm-start。full_reference精确评估全池；beam/value/rank取前3再加入FCFS/HOLD anchors，≤5根。原FCFS与原rolling分别保留为对照，rolling目标不同。各方法独立生成，不跨方法缓存。

若池只有一个根，使用共同快路径，不rollout/infer；最终独立检查与应急规则仍执行。正常批次只有整批可行才提交，旧完整承诺不撤销，未选holder保持。未知正常可行性不授权应急。每5s决策，物理dt1s不变。

候选生成本身已有全部相关feasibility工作，学习只减少后续精确rollout数量，不能把整个调度器说成仅验证3个根。有限池最优不等于全动态最优，也不新增CP-SAT oracle。

## 手工状态—候选表征

允许当前时间与剩余时域、已到达state/ring统计、holding/external的SOC与等待、ring theta聚合、已预约剩余完成/资源占用；候选批次数、起始环、SOC、等待圈、离环/完成偏移、能耗与储备余量、资源duration和未选holding摘要。93个FEATURE_NAMES及分组/数值处理已按实现列于协议附录，禁止字段不进入列表。

禁止未来draw、seed、load/SOC档标签、方法来源、aircraft ID数值或真实未来结果。scaler只拟合train；split按完整seed场景分组，同context所有候选保持同集合。

## 价值回归与排序

value模型为两层64单元ReLU MLP，输出七维标准化ΔJ，用MSE训练；反标准化后预测计数四维用Python round最近整数（.5 ties-to-even），按词典序排列，其他维保留raw。rank模型同hidden结构，输出一个分数，同context全部非tie pairs用logistic loss，context等权，跳过真实same-key ties。二者都不预测teacher动作分类。

模型分数只负责取top3；anchors保留后，最终决定仍由精确rollout的真实key产生。单根跳过推理；未知或不可行不变为人为奖励。validation以informative contexts的top3 best-key召回选checkpoint，同分val loss、再最早epoch；冻结训练设置见协议。只训练一个固定模型seed，不以单次结果论证训练稳定性。

## 比较与边界

offline在相同未见snapshot池比较分量误差、排序和top3保留能力、筛选后真实J regret，beam_topk控制相同精确预算；episode则在相同27个fresh需求场景报告六方法全请求结局、安全和成本。两者不能混为同分布状态：完整episode的策略会改变后续状态，joint behavior离线snapshot也不保证覆盖所有在线状态。

完整成本包含生成、特征、推理、精确rollout、最终复查/应急；模型加载单列，不把I/O或预计算控制优势混入求解比较。离线仿真不模拟计算延迟，四进程CPU计时只支持当前实现和运行条件。若学习未优于简单搜索，如实保留；若局部接近参考且更快，也不推导学习唯一必要或真实未来最优。

当前API/CLI、FEATURE_NAMES、checkpoint/scaler与数据schema已按实现核对，见[技术规范](../technical_spec.md)及协议附录；最终预检、正式数据与固定训练已完成，完整评价与性能结论待证据。非tie pair accuracy按带action_key tie-break的诱导全序计算；raw regret为full−filtered分量差，不保证逐维非负。

最终预检使用4001 low/medium四时刻25候选，2epoch两模型复用相同pilot train/val，仅核对序列化、重载和执行链。10项新增及67项合并测试通过，不能将pilot排序或episode结局当成性能结论；详细日志见协议。
