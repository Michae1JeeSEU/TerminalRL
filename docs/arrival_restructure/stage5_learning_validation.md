# 阶段5：联合候选筛选与计算预算验证

阶段5按[用户计划](../fixed_terminal_stage5_learning_validation_plan.md)执行方向A：检验学习能否作为当前状态下的搜索筛选器，在有限闭合参考质量与完整计算成本之间提供收益。**当前处于实现与预检阶段，正式训练、新test和完整评价尚未开始。** 精确种子、模型超参数、预算、矩阵和运行接口以[冻结协议](../experiments/fixed_terminal_stage5_protocol.md)为准；[阶段4诊断](../evidence/fixed_terminal_stage5/diagnosis.md)记录本轮依据，阶段4原结论保留。

## 学习对象与职责

每个评分对象是当前已到达机群的一个完整联合候选批次，包含跨环等待、向内转移、最后进近和pad预约。评分依赖当前状态、已有承诺及批次共同占用，不能将它解释成独立单UAV优先级之和。方向B中的联合批次语义由现有候选表示承接，本轮不更换表示、不新增结构网络；方向C所需的未来需求模型尚未定义，本轮不训练长期Q或期望未来价值。

固定93维可见特征、几何、物理、能源和资源合同。学习只排序或预测批次值；候选生成、连续轨迹间隔、能量储备、资源可行性、最终复查与应急仍由独立确定性层负责。搜索预算耗尽仍为未知，不能解释为正常可行域为空。未到达请求、场景seed/profile、方法身份、真实结局不得进入部署输入。

标签为已到达机群在固定FCFS continuation下至240 s的累计 `J=(-F,N,-E,-C,-T,-D,-energy)`，value拟合相对HOLD的ΔJ。精确闭合比较保持既有词典序及取整口径；full_reference仅是统一有限池内的闭合参考。单一分量改善、pair accuracy提高或有限池最优都不等于系统整体改善。

## 公共候选池与总预算

标准池上限14。给定TOTAL B，先去重纳入HOLD和现场FCFS anchors，再按方法顺序补足，anchors计入B；候选不足时保留实际根数。该口径与阶段4的top3后额外加入anchors不同，不能直接把两阶段的预算列相等比较。

beam沿用确定性顺序；linear rank与两层MLP rank学习context等权的非tie pair排序；gradient boosting拟合context内词典序等级；value预测七维ΔJ。hybrid交错使用beam与MLP rank顺序，去重填满同一预算，不训练额外模型。选出的候选真实闭合rollout后按共同J决策，再做可行性复查和共同应急处理。

rolling20/60保留旧joint continuation与commitment surrogate，共用外层候选池和B5。其内部搜索、模拟步数及完整耗时单独记录，相同外层根数不能视为相同算力。

## 数据与验证关系

复用阶段4train4101/4102/4103及val4201，scaler只拟合train。固定model seed5401，linear rank、MLP rank、gradient boosting、value各训练一次；validation统一按informative best-key-retention@TOTAL B5选checkpoint，随后loss、最早epoch/tree破平。神经模型300epoch、boosting100树；不据test改变表示、参数或预算。pilot5001仅作小样本流程验证。

四模型冻结后首次生成新test5301/5302/5303。30个场景含27个标准load×SOC组合和3个medium/medium初相变体；旋转初相外的draw、SOC和配置完全配对。标准profile新seed与初相变体分别报告，不将同profile新随机样本称为外分布泛化。

评价分为互补证据：330个完整episode给出全请求结局和自适应轨迹成本；90个共同joint_window snapshot给出六种筛选器B3/5/8质量曲线；24个共同状态逐arm实际重新生成与rollout给出计时；9个32根池context只检查池规模敏感性。标签查表只用于离线质量，不能充当在线耗时。另9个旧4303/t15的realized-future重放只检查continuation敏感性，不作为训练标签或新test。

## 结果解释

正式报告须同时保留N/E/F/C、全请求暴露、正常条件延迟、能耗、安全检查及生成/推理/rollout/复查完整成本，逐场景配对比较。学习是否达到相同质量并减少计算，须由同状态实际计时和完整episode共同判断；当前没有阶段5效果结论。单一模型seed不证明训练稳定性，固定behavior的离线差异不能直接归因另一在线轨迹的失败。最终结论限定于冻结表示、池、预算和需求假设，不推广成普遍的学习必要性判断。
