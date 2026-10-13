# 阶段5前置诊断：阶段4排序信号与系统收益

本报告只读取阶段4保存的test标签、冻结模型预测和旧episode记录，未重新训练或运行rollout。原始分析输出为 `tmp_stage5_preflight/stage4_diagnostics.json`；[紧凑摘要](stage4_diagnostics_summary.json)保留统计、模型hash、特征别名、误差和旧online差异，完整历史结果见[阶段4报告](../fixed_terminal_stage4/results.md)。**这些是阶段4的development诊断，阶段5正式结果尚未生成。** 后续设计见[方法](../../arrival_restructure/stage5_learning_validation.md)和[冻结协议](../../experiments/fixed_terminal_stage5_protocol.md)。

## 覆盖与复现

共108个context、1012个候选，其中60个informative、30个单根；18个非单根context的真实key全部相同。冻结rank/value模型hash与阶段4记录一致，保存的top3结果逐context的chosen root、pair counts与raw regret精确复现。分析器7项针对性测试通过；它们是诊断代码检查，不是新实验效果证据。

## 排序命中与anchors

|方法|原始top3命中，全部108|原始top3命中，informative 60|加入anchors后，全部108|加入anchors后，informative 60|anchors补救context|
|---|---:|---:|---:|---:|---:|
|beam|96|48|103|55|7|
|rank|102|54|103|55|1|
|value|92|44|98|50|6|

这里仍采用阶段4的top3加anchors口径。rank非tie pair accuracy为0.830786，beam为0.636331；最终两者各103/108保留最佳key，且逐context所选闭合结局的N/E/F/C均与full一致。加入anchors后两者共同命中100个，共同miss2个，各有3个独有命中。因此，原始排序准确率较高没有在这批固定behavior snapshot改变N/E/F，也没有提高最终保优率。

rank独有的3个命中发生在high/low/4301/t90、medium/high/4301/t90和medium/medium/4301/t90，相对beam仅节省归一化SOC约0.002846018、0.008996478、0.008996478。beam独有的3个命中发生在medium/high/4302/t0、medium/medium/4301/t0和medium/medium/4302/t0，rank在每个context的暴露与正常延迟总和均多约104.719755 s。命中次数相同没有表达这些差异的大小；pair准确率也不按系统损失加权。

当前证据直接显示anchors补救了beam更多context，并且30个单根及18个同key context没有筛选难度。它不单独证明小候选池是beam表现的原因；阶段5预定的预算曲线及有限32根池面板用于补充该比较。

## value误差、表示别名与闭合目标

value加入anchors后98/108保优；相对full的raw J逐分量差额求和为 `[0,2,1,1,701.549669,580.940798,0.539414591]`，分量顺序沿用 `(-F,N,-E,-C,-T,-D,-energy)`。这是有符号向量诊断，不能当作通用非负界，也不能求和成未定义的标量regret。逐context前四个J分量与full一致的是106/108。

|value的原始ΔJ误差|候选数|F MAE|N MAE|E MAE|C MAE|T MAE（s）|D MAE（s）|energy MAE（SOC）|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|全部候选|1012|0.070799|0.583421|0.468949|0.328431|78.549318|68.488080|0.060622|
|HOLD|108|0.024153|0.419547|0.332717|0.185492|28.703530|36.155162|0.022703|

93维特征按精确数值相等分组：1012行共989个不同向量、19组重复，其中3组各含2个同context候选，特征相同而J不同。它们在high/high/4303/t0、high/low/4302/t0和high/medium/4302/t0，摘要保留root与标签。这说明现有表征存在压缩别名；不能将确定性FCFS标签称为随机标签噪声，也不能据此断定所有value误差均由别名造成。

标签描述既定FCFS continuation中的闭合结果，尚不等于真实未来价值。阶段3既有remove-one记录为13可行、20不可行，来自11个context；这里只读，没有新增反事实。它支持已观察条件下的批次关系，不支持全域不可加定理。阶段5保留整个联合批次评分及原93维表示，不在本诊断中更改模型或扩充任务。

## 旧在线失败与证据缺口

medium/low/4303旧episode中，rank N/E/F/C为17/0/1/7，beam与full均为17/1/0/7。可读取的decision记录显示rank相对两者首次不同的normal action发生在t50，零基decision index为10；但没有完整决策前state snapshot。固定joint behavior离线快照不等于这些自适应在线状态，因此不能把上述排序miss、特征别名或闭合误差直接归因为该额外能源失败。

阶段4完整结论保持：rank有可学习的离线排序信号，尚无优于同预算beam的综合证据，学习必要性仍未证明。阶段5按预先确定范围验证闭合搜索筛选与完整计算成本，正式训练及新test尚未开始；不在此报告预写新的效果或因果结论。
