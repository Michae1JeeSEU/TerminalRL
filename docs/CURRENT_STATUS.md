# 当前状态：阶段3机制分析与学习问题定义

更新日期：2026-10-13。用户已授权阶段3机制分析，当前完成阶段2决策inventory、协议/定义、冻结15context正式机制实验及结果核对；**阶段3已完成，学习必要性仍未证明，下一阶段待用户授权。** 不训练actor、网络、PPO或IL/teacher，不设计GNN/attention，不调奖励。

## 保留的阶段2事实

阶段2算法提交`bfa1a8c2ea4e4973972738504ad005756586fc21`，文档证据提交`52e15e06740aca8f0c41318c3b0f5234ce70fdba`。162场和12个有限目录参考已完成，47项针对性测试及2项元数据修正测试通过；这些是阶段2证据，不是阶段3测试结果。

每方法831请求，FCFS正常444/失败7，deadline445/7，joint429/9，rolling441/9。rolling相对joint正常+12，但相对FCFS正常−3、应急+25、失败+2、截尾−24，存在权衡；其632/1296次决策超过5 s。阶段2未建立学习必要性证据。完整事实和限制见[阶段2结果](evidence/fixed_terminal_stage2/results.md)，旧证据不改写。

用户[阶段3计划](fixed_terminal_stage3_learning_problem_definition_plan.md)开头关于joint/rolling有效和传统方法边界的表述按研究动机理解，不能视为已证明事实；当前检验机制，不预设学习必要。

## 阶段3已完成与冻结范围

- 只读162场、27情景、7776次决策，生成本地`tmp_stage3_analysis/stage2_decision_inventory.json`；N/E/F/C、正常与应急承诺、holding数及来源均已核对。这是独立策略rollout差异，不是单决策因果对照。
- [机制实验协议](experiments/fixed_terminal_stage3_protocol.md)固定15个依阶段2结果选择的context，t60使用共同FCFS prefix；不是heldout或全方法性能矩阵。
- 扩展参考cohort8/scan48/keep8/beam128，前12个唯一root加HOLD与六控制根，去重后最多19；同池FCFS continuation比较20/60/until240闭合估计，并在完整固定需求路径上作一次根干预的ex-post对照。
- 价值采用无权重词典序`(-F,N,-E,-C,-T,-D,-energy)`的实际累计量，保留共同prefix；closed-observed与ex-post分世界记录，不泄漏未到达信息。物理、安全、SOC、指标与基础周期不变。
- [学习问题定义](fixed_terminal_learning_problem_definition.md)已定义受约束窗口条件价值或联合批次评分并完成机制核对；不把数据库称为teacher，不许可训练。

首个low/medium/101、t0序列化smoke已完成（17root，11.480 s）；主审在未查看结果时提出数值比较、observed Δ与计时字段修正，旧产物保留为preliminary，不计正式矩阵。场景/预算先于首次预检确定，最终比较口径和接口在正式运行前固定。

## 完成情况与下一阶段边界

机制工具及公开API/CLI/schema已完成。最终阶段1/2/3合并57项测试通过（17.58 s、exit0）；最终low/medium/101 t0 smoke为17根、10.5729187 s、exit0，57个JSON加1个JSONL可解析，两世界根均可行。产物在`tmp_stage3_preflight/low-medium-101-t0-timing-final`，原始日志及`final_validation.json`同目录上层保留。主审已审核最终实现和预检，正式15context按冻结名单以最多4个并行子进程完成，全部真实exit0，共209个唯一root；实现版本`5a67b9ac6d8421ac31172a00ce0de851ed0a4c14`。原始记录与事实汇总分别在`tmp_stage3_results/contexts`和`tmp_stage3_results/summary`。15个FCFS根的ex-post逐请求记录与阶段2精确一致，45次选根及原始价值/差分已重算；执行spacing违反与未认证区间均为0，着陆最低SOC为0.0602558271。三个固定t60状态均无holding飞机，去重后只有一个空HOLD根，作为退化对照保留，不补换时刻，也不把15context当作独立泛化样本。主审结论已依据完整证据写入[阶段3结果](evidence/fixed_terminal_stage3/results.md)：三窗对FCFS实际词典序赢/平/输为0/12/3、5/7/3、6/8/1，延长窗不保证单调收益；33次局部remove-one为13可行、20undefined，不能默认价值可加。medium/low有局部收益，但low/low词典序win伴随更多截尾/暴露，high/low出现闭合与实际未来排序反转，不能盲目蒸馏参考。局部参考改善且慢或有限窗排序敏感均不能单独证明学习唯一必要。后续若获授权，可研究明示窗口、continuation与需求假设的联合或条件价值；当前不增补实验、不训练模型。阶段1/2固定物理合同及原始证据保持不变。
