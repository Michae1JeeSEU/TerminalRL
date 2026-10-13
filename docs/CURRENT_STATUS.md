# 当前状态：阶段4学习辅助调度

更新日期：2026-10-13。用户已授权[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)，主线程已确定[方法](arrival_restructure/stage4_learning_scheduler.md)与[协议](experiments/fixed_terminal_stage4_protocol.md)。**实现与最终预检完成；正式train/val数据生成获主审授权但尚未启动，正式训练与test未授权。** 目标为学习筛选能否以较低完整在线成本逼近有限预算闭合参考，不预设学习有效或必要。

## 已完成准备

用户计划已从父checkout字节不变复制，5016 bytes，SHA256 `c43b11c6aab48e025d24988a4b9f7b1ed98aefef3f622a44b076fb54326ef584`。现有venv为Python3.12.4、torch2.5.1+cpu、NumPy2.4.4、SciPy1.18.1；CPU tensor检查成功，CUDA不可用，sklearn缺失，未安装依赖。完整只读记录为`tmp_stage4_preflight/runtime.json`和`runtime.log`。

设计固定：新池扩展joint8/48/8/128、前12unique+HOLD+现场FCFS，≤14；full_reference全池闭合rollout，beam/value/rank前3+anchors最多5精确根，另比较原FCFS与原rolling。单根共用快路径，独立安全/能源/资源/应急职责不变，无warm-start或跨方法缓存。

train seeds4101–4103、val4201、test4301–4303，pilot4001。每seed九profiles，joint behavior在0/5/15/90 s取共同prefix快照；train108/val36/test108。train-only scaler，两类64/64 CPU MLP各一次固定seed训练；checkpoint固定后才首次生成test。正式episode为27新场景×六方法=162，最多4进程、threads1。所有值是闭合continuation下的有限窗监督标签，不是teacher动作或expected Q。

## 当前下一步

实现、93维特征、模型/筛选与数据/评价接口已完成；最终阶段4新增10项测试通过（3.28 s），阶段1–4合并67项通过（12.64 s），均exit0并覆盖最终生产字节。主审已接受代码与预检。正式trainval调度器准备36任务/144快照，只生成train/val；等待源码提交消息后启动，当前尚未启动。训练、test数据和正式episode未获本步授权。

pilot仅为4001 low/medium流程验证：四时刻根数14/9/1/1、25候选；两模型只训练2epoch，train/val复用同一pilot且标smoke_only，不能给性能或泛化结论。offline和两条240s学习episode完成，模型重载预测相等，59 JSON+8 JSONL（176行）严格finite可解析。原始记录为`tmp_stage4_preflight/validation.json`、`combined_tests.log`等，预检数据不并入正式集。

## 保留的既有证据

[阶段3结果](evidence/fixed_terminal_stage3/results.md)：15context、209root、57项合并测试通过；15个FCFS逐请求重放一致，执行安全违反/未认证0，着陆最低SOC0.0602558271。三个t60为单HOLD退化对照。until240对FCFS实际词典序6赢/8平/1输；13可行/20不可行remove-one；闭合与实际未来存在排序反转。学习必要性仍未证明，旧数据只作动机，不用于阶段4训练。实现版本`5a67b9ac6d8421ac31172a00ce0de851ed0a4c14`，文档证据版本`3fb624b80b93ffc11ecbc8761d4309bbe7a89e74`。

[阶段2结果](evidence/fixed_terminal_stage2/results.md)保留162场/12有限oracle：FCFS444正常/7失败、deadline445/7、joint429/9、rolling441/9；rolling不支配FCFS。以上历史测试不作为阶段4通过证据。阶段1固定物理合同、SOC、负载和原阶段证据不变。

本轮只实现监督价值和排序辅助筛选，不实施actor/PPO、teacher动作模仿、offline RL、GNN/attention、动态几何或自由飞行。实际未来只由评价环境推进，不能作输入。科学结论等待本轮结果与主审。
