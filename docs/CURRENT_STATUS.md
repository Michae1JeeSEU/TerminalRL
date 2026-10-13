# 当前状态：阶段4已完成

更新日期：2026-10-13。[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)的实现、冻结数据与训练、108context离线评价及162个完整episode均已完成，主审已完成研究判断。**rank存在可学习的离线排序信号，但尚无优于同预算beam筛选的综合证据；学习必要性仍未证明。** 完整分母、表格与边界见[结果报告](evidence/fixed_terminal_stage4/results.md)，设计见[方法](arrival_restructure/stage4_learning_scheduler.md)和[协议](experiments/fixed_terminal_stage4_protocol.md)。

## 已执行事实

- 新增10项测试通过（3.28 s），阶段1–4合并67项通过（12.64 s），均exit0，生产代码未再改变。pilot4001 low/medium只作流程检查：25根，2epoch且同一pilot复用train/val，smoke_only，不能作性能结论。
- train/val共36数据任务全部exit0：train108context/1117根/64informative/22单根/3332非tie pairs；val36/384/25/2/1032。两模型各固定seed4401训练300epoch一次，wall30.0575322 s、exit0；value选17、rank选35，val informative召回均24/25，仅用于选checkpoint。
- 冻结后首次test生成27场景、108context/1012根/60informative/30单根/2813pairs，随后完成六方法162episode。189任务于UTC 06:52:40.609609至09:17:13.626789全部exit0；独立offline评价exit0、6.014 s。四模型文件hash不变，无实验重跑或补采样。
- 物理、安全、能源、SOC与负载设置保持不变。全部方法执行spacing违反与未认证均0，最低着陆SOC0.06067783709660538；每方法仍有8–9例能源失败，不能写成零失败或硬件安全认证。

实现版本`08235fb476f4bf483fecd71fdec5dfe9a8e1778e`，原运行工作树`c0f02fbcc5dd078c34c8211bd47b71bb98caebce`，各日志保留实际workspace_head。当前分支在主目录`D:\7_codespace\terminal`；raw保留在`.worktrees/arrival-restructure`的detached `d511438`目录。分支归属调整没有改变实验实现或模型字节。

## 研究结论与当前边界

informative离线rank top3命中54/60、beam48/60，但加共同anchors并精确评估后两者保优均55/60。在线每法795请求：rank N/E/F/C=439/16/9/331，full438/17/8/332，beam436/17/8/334；rank多1正常不能抵销对full多1失败。rank对beam实际J为9赢/11平/7输，尚未形成综合支配。

rank累计在线决策5659.16 s、beam3575.93 s，>5 s分别274/1296与208/1296；状态轨迹、并发竞争与退化比例不同，不解释为纯网络因果贡献。仿真时钟不计CPU等待，不能据服务结果宣称实时可用。标签仅近似已到达闭合FCFS continuation，非真实未来Q；单一模型seed也不证明训练稳定性。

当前工程参考优先保留full_reference与beam_topk，rank保留实验性可学习信号。阶段4已完成；后续研究需另定范围，不自动启动新实验、GNN/attention、PPO、offline RL、teacher动作模仿或动态几何。

## 保留的既有证据

[阶段3结果](evidence/fixed_terminal_stage3/results.md)：15context、209root、57项合并测试通过；15个FCFS逐请求重放一致，执行安全违反/未认证0，着陆最低SOC0.0602558271。三个t60为单HOLD退化对照。until240对FCFS实际词典序6赢/8平/1输；13可行/20不可行remove-one；闭合与实际未来存在排序反转。学习必要性仍未证明，旧数据只作动机，不用于阶段4训练。实现版本`5a67b9ac6d8421ac31172a00ce0de851ed0a4c14`，文档证据版本`3fb624b80b93ffc11ecbc8761d4309bbe7a89e74`。

[阶段2结果](evidence/fixed_terminal_stage2/results.md)保留162场/12有限oracle：FCFS444正常/7失败、deadline445/7、joint429/9、rolling441/9；rolling不支配FCFS。以上历史测试不作为阶段4通过证据。阶段1固定物理合同、SOC、负载和原阶段证据不变。
