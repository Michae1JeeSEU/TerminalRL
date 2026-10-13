# 当前状态：阶段5正式test运行中，continuation比较补丁准备中

更新日期：2026-10-13。用户已授权[阶段5计划](fixed_terminal_stage5_learning_validation_plan.md)，当前落实[方法](arrival_restructure/stage5_learning_validation.md)与[冻结协议](experiments/fixed_terminal_stage5_protocol.md)。**阶段5正式训练与四模型冻结已完成；流程pilot已通过，正式test运行中，独立continuation检查失败，比较补丁准备中，阶段5尚未完成。当前没有阶段5效果结论。** 运行证据见[阶段5运行状态](evidence/fixed_terminal_stage5/run_status.md)。

## 阶段5当前进度

- 阶段4只读诊断已完成：108context、1012候选、60informative、30单根，冻结预测精确复现旧保存top3的chosen/pair counts/regret；诊断7项测试通过。[诊断简报](evidence/fixed_terminal_stage5/diagnosis.md)与[紧凑证据](evidence/fixed_terminal_stage5/stage4_diagnostics_summary.json)保留anchors、分量误差、特征别名及旧online差异。
- 实现源提交`4811abb805ab03997f5ce4186c6b2610f3c6b3b3`已推送，阶段1–5合并101项检查通过（17.21 s、exit0）。早期pilot5001四模型3epoch/trees只验证流程；正式四模型随后已按300epoch/100树完成唯一实际训练（22.092 s、maxRSS913020 KB、exit0），train108/val36。
- 模型已冻结，manifest确认completed/frozen、非smoke、未读test、重载预测一致。validation TOTAL B5选择linear155（24/25）、MLP161（25/25）、GB100（24/25）、value113（24/25）；它们只是validation选择指标，不是test效果，后续不重新训练。
- 首次训练launcher未设cwd，ModuleNotFoundError后0.036 s退出、未训练；修正后成功。四项服务器pilot首attempt因wrapper预建目录，在仿真前FileExistsError；原始失败日志保留。pilot_v2四项seed5001流程检查现已通过：beam/rolling60/full各48decisions、snapshot保存t0/15/90且t0/t15各有11arm真实计时面板，均exit0，四模型hash不变；耗时与内存见运行状态。
- 冻结方向为当前可见状态下完整联合批次的闭合FCFS参考筛选，保留93维表示与物理。复用train4101/4102/4103、val4201；model seed5401，validation TOTAL B5选checkpoint；冻结后首次生成test5301/5302/5303。
- 已定评价范围为330个完整episode、90个固定snapshot、24个真实同状态计时state、9个32根池context及9个旧4303/t15 continuation诊断，精确口径见协议。主对话已授权test360使用20workers、continuation9使用4workers，每进程threads1；24并发预计内存约17 GB，T0约90 min，已于北京时间20:05:07启动；test初验3/360完成、20running、337pending、0failed，有18份非空decisions.jsonl。独立continuation在15.93 s后exit1，3成功6失败，原始错误`ValueError: old visible snapshot mismatch`，主对话已核对九场旧4303：仅已存plan segment坐标有最大1.7763568394002505e-15的末位浮点差异，结构与非浮点内容、scenario及新提议root plans一致。

正式test继续运行。主对话诊断为跨平台浮点末位表示差异触发旧JSON严格相等，不归因具体库；Astra仅在`tmp_stage5_preflight/continuation_fix`准备artifact比较补丁和测试，待test360终态再应用，尚未修改运行源码或重启。test启动/dispatcher PID为3416299/3416301，continuation为3416300/3416302，四模型hash不变。T0约90 min；当前线程heartbeat id=5“阶段5冻结评价接续”已创建并view确认，20:13:57（北京时间）设置81 min，约21:35复查并接续已授权分析/交付，无常驻监控worker。源文件仅本地修改并经Syncthing同步；服务器输出进入`$HOME/training_outputs/terminal/stage5_validation_20261013`。训练PID3412631及实际启动/进度/失败日志位置见运行状态，全部原始记录保留。

新增汇总/打包工具独立验证：summary7项+package6项，共13 passed in 1.10 s，与原101项检查分开记录。真实pilot汇总CLI exit0，标记`fixture_or_pilot_incomplete`，41份源/副本文件hash相同；核对记录`tmp_stage5_preflight/summary_integration/validation.json`。没有新仿真、训练或读取正式530x结果。

## 阶段4已完成事实

[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)的实现、冻结数据与训练、108context离线评价及162个完整episode均已完成，主审已完成研究判断。**rank存在可学习的离线排序信号，但尚无优于同预算beam筛选的综合证据；学习必要性仍未证明。** 完整分母、表格与边界见[结果报告](evidence/fixed_terminal_stage4/results.md)，设计见[方法](arrival_restructure/stage4_learning_scheduler.md)和[协议](experiments/fixed_terminal_stage4_protocol.md)。

### 阶段4执行事实

- 新增10项测试通过（3.28 s），阶段1–4合并67项通过（12.64 s），均exit0，生产代码未再改变。pilot4001 low/medium只作流程检查：25根，2epoch且同一pilot复用train/val，smoke_only，不能作性能结论。
- train/val共36数据任务全部exit0：train108context/1117根/64informative/22单根/3332非tie pairs；val36/384/25/2/1032。两模型各固定seed4401训练300epoch一次，wall30.0575322 s、exit0；value选17、rank选35，val informative召回均24/25，仅用于选checkpoint。
- 冻结后首次test生成27场景、108context/1012根/60informative/30单根/2813pairs，随后完成六方法162episode。189任务于UTC 06:52:40.609609至09:17:13.626789全部exit0；独立offline评价exit0、6.014 s。四模型文件hash不变，无实验重跑或补采样。
- 物理、安全、能源、SOC与负载设置保持不变。全部方法执行spacing违反与未认证均0，最低着陆SOC0.06067783709660538；每方法仍有8–9例能源失败，不能写成零失败或硬件安全认证。

实现版本`08235fb476f4bf483fecd71fdec5dfe9a8e1778e`，原运行工作树`c0f02fbcc5dd078c34c8211bd47b71bb98caebce`，各日志保留实际workspace_head。当前分支在主目录`D:\7_codespace\terminal`；raw保留在`.worktrees/arrival-restructure`的detached `d511438`目录。分支归属调整没有改变实验实现或模型字节。

### 阶段4研究结论与保留边界

informative离线rank top3命中54/60、beam48/60，但加共同anchors并精确评估后两者保优均55/60。在线每法795请求：rank N/E/F/C=439/16/9/331，full438/17/8/332，beam436/17/8/334；rank多1正常不能抵销对full多1失败。rank对beam实际J为9赢/11平/7输，尚未形成综合支配。

rank累计在线决策5659.16 s、beam3575.93 s，>5 s分别274/1296与208/1296；状态轨迹、并发竞争与退化比例不同，不解释为纯网络因果贡献。仿真时钟不计CPU等待，不能据服务结果宣称实时可用。标签仅近似已到达闭合FCFS continuation，非真实未来Q；单一模型seed也不证明训练稳定性。

当前工程参考优先保留full_reference与beam_topk，rank保留实验性可学习信号。阶段4已完成；阶段5按本页列明的已授权范围推进，不扩展GNN/attention、PPO、offline RL、teacher动作模仿或动态几何。

## 保留的既有证据

[阶段3结果](evidence/fixed_terminal_stage3/results.md)：15context、209root、57项合并测试通过；15个FCFS逐请求重放一致，执行安全违反/未认证0，着陆最低SOC0.0602558271。三个t60为单HOLD退化对照。until240对FCFS实际词典序6赢/8平/1输；13可行/20不可行remove-one；闭合与实际未来存在排序反转。学习必要性仍未证明，旧数据只作动机，不用于阶段4训练。实现版本`5a67b9ac6d8421ac31172a00ce0de851ed0a4c14`，文档证据版本`3fb624b80b93ffc11ecbc8761d4309bbe7a89e74`。

[阶段2结果](evidence/fixed_terminal_stage2/results.md)保留162场/12有限oracle：FCFS444正常/7失败、deadline445/7、joint429/9、rolling441/9；rolling不支配FCFS。以上历史测试不作为阶段4通过证据。阶段1固定物理合同、SOC、负载和原阶段证据不变。
