# P3 nominal curve / total wait 修复后的 fresh 数据与训练方案

本文件承接 [中性课程诊断证据](p3_v25_neutral_course_20261008.md)。用户已授权持续修复、验证到新 P3 正式训练实际启动；最后正式训练不设 heartbeat。本文在新的数据首次 reset 前记录后续范围，执行仍取决于前一步实际结果。

## 修复后的数据边界

核心 `55743e9d6a567a4d39424f0f6020f0a9d3c4e77d` 保持 80 m 安全间隔：仅已知 nominal、零宽 holder 轨迹改用相对弦段和加速度误差界；真实预测交叉仍拒绝。`total_wait_v1` 使用 external+internal 等待。当前课程显式开启两项，历史默认保留；profile 为 `p3_v25_neutral_course_nominal_curve_total_wait_20261008_v2`。

任务等级继续关闭：有效 base=1、priority reward=0、policy mission=STANDARD；保留 SOC 紧迫度、总等待、raw mission 的物理服务时间。历史 `random_135` 的 RNG 抽样只保持种子流，不是当前有效随机任务等级。

原 893501–3 的 1003 标签和 fit 保留为旧实现证据，不能混入新轮。旧 dense fixed 的 424 WAIT 中 196 条、mixed 的 406 WAIT 中 119 条含至少一次 holder-tube 拒绝；这 315 条只是可能受影响，并非 315 个已证实错误。两个同状态查询才具体证明旧 WAIT 被过保守判据影响。重新采集采用新物理/调度合同下的完整访问分布，不能只改已知错误标签或保留容易样本。

## 顺序与预算

先完成 `p3_v25_nominal_curve_validation_20261008/source_dense_fixed`：沿用旧失败 env892854/policy929854 和固定旧 actor，以新合同做完整迁移诊断，不授予新 student 资格。只有技术、完整周期、实际安全/hash、F/D/U 检查无阻塞后，才执行下表。若仍失败，保留失败并依具体状态完成最小诊断。

新输出根：`/home/user/training_outputs/terminal/p3_v25_neutral_course_v2_20261008`。数据协议 `p3_v25_neutral_course_data_20261008_v2`。三个条件、服务分布、动作与观测架构均不扩展。

|阶段|seed|条件|目录|
|---|---|---|---|
|独立入口 smoke|892863|3 warm、mixed SOC、2 ticks|smoke|
|train 1|893511|6 请求/0 warm，SOC .55–.90，每 60s 入场|train_flow6_mixed|
|train 2|893512|24 请求/12 warm，fixed SOC .8|train_dense24_fixed|
|train 3|893513|24 请求/12 warm，SOC .55–.90|train_dense24_mixed|

每个正式数据周期分别为 flow 1210 或 dense 1800 ticks、环境 3600s、全量 actor 标注 21600s、观测重建 900s，外层 27000s。全部 eligible pre-state 按 tick/数值 UID 排序，最多 2048 query/51200 RELEASE 意图，多 LZ plan 次数另计。smoke 的环境/标注各 180s、外层 600s，永不并入 train。超过上限原样保留 INCOMPLETE，不临时补标、延时或抽新 seed 替换失败。

运行前核对源 SHA、进程/输出与 seed 元数据；代码只本地修改，经 Syncthing 同步。最多一个环境父进程与其一个顺序标注子进程。各臂终态后 primary 核对完整 raw/actor/selected/OK-label 集合、105+N×50、null/mask、原 teacher/behavior/实际执行分离、三接口、proposal、busy LZ/低 SOC/retry/WAIT→RELEASE 与成本，再决定下一臂。无新增“零应急”门槛；E 是完整应急服务，仍必须如实报告其能耗/等待代价，F/D/U 和实际安全/技术阻塞不得掩盖。

## 拟合、闭环与独立评价

三个 train 完整后，沿用已验证网络和 loss：91901–3 各 fresh 初始化、固定 5000 updates、300s/seed、batch128、Adam3e-4、clip .5、未加权 gate CE + 条件 RELEASE proposal CE。不选最好 seed/epoch，canonical 固定 91901，输出新的 `fit_initial`，不覆盖旧模型。首先检查训练内残差；模型/损失只在有具体残差证据时修改。

随后新模型执行独立 student smoke（892864/929864，3 warm mixed、2 ticks），以及三个 teacher-free 随机完整开发周期（flow6_mixed 892853/929853、dense24_fixed 892854/929854、dense24_mixed 892855/929855），输出新根下 `student_initial/<scenario>`。这些完整场景沿用原开发协议：flow/fixed 已消费，mixed 原先因前臂失败未运行，全部为 development，不能冒称 untouched。fixed还可与本轮旧actor迁移诊断作同合同对照。每臂最多完整flow1210/dense1800ticks，rollout/replay各3600s、外层7600s；保留全部请求、likelihood、mask、奖励与精确 tape replay。无实际安全/hash/F/D/U/技术/周期阻塞，actor takeover 沿用 <=.05。有分布外误决策才实施有界 Safe DAgger，不直接把运行失败归因 PPO。

Q895501–3 保持未消费，仍按同三个条件保留。候选与训练数据最终冻结后才收集独立 Q，一次评价全部三 seed final；沿用既有 imitation 判据，不用 Q 拟合、选择阈值或反复挑模型。state-only teacher 是当前候选集首可行，不是联合批准或全局最优；学习前和评价解释中都保留此边界。

只有新学生完整周期和冻结 imitation 证据通过，才执行已准备的同源 critic 初始化及三周期 PPO block（预留896501–3/936501–3，尚未消费）。PPO 后在相同开发场景比较 N/E、全请求等待/落地/服务周期、能量、reward、安全和真实参数更新；不能凭 no-op、训练内拟合或一个周期宣称训练退化已解决。正式 runner、预算和 seed 在实际有界 PPO 结果审查后冻结。最终确认正式训练 PID/cwd/argv/初始产物，再删除本任务 heartbeat 并结束，正式训练不设监控。

本方案冻结时尚未产生新轮采集/拟合/Q/PPO 数据，后续实际状态见执行记录。证据来源：既有失败及旧标签为 preexisting_artifact，同状态查询与测试为 newly_run；用户授权不是性能证据。

## 执行记录

实现提交`e44e76d7792ac2a6812080e3f80005da163639ec`，15项local/server定向测试通过29.97/15.37s，源SHA同步，无Git remote/未推送。独立892863 smoke已SMOKE_OK/exit0：2tick/6raw/3actor、三接口各1条teacher R12，实际均批准；null/forced与105+N50原标签join逐行检查通过，首tick重放一致。base全1、policy mission STANDARD，raw cargo/standard物理服务保留；无实际安全/hash/能量异常。环境2.136s/标注3.606s，result SHA`425eaf71c9834dae8e52a91e0e07832a0c62c1dd2555a4dd1bce161c3ac2b363`，关键产物两端同步。进程已自然退出，速度较快未捕获live PID/cwd，不能补称捕获过；显式cwd启动器和完整终态/序列化证据均保留。该smoke不重跑、不入train。

primary接受唯一下一臂train_flow6_mixed/893511。后续拟合/student助手已编译，只准备未执行；其student开发种子如上固定，入口只读新根，不覆盖旧结果。

train_flow6_mixed已COLLECTION_OK/exit0：702ticks、5N1E/0F/D/U，2050raw=30actor+2020forced、30/30 OK标签（23R/7W，4种proposal）。三接口actor15/6/9；busyLZ18、retry恢复12、SOC<.55为5/<.65为17，3次同接口WAIT→RELEASE，原始null/actor mask/全部label与105+N50 join检查通过。teacher23R中5次原规则执行WAIT，保留原pre-state候选标签与批量执行差异。17次正常K3批准及1条应急无rank trace分开记录，不按执行改标。

最小实际/扫掠81.145833/81.133264m，最低完成SOC .354109/剩余能量531.164；应急1次尝试/提交/触地，实际安全/hash/负能量/技术/censor均无异常。环境45.167s、标注93.044s；210个RELEASE意图、386次plan调用。不同于旧flow seed的6N，不能把本5N1E差异归因修复。root result SHA`2ba05115052825fb8d105415c0a2eb15f3914cc9eb02d041c79bbd9c79ef666c`、context SHA`b3ee0bf5dfcd438bdfddee9a12120ba5ea22fed47746a64bbd2de780d92c2717`，终态两端一致；首次同步缺3个尾部文件，等待10s后齐全，未重跑环境。timeout2858974/Python2858975均已退出。

primary接受该完整训练周期，下一唯一train_dense24_fixed/893512，既定24/warm12/fixed.8、1800ticks/3600s环境/21600s完整标注，所有上限不变。尚无新fit/student/Q/PPO；dense须终态分析后才继续mixed。由于可能持续数小时，保持60min监控。
