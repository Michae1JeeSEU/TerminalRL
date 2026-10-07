# V2.5 新合流规划器初始训练数据（2026-10-08）

## 执行前范围

用户已授权持续修复至P3正式训练就绪并启动。本方案补足规划器变更后的监督语义：旧1046行及其229R/817W保留为旧模式证据，不与新数据混合。新训练数据仅用于后续fresh imitation，不自动授予I0/student/PPO资格。本文件写入时新forecast C_rule完整对照尚在运行；**该对照完整、无技术/实际安全/hash/F/D/U阻塞，经primary分析后才启动下面采集**。若需进一步修改物理核心，先更新此方案及配置，当前准备不意味着允许带已知阻塞训练。

使用已提交8078e3e的`committed_arrivals_v1`合流成员/相位预测、direct C²、原安全/能量/应急/K3契约。新接口`training/p3_v25_forecast_data.py`复用已经验证的all-actor rule rollout、私有snapshot teacher以及90D raw-label join，随后直接重建105D ego+全邻机50D。只在未被当前C_rule导入的文件准备；当前运行的物理核心保持冻结。

## 固定输入与预算

新根`/home/user/training_outputs/terminal/p3_v25_forecast_train_20261008`，每次单臂，primary完整检查后才推进下一臂：

|臂|seed|请求/warm|SOC|
|---|---:|---:|---|
|smoke|892807|3/3|高SOC，2tick|
|train_A|893301|12/6|高SOC|
|train_B|893302|24/12|高SOC|
|train_C|893303|24/12|.55–.90|

seed精确搜索terminal_marl/tests/docs未见此前使用记录；首次命令的Windows glob参数报错后已用真实目录完整搜索重做，不能用失败搜索证明唯一。首次reset前还需服务器输出/进程及源码SHA核对；任一碰撞先查真实身份，不更名绕过。开发smoke永不进入train，独立I1的未用Q895301–3与初始化91801–3保持未消费。

每完整环境1800ticks/3600s；全部actor pre-state按tick/数值UID顺序，最多2048query/51200 RELEASE意图、标注7200s（多LZ计划调用另计），V2.5重建900s。原raw label保持null，只有独立标注OK行成为监督；forced不入actor loss；WAIT无proposal loss。预算沿用已完成V3数值加速采集的边界，不按结果延时、续标或筛掉困难状态。smoke环境/标注各180s，只验证输入→label→重建→序列化。先定向测试，再smoke，再实际采集。

teacher仍是**当前pre-state单机首可行、最小干预请求**，保留完整25意图前缀/checked证书；不是联合批准、全局排序最优或未来不会再阻塞的保证。rule behavior、teacher请求、实际联合执行分别保存。原snapshot中的配置必须为新mode，loader要求指定三个train seed、完整COLLECTION_OK、partition=train和context SHA，拒绝旧planner/Q/development数据；不会把Q改成train以复用解码器。

## 分析与下一步

每臂检查N/E/F/D/U、完整到达/服务结束、技术/censor、安全/hash、能量及应急完成；核对全raw/actor/selected/唯一OK标签集合、90D原值/新上下文、mask/null/条件proposal及三接口/低SOC/busyLZ/高速inner/retry恢复覆盖。E单列，不任意新增“混合SOC必须零E”门槛；F/D/U、实际安全/hash或标注不全则保留失败并最小诊断，停止后续采集。

本初始集只有三个完整交通周期，数据完整性不等于泛化能力。全部通过后根据训练内残差与已有开发闭环制定fresh固定预算拟合与必要Safe DAgger，不再盲目使用旧1046模型的准确率；学习之前记录I0适用范围。原独立I1方案必须切换为此新规划器/新数据，再冻结fit/Q消费规则；既有I1门槛不随结果降低。若仍需补样，只能用明确train/development分区，失败Q保留而不进入train。最终仍需student-only完整周期、真实critic与有界PPO防退化验证，再冻结正式P3配置并启动；最后正式训练不设heartbeat。

## 执行记录

此处仅追加实际结果，以上准备尚无采集/学习结果。新重建公共实现保留`rebuild_q`的evaluation-only边界，train通过单独严格decoder；两个入口不被正在运行的规则对照导入。

新数据入口/重建/loader定向30项local/server用例全通过（35.74s/17.85s），四文件SHA相同。涵盖旧Q wrapper拒绝train、真实开发fixture的新mode快照→标签→上下文原样重建、train拒绝Q/旧mode/不完整/被改字节及空邻机/WAIT无proposal。这里的fixture不消费正式seed；尚未运行独立892807 smoke或真实train。primary已阅读代码diff及测试输出。
