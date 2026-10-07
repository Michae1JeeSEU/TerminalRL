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

C_rule完整对照已由primary分析：19N5E/FDU0、技术/实际安全/hash/能量异常0，和旧rule正常/应急数相同，周期+24tick/能耗+.75%。额外同状态12步时刻细化虽消除约1.95s早到误差，仍被完整holder包络拒绝，未部署此诊断变体。环境具备当前有限场景继续采集的完整性条件，不等于业务最优或student准入；接受本入口73ef9660的独立smoke，成功核对后依原表逐臂采集，尚未开始真实学习。

独立smoke892807已SMOKE_OK/exit0，两tick/6raw=3actor+3forced，3/3 selected/OK label（各接口1个nominal RELEASE），3条V2.5上下文105+2×50。原raw六行teacher_label均null、forced无监督，pre-state/90D/标签join及首tick完整重放一致。实际安全/hash/能量异常0；预定2tick censor保留，不称完整环境资格。result SHA a9ccc8cd7ada2a6fe428274d83a36f87b26f8592687945063bfde9237fb3a00e，context SHA8666c9d2a0727c65a873b8d844e23431b81d17c1cbc1f2a29583a77620f2a938，13项关键文件两端相同，无残余进程。primary读取完整root结果及全部raw/context后，接受提交记录后只启动train_A；B/C仍依逐臂完整结果，未开始真实学习。

train_A已启动：Python2817000/timeout2816999，SSHsession13973，真实cwd=/home/user/projects/terminal、argv --index1及协议seed893301/12请求/warm6高SOC已核对；175tick/1634raw/23selected/73.185s及first_before/首tick重放/steps/progress均出现，只是启动前缀。运行中核心与重建模块冻结，60min monitor p3-v2-5已更新为本臂，尚无终态或真实学习。

### 正式P3范围复核与未执行步骤调整

primary重新核对`docs/execution_plan.md`的P3阶段及既有`p3_controlled_emergency_formal_v5.yaml`/`p3_adaptive_kl_v7.yaml`：正式P3必须保留priority 1/3/5、SOC .55–.90、mission/资源服务异质性、downstream_priority，以及flow6/dense24/flow12；旧formal每run240周期、三run、固定pattern flow6/dense24/flow12/dense24。历史准入与checkpoint不能继承到新V2.5，但这些任务要求不能因修物理链路被静默删掉。

当前integrated-repair builder明确`task_priority_enabled=False`、priority reward权重0、downstream_soc_priority，12/24 warm场景是基础阶段而不是完整正式P3。此轮此前记录也限定为基础输入，不把其完整性当全P3准入。**正在跑的train_A仍按已冻结原协议自然完成；原未启动train_B/train_C暂停，不按旧helper自动展开。** A终态后primary引入显式P3任务profile（不覆盖原基础profile），先小样本核对priority观测/仲裁/reward与真实自然入场，再以另行冻结的fresh train及Q/student周期验证。不能仅打开PPO而保持task-neutral环境。

这是基于正式任务要求纠正后续范围，未改当前运行参数、未删除原计划/输出或修改任何历史结果。A标签只能作为明确task-neutral桥接数据，不把它标成真实P3；新正式数据方案和seed在首次reset前另存，不消费原未用893302/893303或Q895301–3。


train_A已COLLECTION_OK/exit0自然完成（newly_run）：662ticks、12N/0E/F/D/U，3671raw=64actor+3607forced，64/64独立OK标签=34R/30W、6种proposal；环境172.032s、标注776.955s，无error/censor/实际安全/hash/负能量，最低完成SOC .573195。primary全量分析器逐行断言通过；三接口label12/35/17、32个retry恢复、8次同UID同接口WAIT→RELEASE。两个teacher R/实际W的原候选分别7/13，规则原请求均12，原请求/执行均保留，不按执行改标。32个实际放行rank1/2/3=29/1/2。无应急提交，最小实际/扫掠间隔88.864067/88.863011m。

result SHA a66e362c658bb2c23e29730f0fe041f4b912210e5b8a97bdc090cc6cd5a51588，context SHA723c4879f05b555bcf29d6bb32034488c002a5a50f4987fd08523742d5a26c07；终态关键文件最终两端匹配，初次同步滞后已收敛。Python2817000/timeout2816999及child均退出。完整分析`.tmp/p3_forecast_train_A_analysis.json` SHA dd4bf9dec3cc2df102ced3057f358975e10943bd607d91dc8865cc0a44d30532。仅接受本基础臂的完整性，不将其等同正式P3；原B/C保持未启动。下一步已另冻[真实P3任务方案](p3_v25_actual_task_20261008.md)，先验证任务profile与smoke，再真实flow6/dense24/flow12数据。
