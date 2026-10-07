# P3 V2.5轨迹上下文观测修复（2026-10-08）

## 范围与设计

用户“继续下一步”授权执行上一轮teacher/90D诊断后的观测修复。primary亲自修改核心/数据入口/docs并分析；worker仅执行明确测试、SSH和Git。沿用research-pragmatist与academic-experiments的最小有界验证：不训练、不重查teacher、不重跑环境，不消费新seed/audit/Q。

已定位的问题为target gap绝对角错误、TRANSFER阶段的即将入环邻机遗漏、仅用恒速CPA无法表达完整承诺曲线。修复使用显式版本`phaseA_low_obs_v2_5_trajectory_context_set`：

- `ego[105]`：90D基础的两个target槽改用相对方向模差，并纳入基准时刻已到达目标环的committed incoming；其他72个旧标量逐位保持。附加已有6个scheduler context与3个实际环的radius/height/speed共9值。
- `traffic[N,50]`：**全部当前活跃邻机**，排除自身、不按距离/CPA/top-K截断。包含phase8、当前ring3、committed1、承诺目标4、剩余时间1、相对位置/速度各3、承诺终点相对位置/速度各3，以及未来5/15/30/60/90/120/150s的相对位置21值。
- `traffic_mask[N]`与batch zero padding配套。UID仅作为诊断join信息，loader不把UID、tick、seed、label或hash放入model tensors；后续actor必须按集合编码，不能按UID位置flatten学习。
- 坐标位于ego当前径向/切向/竖向基底，距离除1000m、速度除25m/s；这些向量不clip，避免重演320m以外都等于1的丢失。remaining使用`t/(t+150)`，不把所有长承诺截成同值。

原90D builder/config/默认collector与checkpoint完全保留。新接口是`TerminalParallelEnv.trajectory_context_observations()`与`build_low_observation_v2_5`，返回结构化set；原flat student/PPO入口仍只接受其原schema。这是观测构建和数据准备阶段，新set-aware actor不在本轮创建。

目标槽仍采用历史18.5s等名义reference horizon，明确是reference，不是候选真实到达时间。incoming仅在其承诺end<=reference时进入对应target槽；更晚incoming也始终进入完整traffic集合。计划离开某环的飞机不再仅凭当前ring_role视为永久holder。未承诺未来放行不做推测。

轨迹采样复用当前SafetyFilter的canonical/有限响应预测助手，但创建私有缓存，**不调用filter_plan、candidate生成、teacher扫描或env.step**，也不把可行性位/证书/实际未来动作作为观测。所有数据源为当前状态、中心预约和已承诺计划，未来部署需要中心播报这些计划。7个采样时刻与终点仍是曲线摘要，不宣称覆盖任意细节或保证teacher完全可辨识。

## 运行前协议

只重建已消费train：893001（59 actor，50快照）、893101（613 actor，327快照）、893201（374 actor，268快照）；总1046行/645快照，原229 RELEASE/817 WAIT不变。原始raw/label/checkpoint不覆盖。

新输出`/home/user/training_outputs/terminal/p3_v3_observation_v25_20261008`。先`probe`读取893001的4个既定状态：tick0/UID4、92/UID0、101/UID0、152/UID6，180s上限。验证snapshot SHA/state/旧90D、new target gap/全邻机覆盖（UID4与UID1）、缓存/直接调用一致、无状态修改、数据序列化及decoder，再进行一次`full`重建1046行，900s上限。只恢复原pickle，不reset、不查询6个候选。若失败保留前缀，不自动覆盖/扩预算。

每快照仅在私有对象构建预测，邻机集合不截断，actor mask来自原始合格label；forced/unknown不补成WAIT。每行原teacher gate/proposal直接按decision_id从旧loader复制，old90D仍与原行逐位相同，新ego除两个target槽外的旧字段也逐位相同。full loader只能接收完整full结果，probe/ERROR不能成为未来训练输入。

服务器预检无Python任务，输出根未存在。三个annotation SHA两端相同；各entry/snapshot存在，2.16GB pickle通过Syncthing已可读（实际按每文件SHA验证）。一次环境CPU进程即可；预计分钟级，若明显超预期才建立用户要求的60min监控，终态删除。源码只本地经Syncthing，不手传或server Git。

## 验证记录

初版local/server四文件63项定向tests均通过（13.29s/1.84s），覆盖旧90D runtime/loader、新gap旋转不变、incoming边界、多committed保留、空集合padding、future采样及禁止候选/step调用。新增reconstruction脚本中的72个旧标量比较是数据输出断言，随后用4状态probe验证。此时尚未运行完整重建，其后续终态见下节；本轮没有训练效果结果，I0/student/PPO资格仍false。

4状态probe已`RECONSTRUCTION_COMPLETE/exit0`（newly_run），耗时0.248s，4行/32个邻机行、最多9邻机，原3R/1W不变。primary逐项检查原state/90D不变，缓存与直接构建一致，新增数值与原承诺匹配：

| 状态 | 新target槽 | 完整set中的关键承诺 | gap修正 |
|---|---|---|---|
| tick0/UID4 | UID5/5 | 原holder均保留 | ±0.925→±1 |
| tick92/UID0 | UID4/4 | UID4目标inner，剩余0.991704s | incoming不再遗漏 |
| tick101/UID0 | 空/空 | UID4仍在set，但目标landing、剩余60.553228s | 不误当inner holder |
| tick152/UID6 | UID1/2 | UID1目标inner，剩余4.884610s | successor −1→−0.598899 |

这直接验证了前轮漏掉的UID4/UID1现在进入输入，且能区分“即将进入inner”和“已承诺离开inner”。结果/observations/inputs两端SHA相同；相邻`probe.exit=0`及`probe.log`存在（worker初报在case内找日志而误称不存在，primary已从正确相邻路径核对）。小样本通过后接受执行唯一full1046行重建，仍不学习，不扩大种子/标签范围。

## 完整重建终态

full已自然结束`RECONSTRUCTION_COMPLETE/exit0`（newly_run），重建部分运行46.734s（不含末尾loader复读检查），无残余进程，未设置长期heartbeat。一次处理645个原快照，得到1046个actor输入、13939个邻机行、最多20邻机。229R/817W及所有decision_id原标签保持不变；未并入probe额外行。新loader完整读取并padding通过（ego1046×105，traffic1046×20×50，mask有效13939），`loader_verified=true`。

逐行断言通过：snapshot SHA/原pre_state hash、旧90D与原raw一致；构建前后完整物理/资源/telemetry state一致；新ego除2个target槽外的旧72值一致；traffic UID集合恰为active排除ego，无漏机或重复；labels来自原loader且decode不改gate/proposal；所有张量float32有限、空集合合法。新数据不写原raw或label，不创建模型/optimizer/checkpoint。完整输出中的4个probe rows与先前probe文件SHA精确相同，未因其他行顺序或缓存改变。

主要产物（local/server SHA一致）：

| 路径（输出根下） | 内容/大小 | SHA256 |
|---|---|---|
| `full/result.json` | 终态418bytes | `976577c6d7fba01387ca9ef37a9e5cf81f4ee764269441373cf2d61dd8b3504d` |
| `full/observations.jsonl` | 1046行，12429165bytes | `77163de4a6c24fa854b9f90005c91581b20da66dbb5c781822563c42a141b930` |
| `full/probe_rows.json` | 四状态48874bytes | `6965f67544f805142f8123dfe290d55484c33656f1b84bf7c85ebacad3beecad` |
| `full/inputs.json` | 原snapshot/annotation身份59798bytes | `1302a248b51c69cadcebe26707ee268c4fd2d196a9aaa85ee4f718f9370826a7` |

执行命令：在`/home/user/projects/terminal`用`/home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.reconstruct_p3_v25_context --root /home/user/training_outputs/terminal --output /home/user/training_outputs/terminal/p3_v3_observation_v25_20261008/full --mode full`，单CPU线程，源代码先Syncthing SHA核对。primary本地准备的`.tmp/run_p3_v25_reconstruction_20261008.py`只做固定启动、重复路径/进程检查和有界timeout；worker一次中间只读progress命令变量错误并未影响主任务，已纠正。已完成的full不要重跑。

## 结论与下一步

观测侧这两个具体缺陷已经实现修复并用原状态验证：target gap按相对语义计算；关键incoming及所有其他活跃邻机都进入新输入，承诺目标变化和剩余时间可见。新接口能表达此前90D直接丢掉的信息，但没有证明模型已经会利用它，也没有证明7个时刻的曲线摘要覆盖teacher全部判断。保留旧flat接口是明确兼容，不意味着旧checkpoint获得新能力。

下一项先实现小型**集合actor**及0更新前向/批处理验证，使用共享邻机编码与对顺序不敏感的汇聚，严格使用mask；旧90D checkpoint不直接塞入新接口。随后若开展有界拟合比较，优先采用**同一个新网络、同初始权重/相同1046行/相同固定更新预算**的旧信息与新信息两臂，而不是只把新结构与旧MLP比较后把所有收益归于观测。旧信息臂的新增输入置零并还原原两个target槽；新信息臂使用本轮重建。先冻结方案，再学习；本轮未实现actor或开始该比较。

两臂最多仍是已知train状态可拟合性证据；student-only闭环、独立qualification及PPO仍需后续证据。I0_QUALIFIED/STUDENT_QUALIFIED/PPO_ALLOWED维持false，P3训练退化尚未验证解决。研究docs、核心和判断均由primary完成；其他dirty保留，CURRENT_STATUS只更新本任务最新块。本轮只本地显式提交，无remote不推送。
