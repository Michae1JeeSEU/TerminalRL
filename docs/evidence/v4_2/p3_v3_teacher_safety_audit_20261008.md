# P3 teacher放行与90D上下文诊断（2026-10-08）

用户要求先核对teacher决策合理性，再执行既定阻塞邻机/预测轨迹/90D检查。本轮primary本人负责研究文档、代码、诊断与判断；worker仅运行明确命令。没有学习、环境reset/step、新采集、audit/Q或checkpoint写入。

## 运行前范围

1. 对已完成893001/893101/893201三个train回合的1046个标签做只读结构复核：90D/actor/label join、原25 RELEASE后WAIT候选顺序、首个可行选择、被拒绝前缀、RELEASE对应checked hash。原始标签不改；此项不是重新证明1046个轨迹的物理安全。
2. 只使用893001原快照的4状态/6固定候选：tick92/UID0 proposal12；tick101/UID0 proposal12；tick0/UID4 proposal12及teacher14；tick152/UID6 proposal12及teacher13。每候选最多180s，六候选合计600s；首次候选同时作为小样本输出验证，后续不重复。输入快照/标签/entry SHA两端先核对，6项都来自已消费train。
3. 原生`_candidate_plan`每候选只调用一次，不绕过任何检查。诊断包装原SafetyFilter的距离函数，保留其返回值、观察实际阻塞UID/检查阶段/点间隔及保守下界；caller结构变化即报错。再用1/4步长计算同一预测模型的欧氏距离以区分真实预测冲突和裕量拒绝。更密网格仍共享原轨迹与有限响应模型，不伪称独立物理安全证明。
4. 对应输出邻机槽UID、当前几何状态、 committed计划终点/剩余时间、90D各特征和裁剪前数值，核对目标环`path_gap`与相对间隔。保留当前90D wire语义，不在诊断中修正输入、训练模型或改标签。

新输出根：`/home/user/training_outputs/terminal/p3_v3_teacher_safety_audit_20261008`。服务器无项目Python，路径未存在；4个pickle及对应标签/entry/annotation已确认local/server SHA一致。运行代码只本地修改后Syncthing；不上传源码、不server Git。先定向tests与标签检查，再首候选，再其余5项，所有输出独立新文件、失败保留。

## 源码审查得到的待验证事实

teacher优先最小绝对speed residual，再最小绝对timing residual，最后按index打破并列；其目标是首个单机可行RELEASE，否则WAIT。它未比较吞吐/延误/能量最优性，未联合优化同tick多机申请，因此不能作为全局最优或联合批准标签。它对原pre-state检查资源/完整轨迹/能量/安全，再由实际执行层在批次改变后的状态重检；两种语义须保持分开。

90D直接复用87D邻机特征及retry、service上下文。`nearest_predicted_conflict`是当前恒速线性CPA；它未使用安全层的完整已承诺曲线。5个槽仅编码9个标量/槽，不含邻机计划终点、剩余时间或曲线几何。目标环selector与cache均把选中的预测绝对角×半径写成`path_gap`，而非相对合流点间隔；这是待量化的语义问题，不因历史兼容注释而视为正确。不能据此先行宣称唯一退化原因。

## 判定边界

本轮分别回答：记录是否正确；选中轨迹在既定模型下是否获证/有否明确错误放行；排序是否是有业务理由的局部规则；关键阻塞信息是否进入90D。四状态不能证明teacher所有场景最优，不能凭同一个checker重复通过自证安全，不能凭拒绝reason断言物理无解。I0/STUDENT_QUALIFIED/PPO_ALLOWED仍false。

## 准备与首样本修正

local/server原20项tests通过（本地PATH无python，使用已有`venv/Scripts/python.exe`，没有安装依赖）。1046行只读审计已exit0；229R/817W均首可行标签，证书/join无异常。817次behavior gate不同对应规则原请求RELEASE而单机teacher WAIT，不是执行安全违规数。

首候选0在1.569s后exit1：新增诊断把`state_record`中的telemetry也要求不变，但原`_candidate_plan`合法增加repair/energy query计数。失败log/exit/cost保留，不能当teacher拒绝异常。primary修正为物理/资源/能量/已提交plan等完整字段严格相同，只允许4个已知候选查询计数非负增加，并单独报告delta；新增回归同时证明位置改变或collision计数改变仍失败。修正后在新`witness_v2`子目录重新执行首候选一次，剩余5项不增加；明确总候选调用最多7次（6个unique加1次诊断失败），失败耗时仍计入600s总预算。无生产核心、安全条件或原标签变化。

## 终态证据与teacher判断

证据类型：1046个原label及快照是`preexisting_artifact`；本轮结构审计和6项候选见证是`newly_run`。21项local/server定向tests均通过（2.35s/0.77s）；随后新增一项relative-gap报告回归，所在12项tests单独重跑通过（2.14s/0.70s），共22个不同用例验证，不把两次计数相加。六个成功候选全exit0，合计子进程墙钟11.171s，加首失败1.569s共12.740s，低于600s。每项恰1次候选查询；0teacher完整扫描/0reset/0step/0训练。没有残余进程，无需长任务heartbeat。

全部1046行通过loader的原obs/hash/mask/action join和本轮候选前缀检查；229 RELEASE有相同checked/trajectory hash，817 WAIT均在25个候选拒绝后回落。106个RELEASE首个nominal即通过，另外123个使用后续候选；并不是所有状态都无条件教proposal12。拒绝原因计数以**候选**为分母：final_capacity9375、predicted_separation7729、swept2388、holder tube1180等，不能当独立状态数或事故数。

六项原生reason均与原label前缀相同；3条teacher选中RELEASE的轨迹hash与原证书精确一致。所有候选的轨迹完整性和动力学检查均通过，失败点在间隔。名义holder契约为`nominal_until_release`，safe_distance80m，physics_dt0.2s、safety_sampling_dt1s；更密检查dt0.05s。表中“下界”包含原安全检查裕量，“密采样”是该预测模型的直接欧氏距离；拒绝后原filter短路，拒绝行不是所有邻机/所有后续阶段的全局最小值。

| tick/UID | proposal（speed,timing残差） | 原/本轮判定 | 关键邻机 | 原检查最小下界m | 密采样距离m | 未来时刻/相对当前s |
|---|---|---|---|---:|---:|---|
| 92/0 | 12（0,0） | predicted_separation | UID4 | 29.284 | 29.624 | 166.600 / +74.600 |
| 101/0 | 12（0,0） | certified | 最近UID6 | 105.288 | 105.756 | 143.450 / +42.450 |
| 0/4 | 12（0,0） | holder_reachable_tube | UID5 | 28.429 | 33.516 | 100.350 / +100.350 |
| 0/4 | 14（0,+1） | certified | 最近UID5 | 188.461 | 193.548 | 61.950 / +61.950 |
| 152/6 | 12（0,0） | predicted_separation | UID1 | 50.069 | 50.456 | 219.200 / +67.200 |
| 152/6 | 13（0,+0.5） | certified | 最近UID0 | 127.083 | 127.534 | 152.000 / +0 |

**本次抽查支持teacher的单机安全判定，没有发现“错误放行已被教成正确”的反例。** 三个被拒候选即使去掉保守距离修正，直接预测间隔仍明显小于80m；tick0的holder tube角宽为0，不能归因于假设任意残差导致包络过宽。teacher14/13改变合流时间与合流角后获证，有具体物理依据，不应为了让网络更容易拟合而改成12、降低80m或移除holder检查。

这仍是现有预测/执行契约下的安全判断：tick92 UID4的原transfer在92.992s结束，然后预测为内环nominal holder；到tick101它已进入FINAL，并按已承诺final轨迹预测。未来可能发生但**尚未承诺**的离环不能预先当作安全通道空闲。该保守约定合理，但不是“真实世界无论未来如何调度都必冲突”。本轮没有模拟所有未来多机决策，也未复查1046条轨迹全部安全；不能超出四状态结论。

teacher的质量边界必须保留：其排序为先可行RELEASE，再最小|speed|、|timing|、index，属于局部最小干预启发式，没有证明吞吐/能耗/延误最优。pre-state单机标签也不同于执行时先更新服务、应急及更早机批准后的联合结果。它适合作为有限安全初始化目标，不能当全局最优调度或直接把原label accuracy当闭环性能。先前文档已记录这一限制，本轮没有通过实际执行覆盖原标签。

## 已定位的90D问题

### 1. 已承诺进入目标环的关键飞机被邻机选择遗漏

- tick92/UID0四个环槽全部为空，唯一CPA槽是UID6；真正阻塞者UID4当时为TRANSFER_M2I、距ego564.378m，约0.992s后进入inner，却不在任何槽。selector只认`ring_role is target`，transfer阶段`ring_role=None`，所以即将进入的飞机消失在目标环槽里。
- tick152/UID6的两个目标环槽都是UID2，CPA槽是UID0；阻塞者UID1为TRANSFER_M2I、距ego580.341m、约4.885s后进入inner，同样没有任何槽。

这是两条具体错误预测中**决定候选可行性的对象没有进入actor邻机输入**的直接证据。aggregate occupancy/headway不能恢复该对象的轨迹终点、剩余时间和相位。不能因此断言90D在这1046条数据中存在完全相同输入却标签相反（上一轮已确认没有精确重复），也不能断言所有训练错误都由此导致；但观测表达与teacher所用信息确实不对齐。

### 2. `nearest_predicted_conflict`的名称与完整轨迹风险不同

该槽采用当前速度的直线CPA（30s），而安全判定使用canonical曲线、到达环后的有限响应，以及完整candidate约53–143s和post-plan horizon。

在UID0 tick92→101的对照中，CPA槽d_cpa从200.100m（UID6）变为71.272m（UID10）。单看这个特征，可放行的tick101反而显得更危险；实际完整轨迹检查是tick92距UID4只有29.624m，tick101所有已检查对的密采样最小值105.756m。两个state的其他字段也改变，不能只据这一个量给网络做唯一归因，但它明确不能代替完整trajectory context。

90D每邻机仅present/gap/distance/relative_speed/closing/d_cpa/t_cpa/priority/committed标记，没有plan目标环/终点、剩余时间、未来曲线。一个committed bit不足以告诉模型UID4在tick92还将占据inner，而tick101已经承诺离环。

### 3. target `path_gap`是确定的计算语义错误

selector及cache都把“选中邻机的预测绝对角×目标半径”写成gap，未减去ego预期相位、也未按前/后方向计算模差；不是cache不一致。具体：

- tick0/UID4→UID5保存±296m，归一化±0.925；按**相同18.5s基准预测**的相对间隔应为±628.319m（归一化±1）。
- tick152/UID6的successor保存−1249.108m并截到−1；基准相对后方间隔实际−191.648m，应为−0.598899。

这里修正的只是同一baseline中的相对语义，18.5s本身也远短于实际53–143s候选，不把修正gap称为完整真实合流间隔。绝对角错误与320m裁剪叠加丢失信息，单改网络容量不能恢复。原字段已被旧checkpoint使用，必须通过新schema修复，不能静默改变90D旧字段含义。

## 下一项有界修复方案

1. 先实现显式新观测版本：相对gap用方向模差；目标环邻机纳入已承诺incoming，并区分当前holder、transfer目标和承诺剩余时间；提供部署可取得的已承诺轨迹/环上运动摘要。为未入普通环槽的committed交通保留轨迹上下文，避免仍只依赖当前恒速CPA选一架。
2. 输入只能来自当前公开状态/已承诺计划/中心预约广播；不把teacher扫描结果、候选安全通过位或未来实际动作作为特征。新增维度/表示与checkpoint schema显式版本化，旧90D产物原封保留。不能仅把缺失UID临时硬编码进样本。
3. 首先在这4个原快照验证新表示覆盖UID4/UID1、plan目标/剩余时间正确、相对gap与selector/cache一致；无需再运行6个候选，复用本轮已冻结的安全见证。增加旋转不变性（gap）、incoming完成边界与多committed邻机选择回归。此步没有训练。
4. 然后可从三个已消费train回合的原可信pre-state重建新观测，独立新目录按decision_id与原标签join，不改原teacher、不reset/采新seed。先0优化器检查输入/loader，再记录同数据同预算的后续可拟合性比较。student闭环/独立资格与PPO仍另需证据；本轮不自行开始。

本轮优先修复观测语义与信息覆盖，不更换teacher、不继续调loss/阈值，不宣称P3退化已解决。teacher排序的最优性与多机联合协同仍是后续策略性能问题，不能用本轮安全抽查代替。

## 产物与复核

- 原脚本：`terminal_marl/scripts/diagnose_p3_teacher_context.py`；定向tests：`tests/test_p3_teacher_context_diagnostic.py`。本轮生产环境/teacher/90D接口不变。
- 标签汇总`labels.json` SHA：`cd8ffda1b0fbeeab3b57b942baa597443cb10a1808819910976ee41e08d8f783`。
- `witness_v2/candidate_0..5.json` SHA按序：`5e85068fce84e8c011e7af05efd22793c156fa13395ac6d61fb0f6dbe6557e29`、`cdddda2d0afff57679cfa6482b47b561fd7d40a71eba52163214127c21ba73c8`、`e69ba951a2ae4c1b16ea40772cc2d3098caf3826cc035371b0208c7e37a67436`、`bbb540ee3f5acbca279ec25ee480e0fb882bbaeaf071e4825bbd6268f9879113`、`a47f1d2fae93f4eafbd4b04398cc7f8beeca63f88b6db6c0a0ece6eeb2fc59a8`、`2454f45863d925f92759a309177cbdada7000ab7bd84cb3659eb4a96d24998fb`。
- 上述JSON/exit/cost均local/server SHA相同，primary亲自检查完整JSON并用`.tmp/analyze_p3_teacher_context_20261008.py`断言/汇总，0新查询。
- 运行器`.tmp/run_p3_teacher_audit_20261008.py`分labels/首样本/remaining，源码同步SHA、显式cwd、新输出存在检查、进程timeout及总600s预算。临时helper与大型原始产物不入Git；可由模块CLI复查冻结输入，实际完成的候选不要重复启动。
- 其他dirty保留，CURRENT_STATUS仅本任务最新块；本地显式提交，仓库无remote，不推送。
