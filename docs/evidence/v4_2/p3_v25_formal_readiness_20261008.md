# V2.5至新P3正式训练的持续修复与准入（2026-10-08）

## 用户授权与目标

用户明确要求：根据每步结果持续分析修复，直至新的P3正式训练就绪，再同步服务器启动并结束；**最后的P3正式训练不设heartbeat**。这覆盖必要的数据/模型/执行接口修复、有界学习与资格验证，并允许通过后启动正式训练；不把授权当作实验已经通过，不降低安全约束。主代理亲自读研究文档、修改核心/协议、诊断和判断；luna_worker仅执行明确测试/SSH/Git。代码/docs只本地经Syncthing，服务器不Git、不手传源码。所有失败保留，其他dirty不动。

当前已有：direct C²几何/动力学修复、小样本规则/source完整闭环、三个完整train回合；V2.5观测修复重建1046行；同网络信息对照新输入平均joint86.33%、gate AUC .96569，但proposal仍不足，82/88/99次拟请求属于原teacher拒绝候选。这些都是已有证据，不是新schema正式I0/I1/student/PPO资格。

推进顺序：固定模型拟合曲线→根据残差决定最小修复→V2.5请求/mask/K3/likelihood与执行桥接→完整开发闭环→冻结数据及fresh多seed imitation资格协议与untouched评估→必要Safe DAgger→student-only资格→PPO实现/critic及有界更新防退化检查→正式P3准入审查及启动。每步具体seed/预算/成功条件在执行前补充，使用原有规范的安全、无teacher替换、请求概率、完整业务周期等要求。历史88D资格及旧Q不能直接给V2.5背书，也不机械搬用旧采集样本数。

正式就绪至少需要：相关新schema与策略执行一致；数据分区及teacher语义成立；独立student-only完整周期无技术/实际安全/hash/F/D/U问题，保留应急计数与既有takeover边界；PPO请求log-prob/forced mask/terminal reward与GAE正确、更新预算与防退化机制经过有界验证；冻结正式配置、初始化、训练上限和输出目录。`PPO_ALLOWED=false`保持至实证准入；不能仅凭训练集准确率启动正式PPO。最终服务器启动后检查真实argv/cwd、日志及至少一个正常中间产物，不监控最终完整训练。此前预计长于交互窗口的任务可按用户要求60min监控，正常推进静默。

## 第一步：固定新信息5000更新曲线（运行前冻结）

协议`p3_v25_fixed_fit_curve_20261008`：只原1046条train及相同label；固定V2.5网络59451参数、新信息输入、未加权CE、Adam3e-4/batch128/global clip.5，3个初始化91701/2/3。每seed从原fresh初始化重新跑**5000更新/300s**，固定1000/2500/5000记录，不选最好epoch，不调权重/阈值或临时延预算。1000点必须与上轮同seed新信息臂的参数hash、batch序列hash和完整指标精确一致，证明新增测量未改变原前缀。这里只研究延长预算后的残差，不把5000对旧信息1000的差异称为观测因果效应。

每seed只保存final诊断checkpoint，曲线三个点都保存原1046逐行预测/指标，方便检查gate、proposal及原前缀已知拒绝率。仍不reset/step/teacher/audit/Q。结果决定下一步，但不据本曲线作资格判断。

修改`p3_v25_set_learnability.fit_arm`只增加可选固定测量点/reference-prefix断言，原默认1000入口行为保持；新`p3_v25_fit_curve.py`独立入口。先合成smoke（2更新prefix+3更新curve）验证同参数/同batch/输出，再真实三seed。测试/源码同步后执行，输出`/home/user/training_outputs/terminal/p3_v25_set_curve_20261008/{smoke,curve}`，新目录且禁止重复启动。预检无项目Python，原fit SHA两端一致，新根不存在。

## 实际执行与分析

以下仅追加实测证据。

曲线实现的20项local/server tests均通过（3.25s/1.29s）；5文件SHA相同。合成smoke已CURVE_COMPLETE/exit0，2更新reference与3更新curve的前缀精确相同，固定2/3两个点、final checkpoint及JSON有效，两端SHA一致；root result SHA=`5549e30e1e6696fb7c6abbb4dff09202bef227265ec033c331aaac15b3a6b4d8`。primary核对后准许本地提交后真实3seed×5000更新。尚无真实curve结果，不改变资格。

### 固定曲线终态与下一判断（newly_run）

核心94cabf1后真实curve已CURVE_COMPLETE/exit0，91701/2/3各5000更新；1000点的参数/batch/全指标均精确匹配上轮。result SHA `b74f766d1147f0f2b067e294456ada2cb92f898ccb89a5e4446281da45b46228`，两端同步，3个固定final新checkpoint保留，Python2803088已退出。

| seed | 1000 joint/proposal top1 | 2500 joint/proposal top1 | 5000 joint/proposal top1 | 5000 W/R recall |
|---|---|---|---|---|
|91701|86.62%/65.50%|95.98%/88.21%|100%/100%|100%/100%|
|91702|86.14%/59.83%|95.22%/87.77%|100%/100%|100%/100%|
|91703|86.23%/60.70%|95.51%/89.96%|99.81%/100%|100%/99.13%|

三个final gate AUC均1，proposal top3均100%。这表明当前1046个已知train状态的主要残差可以通过原网络/原loss继续拟合消除，暂不增加loss或更换架构；不能由训练集拟合宣称泛化。接下来直接检查执行与开发闭环，不再反复调训练集分数。固定选seed91701作为开发canonical（按编号，非择优；三seed均保留，正式资格另行全量评价）。

### 第二步：V2.5执行桥接与开发闭环（运行前冻结）

新增`p3_v25_runtime.py`只从当前V2.5 set取样，沿用既有pre-action actor mask、WAIT sentinel、requested gate+conditional proposal log-prob；K3始终以真正请求的proposal为首，其后按logit稳定排序。forced行WAIT/0joint likelihood、不入actor loss。加载显式new-information/V2.5 checkpoint，禁止把旧信息或88/90D模型当新模型。环境内部保留90D物理配置供兼容，**policy实际读V2.5**并在protocol单独记录；不调用teacher/candidate，不动safety。

新增评估入口执行真实ExecutionScheduler/SafetyFilter，记录每tick原请求/likelihood、actor mask、K3、实际trace、状态hash/reward，再从初始pickle以相同请求重放全周期；核对pre/post hash、reward、实际安全与checked/executed一致性、权重不变。完整周期自然终态或1800ticks，rollout/replay各3600s；技术/安全/F/D/U/censor阻塞时停止后续臂，保留failure快照，不扩负载。shield_actor_fraction沿用<=.05要求。此阶段是development，不冒称正式student qualification。

固定checkpoint为`p3_v25_set_curve_20261008/curve/seed_91701/diagnostic_final.pt`。独立smoke env892801/policy928801，3请求/warm3高SOC、deterministic、2ticks；仅验证输出/执行重放。随后顺序A deterministic、A stochastic、B deterministic、B stochastic、C deterministic、C stochastic：A=12/warm6高SOC、env892802/policy928802；B=24/warm12高SOC、892803/928803；C=24/warm12混合SOC .55–.90、892804/928804。各对共用exogenous seed但从头独立rollout；不用Q/audit。代码/docs搜索未发现这些seed实际使用记录（rg no matches），服务器启动前另查无输出/进程。测试及smoke通过后才启动A；每臂完整结果由primary判断后继续，不自动铺开。输出新根`/home/user/training_outputs/terminal/p3_v25_student_dev_20261008`，最多一个本轮评估进程，所有失败保留。预计若分钟级继续当前任务；超过交互窗口再60min heartbeat，最终正式训练无heartbeat要求不变。

执行桥接21项local/server tests通过（10.84s/5.53s），3源码SHA相同。独立892801 smoke已SMOKE_OK/exit0：2ticks、3actor/3RELEASE、0shield，3个正常批准；完整tape replay_exact=true，实际安全/hash异常0，checkpoint不变。primary读取steps确认likelihood和pre/post hash存在（这些字段在steps中，不是root result命名字段），3个未决是预定2tick截断。result SHA=2edd92540d8b591e7a9f270705bbcb9e999f33a5495cd484c6e9d70463446258，两端同步。接受提交后只启动A_det完整开发臂，其他臂仍依完整结果决定。
