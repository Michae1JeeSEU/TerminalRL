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
