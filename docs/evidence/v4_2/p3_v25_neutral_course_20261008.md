# V2.5 当前中性课程与P3继续验证（2026-10-08）

## 范围纠正

primary上一段把`execution_plan.md`历史P3表和旧priority-aware formal配置当成当前规范，新增了priority-enabled数据分支。这是解释错误。`README.md`明确证据文件不能改写规范；`requirements.md`5.2、design/technical与execution_plan0.1/0.2一致规定当前主课程`task_priority_enabled=false`、priority shaping=0，mission只保留物理service/兼容性，等待与能量紧迫度继续参与调度。该规则不是本轮新增，也不因用户要求“正式P3”自动恢复历史任务等级偏好。

保留priority分支的全部代码/结果：flow6为6N完整；dense24环境915ticks/19N5E、无实际安全/hash问题，776actor待标注。primary已授权只停止其未完成标注并保留原ANNOTATING与行政停止说明，不伪造成功、不继续flow12。未训练91901–3、未消费旧Q；没有checkpoint需要撤回。原neutral forecast train_A仍是有效开发证据，但它高SOC/warm的场景不等于以下三条件，不冒称正式三场景采集完成。

## 当前配置与最小数据范围（首次reset前冻结）

新增`configs/p3_v25_course.py`，profile=`p3_v25_neutral_course_committed_merge_20261008_v1`。保留已验证direct C²、committed incoming预测、105+N×50观测、K3、pre-action mask、真实能量/应急和资源服务，沿用foundation的downstream_soc_priority（energy slack、waiting/dynamic priority与可行性共同决定，非只有SOC）。有效base=1、policy mission token=STANDARD，但raw mission和30/120/10/60s service不变，SOC urgency weight2不变。三个条件来自execution_plan0.2：

|条件|请求/初始warm|SOC|到达/周期上限|train seed|Q seed（保留）|
|---|---|---|---|---:|---:|
|flow6_mixed|6/0|uniform .55–.90|10+60i秒；1210ticks|893501|895501|
|dense24_fixed|24/12（6/4/2）|fixed .8|剩余12架每15s；1800ticks|893502|895502|
|dense24_mixed|24/12（6/4/2）|uniform .55–.90|同dense fixed；1800ticks|893503|895503|

fixed .8沿用固定SOC开发值，不提高mixed下限或改变能量阈值来制造通过。每臂一次；自然完成或既定上限。源码、服务分布、能量、安全与调度语义先定向检查，独立smoke892851/3warm3/2ticks验实际入口和输出，永不进入训练。首次预检server新根不存在、7个seed在已见protocol元数据无碰撞；每次启动仍核对无重复进程/输出。

输出根`/home/user/training_outputs/terminal/p3_v25_neutral_course_20261008`，目录`smoke/train_flow6_mixed/train_dense24_fixed/train_dense24_mixed`。入口`training.p3_v25_course_data`。环境3600s、全部actor pre-state按tick/数值UID、cap2048query/51200 RELEASE意图、离线标注7200s、rebuild900s；smoke环境/标注各180s。仅一个父进程和一个顺序query child。超限保持INCOMPLETE，不补标/延时/重跑。原null不作WAIT、forced不入actor loss、单机teacher不当联合执行标签，沿用已测试collector/rebuild语义。

每臂完整后primary核对N/E/F/D/U、technical/censor、安全/hash/能量、全部actor-selected-OK label集合及105+N×50/teacher证书join和成本；安全完成E单列，无新增零E门槛；FDU/技术/实际安全/hash/不完整标注阻止后续。三条件完成再判断训练适用范围，不能因少量train标签就宣布泛化。当前不扩大数据矩阵；需要补充时由实际残差/闭环证据决定。

## 接续学习与正式启动

未曾真实执行的`p3_v25_task_imitation`/`p3_v25_task_student`入口现显式切到本profile及新protocol；priority的原collector/数据不改。拟合仍预定fresh91901/2/3、各5000updates/300s、固定final/canonical首seed、Adam3e-4/batch128/clip.5；输入仅上述三个完整train，fit不打开Q。必须先记录实际输入分析后执行。旧895401–3与895301–3仍未消费，不移作train。

候选按训练内残差和teacher-free开发闭环作必要的Safe DAgger修复，补充只进入新train/development，不借Q调参。新Q895501–3在候选冻结后采集并仅评分一次，保持原独立imitation判据，不事后降低。student-only资格要求完整任务/原请求与likelihood/实际安全/无FDU，原actor takeover<=.05恢复为显式检查；普通候选拒绝不等同安全违规。再做真实critic-only warmup、有界PPO更新与源/更新后同任务对照，检查退化及收益边界。learning预算与具体开发seed在运行前记录；此处不声称任何准入。

用户授权持续修复至新的P3正式训练启动仍有效。必要有界步骤失败则primary诊断修改，不以“等待批准”代替工作；也不以目标期限跳过证据。最终正式配置按本中性课程冻结，核对服务器启动PID/cwd/argv/初始产物后结束，删除前序60min heartbeat，**正式训练不设heartbeat**。无Git remote，代码通过Syncthing同步，不能称Git已推送。

## 实际执行

协议写入时：新profile/入口正在定向测试；新smoke/train/Q未reset、BC/PPO未执行。I0/I1/STUDENT/PPO均未准入。

定向11项local/server tests均通过40.33s/21.11s，8个代码/测试文件SHA相同，实际reset确认fixed.8、effective base1、多物理mission与STANDARD policy token；原始null/完整标注/105+N×50、严格train/Q隔离与无优化器student检查通过。恢复既有takeover<=.05检查，未新增门槛。priority dense父2819670/所属query2821768已SIGTERM，exit143，62个label文件及原ANNOTATING保留、无残余进程，行政停止单列；环境19N5E不是完整采集成功。下一步仅独立892851 smoke。
