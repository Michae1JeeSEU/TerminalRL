# Stage2 修正版成对正式运行：首个共同60秒

用户已授权按原合同继续修正版成对运行。Luna已确认进程、日志、两份60秒summary/checkpoint及
states实际存在，**已正常启动，首个共同60秒已完成**。新轮仍在运行，尚无3600秒或预算截尾终态；
此处的中间结果不判赢家，不替代最终审计及人工观察。

| 状态维度 | 当前证据 |
|---|---|
| implementation/correctness | 修复后有界回归218 passed、1 skipped；不替代新正式运行结果 |
| experiment | **RUNNING；T_common=60秒**，中间状态 |
| real_time_observation | **MEASURED_NOT_CERTIFIED**；正常与完整dispatch分开报告 |
| human_review | **HUMAN_PENDING** |

运行源码提交为`2b045bef2b5f6b2673669899fe58a5f068d58a94`，source hash为
`01749a6cf72239e9f557d004fcf297d39a21fa22346d26f9588b98d9b7653c6c`。
冻结文件为`artifacts/rotating_slots_stage2_20261021/repaired_source_provenance.json`。
本次文档工作没有修改代码、测试或运行产物。

服务器UTC **2026-10-21T02:02:25Z**（北京时间10:02:25）启动；wrapper **99045**、runner **99053**、
tracker **99056**、方法workers **99057/99058**，分别绑定物理核心0/2，四个数值库线程环境均1。
启动证据为`campaign.log`、`run.timing.txt`、`run_launcher.timing.txt`、`run_launch_preflight.txt`。
Luna在UTC **02:04:18Z**观测进程elapsed约112秒，RUNNING、无error/退出标记，并确认共同60秒文件；
本文随后读取同步的两份`summary_0060.json`核对下表。

新服务器输出目录为`/home/user/training_outputs/terminal/stage2_repaired_campaign_v1_20261021`，
对应本地`D:\7_codespace\training_outputs\terminal\stage2_repaired_campaign_v1_20261021`。
旧轮根目录`stage2_first_campaign_v1_20261021`仍保留INVALID_STOP与最后共同840秒证据，见
[first_campaign_status.md](first_campaign_status.md)；旧轮没有被改写为成功。

## 输入与运行边界

新旧目录的`manifest.json`、`fixture.json`、`environment.lock.json`已逐字节核对一致。
因此场景、初始SOC、种子、来流和依赖锁保持原值；候选、正常规划和CP求解预算未改，
没有新增算法组或参数搜索。manifest仍为172条，SHA256
`10b97225bde7992af59c79857b0c387bde69ec3b1153e36867a9d42406ad4bc6`。

本次修复影响公共资源判断，按用户授权，两种方法均从同一初态t=0重新验证，读取同一原manifest。
新轮首次60秒之后必须接续各自同一次运行至目标3600秒或实际终态；不将旧轮840秒拼接到新轮。
新输出目录用于保留来源和证据，不代表重新获得10小时预算。

## 累计预算

依据新目录的`recovery_accounting.json`、`budget.json`和`campaign.json`：

| 项目 | 已记录秒数/规则 |
|---|---|
| 旧轮累计继承 | max(旧campaign wall,旧budget floor)=**4812.677915491629** |
| 恢复验证已知耗时 | 140.347；保守额外记入300，含短预检，并非精确测量值 |
| 新轮preparation wall | **300.55827703222167**，已含上述300 |
| 新prepare wrapper实测wall | 约0.809，单独保留观测口径 |
| carry前新目录已累计 | 334.15497630418395 |
| carry后初始elapsed floor | **5146.832891795813**，后续按既有epoch继续增加 |
| 新旧准备阶段合计 | **901.1101518273063**，小于3600准备上限 |
| 计算截止 / 总上限 | 34200 / 36000，不变 |

旧准备600.5518747950846秒已包含在4812.677915491629秒carry内，不能在总预算再扣一次。
新旧准备合计用于检查一小时准备上限，不作为第三次总预算收费。
主对话已确认累计准备元数据采用901.1101518273063秒；新轮自身300.55827703222167秒与累计口径分开，budget不重置。
旧轮归档后没有运行的编码与人工等待已排除；新轮后续自动运行、检查和导出继续计时。
这一继承关系保留在`recovery_accounting.json`，不能通过新目录或重启清零。

初始T0暂取约5小时，依据旧轮840仿真秒用了4033 wall秒、线性粗推约4.80小时运行段。
这仅用于初始监控安排；后续交通密度、公共处置和数值工作量会变化，不保证新轮一定完成3600秒，
也不是5秒实时性或容量承诺。新轮的实际进度及预算终态以完整检查点和持久化记录为准。

## 首个共同60秒中间结果

| 同窗口指标 | Energy-slack | RH-CP-SAT |
|---|---:|---:|
| F / L | 0 / 0 | 0 / 0 |
| admitted / 空中pending / 上游pending | 41 / 41 / 0 | 41 / 41 / 0 |
| 新请求已释放/已准入 | 5 / 5 | 5 / 5 |
| 总能量J | 173987480.33721232 | 173987480.33721232 |
| C_legacy | 2270.8648355119203 | 2270.8648355119203 |
| 决策 / noop / fallback | 20 / 19 / 0 | 20 / 19 / 0 |
| 首块含检查点wall秒 | 93.2668711 | 100.0416546 |
| 正常callback P95 wall秒 | 4.5017359 | 5.0432798 |
| 正常callback >5秒 | 0/20 | 2/20 |
| 完整dispatch P95 wall秒 | 5.0589809 | 5.7453142 |
| 完整dispatch >5秒 | 2/20 | 5/20 |

双方主键相同：`(0, 0, 0, 173987480.33721232, 2270.8648355119203)`，未完成不是失败。
RH-CP-SAT记录99条raw OPTIMAL阶段状态，不等于99个物理全局最优方案，也不能代替实际前缀认证。
只有20次调用，分位数为描述值；有5秒超时，不能据此宣称实时认证或一小时容量。
新轮两份`replay_0060/`均已导出；主对话已读取同步的两份`independent_audit_0060.json`确认：
均为 **PASS、issue_count=0、continuous_separation=CERTIFIED、physical_outcome=IN_PROGRESS**，
各有69 frames、2555 energy intervals、980 continuous pair runs。
本次由Luna在核心4、线程1于服务器UTC **02:07:47–02:07:49**执行`export_episode+audit_replay`，退出0；
证据为`preparation/first60_export_audit.log`与`first60_export_audit.timing.txt`。
导出只读早期检查点，不推进、不重启、不修改live状态流。
审计仅覆盖已保存的60秒窗口，使用冻结数值内核，既不是完整3600秒成功，也不是现实物理模型的独立经验oracle。
人工PyVista状态仍为 **HUMAN_PENDING**。

## 已确认的接续调度

当前对话heartbeat已由工具成功创建：id **stage2**，名称 **Stage2修正版实验接续**，状态 **ACTIVE**，
间隔298分钟。已读取本地`C:\Users\user\.codex\automations\stage2\automation.toml`，
`created_at=1792548292235`，即UTC **02:04:52.235**；下一次约UTC **07:02:52**、北京时间
**2026-10-21 15:02:52**。工具确认已创建，不是任务完成保证。

该间隔依据T0约5小时扣除启动后已过时间；T0来自旧840秒耗4033 wall秒的约4.80小时粗推，
没有用新轮首60秒直接线性外推整轮。保存上下文包含源码指纹、命令、PID、路径、预算继承和完成后分析/交付。
后续复查仍须核对真实退出状态、共同完整检查点、错误记录及归档审计；本记录不安排额外实验、调参或重新prepare。
