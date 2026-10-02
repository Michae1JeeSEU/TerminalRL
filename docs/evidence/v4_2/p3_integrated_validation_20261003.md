# P3统一修复后的分阶段效果验证（2026-10-03）

## 目的与已有结果

用户要求逐步验证修复是否解决P3退化。沿用commit `d43076b`的新V3/90D profile，先确认有负载的物理/业务闭环，再检查策略迁移和学习。上轮196测试及3请求规则/冻结source完整周期属于preexisting_artifact，不能替代本轮密集场景证据。源码和文档修改、诊断及下一步选择由primary完成；worker仅执行指定命令、读取产物和监控。

本轮读取发现warm-start已进入OutcomeLedger却遗漏request_times，使CLI generated_requests漏算warm请求。补登记原始request_time，不改变运动、调度、奖励或随机采样；在大样本运行前用定向测试和warm-start两步smoke验证。

## 顺序与停止条件（运行前确定）

采用现有 `terminal_marl.scripts.diagnose_p3_integrated_repair`，全周期最多1800 ticks。规则与冻结source使用同一V3物理、安全、动作权限、90D、入场及reward；source为 `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt`，SHA256 `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`，仅显式迁移并冻结。所有seed是开发诊断用途，不使用历史held-out，也不将单场景称为泛化证明。

1. A：12请求、warm6、SOC .90–1.0，env seed890501/policy seed890601。先规则两步smoke；primary核对初始化、统计和输出后运行规则完整周期，再运行同场景冻结source。
2. B（A未暴露阻塞后）：24请求、warm12、SOC .90–1.0，seed890502/890602，规则/source配对。保持高SOC，检查密集构型能否疏通。
3. C（B可运行后）：B相同请求随机流/种子，仅SOC范围改为.55–.90，规则/source配对，检查能量与资源耦合。更低SOC或额外seed只在当前证据需要时另记，不自动扩充矩阵。
4. 基础闭环可接受后，决定V3/90D数据适配与student-only qualification，再预先固定有界学习更新及前后配对比较。不能把零初始化迁移称为训练，也不能用旧V2 qualification自动解锁PPO。当前仍 `PPO_ALLOWED=false`。

每臂完整保留N/E/F/D/U、是否自然终态、技术error/censor、实际安全计数和checked/executed一致性、完成时间与能量。任一技术异常、安全/执行不一致立即停止向下一难度推进并诊断；高SOC仍发生耗尽或未决时先检查可达性/阻塞，不直接开始PPO。备降、应急等业务结果保留原类别；主审结合两控制器结果判断，不把“脚本exit0”视为业务通过。

工作目录为服务器 `/home/user/projects/terminal`，Python `/home/user/envs/terminal_env/bin/python`。输出根 `/home/user/training_outputs/terminal/p3_integrated_validation_20261003/`，各臂新目录，不覆盖历史。运行前核对Syncthing源码同步；不手动传输源码。预计长周期配置heartbeat，正常推进静默，终态清理。

## 本轮结果

待运行；本段后续只追加实际结果。

### A运行前检查（newly_run）

- 两个定向测试文件：30 passed，26.90s，exit0；日志 `preflight_tests.txt`。
- `A_rule_smoke`：2ticks、CENSORED、error=null，初始6架活跃、generated_requests=6、queue=0；安全计数全0，结构化输出正常。CENSORED是预定两步结束，不是完整周期结果。
- 本地/服务器5个指定文件SHA一致，服务器无重复诊断进程。采用前述固定A参数继续规则完整周期，输出 `A_rule/result.json`；冻结source尚未启动。


执行顺序调整（A规则尚未终态时确定）：A规则已推进300ticks、3完成/8活跃/0队列，暂无技术异常。为减少墙钟等待，同一A场景冻结source先做两步smoke，通过后可与A规则并行，使用独立输出和进程；不提前升B、不改变seed/配置/上限。两臂均完整保留，不依据中间优劣挑选结果。进程数限制为本轮2个；任一技术/安全失败仍由primary决定后续，不自动串行启动B/C。


### A中间进度与后续执行

A_rule已确认PID2355453（bash2355452），完整argv与协议一致，初始tick1实际输出存在。UTC2026-10-02 18:46:41快照425ticks、5正常完成/7活跃；随后625ticks、elapsed949.56s仍5正常/7活跃/0queue。A_source两步smoke通过后已开始完整周期；400ticks、elapsed468.59s、4正常/8活跃/0queue。均未终态，不能据此报PASS或断言死锁；规则较长时间没有新增完成作为待检查现象保留。

本轮最多同时保留这两个独立进程。长周期由当前任务heartbeat续查：正常推进静默；两臂终态后primary读取完整证据，先判断A是否存在技术/安全/耗尽/未决阻塞。无阻塞才按既定B、C继续；有阻塞则冻结更高负载及学习，primary亲自分析并选择最小复现，不让worker诊断或改核心代码。不能只发“任务完成”然后停止本次验证。任何新的bounded learning须先满足正文中的环境和student-only资格条件，并记录具体更新方案，当前不训练。

A_source已确认Python PID2357445，完整argv/首tick及checkpoint SHA核对通过；SSH session40720。进程超过10分钟观察窗口后继续运行，未终止。两个只读监控已合并为当前任务heartbeat `p3`（每5分钟），负责结果到齐后的primary分析和有条件继续，不仅提示完成。


### A中途发现能量失败（尚未终态）

UTC2026-10-02 19:30:39的只读服务器快照：A_rule PID2355453仍运行，1450ticks、5正常完成、3 failed_energy（arrival_id 6、7、0）、4活跃、0queue；A_source PID2357445仍运行，1325ticks、4正常完成、8活跃、0queue。两个result.json与exit均尚未生成，日志为空。此处是newly_run的中间事实，最终安全/hash、能量细节和完整N/E/F/D/U仍待终态。

高SOC下已发生实际耗尽，按运行前停止条件，B/C和任何学习保持禁止。保留现有两臂到自然终态/原1800ticks上限，以取得完整配对；不改运行中的核心源码、不新增对照。它说明当前组合尚不能满足A的业务闭环要求，但不能仅由进度推断具体安全阻塞、应急失效原因或PPO退化的因果份额。终态后先读能量phase、emergency和release遥测；若聚合信息不足，再用相同A规则场景的有界重放记录逐次拒绝原因和实际活跃状态，定位最早持续阻塞，不扩大负载。


### A完整结果与最小阻塞重放（newly_run）

两臂自然终态，error=null、censored=false、exit0；服务器UTC2026-10-02 19:40:47确认两个原PID及匹配CLI进程均不存在，本地/服务器result SHA一致。N/E/F/D/U依次为正常/应急/失败/转场/未决；此次F均为failed_energy。

| 臂 | ticks / 墙钟s | N/E/F/D/U | 最后正常完成 / 首次耗尽s | hard / collision / hash mismatch |
|---|---:|---|---:|---|
| A_rule | 1534 / 3332.62 | 5/0/7/0/0 | 413 / 1401 | 0/0/0 |
| A_source | 1537 / 3196.62 | 4/0/8/0/0 | 358 / 1401 | 0/0/0 |

两臂均generated=12、终态active/queue/service=0，外部队列累计20 UAV-s。实际最小swept间隔分别99.7432m/84.7945m；operational buffer、NMAC、joint fail-closed均0。已检查执行采样分别12796/12749，hash mismatch均0。规则reward=-218.9624，source=-287.7294；不同控制器差异不是PPO更新效果。

所有耗尽者均在ring_inner、SOC=0、assigned_lz=null、emergency_landing=false。规则arrival_id为6、7、0、11、9、8、10；source为6、1、7、2、11、9、8、10。两臂最后一次正常完成后约1000秒没有新正常完成。lz_idle_with_inner_demand_steps分别1388/1401。应急attempts为840/960、committed=0；按候选pad累计的blocked_safety为2866/3272、blocked_energy为2174/2488，blocked_lz/dynamics均0。候选pad拒绝次数不能当成独立应急请求数。

结论边界：A业务闭环失败，因此本轮统一修复尚不足以解决P3退化。规则和冻结source都出现内环持续阻塞，不能只归因PPO更新或熵；当前最直接待查的是内环退出轨迹与其他holder的安全可执行耦合，以及应急同样无法提交。零实际违规表示安全约束保持，并不证明拒绝过度保守。需要候选级证据区分真实冲突、保守预测、轨迹生成几何和资源调度，不能据此降低安全阈值或增加电池来掩盖问题。

result SHA256：A_rule=`0201732e9ebb5ca98d63f6493118c47a8d131ed6a0bc3e70420cd76b06dfde2f`；A_source=`b9276974e1a9b462e5a06d0f18da936010e4df57f1287044c6cf28dca24ae423`。原产物保留。

下一步预定为单臂同seed规则有界重放：仍12请求/warm6、高SOC、原动作K3；到900ticks主动截断（诊断CENSORED，不冒充完整周期），记录每tick release拒绝原因、应急累计量、每25tick实际活跃状态，保存400/600/850ticks环境快照供同状态探针。源码仅增加opt-in `--trace-blockers --snapshot-ticks`观察输出；不调用额外候选或改环境决策。先运行定向测试（含快照恢复下一tick一致性）和2tick trace smoke，核对Syncthing和输出后才启动`A_blocker_replay`，最多一臂。聚合记录仍不足时，再针对这些快照调用原SafetyFilter生成最小单候选/单邻机证据；不开新负载矩阵。B/C、训练、held-out保持禁止。主代理亲自写诊断代码并分析。


阻塞重放前置检查：定向25 passed（39.38s），含环境pickle恢复后下一tick观测、奖励、遥测、outcome和SOC逐项一致；4个本地/服务器源码SHA一致。`A_blocker_smoke`为2ticks、error=null、6请求/6活跃，2条JSONL可解析，snapshot_2.pkl存在（621332bytes）；与原A_rule_smoke的业务、reward和全部telemetry相同。主代理核对了逐候选输出：首tick已经可见部分transfer候选`dynamics_trajectory_segment_mismatch`，以及正常headway拒绝。该拒绝不造成脚本异常，但需要与后续内环主阻塞分别识别，不能默认都是预测间隔。继续既定900ticks单臂重放。
