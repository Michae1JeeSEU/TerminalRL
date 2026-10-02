# P3 实现问题与统一修复（2026-10-03）

## 授权、范围和结论边界

用户要求把本轮发现和建议完整记入 docs，由 primary 亲自完成重要实现修改；全部相关修改完成后，再逐步验证效果。文档和核心代码由 primary 编写，worker 仅可执行明确的进程、测试、Git 命令。不得以尚未完成的效果实验声称 P3 已解决。`PPO_ALLOWED=false`，不启动训练、不使用 held-out、不覆盖历史 checkpoint 或实验产物。

本轮依次处理动力学、轨迹、预约、应急资源、能量期限、观测、reward/actor eligibility 与重新 qualification 的入口。V2 历史重放保持原配置；新的行为组合通过明确的新 V3 repair profile 选择。安全间隔、合加速度、最低降落 SOC 不降低。已有用户工作区改动保留；不把别轮改动混入提交。

## 已有证据和正在运行的 R2

历史依据为 [基础复核](p3_foundation_review_20261002.md)、[V3 结果](trajectory_v3_repair_results_20261002.md)、[ring-budget 待应用补丁](trajectory_v3_r2_ring_budget_repair_20261002.md)、[P3 follow-up](p3_followup_results_20260930.md)。这些是 `preexisting_artifact`，不称本轮复跑。

北京时间 2026-10-03 01:36 的服务器快照：R2 完成 2/8 arms。C1 legacy88 在 1017 ticks 因 `ring_hold_tangential_budget_unavailable` 异常，N/E/F/D/U=6/0/2/0/16。C1 repaired88 在 1428 ticks 自然结束，N/E/F/D/U=2/1/20/1/0；20 个失败均为 `failed_energy`，外/中/内环=8/9/3；碰撞、硬间隔违规和 checked/executed hash mismatch 为零。账本 valid 不等于业务通过。C1 repaired90_zero_initialized 随后启动。24 请求不是 24 个独立实验。

repaired88 累计：普通 release approved/rejected=15/2935，shortlist exhausted=1996；emergency attempts=2452，blocked safety/energy=7555/6857，committed/touchdowns=1/1；有空闲 LZ 且内环有需求的 steps=1027/1428。这些候选检查重复计数，不是独立冲突人数；空闲 LZ 不证明当前存在可执行安全路线。原始产物保留于 `D:/7_codespace/training_outputs/terminal/p3_foundation_repair_v3_r2_20261002/`（服务器同名 training_outputs 根）。

主代理决定停止旧 R2：已完成分支暴露技术失败和严重业务失败；余下零初始化迁移比较不能证明新增观测的学习效果，所有 arms 仍共享有缺陷的 V3。继续旧矩阵不足以推迟源代码修复。只终止核对身份后的 PID 2323277，保留旧结果和不完整进度，不将未运行分支写成失败或成功；停止与监控清理结果在本文末补充。源代码修改在确认进程退出后开始。

## 问题、机制和计划修改

### 1. V3 圆周速度边界的减速停滞（已复现）

前一分析轮 `newly_run` 数值探针在服务器 Python 3.12.3 / NumPy 2.5.1、r=200 m、dt=.2 s、a_xy=1.5 m/s² 下执行：V3 由 16 m/s、目标 18 m/s 可达到 sqrt(300)=17.320508075688775 m/s；随后目标改 16，500 步/100 s 后仍停在上限。17.1/17.2 m/s 的 executor 可降速，但 generator 的 .97 reserve 使这两种起点被拒绝。现有 V3 减速单测 r=240/current16/target14 没覆盖边界恢复。

`VehicleDynamics.limited_tangential_speed` 的 max(current,next) 向心项与单步恒切向项在边界耗尽预算。`TrajectoryGenerator._v3_ring_transition_parameters` 又把最大向心负荷与全段最大切向负荷同时计算，并施加 .97 总预算；inner threshold sqrt(200*1.455)=17.05872210923198。它排除的是模板，不是物理上所有连续减速曲线。

修改：V3 采用连续的有界速度过渡，执行和规划共享约束计算；边界从零切向加速度平滑启动减速，积分角位移与返回 P/V/A 一致。区分预期轨迹不可行与坏输入/实现错误。增加合法加速→边界→有限时间恢复、边界 PVA、全曲线加速度及 V2 不变回归。不能只改异常类型，不能把 V2 宽松检查当作修复。该缺陷是 V3 问题，不能解释全部早期 V2 P3 退化；旧 R2 未保存足够状态，不能声称20次失败均由它导致。

### 2. 安全约束下交通不能疏通（代码机制明确，贡献待验证）

应急优先规则按所有 at-risk 飞机数阻挡普通飞机，未绑定兼容 LZ、可执行路线和可用时刻。可能形成“应急路线受阻→普通机被扣留→环道持续占用→更多低电量”的循环。SafetyFilter 对未承诺 holder 的全程 ±2 调速可达管是合理鲁棒假设；V3 路径变长会扩大其保守影响，不能直接缩管而继续允许原动作。

修改：应急优先必须对应实际可兑现的兼容资源/安全候选；已先尝试而无法获得路线的应急机不无限虚占所有空闲 LZ。为相互避让提供可执行的 holder 速度承诺：只有通过完整 safety certification 后才绑定，执行器必须执行承诺，后续 RELEASE 必须重新通过与既有计划的安全检查；不得仅在预测中假设 nominal、执行却仍允许任意 residual。全部状态变化要与成功 commit 一起发生；失败候选不泄漏资源或承诺。

### 3. 预约、等待期限与能量下界（已确认表达局限）

预约初值采用运动时间 surrogate，不能保证 V3 的真实到达时刻可达。`phase_motion_nominal_solo_energy=240.9` 来自旧 solo calibration，不能自动当作 V3 完成保证。候选 fallback 的 energy 下界没有安全和资源承诺，不是多机 viability certificate。warm-start 单机足量也不证明共同可完成。WAIT 缺少和 RELEASE 对称的未来资源期限检查，固定 SOC 阈值可能在可用路线消失后才接管。

修改：V3 预约依据实际生成轨迹的可达时刻闭合，并在真实时刻重查 headway/冲突；保留 speed/timing intent 的作用及偏移记录。新 profile 的入场检查使用当前执行模型的路线/服务资源预测而非旧常数；同 tick 多机入场共享剩余容量。将路线能量余量换成可等待时间，提前触发 route-aware 应急尝试；可用 LZ ETA 与竞争负荷纳入估计，但不把估计命名为安全保证。过载备降和控制失误耗尽分别统计。

### 4. 历史 qualification 与冻结策略的适用范围

旧 coverage 在 known scenes 改善，但 critic warmup/refresh、joint-team weighting、full-MC 和大 batch 均未消除退化；prefix development 提升也未通过 fresh screen。旧 checkpoint 来自 V2/旧 final 物理映射，冻结 proposal 不能自动适应 V3。不是断言所有早期训练错误，而是 qualification 不跨 MDP 自动继承。

修改：建立统一的新环境配置和明确的 V3/90D 初始化入口，冻结历史结果。新增数据和 student-only qualification 必须覆盖高速内环、busy LZ、低 SOC、retry 和拥堵恢复；实现准备完再验证、再决定是否采集/训练。禁止把 checkpoint schema metadata 改名当成重新训练；禁止自动解冻 proposal 做 PPO，K3 排序改变必须另行检查执行分布。

### 5. 观测、奖励和更新资格

task-neutral 88D 统一 mission token，但原始 service/compatibility 仍不同；此 alias 不泛化到历史 priority-aware 88D。已有90D包含 own service duration 和 earliest compatible LZ ETA，应在新 profile 实际使用。zero-new-column 且 frozen 的90D只测试迁移一致性，不测试新增信息是否能学会。90D仍不承诺完整 Markov 性。

修改：新 profile 使用90D及显式迁移、`pre_action_forced_wait`；动作前已确定不可决策的 retry/冻结状态不进入 actor 梯度。把外部排队成本纳入团队奖励，避免对同一等待重复收费，保留旧 reward 配置重放；拒绝惩罚单列，效果以全请求业务账本衡量。共享 team advantage 本身不是数学错误；先保留正确的 requested-action likelihood、terminal/bootstrap 和 joint-team 聚合。提供 gate-only entropy 选项，避免冻结 proposal entropy 通过 gate 系数产生额外 RELEASE 偏好，但不宣称这是退化根因。当前 progress reward 已为零，不重新引入绕圈奖励。

## 实施与验证顺序

先完成上述相关代码、配置、测试和文档修改。期间只做语法/导入/接口等开发检查，不在每个局部修改后选择性跑性能。全部实现完成后依次：定向单元与集成回归→小样本无训练完整周期与输出检查→相同权限的规则/冻结策略对照→根据证据决定新数据及 student-only qualification。新 profile 的安全检查不得绕开，当前阶段不启动 PPO 或硬件。对未证实的假设保留对应开关/对照，不以“修改完成”宣称“根因已证实/效果已改善”。

## 实施记录

- 初始状态：primary 开始实现；本文件是问题与修复计划，尚无新实现验收或效果结果。


### 本轮具体实现（效果待验证）

- R2 PID2323277 于 UTC 2026-10-02 17:45:46 核对完整 argv 后收到 SIGTERM，17:45:57 确认 absent。只停止此 PID，没有删除/重写旧产物。工具删除 `p3-v3-r2` 返回 `not_found`（app 已不存在）；后续核对磁盘 TOML 已不存在，未继续保留旧监控。旧矩阵记作 `STOPPED_FOR_IMPLEMENTATION_REPAIR / INCOMPLETE_COMPARISON`，不伪造原 result 的终态。
- `vehicle_dynamics.py` 增加 shared smoothstep 速度过渡时长上界：利用向心负荷随减速降低的事实，给出包含速度边界的合加速度充分条件；执行每个物理步的速度 smoothstep 在两端切向加速度为零，角位移为平均速度积分。V2 分支保留。generator 使用同一约束计算，候选曲线仍过解析 quintic bounds；有效高速状态不再因 .97 reserve 被一律拒绝。V3 切环终端速度受目标环真实曲率速度上限约束。
- 新 profile 采用全程明确的 `nominal_until_release` 等待约定，属于前述 holder 承诺的简单实现：所有未承诺转移的 holder 保持 nominal，直到其 RELEASE 通过安全检查；SafetyFilter、shield 当前/后继备份、位置预测、forecast 和 executor 一致。它不等价于旧 fixed-rule 的全局 `max_speed_residual_mps=0`：RELEASE 的 ±2 m/s、±6 s 和 K3 仍保留，规则/学习策略使用同一配置。nominal 自身不安全时仍 fail closed，不做未验证的强制执行。当前不实现复杂的按路线动态授予/撤销 holder 调速承诺。
- `committed_route` 应急优先：保留应急先执行、通过安全后预约 LZ 的顺序；移除该新模式下基于纯 at-risk 人数的虚拟扣留。已成功占用的 LZ 仍不能被普通机拿走。route-aware 接管每5秒重估兼容LZ ETA+等待+实际应急路线能量，余量不足一个shield horizon时尝试提前接管；它仍是估计，必须经过真实energy/safety/commit，不称联合可行性保证。
- `ExecutableLandingBudget` 用真实 V3 的 inbound→O2M→M2I→final 孤立路线计算正常完成能量和时间，静态几何下按 gate/LZ 缓存，不依赖240.9旧常数。busy/reserved LZ ETA与未分配飞机/同批已批准入场的服务工作量用于虚拟资源日历；不把日历写成真实预约。生成不了路线时先等待，并保留低SOC/最大等待的备降边界。该有限路线族不可行不等于所有物理路线不可行。
- 入场容量另修复独立问题：同tick批量准入计入本批已批准飞机及已承诺进入外环的飞机，防止超额接受。这项是容量正确性修复，也适用于旧配置；不声称所有旧过载轨迹保持完全相同。
- 预约采用已有 actual-time reconciliation，并在新profile显式开启；late-slot仍重查完整headway和SafetyFilter。旧单项late-slot实验失败保留，不宣称该开关单独有效，也不将其包装为新优化求解器。
- `p3_integrated_repair.py` 统一新profile：V3、90D、K3、canonical WAIT、pre-action forced-WAIT、downstream/SOC顺序和上述修复；PPO仍锁定。88D/V2 source经原始schema加载后显式迁移，新增列零初始化、权重冻结，metadata明确 `NOT_QUALIFIED`。
- 队列成本作为独立team reward进入student rollout，队列无虚构agent/action；队列时间与能耗同区间记账，外部备降单列惩罚，服务期间不重复发低层reward。新profile拒绝惩罚为0，避免用惩罚变化冒充业务收益；历史默认不变。PPO新增 `entropy_mode=gate_only`，供之后准入的训练显式使用；joint likelihood/GAE/终态处理未改。没有自动解冻proposal或进行训练。
- 新CLI分为 `--check-only` / `--smoke` / `--run`，输出目录须新建，运行固定有界；规则与source同环境，输出保留error/censor和所有请求outcomes。它是开发检查入口，不是新的正式qualification协议。

### 修改完成后的验证计划

第一步执行本轮新测试和已有动力学、轨迹、安全、能量、入场、应急、服务上下文、student rollout/PPO相关回归。第二步验证新CLI check-only与2-tick smoke以及结构化输出。第三步只在前两步通过后运行小样本高SOC完整周期，再决定dense冻结策略对照。后续模仿数据采集、student-only qualification与正式PPO需要基于这些结果另立明确协议，不在本轮偷偷启动。完整效果未知，不承诺无能量失败或吞吐提升。


### 验证中的补充发现：入场 tick 的重复能量记账

最终复查发现原 `step` 先推进整步 external queue，再把获准请求在同一个 `[now, now+dt]` 区间执行整步 inbound。即使从空场立即入场，也额外消耗一整步排队能量、累计一整步不存在的外部等待。原 active reward 在入场前创建，因此新入场飞机原先没有首 tick 的 active time reward；不能误称原来已经同时扣了两份 active/queue reward。新奖励必须与新的时间划分一起修改。

增加显式 `external_queue_step_mode=post_admission`，只在新profile使用：先按当前状态决策并安全提交入场，再给仍在队列（包括安全提交失败退回的请求）记一个区间的时间、能量及队列成本；获准请求只推进该区间的入场飞行，并补首 tick active time/urgency reward；当前时刻已备降的请求只记备降成本。历史默认 `pre_admission` 保留以重放旧结果。新增成功入场与提交失败两个回归，加上已有持续排队用例，验证三条路径的互斥计时。这是每次入场一个 tick 的计时误差，不能解释长期耗尽全部现象。

### 修改后的验证记录（newly_run）

服务器运行环境为 Python3.12.3、NumPy2.5.1、PyTorch2.13.0+cu130、pytest9.1.1；运行前核对本地/服务器涉及源文件SHA256一致，无手动复制源码。

- 第一轮新测试：20 passed。相关22文件回归首轮192 passed/1 failed，仅旧fixed-rule异常消息兼容性断言失败；保留旧分支原消息后同组193 passed（7.35s）。未把这次文字兼容性修复称为安全效果改善。
- CLI `check_only`、`smoke_rule`、`smoke_source` 均 exit0，结构化JSON可解析。两步smoke均CENSORED、1个活跃请求、无error；这是输入/迁移/输出检查，不是完成率实验。source只显式迁移并冻结，没有训练；source SHA256 `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`。
- 上述结果位于 `/home/user/training_outputs/terminal/p3_integrated_repair_20261003/`，包括 `first_tests.txt`、`regression_1.txt`、`regression_2.txt`、`check_only/result.json`、`smoke_rule/result.json`、`smoke_source/result.json`。首次完整cycle的两次启动未创建进程或产物，启动失败保留为工具执行问题，不列为仿真失败或成功。
- 队列tick边界补充修改后须重新回归，再开始3请求高SOC规则完整周期；此处结果待后续追加。

- 队列边界修复后的同组回归：**195 passed / 0 failed，29.75s**，`regression_3.txt`，SHA同步一致。全套项目测试未宣称运行。
- 首个新环境完整周期：`cycle_rule_3`，规则策略、3请求、无warm-start、high-SOC=.90–1.0，env seed890301/policy seed890401。`COMPLETED`，475 ticks，3正常/0应急/0失败/0备降/0未决，active/queue/service全0；最终SOC=.7405794682/.7802346154/.7748444022。碰撞、NMAC、硬间隔违规、joint fail-closed、执行hash mismatch全0；checked/executed匹配12，执行样本4371；最小actual/swept间隔105.6105842/105.5777655m。exit0，PID2350964已退出。结果在同根 `cycle_rule_3/result.json`；对应规则监控已删除。
- 这只验证3请求高SOC的基本执行闭环，不证明dense24/随机低SOC场景可疏通、旧策略可迁移或PPO不退化。下一步使用同一配置/种子运行冻结source，结果另列；不把两步smoke替代完整周期。

- 配对冻结source周期：`cycle_source_3`，相同场景/配置/环境seed/策略seed，source路径 `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt`，SHA为上述3fe7。**COMPLETED，540 ticks，3正常/0应急/0失败/0备降/0未决**，active/queue/service全0；最终SOC=.7197472715/.7735140886/.7294348719；碰撞、NMAC、硬间隔、joint fail-closed、hash mismatch全0。匹配12、执行样本4784、最小actual/swept间隔均97.94772585m，exit0，PID2351691已退出。source监控已删除。
- 单场景描述性差异：规则475 ticks、冻结source540 ticks，相差65 ticks；两者均3正常完成。不能把此差异外推成一般策略优劣，更不能与旧R2 dense24直接比较。
- 最终兼容性检查另发现外部最长等待和低SOC同时满足时，旧路径原本优先记low_soc。本轮在新入场路线上优先处理最长等待是为防止未知路线inf无限等待；现已将此顺序明确限定新模式，恢复旧模式原先reason优先级并追加测试。该收尾不改变两次完整周期使用的新profile，随后再做相关回归。

### 交付边界和下一步

所有本轮重要源码、测试和文档由primary亲自修改。worker只执行指定SSH、进程、测试及Git操作。没有降低安全距离、动力学上限或SOC floor；没有训练、更新checkpoint、使用held-out或推广默认配置。

当前解决的是已复现实现缺陷及新环境一致性，尚未证实P3训练退化的因果份额。后续按以下顺序扩大验证，任何阶段失败均先定位失败链，不直接用PPO补偿：

1. 先验证high-SOC warm-start/较密入流的规则与冻结source，分离拥堵可达性与能量约束；保留全部失败/未决请求账本及阻塞原因。
2. 再加入随机SOC与异构服务时长，检查正常/应急/耗尽/备降、等待分位数、LZ利用率和实际安全。使用本轮同一配置，避免规则与学习策略控制权限不同。
3. 物理及业务闭环通过后，在V3/90D重新采集覆盖高速内环、busy兼容LZ、低SOC、retry和拥堵恢复的数据；旧source仅初始化，重新做student-only imitation qualification。零输入列的冻结迁移不替代这一过程。
4. 通过后才准入有界PPO；requested-action likelihood与joint-team目标保留、显式gate-only entropy，评价完整周期业务指标。当前 `PPO_ALLOWED=false`。


最终回归：`regression_4.txt`，**196 passed / 0 failed，28.49s，exit0**；本地/服务器14个指定实现与测试文件SHA一致。它覆盖最终的旧reason兼容性收尾。两次完整周期在该收尾前运行；该收尾只恢复旧配置分支的优先级，新profile分支未改变，未以相同运行重复堆积证据。当前仓库未配置Git remote，提交仅在本地，不能推送到不存在的上游。
