# P3统一修复效果：内环退出阻塞定位与后续修改方案（2026-10-03）

## 结论与证据边界

当前统一修复**尚未解决P3退化问题**。更精确地说，它没有通过学习前的A业务闭环验证，因此本轮没有启动PPO，不能提供更新前后学习效果的比较。高SOC下规则与冻结source都大量耗尽；新证据将一个直接阻塞定位到**V3退出轨迹先在占用内环大幅减速，与其他holder保持名义速度的执行约定不相容**。SafetyFilter正确拒绝了会发生真实间隔冲突的候选，不应降低安全阈值来通过它。

本记录由primary亲自读源码、分析、写诊断代码和文档；worker仅执行指定命令及报告产物。与[分阶段协议](p3_integrated_validation_20261003.md)共同构成本轮验证结论。下列数据均为newly_run；此前196项回归和3请求成功是preexisting_artifact，不能替代本次负载证据。

## A完整配对：没有训练也出现内环耗尽

配置为V3/90D、12请求/warm6、初始SOC .90–1.0、env seed890501/policy seed890601、最多1800ticks。source使用固定SHA `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`，仅冻结迁移。N/E/F/D/U为正常/应急/失败/转场/未决。

| 控制器 | N/E/F/D/U | 终态ticks | 最后正常完成s | 首次耗尽s |
|---|---|---:|---:|---:|
| rule | 5/0/7/0/0 | 1534 | 413 | 1401 |
| frozen source | 4/0/8/0/0 | 1537 | 358 | 1401 |

两臂error=null、censored=false、exit0；hard separation、collision、NMAC、operational buffer、joint fail-closed和checked/executed hash mismatch均0。所有F均为ring_inner的energy_depleted，未分配停机坪、未进入应急降落。两臂应急attempts=840/960，committed=0；pad候选级blocked_safety=2866/3272、blocked_energy=2174/2488，blocked_lz/dynamics=0。这不是“初始电量太低”或“PPO更新了坏动作”就能解释的现象。

A失败后未进入B/C，未改变SOC、电池、间隔阈值、动作权限或seed，也没有挑选成功子集。

## 同场景900tick重放：堵塞早于能量不足

`A_blocker_replay`按预定900ticks主动停止，status=CENSORED、error=null、exit0、5正常/7仍活跃。CENSORED是诊断上限，不是完整周期成功。产物包括900条blockers.jsonl及400/600/850环境快照；快照恢复下一tick的观测、reward、telemetry、outcome与SOC一致性测试通过。

- 600秒时7架全部在内环，速度均16 m/s、无committed plan，剩余能量801.56–933.96，SOC约.534–.623。
- 601–850ticks有145次i2f请求被拒绝，3×6候选的2610条理由全部是`predicted_holder_reachable_tube`；851–900ticks再有30次请求/540候选同样拒绝。
- 到900ticks，energy_reachability_negative_budgets=0；应急51次尝试，306个pad候选全部被安全检查拒绝，blocked_energy=0。
- 因此先出现持续退出冲突，之后绕圈消耗才使能量保护拒绝和耗尽增加。能量不足是这个区段的后果，不能用更大电池替代退出能力修复。

## 已证实的主要机制：在占用环上制动导致后机追近

源码位置为`TrajectoryGenerator._v3_continuous_inward_landing_points()`：先对Bezier离环路径求entry_speed_limit，必要时重复降低entry_cap；然后生成同半径的ring_hold制动前缀，最后才进入向内路径。normal final与emergency共用该构造。其他holder遵守`nominal_until_release`并维持16 m/s；这一约定使预测与执行一致，但不能保证正在减速的退出者仍有足够后方间隔。

在保存状态上重新生成原候选，不调用env.step、不改实际世界，只做相同SafetyFilter及独立解析圆周位置计算。解析参考为theta(t)=theta0+v/R*(t-now)，适用前提为本状态所有邻机无plan且处于steady nominal；脚本明确检查此前提。

| 静态探针 | 候选范围 | 全世界通过 | 单机通过 | 各候选对最近邻机的最小间隔范围 |
|---|---|---:|---:|---:|
| normal @600s | 7架×6可用pad×规则K3 | 0/126 | 126/126 | .06884–24.33473 m |
| emergency @850s | 7架×6可用pad | 0/42 | 42/42 | .00461–10.23168 m |

所有候选动力学校验和执行边界通过，技术错误为0；正常候选已通过原能量guard，应急候选最小剩余能量余量277.02。所有被拒绝候选均有**独立解析名义邻机位置下的采样点距离小于80m**，无需依赖保守插值裕量就能证明不满足间隔。tube角宽度为0，独立参考与Safety holder预测最大偏差低于4.9e-10m。因此这里不是“把holder未来随机调速包络估计过大”的旧问题。

normal前缀22.45–29.21s，离环入口速度2.53–5.65m/s；108/126候选对其最近邻机首次低于80m发生在同环制动前缀。emergency前缀22.47–29.38s，入口2.45–5.64m/s，36/42候选同样在前缀已不安全。

明确例子：600s、uid6→最近pad1、零残差，剩余能量801.56；原路线制动26.6827s至3.6935m/s。uid0在624.0s已与候选低于80m，630.6s的最近距离约.3031m。候选当时仍在半径199.92m、高度40m附近，空间上尚未脱离后机所在环。

范围限制：126不是整个学习策略proposal字典的穷举，只是当前规则K3的全部pad组合；不能据此断言所有可能时序策略都无解。完整source配对也失败，但本轮没有测定本机制对历次P3退化的全部因果份额。

## 一个经过安全校验的修复方向：先向内收缩，再减速

对上述**同一uid6、同一600s状态、同一最近pad1**做一个预定静态替代候选，不改邻机、不改safe_distance、不改变场景实际环半径：

- 以原角速度omega=16/200=.08 rad/s同步前进；60s内半径从200m平滑收缩至100m，高度保持40m。
- r(t)=200-100*h(t/60)，h(s)=10s^3-15s^4+6s^5；theta(t)=theta0+.08t。首尾径向速度和加速度为0，解析p/v/a用同一V3 quintic实现并经过全段bounds检查。
- 到100m构造位置后，复用原可执行landing tail做内部减速、进近和下降；100m仅为该候选构造参考，不修改actual airspace或邻机预测。

| 同状态同pad比较 | 原候选 | 向内收缩诊断候选 |
|---|---:|---:|
| Full-world SafetyFilter | 拒绝 | 通过 |
| 最小解析邻机间隔 | .3031m | 136.9663m |
| 动力学/执行入口错误 | 无 | 无 |
| 总时长 | 99.0897s | 105.8227s |
| 按immediate landing函数计算的所需能量 | 266.8114 | 273.5443 |

新候选保留余量约528.01，且不用降低安全标准。这是**一个状态的可执行替代路线证据**，尚未运行该候选的完整闭环，也没有正式接入normal/emergency生成器。60s/100m不是已验证的通用超参数。该prototype尚未定义完整speed/timing proposal映射、并发退出和其他环几何适用性，不能把它称为修复完成或直接投入PPO。

## 独立数值缺陷：近重复时间节点

首tick出现的`dynamics_trajectory_segment_mismatch`与上述内环堵塞是两个问题。对原初始状态uid0的o2m和uid4的m2i各三个K3候选进行6个无step探针，全部可复现：

- uid0第44段时长3.233e-13s，终端加速度与节点值最大偏差.49463；uid4第56段时长4.459e-13s，偏差.52012。
- position差为0、velocity差约1e-14；错误含`trajectory_segment_mismatch`和`trajectory_continuity`。
- `_v3_ring_hold_points_for_ramp()`仅对ramp_end超过duration时裁齐，ramp_end略小于duration时仍用精确set保留两个浮点节点，产生近零段；quintic加速度除以极小duration平方后出现数值病态。

建议在轨迹生成时将数值精度内重合的分段边界规范到同一**权威结束时间**，保留物理p/v/a与正常非零短段；不要放宽SafetyFilter的连续性或加速度容差。增加duration在ramp_end两侧相差数个ulp以及非零起始时间的回归。本轮只保留复现与建议，未改生产生成器；它也不能单独解释600s已全部到达内环后的126个真实冲突。

## 后续实施与验证顺序

1. 优先在opt-in V3路线中实现带连续p/v/a的向内退出段，让主要减速发生在获得径向净空后。normal与emergency共用可校验primitive，但仍分别经过原资源、能量、SafetyFilter与执行入口检查。参数从实际R、高度、当前速度和动态边界推导，不能直接把本probe的60s/100m推广到所有状态。
2. 同时修正近重复时间节点。保留旧V2路线与checkpoint历史含义，不关闭边界或安全检查。新诊断脚本已记录全部失败样本，供修复前后比较。
3. 保留政策的speed/timing意图：明确每个proposal实际影响的目标速度、出发时刻以及动力学clamp；调度reservation、检查轨迹和执行轨迹使用同一结果，不能在正式动作中偷偷加入训练未建模的fallback目录。
4. 用新可执行路线重新计算能量预算和应急期限。当前入场估计的solo路线+pad工作量不含拥挤退出走廊约束；如新退出段仍有拥堵，再在m2i入场/下游调度中保留退出间隔或通道。此项是设计建议，尚未用入场消融证明其独立改善幅度。
5. 先复现上述旧拒绝与新路线边界，再做原A完整规则/source配对；A通过后恢复既定B、C。不能以这个静态prototype代替闭环，不能通过增电池、降间隔或删失败结果把A变为PASS。
6. 环境闭环通过后才进行V3/90D数据适配与student-only qualification，再预注册有界学习更新和同场景前后比较。冻结88→90 initializer不能自动获得PPO准入；此时再判断gate-only entropy、credit等学习修复是否有效。

## 产物、复现与状态

输出根：`/home/user/training_outputs/terminal/p3_integrated_validation_20261003/`（Syncthing本地镜像`D:/7_codespace/training_outputs/terminal/p3_integrated_validation_20261003/`）。

| 产物 | 用途 |
|---|---|
| A_rule/result.json、A_source/result.json | 完整A配对，原失败保留 |
| A_blocker_replay/{result.json,blockers.jsonl,snapshot_*.pkl} | 有界重放和可信本地快照 |
| inner_exit_smoke、inner_exit_normal600、inner_exit_emergency850 | 单候选smoke及原候选范围核查 |
| transfer_boundary_probe.json、spiral_exit_prototype.json | 初次小探针，源文件随后并入正式诊断模块 |
| inner_exit_transfer_repro、inner_exit_spiral_repro | 合并模块的等价输出验证 |

持久代码：`terminal_marl/scripts/diagnose_p3_inner_exit.py`；kind为normal/emergency/spiral/transfer。normal使用snapshot_600、emergency使用snapshot_850、spiral固定snapshot_600；transfer直接构造已记录的初始fixture。只加载本次可信环境pickle，不接受不明来源pickle。输出目录必须不存在；全部是诊断，不是策略checkpoint。

例如在服务器项目目录使用`/home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.diagnose_p3_inner_exit --kind normal --snapshot <输出根>/A_blocker_replay/snapshot_600.pkl --output <全新目录>`。spiral/transfer为固定小探针，不扩大训练或场景矩阵。

normal600 result SHA=`fd6aed4998176d4668cfcf48852e3ad0b843192ae4a92e7a03baf1bf00bd5a6a`；emergency850 SHA=`5d080dfbad435019f59642adf1d0984f2f085b4bcc85db9b7d5508aa3b439540`；replay SHA=`16d22be7cc66a9a89cf8e5d0729ca544b4f37df9fae9bbcaa63cdedfcd4c1954`。

执行偏差保留：首次emergency命令输出目录日期误拼为20260303，重定向失败并exit1，没有启动对应probe或生成该结果；修正路径后才执行一次42候选probe。所有实际技术诊断输出无异常。

本轮已经具备“尚未解决”的完整A证据、具体阻塞的同状态证据和一个可行修复方向的小探针。生产轨迹实现仍未采用prototype，PPO_ALLOWED=false；无训练、held-out、硬件、checkpoint覆盖或默认promotion。完成最终文档/代码检查后结束本轮监控。

最终验证：诊断模块合并后定向26 passed（37.36s），spiral正式模块输出与原prototype除外层snapshot说明外逐字段相等，transfer 6/6 rows完全相等，两CLI均exit0/technical_errors=0。只有诊断模块、配套测试和docs发生本轮修改，生产环境/安全/轨迹核心未改。

终态清理：UTC2026-10-03 04:48:53服务器确认所有相关诊断进程均不存在，七个正式评估/重放/探针exit文件均为0。本轮已完成问题定位，heartbeat `p3`已由工具确认删除；保留全部结果和快照，不再自动启动评估或训练。
