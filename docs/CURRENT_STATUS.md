# 项目当前状态

> **状态日期**：2026-09-25
>
> **代码快照**：state-only 收集实现为 `768b191`，gate-adapter 训练器为 `b9f1077`，本轮 SOC gate 代码提交为 `a8ea93bf0e41f316fc161c1c9d1d7576395a8850`；本次状态/证据文档为后续文档提交。Git 无 remote，未 push。
>
> **历史阶段背景**：Phase A / A5。fresh fixed-rule、SOC `.75` 的 density six-cell 已完成，`failed_energy=0` 且命名 incident counters 为零，但不建立普适容量或零失败保证；fresh `.55–.90` late-slot pairs 仍未通过 `zero_energy_failure`，变体默认关闭且未晋级。singleton target-gap 修复已完成 `35` 个定向测试及四-cell paired evidence：fixed 与 legacy 的聚合分别为 `32 normal + 17 emergency + 11 failed-energy` 与 `26 normal + 23 emergency + 11 failed-energy`；该结果保留几何修复，但不宣称 zero-energy 已解决或已 qualification。此前固定 SOC `.75` density baseline 不作为该修复的验证。同版本性能链语义保持一致、wall-time 下降 `19.4%/18.6%`，但 100-step prefix 不是 qualification。fallback future route 尚未作未来 SafetyFilter 认证；`P3_QUALIFIED=false`、`checkpoint_promotable=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`。
>
**当前目标（2026-09-25，用户已恢复）：** 本轮目标已从“把 emergency 降到零”调整为“完成受控应急语义下的 P3 PPO 正式大规模训练准入准备”。高密度下安全认证、能量未耗尽并完成着陆的 `emergency_landed` 是一种带代价的降级结果，不单独阻止 PPO 学习；`failed_energy`、未认证/安全失败和 formal/evaluation `unresolved` 仍是硬失败。pilot 训练中的自然 horizon 截断单独报告，不把它自动写成普遍安全失败。P3 仍不应收敛为 SOC 单因素排序：保留 priority `1/3/5`、SOC `.55–.90`、任务/资源混合和 `downstream_priority`，继续报告 priority、low-SOC 与全请求等待的权衡。该目标只授权完成准入检查、受限 pilot 和正式大规模训练前的文档/证据准备；当前仍为 `PPO_ENTRY_ALLOWED=false`、`FORMAL_P3_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

新的准入边界见[受控应急 P3 V5 协议](evidence/v4_2/p3_controlled_emergency_admission_v5_20260925.md)。旧 P3 formal、Q3、canonical comparison 和 heldout 记录保持历史含义，不追溯替换为新语义；`829001–829006` 继续 untouched。V5 首先以不更新的随机策略 entry cohort 验证闭环，再以小规模真实 PPO pilot 验证至少一次被接受的更新、restore 和配对回归，之后才允许正式大规模 P3。该过程不重训 I0/I1/P1/P2；使用 mixed old anchors 做回归保护，避免从 I0 重新开始。

> 本文是滚动状态页。规范语义以 [README.md](README.md) 中列出的主规范为准。

**此前暂停点（2026-09-25，已解除）：** 当前代码迭代已完成：canonical strict loader 为 `29a4dba`，其 `32` 个定向测试通过；evaluator 的 `27` 个定向测试通过；只读服务器 checkpoint/hash/profile-load preflight 通过。暂停时 `848000/848101/848102` 尚未 reset 或 rollout；该句只记录暂停时状态。恢复后 `848000` smoke 与 `848101/848102` full pair 均已完成，结果仍只用于历史基线与修复候选的开发比较，不自动放宽新 V5 准入，也不消费 heldout。`P3_QUALIFIED=false`、`PPO_ENTRY_ALLOWED=false`、`FORMAL_P3_ALLOWED=false`、`P4_ALLOWED=false`。

**canonical pair 冻结结果（2026-09-25）：** `848101/848102` 的 canonical 与 repair 四个 arm 均 exit `0`、full-resolved、zero-energy 且安全硬计数为零；aggregate 为 canonical `31 normal + 17 emergency`、repair `40 normal + 8 emergency`。repair 在两组 priority-weighted wait 与 low-SOC wait 上均更优，因此 V5 entry/pilot 的冻结初始化来源选择 repair candidate E04：SHA256 `e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`，路径 `/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt`。旧 canonical `no_failed_outcomes` gate 的历史 false 保持不变；该开发选择不构成 P3 qualification。

**P3 V5 当前路线：** 固定 `80/40/10 m`、`energy_capacity=1500`、`min_landing_soc=.10`、`soc_reserve=.25`、`emergency_trigger=.40`、`late-slot OFF`、`K=3`、`Observation V2.1=88D`、`Teacher=false`、`downstream_priority`。先运行 `849000` smoke，再运行不更新的 stochastic entry `849101/849102/849103`，然后按顺序运行总计 6 次 update 的 pilot `849201–849206`（每个 registered arm 一次），最后做同初始状态的 paired development `849301/849302/849303`；paired development 的 before 是 pilot 前 baseline，after 是全部 6 次 update 后的结果。每个 paired record 内复用同一初始状态，不消费历史 heldout。这些新种子在协议冻结后才可消费；任何安全/软件硬失败停止后续注册 arm，普通性能结果继续保留并报告。正式大规模 P3 在 V5 pilot 和配对回归完成前不启动。

学生 rollout summary 已记录 `reward_component_totals` 与 `reward_total`，并验证组件和一致；
累计实际 `time/progress/release/touchdown/failure/teacher/shield/urgency/emergency` 字段，
排除 per-UAV `total` 避免重复计算。`urgency` 仍是 priority 与 SOC 的合并项，不虚构独立
分解；实现提交为 `4d34672`，定向测试 `9 passed`。

**当前 trained dense30 V2 bounded diagnostic（2026-09-24）：** 已完成首个有界训练策略诊断
`842101`：`1173` simulated steps、wall `64.53536491142586 s`，`8 normal / 18 emergency /
4 failed_energy / 0 unresolved`；actor `14877 WAIT / 1055 RELEASE`，`24` approved、`1031`
rejected。最小实际/扫掠间隔为 `80.00345665736643/80.0 m`，`safety_error=null`；collision、NMAC、
hard separation、shield/joint fail-closed、executed-plan hash mismatch 均为 `0`。记录了 `9` 次
shield intervention（correction，不是 failure）和 `1315` 次 safety intervention，不能概括为
“all safety counters=0”。weights unchanged、无 Teacher/optimizer、无训练/PPO；该诊断不是 PASS 或
qualification。`842101` 的 read-only trace aggregation 已完成：4 个 failed-energy UAV 的 threshold 前
RELEASE 为 `3/3/3/9` 且全部 rejected，WAIT 为 `411/305/493/238`；inner RELEASE 的主要观察
blocker 为 `primary_blocker=LZ_RESOURCE=593/615`（`510 emergency_lz_priority + 83 lz_unavailable`），
approved `8`。没有 independent feasible RELEASE witness 支持将 WAIT 全部判错或批准
imitation-only fix。详见[诊断结果](evidence/v4_2/trained_dense30_v2_diagnostic_result_20260924.md)。

**当前 random-SOC P3 slot-repair 结果（2026-09-25）：** 详见[结果页](evidence/v4_2/random_soc_p3_slot_repair_result_20260925.md)。
同一 seed `845101` 的 dense24 三 arm 为：late-OFF forced `23/1/0`、late-ON forced `19/5/0`、
late-ON trained `18/6/0`（normal/emergency/failed-energy）；三者均 `unresolved=0`、terminal、
exit `0`。三 arm 均无 `SafetyInvariantError`，collision/NMAC/hard-separation/fail-closed/
hash-mismatch 均为零，weights unchanged、Teacher/PPO off。`845000` late-ON 两步 smoke 有效；
server `29` tests PASS，local diagnostic `17` 与 slot `12` PASS，Teacher `12` PASS 单独统计。
旧 protocol、2026-09-24 日期与 raw root 保持不变；`P3_QUALIFIED=false`、`PPO_ALLOWED=false`、
`P4_ALLOWED=false`，旧 heldout 保持 untouched。

**当前授权/工作状态（2026-09-25）：** 不采纳 late-ON，默认 late-slot OFF 保持。外置
state-only labeler 已由 `768b191` 实现，gate_adapter-only 监督训练器已由 `b9f1077` 实现，
对应 data protocol 为 `56b4c3`；proposal head/encoder 冻结。服务器 smoke `846000` 已以 exit `0`
完成，`18` 条 actor labels 为 `7 WAIT + 11 RELEASE`，全部为 finite 88D 且 schema 映射正确，
weights unchanged；服务器指定的 `5` 个模块测试为 `39 PASS`，本地 `44` 个测试属于另一组，
不合并为同一测试总数。完整 `7` 回合 collection 已完成，输出根为
`/home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925`；详见
[collection/fit result](evidence/v4_2/random_soc_gate_repair_v1_result_20260925.md)。启动 wrapper 曾因
reset 前预建 output 被拒绝，归档为 `launch_failure_train_flow6_846101`；修正独立 `_control` 日志后，
首个实际 `846101` 回合已运行，不称为重复 seed 实验。授权的 gate-adapter-only fit 已完成 `35/35`，
artifact SHA256 为 `e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353`，train/dev
rows 为 `1529/812`；dev RELEASE/WAIT recall 为 `.6131687243→.7860082305` /
`.8119507909→.7768014060`，macro-F1 `.7092638611→.7562677855`，NLL
`1.265022278→1.031340837`，train NLL `1.259755254→.755264521`。仅 adapter 改变，
non-adapter/proposal logits bitwise unchanged；strict load passed、无 optimizer state、无 dev gradient/
early stop/PPO。wrapper 最后有 CRLF shell exit error，但 Python 已完成且 control 保存 Python exit `0`，
未重跑。`846301/846302/846303` paired rollout 已完成且三者 exit `0`；三场合计 `42` requests，
source 为 `36 normal + 6 emergency + 0 failed_energy`，candidate 为 `41 normal + 1 emergency +
0 failed_energy`。详见[collection/fit/pair result](evidence/v4_2/random_soc_gate_repair_v1_result_20260925.md)。
六个 arm 的 replay、safety、Teacher off、no updates、weights unchanged 均为 true；dense24
唯一 false arm gate 为 `no_failed_outcomes=false`，因此不称 P3 通过。整体 lifecycle aggregate 为
source priority-weighted wait `150.6971849500`、low-SOC wait `53.7159174128`（count `12`），
candidate 分别为 `80.1562018040`、`21.4881016766`（count `12`）；selection gates 全部 true。
exact emergency replay 与当前 `32` 个独立单步 shadow checks 已记录：均绑定 exact `795`-step
replay、state unchanged、无 safety error、approved `0`，其中 `safety_wait=16`、
`final_capacity=10`、`safety_reject=6`。这只说明抽查状态未获批，不宣称所有 WAIT 错误或 SOC
排序有效。`847101/847102` arbitration paired development 已完成，baseline 与 soc-priority
各自为 `41 normal + 7 emergency + 0 failed_energy + 0 unresolved`（48 requests）。两 pair 的
安全/完整性检查均通过，唯一 false arm gate 为 `no_failed_outcomes=false`；soc-priority 两 pair
均降低 low-SOC wait 但提高 priority-weighted wait，因此不采纳，默认 `downstream_priority`
保持。详见[arbitration result](evidence/v4_2/random_soc_arbitration_repair_v1_result_20260925.md)。
本轮 repair 来源是 `714d141f...`；历史 P3 canonical initial 是独立的
`4b8be66c...`，不静默替换基线。canonical strict loader、只读 preflight 与 `848000/848101/848102`
smoke/full rollout 已完成；pair 结果与冻结 E04 初始化来源见[受控应急 P3 V5 协议](evidence/v4_2/p3_controlled_emergency_admission_v5_20260925.md)。旧
`829001–829006` 保持 untouched。
`P3_QUALIFIED=false`、`PPO_ALLOWED=false`、`P4_ALLOWED=false`，不放宽既有安全与 P3 gate。

**P3 random-SOC scope（沿用现有 P3 V4 语义）：** initial SOC 为 `.55–.90` uniform，priority 为
`1/3/5`，场景覆盖 `flow6`、`flow12` 和 `dense24`。dense30 pressure 诊断不能单独构成 P3
qualification；后续资格证据仍需 train/dev/heldout 独立性。现有 P3 hard gates 保持原协议口径：
dev/eval completion `1.0`、safety、exact replay、K3 shortlist、low-SOC wait non-worse 和
priority-weighted wait improvement；本轮不新增 completion threshold。预留 `829001–829006` heldout
保持 untouched，不改作本轮开发集。

> **当前阶段更新（2026-09-24）**：固定规则协调流已完成三组 development
> regression（`841301/841501/841502`），均为 `30/30` resolved、`failed_energy=0`，
> normal/emergency 分别为 `29/1`、`29/1`、`28/2`；最小 touchdown SOC 为
> `.301878/.285780/.278703`，最小实际/扫掠间隔均达到 `80 m`，命名安全计数为零。
> 这是 opt-in fixed-rule development evidence，默认旧 route 保持不变，不构成 I1 或 PPO
> qualification。deadline-fallback bounded fresh check 的 `841701/841702` 均已完成并通过硬门槛，
> 两个 fresh case 合计 `60` UAV、`57 normal / 3 emergency / 0 failed-energy`；与三份 development
> regression 合计五场、`150` UAV、`143 normal / 7 emergency / 0 failed-energy`。此前 V7 fresh gate
> 的 FAILED 历史仍保留。`PPO_ALLOWED=false`。

**控制器身份澄清（2026-09-24）：** 上述 `fixed_rule_dense30_coordinated` 回放以及五场、
`150` UAV、`143 normal / 7 emergency / 0 failed-energy` 证据均使用规则控制器，不使用训练的
神经低层 checkpoint；启动时会打印所选模式、低层控制器类型和实际使用的 checkpoint。训练版
V4.2 回放位于 `ppo_checkpoint_viewer`，受冻结 profile/hash 约束；本文不宣称
`dense30_coordinated` 已通过训练策略回放验证。上述历史回放的 `min_landing_soc=.25` 是工程假设，
其文献依据是条件性的，尚未核验为 EH216-S AFM 的限制；见
[landing reserve basis](evidence/v4_2/landing_reserve_basis_20260924.md)。

**能量口径更新（2026-09-24，后续诊断适用）：** 归一化容量保持 `1500`，车辆 profile
profile 已更新为 `eh216s_research_v2`，`min_landing_soc=.10`；`soc_reserve=.25` 单独保留为
queue/admission reserve。其它安全、电量和运行阈值不变。已有 `.25` floor 的实验与 checkpoint
均保留为历史证据，不对 `.10` floor 作性能或 qualification 声明。本轮仅运行
profile/energy/checkpoint metadata 边界相关定向单测，全部通过；没有 fixed-rule 回合复跑或训练，
旧统计不更新为 `.10`。

**最新集成 fixed-rule development 回归（2026-09-24）：** 使用
`fixed_rule_dense30_coordinated` CLI，配置为 30 架 warm-start、初始 SOC `.55–.90`、route
guard on、`continuous_inward_v1`、`downstream_priority`、late-slot off，并启用 opt-in
`rule_flow_coordination_enabled=true`。三份 raw record 均 `30/30` resolved、
`failed_energy=0`、`safety_error=null`；collision、NMAC、hard separation、shield/joint
safety fail-closed、executed-plan hash mismatch 等命名计数均为零。

| seed | sim s | wall s | normal / emergency / failed-energy | min touchdown SOC | min actual / swept m |
|---:|---:|---:|---:|---:|---:|
| 841301 | 757 | 264.76303469995037 | 29 / 1 / 0 | 0.3018781902911374 | 80.1532184002751 / 80.1532184002751 |
| 841501 | 783 | 217.04460870008916 | 29 / 1 / 0 | 0.2857804017484627 | 80.0 / 80.0 |
| 841502 | 840 | 255.41621090006083 | 28 / 2 / 0 | 0.2787034160630745 | 80.0 / 80.0 |

原始 `841301` baseline 为 `18 normal / 8 emergency / 4 failed-energy`；协调回归为
`29 / 1 / 0`。在协调回归中，UID3 的 M→I commit/进入 I 环/normal touchdown 为
`140/236/299 s`，UID4 为 `168/252/327 s`；commit 时刻与进入 I 环时刻不同。
此前 baseline 的三个观察点保持为：早期 O 环候选主要出现
`dynamics_unreachable_merge_time`，552 s 时 UID7/23 在两个空闲 pad 下仍得到
`emergency_lz_priority` 且 `runtime_shortlist_candidates_checked=0`，以及旧的
emergency-before-normal 与 forced energy guard 可在 SOC `.451690/.474427` 触发应急。
后续 inner-queue guard 属于独立的新观察，由 deadline fallback 处理，不改写上述 baseline
事实。

`flow_repair_regression_delivery.txt` 记录 `241 passed`、`2` 个未改变的 frozen-config
source-binding failures（`terminal_marl/core/config.py`）；focused repair subset 的 `82`
项不并入该计数。`validate_plan` 的 `100`-step 等价测量为 scalar `45.319964 s` 到
vectorized `36.037378 s`，仅代表函数/prefix，不能外推为完整 GUI 或整体吞吐 wall-time
加速；三次 development wall time 也只作绝对记录。

deadline-fallback fresh check 已完成 bounded hard gates：`841701` 为
`30 normal / 0 emergency / 0 failed-energy`、`30/30` resolved、863 s、最低 touchdown
SOC `.3451866507009634`、最小实际/扫掠 `80/80 m`；`841702` 为
`27 normal / 3 emergency / 0 failed-energy`、`30/30` resolved、789 s、最低 touchdown
SOC `.2919970856931216`、最小实际/扫掠 `80/80 m`。两份 fresh 记录合计 `60` UAV、
`57 normal / 3 emergency / 0 failed-energy`，最低 fresh touchdown SOC
`.2919970856931216`；连同三份 development regression 的五场总计为 `150` UAV、
`143 normal / 7 emergency / 0 failed-energy`，整体最低 touchdown SOC
`.2787034160630745`，命名物理/安全错误计数均为零。该 bounded check 为 **PASS**，但此前
fresh V7 paired gate 的 **FAILED** 事实保留，当前不进行正式 I1/PPO qualification。

协调 30-UAV 可视化的 PowerShell 命令为：

```powershell
.\venv\Scripts\python.exe -m terminal_marl.visualization.rollout_viewer --mode fixed_rule --scenario fixed_rule_dense30_coordinated --seed 841301 --gui --labels --trajectories --playback-speed 1 --max-simulation-seconds 1800
```

**历史 baseline 可视化回放观测（2026-09-24）：** current-code 精确重放 seed `841301`、30 架全 warm、初始
SOC `.55–.90`、`continuous_inward_v1`、route guard on、`downstream_priority`、late-slot off，使用
`env.step()` 得到 `1364` steps、`18 completed + 8 emergency + 4 failed-energy`，与此前结果一致。
`t=813 s` 时 6 个 LZ 均为 free；UID `4/11/12` 均处于 inner `R=200, z=40`、speed `16` 且
`committed_plan=None`。该回放没有返环让出降落点轨迹证据；`emergency_priority` 只作用于尚未放行的
request，不撤销已经 committed 的 path。完整本地 trace 为 `results/viewer_trace_841301_full.json`
（约 149 MB，未提交）。813 s 的局部只读 probe 进一步显示 UID12 的 6 条 emergency candidate
全部被 `predicted_holder_reachable_tube` 拒绝；即使其他 holder 固定为 nominal 速度，UID4 nominal
swept 最小间隔按 pad `3/4/5/0/1/2` 仍为 `15.040/17.248/17.210/63.639/39.236/20.469 m`，
全部低于 `80 m`。该 probe 未执行候选、未放宽阈值，不能把该时刻归结为 tube 约束过严；当前候选
本身与后方 UID4 的环飞存在冲突。probe 文件为 `results/viewer_trace_841301_probe813.json`。

同一 trace 的 failed-energy 记录为 `UID3@883 outer`、`UID4@991 inner`、`UID11@992 inner`、
`UID12@1363 inner`，reason 均为 `energy_depleted`；`_resolve` 的 `self.now` 是处理 tick 起点，
因此下一 snapshot 时间为 `+1 s`，不把记录解释为 substep 精确瞬间。full vertical execution
trace 显示 `26/26` 个成功 landing plan 均实际 touchdown、`z=0`，pad XY 最大误差 `0`，每架
有 `15–36` 个真实竖降控制步样本且高度单调下降；vertical duration 为 `14.696–34.834 s`，
内环正常 final 约从 `40 m` staging 下降，外/中环 emergency 可从 `120 m` staging 下降。4 架
failed-energy 仍在环飞阶段耗尽，不能与成功落地后退出 active 混淆。UID12 在 `t=991/993` 的
剩余能量为 `372.1257/370.1257`、SOC 为 `24.808%/24.675%`，已低于 `375` reserve，按同一
约束不能通过降落 energy gate。viewer 修复范围已覆盖低 SOC 橙色、应急红色、failed-energy 持久
事件/标记、LZ 状态、顺序 source frame、长间隔禁用插值和正常结束保留最终画面；headless 不等待。
UI 联合回归为 `67 passed`，但 failed-energy 本身尚未修复，`PPO_ALLOWED=false` 保持不变。详见
[viewer flight diagnosis](evidence/v4_2/viewer_flight_diagnosis_20260924.md)。

本轮 PyVista/VTK 离屏 viewer 检查使用 PyVista `0.48.4`、VTK `9.6.2`：30 点云保持
30 个 vertex cells，真实 picker 可选中非零 UID `4/11/29` 并在收缩后保持映射，标签 actor
不可拾取；轨迹线宽为普通 `1.5`、选中 `3.0`，轨迹头跟随插值位置。30 Hz 的 position-only
更新复用了标签缓存和轨迹 mesh，单次中间帧只调用一次 plotter update。八个既有/专项测试文件
本轮合计 `35 passed`。临时 30 UAV 离屏渲染基准（预热 `10` 帧、计时 `60` 帧）记录 before
`3.245240299962461 s`、after `0.6678959000855684 s`；这是 offscreen render-only 测量，
不是仿真或 runtime FPS，也不据此宣称完整 GUI 或整体性能提升。
本轮未改变仿真控制器、能量阈值或轨迹执行语义；rollout 仅新增启动时 controller identity 输出。

**2026-09-24 综合结论（密度、late-slot 与性能）：** fresh fixed-rule、SOC `.75` 的六个密度
cell 已完整结束：12 机为 `24/24 normal`，20 机为 `37 normal + 3 emergency / 40`，30 机为
`46 normal + 14 emergency / 60`；六个 cell 均 `failed_energy=0`，并且
`safety_error`、`hard_separation_violations`、NMAC、collision、`shield_fail_closed`、
`joint_safety_fail_closed` 均为零。这些是 fixed-rule exploratory 结果，不建立普适空域容量上限，
也不建立零失败保证。当前证据显示静态环容量不等于 deadline service capacity；在该混合需求设置中，
输入间隔 `8 s` 小于 final headway `12 s`，若持续则会形成持续过载；此前 mixed 场景的有限 `18` 架 burst 只观察到
transient accumulation，不构成有限批次不可行定理。已有 future fallback energy budget
不是 future trajectory safety certificate；在没有经过完整认证的可行替代路径前，不能据此宣称
SafetyFilter 安全阈值过严。详见 [density efficiency evidence](evidence/v4_2/density_efficiency_20260924.md)。

原始 `.55–.90` fresh late-slot 配对的 off/on 聚合分别为 `36 normal + 19 emergency + 5 failed-energy`
与 `32 normal + 24 emergency + 4 failed-energy`；on 模式的 normal completion 下降、emergency 上升，
且 `zero_energy_failure` 仍未通过，因此 late-slot 变体未晋级，默认保持
`reconcile_late_merge_slots_enabled=false`。性能链四个 100-step prefix 的同语义比较为
`28.607→23.053 s` 与 `28.875→23.501 s`，即 `19.4%/18.6%` wall-time 降低；它们仍是未完成
`29` 架请求的 prefix diagnostic，不能作为 qualification 或容量结论，实际 GUI 也尚未视觉验收。
详见 [trajectory-chain performance evidence](evidence/v4_2/trajectory_chain_performance_20260924.md)。

singleton target-gap 修复已将同 holder point 的零 gap 改为 antipode gap；`35` 个 singleton
定向测试和本轮 combined regression 的 `103 passed` 均已完成。fresh `.55–.90` paired full cells
中，seed `841301` 的 fixed/legacy 分别为 `18 normal + 8 emergency + 4 failed-energy` 与
`12/13/5`，seed `841302` 分别为 `14/9/7` 与 `14/10/6`；聚合为 fixed
`32/17/11`、legacy `26/23/11`。该结果是描述性 paired evidence：保留几何正确性修复，aggregate
failed-energy 未减少且一个 seed 结果更差，不能宣称 zero-energy 已解决或进入 qualification。
该四-cell 证据使用 SOC `.55–.90`，不把固定 SOC `.75` density baseline 作为 singleton 修复验证。
详见 [single-holder gap evidence](evidence/v4_2/single_holder_gap_20260924.md)。保持
`PPO_ALLOWED=false`。

**最新前置观测（2026-09-24，runtime timing reconciliation）：** 只读 timing audit 显示，all-warm
seed `840626` 前 `200` steps 有 `21` 次 approved ring transfer，其中 `20/21` 的 `|end-slot|` 超过
`1 s`，median `-8.379 s`、minimum `-49.229 s`；mixed seed `840627` 前 `200` steps 为 `20/20`
mismatch，median `-8.661 s`、minimum `-41.292 s`。mixed actual arrival adjacent gaps 中，
ring_middle 最小 `1.97 s`（`5` 个低于 `9 s`），ring_inner 最小 `8.29 s`（`4` 个低于 `9 s`）；
prefix safety events 为 `0`。对应 timing audit 仍属于前置观测，不是 qualification。

product viewer 20-step offscreen benchmark 为 headless `2.918 steps/s`、process `2.820 steps/s`，
排除 `1.63 s` startup 后 slowdown `3.37%`，`12.01 fps`，p95 gap `0.192 s`，测试 `22 passed`；
该结果不能替代 GUI 主观验收。CPU profile 的 20-step 总耗时为 `13.18 s`，`ring_transfer` cumulative
`5.912 s`、runtime protect `3.086 s`、SafetyFilter `1.890 s`；profile 调用嵌套，数值不可相加。
timing reconciliation fix 已完成并通过 focused integration 与 actual slot audit；`PPO_ALLOWED=false`。

**最新 merge-slot reconciliation 回归（2026-09-24）：** `MergeScheduler` 现将 ring-transfer slot
对齐到 `plan.end_time`，将 final slot 对齐到已验证的 `scheduled_departure`；integration tests 为
`158 passed, 1 skipped`。mixed30 前 `200` steps 的 actual slot audit 为 `29/29` exact delta `0`，
o2m/m2i/i2f 最小事件间隔 `9/9/12 s`，`0` 个低于 headway，prefix safety events 为 `0`。
all-warm `840626` full-cycle 为 `17 normal + 10 emergency + 3 failed-energy`（step `1067`，
最小实际/扫掠 `81.578/81.550 m`，壁钟 `64.925 s`，`47` approvals）；mixed `840627` 为
`18 normal + 12 emergency + 0 failed-energy`（step `910`，最小实际/扫掠 `80.00062/80.00062 m`，
壁钟 `42.690 s`，`61` approvals）；两者 safety events 均为 `0`。相较此前 all-warm `20/8/2`、
mixed `23/6/1`，该修正改善了 slot timing 的安全与一致性观测，但 throughput/emergency gate 仍未
通过，不能作为 qualification 或训练授权；`PPO_ALLOWED=false`。SOC arbitration pilot 已完成，
但 `zero_energy_failure` 未通过，未晋级默认模式。

**最新 SOC-priority opt-in pilot（2026-09-24）：** `downstream_soc_priority` 保持 downstream ring
precedence，并在同一优先级内按自身 SOC 升序排序；不改变 observation 维度，默认仍为
`downstream_priority`。前 `200` steps 的 all-warm `840626` approvals `23→19`、resolved `5→4`、
墙钟 `26.423→27.693 s`；mixed `840627` approvals `29→28`、resolved `5→5`、墙钟
`13.841→13.420 s`。mixed full-cycle 中 default 为 `18 normal/12 emergency/0 failed`（step `910`，
`42.690 s`），SOC priority 为 `20/9/1`（step `1043`，`43.70 s`）；失败 arrival `17` 为
`UID71`、`ring_middle`、`t=1042`，两种模式 safety events 均为 `0`。该 pilot 因
`zero_energy_failure` hard gate FAIL，未晋级且不改变默认模式；没有 full all-warm SOC pilot 结论，
`PPO_ALLOWED=false`。counterfactual emergency contract probe 已完成，单状态结果见下方。

**最新 nominal-holder counterfactual（2026-09-24）：** 只读 current-code seed `840626` 的 `UID13`
在 `t=200 s`、`ring_outer`、SOC `0.4595651` 状态中，6 条 emergency routes energy `6/6`、
SafetyFilter `0/6`；唯一 free compatible `LZ4` 为 energy `1/1`、SafetyFilter `0/1`，原因计数为
`predicted_holder_reachable_tube=3`、`predicted_separation=3`。将 in-memory nominal-holder exact
trajectory 同时用于 lower/upper tube bounds 后，全部结果和原因计数保持相同。该结果只说明单状态下
nominal speed promise 单独不足以改变 verdict，不构成一般不可能性证明；未放宽运行时安全阈值、未修改
source，`PPO_ALLOWED=false`。

**最新诊断（2026-09-24，dense30 continuous final/LZ exploratory）：** 为检查 30 机潜在冲突场景，新增
opt-in `final_approach_geometry_mode=continuous_inward_v1`。该模式使用连续切向脱环的共享几何，保留
正常 final-approach 语义，并由中心扫描按兼容性排序的空闲 LZ；SafetyFilter `80/40/10 m` 保持不变。
默认 `brake_bezier_v1` 未改变。集成定向测试为 `121 passed`。在 seed `840626`、30 架全 warm、
high-SOC 的 100 s 探索中，该模式获得 `13` 次 approval、完成 `1` 架 normal，emergency、failed-energy
和 safety 均为 `0`；最小实际间隔 `82.03085 m`，服务器墙钟时间 `13.4638 s`（旧模式同条件约
`16.47 s`），但仍未完成完整回合。延长同一模式至终态后，运行在 `t=1117` 结束（服务器墙钟
`51.9834 s`）：`20` 架 normal、`8` 架 emergency、`2` 架 failed-energy、`0` 架 unresolved；失败目标
为 outer UAV `UID13@889 s` 与 `UID14@1116 s`。请求记账为真，累计 `51` 次 approval、`304` 次
emergency attempt/`8` 次 commit；最小实际/扫掠间隔为 `80.0331/80.0226 m`，hard-separation、
NMAC、collision 和 shield fail-closed 均为 `0`。该 full-cycle 结果未通过 `zero_energy_failure`
hard gate，不能进入课程或 PPO，`PPO_ALLOWED=false`。

本轮 Windows 早期 benchmark 的 `2.21 steps/s` 与服务器完整回合的 `21.49 steps/s` 不可直接比较：
后者后段 active UAV 已减少，状态计算量较轻，不能作为 30 架并发实时速率。实际 GUI 尚未完成视觉
验收。UID 级 emergency/release trace 已补齐；下一步前置是量化 merge-slot shortfall，完成安全规则/
应急路线修正，再冻结 fresh normal/stress protocol；在这些前置条件完成前不推进低层课程或 PPO。
详见 [dense30 低层预训练前置诊断](evidence/v4_2/dense30_pretraining_20260924.md)。

**最新归因补充（2026-09-24，dense30 UID trace）：** 同一 seed、同一 continuous 配置的完整回放与原始运行
保持 exact outcome parity：`1117` steps、`20` normal、`8` emergency、`2` failed-energy。`UID13` 首次
emergency 为 `t=290, SOC=.399565`，有 `120` 次实际尝试和 `14` 次 emergency plan commit 尝试，
其中 `8` 次被 holder tube、`6` 次被 predicted separation 拒绝；`UID14` 首次 emergency 为
`t=495, SOC=.414041`，有 `125` 次实际尝试和 `11` 次计划尝试，其中 `9` 次被 holder tube、`2` 次被
predicted separation 拒绝。两条失败终点均有 `6` 个空闲 LZ；UID13 的 `t=200/290` 和 UID14 的
`t=400/495` 窗口中，6 条 emergency routes 均 energy `6/6` 通过但 SafetyFilter `0/6` 通过，
能量 slack 为正（`25.556–177.007`）。因此当前静态 dense30 的 zero-failure gate 明确失败，
不能归结为 LZ 数量不足，也不支持放宽安全阈值。正常 25-proposal 扫描在 `t=100/200` 为
`dynamics_unreachable_merge_time` 全拒，在 `t=500/750` 为 `emergency_energy_guard` 全拒。
相关 forensic artifacts 见 [dense30 证据页](evidence/v4_2/dense30_pretraining_20260924.md)。

**最新 merge-slot 诊断（2026-09-24）：** seed `840626` 的 outer `UID13/UID14` 在 `t=100/200`
各扫描 `63` 个候选（speed residual `-1/0/1`、merge residual `-1/0/1`、delay `0/10/20/40/80/120/180 s`）。
`t=100` 两架均为 `0/63` preliminary certified，`t=200` 分别为 `13/63` 和 `21/63`；但最佳
planned endpoints 比请求 merge slots 提前约 `95–97 s`，因此不能作为可执行安全合流时隙证书。
严格 `<=1 s` alignment search 的 synthetic candidate 通过，但 12 机回归 `840620` 为
`1 normal + 8 emergency + 3 failed_energy`，`840622` 为 `1 normal + 10 emergency + 1 failed_energy`；
该改动已完全回滚，energy guard 定向测试 `8 passed`。详情见 [dense30 证据页](evidence/v4_2/dense30_pretraining_20260924.md)。

**最新 mixed30 探索（2026-09-24）：** seed `840627` 使用 `12` 架 warm-start 加 `18` 架每 `8 s`
到达、SOC `.55–.90`、continuous geometry 和 route guard，运行 `1032` terminal steps、服务器墙钟
`43.3962 s`。结果为 `23 normal + 6 emergency + 1 failed_energy`，`30/30` resolved，`68` 次 approval，
`174/6` 次 emergency attempt/commit，route checks `615`（其中 negative `143`），最小实际/扫掠间隔
`81.2511/81.2478 m`，safety events `0`；唯一 failure 为 arrival `17` 的 outer `UID44@t=1031`。
该回合 `20%` emergency 和 `1` 条 energy failure 使训练 gate 失败，不构成 qualification 或问题已解决
的证据。第二次 20-step 离屏测量为 headless `3.026 steps/s`、async viewer `2.883 steps/s`，开销约
`4.8%`，插值帧 `24`（前次 `3`）；GUI subjective QA 仍不可用。详见 [dense30 证据页](evidence/v4_2/dense30_pretraining_20260924.md)。

**最新 mixed30 UID44 归因（2026-09-24）：** UID44 exact replay 与 mixed30 原始回合一致：`1032`
terminal steps，`23 normal + 6 emergency + 1 failed-energy`，peak active `21`，peak ring
outer/middle/inner `9/4/4`；arrival `17` 的 outer UID44 在 `t=1031` 失败。首个 tracked inbound 为
`t=70, SOC=.63691`，outer 状态为 `t=130, SOC=.600799`，首次 emergency 为 `t=145, SOC=.590799`。
在 `t=130/145`，6 条路线均 energy `6/6` 通过但 SafetyFilter `0/6` 通过，energy slack 约
`+313–386`，normal proposals `25/25` 全拒；`t=750` 为 SafetyFilter `3/6`、energy `0/6`，
`t=900` 为 SafetyFilter `6/6`、energy `0/6`。`t=70`/`t=100` 仍是 inbound，未进行环上 emergency
scan，不能据此推断已经执行 admission check。早期 future-exit certification 与 coordinated
holder/pad traffic plan 仍是 proposed prerequisites，尚未实现或证明；`PPO_ALLOWED=false`。

**最新诊断（2026-09-24，dense30 低层前置与性能）：** 为覆盖 30 机潜在冲突场景，fresh
high-SOC all-warm seed `840626` 在 route guard、ring `600/400/200 m`、SOC `.55–.90`、
K1 `[1,0,0]` 下运行 `100 s`，`0` approved、`0` resolved、`30` unresolved，`3000` 次
release rejection；观察到 `0 failed_energy` 和 `0` hard-separation/NMAC/collision，但
该 horizon 未完成完整回合。25-proposal catalogue 在 `t=0/t=50` 均为 `750/750` rejected；
扩大到 `800/550/300 m` 后 100 s 仍为 `0` approvals。Windows 每步耗时已由 `1.06462` 降至
`0.451737 s`（约 `2.21 steps/s`）；异步离屏 stride1 检查为 headless `2.3685`、viewer
`2.297 steps/s`，约 `3%` 开销且无 queue backlog，实际 GUI 流畅性尚未由此证明。12 机
same-seed 回归 outcome/accounting 不变，高 SOC `12/12` normal，近下限 `6` normal+`6`
emergency，均无 failure/safety event。详见 [dense30 低层预训练前置诊断](evidence/v4_2/dense30_pretraining_20260924.md)。

**历史 exploratory 状态（2026-09-24，emergency reachability guard）：** fresh stress seeds
`840607/840611/840612/840613` 的 old/route 配对每对终态相同；两种模式分别聚合为
`13 completed + 27 emergency_landed + 8 upstream diverted_low_soc = 48`，均
`failed_energy=0`、安全事件为0。route-aware checks 非零，但没有改善这些终态。正常高 SOC
`.55–.90` 的 `840620/840621` 合计为 `23 completed + 1 emergency_landed`；近下限
`.41–.55` 的 `840622/840623` 合计为 `10 completed + 13 emergency_landed + 1 upstream
diverted_low_soc`；两组均 `failed_energy=0`、安全事件为0。全部记录为 exploratory，未通过
exact replay/full qualification，不能据此启动 PPO。V2.3 旧 source-bound observation-only
证据独立保留。低层 `12` 机课程与 `30` 机离线回放的先后顺序，以及是否先重构 `30` 机安全层
实时性能，当时仍待决定；现由上方最新的 [dense30 低层预训练前置诊断](evidence/v4_2/dense30_pretraining_20260924.md) 结论覆盖；当前不宣称可正式 PPO。

**最新实现补充（2026-09-23，source-bound Observation V2.3）：** 已加入保持 88D 宽度的
opt-in `phaseA_low_obs_v2_3_source_bound_88d`。自身状态、有限
`neighbor_comm_range_m` 内的 5 个邻机槽，以及由起降场中心广播的环/队列状态、紧急未分配
数量和兼容降落点释放 ETA 已接入；不增加观测维度，V2.1/V2.2 保持原 schema。当前实现采用
零延迟、无丢失的理想通信假设，后续仍需鲁棒性验证。新旧 observation/audit 定向测试共
`28 passed`；该实现说明不改变既有 evidence，`PPO_ALLOWED=false` 保持不变。

**最新实现补充（2026-09-23，warm-start 能量前置检查）：** 已加入默认关闭的
`warm_start_energy_reachability_check_enabled`。启用后，warm-start reset 会在每架 UAV
进入 `active` 前计算其当前 ring 到兼容 LZ 的最低单机可执行应急路线 SOC 要求；采样 SOC
低于该要求时直接报出 uid、ring、初始 SOC 和要求 SOC，不裁剪、不重采样，也不把 UAV
分流为上游请求。`.35` OUTER 样本被拒绝，`.41` 的三环小样本通过；默认关闭时历史
warm-start cohort 语义保持不变。这只是单机 reset 前置条件，不提供拥堵、等待或时间耦合
保证。独立 candidate energy/LZ budget 模块已覆盖 candidate 能耗、LZ release ETA、ring
hold 等待、应急 fallback route、landing reserve 与 margin；截至 2026-09-23，该模块尚未
接入 admission/release gate；截至 2026-09-24，admission pad ETA 与 candidate
route/等待/reserve/margin budget 已在 opt-in exploratory 链中接入。fallback future route
仍未作未来 SafetyFilter 认证。warm-start、budget、emergency 定向测试共
`24 passed`；`PPO_ALLOWED=false` 保持不变。

**最新结论（2026-09-23，emergency-priority guard V2）：** fresh `840501/840502` × rule/original K3 × guard off/on 的 `8/8` 条 formal records 均通过终态、请求记账、exact replay、安全计数为零和 Teacher-free；聚合为 `33 normal + 33 emergency + 24 upstream-divert + 6 failed_energy`，guard-on/off 分别为 `4/2` 次 failed energy，hard gate 失败，`PPO_ALLOWED=false`。只读 evidence 中，guard-on 的 K3 失败目标多次触发 `emergency_energy_guard`，但最后窗口 6 个 LZ 均 free；guard-off 的一个失败目标因长时间 `safety_wait`。V2 不晋级，guard 默认关闭。该 formal 运行使用的历史 `_warm_start()` 直接注入 `.35–.55` SOC；当前已加入默认关闭的单机可达性前置检查，详见上方实现补充。下一步是待实现的三段修复：计划承诺前的可达能量/时间预算与上游分流；有 abort/hover/返环证书、足够 SOC、原子切换和释放 LZ 条件时才允许安全抢占；规则/安全层通过后扩展共享 policy 的 deadline、pad 时隙和自身可达能量观测并加入非法动作 mask，区分 normal/stress、保持 `failed_energy=0` 硬门槛和 `PPO_ALLOWED=false`。详见 [Emergency-priority guard V2](evidence/v4_2/emergency_priority_guard_pilot_v2_20260923.md)。

**最新结论（2026-09-23，emergency-priority guard V1）：** opt-in guard V1 默认关闭。fresh `840401/840402` × rule/original K3 × guard off/on 的 `8/8` 条 formal records 均通过终态、请求记账、exact replay、安全、Teacher-free、无 labels/updates 和 K=3 检查；聚合为 `28 normal + 51 emergency + 16 upstream-divert + 1 failed_energy`，guard-on `zero_failed_energy` gate 失败，`PPO_ALLOWED=false`。rule `840401` 的 witness 显示固定 `.40` 能量投影切环硬阻断可能造成低 SOC UAV 长时间等待：guard-on 的 `o2m` release 连续 `647` 次被 `emergency_energy_guard` 拒绝并在 `t=647 s` 电量耗尽，而 guard-off 同一 UAV 在 `t=0` 获准切环并于 `t=274 s` 应急降落。Guard V1 不晋级，不宣称预留 LZ 或 PPO 能保证零失败；下一步只允许移除该硬阻断，设计带可达能量预算的应急替代路径，并以全新 seeds 验证，不能重跑 `840401/840402`。详见 [Emergency-priority guard V1](evidence/v4_2/emergency_priority_guard_pilot_v1_20260923.md)。

## 一句话状态

**最新实现状态（2026-09-23，应急路线 repair）：** 应急路线已由源环有限刹停改为从真实初始位置/速度连续切向脱环的 cubic 路径；OUTER/MIDDLE 进入 `120 m` staging，INNER 保持当前高度，再经 LZ 上方垂直下降，仍使用原 SafetyFilter 和 `80/40/10 m` 契约。第一 repair 协议（`e99cb...`）在 fresh `840101/840102` 上为 `7 normal + 14 emergency + 2 upstream-divert + 1 failed_energy`，安全事件为0，因此 FAIL；同 seed forensic replay 已使目标 UAV 在 `t=118 s` 应急提交并落地。route-repair 协议（`8c13dc...`）在 fresh `840201/840202` 上为 `9 normal + 11 emergency + 4 upstream-divert`，`failed_energy=0` 且安全事件为0，但结果仍是 rule-only 诊断证据，不构成 qualification 或真实 EH216-S 标定；当前 phase_motion_v2 未显式设置初始 SOC 范围时 default uniform floor 为 `.4106`，显式 `.35–.55` 仍保留为 stress；`PPO_ALLOWED=false`。详见[应急迫降与分阶段能耗 V2](evidence/v4_2/emergency_energy_v2_20260923.md)。

**最新结论（2026-09-23，SOC 优先级共享策略配对诊断）：** 在同一 `phase_motion_v2`、应急阈值 `.40`、`80/40/10 m` 安全契约和显式初始 SOC `.35–.55` 下，fresh seeds `840301/840302` 的 rule/original K3 四条记录均完成终态、请求记账、exact replay、K=3 shortlist、Teacher-free 和安全检查；安全事件总数为 `0`。合计为 `7 normal + 32 emergency + 6 upstream low-SOC diverted + 3 failed_energy`，因此唯一失败 hard gate 为 `zero_failed_energy`，aggregate `PASS=false`。seed `840301` 的全请求延误为 rule/original K3 `426.25/332.33 s`，seed `840302` 为 `326.17/328.92 s`；两颗种子仅作描述性配对证据，不能据此宣称 SOC 优先级已闭合或判断 failed-energy 根因。详细结果见 [SOC 优先级配对诊断](evidence/v4_2/emergency_soc_priority_paired_v1_20260923.md)。`PPO_ALLOWED=false`。

**最新归因（2026-09-23，immutable replay）：** `failure_trace_evidence.json` 的三个失败目标回放均 exact：rule `840301` arrival `2` 为 `609/609` steps，rule `840302` arrival `11` 为 `742/742`，original K3 `840302` arrival `9` 为 `751/751`；文件 SHA256 为 `a17820aa59bbbece91cc96d066981c20e4b42ba57e2edc317be23bec24da1a57`，witness 文件 SHA256 为 `1a93a063db4982946532c26516629e98ff0224c04bcfee51c112290642a03844`，且未新增 seeds。三者第一次 `SOC<=.40` 时刻为 `t=1/142/143`，均在 committed plan 中；当前 `_try_emergency_landing` 只考虑无 committed plan 的 ring holder。其后首个无 committed plan 状态分别是 rule `840301` 的 `t=70, SOC=.35922`（LZ `5/1` free，候选 energy margin `+37.1625/+20.6904`，下一步 `predicted_separation`、两候选均 rejected、`0` commit）、rule `840302` 的 `t=148, SOC=.39562`（occupied `0/3/5`、reserved `1/2/4`，下一步 `blocked_lz`）和 original K3 `840302` 的 `t=181, SOC=.37948`（occupied `0/2/5`、reserved `1/3/4`，下一步 `blocked_lz`）。三条 target tape 的 exact/reset/base/config/recipe 与 pre/post state hashes 均匹配。该代码观察不证明提前抢占长计划必然安全；全局计数也不能一般性归属于目标 UAV。formal 的 `39` 次成功 touchdown 中有 `32` 次 emergency，因此该 stress diagnostic 不足以证明稳健 SOC sorting。下一步改为独立预注册 low-SOC commitment gate/有证书的长计划安全 preemption 与 emergency pad/corridor capacity 的最小 repair，保持现有 SafetyFilter。

**最新结论（2026-09-22，P3 当前 MDP formal baseline）：** high-SOC qualification 的 6 条记录已完成落盘，但 rule seed `835001` 与 original K3 seed `835003` 各有 2 条 `diverted_low_soc`，因此 all-complete gate 失败；双方 controller 汇总均为 `70/72` 完成，安全、exact replay、Teacher-free、labels/updates 与 runtime shortlist gates 均通过。独立 low-SOC stress 的 6 条记录也已完成：rule 为 `71/72`、1 条低 SOC 分流，original K3 为 `62/72`、10 条低 SOC 分流；其完整性、安全与 replay gates 通过，完成率和分流按协议仅作报告。完整结果见 [P3 high-SOC 正式结果](evidence/v4_2/p3_current_mdp_baseline_result_20260922.md) 与 [P3 low-SOC stress 正式结果](evidence/v4_2/p3_current_mdp_low_soc_stress_result_20260922.md)。不授权 P3 PPO 或 checkpoint promotion；`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

**此前诊断结论（2026-09-22，高 SOC paired pilot）：** dense24 pilot seed `831002` 中，rule 与 original K3 均完成 `24/24`，diversion 与安全计数均为零，action tape exact replay；完成时间为 `686/740 s`，平均等待为 `99.5899/114.6562 s`。该回合仅作诊断，不是 qualification。它与后续 formal high-SOC seeds `835001–835003`、low-SOC stress seeds `835101–835103` 分开，不能替代 formal gate。

**最新结论（2026-09-22，P1/P2 当前 MDP 复评）：** 复评协议已冻结于 [`p12_current_mdp_requalification_v1.yaml`](../terminal_marl/configs/low_v4_2/p12_current_mdp_requalification_v1.yaml)，协议与 formal 结果见[当前 MDP P1/P2 复评协议](evidence/v4_2/p12_current_mdp_requalification_protocol_20260922.md)和[复评结果](evidence/v4_2/p12_current_mdp_requalification_result_20260922.md)。服务器 formal 已完成 `20/20` records：P1 seeds `834101–834104`、P2 seeds `834201–834206` 的 candidate/rule 配对记录均完成，safety、exact replay、terminal 与 hard gates 全部通过，shield actor fraction 为 `0`。P1 candidate/rule mean wait 为 `59.21658969142434/8.351137293207188 s`，paired delta `+50.86545239821715 s`；P2 为 `5.734572452355265/7.794466418817502 s`，paired delta `-2.059893966462237 s`。这些 paired 指标只作描述性报告，协议没有预设效率改善阈值；硬门槛通过不等于 PPO 授权。manifest SHA256 为 `57c50b926fc03a907a21c83738626a41b0c65077a7b4547d7f9ac43aaada80af`，`PPO_ALLOWED=false`、`P3_ALLOWED=false`。首轮 wrapper 因提前创建 output 目录在 reset 前失败，其 stub 与 log 保留；第二次正式运行未重复 seeds。

**最新结论（2026-09-22，P3 当前 MDP baseline evaluator）：** 新增 `terminal_marl/training/evaluate_v4_2_p3_current_mdp_baseline_v1.py` 与 `tests/test_p3_current_mdp_baseline_v1.py`；10 个定向测试和 compile 检查均通过。使用真实 original K3 SHA `4b8be...b8c8c` 的 high-SOC diagnostic seed `835000` 做 CLI 20 步 smoke，得到 rule/K3 两条 records；safety、exact replay、Teacher-free、labels/updates 与 K=3 shortlist gates 通过，`24` 架未全部完成属于 smoke 的预期，formal seeds 消耗为 `0`。正式 high-SOC qualification 与 low-SOC stress 均已按协议完成并逐 episode 落盘：high-SOC all-complete gate 失败，low-SOC `PASS=true` 仅表示其 integrity gates 通过，完成率和分流是报告结果。SSH 地址已修复，源码、协议和 checkpoint 哈希已核对一致，服务器 10 个定向测试通过。详见 [P3 当前 MDP 基线协议](evidence/v4_2/p3_current_mdp_baseline_protocol_20260922.md)。

**最新结论（2026-09-22，SOC 下限核验）：** 在当前 `80/40/10 m`、P3R 资源语义和 rule controller 下，完整单机 `gate(4) x mission(4)` flow-from-empty 矩阵 16/16 完成，touchdown 最大能量消耗为 `240.9`；按 `SOC=.25` reserve 与 `1500` 容量计算，单机名义最低初始 SOC 为 `.4106`。因此当前随机 `.55--.90` 下限高出 `.1394` SOC（`209.1` 个能量单位）；固定 `.55` replay 仍 16/16 完成并保持请求绑定 exact。历史 legacy_phase 的 `ArrivalProcess` 默认 fallback 下限为 `.35`；现行 phase_motion_v2 在 uniform 且未显式设置 `initial_soc_min/max` 时已改为 `.4106`，显式 `.35–.55` 仍仅用于低 SOC stress。该证据只覆盖无拥堵单机名义流程，不保证密集等待/交通可达性；本地命令退出码 `0`，不作为服务器 qualification。该单机证据本身不授权 PPO；当前动作以本页最新 formal baseline 与当前允许的下一步为准；`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。详见 [P3 单机 SOC 下限核验](evidence/v4_2/p3_solo_soc_floor_20260922.md)。

**最新结论（2026-09-22，覆盖下列历史时间线的下一步表述）：** 已实现目标方向的速度可行性修正；运行时首选 80 m，必要时在 swept + recursive 证明满足 40 m 保护下限的条件下由 shield 接管并继续执行，接管进入 reward 惩罚；collision event 门槛改为 `<10 m`。EH216-S 归一化能量参考改为 1500 s-equivalent，touchdown 要求 SOC≥.25。fresh dense24 seeds `830102/830103` × 间隔 `60/80/100 m` × rule/original K3 的 12/12 诊断回合中，288 请求、269 完成、19 低 SOC 分流；0 安全错误、0 `<40 m`、0 NMAC/collision、生成时的服务器同环境 12/12 exact replay（不宣称跨平台字节级一致）。80 m 的原始 K3 仅 41/48 完成，rule 47/48；本结果是诊断而非 qualification，不能启动旧 V4 PPO。P1/P2 旧结果不重写，但新 MDP 下需先 fresh 资格/效果复评，不能凭旧 gate 晋级；P3 需重新冻结能源可达的 dense24 资格场景与低 SOC 压力场景，再决定训练。详见 [P3 安全、能量与间距敏感性诊断](evidence/v4_2/p3_safety_energy_recalibration_20260922.md)。

**最新结论（2026-09-21，优先于下列历史时间线）：** P3 V3 urgency-shaped PPO 正式训练 12/12 回合、全部安全与更新 gate PASS，但 final checkpoint 12 在一次性 heldout dense24 seed `825006` 第684秒触发 `SafetyInvariantError`；heldout 停在 15/18，不能晋级。对这个已暴露 seed 的 13 个检查点只作诊断：original K3 与 checkpoint 1–11 安全，只有 12 失败；诊断 manifest SHA256 `e2be6b10f9cad30262bf68c3b9b0d57e73d9d5408f08cb7fe77f9e9f0ed40dfa`。事后选定 checkpoint 5，用完全新鲜的 Q3 seeds `826001–826006` 做一次配对确认：18/18 完整性与安全 gate PASS，但 seed `826006` 的 trained dense24 仅 `13/24` 完成（initial `24/24`），dense 聚合完成率 `.771<1.0`，故 Q3 `PASS=false`。manifest SHA256 `2ad3c6e52d52a2431f86284cd604911aed99329bbc9af897fbbb723548309e8c`。等待改善只在已完成 UAV 中聚合，不得遮蔽未完成；全部 825xxx/826xxx seeds 已消费，不能用于训练、参数选择或再次资格评估。详见 [P3 V3 与 Q3 证据](evidence/v4_2/p3_v3_early_stop_q3_20260921.md)。

将同一 Q3 dense24 回合延至 1800 秒后，trained 仍 `13/24`、末次 touchdown 第 380 秒，原始 K3 在第 911 秒 `24/24`，因而单纯延长上限不能解决停滞。新 V4 保持安全阈值和 1 秒决策步长，dense24 上限改为 1800 秒并采用“全部完成即停、到上限仍未完成则失败”；正式随机回合完成率仅作学习反馈，两个 fresh 确定性 dense24 开发回合须均 24/24 且安全、延误门槛通过，才可选择首个合格检查点。预留 heldout `829001–829006` 不在训练中评估。启动时一次 JSON 序列化错误发生于预约标记落盘后、首次 reset 前；严格核对标记与仅有扫描进度文件后已恢复，entry 2/2 PASS。但 dev baseline 的首个 fresh dense24 回合在第 533 秒、完成 12/24 时出现无可认证联合备份动作，进程于 exact-likelihood replay 阶段退出；formal PPO 更新 0 次、无检查点、无 heldout。见 [P3 V4 证据](evidence/v4_2/p3_v4_completion_guard_20260921.md)。

下文 2026-09-17–20 的“当前/下一步”句子是当时的历史状态，不覆盖以上终态；以后续本页末尾的最新允许动作与证据边界为准。

历史 P2 门槛：低层 V4.2 / Observation V2.1（88D）完成 R4 → I1 V2/Q2 → Safe DAgger → I3 → 初始 PPO 准入，并在 target-KL 保护下完成 P2 V4 与一次性确定性heldout。其后 V2→V5 on-policy support、learnability audit V2、anchored repair V3 与 fresh student-only Qualification V2 均 PASS；但当时 fresh paired effectiveness V1 仅关闭 baseline–rule 等待差距的 `46.01%`，低于预注册 `50%`，故该独立协议 **FAIL**、不授权从它直接启动 PPO/P3。后续 K3 独立 admission 和 P3 V3 训练已有新证据，不能用这条历史停止语句描述现在。已消费的 Qualification V1/V2 与 effectiveness V1 seeds 均不得重跑、reset 或用于调参。

初始 N3–N6 的 `PPO_ALLOWED=true` 是对应冻结版本及协议的历史准入。P2 V1/V2 累计 28 次合格更新；V3 在入口 KL 超限后保持历史 FAIL。独立预注册的 V4 重新资格与入口通过，并补足剩余 8 回合；这不改写 V1–V3 的历史终态。

2026-09-19 V4正式训练、确定性heldout、WAIT/credit诊断、support successor chain、Qualification V2、effectiveness V1、feasibility attribution、exact-proposal 与 set-valued proposal target chain 均已终止，当前没有遗留服务器进程。历史 PASS/FAIL 均保留；effectiveness 终态见[配对效果评估](evidence/v4_2/ppo_p2_gate_effectiveness_v1_summary.md)，fresh gate/proposal 归因见[可行性归因](evidence/v4_2/ppo_p2_gate_feasibility_attribution_v1_summary.md)，proposal target 闭环见[条件提案目标链](evidence/v4_2/ppo_p2_proposal_target_chain_summary.md)。

2026-09-21 完成 P2R 密集降落流资源修复：历史协议继续使用 lowest-ID、插入顺序和 final
容量1；`p2_dense24` 显式改用 nearest-available、downstream-first 和 final容量3。冻结 original
K3 candidate 在 seeds `809006–809009` 的300秒配对诊断中，touchdowns `23→30`、service
completions `17→22`，每个 seed 最大并发 final `1→2`，新语义 LZ assignment regret 全为0，
全部 collision/NMAC/hard-separation/uncertified-execution 计数为0。seed `809010` 反事实扫描中，
108个内环资源机会时刻仅4个存在“已单机认证但策略未请求”的 RELEASE，故当前问题主要属于资源
语义而非 gate 全面失效；未据此启动新 PPO。证据见
[P2R dense landing-flow V1](evidence/v4_2/p2r_dense_landing_flow_v1_summary.md)。

同日完成 `p2_dense24` 仿真与可视化性能修复。无GUI基准证明主要瓶颈位于递归安全预测，
而非单纯PyVista绘制：20步 `env.step` 的cProfile累计时间由 `15.726 s` 降至 `8.052 s`
（`-48.8%`）。修改只消除递归投影对不可变轨迹数组的深拷贝、复用单步内完全相同的已提交
航迹，并为仅需位置的碰撞审查使用等价的轻量插值；安全阈值、预测时域、采样间隔与动作语义
均未改变。同检查点/seed的放行计数、最小间隔与全部安全遥测逐值一致。viewer现在只显示“当前
真实位置 + 尚未执行的计划点”，按UAV稳定着色，选中航迹加粗，并在plan/topology变化或结束时
正确重建/删除actor；P2 viewer默认 `frame_stride=2`、每条航迹最多12点。完整证据见
[P2 dense visualization/performance](evidence/v4_2/p2_dense_visualization_performance_20260921.md)。

2026-09-20 冻结 Observation V2.2 candidate：保持 V2.1 的 88D byte-identical prefix，追加 6 个
scheduler/reservation/同一步竞争上下文特征，形成 94D schema；runtime 仍为单一 intent + 单次
SafetyFilter validation，不引入 catalogue search。V1 fresh pilot 完成 6/6 episodes，但发现
`interface_committed_transfers_norm` 因 reservation stage 与 trajectory phase key 错配而恒为零，故 V1
只作诊断。修复后的 V2 使用全新 seeds；前三个 train episodes 完成，seed `800004` 在 1570 s
仅完成 10/12，末态两机均停留 `ring_middle/m2i`，安全计数为零且 exact replay PASS。V2 保持
immutable FAIL，未运行 dev/probe、未复用 rows。该失败证明 complete-cycle hard gate 混入了 frozen
behavior 的 reachability 要求，不能归因于尚未参与执行的 V2.2 feature。已预注册 V3：fresh
`801001–801006`，固定 1570 s/terminal stop，12/12 arrivals generated 仍为 hard gate，completion
仅报告；其用途只限 88D+proposal 与 94D+proposal 的配对表示诊断，不能授权 repair/qualification。

V3 formal 随后 6/6 episodes、72/72 requests 全部完成，安全与 exact replay hard gates 全 PASS；
train/dev 为 247/106 rows，manifest SHA256
`20cb9d6ac2377aa5bb3e20c87707405f4aa553eecadae0581ac64a0631cc8211`。94D 相对
88D+proposal shared MLP 的 ROC-AUC 三个 seed 分别提高 `.0255/.0340/.0444`，但 infeasible
recall 为 `+.0456/+.0743/-.0695`，macro-F1 为 `+.0161/+.0326/-.0010`，state Top-1 为
`-.0094/+.0094/+.0849`，不构成一致的 observation-only closure。新增特征中
active-reservation 与 committed-transfer 完全同值，future-reservation 恒零，predecessor 仅 6/353
rows 非零，进一步说明 6 个全局 scalar 未充分表达 proposal-specific joint feasibility。下一步仅允许
探索性比较固定 94D shared scorer 与 `94D→25 independent feasibility logits`；后者仍只做一次前向、
选择一个 proposal 并交 SafetyFilter 验证一次，不改变 runtime 边界。

25-logit exploratory audit 已在冻结 V3 artifacts 上完成，manifest SHA256
`9d07f8cb2de43c0edf1f5875aedd9966d72b83df911d31a21996ba5566a43d06`。相对 94D
conditioned shared scorer，infeasible recall 三次提高 `.0552/.0456/.1295`，macro-F1 三次提高
`.0142/.0052/.0361`；但 ROC-AUC 三次下降 `.0054/.0278/.0205`，state Top-1 为
`+.0283/-.0377/-.0566`，因此不允许 architecture freeze。为避免继续使用同一 dev 调结构，V4
一次性冻结 fresh `802001–802006` 与 probe seeds `802101–802103`，使用
`0.5 balanced BCE + 0.5 feasible-set NLL`；每个 probe 必须同时满足 feasible recall≥`.85`，且
infeasible recall、macro-F1、AUC、state Top-1 相对 fresh 94D conditioned baseline 均不下降。
只有 3/3 probes 全 gate PASS 才允许冻结架构；无论结果如何都不直接授权 policy repair。

V4 formal 已完成，6/6 episodes、72/72 requests、安全与 replay 全 PASS，train/dev 236/145 rows；
manifest SHA256 `1983d5dcbb4cc7ceea28bd02a47ccb7829c33d3e8514a2c53bc86f8f1416270d`。
Hybrid probe seed `802101` 全 gate PASS，但 `802102` 的四项相对指标失败，`802103` 的 AUC 与
state Top-1 失败，aggregate `all_three_probes_pass=false`，architecture freeze 保持 false。此结果
触发预注册停止条件：不再使用 V4 dev 调整 learned proposal head/loss。下一步改测 proposal source
边界：gate 仍取 anchored repair V3 checkpoint 的 deterministic argmax，RELEASE 时固定使用 catalogue
index 12（speed/merge residual 均为 0，minimum-intervention 首选），随后只调用一次 SafetyFilter；
该路径无 Teacher、无候选枚举、无训练更新。fresh paired evaluation 使用 flow12 seeds
`803001–803006`，与原 candidate 及 nominal rule 对比。

fixed-neutral V1 虽完成 18/18 records 且表面 decision gate PASS，但 action-tape 复核显示 candidate 与
fixed controller 的 requested gate/proposal 序列在六个 seed 上逐步完全相同；candidate 中仍出现
非 neutral proposals。这证明 `student_rollout_v4_2.student_request()` 直接调用
`policy.model.distributions()`，绕过了 V1 对 `act_details()` 的覆写。V1 manifest
`15bd56cf439be13796a292ee1a745a683e4c8c9632466667676f25c597b3cf35` 保留为
**INVALID_IMPLEMENTATION**，其 PASS 不得使用。V2 已把 proposal distribution 固定逻辑移到 model
boundary，保持原 gate logits/critic 不变，并新增 action-tape hard gates：全部 fixed RELEASE 必须为
proposal 12，且 candidate/fixed 请求序列必须存在实际差异。V2 使用全新 `804001–804006`，不重跑
或复用 V1 seeds/metrics。

Corrected fixed-neutral V2 随后使用 fresh `804001–804006` 完成 18/18 paired records；实现 hard
gates 证明六个 seed 的 candidate/fixed 请求序列均有差异，且 fixed controller 的全部 RELEASE 均为
proposal 12。三种 controller 均 72/72 requests 完成、安全/回放/likelihood 全 PASS、shield 为零。
fixed-neutral 相对 candidate 的 request-to-touchdown 改善 `4.3611 s`，但 weighted mean wait 从
`13.0303 s` 增至 `17.5845 s`（delta `+4.5542 s`），违反预注册 non-worse gate，故 aggregate
FAIL、architecture freeze=false。manifest SHA256
`df7ab01ef8a828189300d2e79e4f2073b2de8fe0937ae4f0b47faa010cbedac0`。
按停止规则，不再用本轮 dev 调固定 proposal index，也不再追加 learned head/loss。该结果把下一步
限定为显式改变 runtime boundary 或回退 rule-only；用户已在随后选择并授权前者，历史 K=1 停止
结论仍保留，不因新架构候选而改写。

用户随后显式授权 K=3 bounded shortlist certification。实现保持一次 policy forward：确定性
RELEASE 的原请求为 proposal Top-1，再附带两个最高分且不重复的候选；runtime 按序最多认证三项，
只执行首个通过的 canonical trajectory。fallback 记为 runtime safety selection，不记 Teacher、
不扣 Teacher penalty，requested likelihood 仍绑定 rank 1；ActionTape 已覆盖 shortlist/rank/check count
并纳入 exact replay。fresh protocol
`ppo_p2_gate_runtime_shortlist_evaluation_v1.yaml` 冻结 smoke seed `805000`、formal seeds
`805001–805006`，比较 candidate K=1 / candidate K=3 / rule。正式 hard gates 包括 safety、completion、
replay、likelihood、每次最多三候选、fallback 非零，以及 K=3 wait/RTT 均不劣且至少一项严格改善。
V1 smoke seed `805000` 完成预定 160 s：服务器 43 tests PASS，三 controller 的 safety、exact
environment replay、requested likelihood 与 shortlist contract 均通过；短 horizon 未完成 12 架，按
smoke 非授权语义保持 `INCOMPLETE`。V1 formal 在 seed `805001` 的 K=1 与 K=3 后 fail-closed 停止：
两者均 12/12 完成且安全、trajectory replay 通过，K=3 提供 38 个 shortlist、实际选择 4 次 fallback、
每次最多检查 3 项；但旧 `_policy_tape_logprob_exact` 在复核时只回放 requested action，没有把 tape
中的 runtime shortlist 交回环境，导致第一次 fallback 后复核世界错误并把 likelihood 标成 false。
因此 V1 是 **INVALID_EVALUATOR**，不是 policy/runtime hard-gate FAIL；`805001–805006` 已消费，
不重跑、不复用其指标。

复核器现与 exact environment replay 一样恢复 `runtime_proposal_shortlists`。fresh V2 successor 已
预注册 protocol SHA `67e61165ef5c056d1b1ebd0a789cafd17f5e0554d07b29128617df3aa8bf6d5d`，
使用 smoke `806000`、formal `806001–806006`，其他 controller、checkpoint、边界和 hard gates 均不变。
V2 formal 已完成 18/18 records、三 controller 各 72/72 requests；全部 safety、completion、exact
replay、requested likelihood、Teacher isolation 与 shortlist contract hard gates PASS。K=3 六个 seed
均实际选择 fallback，每次最多检查 3 项；相对 K=1 的 weighted wait 为 `-9.519080 s`，
request-to-touchdown 为 `-2.055556 s`，两项均改善；K=3 的 wait `11.303291 s` 也优于 rule
`19.883828 s`。manifest SHA256
`6050088d3c28d1531c7e8dde5b828403390bfdeb51dde8bde54e0f2759323905`；正式结论见
[K=3 runtime shortlist V2 summary](evidence/v4_2/ppo_p2_gate_runtime_shortlist_v2_summary.md)。
`RUNTIME_SHORTLIST_FREEZE=true`；按预注册 `PPO_ALLOWED=false`，下一步只允许独立 admission review。

独立只读 admission review 随后全 gate PASS，review SHA256
`a549ffc60c35cb5ac1403ba0c4b59412096b2936d4a209ddb1a25219e09e8783`；它只授权 fresh entry
protocol 预注册，未授权直接训练。fresh entry V1 已冻结 seeds `807001/807002`、flow6/flow12、
160/256 steps 与 K=3 runtime。为保持 PPO requested-action ratio 与实际 shortlist 的一致性，entry
只允许更新 `gate_adapter + critic`，冻结 `actor_encoder/gate_head/proposal_head/temperature`，并把
更新后 shortlist 排序逐状态不变设为 hard gate；任一 update/排序/恢复 gate 失败即事务回滚并保持
`PPO_ALLOWED=false`。协议 SHA256
`d3400bf47d0f7f0d86ce4ed16c5b547b6ab30e0220691eb1f33eef320af932be`。entry 随后
2/2 records 全 gate PASS：26/26 optimizer steps 接受、0拒绝；proposal 子图 byte-exact 不变，
shortlist 排序稳定，安全/replay/likelihood/restore 全 PASS。`admission.json` SHA256
`8d425da16345dd0546851d83bfe706877dd4e6298437b8ef9bd91da324175db3`，因此
`PPO_ALLOWED=true`；entry checkpoints 仍禁止作为 formal 初始化。详见
[K=3 PPO entry V1](evidence/v4_2/ppo_p2_runtime_shortlist_ppo_entry_v1_summary.md)。
formal V1 已独立预注册：fresh training seeds `808001–808006`，按 flow6/8/12 重复两轮并运行
完整 cycle；heldout `809001–809006` 已冻结且训练阶段不可访问。formal 仍从原始 candidate 以
fresh optimizer 启动，只训练 gate adapter + critic；协议 SHA256
`7e6cf765af93925ffe960ff2ab8a4f141e13c1101e8b2f933fd64a2296bf3048`。formal 已完成
6/6 episodes、52/52 requests，354/354 updates 接受、0拒绝；全部 safety/geometry/replay/
likelihood/frozen-shortlist/restore gates PASS。manifest SHA256
`3c1bc69a518f7e5d99bb606d98b0356397e86f97b493947a0f14b0c8b49b010b`，final checkpoint SHA256
`714d141f7b472a17321e95e9c4c5574fdc3bcaa0c2911494f67b5b1a9f27592e`。详见
[K=3 formal PPO V1](evidence/v4_2/ppo_p2_runtime_shortlist_formal_v1_summary.md)。heldout V1 已按
trained K3 / initial K3 / rule 三控制器预注册，协议 SHA256
`eee2543918bf80610e80ccf11efdac602369d3bb46dd2226ce53ed87d57950e6`。heldout 18/18 records
全部 completion/safety/replay/likelihood/shortlist gates PASS；trained K3 相对 initial K3 的 wait
改善 `0.033386 s`，但 RTT 恶化 `0.019231 s`，违反预注册双指标 non-worse gate，故 immutable
`PASS=false`、checkpoint 不可晋级。两种 K3 均显著优于 rule，说明失败是 PPO 无一致增益，而非
K=3 runtime 边界失效。manifest SHA256
`f44acf5e5c78925b71202e0bb81ce4c2c56d302a51fd207025b2228714a666ce`；详见
[K=3 paired heldout V1](evidence/v4_2/ppo_p2_runtime_shortlist_heldout_v1_summary.md)。
独立只读 selection review 的 9 项 gate 全 PASS，review SHA256
`5617c96255c040797abf2d2f44ed81ce53be955ed94d3a8447a5c943b2f59969`。结论为：保持 heldout
FAIL，拒绝 formal checkpoint，停止追加 P2 PPO，选择 original K3 candidate；仅
`P3_PROTOCOL_PREREGISTRATION_ALLOWED=true`，`P3_ALLOWED=false`、`checkpoint_promotable=false`。
据此已冻结 P3 V1：保持固定几何与 N6/N8/N12、60 s 到达间隔及 K=3 runtime，只把 priority 从
固定 1 恢复为随机 `1/3/5`，SOC 从固定 `.8` 改为可完成区间 `[.55,.90]`；qualification seeds
`810001–810006`，entry `811001/811002`，formal `812001–812006`，并保留 untouched heldout
`813001–813006`。entry 更新不得作为 formal 初始化，formal 仍从 original K3 + fresh optimizer
启动。协议 SHA256 `ea5efc064f763b39cd18f07f55e84f07754aebbe40d8ded588b9b61998559b25`。

fresh P2 gate support V1 因 seed772009 出现 3/959 次 shield 而保持 FAIL；其 rows 不得训练。随后
V2→V5 successor support chain 在不混入旧 support/qualification rows 的前提下完成：V3 为 15/15
安全 episodes 但 train/dev 仅 1,150/908 rows；V4 追加后为 27 个安全 episodes、combined
3,786/1,787 rows，仅 dev cardinality 少 13；V5 dev-only 追加 3 个安全 episodes 后冻结
train/dev 为 3,786/2,415 rows，全部 support gates PASS。V5 manifest SHA256 为
`100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960`，train/dev rows SHA256 为
`40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4` /
`da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b`。其后 learnability audit V2
PASS、anchored repair V3 PASS、Qualification V2 PASS；完整链条见[汇总证据](evidence/v4_2/ppo_p2_gate_support_repair_qualification_summary.md)。

## 2026-09-17 切环轨迹局部修复

checkpoint12 演示确认 m2i 中段逆环向并进入目标环内侧，根因为旧 cubic 仅约束端点切向。已完成局部修复及[固定回归验证](evidence/v4_2/transfer_geometry_repair_20260917.md)：3场景12/12完成，实际切环无逆环向、安全0、回放一致；UID6最小切向速度16.02m/s。历史 PASS/FAIL 不变，但不外推到新生成器；**修复后资格与入口复核已通过，第二个 PPO block 已完成**：24回合96/96架完成，安全/逆环向/shield全0。详见[续训结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。

## 当前冻结的主线语义

| 项目 | 当前值 |
|---|---|
| 研究范围 | Phase A，landing-only；Phase B/C 仅为未来路线 |
| 代码主线 | `terminal_marl/`；`src/` 只读参考 |
| 空域 | `three_ring_v1`：OUTER / MIDDLE / INNER |
| 研究机型 | 当前为 `eh216s_research_v2`；`eh216s_research_v1` 仅保留为历史 checkpoint/provenance |
| 动力学 | `performance_constrained_v2`；`physics_dt = 0.2 s` |
| 低层观测 | 当前合格策略仍为 `phaseA_low_obs_v2_1_retry_state` 88D；`phaseA_low_obs_v2_2_scheduler_context_94d` 为未晋级 candidate |
| 低层动作 | `low_action_schema_v4_2_hierarchical` |
| 低层分布 | `WAIT/RELEASE` 二分类 gate + RELEASE 条件下 25-way proposal |
| 执行解码 | 1 个 canonical WAIT + 25 个 `speed × merge-time` RELEASE 意图 |
| 高层动作 | `high_airspace_v1_10d`，当前训练关闭 |
| 安全权威 | TrajectoryGenerator → SafetyFilter → PredictiveSafetyShield |
| Teacher 角色 | 训练期 expert labeler / auditor；不是部署控制器或 PPO behavior action |

低层完整语义见 [reference/low_level_policy.md](reference/low_level_policy.md)。

## 里程碑与证据

| 里程碑 | 状态 | 说明 / 证据 |
|---|---|---|
| Phase A 核心环境与 rule pipeline | 已进入后续训练使用 | 具体接口见主规范；历史迁移过程不再作为状态源 |
| D0–D7 动力学 qualification | PASS | [正式报告](evidence/dynamics/dynamics_qualification_report_2026-08-28.md) |
| P0 仿真性能 qualification | PASS | 结果摘要保留在 [历史升级计划](history/plans/CRAZYFLIE_DYNAMICS_LAYER_UPGRADE_PLAN.md) 末节 |
| C0 capacity calibration | PASS | 作为 user-provided frozen evidence 记录于 [V4 migration summary](evidence/v4/v4_new_mainline_migration_summary.md) |
| V4.2 hierarchical S0 | PASS | [S0 qualification](evidence/v4_2/v4_2_s0_qualification_summary.md) |
| Observation V2.1 final freeze | PASS | [final freeze v2](evidence/v4_2/v4_2_v2_1_final_freeze_decision_v2.md) |
| Fresh V2.1 I0-R | PASS | [I0-R qualification](evidence/v4_2/v4_2_v2_1_formal_i0_r_summary.md) |
| Fresh multi-seed I1 | FAIL | RELEASE manifold/context generalization gap；[qualification](evidence/v4_2/v4_2_v2_1_i1_formal_qualification_summary.md) 与 [attribution](evidence/v4_2/v4_2_v2_1_i1_failure_attribution.md) |
| I0-R2 repair collection | FAIL | 未闭合所需 support；协议和结果保存在 `evidence/v4_2/` |
| I0-R3 repair collection | FAIL（历史） | [R3 result](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md) |
| I0-R4 repair collection | PASS | [正式结果](evidence/v4_2/v4_2_v2_1_i0_r4_collection_summary.md) |
| I1 Qualification Protocol V2 / Q2 | PASS | [正式结果](evidence/v4_2/v4_2_v2_1_i1_v2_summary.md)，Q2 1,787 rows，三 seed 全通过 |
| I2 Safe DAgger | R1 PASS / I3 PASS | [I3 协议与 R1 结果](evidence/v4_2/v4_2_v2_1_student_qualification_protocol.md)，976 新事件、35 epochs、全部 gate PASS |
| 初始 PPO 历史准入 | **PASS（冻结 N3–N6 配置）** | [准入结果与训练命令](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，正式 train 模式已接受最新 admission |
| P2 自然流 V3 | **FAIL（entry KL）** | 资格 6/6 PASS；entry seed763001 的 `approx_kl=0.11257042959332467>0.05`，未执行 formal train；[持续课程终态与诊断](evidence/v4_2/continuous_curriculum_20260917.md) |
| P2 自然流 V4 | **训练 PASS / 效果审查 BLOCKED** | target-KL 资格6/6、入口2/2、训练8/8 PASS；heldout 18/18 PASS，但等待55.11s仍高于rule 16.18s；[正式终态](evidence/v4_2/ppo_p2_v4_training_evaluation_summary.md) |
| P2 WAIT/credit诊断 | **FAIL** | fresh stochastic 12/12 records；candidate50/52、baseline48/52；false-WAIT 83.21%/80.76%，安全与shield全0；[诊断终态](evidence/v4_2/ppo_p2_v4_wait_credit_diagnostic_summary.md) |
| Gate forgetting离线审计 | **COMPLETE / 不支持单纯PPO遗忘** | I3 false-WAIT 82.85%，checkpoint24 83.11%，V4 80.37%；I3并未解决P2状态支持缺口 |
| P2 gate support V1 | **FAIL（shield gate）** | 15/15 records、train/dev覆盖通过、安全0；seed772009有3/959次shield，V1零容忍gate失败；rows不得训练 |
| P2 gate support V2 | **PASS** | train 78/78、dev 52/52；5,627/2,009 rows；manifest SHA `09402508bc02155c1369de477dc320d5ca596768d4de6461f4a0526f2c7d9304`；仅进入后续离线审计 |
| P2 gate support V3 / V4 / V5 successor | **V5 PASS** | V3 15/15 安全但 train/dev 1,150/908；V4 后 combined 3,786/1,787（仅 dev cardinality 少13）；V5 dev-only 追加3个安全 episodes，最终 train/dev 3,786/2,415，全部 gates PASS；V5 manifest `100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960` |
| P2 gate learnability audit V2 | **PASS（离线）** | 3/3 fresh probes；manifest SHA `5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679`；support/dev 通过，unanchored adapter probe 在 D1 降级；无 environment/Teacher/formal policy training |
| P2 gate repair V1 / V2 | **FAIL（历史）** | V1 non-finite loss；V2 仅 support-dev NLL `.969>.50` 失败，其他主要 metric 与 D1 anchor gates 通过 |
| P2 gate repair V3 anchored | **PASS** | protocol `f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8`；manifest `e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410`；仅 gate_adapter 六参数变化，T/proposal/critic/common state invariant |
| P2 gate calibration V3 | **PASS（离线）** | fixed `T=2.5`；无 training/optimizer/environment/Teacher；仅授权 fresh student-only qualification |
| P2 student-only qualification V1 | **FAIL（历史终态）** | formal seeds `774001–774006`；原 false-WAIT / RELEASE recall gates失败；seeds 不得重跑或调参 |
| P2 student-only qualification V2 | **PASS（当前终态）** | formal seeds `782001–782006`；6/6、52/52、621 actor rows；sampled false-WAIT `18/206=.0873786`、false-RELEASE `16/415=.0385542`；仅新授权 P2 effectiveness evaluation |
| P2 paired effectiveness V1 | **FAIL（该协议历史终态）** | formal seeds `788001–788006`；18/18 controller episodes、各52/52请求；全部 hard gates PASS；等待 gap closure `.4600799557<.50`，其余效果 gates PASS；该协议本身不授权 admission review/PPO/P3，后续独立 K3 路线另计 |
| P2 flow12 feasibility attribution V1 | **COMPLETE（诊断）** | formal seeds `789001–789006`；6/6、72/72请求、1,050 actor rows；89次 rejected RELEASE 中66次为 Teacher也判RELEASE但proposal不同，74次发生于后半到达；仅支持转入fresh proposal support/learnability设计 |
| P2 exact-proposal support V1 | **PASS（数据）** | train/dev seeds `790001–790009` / `791001–791006`；540/455 rows；15/15 episodes、180/180请求；仅授权离线exact-label learnability audit |
| P2 exact-proposal learnability V1 | **FAIL** | anchored residual probes fresh-dev Top-1约`.433`、Top-3约`.577`，D1 Top-1约`.962`；未授权repair |
| P2 exact-proposal failure attribution V1 | **COMPLETE** | standalone MLP Top-1 `.580–.600`、Top-3 `.712–.732`；分类为`observation_or_label_learnability_limit_supported` |
| P2 proposal feasible-set diagnostic V1 | **COMPLETE** | fresh seeds `794001–794003`；36/36请求；166个requested-RELEASE状态，Top-1/Top-3可行107/123，27个状态无可行RELEASE |
| P2 proposal-set support V1 | **PASS（数据）** | train/dev seeds `795001–795009` / `796001–796006`；463/372 rows；15/15、180/180请求；全部episode/split/catalogue gates PASS |
| P2 proposal-set learnability V1 | **FAIL** | 3个D1-anchored probes整体Top-1约`.777–.780`，partial-set Top-1约`.478–.484`；Top-3/D1通过，Top-1与o2m失败 |
| P2 proposal-set failure attribution V1 | **COMPLETE** | standalone linear仅1/3全gate PASS，MLP 0/3；分类为`observation_or_set_target_learnability_limit_supported`；不授权repair |
| P2 proposal-conditioned feasibility V1 | **FAIL** | 88D+proposal参数的binary MLP feasible recall约`.90`，但 infeasible recall `.454–.462`、AUC `.793–.810`，o2m与全部综合gate失败；不授权factorized architecture或repair |
| Observation V2.2 closure V3 | **COMPLETE（诊断，不闭合）** | 6/6 episodes、247/106 train/dev rows；94D 的 AUC 3/3 提高，但 negative recall、F1、Top-1 不一致；observation-only repair 不授权 |
| V2.2 25-logit feasibility head exploratory | **COMPLETE（不闭合）** | negative recall/F1 3/3 改善，但 AUC 3/3 下降、Top-1 仅1/3改善；architecture freeze 不授权 |
| Observation V2.2 hybrid confirmation V4 | **FAIL（架构）** | collection 6/6 PASS；hybrid 仅1/3 probes 全 gate PASS；学习式 proposal 分支停止，不再基于 V4 dev 调参 |
| Learned gate + fixed neutral proposal V1 | **INVALID_IMPLEMENTATION** | rollout 绕过 `act_details()` adapter；candidate/fixed action tapes 完全相同；表面 PASS 作废，803xxx 不重跑 |
| Learned gate + fixed neutral proposal V2 | **FAIL（效果）** | 18/18 records 安全完成且实现 hard gates PASS；RTT改善4.36s，但平均等待恶化4.55s，aggregate gate FAIL |
| K=3 bounded runtime shortlist V2 | **PASS / FROZEN** | fresh `806001–806006`；18/18 records、216/216 requests；全部 safety/replay/likelihood/contract gates PASS；K3 相对 K1 等待改善9.52s、RTT改善2.06s |
| K=3 PPO admission review V1 | **PASS（只读）** | review SHA `a549ffc60c35cb5ac1403ba0c4b59412096b2936d4a209ddb1a25219e09e8783`；仅授权 fresh entry protocol 预注册，不直接授权 PPO |
| K=3 PPO entry V1 | **PASS / PPO_ALLOWED** | fresh `807001/807002`；26/26 updates 接受、0拒绝；全部 safety/replay/likelihood/frozen-shortlist/restore gates PASS；entry checkpoints 不得作为 formal 初始化 |
| K=3 formal PPO V1 | **PASS / HELDOUT ALLOWED** | fresh `808001–808006`；6/6、52/52完成；354/354 updates 接受、0拒绝；全部 hard gates PASS |
| K=3 paired heldout V1 | **FAIL（效果）** | 18/18 records hard gates PASS；trained vs initial wait `-0.033386s`、RTT `+0.019231s`；formal checkpoint 不可晋级，seeds 不重用 |
| P2 candidate selection review V1 | **PASS（只读）** | original K3 被选为下一阶段候选；formal checkpoint 淘汰；`P2_PPO_STOP=true`；仅授权 P3 protocol 预注册 |
| P2R dense landing-flow semantics V1 | **PASS（诊断/资源语义）** | seeds `809006–809009`：300秒 touchdowns `23→30`、completed `17→22`、并发final `1→2`、LZ regret归零且安全计数全0；seed `809010` confirmed gate miss `4/108`；不授权新 PPO 或 checkpoint promotion |
| P3 priority/SOC + dense24 PPO | **V3 formal PASS，final heldout FAIL；早停 Q3 FAIL** | V3 12/12 训练 gate PASS，但 checkpoint12 heldout 安全不变量失败；checkpoint5 事后选择，fresh Q3 18/18 安全/实现完整性通过而 dense 完成率 `.771<1.0`；全部检查点不晋级。见 [P3 证据](evidence/v4_2/p3_v3_early_stop_q3_20260921.md) |
| A6 高层动态空域训练 | BLOCKED | 需要低层 static-geometry qualification 后冻结低层策略 |
| Crazyflie 实物验证 | FUTURE | 仅在主仿真策略冻结后进入 small-N 缩尺验证 |

## 保留的历史失败：I0-R3

I0-R3 的 provenance、Teacher determinism、exact replay、Safety、partition independence、
aliasing 与 nominal anchors 均通过，但正式结果仍为 FAIL：

- complete-cycle 停止产生 4,541 个训练 rows 和 1,511 个内部审计 rows，超过冻结上限
  4,200 / 1,400；
- proposal-14 的 N=5 支持只有 12 个独立 lifecycle，低于最低 15；
- shared gate false-negative、shared proposal error，以及 post-WAIT、transition/retry、
  post-conflict、proposal-11、proposal-14 的 radius coverage gate 全部未达预注册阈值。

R3 仍为 FAIL；后续通过的是独立预注册的 R4，不能追溯升级 R3。完整数值见
[I0-R3 collection summary](evidence/v4_2/v4_2_v2_1_i0_r3_collection_summary.md)。

## 当前进度与结果

| 阶段 | 已核验进度 | 结论 |
|---|---|---|
| 初始 PPO block1 | 12回合，48/48架完成；checkpoint12独立评估24/24架完成 | 安全与闭环 gate PASS |
| 轨迹修复后 block2 | 24回合，96/96架完成；checkpoint24配对评估双方各24/24架完成 | 全 gate PASS，但候选相对 checkpoint12 的完成时间差为0/0/+12/0/+9/-2秒，没有明确改善 |
| P2 可达性 pilot | N6/N8/N12，rule/student各52/52架完成 | 可达性通过；N12 student等待65.66/73.91秒，rule为19.07/26.88秒，效率差距仍明显 |
| P2 V1 | 前23次更新合格，第24回合未更新 | initial ratio 检查误差超限，历史 FAIL |
| P2 V2 | 从 checkpoint23 恢复后5次更新合格，第6回合未更新 | 连续轨迹内侧偏差超限，历史 FAIL |
| P2 V3 | 新生成器资格6/6通过；entry仅执行1/2 | seed763001 的 KL 超限，formal train 未启动，历史 FAIL |
| KL 双种子诊断 | fresh 765001/765002；恢复 Adam 与清零 Adam 配对 | 仅诊断；未复现原失败，不支持清零 Adam 作为修复 |
| KL 更新保护 | 已提交并接入独立 V4 协议；资格6/6、入口2/2、训练8/8 | 468/468 formal optimizer steps 接受，0拒绝；最大 accepted rollout KL 0.006534<0.01 |
| P2 V4 heldout | 755001–755006 已一次性消费；candidate/baseline/rule 各52/52完成 | candidate 等待55.11s，优于baseline 65.02s但落后rule 16.18s；效果审查不通过 |
| P2 WAIT/credit fresh diagnostic | 771001–771006 已消费；12/12 records生成 | `DIAGNOSTIC_COMPLETE=false`；candidate50/52、baseline48/52；false-WAIT和team credit失配 |
| P2 gate support V2 | 775001–775009 / 776001–776006 已消费；78/78、52/52请求完成 | train/dev 5,627/2,009 rows；support V2 PASS，仅进入离线 learnability |
| P2 gate support V3 → V4 → V5 | successor seeds 已按协议消费；V3 15/15 安全，V4 27 个安全 episodes，V5 dev-only 3 个安全 episodes | 最终冻结 train/dev 3,786/2,415，V5 全部 support gates PASS |
| P2 gate learnability V2 → repair V3 | support V5 后已完成；learnability V2 与 anchored repair V3 均 PASS | learnability 3/3 fresh probes；repair 仅 gate_adapter 六参数更新，D1 anchor 与非 adapter state 保持不变 |
| P2 student-only qualification V1 | 774001–774006 已消费；6/6 episodes、52/52 UAVs、983 actor rows；recovery仅补写manifest | **FAIL（immutable）**：sampled false-WAIT 0.333333、RELEASE recall 0.625786；不得重跑/调参 |
| P2 student-only qualification V2 | 782001–782006 已消费；6/6 episodes、52/52 UAVs、621 actor rows；formal manifest/marker 已落盘 | **PASS**：sampled false-WAIT 0.0873786、false-RELEASE 0.0385542；概率 macro-F1 0.9532325、WAIT recall 0.9590361、RELEASE recall 0.9563107、NLL 0.1034486、Brier 0.0314997；仅 effectiveness evaluation 新获允许 |
| P2 paired effectiveness V1 | 788001–788006 已消费；candidate/baseline/rule 各6 episodes、52/52请求 | **FAIL**：candidate等待15.9619s、baseline17.3097s、rule14.3801s；等待 gap closure 46.01% 未达50%；request-to-touchdown与全部hard gates通过 |
| P2 flow12 feasibility attribution V1 | 789001–789006 已消费；6/6 episodes、72/72请求、1,050 actor rows | **COMPLETE**：66/89 rejected RELEASE 为 Teacher-RELEASE + proposal mismatch；late half 占74/89 rejections；o2m/m2i分别贡献55/11个该类事件；无训练授权 |
| P2 exact-proposal target chain | support 15/15 episodes PASS；learnability 3个probe FAIL；failure attribution COMPLETE | exact canonical Teacher proposal 在fresh dev上不可稳定学习；未保存或修改策略 |
| P2 proposal feasible-set / support | diagnostic 3/3 episodes COMPLETE；support 15/15 episodes、180/180请求 PASS | train/dev 463/372集合标签rows；Top-1可行率351/463与265/372；仅进入离线audit |
| P2 proposal-set learnability / attribution | anchored audit 3/3 probe FAIL；standalone linear/MLP各3 seeds attribution COMPLETE | Top-3、D1保持与接口aggregate可过，但partial-set Top-1未闭合；repair/PPO仍禁止 |
| P2 proposal-conditioned feasibility | linear/MLP各3个probe，11,575/9,300 train/dev state-proposal pairs | **FAIL**：MLP 0/3；o2m、negative recall、macro-F1、AUC与state Top-1未闭合；旧closure-bank可分性未泛化到fresh flow12 |
| P3–P5 / static qualification | P3 尝试后未合格 | Q3 dense24 吞吐失败；P4/P5 与 A6 高层动态空域均未获授权 |

## 已识别的问题与处理状态

1. **奖励与等待目标冲突已修正，但效率改善仍不足。** 旧 progress 奖励计入绕环距离：外环22m/s约每秒+0.022，高于时间惩罚-0.01。P2配置将该系数置0，物理、安全及其他奖励保持不变。V4 heldout 已验证 candidate 等待相对 checkpoint24 baseline 下降9.91s，但仍比rule高38.93s；修正有效果，不足以认定收敛。
2. **初始动作概率检查存在批次浮点差异，已局部修复。** V1 seed754024 的误差为1.1444091796875e-5，超过1e-5。原action tape只读复放表明完整world批次误差0，actor子批次复现误差。检查器现保留原采样批次再mask，阈值和PPO行为不变；后续V2/V3初始ratio检查通过。
3. **轨迹节点检查漏掉连续线段内切，已局部修复并通过V3资格复核。** V2 seed757006实际内侵0.0592194m，超过0.05m。UID39见证曲线节点半径均合格，但线段最近点半径199.9275204m。修复使用每段投影最近点，仅超限旧曲线进入已有forward-annular fallback，保持有效旧曲线和原安全容限；此前38项定向测试通过。V3资格最大内侵0.0186652m，安全/shield/逆环向均0。
4. **V3 的 PPO 更新漂移已由独立 V4 协议闭合，但历史 FAIL 不变。** V4 每个 optimizer step 后检查完整 rollout requested-action KL，超限或非有限值时事务回滚模型与 Adam；原0.05 gate不放宽。资格、入口和8回合训练均通过，formal 468次更新全部接受，最大 accepted rollout KL 0.006534<0.01。该结果恢复了独立 V4 的训练准入，不追溯升级 V3。
5. **P2 gate support 的 distribution successor 与 Qualification V2 已闭合。** support V3/V4/V5 逐步补足 calibrated candidate 自身状态分布，最终 train/dev 3,786/2,415 且全部 support gates PASS；learnability audit V2 3/3 PASS，anchored repair V3 PASS。Qualification V1 的 `774001–774006` FAIL 保持 immutable；Qualification V2 的 `782001–782006` 以 6/6、52/52、621 rows 和全部 episode/safety/provenance gates PASS 终态 PASS。两组 seeds 均不得重跑或用于调参。
6. **fresh attribution 将主要剩余瓶颈定位到 proposal/执行可行性。** 6个全新 flow12 episodes 全部安全完成；1,050 actor events 中 false-WAIT/false-RELEASE 为43/23，而89次 rejected RELEASE 中66次发生在 Teacher同意RELEASE但student proposal不同的事件，且这66次无一proposal match；其中55次在o2m、11次在m2i。late arrival half 占74/89 rejections与52/66上述proposal mismatch rejects。i2f 的6次reject则全部对应false-RELEASE。
7. **单标签、可行集合和动作条件化三条proposal监督路线均未闭合。** exact-label audit 与独立failure attribution先后失败；完整25-proposal catalogue显示166个requested-RELEASE状态中107个Top-1可行、123个Top-3可行，说明单一Teacher tie-break并非正确监督目标。fresh set support 463/372 rows全部gate PASS，但D1-anchored set probes的overall Top-1仅`.777–.780`、partial-set Top-1仅`.478–.484`，o2m仍失败。去掉D1 anchor和旧logits后，standalone linear仅1/3全gate通过，MLP 0/3。再将每个状态展开为25个`88D+proposal` binary样本，MLP feasible recall虽约`.90`，infeasible recall仍仅`.454–.462`，AUC`.793–.810`且o2m失败。因此当前证据支持88D信息/多UAV同一步竞争或target factorization限制，不支持直接proposal repair。
8. **课程覆盖仍不完整。** 历史 P2 N6/N8/N12是总请求数，N12 pilot在空峰值仅6–7架，不能称12架并发或高负载验证。P3 已加入 priority `1/3/5`、SOC `.55–.90` 和 dense24 warm+flow，但 Q3 吞吐失败；仍未合格覆盖间隔扰动、20/40/60总请求、持续/突发流和合法静态几何泛化。离最终终端区场景仍有明确跨度。
9. **文档参数及运行交接需保持一致。** 需求文档仍有旧高度/速度数字；当前验证配置为三环高度100/70/40m、速度22/20/16m/s，不能静默切换。后续长任务按用户要求由`luna_worker`跟踪、主agent静默等待终态。Git无remote，代码仅经Syncthing同步，不能声称已推送。
10. **密集可视化中的三个资源问题已闭合。** nearest-available 消除了可避免的 LZ 距离后悔；downstream-first 防止上游插入顺序抢占同一步认证；final容量3允许安全过滤后的并发进近。环空闲不再直接解释为低效：seed809010 的95个“有名义资源但未请求”时刻中只有4个存在 individually-certified RELEASE，另外91个受航迹/headway/安全约束。环容量是上限而非占满目标。

## 模型、评估与证据边界

- 历史 P2 V4 候选：`p2_resume_v4_20260918/train/checkpoint_8.pt`，SHA256 `5e6811b8349d875fb007943476aec62d93550a74ffaf4ae46614c5c188ab004f`。其训练与 heldout hard gates PASS，但该协议因 rule 效率差距未晋级；后续 P3 从另一条 original K3/anchored repair 路线开展，不能混淆模型血缘。
- 配对 baseline：`ppo_block2_transfer_repair_20260917/train/checkpoint_24.pt`，SHA256 `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`。
- V4恢复起点：`p2_resume_v2_20260917/train/checkpoint_5.pt`，SHA256 `8a8466f61119f7f5b5f780249d26f218ee9b8ae3e4aededc5c7f5fe2e51b201b`。它是V2失败训练块中的最后合格中间模型，现仅作为V4 provenance，不是当前最终候选。
- V3 root/entry manifest SHA256：`2098af1c3109c17fc489b7c9f20a9db11eb67aa62b77b5a03beb2de66909cd7d`；KL诊断manifest：`41017e10f45105a6125213ac51a630bdcfda9c35bac196608e2affc891955f69`。结果位于输出根目录下 `p2_resume_v3_20260918` 与 `p2_v3_kl_diagnostic_20260918`。
- checkpoint24评估seeds710001–710006与P2最终评估755001–755006均 **已消费**，不得重跑或用于调参。V4 heldout manifest SHA256为 `167d22cb29d8ef6d08275cb0ff4cbe270c0d9005611a176b85d81fd64730f104`。
- WAIT/credit diagnostic seeds771001–771006亦 **已消费**；manifest SHA256为 `b290e2d5f69749e1109bff2c5f348458732a22fbe8c0ce34c15ce98880eefc93`，终态FAIL不得重跑或升级。
- P2 gate support→learnability→repair→qualification 链的完整证据见[汇总](evidence/v4_2/ppo_p2_gate_support_repair_qualification_summary.md)：最终 V5 support manifest SHA256 为
  `100448260298b13cb33ce301267a44f9503a9cc1baea5c93f6121bff22fae960`，train/dev rows SHA256 为
  `40002db70c2e0d5d181b53b2f23290c65954f6bbe6282be1066b6f558e6153d4` /
  `da106e0d862c8e435748514abb10c4e0e5c500fb3bbd73f03cbfa6aa186a261b`；learnability audit V2 manifest SHA256 为
  `5a72549d05c4733c2163512bf30cfe28d601c38e3ddfe23dc8291ec2f17b3679`。
- Anchored repair V3 protocol/manifest/checkpoint SHA256 分别为
  `f8a84632fea21e6789c8a40c0a22938cb5f0bdfcdb4ec9efd5e48304867778e8` /
  `e1e0883e355676593ab4d561aeb46fb37656629ec514ad6f01e12c3bbd17d410` /
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`；仅 gate_adapter 六参数改变，T/proposal/critic/common state invariant。
- Qualification V2 protocol SHA256 为 `7ab9e2a84b9aea8da2ea975d98b4867de4a45bd5725bc72019b2e95ddbb75094`，formal
  `782001–782006` 已消费；manifest SHA256 为
  `7cd4f02c88dc953fe1545c613424e8c961c295d4f8c099456d6697f300088feb`，marker SHA256 为
  `43fafcd99e22d48bffb6c2389767a265c2376b05f95a070ae3d823f4066949d9`。形式化 rollout 的 source runner
  hash 为 pre-fix `b91e0d3794495bb768cf80ab266f1584b0af01addd2b0998c9c07e30af8d4a82`；CLI serialization 修复提交
  `d486bab` 仅修复终端打印，没有 rerun/reset，不能声称 formal rollout 使用了 patched source。
- Effectiveness V1 protocol SHA256 为 `5882361adcabaff4c0017cae73a8b95dc8029e204942bf4a83879981bd321d84`；formal
  `788001–788006` 已消费。manifest/log/marker SHA256 分别为
  `6aaebeac4dc8dcf425ce250df6c6abcd991921263605b3dd71d3d717f0783e88` /
  `df9aebf8860212a72ede78f0078e22d0a8d8df90535d20798a436ba4f28c3acd` /
  `b9ca4306835b19756c070eb7917d3e01de429209329e59f397257a5e4c54fd8c`。终态 FAIL，不得重跑或以其 seeds 训练/调参。
- Feasibility attribution V1 protocol SHA256 为 `50b2cb7dc40160a54ecfe150122d7bcc3407b638a40e887b20045120e220b1fd`；formal
  `789001–789006` 已消费。manifest/rows/log/marker SHA256 分别为
  `f2382b1a8c5ee486c4246e716fc27b97a34d2948bfe90633e33d0f428a56f35c` /
  `b07f51b0354d92943864232b7698a0d53061de22e6593911b10e40a8192c412f` /
  `371fd775e823fe347a593fec4ba79ca03217bdfb44ffc38ddbceb382bcbc5161` /
  `89e7af446d312c06019d88f2c4b1e8b4424bde8711dae4cba9294550d1ddfa2e`。rows 仅用于归因，不得训练。
- Exact-proposal support/audit/failure-attribution manifest SHA256 分别为
  `48c0b1cd0b92bd746bd52f18090bbffd8facf59457eb82d503802eb4fc2c512b` /
  `87bdf9b5ce0b9d9cbf967d858669a6f17679c0cf271d83c8c39e7f4fa4e52e95` /
  `3c8e0980f4f970bb66ca207ceb3305dc3fbe9e3eb79e9e7a827a1b086acf11f5`。support rows冻结；audit FAIL，不能进入exact-label repair。
- Proposal feasible-set diagnostic protocol/manifest/catalogue SHA256 分别为
  `dbe387cc134741f87cde8caa2b8910410e42e5ad756dcb197bc2c04c208eabca` /
  `957bc2535fbef767a90fbf6cd299ea5aa38cd2f680099a3c5f79aff97fe4821f` /
  `c7dbbf203084d1b90db8662f42514b8c3f45a5e94b2d7e9e706ad80143784242`；仅诊断，不可训练。
- Proposal-set support protocol/manifest/train/dev SHA256 分别为
  `fa2d764604e05526262549559c718291caeef3ca23202bce38d20d3e038d2972` /
  `d057d0f39a3a99f6641c331eecd1a127a2b639b5046bf78d39b55d1b8caf7f37` /
  `f5511da554f21614e399b6384609bac93a41e9fdef94b8735eb2ecd6cb5d3f14` /
  `dd07e8f2b877bd7f08db113aa0fc8988ed8d106223d799284b3e1f8b6324d17a`。support PASS，但后续learnability FAIL，rows不得用于repair。
- Proposal-set learnability audit 与 failure attribution manifest SHA256 分别为
  `a1b76fa16427cecdc0c70e0d38c6247d8e6e6d943063aeeddd00a8cede28e937` /
  `4a1d6d9d9b594c1d680203e8c8b728e08101ec1fdfc26aa64028be9f800c5e10`。二者均未保存probe checkpoint，candidate state digest始终为
  `fa8740a8a4cb54fffd3bb669537133ede8f231edd754e4b10ebfc0f982a0ed61`。
- Proposal-conditioned feasibility audit protocol/manifest SHA256 分别为
  `68694cefa2e8907b3d4beb77ca3abc2daf569bc9b5ffab6b46b1d70b9888a896` /
  `4b37952e8b624284f1d45280043f89aaad9ef97be4e1f33e690644c03827067f`。终态FAIL；未保存probe checkpoint，未调用environment/Teacher/policy optimizer。
- 2026-09-19 该批任务终止时没有遗留 server process；2026-09-21 已另行启动 P3 V4。Qualification V1 的 recovery manifest/marker 只属于旧 `774001–774006` FAIL 证据，不改写新 V2 PASS。
- 入口 `entry/checkpoint_1.pt` 和Adam诊断快照均不能晋级正式模型；历史失败manifest不得重写。
- 服务器输出根为 `~/training_outputs/terminal`，本地镜像为 `D:/7_codespace/training_outputs/terminal`。源码只在本地修改，经Syncthing同步，服务器仅运行。

## 当前允许的下一步

**2026-09-24 当前顺序：** merge-slot timing reconciliation、fixed-rule dense30 development 与
viewer bounded checks 已完成；历史结果、已消费 seeds 和 `.25` floor 记录保持 immutable。当前按
staged authorization 推进，不把历史 fixed-rule evidence 写成 learned-policy qualification。

1. 保持已冻结的 `eh216s_research_v2` profile、checkpoint/hash contract 和无 Teacher/optimizer
   语义；首个 bounded diagnostic 的协议与结果分别见
   [protocol](evidence/v4_2/trained_dense30_v2_diagnostic_protocol_20260924.md) 与
   [result](evidence/v4_2/trained_dense30_v2_diagnostic_result_20260924.md)。global loader/core
   保持不变，不自动授予 qualification。
2. `842101` 的 read-only trace aggregation 已完成；failed-energy trace 与 inner `LZ_RESOURCE`
   统计见结果页。该证据用于区分 controller request 与 execution feasibility，不能将 WAIT 全部判错，
   也不能写入 imitation-only fix 或新的 PASS。`842101` 历史证据保持 immutable，`842102` 保持
   unused。
3. fresh random-SOC development comparison 已完成，详见
   [protocol](evidence/v4_2/random_soc_repair_development_20260924.md) 与
   [result](evidence/v4_2/random_soc_repair_development_result_20260924.md)。`843000` smoke 与
   `843101` trained/nominal_rule 配对均已记录；trained `9/18/3/0`、nominal `17/13/0/0`
   （normal/emergency/failed-energy/unresolved），两者均 terminal 且无 collision/NMAC/
   hard-separation/fail-closed。该结果包含 19 条 trained probe、4 个 approved witness，不能写成
   gate-only 归因或 qualification。
4. 当前目标转为准备 fresh P3 `flow6`/`flow12`/`dense24` development 与 dense24 forced-gate/
   frozen-proposal 消融；代码与新 protocol pending。配对结果不授权 targeted repair 训练或
   qualification，`PPO_ALLOWED=false`，保持 `80/40/10 m`、`min_landing_soc=.10` 与
   `soc_reserve=.25`；低层 static qualification 完成后才进入高层 A6。
5. P3 random-SOC 范围沿用现有 P3 V4：initial SOC `.55–.90` uniform、priority `1/3/5`、场景
   为 `flow6`/`flow12`/`dense24`。dense30 pressure 诊断不能单独构成 qualification；后续资格证据
   仍需 train/dev/heldout 独立性。现有 P3 hard gates 保持 `dev/eval completion=1.0`、safety、
   exact replay、K3 shortlist、low-SOC wait non-worse 和 priority-weighted wait improvement；本轮不
   新增 completion threshold。预留 `829001–829006` heldout 保持 untouched，不改作本轮开发集。

**历史顺序（2026-09-23，已由上方 2026-09-24 当前顺序取代）：** SOC-priority paired diagnostic 保持 FAIL，原因是 `3` 条 `FAILED_ENERGY`；immutable replay/witness attribution 已完成。下一步独立预注册 fresh-seed 的最小 repair，覆盖 low-SOC commitment gate 或有证书的长计划安全 preemption，以及 emergency pad/corridor capacity；保持现有 SafetyFilter 和安全阈值，先评估最小修复，再决定后续课程，repair 证据完成前不启动 PPO，`PPO_ALLOWED=false`。P3 high-SOC 与 low-SOC formal seeds 仍不得 formal 重用、reset 或用于调参。

1. 保留原始失败 manifest/tape、`failure_trace_evidence.json` 与 witness 文件为 immutable evidence；不 rerun、reset 或调参。
2. 单独冻结 fresh-seed 的最小 repair protocol，覆盖 low-SOC commitment gate/有证书的长计划安全 preemption，以及 emergency pad/corridor capacity；保持现有 SafetyFilter 和安全阈值，该诊断不授权 PPO。
3. 保留 support、learnability、repair、Qualification V1/V2、effectiveness、exact-proposal 与 proposal-set 全链条为 immutable evidence；已消费环境seeds不得 rerun/reset，也不得用于训练、阈值或模型调优。
4. 暂停 proposal repair、learned proposal head/loss 与 fixed proposal index 调整；V1/V2/V3/V4 及 fixed-neutral V1/V2 seeds、rows、metrics 均为 immutable evidence，不得重跑或作为新阈值调优集。
5. K=3 runtime boundary 已冻结；P2R 只改变 scheduler-owned LZ/同一步仲裁/final资源语义，不得隐式加入 runtime catalogue search 或让策略直接输出 `lz_id`。
6. P2R 当前不需要因4/108个单种子漏放行立即重训。若后续 fresh 多种子确认稳定的 individually-certified-but-not-requested 缺口，先独立预注册 train/dev/heldout seeds 和 warm+flow mixed curriculum，再决定 gate support/训练；当前诊断不得直接授权 PPO。
7. **最新 P3 停止规则：** checkpoint12 的安全 FAIL 与 checkpoint5 的 fresh Q3 完成率 FAIL 均不可改写；不得按 825006/826006 的表现继续挑选旧检查点，或将低等待（只统计完成者）当作高吞吐证据。Q3 dense24 的1080步 exact replay 显示 380秒后放行批准持续为0，后续累计 `NO_MERGE_GAP=305`、`PREDICTED_SEPARATION=344`，末态11架积压、6个LZ空闲。1800秒对照中 trained 仍停在13/24，而初始 K3 在911秒24/24；V4 fresh dev 则证明原始 K3 对所有 dense24 seeds 并非安全保证。828001 失败前60秒 509 个 actor 状态仅17次请求 RELEASE、1次获批；失败当步两架内环 UAV 均未请求 RELEASE，触发的是 `backup_existence` 递归可恢复性门槛。V4 在正式更新前已停，不得因基线失败而继续同协议或放松 SafetyFilter；下轮应先核验该备份认证失效是可行性边界还是求解器漏证，再预注册渐进密度课程和独立安全/吞吐开发集。P4/P5 与 A6 仍禁止。

完整协议、旧失败与诊断详情见[持续课程证据](evidence/v4_2/continuous_curriculum_20260917.md)；初始准入见[准入结果](evidence/v4_2/v4_2_v2_1_ppo_admission_summary.md)，轨迹修复与第二训练块见[续训结果](evidence/v4_2/ppo_block2_transfer_repair_protocol.md)。

## 不应恢复的旧路线

- V3 factorized 三头策略；
- V4 30-way 与 V4.1 flat-26 actor；
- FULL-Teacher PPO 或用 Teacher corrected action 计算 requested-action PPO ratio；
- 87D V2 checkpoint 直接续训 88D V2.1；
- 两环、8D high-level、自由 `ψ/γ/V` 或姿态/电机 action；
- 把日期型 handoff 或单次 diagnostic 当作新的 canonical specification。
