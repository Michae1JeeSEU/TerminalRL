# Dense30 低层预训练前置诊断（2026-09-24）

## 证据边界

本文件记录 30 机潜在冲突场景的低层预训练前置 exploratory 诊断。记录不构成
qualification，不构成完整 30 机 flow 证明，也不授权 PPO；`PPO_ALLOWED=false` 保持不变。
其中 high-SOC all-warm dense30 运行使用固定 K1 rule action `[1, 0, 0]` 和 100 s 截止时间，
因此不能代表训练后策略，也不能据此判断完整回合的最终吞吐量。

## Opt-in continuous final/LZ exploratory 运行

为检查固定几何与最终进近语义在 30 机潜在冲突场景下的表现，加入了 opt-in
`final_approach_geometry_mode=continuous_inward_v1`。该模式使用连续切向脱环的共享几何，保留
正常 final-approach 语义；中心扫描按兼容性排序空闲 LZ。SafetyFilter `80/40/10 m` 保持不变，
默认 `brake_bezier_v1` 未改变。该模式不是默认配置，也没有扩大 SafetyFilter 的安全阈值。

本轮集成定向测试为 `121 passed`。在 seed `840626`、30 架全 warm、high-SOC、100 s horizon
的探索中，continuous 模式产生 `13` 次 release approval，完成 `1` 架 normal；emergency、
failed-energy 和 safety 均为 `0`。最小实际间隔为 `82.03085 m`，服务器墙钟时间为 `13.4638 s`，
旧模式同条件约为 `16.47 s`。该运行在 horizon 结束时仍未完成完整回合。

同一 seed 和模式延长至终态后，服务器在 `t=1117` 结束，墙钟时间为 `51.9834 s`：

| 指标 | 观察值 |
|---|---:|
| normal completion | 20 |
| emergency completion | 8 |
| failed-energy | 2 |
| unresolved | 0 |
| release approval | 51 |
| emergency attempt / commit | 304 / 8 |
| 最小实际 / 扫掠间隔 | `80.0331 / 80.0226 m` |
| hard-separation / NMAC / collision | 0 / 0 / 0 |
| shield fail-closed | 0 |

两条 failed-energy 记录对应 outer UAV `UID13@889 s` 和 `UID14@1116 s`。请求记账为真，
但该 full-cycle 结果未通过 `zero_energy_failure` hard gate，因此仍是 exploratory evidence，
不进入低层课程或 PPO，`PPO_ALLOWED=false`。UID-specific emergency/release trace 已补齐；下一步
前置是量化 merge-slot shortfall，随后完成安全规则与应急路线修正，再冻结 fresh dense30
normal/stress protocol。实际 GUI 尚未完成视觉验收。

## UID-specific emergency/release trace

同一 seed、同一 continuous 配置的完整回放与原始运行保持 exact outcome parity：终态为
`1117` steps、`20` normal、`8` emergency、`2` failed-energy。两条失败目标的 UID 级 trace 为：

| UAV | 首次 emergency 时刻 / SOC | 实际 emergency attempts | 被拒 emergency plan attempts | 拒绝原因计数 |
|---|---:|---:|---:|---|
| `UID13` outer | `t=290 / .399565` | 120 | 14 | `8` holder tube，`6` predicted separation |
| `UID14` outer | `t=495 / .414041` | 125 | 11 | `9` holder tube，`2` predicted separation |

在两条失败终点，6 个兼容 LZ 均为空闲；对应的早期窗口为 UID13 的 `t=200/290` 和 UID14 的
`t=400/495`。在这些窗口中，6 条 emergency route 均有正能量余量并通过 energy check（`6/6`），
但 SafetyFilter 认证为 `0/6`，即使忽略 LZ 占用状态也相同。观察到的能量 slack 范围为
`25.556–177.007`。因此该证据不能归结为终点 LZ 数量不足，也不支持放宽现有安全阈值。

正常 25-proposal 离线扫描中，`t=100/200` 的 `0/25` proposals 均因
`dynamics_unreachable_merge_time` 被拒；`t=500/750` 的 `0/25` proposals 均因
`emergency_energy_guard` 被拒。扫描用于归因，不扩展运行时 proposal 搜索范围。

相关 forensic artifacts 位于
[`training_outputs/terminal/dense30_continuous_preflight_20260924`](../../../../training_outputs/terminal/dense30_continuous_preflight_20260924/)，
包括 [完整回放 JSON](../../../../training_outputs/terminal/dense30_continuous_preflight_20260924/seed840626_continuous_full_1200.json)、
[failed-energy trace JSON](../../../../training_outputs/terminal/dense30_continuous_preflight_20260924/seed840626_failed_energy_trace.json)
及对应日志和 progress 文件。

## Outer merge-delay scan and timing alignment attempt

针对 seed `840626` 的 outer UAV `UID13/UID14`，执行了只读 merge-delay scan。每个 UAV 在
`t=100` 和 `t=200` 各检查 `63` 个 candidates，组合为 speed residual `-1/0/1`、merge
residual `-1/0/1`，以及 delay `0/10/20/40/80/120/180 s`：

| 时刻 | UID13 preliminary certified | UID14 preliminary certified |
|---|---:|---:|
| `t=100` | `0/63` | `0/63` |
| `t=200` | `13/63` | `21/63` |

这些 preliminary certified candidates 的最佳 planned endpoints 比请求的 merge slots 提前约
`95–97 s`。因此这些 approval 不能验证可执行的安全 merge slot：reservation timing 与 trajectory
timing 不一致。扫描结果保存于
`/home/user/training_outputs/terminal/outer_merge_delay_scan_20260924/seed840626_outer_merge_delay_scan_t100_t200.json`。

另一次严格 `<=1 s` alignment search 中，synthetic aligned candidate 通过；但在 12-UAV 回归中，
seed `840620` 终态为 `1 normal + 8 emergency + 3 failed_energy`，seed `840622` 为
`1 normal + 10 emergency + 1 failed_energy`。aligned candidate 的计算耗时为 `35.9 ms`，
unavailable candidate 的计算耗时为 `176 ms`。该改动已完全回滚，相关 energy guard 定向测试
为 `8 passed`。这组回归不改变当前 SafetyFilter 阈值，也不构成 PPO 准入证据。

## Mixed30 exploratory 运行

补充完成了 seed `840627` 的 mixed30 exploratory：`12` 架 warm-start 加 `18` 架每 `8 s`
到达，SOC 范围 `.55–.90`，启用 continuous geometry 和 route guard。回合在 `1032` 个 terminal
steps 结束，服务器墙钟时间为 `43.3962 s`：

| 指标 | 观察值 |
|---|---:|
| normal / emergency / failed-energy | `23 / 6 / 1` |
| resolved | `30/30` |
| release approval | `68` |
| emergency attempts / commits | `174 / 6` |
| route checks / negative route checks | `615 / 143` |
| 最小实际 / 扫掠间隔 | `81.2511 / 81.2478 m` |
| safety events | `0` |

唯一 failed-energy 为 arrival `17` 的 outer `UID44`，发生在 `t=1031`。该回合的 `20%`
emergency 比例和 `1` 条 energy failure 使训练 gate 失败；这仍是 exploratory evidence，
不构成 qualification，也不能作为当前静态 dense30 问题已经解决的证据。结果文件为
`/home/user/training_outputs/terminal/dense30_continuous_preflight_20260924/seed840627_mixed12plus18_full_1500.json`，
本地同步路径为
[`training_outputs/terminal/dense30_continuous_preflight_20260924/seed840627_mixed12plus18_full_1500.json`](../../../../training_outputs/terminal/dense30_continuous_preflight_20260924/seed840627_mixed12plus18_full_1500.json)。

## UID44 mixed30 exact replay and early-entry scan

UID44 的 exact replay 与 mixed30 原始回合保持一致：`1032` terminal steps，`23` normal、`6`
emergency、`1` failed-energy，peak active 为 `21`；峰值环上数量为 outer/middle/inner
`9/4/4`。arrival `17` 的 outer UID44 在 `t=1031` 失败。

UID44 的追踪窗口如下：

| 时刻 | 状态 | SOC | energy check | SafetyFilter | 说明 |
|---|---|---:|---:|---:|---|
| `t=70` | first tracked inbound | `.63691` | — | — | 仍为 inbound，未进入环上 emergency scan |
| `t=130` | outer | `.600799` | `6/6` | `0/6` | energy slack 约 `+313–386`；normal proposals `25/25` rejected |
| `t=145` | first emergency | `.590799` | `6/6` | `0/6` | energy slack 约 `+313–386`；normal proposals `25/25` rejected |
| `t=750` | outer emergency window | — | `0/6` | `3/6` | — |
| `t=900` | outer emergency window | — | `0/6` | `6/6` | — |

`t=70` 是 inbound 记录，且在 `t=100` 之前；因此该时刻没有执行环上 emergency scan，不能据此
推断已经完成 admission check。早期 `t=130/145` 的 energy 可达性已有通过记录，但当时没有
SafetyFilter 通过的候选；后期 `t=750/900` 的 SafetyFilter 通过记录又没有 energy 通过记录。

相关 artifacts 为
[`seed840627_mixed12plus18_trace_uid44_v2_20260924.json`](../../../../training_outputs/terminal/dense30_continuous_preflight_20260924/seed840627_mixed12plus18_trace_uid44_v2_20260924.json)
和
[`seed840627_uid44_early_entry_scans.json`](../../../../training_outputs/terminal/dense30_continuous_preflight_20260924/seed840627_uid44_early_entry_scans.json)。
早期 future-exit certification 以及 coordinated holder/pad traffic plan 仍属于 proposed
prerequisites，尚未实现或证明。该 evidence 保持 `PPO_ALLOWED=false`。

Windows 早期 benchmark 的 `2.21 steps/s` 与本次服务器完整回合的 `21.49 steps/s` 不可直接比较：
后者后段 active UAV 已减少，状态计算量较轻，不能作为 30 架全并发实时速率。

## High-SOC dense30 all-warm 运行

服务器运行的固定条件为：

- seed：`840626`；30 架 UAV 全部 warm-start active；SOC 范围 `.55–.90`；
- ring 半径：`600/400/200 m`；route guard 开启；
- rule action：K1 `[1, 0, 0]`；
- 仿真上限：`100 s`。

100 s 结束时，30 架 UAV 中 `0` 架获得 release approval，`0` 架 resolved，`30` 架仍
unresolved；共有 `3000` 次 release rejection。rejection 分类为：

| 分类 | 次数 |
|---|---:|
| `safety_reject` | 343 |
| `safety_wait` | 2655 |
| `emergency_energy_guard` | 2 |
| 合计 | 3000 |

`safety_reject` 的 secondary constraint 计数为
`dynamics_unreachable_merge_time=181`、`predicted_holder_reachable_tube=153` 和
`predicted_post_plan_holder_reachable_tube=9`；`emergency_energy_guard` 为 `2`。本次运行
观察到 `failed_energy=0`，且未观察到 hard-separation、NMAC 或 collision；由于 100 s
时仍有 30 架 unresolved，以上安全观察不等于完整回合通过。

服务器结果文件为
`training_outputs/terminal/energy_reachability_dense30_preflight_20260924/high_soc_dense30_840626_100steps_secondary.json`
及其同目录日志和进度文件。诊断入口为
`terminal_marl/training/diagnose_energy_reachability_guard_smoke.py`。

## Proposal、占用和几何诊断

在同一 seed 的只读 proposal 诊断中，25-proposal catalogue 在 `t=0` 和 `t=50` 均为
`750/750` rejected：

| 时刻 | `dynamics_unreachable_merge_time` | `predicted_holder_reachable_tube` | `predicted_post_plan_holder_reachable_tube` | energy | rejected |
|---|---:|---:|---:|---:|---:|
| `t=0` | 382 | 341 | 27 | 0 | 750 |
| `t=50` | 389 | 325 | 34 | 2 | 750 |

该 catalogue 只用于离线覆盖诊断，没有扩展运行时策略的 proposal 搜索范围。

同一 seed 在 `t=0` 的 occupancy 扫描结果如下。`approvals` 是 proposal approval 次数，
`distinct UAVs` 是获得过 approval 的不同 UAV 数量。

| active UAV 数 | outer/middle/inner occupancy | approvals | distinct UAVs |
|---:|---:|---:|---:|
| 6 | `3/2/1` | 63 | 5 |
| 12 | `6/4/2` | 92 | 10 |
| 18 | `9/6/3` | 25 | 1 |
| 24 | `12/8/4` | 0 | 0 |
| 30 | `15/10/5` | 0 | 0 |

将 ring 半径扩大为 `800/550/300 m`，保持相同高度和速度后，100 s 运行仍为 `0` approvals。
在 t=0 的一个状态中，5 架 inner UAV 对应的 6 个兼容 LZ 均为 free，但 750 个 proposals
全部被 `predicted_holder_reachable_tube` 拒绝。

另有一次临时 nominal-holder-speed tube counterfactual 在 t=0 得到 `43/750` approvals，
覆盖 17 架 UAV，其中 outer/middle/inner 分别为 `26/17/0`。该 counterfactual 除非有
合同化速度锁定，否则不满足当前安全契约；没有进入运行时代码或结果。

其中，5 架 inner UAV 到 6 个兼容 LZ 的全部 `750` 个 inner→final candidates 在该
counterfactual 下仍为 `0` approval，且每次都由 `predicted_holder_reachable_tube`
拒绝。该观察不构成 inner final approach 的安全证书。

## 仿真与可视化性能

Windows dense30 benchmark 的每步耗时依次为：优化前 `1.06462 s/step`，完成前两项优化后
`0.70742 s/step`，加入 speed reuse 后 `0.53378 s/step`，完成本轮全部优化后
`0.451737 s/step`，即约 `2.21 steps/s`。使用 `frame_stride=5` 时 total viewer 约
`0.462 fps`。

逐步送帧的异步离屏检查中，headless 为 `2.3685 steps/s`，带异步 viewer 为
`2.297 steps/s`，差约 `3%`；检查期间没有 queue backlog。PyVista 单次 update 约
`0.095 s/frame`。这些是离屏/基准测量，尚未证明实际 GUI 播放已经流畅。

第二次 20-step 离屏测量中，headless 为 `3.026 steps/s`，带异步 viewer 为
`2.883 steps/s`，viewer 开销约 `4.8%`；浏览器端生成了 `24` 个插值帧，前一次为 `3` 个。
这仍不能替代实际 GUI 的主观视觉验收，GUI subjective QA 当前不可用。

本轮相关修改包括精确保持语义的 safety 计算优化、live viewer 默认 `stride=1` 和浏览器端
仅用于显示的 position interpolation；没有放宽 SafetyFilter 阈值，也没有启动 PPO。
相关代码文件为 `terminal_marl/core/safety_filter.py`、
`terminal_marl/core/runtime_traffic_safety.py`、
`terminal_marl/core/trajectory_time_parameterization.py`、
`terminal_marl/visualization/config.py`、
`terminal_marl/visualization/rollout_viewer.py` 和
`terminal_marl/visualization/snapshot_playback.html`。

## 回归检查

同 seed 的 12 机回归 `840620/840622` 保持 scalar outcome 和 accounting 不变：高 SOC
回归为 `12/12` normal，近下限回归为 `6` normal 加 `6` emergency，均为
`0` failure 和 `0` safety events。

本轮 focused safety tests 为 `110 passed`；dense diagnostic tests 为 `3 passed`；
visualization focused tests 为 `14 passed`；browser playback behavior tests 为 `2 passed`。

## 后续前置条件

后续继续量化 merge-slot shortfall，并完成 coordinated emergency corridor 的安全认证，或建立
可执行的 safe admission envelope；当前 merge-delay scan 仍因 reservation/trajectory timing
不一致而不能作为可执行 slot 证书。在此之前保持 `PPO_ALLOWED=false`，不得将当前问题标记为
单纯 LZ 数量不足，也不得放宽现有 SafetyFilter 阈值。完成上述前置后，再冻结 fresh dense30
normal/stress protocol，硬门槛保持 `failed_energy=0` 和安全事件为零，并分别报告 measurable
normal completions 与 emergency outcomes。当前推进不以 12 机课程优先级替代 30 机前置验证。

## 新增 runtime timing 与 product viewer 观测

只读 runtime timing audit 的 prefix 结果如下：

| 场景 | prefix 观测 |
|---|---|
| all-warm seed `840626` 前 `200` steps | `21` 次 approved ring transfer；其中 `20/21` 的 `|end-slot|` 超过 `1 s`；median `-8.379 s`，minimum `-49.229 s` |
| mixed seed `840627` 前 `200` steps | `20/20` 的 `|end-slot|` mismatch；median `-8.661 s`，minimum `-41.292 s` |
| mixed actual arrival adjacent gaps | ring_middle minimum `1.97 s`，其中 `5` 个低于 `9 s`；ring_inner minimum `8.29 s`，其中 `4` 个低于 `9 s` |
| prefix safety events | `0` |

对应文件为 [all-warm timing audit](../../../../training_outputs/terminal/merge_slot_timing_audit_20260924/seed840626_allwarm_continuous30_200.json)
和 [mixed timing audit](../../../../training_outputs/terminal/merge_slot_timing_audit_20260924/seed840627_mixed12plus18_continuous30_200.json)。
以上是观测到的 slot timing 与 arrival-gap 前置事实，不是 qualification 结论。

product viewer 的 20-step offscreen benchmark 为：headless `2.918 steps/s`，process viewer
`2.820 steps/s`，排除 `1.63 s` startup 后 slowdown 为 `3.37%`；process viewer 为
`12.01 fps`，p95 frame gap `0.192 s`。该 benchmark 仅为 offscreen 测量，不能替代 GUI 主观验收；
对应测试为 `22 passed`。结果见 [product viewer benchmark](../../../results/_buffered_process_product_benchmark_20260924.json)。

CPU profile 的 20-step 总耗时为 `13.18 s`，`ring_transfer` cumulative `5.912 s`，runtime
`protect` cumulative `3.086 s`，SafetyFilter cumulative `1.890 s`。profile 调用存在嵌套，
这些 cumulative 数值不可相加。结果见 [CPU profile](../../../results/_profile_dense30_cpu_20260924.top20.txt)。

merge-slot timing reconciliation fix 仍在进行，尚未完成验证；当前保持 `PPO_ALLOWED=false`。

## Merge-slot reconciliation completed and full-cycle regressions（2026-09-24）

后续实现已在 `MergeScheduler` 中将 ring-transfer slot 与 `plan.end_time` 对齐，并将 final slot
与已验证的 `scheduled_departure` 对齐。对应 integration tests 为 `158 passed, 1 skipped`。

修正后的 mixed30 前 `200` steps actual slot audit 中，`29/29` 个事件的 slot delta 为 exact
`0`；o2m、m2i、i2f 的最小事件间隔分别为 `9/9/12 s`，低于 headway 的事件为 `0`，prefix
safety events 为 `0`。结果见 [mixed actual slot audit](../../../../training_outputs/terminal/merge_slot_reconcile_20260924/seed840627_actual_slot_audit_200.json)。

两组 full-cycle regression 结果如下：

| 场景 | terminal steps | normal / emergency / failed-energy | approvals | 最小实际 / 扫掠间隔 | 墙钟时间 | safety events |
|---|---:|---:|---:|---:|---:|---:|
| all-warm `840626` | 1067 | `17 / 10 / 3` | 47 | `81.578 / 81.550 m` | `64.925 s` | 0 |
| mixed `840627` | 910 | `18 / 12 / 0` | 61 | `80.00062 / 80.00062 m` | `42.690 s` | 0 |

对应完整结果为 [all-warm full run](../../../../training_outputs/terminal/merge_slot_reconcile_20260924/seed840626_allwarm30_continuous_full.json)
和 [mixed full run](../../../../training_outputs/terminal/merge_slot_reconcile_20260924/seed840627_mixed12plus18_continuous_full.json)。
相较此前 all-warm `20/8/2` 与 mixed `23/6/1` 的 normal/emergency/failed-energy 计数，
本次修正说明 slot timing 的安全与一致性观测已改善，但 throughput/emergency gate 仍未通过，
不构成 qualification 或训练授权；`PPO_ALLOWED=false`。新的 SOC arbitration pilot 仍在进行，
尚未验证。

## SOC-priority opt-in pilot（2026-09-24）

`downstream_soc_priority` 保持 downstream ring precedence，在同一优先级内按自身 SOC 升序排序；
不改变 observation 维度，默认模式仍为 `downstream_priority`。该 pilot 不改变 SafetyFilter，
也不构成训练授权。

前 `200` steps 的只读对照为：

| 场景 | approvals（default → SOC priority） | resolved（default → SOC priority） | 墙钟时间（default → SOC priority） |
|---|---:|---:|---:|
| all-warm `840626` | `23 → 19` | `5 → 4` | `26.423 → 27.693 s` |
| mixed `840627` | `29 → 28` | `5 → 5` | `13.841 → 13.420 s` |

完整 mixed `840627` 对照中，default downstream priority 为 `18 normal / 12 emergency / 0 failed`
（step `910`，墙钟 `42.690 s`），SOC priority 为 `20 normal / 9 emergency / 1 failed`
（step `1043`，墙钟 `43.70 s`）。SOC priority 的 failed arrival `17` 为 `UID71`，位于
`ring_middle`，发生在 `t=1042`。两种模式的 safety events 均为 `0`。

该 pilot 因 `zero_energy_failure` hard gate FAIL，未晋级且不改变默认模式；`PPO_ALLOWED=false`。
当前没有 full all-warm SOC-priority pilot 结论。对应 artifacts 为 [all-warm prefix](../../../../training_outputs/terminal/soc_release_order_20260924/seed840626_allwarm30_soc_priority200.json)、
[mixed prefix](../../../../training_outputs/terminal/soc_release_order_20260924/seed840627_mixed12plus18_soc_priority200.json)
和 [mixed SOC-priority full run](../../../../training_outputs/terminal/soc_release_order_20260924/seed840627_mixed12plus18_soc_priority_full.json)。

## Single-state nominal-holder contract counterfactual（2026-09-24）

只读 counterfactual probe 固定在 current-code seed `840626`、`UID13`、`t=200 s`、
`ring_outer`、SOC `0.4595651`。当前 SafetyFilter 对全部 `6` 条 emergency routes 的结果为
energy `6/6`、SafetyFilter `0/6`；其中唯一 free compatible LZ 为 `LZ4`，该候选 energy `1/1`
通过、SafetyFilter `0/1`。当前原因计数为 `predicted_holder_reachable_tube=3`、
`predicted_separation=3`。

将 in-memory nominal-holder 的 exact trajectory 同时用于 lower/upper tube bounds 后，全部
`6` 条路线仍为 `0/6` SafetyFilter 通过，原因计数保持相同；`LZ4` 仍为 `0/1`。该单状态结果
显示 nominal speed promise 单独不足以改变该状态的 SafetyFilter verdict，但它是 read-only、
single-state 观察，不构成一般不可能性证明。未放宽运行时安全阈值，也未修改 source。

结果见 [nominal-holder contract probe](../../../results/emergency_nominal_holder_contract_probe_20260924.json)。
该 probe 未调用 training，且保持 `PPO_ALLOWED=false`。
