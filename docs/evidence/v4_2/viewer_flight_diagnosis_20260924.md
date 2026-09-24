# 30 机可视化回放与飞行终态诊断（2026-09-24）

## 证据边界

本文件记录 current-code 对用户所看的 30 机场景进行的精确重放，以及主 agent
已经作出的解释结论。它是 exploratory evidence，不是 qualification，不给出一般
空域容量或普适安全保证；`PPO_ALLOWED=false` 保持不变。完整 trace 位于本地
`results/viewer_trace_841301_full.json`，约 149 MB，未提交。

## 重放条件与终态

重放固定为 seed `841301`、30 架 all-warm、初始 SOC `.55–.90`、
`final_approach_geometry_mode=continuous_inward_v1`、route guard on、
`release_arbitration_mode=downstream_priority`、late-slot reconciliation off，
逐步调用 `env.step()`。结果为 `1364` steps、`18 completed + 8 emergency + 4
failed-energy`，与此前同配置结果一致。

该回放的最小实际间隔为 `81.353428 m`，最小 swept 间隔为 `81.169130 m`；
named safety counters 为零。该观察不外推为一般安全保证。

## `t=813 s` 的 UID 级记录

在 `t=813 s`，6 个 LZ 均为 `free`。UID `4/11/12` 的共同状态如下：

| UID | ring/geometry | speed | committed plan | SOC | 剩余能量（normalized units） |
|---:|---|---:|---|---:|---:|
| 4 | inner, `R=200, z=40` | 16 | `None` | `0.118900777565` | `178.351` |
| 11 | inner, `R=200, z=40` | 16 | `None` | `0.119458299338` | `179.187` |
| 12 | inner, `R=200, z=40` | 16 | `None` | `0.366750495978` | `550.126` |

6 个候选 emergency pad 的必要能量范围为：UID 4 为 `439.741–458.183`，UID 11
为 `443.795–462.337`，UID 12 为 `441.685–463.972`。这些范围包含 `25%` landing
floor（`375`）、`15` margin 和 route power。UID 4/11 已不能通过 energy gate；
UID 12 的能量足够，但其最近记录中的 6 个 final 候选均出现
`predicted_holder_reachable_tube`。该记录不能当作 `t=813 s` 刚执行的 certification：
该 tube 覆盖其他 holder 的未来合法速度残差，不等于当前 pad 已被占用。

针对该时刻的局部只读 probe 进一步记录到 UID 12 的 6 条 emergency candidate 均被
`predicted_holder_reachable_tube` 拒绝。即使把其他 holder 固定为 nominal 速度，UID 4
nominal swept 最小间隔按 pad `3/4/5/0/1/2` 仍分别为
`15.040/17.248/17.210/63.639/39.236/20.469 m`，全部低于 `80 m`。因此该时刻的
观测不能归结为 tube 约束过严；当前候选本身与后方 UID 4 的环飞存在冲突。该 probe
没有执行这些 candidate，也没有放宽安全阈值；它只是对重建的 `813 s` 状态进行的局部
只读诊断。probe 文件为本地 `results/viewer_trace_841301_probe813.json`。

本次重放没有任何“返环给低电量 UAV 让出降落点”的轨迹证据。现有
`emergency_priority` 只影响尚未放行的 request，不撤销已 committed path。

## I 环到降落点的路径与失败时间语义

当前 I 环到 pad 的路径语义是先飞到降落点上方，再执行垂直下降；内环正常 final
使用约 `40 m` staging 高度，外/中环 emergency 可使用 `120 m` staging 高度，不能
把所有 emergency 路线都解释为 `40 m`。full vertical execution trace
`results/viewer_vertical_execution_841301.json` 记录了 `26/26` 个成功 landing plan
均有真实 `advance_plan` touchdown：`touchdown_return_true=26`、touchdown `z=0`
且 pad XY 最大误差为 `0`。每架有 `15–36` 个实际竖降控制步样本，高度均单调下降，
vertical duration 为 `14.696–34.834 s`。因此本次回放没有漏执行垂直段的证据。

本次回放的 failed-energy 记录为：

| UID | 记录时间 | phase | reason |
|---:|---:|---|---|
| 3 | `883` | outer | `energy_depleted` |
| 4 | `991` | inner | `energy_depleted` |
| 11 | `992` | inner | `energy_depleted` |
| 12 | `1363` | inner | `energy_depleted` |

4 架 failed-energy UAV 均仍在环飞阶段耗尽电量，未进入成功 landing plan；应将这种
energy terminal 与成功落地后退出 active 区分开。UID 12 在 `t=991` 的剩余能量为
`372.1257439666631`、SOC `0.24808382931110873`，在 `t=993` 为
`370.1257439666631`、SOC `0.2467504959777754`；在 UID 4/11 终态移除后，UID 12
已经低于 `375` reserve，按同一约束不能通过降落能量 gate。这一观察不应归结为只有
UI 问题。

`_resolve` 中的 `self.now` 是处理 tick 的起点，因此下一 snapshot 时间为 `+1 s`；
这些时间不解释为 substep 精确瞬间。

用真实 `t=773/813` source 点重建旧 renderer 的直线中点时，UID `4/11/12` 的中点
半径均为 `5.8399 m`，pair distance 为 `4–12: 4.1742 m`、`4–11: 9.0414 m`、
`11–12: 11.0868 m`。这是构造的旧渲染反例，用于说明旧插值可能制造不合理视觉轨迹；
它不是用户 GUI 确切丢帧日志。

## 可视化链路观测与修复范围

旧 viewer 的有界 queue 会丢弃旧帧，且对 source frame 之间直接做线性插值；这可能
制造穿环、突然加速等视觉运动。当前没有用户那次 GUI 的丢帧日志，因此不能声称
用户窗口确实丢失了某个确切的 `40 s` 区间。

当前 viewer 修复范围包括：

- 低 SOC UAV 使用橙色并显示 `LOW SOC`，应急 UAV 使用红色并显示 `EMERGENCY`；
- `failed_energy` 使用持久红色标记，并在累计事件面板中分页记录和打印；
- 3D 中 LZ 使用 `P0`–`P5` 短标，固定 `PAD STATUS` 面板显示 `FREE`、`RESERVED UID`、
  `SERVICE UID` 状态；
- 已采样 source frame 按顺序消费，长 source gap 或 plan 变化时不做直线插值；
- GUI 正常结束时保留最终画面直到用户关闭，headless 不等待。

这些 UI 行为只改变观察和记录方式，不改变 headless 仿真的物理、规划或安全认证。
failed-energy 本身仍未修复，不能将可视化修复解释为能量零失败修复。

最终 UI 联合回归退出码为 `0`，共 `67 passed`，覆盖 energy event snapshot、UI、snapshot、
integration、live interpolation、buffered process、warm-flow 以及 energy reachability/
emergency regression；`results/viewer_energy_ui.png` 也已完成截图审阅。该验证确认的是
记录与显示链路，不改变上述 failed-energy 结论。
