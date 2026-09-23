# 应急迫降与分阶段能耗 V2（2026-09-23）

## 结论边界

本轮实现的是新的、显式启用的研究 surrogate。历史 qualification 继续使用
`energy_model_mode=legacy_phase`、`emergency_landing_enabled=false`，旧结果不重写。
新模型没有 EH216-S 分阶段实测功率，不能作为真实续航认证。

## 文献映射

- Kleinbekman、Mitici、Wei 的滚动时域调度论文（DOI
  [10.2514/1.I010776](https://doi.org/10.2514/1.I010776)）按巡航、浅下降、水平 final、
  垂直 final 分阶段累计 SOC；其 EHang 184 场景报告巡航 `36 kW`、悬停 `42 kW`，以及浅下降
  每增加一秒延误的边际功率 `28 kW`。本实现只使用相对比值，且不把 `28 kW` 写成整段下降
  的平均实测功率。
- Kasliwal 等人的 [Nature Communications 论文](https://www.nature.com/articles/s41467-019-09426-0)
  将 VTOL 任务分成 takeoff hover、climb、cruise、descent、landing hover，并按
  `sum(P_phase * t_phase)` 计算任务能量。该文对象是带翼 VTOL，其零下降能耗简化不适用于
  本项目的 lift-only 多旋翼，因此本实现对垂直下降采用 hover-like power。
- Pradeep 与 Wei 的多旋翼到达轨迹模型（DOI
  [10.2514/1.I010710](https://doi.org/10.2514/1.I010710)）给出基于动量理论的悬停、前飞、
  爬升和下降功率及 vortex-ring-state 下降约束。当前 fixed corridor 仍受已有垂直速度和加速度
  envelope 约束；后续取得转子/功率参数后再升级为物理功率模型。
- EH216-S 公开规格来自 [EHang 官方页面](https://www.ehang.com/ehang216s/)。公开页面没有提供
  可直接标定本模型所需的逐阶段功率遥测。

## 实现语义

`phase_motion_v2` 对已提交轨迹按每个时间段的速度方向积分 normalized power：水平巡航为
`1.0`，水平浅下降为 `28/36`，低水平速度或垂直下降为 `42/36`，爬升暂用保守的 `1.30`。
无 committed plan 的环上等待仍按 phase power 计量。每个 resolved UAV 在
`energy_outcomes` 中记录时间、phase、SOC、剩余/消耗能量、outcome 和 reason。

固定应急航迹只针对未提交其他计划的环上 UAV：从真实当前位置和速度连续切向脱环，
使用 cubic Bézier 进入内部 staging point；外/中环进入 `120 m` staging，内环保持当前高度；
向已预约 LZ 内移，再沿现有垂直 leg 下降。脱环段的 parameterizer 绑定真实 initial velocity，
并以水平速度为零进入 staging。提交条件为：

1. `SOC <= emergency_trigger_soc`；
2. 预计应急航迹能量 + `min_landing_soc` reserve + margin 可满足；
3. LZ 为 FREE 且原子预约成功；
4. canonical trajectory 通过原 SafetyFilter。

应急请求没有安全豁免，原 `80/40/10 m` SafetyFilter 契约保持不变。安全拒绝或无 LZ 时等待五秒后重试；episode 继续。成功 touchdown
占用 LZ 并完成正常地面服务，终态 `EMERGENCY_LANDED`，低层 reward 收取独立 `-8` cost。
若执行中能量耗尽则为 `FAILED_ENERGY`。

共享 actor 已经接收每架 UAV 自身 SOC。开启应急课程后，PriorityEngine 在
`emergency_trigger_soc → min_landing_soc` 区间连续提高该 UAV 的 dynamic priority；共享参数因此
可以根据各自 observation 产生不同 WAIT/RELEASE 意图，同一步仲裁也优先处理低 SOC UAV。

## 首轮本地检查

手工构造空场景的 OUTER/MIDDLE/INNER 环状态，三条路线均通过动力学验证与 SafetyFilter。
当前 continuous-tangent 路线的计划时长（OUTER/MIDDLE/INNER）为
`158.611503/118.406382/65.456200 s`，`EnergyModel.estimate_plan_energy` 分别为
`182.844990/149.283733/67.905555`。三条计划的 `validate_kinematics` 均为空，且在空场景中
均获 SafetyFilter approved；该检查未运行完整 episode。

## 第一轮 repair 诊断

协议 `EMERGENCY_ENERGY_V2_REPAIR_PILOT` 的 SHA256 为
`e99cb11f8cf9756f26643f6e3fae8dc294394b22275ff9b846acafdf00377bc8`，使用 fresh seeds
`840101/840102`，触发阈值为 `.40`，并保持 `80/40/10 m` 与显式能量结果语义。聚合结果为：

| normal | emergency | upstream low-SOC divert | failed_energy | safety events | terminal in-terminal abstract divert |
|---:|---:|---:|---:|---:|---:|
| 7 | 14 | 2 | 1 | 0 | 0 |

唯一 `FAILED_ENERGY` 为 arrival 9 / uid 8。其首次外环应急候选发生在 `t=118 s`、SOC
`.388784`；候选能量可行，但被 SafetyFilter 拒绝。该 UAV 共 117 次应急尝试，主要阻挡来自
`holder reachable tube` 条件。该记录对应的服务器 manifest SHA256 为
`8bf0bba015eb712e95f48200c9326a32e928e4d4014e9188a951635eca57f55d`。

## 同 seed forensic replay

对 seed `840101`、trigger `.40` 的只读 replay 中，arrival 9 / uid 8 在 `t=118 s` 成功
commit 应急路线并得到 `EMERGENCY_LANDED`；最终 SOC 为 `.292909`，最小 swept separation
为 `81.638 m`，全局安全计数均为 `0`。该 replay 仅用于修复证据，不是新的 qualification gate。

## Route-repair fresh 诊断

协议 `EMERGENCY_ENERGY_V2_ROUTE_REPAIR_PILOT` 的 SHA256 为
`8c13dcf6a4b5f3b7f965d01429f4c9e1ffed403e8c21cbc5e7fb0d574f2b0108`，runner SHA256 为
`936c4336f056334328b63c9fd5c4e688c17542f8e940986aa3df2b3c1bf5815d`，使用 fresh seeds
`840201/840202`。该 formal route protocol 显式固定初始 SOC `.35–.55`，因此后续
phase_motion_v2 未显式范围时的 default fallback 修复不改变这轮已运行协议语义。两条记录通过
route-repair hard gates：

| normal | emergency | upstream low-SOC divert | failed_energy | terminal in-terminal abstract divert | safety events |
|---:|---:|---:|---:|---:|---:|
| 9 | 11 | 4 | 0 | 0 | 0 |

两个 case 的完成时间为 `420 s` 和 `341 s`，最小 swept separation 为 `80.0 m` 和
`94.320285 m`。应急候选的 safety blocks 为 `6` 和 `15`；energy、LZ、dynamics blocks
均为 `0`。服务器 manifest SHA256 为
`21756764251332eda9782e5b383f8063560d74820c9890fa1183eeb4c988c5b7`。

## 验证边界

本地定向测试为 `55 passed`，服务器定向测试为 `16 passed`。两轮诊断均为 rule-only；
PPO、training、teacher 和 checkpoint 均为 `false`。上述两个 fresh seed 结果不构成
qualification，也不构成真实 EH216-S 功耗或安全标定。
