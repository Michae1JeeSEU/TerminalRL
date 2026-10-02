# 轨迹运动学一致性修复审核方案（2026-10-02；已实施部分）

状态：用户已同意本轮重大修复；opt-in `executable_trajectory_v3` 已实现并限定为 static geometry，V2 默认路径与 50 条历史 hash 回放保留。统一 p/v/a quintic、全曲线解析 bounds、live initial/ring-handoff guard 已落地，check-only preflight 已通过。此前 compact/strengthened smoke 失败 provenance 保留；最终 corrected V3 smoke 已接受为 `SMOKE_OK`，三臂各 2 tick、validity valid、technical errors 为空、safety none、uncertified/invariant 为 `0`，energy coverage valid，V2/V2/V3 bindings 与 action/metrics/hash checks 通过。counter mapping 已修正，evaluator tests `16 passed`；八臂业务矩阵尚未启动，下一步先做相同两 tick 性能剖析，核对计算热点，再启动已登记八臂无训练对照。P3 未解决，`PPO_ALLOWED=false`。

## 已确认事实

`TrajectoryPlan.sample` 分别线性插值 `position`/`velocity`/`acceleration`；`sample_position`/`sample_positions` 也线性插值位置。parameterizer 按 scalar speed 决定节点时间，首节点 `velocity` 覆盖 live velocity。`VehicleDynamics` 检查节点 velocity 变化以及 displacement-derived 速度上限，但没有验证两者导数一致性，且没有检查实际 position 执行序列在节点交界的加速度。

四 plan 固定 fixture（`continuous_inward_v1`、`residual=0`、`legacy/speed_timing_v1 × reservation=0/30`）全部旧 `validate_kinematics=[]`。下表单位为 duration=`s`、加速度=`m/s²`；duration 来源于 `end_time-start_time`，只有加速度来自按 `physics_dt=.2` 的 position 采样差分，最后短 interval 使用真实 `dt`。

| mode | slot | duration (s) | max actual horizontal accel (m/s²) | max actual vertical accel (m/s²) |
|---|---:|---:|---:|---:|
| `legacy` | `0` | `65.4561997898406` | `28.0235052431024` | `6.51781141164899` |
| `legacy` | `30` | `65.4561997898406` | `28.0235052431024` | `6.51781141164899` |
| `speed_timing_v1` | `0` | `70.1783200237381` | `22.3042676892213` | `5.85048370097321` |
| `speed_timing_v1` | `30` | `86.5132317122911` | `23.3341765134967` | `6.01783603351056` |

现有 profile 的水平与垂直加速度上限分别为 `1.5` 与 `1.0`；成功诊断不等于动力学 PASS。

legacy 首段中点 position 差分速度与 reported velocity 差的 norm 为 `3.887695799217562`；新模式曲线段最大 chord-vavg residual 仍为 `1.0742706573` 与 `0.7061464605`。node chord-vavg residual 与 trapezoidal-integral 指标针对当前 linear interpolation；未来 V3 应以解析导数一致性和真实积分检查为准，不沿用 trapezoid 误差作为通用判据。首阶段修复减少 handoff 问题，但没有解决统一采样问题。

这确认了可执行性检查盲区，尚未测定其对历史 P3 学习退化的因果份额，不能据此宣称全局 P3 根因已唯一确定。当前 `terminal_marl/hardware` 不存在，不能声称真实 Crazyflie/Crazyswarm2 执行路径已经核验。

## 原拟修改范围（用户已批准的实施范围；本节保留历史提案）

1. `core/trajectory_generator.py`：新增显式 version 的可微执行轨迹表示（`executable_trajectory_v3`）。位置、速度、加速度从同一段多项式及其解析导数获得；优先五次 Hermite 段保留节点 p/v/a 和 C2 连续性，scalar/batch position 采样共用该表示。legacy 回放版本继续独立保留。
2. `core/trajectory_time_parameterization.py` 及必要的生成器 prefix：从 live p/v/a 形成连续轨迹；在几何/初速度不兼容时生成有限制动 prefix 或明确 reject，不以独立覆盖 velocity 字段表示满足初态。重新求时序，随后验证完整曲线。
3. `core/vehicle_dynamics.py`：按实际曲线导数检验速度/加速度全段极值、初态、转段与控制采样；沿用当前 profile 上限与现有数值 tolerance，旧节点字段不能充当执行加速度证明。
4. `core/safety_filter.py`、`envs/parallel_env.py` 及必要的轨迹 schema/hash 接口：安全预测与执行使用同一曲线，hash 绑定曲线版本/coefficients；采样之间采用依据曲线加速度界的间隔保守量，沿用既有安全间隔、SOC 规则、动作权限与 reward 定义。只做与新曲线表示有关的适配，不同时重设计 holder tube 或 energy/deadline 策略。
5. 覆盖 inbound、ring transfer、normal final、emergency 的仿真执行一致性；保持各动作业务语义和应急几何，硬件 adapter 实现留在现有后续路线，不新增硬件连接。

## 验收与运行顺序

- 任意段内部采样满足 `v=dp/dt`、`a=dv/dt`；初始 p/v/a 与 live state 一致；节点之间及 handoff 满足 profile 界。导数数值复核与解析界检查相互验证。
- `sample`/`sample_position`/`sample_positions` 的 position 一致；被安全检查的曲线/hash 与执行曲线/hash 相同，canonical 预约时刻匹配真实计划 departure。
- 四条当前反例及代表性 inbound/transfer/emergency fixtures 通过新检查；已知不一致轨迹必须被拒绝。legacy 50 条历史 hash 保留一致。
- 定向 core tests 与小样本 CLI 字段/序列化检查通过后，才执行已登记的相同八臂零训练开发对照（两 env seeds，1800 ticks/24 requests 每臂，source/rule authority ±2）。报告全部 outcomes、wait、elapsed、energy 与硬安全，达到 cap 标 `censored`。
- 八臂沿用既定 seeds、arm 数量与上限，但所有臂统一采用通过资格检查的 V3 物理执行基础；`legacy88` 仅指旧 final 意图映射作为动作消融，不能用未通过运动学检查的 V2 曲线作为安全/可执行性 baseline。V2 留作历史精确回放/50-hash 兼容核查。对照只在共同 V3 基础上比较 final 意图绑定与 88/90 schema；运行前同步 protocol/base_config 这一公共字段。此为待审范围内容，尚未更改当前 runner 或 matrix 配置。
- 本审核仅准入代码修复与已有零训练对照；完整训练、heldout、optimizer step、checkpoint 写入、default promotion 仍属于之后单独准入。

## 首阶段已完成证据

core commit `aff605e4f74a560853ef111d6f8c7c4900d12c1c`，14 文件 116 tests PASS；runner/protocol commit `848b2d8570f8ae9b680b98273a45e4e34fd93bec`，10 tests PASS；preflight `CHECK_ONLY_OK`（2.065 s）与三 source 各 2 tick `SMOKE_OK`（3.701 s），其中 energy coverage、requested action tape、metrics/hash 与 frozen flags 是 smoke 运行事实。该 smoke 验证接口与输出，不代表性能提升或可执行性 qualification。运动学诊断脚本及 4 项 diagnostic tests 已 PASS（commit `d67597aec27b7f703d952759647b9a974e0f7aa1`；server/local 指定文件 SHA 一致），日志 SHA256 为 `ff9f6c40d73d36f40389303ff64d302f2ef8f04bb289ec72ce3cb10439f18a78`；compact CLI 为 `KINEMATIC_CONSISTENCY_FAIL`、`technical_errors=0`、`finding_count=20`，其中 findings 是重复检查项，不是 20 个独立缺陷；artifact `.tmp/p3_trajectory_kinematic_consistency_compact_20261002.json` SHA256 为 `FC4FC02E98600174EBD9EC1DA18F725EEAC3C57D31478F1CC2AFA754E95111CA`。

相关链接：[基础复核报告](p3_foundation_review_20261002.md)、[基础修复协议](p3_foundation_repair_protocol_20261002.md)、[运动学诊断脚本](../../../terminal_marl/scripts/diagnose_trajectory_kinematic_consistency_20261002.py)。
