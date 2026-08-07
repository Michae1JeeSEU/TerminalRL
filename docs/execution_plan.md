# 执行计划（Q1 创新主线重构版）

> 本计划替代旧版 Phase 1–7 路线。后续开发以“动态终端区空域构型 + 多 UAV 降落流协同 + 安全可执行轨迹 + Crazyflie 缩尺验证”为核心推进。

## 0. 当前状态判断

当前项目已有两条历史路线：

1. `src/`：早期三环 Gym/HRL 原型，文档完整但动作空间曾直接控制 `ψ, γ, V`，不适合作为最终落地路线；
2. `terminal_marl/`：后期多智能体终端区环境，包含规则基线、VTAC、残差 PPO，更适合继续扩展。

后续原则：

```text
保留 src/ 作为 legacy prototype；
以 terminal_marl/ 为唯一主开发线；
先实现动态空域和轨迹生成器；
再重构低层交通意图动作；
再做完整 HRL 和消融实验；
最后接入 Crazyflie/Crazyswarm2 缩尺验证。
```

## 1. 阶段总览

| 阶段 | 目标 | 关键输出 | 对论文的作用 |
|---|---|---|---|
| Phase A | 研究定位与文档重构 | Q1 主线 docs | 明确 gap 与贡献 |
| Phase B | 动态空域模块 | `DynamicAirspace` + tests | 支撑核心创新 1 |
| Phase C | 轨迹生成器 | `TrajectoryGenerator` + tests | 支撑可执行性创新 |
| Phase D | 低层交通意图动作 | release/speed/timing residual | 避免自由飞行动作 |
| Phase E | SafetyFilter 统一化 | safety events + tests | 支撑实机部署 |
| Phase F | 实验模式与消融 | fixed/dynamic/full/no-safety/direct-action | 支撑 Q1 实验体系 |
| Phase G | HRL 训练闭环 | high-level + low-level policy | 支撑核心创新 2 |
| Phase H | Crazyflie lab 配置与 dry-run | `crazyflie_lab.yaml` + trajectory export | 支撑 sim-to-lab |
| Phase I | Crazyswarm2 adapter | real flight interface | 支撑实物验证 |
| Phase J | 论文实验整理 | metrics/figures/tables | 支撑投稿 |

## 2. Phase A：研究定位与文档重构

### 目标

把项目从“RL 控制无人机降落”重新定位为：

```text
Dynamic vertiport terminal airspace management
+ Multi-UAV landing-flow coordination
+ Safety-filtered executable trajectory generation
```

### 完成标准

- `docs/research_positioning.md` 明确 gap、贡献和实验矩阵；
- `requirements/design/technical/execution` 文档均围绕动态空域主线；
- `AGENTS.md` 和 `CLAUDE.md` 明确后续唯一主线为 `terminal_marl/`。

## 3. Phase B：DynamicAirspace 模块

### 目标

实现高层动态终端区空域构型调节，使 ring 半径、高度、速度和 headway 成为运行期变量。

### 任务

1. 新增 `terminal_marl/core/dynamic_airspace.py`；
2. 定义 `DynamicRingSpec`、`AirspaceBounds`、`DynamicAirspace`；
3. 实现 `apply_high_level_action(action)`；
4. 实现 ring gap、height gap、速度、边界和变化率约束；
5. 支持固定空域模式和动态空域模式；
6. 新增 `tests/test_dynamic_airspace.py`。

### 完成标准

- 输入随机高层动作，ring 参数始终合法；
- 非法动作会被 clip 并返回 violation info；
- 当前 `TerminalParallelEnv` 不启用高层时行为不变；
- 测试通过：

```bash
python -m pytest tests/test_dynamic_airspace.py -v
```

## 4. Phase C：TrajectoryGenerator 模块

### 目标

把低层交通意图转化为仿真和 Crazyflie 都可执行的轨迹。

### 任务

1. 新增 `terminal_marl/core/trajectory_generator.py`；
2. 定义 `TrajectoryPoint` 和 `TrajectoryPlan`；
3. 实现 `ring_hold`、`ring_transfer`、`final_approach`、`landing`、`emergency_hover`；
4. 加入速度、高度、时间单调性检查；
5. 新增 `tests/test_trajectory_generator.py`。

### 完成标准

- ring_hold 轨迹点近似位于指定半径和高度；
- transfer 轨迹连续，半径和高度平滑变化；
- final/landing 轨迹满足速度和高度约束；
- 测试通过：

```bash
python -m pytest tests/test_trajectory_generator.py -v
```

## 5. Phase D：低层交通意图动作重构

### 目标

把当前低层 `[heading_residual, speed_residual, altitude_residual]` 逐步改成：

```text
[release_request, speed_residual, timing_residual, slot_residual]
```

### 任务

1. 新增 action mode：`intention_residual`；
2. 保留旧 action mode 作为 legacy 或 direct-action baseline；
3. 将 release_request 接入 VTAC 阶段转移；
4. 将 speed/timing residual 接入规则基线；
5. 所有阶段转移必须经过 SafetyFilter。

### 完成标准

- rule-only 模式仍可运行；
- intention residual 模式可训练或随机 rollout；
- direct action 模式仅作为对照实验，不作为推荐路线。

## 6. Phase E：SafetyFilter 统一化

### 目标

将 VTAC、规则基线和新增轨迹检查统一为可解释的安全过滤层。

### 任务

1. 新增或重构 `terminal_marl/core/safety_filter.py`；
2. 实现 intention-level filter；
3. 实现 trajectory-level filter；
4. 记录 safety events；
5. 新增 `tests/test_safety_filter.py`。

### 完成标准

- 所有 release/transfer/final/landing 动作可被批准、修改或拒绝；
- 所有拒绝都有 violation reason；
- 后续可做 `full_hrl_no_safety` 消融。

## 7. Phase F：实验模式与消融框架

### 目标

为 Q1 论文实验提前准备可切换模式。

### 必须支持模式

```text
fixed_airspace_rule
fixed_airspace_low_rl
dynamic_airspace_rule
full_hrl
full_hrl_no_safety
direct_action_rl_baseline
```

### 任务

1. 新增统一实验配置字段 `experiment_mode`；
2. 新增 `scripts/evaluate_ablation.py`；
3. 输出统一 metrics；
4. 支持多流率、多 seed、多配置评估。

### 完成标准

- 同一脚本可运行不同模式；
- 结果保存到标准目录；
- metrics 支持论文表格和图形生成。

## 8. Phase G：完整 HRL 训练

### 目标

建立高层动态空域策略与低层交通意图策略的训练流程。

第五轮已完成 high-level PPO 训练/评估入口、dynamic-rule baseline、运行期到达、
nominal delay、统计聚合与图表骨架。第六轮进入小规模训练、参数扫描和论文结果
生成，优先形成多 seed × demand 的均值、标准差和置信区间，不接入真实硬件。

### 推荐训练顺序

```text
1. rule-only baseline
2. fixed airspace + low-level residual policy
3. dynamic airspace + rule low-level
4. full HRL
5. ablation and robustness evaluation
```

### 高层训练要点

- 低频更新；
- 奖励关注 throughput、delay、conflict risk、airspace change penalty；
- 限制半径/高度/速度变化率。

### 低层训练要点

- 高频更新；
- 奖励关注阶段推进、等待时间、动作被拒绝次数、安全约束和轨迹平滑性；
- 不奖励激进切环或贴近冲突边界的行为。

## 9. Phase H：Crazyflie lab 缩尺配置与 dry-run

### 目标

先不真实飞行，完成实验尺度配置和轨迹导出检查。

### 任务

1. 新增 `terminal_marl/configs/crazyflie_lab.yaml`；
2. 新增 `scripts/export_lab_trajectories.py`；
3. 输出 3D 轨迹图和 CSV；
4. 检查边界、速度、加速度和最小间距。

### 完成标准

- 可导出 1–3 架 Crazyflie 的完整降落流程轨迹；
- 所有轨迹在 lab boundary 内；
- 可视化无明显交叉冲突。

## 10. Phase I：Crazyswarm2 adapter 与实机验证

### 目标

把 `TrajectoryPlan` 接入 Crazyswarm2，而不是把 RL action 直接发给 Crazyflie。

### 任务

1. 新增 `terminal_marl/hardware/crazyflie_adapter.py`；
2. 新增 `terminal_marl/hardware/trajectory_executor.py`；
3. 新增 `terminal_marl/hardware/safety_monitor.py`；
4. 先完成单机 rule-only；
5. 再完成 2–3 架 rule-only；
6. 再测试 residual policy；
7. 最后测试低频 dynamic airspace。

### 完成标准

- 真实平台只执行轨迹，不直接执行裸 action；
- 有 emergency land/hover 接口；
- 记录 tracking error、safety event 和 command log。

## 11. Phase J：论文实验整理

### 目标

形成支撑 Q1 投稿的实验证据链。

### 必须输出

1. 方法总体示意图；
2. DynamicAirspace 更新过程图；
3. trajectory intention → executable trajectory 图；
4. 主要性能对比表；
5. 消融实验表；
6. 不同 traffic demand 下鲁棒性曲线；
7. safety violation / near miss 对比；
8. Crazyflie 缩尺验证照片、轨迹图、误差统计。

### 最终判断标准

论文不能只证明“RL 能飞”，而要证明：

```text
动态空域构型确实优于固定结构；
高层空域调节与低层降落协同具有互补性；
安全过滤和轨迹生成使策略具备真实平台可执行性；
Crazyflie 实验验证了仿真策略的缩尺落地能力。
```

## 12. Phase G-Server：服务器训练准备

第六轮将 high-level PPO 从本地 smoke 入口扩展为服务器训练入口，但仍不在本地进行正式训练：

- 并行环境：`n_envs` + `dummy/subproc` vector env；
- CPU 默认：`device=cpu`，脚本设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`；
- 训练阶段：Stage 0 smoke、Stage 1 pilot、Stage 2 三 seed 小规模、Stage 3 五 seed 主实验、Stage 4 reward/LR sweep、Stage 5 统一评估聚合绘图；
- 输出：checkpoint、training progress、eval summary、config snapshot、episode logs、aggregate CSV 和 figures；
- 打包：`scripts/server/pack_for_server.py` 生成服务器上传包和 manifest。

该阶段仍然只训练 high-level dynamic airspace reconfiguration policy，不接入真实硬件，不改变底层飞控边界。

## Phase G-Readiness hotfix: dynamic-rule pipeline and reward sanity

This hotfix sits before any new PPO pilot.  It is a simulation-only readiness
gate for the server training workflow.

Completed tasks:

1. Diagnose fixed-rule versus dynamic-rule landing-pipeline behavior on the
   server mirror.
2. Repair the dynamic-rule high-level heuristic so it no longer starves
   `inner -> final -> land` transitions through excessive geometry/headway
   changes.
3. Normalize pairwise conflict, waiting, and congestion metrics for reward and
   logging.
4. Add phase-progress reward and detailed reward component statistics.
5. Add low-level-step phase transition diagnostics, high-level action isolation
   modes, and rule-only capacity calibration.
6. Verify locally with the full test suite and remotely with targeted tests and
   server dry-runs.

Gate to proceed:

```text
local pytest: all pass
server targeted pytest: pass
single_rule: 1/1 landed
six_rule: >=5/6 landed
six_dynamic: >0 landed and >=80% of six_rule
reward diagnostics: no 20x component dominance warning
capacity calibration: recommended demand intervals generated
```

Next execution step:

Run Stage-0/Stage-1 high-level PPO smoke/pilot only after reviewing
`training_readiness.json`, `reward_component_summary.csv`,
`capacity_calibration.csv`, and `recommended_pilot_config.yaml` under the server
output root.  Do not launch a long multi-seed sweep until the smoke/pilot logs
show non-zero traffic, non-zero phase progress, and plausible reward component
scales.
# Event-gated 180s HRL execution addendum

For the current phase, execute in this order: audit 8D action/timing semantics;
calibrate geometry-gate thresholds from fixed-geometry traces; validate
`fixed_rule` and `event_rule` regressions; run 512-step event-gated smoke from
scratch; run exact 5k `event_gated_hrl` and `periodic_hrl` pilots from scratch;
only if ready, run exact 20k event-gated and periodic pilots from scratch;
evaluate and stop. Do not continue to 50k/main/sweep or hardware experiments.
