# CODEX_NEXT_STEPS（Q1 创新主线版）

> 本文件用于交给 Codex 继续开发。请先阅读 `AGENTS.md`、`docs/research_positioning.md`、`docs/requirements.md`、`docs/design_spec.md`、`docs/technical_spec.md`、`docs/execution_plan.md`，再执行本文件任务。

## 当前开发状态

前五轮基础工程已经完成：

1. **第一轮**：实现 `DynamicAirspace`、`TrajectoryGenerator` 及其测试；
2. **第二轮**：实现 `ExperimentMode`、fixed/dynamic 环境接入、`crazyflie_lab` preset、`SafetyFilter` 骨架和 `intention_residual` 最小闭环；
3. **第三轮**：
   - 实现 YAML → `SimConfig` 加载器，并增加 fixed/dynamic/lab presets；
   - 建立显式 high-level interval、动作缓存和高层观测接口；
   - 将 SafetyFilter 接入 outer→inner、inner→final、final→land 审批；
   - 完善 trajectory boundary/time/speed/acceleration/height/min-segment 检查；
   - 建立 rule-only/random-policy/dry-run 消融 runner 和统一指标。
4. **第四轮**：
   - 新增 Gym-style `HighLevelHRLWrapper`，一个高层动作执行完整低层时间区间；
   - 新增 interval-level high-level reward 与可解释 diagnostics；
   - 增加 nominal landing schedule、严格 landing delay 及分位数指标；
   - 建立 episode summary、step、airspace、safety、agent 五类日志；
   - 支持 multi-seed、low/medium/high/bursty demand 和多 episode sweep；
   - 增加 simulation-only `full_hrl_no_safety` 消融；
   - 增加 high-level-random 与 high-level-static baseline。
5. **第五轮**：
   - 接入独立 SB3 high-level PPO 训练与模型评估入口；
   - 增加 dynamic-rule 启发式空域调节 baseline；
   - 增加 deterministic/Poisson/bursty runtime arrival process；
   - 增加 analytic/rule-only-reference nominal schedule；
   - 输出 mean/std/min/max/median/95% CI 聚合结果；
   - 建立 throughput、delay、conflict、safety 和轨迹图生成骨架。

以上实现仍然全部属于 simulation/dry-run，不包含 ROS2、Crazyswarm2 或真实 Crazyflie 控制。

## 一、核心原则

后续开发必须服务于论文主线：

```text
Dynamic terminal airspace reconfiguration
+ Multi-UAV landing-flow coordination
+ Safety-filtered executable trajectory generation
+ Crazyflie scaled validation
```

代码主线：

- `terminal_marl/`：唯一主开发线；
- `src/`：legacy prototype，仅参考，不新增主功能。

禁止事项：

- 不要把最终路线继续扩展为自由 `heading/altitude/speed` 飞行动作控制；
- 不要把 ROS2/Crazyswarm2 import 到 `terminal_marl/core/`；
- 不要删除旧代码；
- 不要跳过测试直接接入真实 Crazyflie；
- 不要把 Crazyflie 写成研究对象本身，它只是缩尺验证平台。

## 二、第一轮开发目标

第一轮只做核心基础模块，不训练模型，不接入 ROS2，不真实飞行。

目标：

1. 新增动态空域模块；
2. 新增轨迹生成器初始结构；
3. 新增测试；
4. 保持现有主流程兼容。

## 三、Task 1：新增 DynamicAirspace

创建：

```text
terminal_marl/core/dynamic_airspace.py
```

实现：

```python
@dataclass
class DynamicRingSpec:
    index: int
    radius: float
    height: float
    speed: float
    capacity: int

@dataclass
class AirspaceBounds:
    radius_min: float
    radius_max: float
    height_min: float
    height_max: float
    speed_min: float
    speed_max: float
    min_ring_gap: float
    min_height_gap: float
    max_radius_change_per_update: float
    max_height_change_per_update: float
    max_speed_change_per_update: float
    headway_min: float
    headway_max: float

class DynamicAirspace:
    def __init__(...): ...
    def state_vector(self) -> np.ndarray: ...
    def apply_high_level_action(self, action: np.ndarray) -> dict: ...
    def validate(self) -> bool: ...
    def to_ring_specs(self) -> list: ...
    def get_headways(self) -> dict: ...
```

要求：

- 高层 action shape 初期为 `(8,)`；
- action 从 `[-1, 1]` 映射为小幅增量；
- 更新对象包括 `R_outer, R_inner, H_outer, H_inner, V_outer, V_inner, headway_ring, headway_final`；
- 必须保证 `R_outer > R_inner + min_ring_gap`；
- 必须保证 `H_outer > H_inner + min_height_gap`；
- 所有参数必须 clip 到 bounds；
- 返回 info，例如：

```python
{"clipped": bool, "violations": list[str], "old_state": ..., "new_state": ...}
```

## 四、Task 2：新增 DynamicAirspace 测试

创建：

```text
tests/test_dynamic_airspace.py
```

至少测试：

1. 初始化状态合法；
2. 随机 action 后状态仍合法；
3. 极端 action 会被 clip；
4. ring gap 和 height gap 永远满足；
5. `state_vector()` shape 稳定；
6. `to_ring_specs()` 能兼容现有 ring spec 或返回可被现有代码读取的结构。

## 五、Task 3：新增 TrajectoryGenerator

创建：

```text
terminal_marl/core/trajectory_generator.py
```

实现：

```python
@dataclass
class TrajectoryPoint:
    x: float
    y: float
    z: float
    yaw: float
    t: float

@dataclass
class TrajectoryPlan:
    uav_id: str
    mode: str
    points: list[TrajectoryPoint]
    metadata: dict

class TrajectoryGenerator:
    def ring_hold(self, uav_state, ring, duration: float, dt: float) -> TrajectoryPlan: ...
    def ring_transfer(self, uav_state, from_ring, to_ring, duration: float, dt: float) -> TrajectoryPlan: ...
    def final_approach(self, uav_state, landing_pad, duration: float, dt: float) -> TrajectoryPlan: ...
    def landing(self, uav_state, landing_pad, duration: float, dt: float) -> TrajectoryPlan: ...
    def emergency_hover(self, uav_state, duration: float, dt: float) -> TrajectoryPlan: ...
```

本轮重点：

- 先实现 `ring_hold`；
- 其他函数可以给出可运行的简单版本或清晰 TODO；
- 不依赖 ROS2；
- 输出 trajectory points 可用于 dry-run 和绘图。

## 六、Task 4：新增 TrajectoryGenerator 测试

创建：

```text
tests/test_trajectory_generator.py
```

至少测试：

1. `ring_hold` 输出点数正确；
2. 时间严格递增；
3. 所有点近似位于目标 ring 半径；
4. 所有点高度近似等于 ring height；
5. yaw 为有限值；
6. `TrajectoryPlan.mode == "ring_hold"`。

## 七、兼容性要求

本轮开发后必须满足：

- 不破坏现有 `terminal_marl` imports；
- 不破坏当前训练/评估脚本；
- 不强制安装 ROS2 或 Crazyswarm2；
- 不删除 `src/`；
- 不修改大规模训练逻辑，除非只是为了 import 兼容。

## 八、运行测试

完成后运行：

```bash
python -m pytest tests/test_dynamic_airspace.py tests/test_trajectory_generator.py -v
```

如果项目现有测试存在与本轮无关的问题，请单独说明，不要为了通过测试删除功能。

## 九、完成后汇报格式

请按以下格式汇报：

```text
1. 修改/新增文件
2. 新增核心类和函数
3. 与论文主线的对应关系
4. 测试命令和结果
5. 兼容性说明
6. 下一轮建议
```

## 十、第六轮建议

下一轮继续围绕 HRL 训练证据链，而不是底层飞控：

1. 执行多 seed × demand × mode 小规模 PPO 参数扫描；
2. 固化 low-level intention policy checkpoint 接口；
3. 校准 reward 权重和 rule-only-reference delay；
4. 生成论文主表、置信区间和初版性能图；
5. 增加训练稳定性、泛化和 safety ablation 分析；
6. 增加 Crazyflie lab 轨迹 CSV/3D dry-run 导出，仍不连接真实硬件。

## 第六轮完成记录：服务器训练准备

第六轮不在本地正式训练，只完成 simulation-only 的服务器训练准备：

1. 扩展 `terminal_marl/training/train_high_level_ppo.py`：
   - 支持 `--n-envs`、`--vec-env dummy/subproc`、`--device cpu/cuda/auto`、`--num-threads`；
   - 支持 `--checkpoint-freq`、`--eval-freq`、`--save-replay-traces`；
   - 支持 `--resume-from`；
   - 输出 `training_progress.csv`、`eval_summary.json`、`config_snapshot.json`。
2. 新增 `terminal_marl/training/training_config.py` 与 `terminal_marl/configs/training/*.yaml`，用于服务器训练 YAML 参数加载；显式 CLI 参数覆盖 YAML。
3. 新增 `terminal_marl/training/evaluate_high_level_policy.py`，用于加载训练后的 high-level PPO policy 并进行多 seed、多 demand、多 episode 评估。
4. 新增 `scripts/server/`：
   - 环境检查、smoke、pilot、main、sweep、评估、聚合绘图脚本；
   - `pack_for_server.py` 可生成 `terminal_server_training_package.zip` 与 `SERVER_UPLOAD_MANIFEST.txt`；
   - `README_SERVER_TRAINING.md` 记录无 sudo Linux/SLURM 使用流程。
5. 新增 `terminal_marl/experiments/reward_diagnostics.py`，用于在正式训练前检查 high-level reward component 是否被单项支配。
6. 新增 `terminal_marl/experiments/experiment_matrix.py`，导出论文主实验矩阵。

约束保持不变：不接入 ROS2/Crazyswarm2/真实 Crazyflie；PPO 只训练 high-level dynamic airspace policy；不扩展自由 heading/altitude/velocity-vector 控制。

## 第七轮建议

下一轮建议从“服务器产物回收与论文证据链”推进：

1. 在服务器执行 Stage 0/1 smoke 与 pilot 后，回收 `episode_summary.json`、`aggregate_results.csv`、`aggregate_summary.csv` 和 figures；
2. 根据 `reward_diagnostics.csv` 调整 high-level reward 权重，避免 safety 或 airspace-change 单项支配；
3. 固化 high-level PPO checkpoint 命名、评估目录规范和论文表格生成脚本；
4. 增加 `dynamic_hrl + high-level-ppo` 与 `dynamic_rule`、`high-level-static`、`full_hrl_no_safety` 的统一显著性/置信区间表；
5. 继续保持 simulation-only，除非后续明确进入 Crazyflie lab dry-run trajectory export。

## 服务器 pilot 问题修复记录：traffic / logging / reward sanity

服务器 pilot 暴露的问题：

- `dynamic_hrl` high-level PPO 从 10k 到 100k steps `landed_count=0`；
- `final_eval.csv` 中 `active_count=0`，但 `safety_override_count` 持续增长；
- `reward_diagnostics.py` CLI 与文档不一致，缺少 `--config` 与 `--demand-level`。

本轮修复：

1. `reward_diagnostics.py` 支持 `--config`、`--demand-level` 与 `--demand-levels`，其中 `--demand-level medium` 等价于单元素 demand list。
2. `HighLevelHRLWrapper` 不再在 `env.agents` 为空时直接停止低层推进，因此纯 runtime-arrival 场景仍可随时间生成 UAV。
3. `TerminalParallelEnv.metrics()` 增加 `active_count`、`landed_count`、`total_spawned_count`、`total_landed_count`、`total_safety_override_count`，避免 eval trace 静默写 0。
4. `final_eval.csv` 增加区间计数字段：
   - `active_count_start`
   - `active_count_mean`
   - `active_count_max`
   - `active_count_end`
   - `spawned_count_delta`
   - `landed_count_delta`
   - `total_spawned_count`
   - `total_landed_count`
   - `safety_override_delta`
   - `total_safety_override_count`
5. high-level reward 继续使用 interval delta/average 口径，并在 diagnostics 中同时输出 cumulative count，避免把累计 safety override 直接作为区间惩罚。
6. 新增 `--fail-if-no-traffic`、`--min-spawned-before-training-check`、`--traffic-check-steps`、`--sanity-check-baseline`，在 PPO learn 前做 no-traffic fail-fast 与 rule-only baseline sanity check。
7. 新增 `terminal_marl/experiments/debug_arrivals.py` 与 `terminal_marl/training/check_parallel.py`，用于低耗时本地排查 arrivals 与并行 step 性能。
8. 新增 `ppo_debug_local.yaml`，用于 256/512 timesteps 的本地 tiny PPO 通路验证。

重新上服务器前建议检查：

```bash
python -m terminal_marl.experiments.debug_arrivals --config dynamic_hrl --demand-level medium --policy rule-only --high-level-steps 8 --output-dir runs/debug_arrivals_local
python -m terminal_marl.experiments.reward_diagnostics --config dynamic_hrl --demand-level medium --policies rule-only dynamic-rule high-level-random --episodes 1 --high-level-steps 8 --output-dir runs/reward_diagnostics_local
python -m terminal_marl.training.train_high_level_ppo --training-config terminal_marl/configs/training/ppo_debug_local.yaml --total-timesteps 256 --n-envs 1 --vec-env dummy --fail-if-no-traffic --sanity-check-baseline --output-dir runs/high_level_ppo/ppo_debug_local
```

若 `debug_arrivals.csv` 中 `total_spawned_count=0` 且 `active_count_max=0`，不要提交服务器 pilot；先检查 `arrival_flow`、`demand_level` override 和 `HighLevelHRLWrapper` reset/step 路径。
