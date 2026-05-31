# 技术规范

## 1. 开发语言与运行环境

| 项目 | 选型 | 说明 |
|------|------|------|
| 语言 | Python 3.10+ | 主流 RL 生态支持 |
| 包管理 | pip / conda | requirements.txt 管理依赖 |
| 虚拟环境 | venv 或 conda env | 隔离项目依赖 |

## 2. 核心依赖

### 2.1 强化学习

| 包 | 版本 | 用途 |
|----|------|------|
| gymnasium | >= 0.29 | 环境接口标准 |
| stable-baselines3 | >= 2.0 | RL 算法实现（PPO, SAC 等） |
| ray[rllib] | >= 2.0（Phase 7 引入） | 大规模分布式 RL |

### 2.2 数值与科学计算

| 包 | 版本 | 用途 |
|----|------|------|
| numpy | >= 1.24 | 数组运算 |
| scipy | >= 1.10 | 优化与统计工具 |

### 2.3 可视化与日志

| 包 | 版本 | 用途 |
|----|------|------|
| matplotlib | >= 3.7 | 基础绑图 |
| tensorboard | 随 SB3 安装 | 训练曲线 |
| wandb | >= 0.15（可选） | 实验管理 |

### 2.4 代码质量

| 工具 | 用途 |
|------|------|
| black | 代码格式化 |
| isort | import 排序 |
| mypy | 静态类型检查（可选） |

## 3. 代码规范

- 所有公开函数/类必须有类型注解
- 模块命名：snake_case，类命名：PascalCase
- 每个模块文件顶部包含一行 docstring 说明用途
- 环境接口严格遵循 Gymnasium Env 规范

## 4. 环境接口规范

```python
class TerminalAreaEnv(gym.Env):
    """终端区降落环境，遵循 Gymnasium 接口"""

    def __init__(self, config: dict): ...
    def reset(self, *, seed=None, options=None) -> tuple[ObsType, dict]: ...
    def step(self, action: ActionType) -> tuple[ObsType, float, bool, bool, dict]: ...
    def render(self) -> None | np.ndarray: ...
    def close(self): ...

    @property
    def observation_space(self) -> gym.Space: ...
    @property
    def action_space(self) -> gym.Space: ...
```

### 4.1 观察空间设计（当前实现）

```
Per-UAV 局部观测矩阵 (max_n, 91):
  - 空域状态 (9): R1,R2,R3,H1,H2,H3,V1,V2,V3 归一化
  - 自身状态 (9): sin(θ), cos(θ), ring_id/3, r/R1, z/H1, battery/max, priority/5, wait/3600, can_descend
  - k 近邻 (k×6): 每邻居 [sin(θ), cos(θ), ring_id/3, r/R1, z/H1, battery/max]，k=8 默认
  - 环计数 (3): 各环 UAV 数 / 活跃总数
  - LZ 占用 (num_landing_zones): 各 LZ 剩余占用时间归一化，6 LZ
  - 车流上下文 (12): nearest_3d, predicted_min_dist, front_arc_gap, back_arc_gap, front_closing_speed, back_closing_speed, front_ttc, ring_mean_gap, executed_circle_speed, requested_circle_speed, last_safety_active, descent_blocked
  - Safety 介入信号 (4): speed_mult, psi_offset, gamma_offset, blocked
  合计: 9 + 9 + 48 + 3 + 6 + 12 + 4 = 91 维
```

- max_n = 100（最大并发 UAV），inactive UAV 行全零
- 邻居按 3D 距离取最近 k 个，不足补零
- 车流上下文提供显式弧距、闭合速度、TTC 与请求/执行速度；介入信号来自 SafetyController，让 RL 学会通过自主调速避开干预

### 4.2 动作空间设计（当前实现）

```python
# 高层动作（9维连续，每 high_level_interval 步应用一次）
high_level_action_space = spaces.Box(
    low=[R_min*3, H_min*3, V_min*3],
    high=[R_max*3, H_max*3, V_max*3],
)

# 低层动作（per-UAV）
low_level_action_space = spaces.Dict({
    "control": spaces.MultiDiscrete([2] * max_n),        # 0=CIRCLE, 1=DESCEND
    "psi": spaces.Box(0, 2π, shape=(max_n,)),            # 航向角
    "gamma": spaces.Box(-π/2, π/2, shape=(max_n,)),      # 航迹角（负=下降）
    "speed": spaces.Box(5, 40, shape=(max_n,)),          # 飞行速度 m/s
    "landing_zone": spaces.MultiDiscrete([m+1] * max_n), # 0..m-1 选 LZ，m=no-op
})
```

**训练动作空间（Phase 3b 安全修订）**：PerUAVDiscreteWrapper 使用 `Box(low=-1, high=1, shape=(max_n, 2))`，每 UAV 独立输出：

| dim | 名称 | 缩放目标 | 生效 |
|-----|------|---------|------|
| 0 | control_raw | > 0 → DESCEND, ≤ 0 → CIRCLE | 始终 |
| 1 | circle_speed_raw | [-1,1] → [0.3, 1.3] 环速系数 | CIRCLE |

DESCEND 现在只表示“下降请求”。环境规则层负责生成 (ψ, γ, V)、选择可用 LZ、执行下降放行计时、预测冲突过滤和目标环捕获；RL 不再直接控制自由下降航迹。

**动作约束（环境层强制）**：
- 环驻留不足 → CIRCLE 强制覆盖 DESCEND
- LZ 被占用/预留 → CIRCLE 强制覆盖 DESCEND
- CIRCLE 动作 → UAV 吸附到当前环轨道 (R_i, H_i) 锁定
- 同环日常间距由 PPO 自主调速；仅进入 shield 紧急缓冲区且后机仍在追近时触发最小干预刹车
- 下降请求不满足放行间隔或预测安全距离 → CIRCLE 强制覆盖 DESCEND

## 5. 仿真步长

| 层级 | 决策间隔 | 仿真步长 | 说明 |
|------|----------|----------|------|
| 高层 (option) | ~30 min（仿真时间） | 1800 秒 | 每 900 low-level steps 调用一次 |
| 低层 (control) | 1 s（仿真时间） | 1 秒 | 每步推进所有 UAV 1s |

Phase 3b 训练用 max_steps=1200（20 分钟仿真），dt=1.0s。

## 6. 平台

| 项目 | 说明 |
|------|------|
| OS | Windows 11 (开发环境) |
| GPU | 不需要（初期纯数值仿真） |
| CI/CD | 暂不配置 |

## 7. 配置文件格式

环境配置使用 Python dataclass（`src/config.py`），实际默认值：

```python
# 环配置
RingConfig:
  r_min=500.0, r_max=5000.0          # 半径范围 (m)
  h_min=50.0, h_max=1000.0           # 高度范围 (m)
  v_min=5.0, v_max=30.0              # 速度范围 (m/s)
  default_radii=(4000, 2500, 1000)   # 默认外→内环半径
  default_heights=(800, 500, 200)    # 默认外→内环高度
  default_speeds=(25, 20, 15)        # 默认外→内环速度

# UAV 配置
UAVConfig:
  max_speed=40.0, max_battery=100.0
  battery_critical=0.15               # 临界电量 (15%)
  safe_separation=10.0                # 碰撞危险区 (m)
  near_miss_threshold=50.0            # 近距预警区 (m)
  min_ring_wait=30.0                  # 每环最少盘旋时间 (s)
  min_ring_spacing=100.0              # 同环 UAV 最小角间距 (m 弧长)
  proximity_penalty_threshold=30.0    # 接近惩罚触发距离 (m)
  proximity_penalty_scale=5.0         # 接近惩罚缩放系数
  safety_margin_threshold=100.0       # 最近距离观测归一化范围 (m)
  ring_gap_observation_scale=1000.0   # 同环弧距观测归一化范围 (m)
  ttc_observation_horizon=60.0        # TTC 观测归一化范围 (s)
  power_cruise=1.0, power_descend=0.7 # 功耗 (单位/s)
  priority_emergency=5                # 满豁免阈值
  priority_alpha=0.5, priority_beta=1.5   # 等待时间优先级参数
  priority_gamma=1.0, priority_delta=2.0  # 电量优先级参数

# Safety 配置（SafetyController 参数）★ 新增
SafetyConfig:
  angular_separation_threshold=25.0   # Rule 1: 最终紧急介入距离 (m)
  initial_intervention_threshold=40.0 # 训练初期 shield 介入距离 (m)
  final_intervention_threshold=25.0   # 训练后期 shield 介入距离 (m)
  intervention_curriculum_episodes=160
  descend_crossing_lookahead=10.0     # Rule 2: 预测窗口 (s)
  descend_crossing_slowdown=0.85      # Rule 2: 盘旋者减速系数
  dual_descend_threshold=50.0         # Rule 3: 触发距离 (m)
  dual_descend_psi_offset=15.0        # Rule 3: 偏转角 (deg)
  universal_brake_threshold=25.0      # Rule 4: 最终触发距离 (m)
  universal_brake_low=0.70            # Rule 4: 低优先减速
  universal_brake_high=0.85           # Rule 4: 高优先减速
  emergency_yield_slowdown=0.60       # Rule 5: 让路减速
  emergency_yield_psi=25.0            # Rule 5: 让路偏转 (deg)

# 仿真配置
SimConfig:
  dt_high=1800.0, dt_low=2.0          # 高低层步长 (s)
  max_steps=7200                       # 默认最大步数
  num_landing_zones=6                  # 着陆区数量

# 环境总配置
EnvConfig:
  num_uavs=50                          # 默认 UAV 数量
  uav_inflow_rate=0.0                  # 流入率 (UAVs/h)，0=无流入
  max_uavs=100                         # 最大并发 UAV 数
  k_nearest=8                          # 局部观测邻居数
  safety_enabled=True                   # 启用 SafetyController ★ 新增
  reward_landing=2.0                  # 每架新着陆奖励，乘 w_throughput
  penalty_conflict=500.0               # 碰撞基础惩罚
  penalty_near_miss=2.0                # 近距预警惩罚（按受控 UAV 数归一化）
  penalty_safety_rule=0.25             # 每条 SafetyController 规则介入惩罚
  penalty_safety_speed=1.0             # 安全层减速幅度惩罚
  penalty_safety_block=2.0             # SafetyController 阻塞惩罚
  penalty_descent_block=0.0            # DESCEND 放行阻塞仅作观测信号，不默认惩罚
  penalty_ring_spacing=0.20            # 同环短前向弧距 softplus 惩罚
  penalty_ring_ttc=0.10                # 同环短 TTC softplus 惩罚
  penalty_speed_smooth=0.10            # 盘旋速度请求振荡惩罚
  ring_spacing_softness=20.0           # 弧距 softplus 过渡宽度 (m)
  ring_ttc_threshold=12.0              # TTC 开始惩罚阈值 (s)
  ring_ttc_softness=3.0                # TTC softplus 过渡宽度 (s)
  w_energy=0.3, w_wait=1.0, ...       # 奖励权重
```

**任务类型与 LZ 占用**：
```python
MISSION_TYPE_LZ_OCCUPANCY = {1:30.0, 2:120.0, 3:10.0, 4:60.0}  # 秒
MISSION_TYPE_WEIGHTS = [0.45, 0.25, 0.10, 0.20]                  # 生成概率
```
