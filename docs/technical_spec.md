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

### 4.1 观察空间设计（初稿）

```
高层观察 (per option):
  - 当前各环参数 (R, H, V) × 3 = 9维
  - 各环 UAV 数量 × 3 = 3维
  - 平均等待时间 × 3 = 3维
  - 吞吐量滑动平均 × 3 = 3维
  合计: ~18维

低层观察 (per UAV):
  - UAV 自身状态: 位置(x,y,z), 速度, 当前所在环, 剩余电量 = 7维
  - 目标环信息: 半径, 高度, 速度 = 3维
  - 邻近 UAV 相对位置: K × 3维 (最近K架)
  - 当前空域构型参数: 9维
  合计: ~19 + 3K 维
```

### 4.2 动作空间设计（初稿）

```python
# 高层动作
high_level_space = gym.spaces.Dict({
    "radii": gym.spaces.Box(low=[R_min, R_min, R_min],
                            high=[R_max, R_max, R_max], shape=(3,)),
    "heights": gym.spaces.Box(low=[H_min, H_min, H_min],
                              high=[H_max, H_max, H_max], shape=(3,)),
    "speeds": gym.spaces.Box(low=[V_min, V_min, V_min],
                             high=[V_max, V_max, V_max], shape=(3,)),
})

# 低层动作
low_level_space = gym.spaces.Dict({
    "control": gym.spaces.Discrete(3),       # 盘旋/悬停/下降
    "waypoints": gym.spaces.Box(             # 下降路径点（仅下降时有效）
        low=-1, high=1, shape=(K, 3)         # K个路径点 × (dx, dy, dz)
    ),
    "landing_zone": gym.spaces.Discrete(M),  # 降落区域（仅内环+降落时有效）
})
```

## 5. 仿真步长

| 层级 | 决策间隔 | 仿真步长 |
|------|----------|----------|
| 高层 (option) | ~30 min（仿真时间） | 1800 秒 = 1 step |
| 低层 (control) | ~1-5 s（仿真时间） | 1-5 秒 = 1 step |

一个高层 step 包含约 360-1800 个低层 step。

## 6. 平台

| 项目 | 说明 |
|------|------|
| OS | Windows 11 (开发环境) |
| GPU | 不需要（初期纯数值仿真） |
| CI/CD | 暂不配置 |

## 7. 配置文件格式

环境配置使用 YAML 或 Python dict，示例：

```python
DEFAULT_CONFIG = {
    "num_uavs": 50,
    "num_landing_zones": 6,
    "ring": {
        "r_min": 500.0,   # 内环最小半径 (m)
        "r_max": 5000.0,  # 外环最大半径 (m)
        "h_min": 50.0,    # 内环最低高度 (m)
        "h_max": 1000.0,  # 外环最高高度 (m)
        "v_min": 5.0,     # 最低巡航速度 (m/s)
        "v_max": 30.0,    # 最高巡航速度 (m/s)
    },
    "uav": {
        "max_speed": 40.0,
        "max_battery": 100.0,     # 最大电量 (归一化)
        "safe_separation": 50.0,  # 安全间隔 (m)
        "battery_critical": 0.15, # 临界电量阈值
    },
    "simulation": {
        "dt_high": 1800.0,  # 高层决策间隔 (s)
        "dt_low": 2.0,      # 低层决策间隔 (s)
    },
}
```
