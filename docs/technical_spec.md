# 阶段1技术规范

## 文件与依赖边界

| 路径 | 内容 |
|---|---|
| `terminal_marl/core/fixed_terminal/` | 新固定终端区模型与可行性实现 |
| `terminal_marl/envs/fixed_terminal_env.py` | 新仿真环境 |
| `terminal_marl/configs/fixed_terminal_stage1.py` | 冻结阶段1默认值 |
| `terminal_marl/scripts/evaluate_fixed_terminal_stage1.py` | 评价入口 |
| `tests/test_fixed_terminal*.py` | 针对性测试 |

新入口不要求加载旧策略权重、88D observation、V4.2 action、teacher、Gym训练课程、ROS2或硬件连接。已有历史代码保留原用途。

## 核心配置与公开接口

`TerminalConfig`、`Ring`、`EnergyRates` 是冻结 dataclass；`DEFAULT_CONFIG`、只读 `LOAD_PROFILES` / `SOC_PROFILES` 与 `STANDARD_SEEDS=(101,202,303)` 定义阶段1配置。环索引内到外为 0、1、2，几何容量为 25、41、58，运营容量为 6、8、9；公式和全部数值见[问题定义](fixed_terminal_problem_definition.md)。

| 模块 | 已实现接口 | 语义 |
|---|---|---|
| `domain.py` | `TerminalConfig`, `Aircraft`, `Intent`, `Plan`, `Segment`, `Reservation` | 几何、飞机状态、联合等待意图、轨迹与资源记录 |
| `routes.py` | `build_plan(aircraft, now, config, wait_cycles=(), emergency=False)` | 构建规范路线；普通等待向量按内环到当前环排列 |
| `routes.py` | `plan_energy(plan, config, start=None, end=None)` | 累计指定时间区间实际路径耗能 |
| `routes.py` | `required_energy(aircraft, now, config, wait_cycles=(), emergency=False)` | 路径总能耗加最低储备 |
| `safety.py` | `Feasibility(config).check(plan, aircraft, fleet, now, *, check_emergency=True, search_limit=256)` | 独立验证规范路线、能源、资源、容量与连续安全 |
| `safety.py` | `search_normal(aircraft, fleet, now, limit=256)` / `search_emergency(...)` | SOC 有界候选域搜索；计算预算耗尽返回未知 |
| `scenarios.py` | `generate_scenario(config, seed=101, load="low", soc_profile="high")` | 生成初态及观察窗内全部请求，独立需求/SOC随机流 |
| `scenarios.py` | `truncated_normal(rng, mean, std, lower, upper)` | 拒绝采样截断正态 |

`Check` 包含 `status, reason`；`SearchResult` 包含 `status, plan, searched, reason`。状态为 `feasible / infeasible / unknown`；有限预算或连续间隔未获证明不得改记不可行。设置 `limit=None` 可完整枚举 SOC 允许的正常域；计划完成时间不受观察窗限制。应急前的正常不可行只针对当前已承诺预约和未预约飞机持续保持预测。

环境为 `FixedTerminalEnv(config=DEFAULT_CONFIG, scenario=None)`，提供 `reset()`、`step(intents=())`、`propose(intent)`、`submit(intent, search_limit=256)`、`exercise_feasibility(search_limit=256)` 与 `report()`。`Intent(aircraft_id, wait_cycles=(), emergency=False)` 的正常向量按内环到当前环排列；应急向量只有当前环的一项。`exercise_feasibility` 按确定飞机顺序调用候选搜索，仅用于运行基础设施检查，不是强调度基线。

CLI 示例（输出目录必须尚不存在）：

```powershell
python -m terminal_marl.scripts.evaluate_fixed_terminal_stage1 --load low --soc high --seed 101 --output D:\7_codespace\training_outputs\terminal\fixed_stage1\low-high-101
```

可选参数：`--load {low,medium,high}`、`--soc {high,medium,low}`、`--seed`（默认 101）、`--search-limit`（默认 256，0 表示穷尽）、`--matrix`（指定种子的九个负载/SOC组合）、`--output`（必填）。每个场景目录产生 `frozen_input.json`、`result.json`、`requests.jsonl`、`events.jsonl`；输入文件在首次模拟结果之前写入。CLI 调用本身不意味着运行域已通过审查。

配置至少明确固定几何、空间与时间间隔、运动学界限、分模式能耗、最低储备、pad服务、仿真步长与观察窗、负载/SOC情景及两个独立随机流的种子。配置在 pilot 前冻结，输出保存实际参数与种子，不创建额外登记系统。

## 输出约定

评价必须保留总请求数及互斥结局计数（正常、应急、失败、截尾），外部等待/积压和已准入状态。请求级记录应能复核到达、准入、完成/终止、SOC消耗和原因。汇总区分全请求观察时间、成功子集完成时间、正常吞吐及应急代价；详细公式见问题定义。报告 schema 为 `fixed-terminal-stage1-v1`，主要字段为 `config/scenario/capacity/assumptions/summary/requests`；完整观察窗请求结局为 `normal_landed/emergency_landed/failed_energy/censored`；中间报告另有 `not_arrived`，由 `arrived_request_count` 区分已到达数。截尾前状态由 `state_at_cutoff` 保存，汇总提供 `state_at_cutoff_counts/censored_state_counts`。`landing_metric_denominator` 明示完成时间和延迟统计分母；`energy_mean_soc_all_requests` 保留全请求分母。正常/应急比例均以全请求计；两类着陆各自提供吞吐、完成时长、延迟与样本数。`observed_request_exposure_seconds` 是全请求已观察在系统时长之和。准入拒绝次数与原因保存在 `admission_rejection_counts` 和 `admission_rejected` 事件，不是唯一拒绝请求数。

未知正常可行性与已证明正常不可行必须可区分。搜索规模/耗时不得被解释为物理安全结论。输出路径显式指定，运行产物不写入项目源码；服务器统一写入 `$HOME/training_outputs/terminal`。

## 针对性检查

新主线只收集三个针对性测试文件，避免额外加载旧训练依赖：

```bash
python -m pytest tests/test_fixed_terminal_domain.py tests/test_fixed_terminal_safety.py tests/test_fixed_terminal_env.py -q
```

上述三个文件已完成 32 项测试；九格单种子 pilot 和输出重算记录见[验证记录](evidence/fixed_terminal_stage1/validation_20261013.md)，滚动进度见[当前状态](CURRENT_STATUS.md)。先检查小样本 JSON 等输出字段、分母和序列化，再启动长运行；服务器启动与监控继续遵守根目录 `AGENTS.md`。
