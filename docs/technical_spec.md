# 固定终端区与阶段3机制技术规范

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

## 阶段2接口与依赖状态

阶段1的物理配置和现有单计划API保持冻结。阶段2增加以下接口，已按当前实现核对；测试与最终运行状态独立记录。

现有 `D:\7_codespace\terminal\venv\Scripts\python.exe` 环境已验证 Python 3.12.4、SciPy 1.18.1、NumPy 2.4.4，`from scipy.optimize import milp` 可导入。只在本地现有虚拟环境安装所需 SciPy；不依赖服务端或新学习框架。

调度配置每5s决策，基础dt仍1s，搜索/候选/滚动/oracle预算见[阶段2协议](experiments/fixed_terminal_stage2_protocol.md)。完整比较与闭合目录比较分开输出；预算截断及oracle状态是数据字段，不应由失败异常或`null`悄悄抹去。阶段2主要接口：

| 模块/对象 | 接口 | 语义 |
|---|---|---|
| `safety.Feasibility` | `check_batch(plans, fleet, now)` | 独立验证普通计划批次、未选保持和现有承诺 |
| `FixedTerminalEnv` | `submit_batch(plans)` | 整批可行时原子提交完整路线 |
| `baselines.py` | `observed_clone(env)` | 删除未到达飞机及其场景请求信息 |
| `baselines.py` | `analyse_windows(env, aircraft, limit=256, stats=None)` | 能源/最晚窗口值与 exact/界限/未知状态 |
| `baselines.py` | `choose_normal_batch(method, observed_env, settings=DEFAULT_SCHEDULER_CONFIG, catalogue=None)` | 当前可观测状态上的普通批次选择 |
| `baselines.py` | `apply_scheduler(env, method, settings=DEFAULT_SCHEDULER_CONFIG)` | 提交普通决策并使用共同应急逻辑 |
| `oracle.py` | `finite_catalogue_milp(env, catalogue, seconds=30., iteration_limit=1000)` | 有限目录三层目标 MILP 及独立验证 |
| `evaluate_fixed_terminal_stage2.py` | `run(...)` / `run_reference(...)` / `aggregate(...)` | 完整运行、闭合目录参考与摘要 |

`terminal_marl/configs/fixed_terminal_stage2.py` 中的冻结 `SchedulerConfig` 保存计算预算，`METHODS` 保存六方法。CLI 入口为 `python -m terminal_marl.scripts.evaluate_fixed_terminal_stage2`：单场使用 `--method/--load/--soc/--seed/--output`；`--matrix` 为162场；`--oracle --size N --seed S` 为单个闭合实例；`--oracle-matrix` 为12实例；`--aggregate <root> --output <prefix>` 产生汇总JSON/CSV。每个结果目录要求尚不存在。上述接口已用于完成并核对本轮162场与12个oracle；精确单元命令和结果见阶段2证据。

完整场景文件为 `frozen_input.json/result.json/requests.jsonl/events.jsonl/decisions.jsonl`。结果schema为 `fixed-terminal-stage2-v1`，增加 `scheduler/scheduler_config`；决策记录保存 `generated/examined/joint_assignments/truncated/rejection_reasons/priority_windows/lookahead_states` 与耗时。摘要含 `online_call_seconds_total/p50/max`、`normal_delay_max_seconds`、候选数、联合检查数、截断次数及约束拒绝原因。

优先级分析字段包括 `minimum_feasible_required_soc/minimum_status`、`energy_slack`、`latest_feasible_completion/deadline_status`、`deadline_slack`、`completion_energy_only_upper_bound/domain_size/examined`。已发现可行最低能源若未认证为最小则为上界，对应能量余量是下界；已发现最后完成窗口若未认证为最晚则为下界。energy-only完成上界不代表资源可行。

oracle输出 `status/objective/seconds/candidates/iterations/cuts/solver_status/solver_gap/certified_tiers/unknown_assignments/reason`，另含 `solver_dual_bound/solver_bound_tier/solver_bound_scope`。tier为负正常数、延迟秒或路线SOC能耗；bound只对应当前有限MILP松弛目标层，不是动态全局界，缺失或非有限值为 `null`。闭合实例结果带 `full_episode_optimality_claim=false`，不得与完整rollout的收益混用。最终合并针对性测试已由实现任务保存：47 passed in 1.58s，exit 0，原始日志在忽略目录 `tmp_stage2_validation/final_tests.log`。测试文件为阶段1三个文件以及 `tests/test_fixed_terminal_stage2_joint.py`、`tests/test_fixed_terminal_stage2_baselines.py`、`tests/test_fixed_terminal_stage2_oracle.py`、`tests/test_fixed_terminal_stage2_runner.py`。正式162场与12个oracle已另行完成原始文件和指标核验。

阶段2实现与冻结实验已完成，47项针对性测试及2项元数据修正测试通过；162场完整比较和12个有限目录参考的事实与边界见[结果报告](evidence/fixed_terminal_stage2/results.md)。本轮尚未建立学习必要性证据，不自动启动后续训练。

## 当前阶段3机制接口

阶段3实现与正式15context机制实验已完成；上文阶段1/2测试和CLI结果均为历史已完成事实。实现任务已提供以下稳定接口，最终阶段1/2/3合并57项测试通过（17.58 s、exit0）；最终修正smoke完成，但不把预检记录称为正式机制实验结果。

固定设置为cohort8/scan48/keep8/beam128，扩展前12根加HOLD与六控制、去重最多19，公共5s/256不变。15context与三种终点见[阶段3协议](experiments/fixed_terminal_stage3_protocol.md)。原始价值必须记录`root_time`、`evaluation_stop`、`episode_cumulative=true`及全部分量；闭合观测与ex-post各记分母；value实际字段为`request_denominator/arrived_request_count`。公共prefix保留、Δ中抵消，不另定义增量J。

输出应包括候选来源labels（仅审计，不作特征）、feasibility与undefined、根前状态、完整既有承诺、实际J/ΔJ、配对请求/events与remove-one。t0旧日志控制保留原`decision.seconds`，新读取时间另列；t60在共同状态重新求解。计时拆候选生成/控制获取/closed-observed rollout/actual ex-post rollout/remove-one，不能把读取时间冒充求解时间。每个J stop另存`elapsed_through_stop_seconds`，记录同一closed rollout至该stop的累计耗时，不参与value比较；参考成本（生成+控制获取+observed）与ex-post/remove-one分析成本分开。

部署输入只含已到达状态和固定资源预约；未到达draw、全程标签和方法来源不可输入。价值接口见[学习问题定义](fixed_terminal_learning_problem_definition.md)，不训练或设计模型。

### 阶段3公开符号

| 文件 | 公开接口 | 用途 |
|---|---|---|
| `core/fixed_terminal/value_analysis.py` | `load_stage2_episode(directory)`、`replay_episode(directory, stop_time=None)` | 读取已存阶段2记录、恢复决策前环境 |
| 同上 | `visible_snapshot(env)`、`system_value(env)` | 可见状态与累计实际J |
| 同上 | `evaluate_root(env, plans, stops, *, realized_future=False)`、`delta_value(value, baseline)` | 指定终点/世界的根评估与分量差 |
| 同上 | `conditional_margin(env, full_batch, aircraft_id, full_evaluation=None)` | 可行条件remove-one对照 |
| `core/fixed_terminal/reference.py` | `Root`、`acquire_controls`、`make_root_pool(env, context, stage2_root=None)` | 六控制与确定性候选根池 |
| 同上 | `observed_evaluation`、`select_observed_roots`、`value_comparison_key`、`action_key` | 闭合估计、选择及冻结比较/tie |
| `configs/fixed_terminal_stage3.py` | `Context(load, soc, seed, time=0)`、`FORMAL_CONTEXTS` | 15个固定context |
| 同上 | `REFERENCE_SETTINGS`、`ROOT_SEARCH_KEEP`、`REMOVE_ONE_LIMIT` | 8/48/8/128、12根、最多3个remove-one |
| `scripts/analyze_fixed_terminal_stage3.py` | `run_context`、`aggregate` | 单context机制产物与汇总 |

以上路径以`terminal_marl/`为根。CLI示例：

```text
python -m terminal_marl.scripts.analyze_fixed_terminal_stage3 --load low --soc medium --seed 101 --time 0 --stage2-root tmp_stage2_results/episodes --output NEW_DIRECTORY
python -m terminal_marl.scripts.analyze_fixed_terminal_stage3 --list-contexts
python -m terminal_marl.scripts.analyze_fixed_terminal_stage3 --aggregate ROOT --output SUMMARY.json
```

schema为`fixed-terminal-stage3-v1`。每context保存`input.json/snapshot.json/root_pool.json/observed_selections.json/conditional_margins.json/result.json/decision_database.jsonl`；各唯一root在`<context-output>/<root_id>/root.json`、`observed.json`、`realized.json`保存根定义和两类重放。异常写`errors.json`并向上报错，不静默丢根。数据库分开保存两世界raw values，以及每stop的`delta_observed_vs_HOLD/FCFS`与`delta_realized_vs_HOLD/FCFS`，不是teacher训练集。

selection key前四项整数精确比较，T/D round8位、energy round10位；raw J/ΔJ仍保存。相同key按action_key最小者，no-worse使用同key。首次17root/11.480 s仅为修正前序列化smoke，保留旧目录；后续输出接口修订smoke也单独保留。最终含stop累计计时的smoke位于`tmp_stage3_preflight/low-medium-101-t0-timing-final`：17根、10.5729187 s、exit0，57 JSON+1 JSONL、3,155,951 bytes。closed请求分母7、ex-post分母12，不跨世界直接比较。正式15context已全部exit0，共209根，原始结果及汇总在`tmp_stage3_results`；[阶段3结果](evidence/fixed_terminal_stage3/results.md)给出事实和边界。

最终测试命令在既有阶段1/2七个文件外加入`tests/test_fixed_terminal_stage3_values.py`、`tests/test_fixed_terminal_stage3_replay.py`、`tests/test_fixed_terminal_stage3_reference.py`，完整原始命令与57 PASS见`tmp_stage3_preflight/final_combined_tests.log`；smoke命令/exit见`final_smoke.log`，产物检查见`final_validation.json`。最终smoke累计observed stop时间合计20/60/240为0.343890/0.684381/1.899963 s，根生成5.802359 s、控制获取0.031774 s，ex-post2.518635 s、remove-one0.154133 s；原控制求解成本单列。这些仅是预检记录，不作为正式机制收益结论。
