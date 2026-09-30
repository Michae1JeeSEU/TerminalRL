# 研究与 P3 不足逐项复核及修复登记（2026-09-30）

## 0. 范围与证据纪律

本页对应“评估研究与训练”对话中提出的研究定位、方法设定、P3 训练退化和后续修复计划。
本轮代码修复及文档同步；已运行测试结果填入第 8 节，qualification 与 paired 证据仍待完成。没有启动新训练、没有消费 heldout、没有修改旧 evidence，
也没有把待验证设计写成 PASS。`CURRENT_STATUS.md` 仍是滚动状态入口。

证据标签使用如下：

- `preexisting_artifact`：对话前已经存在的代码、协议或结果；
- `newly_run`：本轮若有新执行结果，由 primary 另行补入精确路径和 exit/status；
- `pending`：已经冻结为待验证实验或尚未完成的实现检查。

## 1. 对话中研究不足的逐项判定

| 编号 | 对话指出的不足 | 判定与修复状态 | 直接依据与当前边界 |
|---|---|---|---|
| 1 | 研究创新需要独立的方法性贡献，而不是模块组合 | 合理评估；当前未证原创性 | 四模块保留为组成机制；`research_positioning.md` 登记两个待验证方法性候选，见第 2 节。 |
| 2 | 需要区分存储能力 $K(g)$ 与服务率 $\mu(g,\mathrm{rule},\mathrm{LZ})$；dense24 的 `<47/31` 不能证明 outer/middle 已饱和 | 属实限定 | 两个量分开记录；该结果不能验证 storage saturation，也不能排除路径约束影响。 |
| 3 | 低负荷缩容需要 sustained-low gate | 代码事实；实现回归 PASS，未构成 qualification | `low600 + dwell` wrapper 和持续 safety checks 已由 high-level implementation regression 覆盖。 |
| 4 | 宏步队列和延误应按经过时间累计，避免已完成历史 mean delay 重复计入 | 代码事实；实现回归 PASS，未构成 qualification | queue UAV-seconds、unresolved request-seconds 和历史 mean 去重已由 high-level implementation regression 覆盖。 |
| 5 | 移除 energy object 会丢失到达后的能源成本 | 代码事实；实现回归 PASS，未构成 qualification | energy arrival ledger 已由 high-level implementation regression 覆盖。 |
| 6 | SharedPPO 接 V4.2 时不应把非等价 adapter 当作已兼容 | 代码事实；实现回归 PASS，未构成 qualification | V4.2 adapter 在不兼容时提前失败；高层实现回归已通过，但这不等于完整等价支持或 A6 qualification。 |
| 7 | low-level reward 不覆盖全过程，`soft_normal_pref` 只能表达软权衡 | 属实边界；已澄清但不认定根因 | raw reward 不是词典序、也不是全系统成本；不能单独归因训练退化。 |
| 8 | collision/conflict 为 0 不代表能源、资源和长期可行性成立 | 属实；方法检查仍待完成 | energy reachability、trajectory executability、resource serviceability 与安全间隔分开记录；当前不声明 joint recursive feasibility。 |
| 9 | service duration 与 ETA 观测存在混叠 | 属实；实现测试 PASS，尚未 qualification | V2.4 已加入 `self_service_duration_norm`、`center_earliest_compatible_lz_release_wait_norm`，形成新 90D schema；正式 P3 仍为 88D。 |
| 10 | speed/turn 能耗代理需要敏感性验证 | 合理；实验待完成 | 当前 phase/speed/turn 是归一化代理，未完成实测标定。 |
| 11 | gate-only 诊断覆盖不了实际 action proposal 的限制 | 属结构事实；因果尚未证实 | proposal、gate、feasibility 与长期收益需同一 pre-state 的 paired expected outcome；不据此单独归因。 |
| 12 | team credit 存在动作信用混杂 | 假设性诊断项；不是已确认实现 bug | team advantage 的因果影响待 paired 证据，暂不改算法。 |
| 13 | retry 仍由 actor 选择，需提供 pre-action forced-wait 选项 | 属实；实现测试 PASS，尚未 qualification | `student_decision_mask_mode=pre_action_forced_wait` 为 opt-in，默认 `legacy`；动作前阻塞不事后删除。 |
| 14 | warm-start 单机状态不等于多机全可行，需要 energy-slack 课程 | 属实；课程计划已登记 | 课程拆为低并发随机 SOC、固定 SOC 密集异质 service、合并条件；按 slack 和完整 schedule 分级。 |
| 15 | entropy 与 global clip 需要区分结构条件、实现事实和因果结论 | 条件/实现事实已澄清；因果待测 | 对 factorized entropy，$H=H(\mathrm{gate})+p_{\mathrm{release}}H(\mathrm{proposal})$，驻点为 $p^*_{\mathrm{release}}=\sigma(H(\mathrm{proposal}))$；只有均匀 25-proposal 时 $H(\mathrm{proposal})=\log 25$，才有 $p^*=25/26$。当前 frozen proposal 通常非均匀，因此 25/26 不是当前实际最优值。global joint clipping 的 actor/critic 耦合是实现事实，对退化的贡献尚未证实。 |
| 16 | 需要四级 paired 诊断，并在真实参数概率下复核更新 | 合理方法；修正后的 helper 与真实 artifact smoke 已测试，完整 causal paired outcome 仍待完成 | API 按 all-25 sampled-first K=3 真实概率计算含首步 reward 的 full undiscounted return，suffix diagnostic 单独记录；UID/tick/channel keyed RNG，deep-copied model 的 autograd 使用实际参数差异。修正后的 smoke 受注册 horizon 截断并因 `censored=true` withholding `qualified_delta_Q`，不产生因果或性能结论。 |
| 17 | 需要强基线、独立消融、长持续流和 episode 级统计 | 合理要求；实验待完成 | 待补 fixed geometry、strong backpressure/min-slack、rolling MILP/MPC、可行解/下界；episode 为 paired 单位，训练与评估 seeds 分离。 |

## 1.1 对话中的排除性澄清（不是原对话主张）

- 原对话没有把单独调整 discount、batch 或 critic 作为已证修复；当前同意其限定：不足以解释或修复 pre-existing degradation，不能形成单因果结论。
- 原对话没有把 SOC 不可见或 progress 绕圈写成正式根因；正式记录中的 progress gate 为 `0`，本页不把这两项写成证据结论。
- team advantage 保留为待测动作信用假设；entropy 的 $p^*=25/26$ 只在均匀 25-proposal 条件下成立，global joint clipping 的 actor/critic 耦合是实现事实；两者都不是已证退化因果。
- 25 个 proposal 的 best-of-25 不能替代实际 weighted K=3 stochastic execution；比较采用同一 pre-state 的 paired expected outcome。
- rule/MPC、fixed geometry 和既有 counterfactual 不能自动放行 A6 或证明训练退化已解决。

## 2. 方法性候选与创新性边界

当前四个系统模块（动态三环、低层共享意图、资源/轨迹/安全链、LZ service coupling）保留为系统机制，
不称为已验证原创方法。待验证的方法性候选为：

1. **有重构成本的终端资源排队网络**：对几何构型 $g$ 分别定义存储能力 $K(g)$ 与在给定
   rule、LZ 服务资源和安全约束下的可持续服务率
   $\mu(g,\mathrm{rule},\mathrm{LZ})$。扩大 $K(g)$ 不保证 $\mu$ 增加；比较时必须记录
   reconfiguration、transition 和 temporary capacity-loss cost。
2. **能量 deadline 与可执行预约耦合调度**：energy reachability、merge/LZ reservation、轨迹可执行性、
   headway 和 service occupancy 同时进入候选可行性与配对比较。

动态构型的待研究决策条件是 predictive cost saving 大于 reconfiguration cost 与 transient cost；这是
实验条件，不是定理。安全间隔、轨迹可执行性、energy reachability、resource serviceability 分开记录，
当前不声明 joint recursive feasibility。phase/speed/turn 能耗仍是归一化代理，未完成实测标定；speed/turn
能耗需做敏感性验证。

## 3. Qualification、指标和公平比较

正式 qualification 的顺序冻结为：

1. 先核对 `failed`、`unresolved`、collision/NMAC、protected-floor 和 uncertified-execution 等既有 hard limits；
2. 再做 `normal/emergency` 业务 screen；
3. 最后比较等待、吞吐、service rate、资源利用和尾部。

低层 raw reward 是软权衡，不是词典序，也不是全系统成本。尚未证实的 reward mismatch 不得单独写成训练退化归因。

待完成的公平对照包括：调优 fixed geometry、strong backpressure/min-slack、rolling MILP/MPC，以及小规模
可行解/下界。所有对照使用相同 safety、resource pipeline 和 control authority。`two_ring` 按 storage、
airspace footprint、service resource 匹配；geometry-only、flow-only、joint 三类变化独立注册。episode 是
配对统计单位，训练 seeds 与 evaluation seeds 独立。service rate 用长持续流估计；短峰也可作为机制和响应研究，
但要观察动态重构收益及其成本，比较需覆盖持续时间跨越 geometry dwell time 的长峰场景。

已有 dense24 `<47/31` 结果不能验证 outer/middle storage 饱和，但不能据此排除路径约束影响。rule/MPC 可以
作为 mechanism study，不自动授权 A6。

## 4. 能量、安全、资源与实现设计登记

已有主规范已经明确：

- `energy_capacity=1500` 是 normalized/surrogate profile；`.10` 是当前研究 touchdown floor，`.25` 是独立
  queue/admission reserve；speed/turn/phase 能耗不是实测功耗或适航保证；
- `80 m` desired、`40 m` protected floor、`<10 m` collision proxy；SafetyFilter/SafetyShield、trajectory
  certification 和 fail-closed 保持为 hard constraints；
- LZ reservation、final capacity/headway、energy/dynamics、trajectory safety 和 service occupancy 必须
  联合检查后 commit；当前不保证 joint recursive feasibility。
- `footprint_seconds` 按每 tick 累计归一化环面积和/成本代理 `sum(actual R²) / sum(initial R²)`；它不是实测
  物理占地，也不是三维体积。新输出 metadata 使用 `footprint_definition` 记录该定义及版本。

本轮 code 修复状态登记；实现回归结果见第 8 节，不写 qualification PASS；其他 worker 的实现检查另行记录：

- sustained-low wrapper：`low600 + dwell`，安全检查继续执行；
- 每 tick 累计 queue UAV-seconds、unresolved request-seconds；energy 永久写入 arrival ledger；
- `high_reward_v2_time_integrated` 是新 reward/schema，旧 high-level 结果不等价；
- SharedPPO V4.2 的 early-fail-fast 已实施并通过回归；这只证明不兼容时提前失败，不等于 adapter 完整等价支持；
- `student_decision_mask_mode=pre_action_forced_wait` 为 opt-in，默认 `legacy`；retry/SOC/interface 的动作前
  阻塞不能事后从样本删除；
- V2.4 已实施为旧 88D 加 `self_service_duration_norm`、`center_earliest_compatible_lz_release_wait_norm` 两字段的 90D，使用新 schema/feature hash，
  旧 checkpoint 拒绝加载；实现回归已通过但尚未 qualification，正式 P3 在新 qualification 前继续 88D。

## 5. P3 counterfactual 与 credit 证据边界

`preexisting_artifact` 证据：

- `diagnose_p3_action_credit.py` 对选定状态执行 WAIT 与 25 个 RELEASE 分支，检查 runtime branch、state/model/RNG
  不变；其 utility 是 one-step critic proxy，不是完整周期 return。
- `p3_action_credit_runtime_labels_20260926.md` 记录 `16/16` 状态、每状态 `51` branches、共 `816` 条 runtime
  branch 记录，明确标记为 partial diagnostic，不是完整训练集。
- `p3_terminal_credit_protocol_20260928.md` 冻结了按 `(step, uid, arrival)` 选择状态、每状态 2 个 continuation
  RNG repeats 和 WAIT/source-release/runtime-certified-release 三分支；后续 continuation 运行到完整 scene terminal。
- `p3_terminal_credit_result_20260928.md` 记录 `12 states × 2 repeats`、`24 reports`、`72` logical branches，
  其中 `52` 次真实执行、`20` 次复用；5 个 stable labels、7 个 uncertain；重复汇总不作独立性能证据。
- `diagnose_p3_task_neutral_episode_gradients.py` 对 6 条完整 episode tape 计算 complete-episode MC、team GAE、
  shared actor normalization 和 aggregate-sum gradient check，但不采样、不更新、不调用 optimizer。

paired helper 已实现为 `terminal_marl/training/paired_gate_diagnostic.py`，定向测试为
`tests/test_paired_gate_diagnostic.py`；服务器 `8 passed`、`1.32 s`，hashes matched，真实 Terminal 两 tick
smoke 通过。API 按 all-25 sampled-first K=3 计算含首步 reward 的 full undiscounted return，suffix diagnostic
单独保留；`qualified_delta_Q` 在 censored/error、nonidentical request prefix 或 parent-model/RNG invariant
失败时 withheld。随机数按 UID/tick/channel keyed，模型用 deep copy 和 autograd 取得实际参数差异。

真实 artifact smoke 已完成（`COMPLETED exit0`）：JSON 为 `/home/user/training_outputs/terminal/research_review_repairs_20260930/real_artifact_smoke.json`
（本地同步路径 `D:/7_codespace/training_outputs/terminal/research_review_repairs_20260930/real_artifact_smoke.json`），
SHA256 为 `c471f96589e9af071e06c5ef3df98bde1ad544abf6d20780eb0bb15d83deb00e`。d449 policy branches 对齐
`before=3fe7+a82` 与 `after=d449`；reset `877101`、continuation `878201`、UID `0`，`26` branches、每 branch `2` ticks、
无 errors，invariance 全为 `true`，external request prefix length 为 `2`，reliable fingerprint 为 `safety_trace_plus_ledgers`。
修正后的实际 Terminal config 使用 `max_simulation_seconds=2`、`max_steps=3`，每 branch 恰好 `2` ticks，并按
`environment_time_limit` 截断；`censored=true`，因此 `qualified_delta_Q` absent。18 个 parameter tensors changed
（12 critic、6 gate adapter），`12×25` proposal-probability delta exact `0`；gate probability delta 为异质范围
`-0.0003174..+0.0619840`。这些是实现与诊断事实，不产生 causal/performance conclusion。

`pending` 的完整 causal paired outcome 仍要求旧/新 checkpoint 的真实参数概率、同一 request prefix 和 parent-model/RNG
不变量全部通过，并处理任何 censored/error 情况；当前不把该 smoke 写成 qualified delta。

## 6. 课程修复登记

低层课程拆成三个独立条件：

1. 低并发、随机 SOC；
2. 固定 SOC、密集到达、异质 service time；
3. 随机 SOC 与密集异质 service 的合并条件。

每阶段按 energy slack 与是否找到完整可行 schedule 分级。未找到 schedule 只记录为观察结果，不等同于 infeasible；
不得提高 SOC 下限、删除失败或关闭安全/能量/资源约束来制造 qualification。本轮不启动新训练、不消费 heldout。

## 7. 文献与期刊核实边界

- Shao et al. (2021), *Terminal area control rules and eVTOL adaptive scheduling model for multi-vertiport system in urban air Mobility*, DOI [`10.1016/j.trc.2021.103385`](https://doi.org/10.1016/j.trc.2021.103385)。
- Waltz, Okhrin & Schultz (2024), *Self-organized free-flight arrival for urban air mobility*, DOI [`10.1016/j.trc.2024.104806`](https://doi.org/10.1016/j.trc.2024.104806)。

上述两条用于支持方法重叠的边界判断；本页不宣称已完成两篇全文核验。

Chang et al. (2026), *Vertiport arrival airspace design, analysis, and capacity assessment: Integrating the point merge system into UAM*，
目前只做 publisher metadata/abstract-level 核实：DOI [`10.1016/j.ast.2026.113051`](https://doi.org/10.1016/j.ast.2026.113051)，
[publisher abstract](https://www.sciencedirect.com/science/article/abs/pii/S127096382601429X)。摘要层面支持其涉及 geometry/capacity
评价；卷期/线上正式发表时间与全文仍待核实，不使用 ResearchGate 日期。

Han et al. (2026) 目前只有 SciOpen author-page 级线索，未完成全文核实；不据此下原创性结论。TR Part B aims 页面本轮返回
HTTP 403，因此本页不宣称已核实期刊准入标准，也不宣称必须有定理。

## 8. 报告填充边界

### `newly_run`

本轮服务器回归检查均通过 `ssh lab-server` 执行并 exit 0：

- group 1：`tests/test_research_low_level_repairs.py`、`tests/test_student_rollout_v4_2.py`、
  `tests/test_student_rollout_reward_totals.py`，`41 passed`，`1.56 s`；
- group 2：`tests/test_task_neutral_curriculum.py`、`tests/test_source_bound_observation.py`、
  `tests/test_rejected_release_hold_v4_2.py`、`tests/test_random_warm_start_layout.py`、
  `tests/test_low_level_action_v4_2.py`，`48 passed`，`6.61 s`；
- group 3：`tests/test_p3_task_neutral_evaluation.py`、`tests/test_p3_task_neutral_joint_batch.py`、
  `tests/test_p3_task_neutral_fresh.py`，`37 passed`，`0.88 s`。
- 三组共 `126` 个 low-related checks。该计数来自独立命令的测试计数，不作为独立实验样本数。
- high-level 回归：`PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_research_high_level_repairs.py tests/test_a6_effectiveness.py tests/test_a6_dynamic_high.py -q`，`18 passed`，`20.60 s`，服务器 exit 0；日志为 `/home/user/training_outputs/terminal/research_review_repairs_20260930/high_tests.log`。
- 累计测试计数为 `144`（`126 + 18`，重复独立命令计数，不作为独立实验样本）；A6 qualification 和 paired 诊断仍未完成。
- `compilepass` 已通过；旧 `episode_01` 的 `config_equal=True` 与 `legacy_hash_compatible=True` 已观测。source A82 checkpoint header 与 `725 steps`、`seed 877101`、`policy 878201` 对应；该项是只读 header 检查，不是 replay 验证。
- 本地环境缺少 `torch` 与 `pytest`，本地仅完成哈希同步检查；服务器回归结果以上述服务器命令为准。
- paired helper（修正后）：`terminal_marl/training/paired_gate_diagnostic.py` 与 `tests/test_paired_gate_diagnostic.py`，`9 passed`，`1.54 s`，无 `HOME` override，服务器 exit 0；local compile/diffcheck/hash match 通过。此前 `8 passed` 的 helper 结果已被该命令 supersede。真实 artifact smoke 同一 SHA 已 `COMPLETED exit0`，并按注册 horizon 截断。
- low worker 对相同 helper cases 的 rerun 为 `9 passed`、`1.48 s`；该 rerun 不重复计数。
- 后续 high-focused 命令为 `7 passed`、`0.24 s`，其中包含既有测试重跑和 1 项新增 endpoint 检查；日志为
  `/home/user/training_outputs/terminal/research_review_repairs_20260930/test_research_high_level_repairs.log`。
  metadata 已生成，equal-endpoint queue 为 `0`，但 history 为 `2 UAV-seconds`、reward 为 `-0.1` 对 `0`；该计数单独记录，
  不与前述 `126 + 18` 合并为独立测试或实验样本数。

实现回归中，表中第 3、4、5、6、9、13、16 项的实现测试已通过，但均未构成 qualification。第 15 项未修改 entropy 或 global clip；第 16 项的 real-artifact smoke 已通过但因 `censored=true` 未产生 `qualified_delta_Q`，完整 causal paired outcome 仍待完成。high metadata/one-endpoint regression 已通过并记录 endpoint/history 差异，不作性能结论。distinct pytest cases 按 `126 + 18 + 1 + 9 = 154` 记录：high-focused 的 7 项为 6 项重跑加 1 项新增，helper 的 9 项为 8 项重跑加 1 项新增；不把重跑次数重复累加。

### 8.1 已闭合与部分闭合

- 已闭合到实现回归层：第 3、4、5、6、9、13 项代码修复通过服务器回归检查。
- 部分闭合：新的 90D schema、`pre_action_forced_wait` mask、paired helper 和真实 artifact smoke 已实现并通过实现检查，但尚未独立 qualification；V4.2 high adapter 目前只保证 fail-fast，完整等价支持仍待完成。
- 仍待研究证据：方法性贡献、strong baselines/消融、speed/turn 敏感性、joint recursive feasibility，以及完整 causal paired outcomes，不能由文档或实现回归替代。

## 9. 本轮修改文件（selective staging）

本轮交付只包含以下 20 个路径；`CURRENT_STATUS.md` 仅暂存标题后的 2026-09-30 顶部段，工作区其余历史 dirty 内容保持未暂存：

- **文档（7）**：`docs/README.md`、`docs/research_positioning.md`、`docs/requirements.md`、`docs/design_spec.md`、`docs/technical_spec.md`、`docs/execution_plan.md`、`docs/evidence/v4_2/research_review_repairs_20260930.md`。
- **核心（3）**：`terminal_marl/core/config.py`、`terminal_marl/core/observations.py`、`terminal_marl/core/safety_trace.py`。
- **环境（2）**：`terminal_marl/envs/high_level_wrapper.py`、`terminal_marl/envs/parallel_env.py`。
- **训练（3）**：`terminal_marl/training/evaluate_p3_task_neutral.py`、`terminal_marl/training/student_rollout_v4_2.py`、`terminal_marl/training/paired_gate_diagnostic.py`。
- **脚本（1）**：`terminal_marl/scripts/run_research_gate_diagnostic_smoke.py`。
- **测试（3）**：`tests/test_research_low_level_repairs.py`、`tests/test_research_high_level_repairs.py`、`tests/test_paired_gate_diagnostic.py`。
- **状态（1）**：`docs/CURRENT_STATUS.md` 的 selective top-block insert。

### `preexisting_artifact`

本页引用的既有结果均保持原 evidence 页面、manifest、seed 和 qualification 边界；不改写旧 `candidate_retained`、formal、heldout
或 `PPO_ALLOWED` 字段。

### 结论边界

本轮代码修复及文档同步已完成部分实现回归；报告列明的 qualification、paired 和完整诊断仍未完成。它们没有证明训练退化已解决，
也没有授权新训练、PPO、heldout 或 A6。
