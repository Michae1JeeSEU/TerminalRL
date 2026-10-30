# Service-opportunity follow-up：实现与固定比较协议

本轮显式启用 `service_opportunity_v1`，保留默认 `legacy_window_tail_v8` 和已有 `stitched_continuation_v1`。目标是在相同物理起点和截止条件下，比较 FULL_H / RULE_LOCAL_K8 的可达服务机会利用；不加入 transfer bonus、inner bonus、强制 MOVE 或任意容差。旧结果、checkpoint、账本及报告保持只读。本文件不是效果结论，正式结果须由主对话复核后填写。

实现入口为 [completion/queue/scoring](../../terminal_marl/core/rotating_slots/stage2_service.py)、[profile routing](../../terminal_marl/core/rotating_slots/stage2_downstream.py)、[preselection frame](../../terminal_marl/core/rotating_slots/selective_replanning/frame.py)、[candidate/CP/certificate](../../terminal_marl/core/rotating_slots/stage2_planning.py) 和 [local planning](../../terminal_marl/core/rotating_slots/selective_replanning/local_problem.py)。

## 统一完成估计与六项字典序

MOVE 与 KEEP 均计算：已有/本次显式动作与必要间隙的真实积分能量、同一调整时点约定下的等待、缺失后续动作的同一 nominal 模型，以及共享 pad 队列的等待。显式前缀先去重；不把 horizon holding 再叠加到同一段时间上。缺失后续动作一律用 nominal 模型，已有 reference cache 不改变其点估计；完整显式落地链仍使用实际计划。真实安全、SOC/reserve、peak、资源及提交检查保持原规则，评分估计不构成执行授权。

| 字典序项 | 含义 | 对外 key 单位 / CP 整数单位 |
|---|---|---|
| `F` | 显式 peak 超限或按完成能量预测耗尽的飞机数 | count / count |
| `-L` | horizon 内完整显式链实际预测正常 touchdown 数的负值；partial 不假算落地 | count / count |
| `R` | `sum(max(0, estimated_completion_energy - energy_above_25pct_reserve))` | Wh / 0.1 J |
| `G_service` | 下述同一 baseline 的累计服务机会延误 | s / 0.1 s |
| `E` | 完成能量预测和队列等待能量之和 | J / 0.1 J |
| `tau` | 累计预测 touchdown 相对当前时刻的时间 | s / 0.1 s |

未知后续可行性、相位和占用释放的不确定性保留在 diagnostics；不再作为额外 Wh 混进 `R`。物理能量/时间下界只保留必要性检验的用途，不能充当未知后续的期望能耗。

nominal 模型每个 config 首次调用时，在 callback 的实际时钟内，用通过运动学检查的三个 canonical primitives 和现有能量 quadrature 校准，后续飞机复用标量。冻结 baseline config 的观测校准如下，数值仅说明估计模型，不是每架飞机或每个 pad 的可行性证明。

| primitive | 代表构型 | duration | energy，约 |
|---|---|---:|---:|
| outer → middle | slot 0 → slot 0 | 40 s | 2.503 MJ |
| middle → inner | slot 0 → slot 0 | 40 s | 2.992 MJ |
| normal final | inner slot 0 → pad 5 | 48.5 s，含 13.5 s descent | 5.016 MJ |

因此 outer nominal completion 为 128.5 s / 10.511 MJ，middle 为 88.5 s / 8.008 MJ，inner 为 48.5 s / 5.016 MJ。它们是 representative point estimates；不生成全体 background aircraft 的完整轨迹，也不声称下界、相位适配、资源可达或全路径安全已认证。

## G 的共同 baseline 与队列边界

每次 dispatch 在选择 replan 集合之前，冻结全部已准入、已揭示飞机的 baseline demand、真实 pad calendars 和全局顺序。未准入请求没有 flight state，不进入目标。共同评价窗口为 `H = now + 240 s`，使用既有 0.1 s 网格。

baseline 从同一物理 source/node 和不可改写的 explicit prefix 终点出发，只去掉 FOLLOW 的 discretionary adjustment wait（通常为额外 20 s），不去掉已承诺动作、真实间隙或物理飞行时间。baseline 和候选均以同一队列算法联合分配；baseline 完成后在本次 frame 内固定。

令 `b_i` 为共同 baseline 的 pad 占用起点（descent 起点），`s_i` 为所选动作预测的对应起点。对 baseline 在窗口内具有 forecast opportunity 的飞机，

```text
G_service = sum_i max(0, min(s_i, H) - b_i),  with b_i < H.
```

若该 baseline 有机会而所选 continuation 无预测可达窗口，按 `s_i = H` 截断，不能靠自行变成 blocked 来抹掉 G。无 demand、baseline 无窗口内机会或共同的已知不可达情形贡献为零。无限真实 pad 占用的释放未知时，标记无 forecast window 和 occupancy uncertainty，不把未知释放称为已证明不可达。普通 HOLD slot 可在后续决定释放，不能凭整环 slot 当前占满就证明不可达。相同窗口落位具有相同 G，holding ring 本身没有奖励；能量在其后的项打破平局。

队列按冻结的 preselection 全局顺序逐架处理。每个 nonfull demand 在 nominal reachable-pad 集合中选择最早能容纳 descent+service 的完整间隙，同起点按 pad index；真实 calendars 和所选完整链的实际 pad intervals 均纳入。每次虚拟服务只分配一次，每个 pad 的后继任务位于上一虚拟服务结束之后。Python evaluator 和 CP 用相同 earliest-gap/min-choice 语义。该近似允许换用可用 pad，避免固定虚拟 pad 制造排队；它没有穷举服务顺序，也不声称全局最优 pad allocation。nominal reachable 仍明确标记为未认证，真正执行另经原 validator。

## 搜索与交付

初始有限 slice 优先提供 inner final 和近期 middle/outer 供给候选；保留所有决策变量。已预留的有限 suffix-completion slice 在第一次 CP 前运行，均计入原 deadline，不增加预算。完整链才影响 `L`。CP 首阶段以 `(N+1)*F - L` 合并求解，其中 `N` 是本次计分飞机数，保证一个 failure 的优先级超过可能的 landing 数差；再按 `R, G_service, E, tau` 接续。该改动避免将有限时间全部花在单独 `F` 阶段，但不承诺有限预算内找到全局最优解。

append-only candidate 增长只在 snapshot、scope、frame、评分上下文和已有 column 内容一致时复用原 certificate/cuts。deadline 到达后保留已有非空认证前缀；UNKNOWN 不授权执行。投影、交付及 source/effective snapshot binding 与实际提交证据分别保留，不以“求解完成”代替物理启动。

## 固定输入、四臂与预算

复用旧 [A4 common initial](selective_quality_a4_20261030.md) 的字节级相同物理 checkpoint，不重建或挑选初态。两根真实 `t0=20 s`，正式增量窗口均为绝对 `20→320 s`；历史 initializer 提交的 FINAL touchdown 单列 inherited。common 来源提交为 `b6590c8a5ede25b5f63e89834dd4cd5ec36a646d`，新源码另行 freeze，不能把两者混成同一版本。

| 根 | seed | common checkpoint SHA256 |
|---|---:|---|
| s3r23 / medium | 7324 | `0f67e40052c6d80ba35efebaf1e47fb113a1aca7046332d20b0c44f8cc29efc7` |
| s3r24 / high | 7325 | `255f3ce5a3c45b01d3fb59ce5069ce39dec73b420735759ecdd6ce510f5eda77` |

lead 20 s、preparation margin 1 s、ready offset 19 s、search handoff reserve 4 s、search offset 15 s、heartbeat 5 s 固定；不扫描参数。算法缓存从独立冷状态开始，物理 SOC、peak、账本、active/soft、真实 slots/pads 和已有承诺完整保留。

新服务器输出根为 `/home/user/training_outputs/terminal/rs_service_opportunity_20261030`，经 Syncthing 回到 `D:\7_codespace\training_outputs\terminal\rs_service_opportunity_20261030`；旧 `rs_selective_quality_a4_20261030` 只读。入口为 [bounded runner](../../scripts/stage3/run_service_opportunity_comparison.py) 和 [read-only report](../../scripts/stage3/report_service_opportunity_comparison.py)。

| 臂 | physical CPU | 数值库线程 |
|---|---:|---:|
| s3r23 / FULL_H | 16 | 1 |
| s3r23 / RULE_LOCAL_K8 | 17 | 1 |
| s3r24 / FULL_H | 18 | 1 |
| s3r24 / RULE_LOCAL_K8 | 19 | 1 |

四臂使用相同 profile 与物理协议。每次 launcher 仅执行一个有界 action；并行由主对话安排，避免嵌套数值线程。正式臂每臂 child-wall cap 1200 s，audit 每次 300 s，smoke 为 20 simulation seconds（20→40 s）且 child-wall cap 120 s；prepare cap 120 s。prepare、smoke、正式臂和 audits 共用本轮 7200 child-seconds 账本，旧账本不重置、不抵扣。preflight/tests 的实际耗时另列，不能冒充正式 rollout 成本或效果证据。

## 小验证与当前状态

下列是有界机制验证，不替代四臂闭环比较。[scoring tests](../../tests/test_service_opportunity_scoring.py)、[search tests](../../tests/test_service_opportunity_search.py) 和 [measurement tests](../../tests/test_service_opportunity_campaign.py) 保留各自的输入与断言。

| 验证类 | 精确测试入口 | 当前证据 |
|---|---|---|
| 1. inner 飞机/free pad/预构造安全 FINAL：及时交付 | `test_known_safe_final_has_joint_certificate_and_is_physically_committed`、`test_real_service_final_survives_20s_lead_delivery_and_starts_at_effective` | integrated 及最终 focused server 用例已通过 |
| 2. inner 即将空、middle 供给可行：及时服务 | `test_minimal_middle_supply_is_completed_before_first_cp_and_committed`、`test_service_priority_seed_precedes_reference_and_keeps_every_variable` | 首次 CP 前补链及提交机制已通过；不据此声称完整动态场景已验证 |
| 3. pad 暂忙、inner stock 足够：允许 outer 继续等待 | `test_real_busy_pads_sufficient_inner_stock_keeps_outer_at_lower_energy`、`test_alternative_free_pad_and_each_interval_allocated_once` | 真实 world fixture + 同一 frame scoring/原 validators 本地及最终 focused server 通过；正式 rollout pending |
| 4. MOVE/KEEP 同 landing window、不同 wait ring：低总能量胜 | `test_real_busy_pads_sufficient_inner_stock_keeps_outer_at_lower_energy`、`test_identical_move_keep_physical_prefix_energy_and_timeline` | 实际 O→M quadrature 与共享 missing-suffix 模型比较通过；nominal 后续未称为真实已落地 |
| 5. low SOC 或真实阻挡：安全规则不变、解释等待/改选 | `test_genuine_opportunity_shortfall_and_risk_are_separate`、`test_preconstructed_low_soc_final_never_bypasses_physical_safety`、`test_no_demand_out_of_window_and_known_blocking_masks` | reserve/低 SOC validator/mask 用例已通过；等待和其他飞机选择须按实际产物解释 |
| 6. 压缩计算/慢 validation：保留较优认证前缀，未认证不执行 | `test_slow_later_growth_keeps_original_nonempty_certificate_for_handoff`、`test_expired_generation_preserves_exact_original_certificate_without_validation`、`test_unknown_joint_validation_never_authorizes_execution` | integrated 及最终 focused server 用例已通过 |

另有 `test_cp_and_python_exact_six_components`、`test_cp_parity_for_masks_background_and_ground_times` 覆盖六项 queue CP/Python 一致；`test_service_cp_combines_failure_and_landing_before_one_validation` 覆盖合并 F/L 阶段。UNKNOWN、无 demand 与 occupancy uncertainty 的 unit 边界独立保留。

较早服务器 scoring + stitched 回归为 **40 passed，0 skipped，3.73 s**，8 文件 integrated suite 为 **122 passed，0 skipped，26.43 s**。最终 hint 修改后的 focused server 集合为 **62 passed，0 skipped，11.05 s**：`test_service_opportunity_scoring.py`、`test_service_opportunity_search.py`、`test_service_opportunity_campaign.py`、`test_rotating_slots_search_handoff.py`，以及精确 legacy node `test_stage2_candidate_coverage.py::test_fixed_background_rejections_do_not_consume_accepted_first_target_cap`。其中包含 class 3/4 真实 busy fixture 和 `test_service_hint_tries_inner_final_before_many_keep_only_variables` 的九飞机 KEEP-only starvation 回归。本地早期无 OR-Tools 的 skips 已由服务器对应 CP 用例补齐，不能将早期 122 个用例的运行时点与最终 focused 集合混同。

最终 focused 运行 UTC 为 **2026-10-30T09:13:48Z → 09:14:00Z**，PID **399413**，exit code **0**。证据位于新输出根的 `focused_final.log`、`focused_final.xml`、`focused_final.exitcode.json`；本地镜像的日志、JUnit 与退出记录已读取确认。源码 freeze、smoke、正式四臂及 audits 仍 pending，此处不预填通过。

class 3/4 真实 fixture 单测本地为 **1 passed，2.69 s**，并已纳入上述最终 server focused 通过集合：以原 `seed_hold` 建立 SOC 0.9 的 outer slot 0 和六架 inner（slots 0/3/6/9/12/15），用原 `seed_service` 建立六个实际 ground pad service，均在 100 s 释放。六架 inner 为冻结 background，标准 service 为 120 s；按共同 frame 顺序，第一组 inner 预测从 100 s 占 pad，outer KEEP/MOVE 都在 233.5 s 开始 descent、247 s 预测 touchdown，F/L/R/G 相同、outer partial `L=0`、`R=G=0`。实际安全 O→M primitive 为 40 s / 2,502,687.966 J；KEEP 估计总能量为 18,577,756.7 J，MOVE 为 19,375,854.0 J，分别在 outer/middle 等待合计 118.5 s，故原六项 key 选择 KEEP。两种选择在仅 outer 为 decision、其余飞机为 fixed background 的原 single/batch 检查均为 SAFE，prefix 检查均为 ACCEPTED，world snapshot 未改变。该单测没有修改物理参数或评分系数，没有把 nominal 完成估计称为已执行的后续轨迹。

## 指标语义与待填结果

`realized physical idle` 从实际启动的 FINAL descent+service 占用、初始已服务飞机和真实已承诺占用构造每个 pad 的区间并集，再取补集；重叠不重复计时。保存 idle pad-seconds、全部 pad 同时空闲的最长区间，以及无已准入 airborne demand 的 idle。它只是物理空闲，不能直接称为可避免浪费。

`witnessed avoidable idle` 只使用导出的 joint-prefix certificate：绑定 source/projection/effective snapshot、正 delivery margin、明确 FINAL 占用区间，并按首次相关 state change 截断有效期；再与真实 idle 和已准入 demand 区间取交集、按 pad 去重。它是 **有证据支持的保守下界**，不是全部可行路径或全局可避免量。没有 witness、绑定缺失、事件日志或物理覆盖不足时记 `null/unknown`；其余 idle 保留为 unclassified，不能一律判为合理或浪费。无 demand 的 idle 单列。

M→I 指标区分实际 maneuver start 和实际完成后形成的有效供给，按 trajectory source/target 与事件识别，不靠 aircraft ID 名字。`longest_no_inner_supply_s` 是 admitted airborne demand 存在期间、M→I **完成供给事件**之间的最长间隔，窗口端点注明 censoring。它与 `longest_inner_stock_empty_with_upstream_demand_s` 分开：后者要求实际无 HOLD_INNER 库存且存在 admitted inbound/outer/middle airborne demand；没有 M→I completion 不等于 inner stock 已空。

| 根 / 方法 | 正常落地增量（inherited/treatment 分列） | M→I 实际启动/完成供给 | 物理 idle / witnessed 下界 / unclassified | 分环 holding 与 energy | emergency/failure、timely prefix、audit |
|---|---|---|---|---|---|
| s3r23 / FULL_H | PENDING | PENDING | PENDING | PENDING | PENDING |
| s3r23 / RULE_LOCAL_K8 | PENDING | PENDING | PENDING | PENDING | PENDING |
| s3r24 / FULL_H | PENDING | PENDING | PENDING | PENDING | PENDING |
| s3r24 / RULE_LOCAL_K8 | PENDING | PENDING | PENDING | PENDING | PENDING |

结果表只在原始产物、完成状态、预算、审计和指标覆盖经主对话核对后更新。两根各一次开发配对不支持统计泛化；对旧结果的差异需保留新评分/新搜索版本的共同影响，不作未控制的单因素归因。
