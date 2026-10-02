# P3 基础修复小样本无训练对照协议（2026-10-02）

本文件登记 `evaluate_p3_foundation_repair_20261002.py` 的固定输入与执行上限。
它只用于开发集小样本对照，不是 qualification，不使用 held-out，不改变
`PPO_ALLOWED=false`，也不产生训练或 checkpoint 写入。

## 固定输入

- source checkpoint：
  `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt`
- source SHA256：
  `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`
- 基础配置：`evaluate_p3_task_neutral._build_neutral_inputs()[1]["dense24"]`。
  保留该配置的安全、能量、动作权限、reward、mask 和 arrival 设置；对照只改变
  `final_approach_execution_mode`、观察 schema，并将 runtime environment 显式绑定到
  `trajectory_schema_version=executable_trajectory_v3`；source checkpoint 的严格 load
  boundary 单独使用 `trajectory_schema_version=executable_trajectory_v2`。
- 开发 cell 与 keyed policy seed：

  | cell | env seed | policy seed |
  |---|---:|---:|
  | `cell_1` | `877101` | `878201` |
  | `cell_2` | `877102` | `878202` |

- 每个 cell 的 source arms：
  - `legacy88`：`phaseA_low_obs_v2_1_retry_state`（88D）+ `legacy`；
  - `repaired88`：同一 88D source + `speed_timing_v1`；
  - `repaired90_zero_initialized`：通过
    `initialize_service_context_policy(source_policy,
    source_checkpoint_metadata=payload["metadata"])` 迁移到
    `phaseA_low_obs_v2_4_service_context_90d`（90D），追加字段零初始化，且只做
    inference。
- `cell_1` 另跑两个 nominal rule arms：`nominal_rule_legacy88` 与
  `nominal_rule_repaired90`。规则臂使用同一环境 authority `±2`、K=3、恒定
  requested RELEASE `[1, 0, 0]`，固定 shortlist
  `[(1, 0, 0), (1, -1, 0), (1, 1, 0)]`；非 actor 使用 canonical WAIT，不启用
  `rule_flow_coordination`。
- source arm 的 gate/proposal 采样使用既有 `keyed_uniform`、CDF 和 shortlist helper，
  key 为 `policy_seed|absolute_tick|arrival_id|stream`，stream 分为 `gate` 与
  `proposal`。context 由 runner 独立构造，actor mask 与 frozen `_legacy_actor`
  一致，不调用 88D 专用 `_details_context`。

## 执行边界与停止规则

- 每臂最多 24 个 requests、最多 1800 ticks；固定八臂，合计 8 个 arm 记录。
- 循环只有在 arrivals exhausted 且 active、admission queue、service 全部为空时才算
  natural terminal；达到 tick 上限一律标记 `censored`，不能标记完成。
- runner 不调用 optimizer、Teacher 或训练步骤，不写 checkpoint，不读取 held-out。
- 每个结果保存 progress、ledger、terminal rows、wait/elapsed/energy、独立的
  `normal/emergency/failed/diverted/unresolved` 计数、validity/censor/error、硬安全
  记录，以及去重后的 committed `final_approach` plan 元数据。安全记录单独保留
  `SafetyInvariantError`、collision、NMAC、hard separation 和 uncertified 字段。
- terminal rows 先合并 lifecycle rows 与 `energy_outcomes` 的直接 terminal/service
  rows，再做唯一性与 ledger 对齐校验。`total_elapsed_seconds` 是全部 request 从
  request time 到 resolution time 的和（包含 ground service 完成以及 failure/diversion）；
  `total_energy_consumed` 是 energy ledger rows 的 `energy_consumed` 之和。字段缺失或
  非有限值保留为 `null`，并写入 validity technical errors，不填充为零。
- 每个 arm 目录写 `progress.json`（开始、每 25 ticks、终态），根目录也写当前
  cell/arm、ticks、wall time 与状态。每个 arm 另外保存 requested action groups 与
  config snapshot，供 paired comparison 使用。
- 每个 cell 记录 `legacy88→repaired88` 的业务 scalar 差值；记录
  `repaired88→repaired90_zero_initialized` 的 outcomes、wait/elapsed/energy、requested
  action tape 与 final trajectory hash 对照。zero-init 数值一致性容差为 `1e-6`；UUID
  `plan_id` 不比较，trajectory hash 独立记录，不因微小浮点差异强制相等。
- matrix status 只有所有 arm natural completed、valid、无硬安全指标非零或不可用时为
  `COMPLETED`；任一 arm error、validity invalid 或 censored 时为
  `COMPLETED_WITH_ISSUES`，业务 failed/emergency 计数本身不触发技术 issue。发生
  `SafetyInvariantError` 或硬安全 violation 时保留已完成 arm 输出并停止余臂，CLI 返回
  nonzero。
- 历史 V2 输出根目录（仅记录旧版 `--run`）：
  `/home/user/training_outputs/terminal/p3_foundation_repair_20261002`。
- 当前 V3 对照输出根目录：
  `/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002`。

## 命令边界

```text
python -m terminal_marl.scripts.evaluate_p3_foundation_repair_20261002 --check-only
python -m terminal_marl.scripts.evaluate_p3_foundation_repair_20261002 --smoke
python -m terminal_marl.scripts.evaluate_p3_foundation_repair_20261002 --run
```

`--check-only` 只验证固定 checkpoint、SHA、metadata、88D/90D schema、迁移元数据和
八臂登记；`--smoke` 执行 `cell_1` 的三个 source arms 各两 tick，检查 serialized
schema dimension、progress、energy coverage、validity、error/lifecycle_error、硬安全
状态、final-plan trace 字段、V2 source/load→V3 runtime binding，以及 repaired88 与
zero-initialized repaired90 的初始 requested action tape，三臂预期均为 `censored`，不计
入业务性能；`--run` 才执行固定八臂开发对照。

## 首阶段批准与历史 V2 实际执行记录

用户已批准本轮首阶段的小样本、无训练闭环验证。以下记录属于历史 V2 输出根目录，
不作为当前 V3 对照结果：

- `--check-only` exit `0`，JSON 状态为 `CHECK_ONLY_OK`，登记 `8` 个 arms；输出
  `/home/user/training_outputs/terminal/p3_foundation_repair_20261002/preflight.json`，
  SHA256 为
  `751baf2e44e8ccc050ed99804ae07a2f16bedb09b84e9bc143c5a859abb61e85`。
- `--smoke` exit `0`，JSON 状态为 `SMOKE_OK`，执行 `cell_1` 的
  `legacy88`、`repaired88`、`repaired90_zero_initialized` 三个 source arms，各
  `2` ticks 且均为 `censored=true`；三臂 `energy_coverage.valid=true`，最终 trace
  count 依次为 `1/2/2`。repaired88→repaired90 的 requested action tape 最大差为
  `0.0`，wait/elapsed/energy 对照通过，final trajectory hash 独立比较且相等，所有
  frozen flags 均为 `false`。该 smoke 输出仅作字段与序列化检查，
  `performance_interpretation=not_available`；输出
  `/home/user/training_outputs/terminal/p3_foundation_repair_20261002/smoke.json`，
  SHA256 为
  `e0c9d23984c158f29ba8dc36151f118fd88e5c7aba2ca93fc3e09da345c5edac`。
- server targeted test 为 `10 passed in 0.75s`。
- 完整八臂 `--run` 未启动。新的 physical derivative consistency prerequisite
  检查为 `FAIL`：四条 plan 数值复核观测到水平加速度范围
  `22.3043–28.0235 m/s²` 高于 `1.5 m/s²` 上限、垂直加速度范围
  `5.8505–6.5178 m/s²` 高于 `1.0 m/s²` 上限，而旧 `validate_kinematics` 返回空列表。
  该阻断是前置一致性检查记录，不是性能结果；原定 seeds 与每臂 `24 requests`、
  `1800 ticks` 上限均未改变。

## 输出字段约束

业务 outcome 分类与 validity、censor、error 分开记录。缺失或重复 terminal rows、
energy coverage 或 wait 字段只能造成 validity/error 记录，不得把这些技术问题吞成
`failed` outcome。source arm 的 checkpoint metadata、SHA 和 90D migration metadata
随结果保存；nominal rule arm 不声称使用了已学习的 service context。

## 注册对照的 V3 共同物理基准修订（2026-10-02）

用户已批准首阶段的无训练小样本闭环验证；注册对照的共同物理基准现改为
`trajectory_schema_version=executable_trajectory_v3`，并要求
`vehicle_execution_mode=performance_constrained`。八个 arm 均使用这一共同轨迹
schema；`legacy88` 仅表示旧 final intent mapping，观察 schema、动作权限、安全、
能量、reward、mask、arrival 设置以及两 cell 的 seeds、每臂 24 requests 和 1800
ticks 上限保持登记值不变。V3 轨迹在执行边界复核起点状态、时间和 ring endpoint
的几何/切向速度/向心加速度；诊断同时保留 V2 的 chord/trapezoid 行作为
`v2_only_nonbinding` 观测，并以 V3 多项式解析导数、三点 Gauss-Legendre 积分和
physics-grid 指标作为对应 schema 的检查字段。

本修订使用新的输出根目录：
`/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002`。
旧 V2 预检和 smoke 产物仍保留在
`/home/user/training_outputs/terminal/p3_foundation_repair_20261002`，不覆盖历史记录。
source checkpoint 路径、SHA、90D migration 入口、规则臂、无训练/无 held-out/无
checkpoint 写入边界均保持不变。source checkpoint 必须使用显式的
`trajectory_schema_version=executable_trajectory_v2` load config 经过严格 metadata
校验；随后以同一份冻结权重在 `trajectory_schema_version=executable_trajectory_v3`
的 runtime environment 中只做 inference。这是登记的 V2-source → V3-runtime frozen
inference transfer，不表示 checkpoint 已用 V3 训练，也不构成 promotion。
V3 `--check-only` 已完成：exit `0`，状态 `CHECK_ONLY_OK`，登记 `8` 个 arms；
JSON 为
`/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002/preflight.json`，
SHA256 为
`8f21a8203efbdefcdbb88e9e820ce8c1a7a096179e7244f8cce52eddefc6fc40`。其
`checkpoint_inference_binding` 明确记录 source/load 为 V2、runtime 为 V3，且
`source_checkpoint_is_v3_trained=false`；frozen flags 全部为 `false`。

V3 两 tick smoke 的第一次加强校验在发现 safety counter 映射缺失时以 exit `2`
结束；原始日志为
`/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/v3_smoke_verified_retry.log`，
SHA256 为
`ad384d78593bfe95d289fa4be39b1fef7ecad5499d2f1c39943496b71b3e4885`，原始错误为
`two-tick smoke arm has a hard safety issue`。

随后将 smoke reporter 的 uncertified counter 明确绑定到 runtime telemetry
`joint_safety_fail_closed`；缺失时仍保留 `None` 并 fail-closed，不使用 trace dump
reason 替代。最终 retry exit `0`，状态 `SMOKE_OK`，日志为
`/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/v3_smoke_counter_mapping_retry.log`，
JSON 为
`/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002/smoke_verified.json`，
两者 SHA256 均为
`967426e3fa6bc24e85847ea3e1da9352ef1e473e01197a2aeda26689a8ec3710`。三个 source arms
各运行 `2` ticks 且 `censored=true`，validity 均为 `valid`、technical errors 为空、
error/lifecycle error 为 `null`；三个 arm 的 `uncertified=0`、counter source 均为
`joint_safety_fail_closed`、`SafetyInvariantError=0`，collision/NMAC/protected-floor/
hard-separation 事件均为空。repaired88 与 zero-initialized repaired90 的 requested
action tape、wait/elapsed/energy 和 final trajectory hashes 均相等。

完整八臂 `--run` 仍未启动。

预检与每个 arm 输出同时记录 source checkpoint schema、source load schema、runtime
environment schema 和 frozen transfer 标识，便于区分 checkpoint load boundary 与
实际执行环境；全局 checkpoint loader 的严格兼容性规则保持不变。

V3 轨迹一致性诊断已在生成器与参数化器 SHA 对齐后以四条 fixture 运行：exit `0`，状态
`KINEMATIC_CONSISTENCY_PASS`，`finding_count=0`，`technical_error_count=0`，四条的
V3 analytic derivative/integral checks 均通过，旧 node rows 均标记为
`v2_only_nonbinding`。旧产物保留；修正 altitude aggregation 后的产物为
`/home/user/training_outputs/terminal/p3_foundation_trajectory_diagnostic_v3_corrected_20261002.json`，
SHA256 为 `380da099c4595a4370d979f7d9730b782267a48c6de8cb53aa70e11c88e72305`。
四条 fixture 的 analytic altitude bounds 均为
`min_altitude=4.440892098500626e-16`、`max_altitude=40.0`；状态仍为
`KINEMATIC_CONSISTENCY_PASS`，`finding_count=0`，`technical_error_count=0`。
