# 轨迹 V3 修复结果与审核记录（2026-10-02）

状态：用户已同意本轮重大修复。`executable_trajectory_v3` 已以 opt-in 方式实现，当前范围限定为 static geometry；V2 默认路径与 50 条历史 hash 回放保留。本文记录已运行证据和资格边界，不将 P3 标记为已解决或已准入。

## 已实施的实现合同

- V3 使用同一条 quintic 曲线产生 position、velocity、acceleration，并提供全曲线解析的速度、加速度和 altitude bounds。
- 执行前增加 live initial-state guard 与 ring-handoff guard；有限 prefix、同曲线 Safety/runtime error budget 和当前阈值、SOC、reward、动作 authority 继续沿用既有合同。
- V2 default、legacy replay 与 50 条历史 hash 对照保留；V3 只作为 opt-in 物理执行基础。dynamic geometry 尚未 qualification，当前没有真实 Crazyflie/Crazyswarm2 hardware adapter 可核验。

## Newly-run V3 运动学诊断

四 fixture 诊断没有 environment step、training 或 optimizer 操作。修正后的 artifact 为：

`training_outputs/terminal/p3_foundation_trajectory_diagnostic_v3_corrected_20261002.json`，SHA256=`380da099c4595a4370d979f7d9730b782267a48c6de8cb53aa70e11c88e72305`；状态 `KINEMATIC_CONSISTENCY_PASS`，`finding_count=0`，`technical_error_count=0`。

| mode | reservation slot | duration (s) | canonical departure (s) | trajectory hash grouping |
|---|---:|---:|---:|---|
| `legacy` | 0 | 104.74858412577643 | 24.622751480485736 | `c1c127…` |
| `legacy` | 30 | 104.74858412577643 | 24.622751480485736 | `c1c127…` |
| `speed_timing_v1` | 0 | 104.74858412577643 | 24.622751480485736 | `c1c127…` |
| `speed_timing_v1` | 30 | 108.25511970686497 | 30.0 | `c6bb6c…` |

四 fixture 的 position-sampled 最大水平加速度为 `1.2796908 m/s²`，最大垂直加速度绝对值不超过 `0.8205116 m/s²`；同一 artifact 的解析上界为水平 `1.2800000 m/s²`、垂直 `0.8243633 m/s²`。解析导数误差最大值分别为 position→velocity `3.392528924450744e-9`、velocity→acceleration `7.549545766316612e-8`；velocity integral 与 acceleration integral 误差最大值分别为 `4.440892098500626e-14` 与 `9.992007221626409e-15`。解析 altitude 范围约为 `0–40 m`（min=`4.440892098500626e-16`，max=`40.0`）。

这些数值属于 `newly_run` isolated diagnostic。历史 V2 fixture 曾出现 node validator 返回空结果但 position-derived derivative 超出 profile 的记录；V3 artifact 的解析/积分/采样复核为当前物理实现的单独证据，不能替代 closed-loop qualification。

## 已确认的基础实现事实

本轮复核确认了三个 foundation defects：早期 final action proposal 没有进入 physical trajectory（首阶段已实现 opt-in 修复路径）；position/velocity 的执行一致性与 node-only validator 曾不一致；ring braking 的 endpoint-only load 存在 handoff 覆盖缺口。它们足以影响 credit/evidence 的可解释性，但历史 P3 中的因果份额尚未测量，也没有据此确定唯一根因。

既有 critic、larger-batch、gate-fit 等尝试保留为 controlled diagnostic evidence；它们不能单独修复未执行的 proposal 或不一致的物理轨迹。`gamma=lambda=1`、full-MC 等旧结果仍按原 cohort 边界解释。

## 回归、预检与准入边界

- `2×25` 条 V2 historical hash comparison 保持一致；本轮 newly-run safety/dynamics regression 记录为 `123 passed / 1 skipped`、`89.26 s`，持久化日志 SHA 前缀/后缀为 `3272f8…7010`。Unified 首次运行的失败 provenance 保留；最终 retry exit `0`，为 `298 passed / 1 skipped`、`99.21 s`，日志 `/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/unified_named_regression_retry.log`，SHA=`606b5cc5278da7ff3a2637f28099e945c9080b2e5a826e9faa91a8643e94e485`。
- 初始 V3 preflight 记录保留，日志 SHA=`9df70d920006af0879e48355995e5c518b03a176239006014eb1ffc51ed286a5`；随后 retry 为 `CHECK_ONLY_OK`、exit `0`、`arm_count=8`。成功 artifact 为 `/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002/preflight.json`，SHA=`8f21a8203efbdefcdbb88e9e820ce8c1a7a096179e7244f8cce52eddefc6fc40`。其中 source/load schema=`executable_trajectory_v2`、runtime=`executable_trajectory_v3`、transfer=`frozen_v2_source_checkpoint_to_v3_runtime`、`source_checkpoint_is_v3_trained=false`，source SHA=`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`。
- loader-specific tests 为 `12 passed`、`0.78 s`，日志 SHA=`f7ffb14763b8aedeb310e1f3b397d195656f6d09f825197e8efb5aa698bc959b`。Unified 首次运行观察到 `297 passed / 1 failed / 1 skipped`；唯一失败为 altitude exact assertion（actual=`1.9999999999999911`，expected=`2.0`），该 provenance 保留。Focused diagnostic 为 `6 passed`、`0.17 s`；仅两处 assertion 改为 `np.testing.assert_allclose(atol=1e-12, rtol=0)`，无 production change。最终 retry 已 exit `0`。
- 首个 compact smoke 为三 source 各 `2 ticks`、`SMOKE_OK`，runtime V3；energy coverage valid，repaired88/90 requested action tape 与 metrics/hash 完全相等。对应 log/JSON SHA=`5560975f3039b9308097eb3adcb7e1586e07fa18cf43834b2ea67878bfafcab6`，wall time=`162.1 s`。compact output 缺少 arm validity、safety/error 与 load/runtime binding 字段，主审尚未接受为 complete smoke。
- Strengthened smoke retry exit `2`、wall=`162.297 s`，日志 `/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/v3_smoke_verified_retry.log`，SHA=`ad384d78593bfe95d289fa4be39b1fef7ecad5499d2f1c39943496b71b3e4885`；错误为 `two-tick smoke arm has a hard safety issue`，失败输出仍缺少上述具体字段。reporter 使用的 `uncertified_joint_safety_action` 是 trace-dump reason，不是 runtime counter；实际初始化/递增 counter 为 `joint_safety_fail_closed`。字段缺失保持 `None/reject`，不弱化安全规则。evaluator-specific tests 为 `14 passed`、`0.77 s`，日志 SHA=`d5943a9aaa948b3f3e93f328500c1913034f75d2c4c784d5076771172a62e62e`；该次失败时 corrected retry 尚待执行（后续 PASS 见下一条），frozen flags 保持 false。
- 最终 corrected V3 smoke 已被接受：log=`/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/v3_smoke_counter_mapping_retry.log`，JSON=`/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002/smoke_verified.json`，SHA=`967426e3fa6bc24e85847ea3e1da9352ef1e473e01197a2aeda26689a8ec3710`；exit `0`、wall=`161.610 s`、`SMOKE_OK`。三臂各 2 ticks 且 `censored=true`，validity valid、`technical_errors=[]`、error/lifecycle error 为 null；safety 全部 none、`uncertified=0`、source=`joint_safety_fail_closed`、invariant=`0`，无 collision/NMAC/protected/hard events，energy coverage valid。source/load/runtime binding 为 V2/V2/V3，使用同一 fixed source SHA；repaired88/90 action tape 与 metrics 最大 diff 为 `0`，hash/outcome/request identity 相等，final trace count=`1/2/2`，per-arm wall=`38.9024/42.4635/42.6102 s`。
- corrected evaluator tests 为 `16 passed`、`0.76 s`，日志 SHA=`14ebcecee0f92e2d789387eb3b1dc42b070b8fca462e2c1406cf81f9c279b7e8`。
- Core implementation commit 为 `f1dfb1034d6517fb1b5197035dc92e3eabe482ba`（24 files），evaluator/docs commit 为 `9b0d13cdd85a7c59170b68a25f8e12a1dfbad22e`；index empty，remote-v 为空，未 push。
- 上述历史预检汇总阶段，八臂 zero-training matrix 尚未启动；随后 exact-one registered launch 的初始健康快照见下方，不能把该历史行解读为当前 launch 状态。此前记录的 held-out、training、optimizer step、checkpoint write、default promotion 均未执行，`PPO_ALLOWED=false`；P3 仍未 solved/admitted。

## Bounded performance preflight（`newly_run`）

同一 cell 1 `legacy88`、相同 2-tick 输入的 profile artifact 为 `/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/profile_legacy88_two_ticks`，结果 SHA=`c8b056d839b63947d696c9b4e1d2240332769476a388fe16453854a46bc9e077`，pstats SHA=`46593f320e7a3dafa6df6914bff10819fea325eba6ccdb4d69364496b048d2c6`。命令 exit `0`；2 ticks、`censored=true`、validity valid、safety zero。

profile 记录 `202221455` calls、`281584` bounds calls、累计 `69.340 s`；under-profiler actual step 为 `73.735 s`，含 reset 的 total 为 `97.060 s`。该范围包含 reset，不能外推完整矩阵运行时。

主审选定并完成同一 scalar analytic-bounds algorithm 的 batch priming targeted qualification；不改变 threshold、tolerance、grid、curve 或 hash。最终 unified regression 与 before/after compare 已完成，`PPO_ALLOWED=false`。

## Batch bounds targeted qualification（`newly_run`）

server NumPy `2.5.1` 的 `256`-segment 对照中，scalar 为 `0.0893419 s`、batch 为 `0.0448305 s`，观测比值约 `1.9929×`，`max bounds diff=0`；local NumPy `2.3.5` 的 `64`-segment 对照同样为 `max diff=0`。focused helper tests 为 `19 passed`，日志为 `/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/batch_bounds/focused_pytest_retry.log`，SHA=`792cb1ad7160e02b5e4f0e2a909338ec8afae341a9ee92c12be8b58ab50412c3`；V3 caller integration 为 `55 passed in 5.37 s`，日志 SHA=`1fa45367ac07d2137d8adf114931054665819e8505caf49833015acf1db46ffa`。

真实三臂、每臂 `2` ticks 的 batch smoke 为 exit `0`、`SMOKE_OK`；JSON=`/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002/smoke_batch_verified.json`，log/JSON SHA=`68a02d277dee0b847827188eeab38415eb6bb2708d9d02fd687f6666f4a3373a`。三臂 source/load/runtime schema 均为 V2/V2/V3，使用 fixed source checkpoint，frozen flags 全为 false；除 UUID plan id 外，requested/accepted/actual metadata、hashes、safety、validity、wait/elapsed/energy 等非计时字段均与 corrected smoke artifact 相等。观测到的三臂耗时为 `27.885836/30.494660/30.382251 s`，此前 corrected smoke 为 `38.902411/42.463497/42.610227 s`；该计时比较未独立测量并发效应。

最终 `38`-file unified suite 已 exit `0`，为 `319 passed / 1 skipped`、`99.25 s`；日志为 `/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/unified_named_regression_batch.log`，SHA=`4bbc1fff9c5bb823e7476c28e8a2e646d35852af459cad69f4efacc931157498`。四个既有 before/after JSON compare 为：`field_diff_count=0`、`point_pva_time_diff_count=0`、`segment_length_diff_count=0`、`bound_diff_count=0`、`max bounds diff=0`，所有 validation errors 均为 `[]`；before=`/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/batch_reference_before.json`（SHA=`bc1f4d3d0f9ea7ce6009705b1516757a551b1e7fe104daa142ab75ff761d955b`），after=`/home/user/training_outputs/terminal/trajectory_v3_repair_20261002/batch_reference_after.json`（SHA=`168be29fc93d1c5f1281e889e8701475535464132e534817ff4eebfe946d188f`）。

## Registered matrix launch（`newly_run`）

已确认本轮 exact-one launch：UTC `2026-10-02T12:15:41.616174+00:00`，PID=`2303790`，argv=`/home/user/envs/terminal_env/bin/python -u -m terminal_marl.scripts.evaluate_p3_foundation_repair_20261002 --run`，cwd=`/home/user/projects/terminal`，outroot=`/home/user/training_outputs/terminal/p3_foundation_repair_v3_20261002`。launch metadata 绑定 execution/performance commit=`e5aa7af315ecefb786464740122d08d8a6816c90`（7 files，remote-v 为空，未 push）、source checkpoint SHA=`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`；14 个 source hashes 与 manifest 一致。

启动时初始 root progress snapshot 为 `running`、`ticks=1`、`completed_arms=0/8`、`cell=1 legacy88`、wall=`37.6963 s`。对应 arm1 为 `ticks=1`、`simulation_seconds=1`、`active=12`、`generated_requests=12`、`queue=0`、`service=0`、reward=`-0.1778916698542347`、`safety_violation_status=none`。该启动快照中 PID correct alive；`matrix.log` 已存在但为 `0` bytes，未见 warnings。 最新只读快照（UTC `12:18:33`）显示 PID alive、`Rsl`、CPU=`105%`，root progress 仍为 `running`、`1/8` 臂完成，cell=`1/repaired88` 已到 `25 ticks`、safety=`none`；legacy88 `result.json` 已存在，comparison 尚无。该快照仅为运行进度，不代表 legacy 首臂 qualification。 cell1 legacy88 result SHA=`cb1cae6bab957f2af1ceb34ada7bb0dbc921e8881a8c854e9a98274acdffbdb1`，status=`error`、stop_reason=`exception`，RuntimeError 为 `could not time-parameterize a dynamically feasible continuous landing`；记录 `13` successful ticks（last tick=`12`、simulation=`13`），hard/invariant/uncertified=`0`、safety=`none`，ledger-only validity=`valid` 但 arm error 存在，`12` requests、`0` resolved、`12` unresolved，expected cohort `24` 未达到、`censored=null`。这是 technical trajectory failure，不作 business performance 或 learning 结论。primary 正在 bounded replay exact failed first `14` ticks 以获取 traceback，matrix source 保持 immutable。

上述 launch 段记录启动时与中途快照；后续终态见下方 registered matrix terminal snapshot。heartbeat automation `p3-v3` 已在终态由 primary 删除。此前 held-out、training、optimizer step、checkpoint write、default promotion 均未执行，`PPO_ALLOWED=false`。

## Registered matrix terminal snapshot（`newly_run`）

comparison artifact SHA=`d7cee7a34092cf885aa213961fa86ceba68d5dfb3b125607164d4e126183381b`，status=`COMPLETED_WITH_ISSUES`，`executed_arms=8/8`、`stopped_early=false`、wall=`1542.282646 s`。UTC `12:44:37` 时 PID=`2303790` 已 absent，OS exit code 尚未观测到。八臂均为 `status=error`、`stop_reason=exception`、`censored=null`、`resolved=0`；ledger-only validity 均为 `valid`、`errors=[]`，但 arm error 使其不构成 arm qualification。

| cell / arm | successful ticks / requests | observed error |
|---|---:|---|
| C1 `legacy88` | `13 / 12` | continuous landing time-parameterization error |
| C1 `repaired88`、`repaired90_zero_initialized` | `84 / 17` each | `ring_hold_duration_shorter_than_required` |
| C1 `nominal_rule_legacy88` | `45 / 15` | continuous landing time-parameterization error |
| C1 `nominal_rule_repaired90` | `9 / 12` | `ring_hold_duration_shorter_than_required` |
| C2 `legacy88` | `2 / 12` | continuous landing time-parameterization error |
| C2 `repaired88`、`repaired90_zero_initialized` | `118 / 19` each | `ring_hold_duration_shorter_than_required` |

八臂 hard/invariant/uncertified 均为 `0`，event lists 均为空，frozen flags 均为 false。两 cell 的 zero-initialized comparison 中 requested tape 最大 diff=`0`，wait/elapsed/energy metrics 最大 diff=`0`，trajectory hashes、request identity、outcomes 相等；这些记录均只到 error 为止，属于 partial evidence，不是 performance qualification。

root 已确认 isolated legacy 14-tick probe JSON SHA=`dabeed4586a932d5468651340d1e6280f094516a7ee18f4f51112a2fd605c0dc`；caller stack 路径为 `envs/parallel_env.py:709 → core/energy_reachability.py:289 → core/trajectory_generator.py:1862 → core/trajectory_time_parameterization.py:1391`。该 probe 记录 uncaught fallback candidate error 的 caller defect；参数化的后续数值复核见下方。

旧 matrix 的 artifact 与 source binding 保留，不覆盖失败记录。当前修复与重试状态见下方，`PPO_ALLOWED=false`。

## Energy-query 与 altitude numeric repair（primary 实施）

新运行的 isolated bounds probe（`numeric_probe/bounds.json`，SHA=`20f3d7433bea72d0de9d3cfbe1466c46f3377eb147e21232b9ba129212195fae`）复用首臂最后失败的七节点 departure。原有限搜索的 16 个 profile 均在末段达到 `120.00617783094059 m`，超出 `120 m + .001 m` 的既有 altitude tolerance；水平加速度与速度均在上限内。四档 speed scale 均保留相同高度 overshoot，说明单纯继续降速不能消除这个几何插值反例。这是该捕获 fixture 的数值原因，不泛化为所有不可行轨迹的原因。

另一个 registered C1 rule-repaired 10-tick probe（`numeric_probe/ring.json`，SHA=`78dd57f6bc840261fad65e9109f323217d38d244b142509c26092f9eae97f58c`）在 9 个 successful ticks 后复现能量预算的短 pad wait：等待 `0.21230663739135025 s`，current/target speed=`18/20 m/s`，而完整 smooth speed ramp 至少需要 `2.8385059450509385 s`。预算使用完整-ramp API，实际环上执行使用 partial finite-step integrator，二者合同不一致。

primary 亲自修改三个 core 文件，并编写 `tests/test_v3_energy_reachability_failures.py`（含旧 fixture 修正的 code/tests commit=`5bd9b62e218d71640deb8bd73a9c691af454052b`，6 files，remote 未配置、未 push）：

- V3 原曲线通过时继续使用原曲线；仅对 altitude 越界的未通过候选，约束内部节点的 vertical slope，使 quintic Bernstein controls 在既有高度包络内。保持 live/terminal p/v/a、节点、时序、XY 与 acceleration；之后仍需原全曲线 dynamics validator 通过。
- V3 energy-query 的未来 holder 使用与环境执行相同的 `ring_motion_step` 和 `physics_dt`，在 pad ETA 保留尚未完成的 speed transition。
- 明确耗尽有限搜索的轨迹使用 `TrajectoryInfeasibleError`；能量查询跳过无可执行 witness 的兼容 pad，全部不可行时返回原 `inf/-inf` 拒绝预算。其他 RuntimeError 继续暴露。安全间隔、高度上限、SOC/energy 门槛、reward、authority 和 V2 默认路径未调整。

首轮扩大 targeted suite 为 `56 passed / 1 failed`（日志 `energy_numeric_repair_targeted.log`，SHA=`6bf2ac10aec1d066c65d1a99a73946a61390d532abf626a9c95b5e4e49624e9a`），所有新增 cases 已通过。唯一失败是旧 warm-start fixture 假定 `.35` SOC 必须拒绝。冻结 pre-repair helper 与当前 helper 的 V2 numerical comparison 都为 required SOC=`0.2060807011022114`；artifact=`warm_start_soc_probe.json`，SHA=`8f434049537e830f4ca5a5559e44e6a79d0f39cfb11708c1b7b1dd720644e6a0`。因此 primary 仅将拒绝测试的 SOC 改为明确不足的 `.15`，未改变生产 energy 规则。targeted retry 为 `57 passed in 3.38 s`、exit `0`，日志 SHA=`65813bbb49b79be2e83ad6367ba7214d50319a8160e021b9f3e479336b2d2296`。

既有四条 V3 reference 的新 capture（`energy_numeric_reference_after.json`，SHA=`b834bcd91029e386084bcb95111f062bf341c9275cf206be3a65cab29c9bdfcf`）exit `0`、所有 validation errors 为空；除 UUID plan_id 与 source hashes 外，与 `batch_reference_after.json` 的全部 field、point p/v/a/time、segments、bounds 和 trajectory hashes 完全相同。

两个原异常 prefix replay 已完成为 `PREFIX_SMOKE_OK`、exit `0`；artifact=`energy_numeric_repair_smoke/result.json`，SHA=`033492a6396bf942c706153a3ed92e1432ab9417b94342d1f9c09bbd900f1403`。C1 legacy88=`14 ticks`、rule-repaired90=`10 ticks`，均越过各自原 exception 位置；两臂 error=null、technical errors=[]、safety none、uncertified/invariant=0、source hashes before/after 相等。两臂均 censored、resolved=0，仍是接口/异常修复证据，不能作为完整性能 qualification。预编译首次因未使用正确 cwd 返回 ENOENT，指定 cwd 后 exit 0；诊断没有生成 launch metadata 或持久化 stdout log，实际 JSON 保留，PID=`2315795` 已退出。扩大 unified suite 尚待执行。

扩大回归采用本轮明确列出的 `33` 个 test files；之前 `38`-file 命令清单未保存，本轮不声称重复原清单。首轮为 `280 passed / 3 failed / 1 skipped in 167.42 s`、exit `1`，日志=`energy_numeric_repair_unified.log`，SHA=`36900b80bc5824a42d4eb5d2ba9b773d2ad870d640ec6921eabddcc50bee373e`。三个失败均在旧 `test_emergency_energy_v2.py`：solo energy `1200/1500+.1=.9` 并没有超过 `1`；middle `.30` 和 inner `.249` SOC 高于各 pad 立即应急所需 SOC，不能作为 insufficient/under-reserve fixture。

primary 的 frozen old-source/current-source probe 确认三个前提在修复前后相同；verified artifacts 为 `emergency_fixture_binding_verified_baseline.json`（SHA=`e77b94af71240388e5ee59d33788f58fcc9d64f16f84e0a57ab7b2c667b3666b`）及 `emergency_fixture_binding_verified_current.json`（SHA=`649c6ec962467722d2aea392c7137550b107dc0d8af85ab591da86a4863f92ff`）。baseline 的实际 generator / parameterizer definition files 明确绑定 `.tmp/*pre_numeric_repair*`，current 绑定 core 文件；除路径外输出相同，middle required SOC=`.1829324892–.2095224883`、inner=`.1465957755–.1552703699`，均 accepted。初次 probe 只替换模块但未重绑预先导入的 env globals，因此其 baseline artifact 不接受为旧执行证据，两个初始 artifacts 仍保留。

primary 将 nominal solo energy fixture 设置为 capacity（加 reserve 后超过 `1`），两个不足 SOC fixture 设置为 `.5 * min_landing_soc`。生产门槛未改变；之后只重跑修改的 emergency-energy 文件及 warm-start / 新 V3 regression files，不重复已通过的其他 30 个文件。focused retry 为 `27 passed in .31 s`、exit `0`，日志=`energy_numeric_repair_emergency_retry.log`，SHA=`14406e8063484cccb15dc872f1057585b369b748b59f511ad27f39f31c1e74df`。本轮 33-file 集合的三处失败均已闭合；该结论综合初次回归与 focused retry，不声称重新运行完整集合。唯一 skip 仍为旧 V2 `diagnostic_snapshot` API 已移除。

R2 preflight=`CHECK_ONLY_OK`、exit `0`、8 arms；artifact=`/home/user/training_outputs/terminal/p3_foundation_repair_v3_r2_20261002/preflight.json`，SHA=`9bd34d9217d19ff0d2555d7219858dc235db9a579b808df91391c993534d8ad2`。新 manifest SHA=`ddaf93ecdf774224b3fd2cd1a4c2a69fb8dde8745a3b44791475bb33bd0914e1` 绑定 15 个 source hashes（增加 energy_reachability），local/server 全相同；checkpoint SHA 与 V2/V2/V3 binding 保持固定，frozen flags 全 false，compile exit `0`。launcher dry-run 正确指向 R2 root，未启动 full matrix。

上述结果区分 `newly_run` 与 `preexisting_artifact`，不构成完整 audit 或业务性能结论。相关协议见 [P3 foundation repair protocol](p3_foundation_repair_protocol_20261002.md)；基础链路记录见 [P3 foundation review](p3_foundation_review_20261002.md)。
