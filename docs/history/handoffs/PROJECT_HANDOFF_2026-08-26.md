# Terminal MARL 项目交接（2026-08-26 架构修订）

> **用途**：下一次 Codex 对话的当前状态入口。本文不是独立 canonical specification；具体机制以 `research_positioning.md`、`requirements.md`、`design_spec.md`、`technical_spec.md`、`execution_plan.md`、`revision.md` 和新增 `A5V2_TEACHER_STUDENT_MODIFICATION_PLAN.md` 为准。

## 1. 当前主线结论

- 主代码：`terminal_marl/`；`src/` 为 legacy。
- A0–A4 的三环、arrival/admission、LZ/service、SOC/priority、canonical trajectory 与 hard-safety thresholds 保留。
- 旧 A5 共享 PPO checkpoint 对应 41D actor observation。它支持 shared actor + variable-N centralized critic，但 actor **没有显式 leader/follower/pred/succ/CPA/TTC traffic context**。
- A6.11–A6.21 audits 证明 scheduler/safety/reconfiguration 机制对系统行为有很强影响，并暴露 observation/action causality 与 identifiability 问题。这些 evidence 保留，但不再意味着“下一步只能继续 A6，不得重训低层”。
- 2026-08-26 起主开发线切换到 **A5-v2 teacher-guided topology-aware low-level learning**。

## 2. 新 canonical architecture

```text
π_L requested 3D intent
→ ExecutionScheduler (single requested candidate; no repair search)
→ canonical trajectory
→ SafetyFilter verifier
→ Predictive Minimal Safety Shield (future-risk only)
→ execute checked plan

TRAINING ONLY:
TeacherScheduler / teacher reconfiguration repair
→ bounded correction witness
→ intervention penalty + optional imitation loss
→ anneal to OFF
```

低层动作保持 `[release_request, speed_residual, merge_time_residual]`。`safe_distance=80 m` 等 hard thresholds 不降低。

## 3. Observation V2

新 actor schema：`phaseA_low_obs_v2_topology_87d`：

- self/flight 18D；
- mission/resource identity 12D；
- 5 semantic neighbor slots × 9D = 45D；
- system/flow 12D。

邻机槽位：same-ring leader/follower、target-ring predecessor/successor、nearest predicted conflict。每槽包含 presence、gap/distance、relative speed、closing rate、d_CPA/t_CPA、priority delta、reservation/committed flag。

## 4. Variable-N position

当前 A5 的 shared actor + masked mean/max critic 只部分解决 variable-N。新主方法候选用 topology slots 让 actor input 与 N 解耦；all-neighbor recurrent/attention 方法作为正式 comparator。不要直接因为 Waltz et al. (TRC 2024, DOI 10.1016/j.trc.2024.104806) 使用 LSTM-TD3 就切换 optimizer；先比较 representation，再决定 temporal memory/algorithm 是否需要改变。

## 5. 正确训练顺序

```text
T0 freeze legacy evidence
T1 implement V2 observation + role split + shield
T2 teacher dataset / optional BC
T3 teacher-guided low PPO: full → immediate → label-only → off
T4 teacher-OFF student qualification under ID/OOD-N
T5 freeze low-v2; train high-level
T6 full hierarchical evaluation, all teachers OFF
T7 optional alternating fine-tuning
```

禁止两个随机 policy simultaneous joint training。

## 6. PPO correction critical rule

必须记录 `requested / teacher_corrected / execution_applied / shield_applied`。PPO buffer 的 action/log_prob 永远对应 `requested`。Teacher corrected action 不得替换 PPO action 后继续使用 requested log_prob。

## 7. 下一次 Codex 的第一批任务

1. `observations.py` + topology utilities：实现 87D V2；
2. scheduler split：Execution vs Teacher；
3. `runtime_traffic_safety.py` → PredictiveSafetyShield；
4. `parallel_env.py` action provenance；
5. tests 先行，不立即启动长训练。

详细逐文件协议见 `docs/A5V2_TEACHER_STUDENT_MODIFICATION_PLAN.md`。

## 8. Performance gate status (2026-08-28)

Dynamics correctness remains PASS.  P0 performance refactor is in progress:
trajectory hash caching, analytic ring limiting, duration estimation,
TopologyCache and shield prediction reuse are implemented and covered by
regression tests.  A holder-only 60-UAV witness improved from 3.98 to 9.09
env steps/s with zero safety events.  However Full Teacher still reaches the
legacy deepcopy forecast path and GUI/viewer decoupling has not been
qualified.  Therefore Performance Qualification is **FAIL** and the next
step is continued P0 profiling/refactoring, not C0 capacity calibration or
official A5-v2 PPO training.  Benchmark artifacts are under
`D:\\7_codespace\\training_outputs\\terminal\\p0_performance_refactor`.

## 9. P0 final handoff (2026-08-29)

Section 8 is superseded: Performance Qualification is **PASS**.

- Full Teacher future-delay repair is implemented by lightweight
  `ForecastWorld`; the production forecast no longer deep-copies or steps a
  `TerminalParallelEnv`.
- `physics_dt` remains 0.2 s. Safety sampling converged across 0.2/0.5/1.0 s;
  1.0 s is selected as the coarsest safety-equivalent value.
- Trajectory density converged to Bezier 41 / Final 51 with a 0.05 m
  dense-reference tolerance and unchanged EH216-S feasibility decisions.
- Prediction/shield work follows cheap broad phase -> cached prediction ->
  swept narrow verification, including post-handoff holder transients.
- Viewer rendering is latest-only asynchronous, drops stale frames, reuses
  persistent actors, defaults to stride 5 with trajectories hidden, and does
  not pace the simulation loop.
- N=30 sustained: mixed 11.43, committed-heavy 27.26, shield-risk 20.93,
  Full Teacher 12.04 steps/s; 198/198 regression tests passed.

Canonical evidence:
`D:\\7_codespace\\training_outputs\\terminal\\p0_performance_refactor\\final_qualified_v4\\PERFORMANCE_QUALIFICATION.md`.
No C0 or PPO was run. Next authorized phase is C0 capacity calibration.
