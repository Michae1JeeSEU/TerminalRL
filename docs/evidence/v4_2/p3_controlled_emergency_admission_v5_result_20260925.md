# P3 受控应急 PPO 准入 V5：Smoke、Entry、Pilot 与 Formal Readiness

协议：[P3 Controlled-Emergency PPO Admission V5](p3_controlled_emergency_admission_v5_20260925.md)。本页记录已完成的 smoke、entry、bounded pilot technical evidence 和最终 formal readiness check；正式大规模训练尚未运行，P3 qualification 与 checkpoint promotion 仍保持关闭。

## 冻结来源与实际绑定

本轮使用 repair candidate E04：

```text
SHA256: e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353
server: /home/user/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt
local: D:/7_codespace/training_outputs/terminal/random_soc_gate_repair_v1_20260925/fit846401/random_soc_gate_repair_v1.pt
```

Entry manifest 的实际绑定为：protocol SHA256 `732079236fc1554f63c66de46bf910e23ecd189d2e6bf5baa06873c4ae386676`，schema digest `f896d7e20d49f063c3fa342b7a1a510742a3fd8fd09a7769fb60d537ea69b5dc`；Observation V2.1 `88D`、V4.2 hierarchical action、`Teacher=false`、execution scheduler、fixed geometry、shield enabled。entry 使用 fresh Adam 且 `entry_updates_discarded=true`；训练 scope 为 `gate_adapter`、`critic_encoder`、`critic_head`，actor/proposal/temperature 相关 scope 冻结。实际字段见 [entry manifest](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/entry/manifest.json)。

## Smoke

`849000/dense24` 绑定完整 `1800 s` horizon 配置，实际执行 `2` steps、仿真时间 `2.0 s`，`terminal=false`、完成数为 `0`。安全计数（collision、NMAC、hard separation、uncertified execution、`SafetyInvariantError`）均为零；reward component totals、reward total 和 `reward_components_consistent=true` 已落盘。Smoke manifest 的 `PASS=true`，但它仍是两步协议检查，不是性能或资格结论：[smoke manifest](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/smoke/manifest.json)。

## Stochastic no-update Entry

三个 full-horizon entry arm 均 `status=completed`、`hard_gate_pass=true`、`full_resolution=true`、exact replay true、zero failed-energy、zero unresolved，且安全计数为零。等待指标由各 arm lifecycle rows 计算；low-SOC 列同时给出样本数。

| seed | 场景 | steps / 仿真秒 | normal | emergency_landed | failed_energy | unresolved | priority-weighted wait (s) | low-SOC wait (s, count) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 849101 | flow6 | 632 / 632 | 6 | 0 | 0 | 0 | 14.733390 | 4.704212 (1) |
| 849102 | flow12 | 1000 / 1000 | 12 | 0 | 0 | 0 | 33.533007 | 32.775745 (1) |
| 849103 | dense24 | 720 / 720 | 21 | 3 | 0 | 0 | 114.736657 | 89.956835 (8) |

直接结果来源为 [entry manifest](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/entry/manifest.json) 及三个 [flow6 summary](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/entry/flow6_849101/summary.json)、[flow12 summary](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/entry/flow12_849102/summary.json)、[dense24 summary](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/entry/dense24_849103/summary.json)。三个 summary 均记录 `terminal=true`、`reward_components_consistent=true` 以及全零安全计数。

该 entry 满足受控应急语义：`emergency_landed` 是经过安全层认证且没有能量耗尽的完成结果，可以进入性能与 reward 统计；dense24 的 3 个应急结果没有被改写为失败，也没有触发 entry 的自动拒绝。协议没有把 emergency 数量归零设为 admission hard gate。`failed_energy`、unsafe、未认证执行，以及 entry/development/evaluation 阶段的 `unresolved` 或完整请求未闭合，仍属于相应技术失败条件；formal training 的 horizon 截断只作训练报告字段，不直接改写为 evaluation 安全失败。

Entry manifest 的授权字段为 `PASS=true`、`PPO_ENTRY_ALLOWED=true`、`PPO_ALLOWED=false`、`FORMAL_P3_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`checkpoint_promotable=false`。这只授权 bounded pilot 候选状态。

Entry 控制文件的 command mtime 到 completed status mtime 约为 `118.02 s`；这是 wrapper 控制文件的时间跨度，不是 summary 提供的独立 wall-clock 计时。Entry 目录实际占用 `63,532,070` bytes（约 `63.5 MB`）。当前不据此外推正式 `720` 个训练 episode 或 `126` 个 development episode 的总时长和磁盘需求。

## Pilot 状态

Pilot 已完成，实际 manifest 为 [pilot manifest](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/pilot/manifest.json)：

```text
stage: V5_PILOT
status: COMPLETED
PASS: true
PILOT_TECHNICAL_PASS: true
FORMAL_P3_ALLOWED: false
selected initial SHA: e04e5b4a8f8365b52a41f66c96ad1f66a09a554ea64118fcab8040d64175d353
final pilot checkpoint SHA: 6ef5dfd44ae46ef1d3f11182983b66e789043e6498afcdb9794127916926d85c
final pilot checkpoint: /home/user/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/pilot/checkpoint_6.pt
```

Pilot 的 6 个是 episode-level PPO update calls；每次 call 内的 minibatch/accepted-update 统计是另一层指标。本轮 6 个 call 均通过，累计 `296` 个 accepted minibatches、`0` 个 rejected minibatches；六个 checkpoint restore 均 exact，proposal subgraph freeze、requested likelihood、runtime shortlist、KL、safety 和 reward gates 均通过。训练记录合计 `96` requests、`82` normal、`14` emergency_landed、`0` failed_energy、`0` unresolved；其中 dense24 为 `14/72` emergency，sparse flow6/flow12 为 `0/24`。

paired development 的 performance report-only 对比均表示 after-before 增量：flow6 priority-weighted wait `Δ+0.000 s`、low-SOC wait `Δ+0.000 s`；flow12 priority-weighted wait `Δ+0.179 s`、low-SOC wait `Δ+0.500 s`；dense24 priority-weighted wait `Δ+2.763 s`（约 `+2.25%`）、low-SOC wait `Δ-44.036 s`。sparse emergency 为 `0→0`，dense emergency 为 `4/24→3/24`。主代理审查认为这组 balanced performance evidence 可接受；这些差异是有限 development evidence，不是统计证明，也不代表课程毕业。

pilot training 的 horizon 截断仍按 training report-only 记录，不直接改写为 evaluation 安全失败。pilot seeds `849201–849206` 已各完成一次；development seeds `849301–849303` 按协议成对复用 before/after。formal seeds `850/851/852` 与 development/heldout seeds `853/854` 尚未消费；历史 seeds `829001–829006` 保持 untouched。

## 最终只读 preflight

最终 genuine check-only preflight 已通过，实际决策文件为 [formal admission decision](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_admission_decision.json)。该文件的 stage 为 `P3_FORMAL_ADMISSION_DECISION_V5`，`FORMAL_P3_ALLOWED=true`、`PPO_ENTRY_ALLOWED=true`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`、`PPO_ALLOWED=false`、`formal_not_started=true`；因此 formal readiness 已达成，但正式训练尚未开始，qualification 和 P4 promotion 仍关闭。

只读检查结果见 [formal check-only run 0](D:/7_codespace/training_outputs/terminal/p3_controlled_emergency_admission_v5_20260925/formal_check_only_run0.json)：`status=CHECK_ONLY_OK`、`check_only=true`、`strict_load=true`、`fresh_optimizer=true`、`trainable_parameter_count=18`、`output_created=false`、`heldout_executed=false`，且 `run_index=0`。当前实际配置与 formal protocol 已验证；没有 reset、update 或 formal rollout。9 个 formal tests 在最终 YAML 状态变更后再次通过。

正式决策文件 SHA256 为 `d4f34955bd2230e5f531ab71f451d7eb538fbd6f16cc6bf05387ff0a673d2d8d`。决策使用 E04 repair candidate 作为 fresh formal initialization，pilot manifest 的原始 `FORMAL_P3_ALLOWED=false` 保持不变；pilot checkpoint 仍不可作为 formal 初始化来源。formal campaign 尚未启动。

## 定向测试与未改变历史失败

本轮同一组 `29` 个定向测试（formal `9` + V5 admission `11` + student rollout/reward `9`）在本地通过；服务器也记录同一组 `29 passed`，耗时 `1.61 s`。两者是同一测试组，不相加为 `58`；没有远程代码变更。

以下三个历史 V3/V4 测试失败保持原样，未修改历史 source hash 或旧测试：

1. `tests/test_p3_mixed_priority_soc_training_v3.py::test_v3_source_hashes_include_runner_protocol_and_reward_sources`：`V3InputError: reward source changed: terminal_marl/core/rewards.py`。
2. `tests/test_p3_mixed_priority_soc_training_v4.py::test_v4_source_hashes_are_relative_and_include_runner_protocol`：同一 `reward source changed` 错误。
3. `tests/test_p3_mixed_priority_soc_training_v4.py::test_v4_formal_stops_after_first_passing_checkpoint`：`assert result["PASS"] is True` 失败，实际为 `False`。

当前结果页的状态为：`FORMAL_P3_ALLOWED=true`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。正式大规模训练尚未启动，也不由本页自动启动。
