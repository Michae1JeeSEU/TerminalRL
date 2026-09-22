# P3 当前 MDP 基线协议（2026-09-22）

## 目的与冻结边界

本协议冻结新安全/能量语义下的 P3 baseline evaluation，不启动 PPO、不更新参数，
也不把结果直接升级为 P3 qualification。执行顺序固定为：

1. 先用全新种子复评 P1/P2 checkpoint 的资格与效果；
2. 再执行本协议的 P3 高 SOC qualification cohort；
3. 再执行独立的低 SOC stress cohort；
4. 只有高 SOC cohort 中 rule 与 original K3 均通过，才允许另行设计 P3
   train/dev/heldout protocol；该条件不自动启动训练。

所有本协议种子必须在 protocol freeze 后才可评估，且不得与 pilot、历史或彼此重叠。
种子一旦消费，禁止 rerun、reset 或 tuning；发生安全错误时立即停止并保留失败证据。

## 固定实现与场景

- Observation V2.1 `88D`，V4.2 hierarchical action，runtime proposal shortlist `K=3`。
- candidate 是 original K3 frozen checkpoint，SHA256：
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`；逐 seed 与
  rule 配对。无 Teacher、无 labels、无 updates。
- 安全语义为首选间隔 `80 m`、保护下限 `40 m`、`<10 m` collision event；进入
  `40--80 m` 缓冲区必须有 swept-segment 与 recursive backup certificate。
- 能量容量为 `1500`，touchdown 必须满足 `SOC >= .25`。priority 为 `{1,3,5}`，
  mission probabilities 为 `[.45,.25,.10,.20]`；沿用 priority time weight `1.0`、
  SOC urgency time weight `2.0`、阈值 `.65`。
- dense24 使用 12 架 warm start，环分布 `[6,4,2]`，再以 `15 s` 间隔进入 12 架，
  horizon `1800 s`；P2R 使用 `nearest_available`、`downstream_priority` 和 final
  capacity `3`。

## Cohorts 与 gate

### High SOC qualification

SOC 为 `Uniform(.90,1.00)`，fresh seeds 为 `835001–835003`，每个 seed 运行 rule
与 original K3 的 paired dense24 record，共 6 records。每个 controller 必须完成
`24/24`，无 diverted/unresolved request，terminal 在 `1800 s` 内，安全计数全零，
action tape exact replay，且满足 no Teacher、no labels、no updates 与 K3 runtime
shortlist contract。报告 request-to-touchdown、等待和 paired deltas；完成者条件等待
不能单独代表全体表现。

此前 high-SOC pilot `seed=831002` 仅作 pilot：rule/K3 均 `24/24`，安全与 diversion
均为零，exact replay；completion time 为 `686/740 s`，wait 为 `99.5899/114.6562 s`。
它不属于本协议 qualification，也不证明普遍可达性。

### Low SOC stress

SOC 为 `Uniform(.55,.90)`，fresh seeds 为 `835101–835103`，同样 paired rule/K3、共
6 records。hard gates 是 24 请求完整记账、安全计数全零、exact action tape replay、
no Teacher/labels/updates 和 runtime shortlist contract。完成比例与低 SOC 分流是报告
结果而非 all-complete gate；必须报告完整请求分母、completed-only wait 的条件性、
request-to-touchdown 以及 paired deltas。

## Evaluator 与服务器执行

评估器为 `terminal_marl/training/evaluate_v4_2_p3_current_mdp_baseline_v1.py`，配套
定向测试为 `tests/test_p3_current_mdp_baseline_v1.py`。当前 10 个定向测试与 compile
检查均通过。使用真实 original K3 SHA
`4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c` 的 high-SOC
diagnostic seed `835000` 做 CLI 20 步 smoke，产生 rule/K3 两条 records；safety、exact
replay、Teacher-free、labels/updates 与 K=3 shortlist gates 通过，未完成 `24` 架是
smoke 结果边界，formal seeds 消耗为 `0`。评估器按 cohort 分开运行，逐 episode 落盘；
safety/replay 错误即停并保留已有证据。low-SOC completion fraction 与 low-SOC diversion
是报告结果，不是 all-complete gate。

正式 high-SOC baseline 已在 P1/P2 formal 复评和源码同步确认后完成。服务器输出
`/home/user/training_outputs/terminal/p3_current_mdp_baseline_high_soc_20260922`，
manifest 状态为 `COMPLETE`，6/6 records 已落盘；但 rule seed `835001` 与
original K3 seed `835003` 各有 2 条 `diverted_low_soc`，所以 high-SOC all-complete
gate 失败，`PASS=false`、`P3_QUALIFIED=false`、`PPO_ALLOWED=false`。完整数值和
provenance 见 [P3 high-SOC 正式结果](p3_current_mdp_baseline_result_20260922.md)。
low-SOC stress 已按原冻结边界独立完成：服务器输出
`/home/user/training_outputs/terminal/p3_current_mdp_baseline_low_soc_stress_20260922`，
manifest 状态为 `COMPLETE`、6/6 records 已落盘，`PASS=true` 仅表示完整记账、安全、
replay、Teacher-free、labels/updates 为零与 runtime shortlist 等 integrity gates
通过；完成率和低 SOC 分流仍是报告结果，不授权 PPO。完整数值和 provenance 见
[P3 low-SOC stress 正式结果](p3_current_mdp_low_soc_stress_result_20260922.md)。
checkpoint 与策略只读，两个 cohort 使用不同输出目录，执行命令如下：

```bash
# High-SOC qualification: 835001--835003
python -m terminal_marl.training.evaluate_v4_2_p3_current_mdp_baseline_v1 \
  --cohort high_soc_qualification \
  --output "$HOME/training_outputs/terminal/p3_current_mdp_baseline_high_soc_20260922" \
  --history-root "$HOME/training_outputs/terminal" \
  --checkpoint "$HOME/training_outputs/terminal/p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt"

# Low-SOC stress: 835101--835103
python -m terminal_marl.training.evaluate_v4_2_p3_current_mdp_baseline_v1 \
  --cohort low_soc_stress \
  --output "$HOME/training_outputs/terminal/p3_current_mdp_baseline_low_soc_stress_20260922" \
  --history-root "$HOME/training_outputs/terminal" \
  --checkpoint "$HOME/training_outputs/terminal/p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt"
```

## Authorization

本协议冻结为 baseline evaluation：`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、
`P4_ALLOWED=false`。不作 universal feasibility claim；任何后续 P3 课程训练都必须在
上述执行顺序和 high-SOC 双 baseline gate 通过后另行预注册。

相关背景：[P3 单机 SOC 下限核验](p3_solo_soc_floor_20260922.md)、
[P3 安全接管与能量诊断](p3_safety_energy_recalibration_20260922.md)。
