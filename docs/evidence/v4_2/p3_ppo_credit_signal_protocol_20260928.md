# P3 PPO credit signal diagnostic protocol（2026-09-28）

## 目的与状态

本文件冻结一次只读 PPO credit signal 诊断，用于对同一条既有 source replay 比较实际的 `team_gae` 折扣信号与不折扣诊断信号。本文是执行协议，不是训练、修复或准入结论；写入时诊断尚未启动。

固定边界如下：

- source 保持 `03c0`，`PPO_ALLOWED=false`；不改变 reward、scheduler、SafetyFilter、PPO 配置或既有判定。
- 只复用 train dense24、seed `873103` 的既有 action tape，执行完整的只读 replay；预期 `702` steps，允许的最大步数为 `1000`，必须保留 `full24` 请求完整性。
- 不采样、不拟合、不更新参数、不启用 Teacher、不消费 heldout；不根据诊断输出调整旧 qualification 判定。
- 输出目录固定为：

```text
/home/user/training_outputs/terminal/p3_ppo_credit_signal_20260928/trace
```

## 语义与证据边界

`mission_type` 是 policy 可见的独立 one-hot 输入。`mission_type=emergency` 会使 `base_priority=5`，但普通任务也可以是 `base_priority=5`；`mission_type`、`base_priority` 与最终 `outcome=emergency_landed` 不能互相替代。当前运行时的应急接管依 SOC/energy 和轨迹可执行性条件，普通调度按 ring/dynamic priority 排序；本诊断不把现有运行时写成已满足严格 mission 优先保证。

既有 `84` 条 evaluation lifecycle rows 缺少 mission 字段，因此不能由 priority 反推历史 mission 分组。本次只读 replay 若补充 mission 字段，只写入新增诊断 sidecar；应以显式字段为准，不改原历史输出，也不追溯改写既有历史分组。

既有 `credit_v2` 的固定局部证据为 train dense24、seed `873103`、UID42/arrival12：step `102` 的三次 RELEASE−WAIT normal 差值为 `[2,1,1]`，step `103` 为 `[3,-3,0]`，step `104` 为 `[0,0,0]`。raw suffix reward `[1.624176, 10.695595, 5.589250]`、`gamma=.99` suffix `[-7.247834, 1.959001, -6.191158]` 和 first proxy `+.100307` 均只属于 step `102` 的三次 repeat。这组固定局部结果只作为输入事实，不外推为所有轨迹或 PPO 退化的根因。

## 冻结诊断量

对同一 tape、同一 policy weights、同一 critic 输出和同一 replay 顺序同时计算：

1. 实际配置的 `team_gae(gamma=.99, lambda=.95)`；
2. 仅用于诊断的 `team_gae(gamma=1.0, lambda=.95)`；
3. Monte Carlo return；
4. critic value/prediction 与 return 的对应记录；
5. actor 使用的 mask、actor-step normalization 信号及每步 reward 分项；
6. `mission_type`、`base_priority`、mission lifecycle/outcome 与 UID/arrival 标识。

诊断重点固定为 steps `102/103/104`、UID42/arrival12，同时保留完整 `full24` replay 的技术记录。当前 PPO 的 reward 是 team reward 求和，实际折扣为 `gamma=.99`、`lambda=.95`；同一 world 的 actors 共享优势输入，actor-step 使用标准化。`gamma=1.0` 只生成对照诊断，不改变实际训练配置；`lambda=.95` 固定不变。报告必须保留上述 team 聚合、共享优势和标准化字段，不将任一局部差值写成全局性能结论。

## 执行顺序

执行顺序固定为：

1. 在本地用 fixture/tests 验证脚本输入、字段、`team_gae` 对照计算、序列化和输出 schema；
2. 通过 Syncthing 确认服务器项目文件已同步；不使用手工 `scp`、`rsync` 或打包覆盖源码；
3. 只运行一次服务器诊断命令；启动后检查 PID、日志和预期 trace 产物；
4. 诊断若失败，保留失败输出和原始日志，不自动重跑性能诊断；
5. 仅记录直接观察到的技术检查和输出，不把诊断状态写成训练完成或资格通过。

CLI 固定为：

```text
python -m terminal_marl.training.diagnose_p3_ppo_credit_signal --output-dir /home/user/training_outputs/terminal/p3_ppo_credit_signal_20260928/trace
```

服务器输出只写入 `/home/user/training_outputs/terminal`，本地通过既有 Syncthing 链路读取。服务器运行前后均不得写入 `~/projects/terminal` 的 runs 或训练产物目录。

## 技术一致性检查

诊断必须至少记录以下检查的直接结果：

- source weights/checkpoint digest 与 source `03c0` 一致；
- action tape 的 seed、scene、step 数、UID/arrival signature 和执行 action 一致；
- replay 使用完整 `full24` 请求集合，expected `702` steps，未超过 `max1000`；
- reward 分项、team reward、`gamma=.99/1.0` 两套 GAE、Monte Carlo return、critic 输出的行数和索引可对齐；
- 关键 steps `102/103/104`、UID42/arrival12 的 actor mask、归一化信号、reward 分项和 mission lifecycle 字段存在；
- 终态 ledger、全请求技术检查、安全/能量计数和序列化一致。

缺少 mission 字段时必须明确记录为 unavailable，不得用 `base_priority` 代替。任何输入、索引、字段或序列化检查失败均保留原始输出并停止本次诊断。

## 解释边界

本协议不预设折扣时域、critic、归因或其他方向的结论；主审在取得 trace 后另行决定最小后续验证。任何局部三 repeat 结果、`gamma=1.0` 对照、critic 偏差或 actor 信号差异均不得单独写成 PPO 退化根因。普通 `priority wait` 与 `low-SOC wait` 仍是软优化/分析指标；它们不撤销 energy/admission、route-energy/slack、`min_landing_soc` 等硬边界，也不撤销既有危急 SOC 应急接管或 failed/safety 要求。

本协议不新增普通 priority 硬门槛，不修改调度器，不启动 PPO/formal/heldout，不改变 source `03c0`，也不构成 `PPO_ALLOWED`、`P3_QUALIFIED` 或 `P4_ALLOWED` 的通过依据。

## 实际执行记录（2026-09-28）

本协议按代码提交 `77a549bbc12c2dfdc7abb12f9361e325eb72b2e4` 执行；定向测试为 `32 passed`（`9` diagnostic、`11` lifecycle、`12` evaluation）。首次启动预检因远端 cwd 遗漏产生 `ModuleNotFoundError`，发生在启动前，未执行 reset/step；修正 cwd 后只执行一次 replay，不计预检为第二次实验。

唯一 replay 为 `COMPLETED exit0`，输出根为：

```text
/home/user/training_outputs/terminal/p3_ppo_credit_signal_20260928/trace
```

直接观察到：`702/702` exact steps、`24/24` ledger/all-requests、`512` actor steps、terminal step `701`；source weights unchanged、RNG restored；collision、hard-separation、NMAC 均为 `0`。`report.scope` 明确无 training、optimizer、sampling、Teacher、heldout。诊断在 worker 监控窗口内完成，无 active run、无 heartbeat，且未创建 heartbeat。

source fixed tape raw reward 为 `448.98876382803303`，reward components 对齐。以 `dt=1` 计，`gamma=.99` 半衰期为 `68.96756 s`。关键 step 的 source requested gate、critic value、单轨迹 MC return（`gamma=.99`）、GAE raw、actor-step normalized GAE，以及 fixed critic 仅切换 `gamma=1.0` 的 normalized 对照如下：

| step | UID / arrival | source requested | critic V | MC `gamma=.99` | GAE raw `gamma=.99` | normalized `gamma=.99` | normalized fixed-critic `gamma=1.0` |
|---:|---|---:|---:|---:|---:|---:|---:|
| 102 | 42 / 12 | 0 | 14.9018993 | 69.9018934 | 2.24120283 | -0.590941548 | -0.516637743 |
| 103 | 42 / 12 | 1 | 14.9006491 | 70.7249064 | 2.66584110 | -0.548706889 | -0.467637658 |
| 104 | 42 / 12 | 0 | 14.9006081 | 71.7584548 | 3.32892299 | -0.482756525 | -0.396039546 |

`lambda=.95` 固定。负的 normalized 局部代理信号不能宣称 step102 必然鼓励 WAIT；它只表示该样本局部代理项倾向降低已采样 WAIT 概率，完整 batch 的其他梯度与 KL 仍会影响最终更新。actor-step 标准化不据此写成 bug；fixed critic 下仅切换 gamma 未改变三步信号符号。V 与单轨迹 MC 的差异不等于已证明所有 V7 退化根因或 critic 统计失准。

mission pairing 只覆盖 dense24、seed `873103` 的 `24` requests；source 新 trace 记录 mission，candidate 仅在内存分析副本中按 source exact replay 的 immutable request 属性、`seed+arrival_id` 做 join，结果列于本文，不写候选 JSON sidecar，也不改历史 outputs。`request_time` 与 `base_priority` 完全一致，SOC 重建容差为 `1e-12`，实际 `maxdiff=3.885780586188048e-15`；不能把全 `84` rows 写成已完成 mission 核对。

| mission | source（count / normal / emergency_landed / failed / mean wait） | candidate（count / normal / emergency_landed / failed / mean wait） |
|---|---|---|
| cargo | `6 / 5 / 1 / 0 / 116.960214809` | `6 / 6 / 0 / 0 / 112.718649405` |
| emergency | `1 / 1 / 0 / 0 / 168.329161346` | `1 / 1 / 0 / 0 / 179.505236768` |
| standard | `13 / 11 / 2 / 0 / 128.954875674` | `13 / 12 / 1 / 0 / 100.700289030` |
| surveillance | `4 / 4 / 0 / 0 / 76.481889822` | `4 / 4 / 0 / 0 / 119.457418021` |

变化为：arrival5 `STANDARD` `normal→emergency_landed`；arrival12 `CARGO`、arrival13/21 `STANDARD` `emergency_landed→normal`。唯一 emergency mission 为 arrival20，两边均为 `normal`，wait `+11.176075422`，touchdown `545→567`；没有完成同时争用同一资源的冲突层检查，因此不据此宣称发生普通任务抢占紧急任务，也不据此宣称已保证紧急任务优先。

candidate reward 补齐为：six-cell 总体 source `1606.1318631737224` → candidate `1621.4070350399447`（`+15.2751718662`），normal `76→78`、emergency `8→6`、priority-weighted wait `+0.95309 s`、low-SOC wait `−2.67283 s`；dense24/873103 单 cell source `448.9887638280331` → candidate `464.2639356942554`，normal `21→23`、emergency `3→1`。按当前 soft preference 语义，这是待进一步验证的 normal 收益信号；旧 `candidate_retained=false` 历史保持，不事后改旧 manifest，不自动晋级或切换 source。source fixed-tape MC 与此前 counterfactual 三-repeat 结果来自不同 future trajectories，不能混作同一数据。

## Evidence links

- [new trace report](../../../../training_outputs/terminal/p3_ppo_credit_signal_20260928/trace/report.json)
- [prior normal-advantage evaluation record](../../../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/evaluation/records/dense24_873103.json)
- [prior reward-order summary](../../../../training_outputs/terminal/p3_terminal_credit_20260928/reward_order/summary_corrected.json)

主审结论：本次只读诊断未改变实际 `gamma=.99`、reward、scheduler 或 source，未启动新的 PPO；`PPO_ALLOWED=false`。后续回到完整周期 reward/discount objective 与 critic/team-credit 一致性验证，不把本次固定 replay 写成对所有 V7 退化的根因证明。
