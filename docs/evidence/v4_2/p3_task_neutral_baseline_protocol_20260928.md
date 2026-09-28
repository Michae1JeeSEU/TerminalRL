# P3 Task-neutral Baseline Protocol

状态：已注册；首轮 check-only 启动失败已保留，`check_only_v2` 为 `CHECK_ONLY_OK exit0`；
smoke 已授权执行中，完整 full-cycle 评估尚未运行。

本协议冻结一轮低层 task-neutral course 的最小实现核对、配对 baseline 和完整周期评估边界。
它不授权 formal training、heldout evaluation、gamma 调整或 PPO promotion；baseline 完成后由
主审另行决定是否注册有界 PPO repair pilot。失败结果保留，不自动重跑。

## 1. 研究范围与语义

当前课程目标是正常降落、landing throughput、完整生命周期等待和等待尾部监测。课程保留随机
到达、SOC 差异、能量约束、安全约束、ring/path feasibility、headway、resource reservation
和 LZ service occupancy。

课程配置固定：

```text
task_priority_enabled = false
low_level_priority_time_weight = 0
Observation V2.1 = 88D
V4.2 action / K=3 / Teacher=false
```

ArrivalProcess 按历史逻辑消耗 base priority 的随机抽样，但 task-neutral request 只存储有效
`base_priority=1`；raw `mission` 原样保留为物理服务/兼容性 metadata，policy mission one-hot
固定为 `STANDARD`。服务时长分布保持：standard `30 s`、cargo `120 s`、emergency `10 s`、
surveillance `60 s`。raw `mission=emergency` 不表示本课程启用了 mission-level emergency
preference；能量应急 takeover 始终保留，并与 mission 标签独立。

每个 arm 的 normal/emergency/failed/unresolved 结果均按完整 ledger 报告。等待指标报告全请求
full-cycle mean、p95、max、low-SOC wait 和 reward；新 task-neutral reward 不与旧 task-priority
课程 reward 直接比较。普通 priority-weighted wait 或每个 cell/子组 wait 不设无退化硬 gate；
等待尾部只用于 starvation 监测。新 source/candidate 不自动 `candidate_retained`、不晋级，
也不表示稳定收益已解决。

## 2. 固定输入与 lineage

输出根目录固定为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_baseline_20260928
```

source 使用原 `03c0`。candidate 只使用已知 normal-advantage `fit_smoke/terminal_gate_repair.pt`：

```text
SHA256 3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab
```

模型权重、actor/proposal/temperature 冻结；两臂允许按注册 seed 进行既定 student stochastic
rollout 采样，但禁止 optimizer、Teacher 和训练更新。两臂使用相同的
`task_priority_enabled=false`、`low_level_priority_time_weight=0` 和其余 V7 实际配置。Arrival
request time、gate、SOC、raw mission、service time 必须与 legacy 同 seed 输入一致。

注册 development seeds 为原已消费集合：

```text
873101, 873102, 873103, 873104, 873105, 873106
```

这些 seed 的 train/旧 validation 均属于 development；本协议不注册新 heldout，也不消费旧
untouched heldout。

## 3. 注册执行顺序与预算

所有命令通过服务器 `ssh lab-server` 执行，输出只写入上述 root；源码仍以本地同步工作树为准。
`ROOT` 表示该固定输出根。

### 3.1 Check-only

一次，不执行 env step：检查配置、权重 lineage/hash、ArrivalProcess request schema/seed 输入和
task-neutral 字段映射。命令为：

```bash
python -m terminal_marl.training.evaluate_p3_task_neutral \
  --output-dir ROOT/check_only --check-only
```

首轮 server wrapper 在 `ROOT/check_only/check_only.log` 预创建了 leaf，导致 fresh output
directory refusal（exit `2`）；未进入程序预检，也未执行 env reset/step。该失败按启动事实保留，
不能记为 check-only 通过。

修正 wrapper 后，`ROOT/check_only_v2` 已 `CHECK_ONLY_OK exit0`。6 个 cells、84 个 requests 的
arrival time、gate、SOC、raw mission 和 service 等物理属性与 legacy 同 seed 一致；task-neutral
request 的 effective base 全为 `1`。changed fields 仅为 `task_priority_enabled: true→false` 和
`low_level_priority_time_weight: 1→0`；`low_level_soc_urgency_time_weight=2` 保持不变。
该 check-only 为 `zero env steps / zero records`，不构成 rollout 或性能结果。

### 3.2 Smoke

仅一次；source 与 candidate 两臂各运行 `flow6`，seed `873101`，最多 `2` steps。Smoke 只核对
初始化、88D/action、task-neutral reward 字段、finite 输出和启动后的技术记录，不构成性能或
qualification 结论。命令为：

```bash
python -m terminal_marl.training.evaluate_p3_task_neutral \
  --output-dir ROOT/smoke --smoke
```

本协议更新时 smoke 已授权并执行中：source/candidate 两臂各为 `flow6`、`2` steps。首个 arrival
为 `10 s`，因此该 smoke 仅用于启动/输出链路核对，不据此验证真实 actor 动作；88D 与 reward
语义由单测覆盖。smoke 尚未形成终态报告。

### 3.3 Full-cycle paired baseline

由主审在 check-only/smoke 技术核对通过后启动。source 与 candidate 两臂各覆盖 6 个已消费
development cells，共 `12` rollouts；每个 arm 总计 `84` requests。另执行既有 ActionTape
exact lifecycle replay，检查 requested/applied provenance、完整 ledger、terminal rows 和 safety
counts。命令为：

```bash
python -m terminal_marl.training.evaluate_p3_task_neutral \
  --output-dir ROOT/full
```

full 不追加模式参数，不扩展 seed、cell、请求数或训练预算；技术异常、安全违规或 ledger
不完整时停止。完整记录的 `failed_energy` 作为性能结果保留，不因 normal 数量或
`failed_energy` 数量筛选早停；终态按实际结果记录，不自动 rerun。

## 4. 技术核对与报告字段

每臂必须核对：模型冻结、配置/权重 hash、ArrivalProcess same-seed request fields、88D schema、
V4.2 requested likelihood/provenance、ActionTape lifecycle replay、完整 outcome ledger、
normal/emergency/failed/unresolved 分账、full-cycle request closure、SafetyFilter/shield/
hard-separation/NMAC/collision 计数和 finite/serialization 输出。能量应急 takeover 继续按原
安全契约核对；不得用 mission label 替代该安全路径。

正式摘要至少包含：

```text
normal
emergency
failed
unresolved
full_cycle_mean_wait
full_cycle_p95_wait
full_cycle_max_wait
low_soc_wait
reward
```

`reward` 与历史 task-priority 课程 reward 只作协议内记录，不能直接跨任务比较。技术检查通过
不等于 candidate retained、qualification、formal/heldout 或 `PPO_ALLOWED`；这些决定由主审按
独立 gate 作出。

## 5. 停止与证据纪律

check-only、smoke、full 的实际 exit/status、输出路径、progress、report、records、hash、请求
分母和 replay 结果必须写入对应 run 目录。失败或缺失产物只记录事实，不改阈值、不换 seed、不重跑、不把历史
candidate_retained 或旧失败洗成当前 PASS。
