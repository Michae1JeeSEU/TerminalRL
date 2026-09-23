# SOC 优先级共享策略配对诊断 V1（2026-09-23）

## 结论边界

本轮是 phase_motion_v2 能耗模型和固定应急迫降语义下，冻结 original K3 与 nominal rule 的
两种控制器配对诊断。它检验共享 policy 是否在相同交通与 SOC 条件下表现出可观察的 SOC
优先级差异，同时记录正常降落、应急降落、上游低 SOC 分流和能量耗尽。该诊断不是
qualification，也没有训练、参数更新或 checkpoint promotion 授权。

四条 formal records 均完成终态、请求记账、action-tape exact replay、K=3 shortlist、
Teacher-free、无 labels/updates 与安全检查；安全事件总数为 `0`。但四条记录合计出现
`3` 条 `FAILED_ENERGY`，因此预注册的 `zero_failed_energy` gate 失败，aggregate
`PASS=false`，`PPO_ALLOWED=false`。四条 formal records 中的 `39` 次成功 touchdown
包含 `32` 次 `EMERGENCY_LANDED`，因此这组 stress diagnostic 不能单独证明 shared policy
已经按 SOC 稳健排序。

## 冻结协议与 provenance

- 协议：[`emergency_soc_priority_paired_v1.yaml`](../../../terminal_marl/configs/low_v4_2/emergency_soc_priority_paired_v1.yaml)，SHA256
  `6504740d11dc7ceb005155e57ed26ff5edfdf392bf7056bd9d8a5f1ef6893842`。
- runner：[`diagnose_emergency_soc_priority_paired_v1.py`](../../../terminal_marl/training/diagnose_emergency_soc_priority_paired_v1.py)，SHA256
  `2f47edf704ae21331e86bd9ee67555198bc3186644f8fe2a91b876b35448bd47`。
- 定向测试：[`test_emergency_soc_priority_paired_v1.py`](../../../tests/test_emergency_soc_priority_paired_v1.py)。
- frozen original K3 checkpoint：
  `p2_gate_repair_v3_anchored_20260919/gate_repair_v3_qualification_candidate.pt`，SHA256
  `4b8be66c100aaaeb5b5699dcce68a96dbe1237964e4c308e7b23b95c62fb8c8c`。
- formal seeds：`840301`、`840302`；每个 seed 同时运行 rule 与 original K3，未重用
  P3 high-SOC/low-SOC formal seeds。
- 固定场景：12 个请求、warm start 6 架、请求时刻
  `[0,0,0,0,0,0,15,30,45,60,75,90]`、初始 SOC `Uniform(.35,.55)`、1200 s
  horizon、nearest-available LZ、downstream priority、final capacity `3`。
- 能耗/应急：`phase_motion_v2`、`emergency_trigger_soc=.40`、margin `15 s`；应急路线
  仍通过原 `80/40/10 m` SafetyFilter，不放宽安全约束。

Formal manifest SHA256 为
`fb8993507a408940e33406425b25fccb1376e294b231faa924c796ad5427f308`。
服务器 manifest 位于
`/home/user/training_outputs/terminal/emergency_soc_priority_paired_v1_20260923/manifest.json`，
本地 Syncthing 镜像位于
`D:/7_codespace/training_outputs/terminal/emergency_soc_priority_paired_v1_20260923/manifest.json`。

## Formal 结果

`all-request horizon-penalized delay` 的定义是：已成功请求取 touchdown 时刻减 request
时刻；未成功、上游分流或能量失败的请求按 `1200 - request_time` 计入。它保留完整请求
分母，不以 completed-only wait 代替总体结果。

| controller | seed | normal completed | emergency landed | upstream low-SOC diverted | failed energy | all-request delay (s) |
|---|---:|---:|---:|---:|---:|---:|
| rule | 840301 | 2 | 7 | 2 | 1 | 426.25 |
| original K3 | 840301 | 3 | 7 | 2 | 0 | 332.33 |
| rule | 840302 | 1 | 9 | 1 | 1 | 326.17 |
| original K3 | 840302 | 1 | 9 | 1 | 1 | 328.92 |

四条记录合计为 `7` normal、`32` emergency、`6` upstream low-SOC diverted 和 `3`
failed energy。成功 touchdown 共 `39` 次，即 `7` 次 normal + `32` 次 emergency；其中
emergency 占 `32/39`，自动应急占主导，因此这组两种子 stress diagnostic 本身不足以建立
policy 的 SOC-priority RELEASE ordering。paired `original K3 - rule` 的 all-request delay 为 seed `840301`
的 `-93.9167 s`、seed `840302` 的 `+2.75 s`；两颗种子只有描述性证据，不能据此宣称
共享策略已闭合 SOC 优先级，也不能据此归因失败原因。

## Hard gates 与 smoke

通过的 formal gates：

- `all_records_terminal=true`；
- `exact_request_accounting=true`；
- `exact_action_replay=true`；
- `zero_safety_events=true`，包括 `SafetyInvariantError`、collision、NMAC、hard
  separation 和 uncertified execution 均为 `0`；
- `no_teacher=true`、`no_labels=true`、`no_updates=true`；
- `runtime_shortlist_k3=true`；
- `zero_in_terminal_low_soc_diversion=true`。

失败的 formal gate 只有 `zero_failed_energy=false`。记录中的失败终态必须作为后续
action-tape 归因的输入保留，当前不据此判断是应急可用性、触发阈值、能量可达性还是其他
执行条件造成。

预注册 smoke 使用 seed `840300`、每条 tape 12 steps。两条 tape 均可 exact replay，安全
计数为零；由于短步数上限，smoke 的整体完成 gate 为预期的 `PASS=false`，不消耗 formal
seeds，也不改变 formal 结论。

## Immutable replay attribution addendum

只读归因文件为
`D:/7_codespace/training_outputs/terminal/emergency_soc_priority_paired_v1_20260923/failure_trace_evidence.json`，
SHA256 为 `a17820aa59bbbece91cc96d066981c20e4b42ba57e2edc317be23bec24da1a57`；其标记为
`replay_only=true`、`no_new_seeds=true`。三个失败目标的回放均 exact：rule seed `840301`
arrival `2` 为 `609/609` steps，rule seed `840302` arrival `11` 为 `742/742` steps，
original K3 seed `840302` arrival `9` 为 `751/751` steps。

三个目标第一次观测到 `SOC <= .40` 的时刻分别为 `t=1`、`t=142`、`t=143`，当时均有
committed plan。随后记录到的首个低 SOC 且无 committed plan 状态为：

| case | arrival | time / SOC | raw availability and next attempt |
|---|---:|---:|---|
| rule `840301` | 2 | `t=70 / .35922` | `2` 个可用 LZ；候选应急能耗 `501.671`、剩余能量 `538.833`；下一次 `t=71` 的全局 telemetry 增量为 `2` 次 safety block、`0` 次 commit |
| rule `840302` | 11 | `t=148 / .39562` | `0` 个可用 LZ；下一次 `t=149` 记录 `blocked_lz`、`0` 次 commit |
| original K3 `840302` | 9 | `t=181 / .37948` | `0` 个可用 LZ；下一次 `t=182` 记录 `blocked_lz`、`0` 次 commit |

上述 safety/block 计数来自全局 telemetry，不能在一般情况下直接归属于目标 UAV；本表只
记录与目标同一时刻的回放状态和未发生 commit 的事实。它不证明 SafetyFilter 过严，也不
证明 LZ 占用是唯一原因。original K3 的目标 tape 在接近零 SOC 时仍出现 WAIT/RELEASE
交替，因而这三条记录不能作为共享策略已经具备稳健 SOC 优先级的证明。

Witness replay 进一步记录：

Witness 文件为
`D:/7_codespace/training_outputs/terminal/emergency_soc_priority_paired_v1_20260923/failure_witness_evidence.json`，
SHA256 为 `1a93a063db4982946532c26516629e98ff0224c04bcfee51c112290642a03844`。

- rule `840301` 在 `t=70` 的两个 free pads 为 LZ `5/1`；两条应急候选的 energy
  margin 分别为 `+37.1625/+20.6904`。下一步 target safety decision 为
  `predicted_separation`，两个候选均 rejected，commit 为 `0`。
- rule `840302` 在 `t=148` 的 LZ 状态为 occupied `0/3/5`、reserved `1/2/4`，
  下一步为 `blocked_lz`。
- original K3 `840302` 在 `t=181` 的 LZ 状态为 occupied `0/2/5`、reserved `1/3/4`，
  下一步为 `blocked_lz`。
- 三条 target tape 的 exact replay、reset/base/config/recipe 匹配以及 pre/post state
  hashes 均匹配。当前 `_try_emergency_landing` 只考虑无 committed plan 的 ring holder；
  三个目标第一次 `SOC <= .40` 都发生在 committed plan 中。这是代码与回放观察，不能
  推出提前抢占长计划必然安全。

## 下一步与授权边界

保留上述 immutable replay attribution 和原始失败 manifest/tape；下一步应独立预注册
fresh-seed 的最小 repair 诊断，分别覆盖：(a) low-SOC commitment gate，或有证书的长计划
安全 preemption；(b) emergency pad/corridor access capacity。repair 保持现有 SafetyFilter
和安全阈值，先评估最小修复，完成前不启动 PPO；`PPO_ALLOWED=false`。
