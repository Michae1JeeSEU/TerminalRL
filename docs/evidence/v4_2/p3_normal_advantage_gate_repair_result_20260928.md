# P3 normal-advantage gate repair result（2026-09-28）

## Scope and credit diagnostic

本轮 credit/训练固定 source `03c0`、train dense24、seed `873103`、有效 steps `102,103,104`；credit 诊断与训练不消费 validation/heldout，未启动 PPO。完整 evaluation 复用既有六个 development cells（含原 validation 三 cell），不涉及 untouched heldout。原 `100..104` credit FAILED 输出保留，不与 V2 数据混写。

credit_v2 `COMPLETED exit0`：`153` runtime branches、`27 logical / 18 actual / 9 reuse`，`9/9` acceptance，`27/27` terminal/full-request-24/ledger/model/RNG/safety checks 通过，failed=`0`。normal deltas 为 `[2,1,1]`、`[3,-3,0]`、`[0,0,0]`；三行 A=`[4/3,0,0]`。

## Fit execution and deviation

原定 fit_smoke 单 logical update 命令实际遗漏 `--smoke`，于 `16:06:04–16:06:07 (+0800)` 在 `fit_smoke` 目录运行 `smoke=false/full120`；真实 metadata 为 `smoke=false`。这是命令执行偏差，不是训练器忽略参数；不将其写作单步预检通过。

主审已明确一次 full budget 已消耗，保留原目录与数据，不重复 fit；全 `120` update 预算内未运行独立单步 fit smoke，`fit/` 未创建，候选位于 `fit_smoke/`。该 fit `COMPLETED exit0`：`120 accepted / 120 logical / 123 optimizer`，final KL mean=`1.0704449096010649e-6`、max=`.0016782581806182861`，freeze/proposal/roundtrip 全为 true，KL anchors=`11592`。candidate SHA256：`3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`。source p=`[.000710682827048,.999929666519165,2.344587062290584e-20]`；candidate p=`[.000841444474645,.999939680099487,2.366394580541294e-20]`；关键 step102 为约 `0.07107%→0.08414%`，仅记录局部 p 微增。

## Evaluation

check-only 为 `CHECK_ONLY_OK exit0`。六 cell evaluation `COMPLETED exit0`、`technical_ok=true`、`candidate_retained=false`，保留 source `03c0`；checkpoint roundtrip exact，proposal/nonadapter 冻结，无更新、Teacher 或 heldout。84 requests：normal `76→78`、emergency `8→6`，failed/unresolved/safety 均为 `0`；PW wait `86.66905761611285→87.62214654891294`，lowSOC wait `89.77926924393867→87.10643757570104`，mean wait `81.38983429949938→78.89363393551653`。

仅 train dense24/873103 变化：normal `21→23`、emergency `3→1`，PW `99.77872368158535→102.66602250742093`，lowSOC `123.60082027309466→110.23666193190647`；其余5 cell不变。by_priority：priority5（16）wait `90.83635980074234→100.83653713035764`、normal/emergency `14/2→13/3`；priority1（39）wait `71.32276779715836→64.84362501084031`、`36/3→37/2`；priority3（29）wait `89.71642690541016→85.68204417499636`、`26/3→28/1`。

## Boundary

开发集存在 normal 收益，但最高优先级子组退化且 aggregate PW wait 保留标准失败；不写成稳定收益或训练退化已解决，不归因于 KL 或单个状态因果。下一方向为终态收益增加优先级保护，尚未预注册或启动。本轮不再 fit/eval；heartbeat `normal-advantage-repair-v2` 已删除，评估完成无需 monitor。`PPO_ALLOWED=false`、`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

## Evidence

- [protocol](p3_normal_advantage_gate_repair_protocol_20260928.md)
- [credit_v2 report](../../../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/credit_v2/report.json)
- [credit_v2 summary](../../../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/credit_v2/summary.json)
- [fit report](../../../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/report.json)
- [candidate checkpoint](../../../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt)
- [evaluation manifest](../../../../training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/evaluation/manifest.json)
