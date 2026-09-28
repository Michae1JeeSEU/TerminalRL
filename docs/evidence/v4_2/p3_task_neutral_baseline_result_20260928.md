# P3 Task-neutral Baseline Result

状态：主审已核对本地同步的 `full/report.json`；full-cycle paired baseline 为
`COMPLETED`，`12/12` paired records 的 `technical_ok=true`。本页只记录 report 中的精简字段，
server `full.exit` 为 `0`；两臂均为 no Teacher、no update、no heldout，
`collisions=0`、`hard_separation=0`、`NMAC=0`、`SafetyInvariantError=0`、
`uncertified_execution=0`，weights/model unchanged 为 `true`。

## 1. 固定范围

本轮使用注册的 task-neutral 配置、6 个 development cells、source `03c0` 与 candidate
`fit_smoke/terminal_gate_repair.pt`。两臂各 `84` requests；无 heldout、Teacher 或 policy update。
旧 `candidate_retained=false`、full 原始 pending-review 字段和历史 outputs 保持不变。

## 2. Overall report

| metric | source `03c0` | candidate `3fe7...d4ab` |
|---|---:|---:|
| normal | 72 | 72 |
| emergency | 11 | 12 |
| failed_energy | 1 | 0 |
| unresolved | 0 | 0 |
| full-cycle mean wait (s) | 111.47949744714144 | 96.37119998288188 |
| full-cycle p95 wait (s) | 428.211635460218 | 329.0057350762542 |
| full-cycle max wait (s) | 804.6897032491129 | 503.6264147256603 |
| low-SOC wait (s) | 148.3647493634526 | 119.71345607558123 |
| new reward | 1530.4372197172681 | 1623.028105994644 |

Five cells have identical report metrics between the two arms. The only changed cell is
`dense24/873106`:

| metric | source | candidate |
|---|---:|---:|
| normal / emergency / failed_energy | 14 / 9 / 1 | 14 / 10 / 0 |
| mean wait (s) | 238.71178555903364 | 185.8327444341252 |

该 cell 的唯一 source failed-energy 事件为 arrival `4`：`standard`、`base_priority=1`、initial
SOC `.60258461984`，在 `ring_inner` 因 `energy_depleted` 终止，wait 为 `804.6897 s`。candidate
对应 arrival `4` 为 `emergency_landed`，touchdown `659`、final SOC `.1546030`、wait
`503.6264 s`。其它 outcome 变化为 arrival `2/13` 从 emergency 到 normal，以及 arrival `5/18`
从 normal 到 emergency；因此 normal 总数不增加，candidate 减少一个 energy failure 但仍有
`12/84` emergency outcomes。

## 3. Interpretation boundary

The `3fe7...d4ab` candidate is the task-neutral follow-up development repair starting point; `03c0`
remains the comparison source. This decision does not alter the historical `candidate_retained=false`
field, the original full pending-review fields, or historical outputs. The result is not formal training
or global source promotion. Training degradation and stable generalization remain unresolved.

The next registered direction is a bounded-budget on-policy PPO pilot to inspect full-cycle returns,
critic behavior, and actual gate updates. Reward and discount are not to be changed together blindly.
The pilot has not started; `PPO_ALLOWED=false` and the existing formal/heldout flags remain false.

Heartbeat `task-neutral-landing-baseline` has been deleted.
