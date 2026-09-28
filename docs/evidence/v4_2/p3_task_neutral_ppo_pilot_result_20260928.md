# P3 Task-neutral PPO Pilot Result

日期：2026-09-28。该页记录一次有界 task-neutral development pilot，不构成 formal、heldout、
P4 或全局 PPO 结果。

## 终态与审计

本地同步的 `full/report.json` 记录 `status=COMPLETED`、`technical_ok=true`；服务器
`full.exit=0`，cmdline/cwd 核对正确，进程已终态且监控已停止。3 次 `update()` 调用共记录
`148` accepted optimizer attempts，按 episode 为 `(40, 62, 46)`，`0` rejected；roundtrip、
profile、freeze 和固定 `q` 检查均为 `true`，安全计数为 `0`。

起点为 candidate `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`，source
`03c0` 为对照。pilot checkpoint SHA256 为
`1f7ca8ad1fa4a041b3e62017c56dc47a7c418865613a6241c86fa18763d0fb81`，主审决定不保留；继续以
`3fe7...d4ab` 作为 task-neutral 开发起点。该 pilot 不证明稳定收益已解决，也不证明任务优先级
是唯一原因；不再启动更多训练，`PPO_ALLOWED=false`、formal/heldout/P4 仍为 `false`。

首次裸 Python check-only 为 exit `127` 且未启动；另一次 shell wrapper 失败也未进入 env；绝对
解释器执行的 `check_only_v2` 为 `CHECK_ONLY_OK exit0`。失败产物保留，heartbeat
`task-neutral-ppo-pilot` 已删除。代码为 `1bafa85`，pilot/evaluation 定向测试共 `20` 项
（`11` pilot、`9` neutral evaluation），compile 与 diff check 已通过。

## Final 配对汇总

6 个 development cells、`84` requests；左列为已完成 baseline candidate，右列为本次 pilot。
新 reward 仅在同一 task-neutral profile 内配对阅读。

| 指标 | baseline candidate | pilot |
|---|---:|---:|
| normal | 72 | 70 |
| emergency | 12 | 14 |
| failed | 0 | 0 |
| unresolved | 0 | 0 |
| mean wait (s) | 96.37119998 | 96.62067402 |
| p95 wait (s) | 329.00573508 | 311.51050060 |
| max wait (s) | 503.62641473 | 500.06115227 |
| low-SOC wait (s) | 119.71345608 | 114.55870852 |
| reward | 1623.02810599 | 1612.54083128 |

安全字段计数为 `0`。四个 cells 指标相同；`dense24/873103` 的 normal 为 `22→20`，mean wait
为 `134.58601511→129.53859278`；`dense24/873106` 的 normal 为 `14→14`，mean wait 为
`185.83274443→191.75332588`。

## 第三次更新的只读诊断线索

training episode 的 pre-update 记录为：episode 1 `6 normal / 0 emergency`，episode 2
`12 / 0`，episode 3 `dense24/873103` 为 `22 / 2`、mean wait `125.435666`；该 cell 的 final
为 `20 / 4`。以下 arrival `20` 和 `21` 的 wait/SOC 对照均为固定 baseline candidate→pilot
final，不是 episode3 pre-update→final；两者均为 `completed→emergency_landed`：arrival 20
wait `189.478595→258.838766`、initial SOC `.645768` 不变、final SOC
`.364540→.321837`；arrival 21 wait `93.353424→255.807450`、initial SOC `.681023` 不变、
final SOC `.467520→.341547`。

第三回合 critic 单轨迹 discounted-MC 误差为 mean absolute `45.997741`、mean signed
`45.890410`；前两回合 mean absolute 为 `11.193401/13.964662`。这些数值保留为后续诊断线索，
不证明统计失准或根因。下一步先使用已保存的 training records、GAE 和 ActionTape 定位
normal→emergency 变化及更新信号，检查第三次 dense 更新前后的 credit allocation 与 critic
adaptation，再决定修复；不直接增加训练量、重新加入任务优先级或盲改 reward/gamma。
