# P3 Controlled-Emergency Formal V5 Restart1 结果

状态：`FAILED`。本页记录 restart1 正式 3-run campaign 的终态事实；不把首轮软件
checkpoint 路径失败与 restart1 的训练结果合并，也不把 restart1 结果写成 qualification。
正式输出根为：

```text
/home/user/training_outputs/terminal/p3_controlled_emergency_formal_v5_restart1_20260925/
```

三个 run 的 manifest 位于各自的 `run0/manifest.json`、`run1/manifest.json`、
`run2/manifest.json`。冻结协议、formal YAML、admission decision、E04 source、场景和
训练 seed 注册表均未改动。`FORMAL_P3_ALLOWED=true` 只表示此前 admission decision；本次
formal 训练终态不通过，`P3_QUALIFIED=false`、`P4_ALLOWED=false`。

## 1. 三个 run 的汇总

下表的 `normal` 是正常完成，`emergency_landed` 是安全认证且能量未耗尽的受控应急着陆，
`failed_energy` 单独计数；`other_failed` 和 `unresolved` 按实际 ledger 字段记录。

| run | accepted episodes | records | requests | normal | emergency_landed | failed_energy | other_failed | unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| run0 | 217 | 218 | 3,594 | 2,983 | 610 | 1 | 0 | 0 |
| run1 | 31 | 32 | 528 | 451 | 76 | 1 | 0 | 0 |
| run2 | 89 | 90 | 1,482 | 1,292 | 189 | 1 | 0 | 0 |
| **合计** | **337** | **340** | **5,604** | **4,726** | **875** | **3** | **0** | **0** |

accepted/rejected minibatches 分别为：run0 `10,931/12`，run1 `1,551/1`，run2
`4,461/4`；含 rejected minibatches 的 records 分别为 `12/1/4`，非
`rollout_hard_gates` 的 `update=false` records 分别为 `0/0/0`。下列 approximate-KL
范围取自 record 的 `update.approx_kl` 字段，不是全 minibatch 范围：run0
`6.95e-6..0.00624532`，run1 `0.00008897..0.00367787`，run2
`8.4e-7..0.00963297`。checkpoint restore 的
`true/missing/false` 分别为 run0 `217/1/0`、run1 `31/1/0`、run2 `89/1/0`；missing
对应失败 rollout 没有保存 checkpoint，不是 restore false。

三个终态失败 seed 均为 dense24：

| seed | run | `zero_failed_energy` | `rollout_hard_gates` | 安全计数 | error |
|---:|---|---|---|---:|---|
| `850218` | run0 | `false` | `false` | 0 | `null` |
| `851032` | run1 | `false` | `false` | 0 | `null` |
| `852090` | run2 | `false` | `false` | 0 | `null` |

## 2. 按场景的 request ledger

列顺序为 `requests / normal / emergency_landed / failed_energy`。

| run | flow6 | flow12 | dense24 |
|---|---|---|---|
| run0 | `330 / 330 / 0 / 0` | `648 / 648 / 0 / 0` | `2,616 / 2,005 / 610 / 1` |
| run1 | `48 / 48 / 0 / 0` | `96 / 96 / 0 / 0` | `384 / 307 / 76 / 1` |
| run2 | `138 / 138 / 0 / 0` | `264 / 264 / 0 / 0` | `1,080 / 890 / 189 / 1` |

## 3. 训练等待统计

训练等待是 record 均值，不是 request 加权均值。priority-weighted wait 为 run0/1/2
分别 `91.827`（`n=218`）、`80.969`（`n=32`）、`84.582`（`n=90`）秒；low-SOC
wait 为 `85.464`（`n=211`）、`78.491`（`n=29`）、`72.071`（`n=88`）秒。

## 4. Development 记录

每组 development 为 `6` 个 episode、`84` 个 requests，`full_resolution=true`，
`heldout=false`，`same_initial_states=true`，硬 gate 通过。等待值是 6 个 episode 的
算术均值；normal/emergency 列为该组汇总。

| development record | normal/emergency | priority-weighted wait (s) | low-SOC wait (s) |
|---|---:|---:|---:|
| baseline | `70/14` | 65.050 | 66.729 |
| run0 after update 40 | `70/14` | 64.915 | 68.126 |
| run0 after update 80 | `70/14` | 73.748 | 77.723 |
| run0 after update 120 | `67/17` | 76.736 | 82.932 |
| run0 after update 160 | `67/17` | 79.941 | 91.883 |
| run0 after update 200 | `64/20` | 90.892 | 78.999 |
| run1 baseline only | `70/14` | 65.050 | 66.729 |
| run2 after update 40 | `70/14` | 65.485 | 70.527 |
| run2 after update 80 | `66/18` | 69.933 | 87.298 |

run1 没有后续 development update-40/80/120/160/200/240 记录；本 campaign 没有
`dev240` 记录。

## 5. 终态与后续边界

主审终态为：小比例安全认证的 `emergency_landed` 仍可作为学习结果，但三 run 各有一条
`failed_energy`，且 development 中出现等待/事件退化，不能推广任何 checkpoint；正式 P3
训练未完成，`P3_QUALIFIED=false`、`P4_ALLOWED=false`。restart1 失败的根因尚未定位，
现有结果不能归因于 SOC 单因素或软件因素。

首轮 checkpoint 保存路径失败及 restart1 的 seed 重复均保持显式记录；已用 seed 不构成新的
独立 qualification 样本。当前不自动恢复、不自动重跑、不追加 seed；heldout 未消费。
尚未启动 dense30、同负荷随机环数量/安全非均匀角度或其他额外 stress 评估。

三个正式进程的终态均为 exit `1`，终态核验时三个 PID 均已不存在；run0 control mtime 为
`2026-09-25 10:57:03+0800`。后续仅保留只读审查边界：检查三个失败 lifecycle 的资源阻塞与
actor gate，并比较训练日志与 development 退化；如需重放，先单独冻结同 seed 的 E04 与失败前
checkpoint paired diagnostic protocol。heartbeat `p3-formal-v5-restart1-monitor` 已删除。
