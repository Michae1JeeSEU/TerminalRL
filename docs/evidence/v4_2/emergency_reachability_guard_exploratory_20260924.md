# Emergency reachability guard exploratory（2026-09-24）

## 范围与证据状态

本页记录当前应急可达性链的多机 exploratory 结果。所有记录都是 rule-only、Teacher-free、
无 optimizer/无 checkpoint promotion 的诊断结果；没有执行 exact replay/full qualification，
不能授权 PPO。`PPO_ALLOWED=false`。

当前诊断链使用：

- opt-in `phaseA_low_obs_v2_3_source_bound_88d`，保持 88D，`neighbor_comm_range_m=320`；
  通信为零延迟、无丢失的理想假设；
- `phase_motion_v2`、warm-start 6 架、固定到达计划
  `[0,0,0,0,0,0,15,30,45,60,75,90]`；
- `warm_start_energy_reachability_check_enabled=true`；
- emergency-priority guard 开启；route-aware energy guard 由 `--guard-on` 控制；
- admission 的 LZ release ETA budget，以及 candidate route energy、pad wait、landing
  reserve 和 configured margin budget；
- 原 SafetyFilter `80/40/10 m` 契约保持不变；fallback future route 尚未作未来安全认证。

旧的 source-bound observation-only 证据独立保留，见
[source-bound observation performance（2026-09-23）](source_bound_observation_performance_20260923.md)。

## 可复现命令参数

运行器为 `terminal_marl.training.diagnose_energy_reachability_guard_smoke`。命令模板如下，
`--guard-on` 的有无分别对应 route 和 old：

```text
python -m terminal_marl.training.diagnose_energy_reachability_guard_smoke \
  --seed <seed> [--guard-on] \
  --initial-soc-min <min> --initial-soc-max <max> \
  --max-steps 1200 --progress-every-steps 100 \
  --output "$HOME/training_outputs/terminal/energy_reachability_guard_exploratory_20260924/<name>.json"
```

本页使用的 cohort 参数为：

| cohort | seeds | initial SOC | route flag |
|---|---|---:|---|
| stress old/route | `840607, 840611, 840612, 840613` | `.365–.55` | old 无 `--guard-on`；route 有 `--guard-on` |
| normal high route | `840620, 840621` | `.55–.90` | 有 `--guard-on` |
| normal near-floor route | `840622, 840623` | `.41–.55` | 有 `--guard-on` |

每个 JSON 同目录有运行进度文件 `<name>.json.progress.json`，但进度文件不是本页的终态
证据。服务器输出根为 `$HOME/training_outputs/terminal`；以下结果哈希从已同步的
`D:/7_codespace/training_outputs/terminal` 镜像读取。

## Stress old/route 配对

每个 seed 的 old 与 route 终态相同。表中 outcome 顺序为
`completed / emergency_landed / upstream diverted_low_soc`。

| seed | old JSON | route JSON | 相同终态 | old checks | route checks |
|---:|---|---|---:|---:|---:|
| 840607 | `stress_old_guard_840607_retry1.json` | `stress_guard_on_840607.json` | `2 / 8 / 2` | 0 | 30 |
| 840611 | `stress_old_840611.json` | `stress_route_840611.json` | `5 / 7 / 0` | 0 | 30 |
| 840612 | `stress_old_840612.json` | `stress_route_840612.json` | `2 / 7 / 3` | 0 | 14 |
| 840613 | `stress_old_840613.json` | `stress_route_840613.json` | `4 / 5 / 3` | 0 | 16 |
| **合计（old）** | 4 records | — | **13 / 27 / 8 = 48** | **0** | — |
| **合计（route）** | — | 4 records | **13 / 27 / 8 = 48** | — | **90** |

8 条记录均为 terminal，`failed_energy=0`，`safety_error=null`；碰撞、NMAC 和 hard-separation
计数均为零。route-aware checks 非零，但没有改变这 4 对 seed 的终态。

### Stress 结果 JSON SHA256

路径前缀均为：
`$HOME/training_outputs/terminal/energy_reachability_guard_exploratory_20260924/`。

| 文件 | SHA256 |
|---|---|
| `stress_old_guard_840607_retry1.json` | `37b5ba62cf276ef51312ab3710a20fe077f6fdc554f1cdc9fece5893d706c1d6` |
| `stress_guard_on_840607.json` | `a2b242e021dc0109a3d2d1cade4e1aa44bd23c826fcece405dd16fff398dcdc2` |
| `stress_old_840611.json` | `c442e3f740db425aad725c956421d79f43e27cebc8f8abf22843aff23c8d0c62` |
| `stress_route_840611.json` | `1964f3c349619f1b269ece54b114f886c780011e7de3abfdf778a1bf748642b7` |
| `stress_old_840612.json` | `eab71f7d2db109fa82096b5458d319f58379f8c71d0ad5c8249ce36443a802a1` |
| `stress_route_840612.json` | `25577c28078cdfb075d8400566e0e2c182b447623e9752c7af261e8a8d454380` |
| `stress_old_840613.json` | `48c2bf872876d30201a9c7c0b506c0b6c977216778cd53a910efe7c1c89cdf71` |
| `stress_route_840613.json` | `7e2f5ec9629bd6196dc472b9317093c05c36d8e3a17a3888b9690309376cfdc7` |

`840607` 的 old 文件名带 `retry1`；本页按实际存在且被读取的 JSON 路径记录，不把它改名
为不存在的 `stress_old_840607.json`。

## SOC 范围探索

| cohort | seeds | 终态汇总 | failed energy | 安全事件 |
|---|---|---|---:|---:|
| normal high `.55–.90` | `840620, 840621` | `23 completed + 1 emergency_landed` | 0 | 0 |
| normal near-floor `.41–.55` | `840622, 840623` | `10 completed + 13 emergency_landed + 1 upstream diverted_low_soc` | 0 | 0 |

两组均为 terminal exploratory records。near-floor 的唯一分流行位于
`phase=external_queue`，属于 upstream `low_soc_slot`；没有 `failed_energy`。

### Normal 结果 JSON SHA256

同一输出根目录下：

| 文件 | SHA256 |
|---|---|
| `normal_high_route_840620.json` | `12b31d6b1322373be29ccd443fbffe451d59c0738a10fef0360962aefba30a1d` |
| `normal_high_route_840621.json` | `9aadc1f030aee2a7b5c671d275d458b322c5e516440f68cb065c071fe1ab8a4b` |
| `normal_nearfloor_route_840622.json` | `104c4cf091f13e5184aab68edef7a764448376b2e2683558174b6dffe1585dd3` |
| `normal_nearfloor_route_840623.json` | `cd02687a59b091660111ebc082901ee913efa6a11dca5420a9423d47b08e8a63` |

## 仿真与 viewer 性能记录

以下性能数字来自对应 exploratory performance report。`840618` 为 `n=12`，`840619` 为
`n=30`；每行数值顺序固定为 `guard-off/default / old / route`。

| seed | n | mean `env.step` wall time (s) | viewer total |
|---:|---:|---:|---:|
| 840618 | 12 | `0.0122423 / 0.0131983 / 0.0098780` | `6.33 / 6.88 / 7.26 fps` |
| 840619 | 30 | `0.9353323 / 1.0619015 / 1.0646215` | `0.219 / 0.206 / 0.192 fps` |

在 dense30 记录中，route 相对 old 为 `+0.26%`，相对 default 为 `+13.8%`；绝对 viewer 实时显示
速度仍很慢。按 seed `840618/840619` 在本地同步镜像中的精确文件名检索，没有读到可对应的
performance JSON/log，因此本段不填写不存在的结果路径或哈希。

## 源码快照绑定与限制

本探索输出 JSON 没有嵌入 `source_sha256` 字段。以下是 2026-09-24 在本地工作树计算的
`S_exp_20260924` post-hoc 源码快照哈希；stress old/route 与 normal high/near-floor 均使用
同一套诊断入口和这组当前源文件语义。该表用于明确批次绑定，不能替代运行时 manifest，不能
声称远端运行具有未记录的字节级 source binding。

| source file | SHA256 (`S_exp_20260924`) |
|---|---|
| `terminal_marl/training/diagnose_energy_reachability_guard_smoke.py` | `ba5c0e2bf2066aa7d400fcb2152a058132e58f6cb8c97a0c69573c42c01ee472` |
| `terminal_marl/core/config.py` | `55a2aef00a336e35bd8a753972a625eabd01c261c87d9603c41fb6c72dbebb99` |
| `terminal_marl/core/admission.py` | `b83ad5e3137aeaaa8237fe123ac620dbf6cbf4cf4c53ce1c8ca3db3b34bb032b` |
| `terminal_marl/core/energy.py` | `b95b25b3ef1fd67ef52b32ecfb8f2d38217d7ab8f2e303a2bd87a741f6c2aabf` |
| `terminal_marl/core/energy_reachability.py` | `c0f850e4a9f2ca9455b2a0a4aa0d4b6c8ac947d778f7bebdded5e37dac327368` |
| `terminal_marl/core/trajectory_generator.py` | `4acb16bf0e8a4229cbb1c237b66400debb5c9cfc5dad29ad1ad98afa98c7ed9f` |
| `terminal_marl/envs/parallel_env.py` | `800a8663ebd1cefd3474ef42164aae584ed7dc3997f6e2874f68208c45b6153a` |
| `terminal_marl/core/observations.py` | `5aeeb7328809f1c3bfe31ea67bf3582ee68b850937daf9612c2b8ea81f24da2e` |
| `terminal_marl/core/topology_neighbors.py` | `3e5efd63075eadab6fdbc5bc41c6cf159eb4c377ab7074fb9cc160e68e15bec0` |
| `terminal_marl/core/runtime_traffic_safety.py` | `8346dee561a1fe34c6fa345b017282100f965f7682f5450ecd3d91c8346025a2` |
| `terminal_marl/core/landing_zone.py` | `d80f936085252251d304587198bb342145e20721adc82567aee1fcdd71dd7333` |
| `terminal_marl/core/uav.py` | `137571791c2053f618ecbed9afae8436da5d763f66e49925caa3a0c9ee8baec8` |

## 当前边界

这些 exploratory 结果说明 route-aware budget 已被执行并产生非零检查计数，但在本 cohort
中没有改变终态；它们没有证明 fallback future route 的未来安全认证，也没有证明共享 policy
已经学会 SOC 优先级。下一步仍需先决定 `12` 机优先课程加 `30` 机离线回放，还是先重构
`30` 机安全层实时性能；两者均需新鲜、单独冻结的协议。当前不启动正式 PPO，且 V2.3 旧
checkpoint 不能直接沿用。
