# R4 continuation and admitted-aircraft drain

The runner resumes each original completed R4 arm at absolute time 320. It restores the saved world and latency wrapper together, including pending computation, absolute future reservations, reference templates, previous predictions, selector history, and all caches that the original checkpoint actually stores. It compares their serialized representations before advancing. The original provider validation cache was never serialized; its reconstruction and subsequent wall times cannot be claimed identical to an uninterrupted resident process. No cache clearing, scheduler cold initialization, reseeding, forced replanning, or algorithm changes are added.

Traffic and shared FIFO admission continue through the exact settled time 600. At that boundary, the runner keeps the already released manifest prefix, retains the original seed and rate, sets its arrival window to 600, and binds the resulting manifest digest in subsequent checkpoints. The cutoff receipt retains the original digest/window/count and every suppressed request ID. Admission is disabled. Released requests still waiting outside remain in the request table and are listed as excluded external pending requests, without becoming aircraft failures. This intervention is reapplied after every post-cutoff restore because the existing restoration API installs FIFO admission by default.

The cutoff airborne cohort includes INBOUND, holding, transfer, normal final, emergency final, and vertical descent states. All admitted historical aircraft outcomes are reported separately; initial ground fixtures and rerouted external requests are excluded. Drain completion requires no global airborne aircraft and a known terminal outcome for every admitted cohort member. Touchdown resolves an aircraft in LANDED_SERVICE or CLEARED without waiting for pad service. Battery depletion and peak-budget exhaustion retain the existing physical failure archive reasons. Invalid execution, unknown admitted outcomes, and exceptions remain errors, never successful completion. Stale reservation-registry entries are reported separately from physical aircraft state.

The runner advances on five-second stable boundaries, publishes compact progress on each boundary, and saves durable checkpoints approximately every 60 simulated seconds, at restore, at cutoff, and at segment termination. A wall segment cap requests a pause only after a stable boundary; it changes no physical horizon. Explicit `--resume` accepts only PAUSED_CHECKPOINT and restores its full state into a new segment stream copy. No error retry is automatic. Original checkpoints, histories, results, and ledgers remain read-only. Streamed checkpoint sources are copied before reopening; in-memory checkpoint histories remain in memory and are preserved in full.

Each segment writes a complete, unique replay archive including the original history and initializer record provenance. The final replay is passed to the existing continuous energy/separation, resource, PVA/C2, and rejected-proposal audit through the separate `audit` command. Replay export and audit failures retain error or incomplete status; automated output never supplies human acceptance.

## Commands

Commands run on the server from `~/projects/terminal`; Syncthing supplies the locally reviewed source. `<original-freeze>` is the existing e3ae9a96 R4 source freeze and `<new-commit>` is the reviewed continuation commit. Algorithm and configuration hashes are checked against the old freeze; historical documentation changes do not invalidate that algorithm check. The new binding records hashes of this runner, its tests, and directly used runner/audit helpers.

```bash
python scripts/stage3/run_future_commitment_drain.py smoke \
  --source-output "$HOME/training_outputs/terminal/rs_future_commitment_20261031" \
  --output "$HOME/training_outputs/terminal/rs_future_commitment_drain_20261101" \
  --source-freeze <original-freeze> --code-commit <new-commit> \
  --root s3r23 --method FULL_H --cpu 16

python scripts/stage3/run_future_commitment_drain.py run \
  --source-output "$HOME/training_outputs/terminal/rs_future_commitment_20261031" \
  --output "$HOME/training_outputs/terminal/rs_future_commitment_drain_20261101" \
  --source-freeze <original-freeze> --code-commit <new-commit> \
  --root s3r23 --method FULL_H --cpu 16 --segment-wall-s 1800
```

The four fixed root/method/CPU tuples are s3r23/FULL_H/16, s3r23/RULE_LOCAL_K8/17, s3r24/FULL_H/18, and s3r24/RULE_LOCAL_K8/19. Each command operates one explicit arm. Restore-only smoke writes `restore_checks/<root>/<method>` without creating a formal arm or calling physics or the solver. To continue a normally paused formal segment, repeat its run command with `--resume`. Complete or erroneous arms cannot resume through that option.

```bash
python scripts/stage3/run_future_commitment_drain.py audit \
  --output "$HOME/training_outputs/terminal/rs_future_commitment_drain_20261101" \
  --root s3r23 --method FULL_H
python scripts/stage3/run_future_commitment_drain.py report \
  --output "$HOME/training_outputs/terminal/rs_future_commitment_drain_20261101"
```

The caller bounds the separate audit process. `report` reads saved arm results and audits into `analysis_summary.json`; missing or paused arms remain explicit. Server execution status is recorded below separately from the local implementation checks.

## Local checks

`tests/test_future_commitment_drain.py` uses small real checkpoint fixtures for streamed and in-memory histories, pending latency and future absolute reservations, cutoff release/admission behavior, post-cutoff resume, normal/emergency/battery/peak classification, initial-ground exclusion, finite JSON output, segmented replay completion, smoke isolation, and invalid-run rejection. Its synthetic advance fixture verifies runner bookkeeping without invoking the expensive normal optimizer. Actual four-arm restore smoke and physical continuation are separate server executions.

Local verification: `venv/Scripts/python.exe -m pytest tests/test_future_commitment_drain.py -q --basetemp=future_drain_units_20261101a` completed with 13 passed in 2.95 seconds. An earlier `.tmp` basetemp attempt returned WinError 5 before fixture setup; the listed top-level workspace basetemp was writable. The runner's `--help` command completed successfully. No server smoke, physical rollout, or saved-replay safety audit is included in these local test results.

## 服务器执行状态（更新至 2026-11-01 02:03:34，Asia/Shanghai）

代码与测试冻结提交为 `fa1c2647a917b035b443b7b5a52e7278d5b17a92`，已推送。服务器单测结果为 **13 passed、0 skipped，0.61 s**。四个仅恢复 smoke 均 PASS：world、scheduler 和绝对时间完全匹配，物理推进及 solver 调用均为 0，原始来源文件哈希未变。

正式任务于 UTC 2026-10-31 17:48:13–15（北京时间 2026-11-01 01:48:13–15）启动。以下为历史启动记录：只读初始计数来自各自原始 t320 checkpoint；新增 release 数指原始 manifest 在 `(320,600]` 中的请求数。

| Arm | CPU | supervisor PID / timeout PID | t320 airborne | `(320,600]` releases | 启动验证进度 |
|---|---:|---|---:|---:|---|
| s3r23 / FULL_H（MF） | 16 | 461435 / 461437 | 28 | 5 | t340，TRAFFIC |
| s3r23 / RULE_LOCAL_K8（MK） | 17 | 461434 / 461439 | 28 | 5 | t335，TRAFFIC |
| s3r24 / FULL_H（HF） | 18 | 461436 / 461440 | 36 | 18 | t325，TRAFFIC |
| s3r24 / RULE_LOCAL_K8（HK） | 19 | 461490 / 461491 | 37 | 18 | t325，TRAFFIC |

启动验证观测时间为 UTC 17:49:56（北京时间 01:49:56）；四个正式任务的 restore 检查均 PASS。产物根目录为 `/home/user/training_outputs/terminal/rs_future_commitment_drain_20261101`，启动日志为根目录下的 `logs/formal_*`；各 arm 的 evaluation 目录保存 `latest.json`、`progress.json` 和 `segment_0001`。

每段软墙钟上限为 1800 s，外部进程上限为 3600 s。正常 `PAUSED_CHECKPOINT` 必须通过完整状态 `--resume` 接续，不能重新初始化。尚未执行正式 resume 或 replay audit，也没有正式任务完成结论。依据上一轮 R4 运行，至 t600 的初始总墙钟估计 `T0` 约为 1200 s；整个 drain 的耗时尚未知。

第一次进度查询观测于 UTC 17:53:26（北京时间 01:53:26）。四个任务均活跃，CPU 使用率为 99.8–99.9%，未观测到错误；以下计数为当时累计值，剩余时间仅估计到 t600 的 TRAFFIC 阶段终点，不代表整个 drain 完成时间。

| Arm | 仿真时间 | airborne | admitted | released | 至 t600 预计剩余墙钟（s） |
|---|---:|---:|---:|---:|---:|
| MF | 480 | 30 | 45 | 21 | 229.77 |
| MK | 490 | 29 | 46 | 22 | 200.13 |
| HF | 385 | 37 | 50 | 14 | 930.87 |
| HK | 375 | 38 | 50 | 14 | 1238.77 |

上述第一次查询之后的固定查询时间为北京时间 01:58:15，此时间现为历史记录；整个 drain 耗时仍未知。

最初一次后台启动因 Python 换行错误，在任何正式任务开始前失败；修正启动命令一次后才产生上述任务。该次失败日志保留，不计为实验重试。

### 最新观测与接续交接

第三次进度查询观测时间为北京时间 2026-11-01 02:03:34。四个任务均活跃，未观测到错误，均尚未到达 t600；以下为当时累计观测值。

| Arm | 仿真时间 | 正常着陆 | 应急着陆 | failed | airborne | admitted | released | 外部 pending |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MF | 570 | 14 | 3 | 0 | 30 | 47 | 23 | 未在本次交接单列 |
| MK | 595 | 17 | 1 | 0 | 29 | 47 | 23 | 未在本次交接单列 |
| HF | 440 | 13 | 0 | 0 | 40 | 53 | 17 | 未在本次交接单列 |
| HK | 415 | 12 | 0 | 0 | 40 | 52 | 17 | 1 |

主对话已通过 `automation_update` 创建当前线程 heartbeat，ID 为 `600`，状态 ACTIVE，间隔 17 min，并通过工具查看确认。下一次唤醒约为北京时间 02:22；这是复查时间，不是完成时间估计。等待期间不保留持续睡眠的监控 worker。

唤醒后检查约 02:18 触发的软墙钟暂停及其稳定边界、checkpoint 和 replay 导出。只有正常 `PAUSED_CHECKPOINT` 才能使用完整状态 `--resume`，保留 pending、未来预约和既有世界状态，继续原任务直到全部已接纳航空器达到约定终态；遇到 ERROR 或无效执行返回主对话处理。核验全部终态与产物后，再继续已授权的 replay audits、汇总报告及 Git 提交推送。整个任务的完成时间仍未知，本交接不声明四个任务均 COMPLETE。
