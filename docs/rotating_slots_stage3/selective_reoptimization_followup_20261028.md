# 阶段 A 后的规则与接入时间探索

八臂结果与预算异常见 [结果报告](selective_reoptimization_followup_results_20261028.md)。当前为 PRELIMINARY_MIXED_EFFECTS / BUDGET_LIMITED_FINAL_REPLAY_AUDIT_PENDING；八臂均推进至300秒并退出0，但累计预算超限，尚未完成新回放物理审计，不是完整验收，也未进入 B。

本轮为用户明确授权的两个因素 2×2 对照，不是阶段 B、训练或旧 A 结果的改写。起点均为原始 s3r23/seed7324、s3r24/seed7325 的 t0；每臂 300 模拟秒，无额外八快照采集。四臂共用同一新科学源码，每根内依次运行，两个根可在物理 CPU12（中流量）和 CPU14（高流量）并行。

| 臂 ID | 规则 | lead |
|---|---|---:|
| baseline20 | 原规则，新的共同控制运行 | 20s |
| outer20 | outer_priority_v1 | 20s |
| baseline15 | 原规则 | 15s |
| outer15 | outer_priority_v1 | 15s |

outer_priority_v1 只在已有四个 round-robin 队列前加入一个外环种子队列。它读取当前投影节点的 ring，按已知 progress_age 降序、unknown 在后、请求时间及稳定 ID 排序；理由记为 OUTER_RING_QUEUE。共同 mandatory 仍先插入，能源队列的排序不变，既有阻塞插入、扩展和 overflow 全量降级保持。外环飞机不会因此全部变成 mandatory。

15 秒是时间协议干预，会改变投影接入状态、搜索可用窗口和普通任务阻塞节奏；保留 1 秒准备、4 秒交接后，搜索窗口由 15 秒变为 10 秒。它不是仅从原计划空等中扣掉五秒，不能保证计划或实际质量相同。K8/max12、5 秒心跳、候选/CP/hint/soft/watchdog 6/8/3/30/60 秒、尝试32/1024、物理、背景、目标和共同应急均保持。

新入口一次只运行一个明确臂，无 all/自动推进。外层受控启动器负责串行次序、进程 watchdog 和同一剩余预算记账。原 A 输出只读，仅读取其中的原始输入血缘；使用新 source-freeze 字节验证，并在 `arms/<root>/<variant>/resolved_run_binding.json` 和 treatment.json 记录新 binding、实际 variant/lead、CPU 和输入哈希。不会声称是原 A binding。

```bash
cd ~/projects/terminal
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/user/envs/terminal_stage2_v1/lib/python3.12/site-packages
/home/user/envs/terminal_env/bin/python -B scripts/stage3/run_selective_reoptimization_followup.py arm --baseline-output "$HOME/training_outputs/terminal/rs_selective_reoptimization_a_20261027" --output "$HOME/training_outputs/terminal/rs_selective_followup_20261028" --source-freeze artifacts/rs_select_a_20261027/followup_source_freeze.json --root s3r23 --variant outer15 --short-lead 15 --cpu 12 --seconds 300
```

先以独立 smoke 输出目录和 `--seconds 10` 做一次真实服务器烟雾验证，再按批准次序运行四臂；smoke 不替代正式臂、不计为正式重复。源冻结由本地提交后生成，通过 Syncthing 同步；服务器不做 Git 或手工源码上传。

各臂复用既有 restore、延迟 wrapper、完整日志、每 5 秒 checkpoint、终点截尾和 ReplayBundle。额外 followup_observables.json 只从保存结果派生：各边界空中飞机原始 ring 计数、真实 O→M/M→I 启动事件、投影决策节点的 eligible/selected/attempted-union ring 计数；没有精确保存边界帧时明确留空。选择不是执行，扩展并集与最终返回 scope 分列。

计时沿用既有完整 normal-work 记账，包含 pending 部分和 history 一次；按 LOCAL/OVERFLOW_FULL/UNKNOWN 分组，不能把全量降级混称纯 K8 开销。另报告所有普通任务 snapshot 起始间隔，以及仅 COMMITTED 且有 ready_time 的 `effective_time-ready_time`；这是成功任务条件下的最终准备后等待，不对被拒绝、无前缀或 pending 任务补造值。现有 A 中该等待的均值约为中流量8.74s、高流量9.81s，不能单独据此诊断外环停滞原因。

本轮只检验规则优先级与时间协议对实际服务、ring 进展和计算的影响。无获胜前提，不推定15秒质量不劣；没有新增物理动作、学习层或 B 准入。
