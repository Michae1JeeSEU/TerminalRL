# P3 Task-neutral Critic Repair Result

日期：2026-09-28。本页记录 sampling repeats、critic warmup 与其 pilot check 的开发证据，
不构成 formal、heldout、P4 或全局 PPO 结果。

## Sampling repeats

代码提交为 `12f7873`，定向测试 `37` 项；check-only 为 `CHECK_ONLY_OK`、`16`-matrix、
`0` records。full report 为 `COMPLETED`、`technical_ok=true`，共 `16` rollouts、两臂各
`2` 个 dense env seeds（`873103`、`873106`）和 `4` 个 policy sampling seeds
（`874301..874304`），共 `192` requests。只分离 torch policy sampling seed，env 物理 seed、
arrival、SOC 和 service 不变，未更新模型。

| 指标 | before (`3fe7`) | after (`1f7`) |
|---|---:|---:|
| normal | 140 | 136 |
| emergency | 51 | 56 |
| failed | 1 | 0 |
| mean wait (s) | 160.633 | 172.684 |
| p95 wait (s) | 369.113 | 445.294 |
| low-SOC wait (s) | 187.337 | 197.268 |
| reward | 3219.130 | 3166.920 |
| safety count | 0 | 0 |

8 个 paired normal deltas 为 `[-2, 0, 1, 0, 3, -3, -3, 0]`，其中 `7/8` 个 paired mean wait
上升。两既有 env 的 action repeats 支持继续修复，但不能归因仅一次采样，也不是独立环境
泛化或显著性证明。该结果不证明 root cause 已定位。

## Critic warmup

代码提交为 `92baea9`，定向测试 `37` 项、check-only 通过；使用 `6` 个既有 neutral baseline
tapes、`4775` steps。固定 `3fe7` actor，critic target 使用完整终态 MC、`gamma=.99`，不使用
GAE bootstrap；只优化 `critic_encoder`/`critic_head`，fresh Adam、`lr=3e-4`、`20` epochs、
minibatch `64`、seed `874501`、`gradclip=1`，actor/proposal/temperature 逐位冻结。

第一次启动 wrapper 返回 launcher `262040` 后立即消失，未创建 warmup/log/pid/exit；第二次尝试
在外层 compile (`launcher_code`) 暴露 `SyntaxError: unterminated string literal`，记录为
`pid_path.write_text` 拼接换行转义错误，未 `Popen`、无 leaf/log/fit；两次均不计作实验重跑。
随后唯一真实 runner PID `397550` 完成 `exit0`：`6` exact、`84` requests、`4775` steps、
`20` epochs。MSE 为 `733.8008877569→104.6766084265`，MAE 为
`20.2631550239→5.6295053403`，EV 为 `.25466145→.76812814`；actor/noncritic bitwise
unchanged 为 `true`，无 PPO、sampling 或 Teacher。`warmup.pt` SHA256 为
`b136a57a8a977b60d5ae71fe3925c8cf133433c0c5fc4e6bef9872295353844d`。这些指标只说明数据内
拟合误差变化，不代表泛化或问题已解决。

## Warmup pilot paired evaluation

`pilot_check_only` report 为 `CHECK_ONLY_OK`；随后唯一 `pilot/` report 为 `COMPLETED`、
`technical_ok=true`。6-cell、84-request 配对为 normal `72→74`、emergency `12→10`、
failed/unresolved `0/0`、安全计数 `0`；mean wait `96.3711999828819→94.8296761036019 s`，
p95 `329.005735076254→349.561162932274 s`，max `503.62641472566→492.246915376754 s`，
low-SOC wait `119.713456075581→110.00386056458 s`，reward `1623.02810599464→1637.18852268814`。
checkpoint SHA256 为
`fa0e4218893f51934772e9629e507b86d51024d9f7b4664e6831e31ac135c289`。

主审判定该 checkpoint 为开发 candidate：单次 6-cell 有收益与尾部 tradeoff，不称稳定修复；
已有 6-cell 已被 critic warmup 使用，`validation` split 名称为历史保留，本轮不作独立验证。
下一项仅批准同两个 env × 四个 policy seeds 的 `8` 个新 after rollouts，并复用既有 `8` 个 before
rollouts；新 repeats ROOT 待主审确定，须先完成实现、测试和 check-only，不自动启动。全局
`PPO_ALLOWED`、formal 和 heldout 边界不变。包装修复要求为 `write_text(str(child.pid))` 与
`str(returncode)` 不带换行，并在启动前完成本地与远端 compile 预检。
