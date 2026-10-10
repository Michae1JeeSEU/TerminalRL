# P3 Task-neutral Critic Refresh Comparison Protocol

日期：2026-09-29。状态：server full `exit0`；control/refreshed 两臂均
`COMPLETED technical_ok=true full_evaluation=true`，PID 已退出，heartbeat 已删除。control `48` training
rollouts 统计为 `960 normal / 192 emergency / 0 failed / 0 unresolved`，refresh 为
`957 normal / 194 emergency / 1 failed / 0 unresolved`，两臂 safety 计数均为 `0`。每臂 evaluation
`96` requests，baseline/control/refresh 为 `80/16`、`81/15`、`78/18`（normal/emergency），全体
evaluation failed/unresolved 均为 `0`；mean/p95/max wait 分别为
`124.5896/281.3078/361.1838`、`130.8607/308.6437/497.1783`、
`137.3319/343.8039/430.7468 s`，low-SOC 为 `119.3941/107.9820/129.9535`，raw reward 为
`1810.5198/1801.5738/1769.6222`。control updates `292+300`、refreshed updates `292+294`，
两臂 rejected 均为 `0`；MC match、freeze/q/roundtrip 均通过，最大 MC 误差 `4.768e-7`。control
checkpoint SHA `3a8aacd7c0e0dfc8cf7828c56a7e4d462f3d1e63d02219e7e3b4e64fee1b1b2c`，refresh final SHA
`370344b6523f250bd6816bbfd6ef51bd065c4aea1352b21d8910adc1275e942c`。refresh failed record 为
batch02 episode24、env `881148`/policy `881248`、arrival `14`/UID `57`、`standard`，SOC
`.553655769→0`，wait `674.2165` s、terminal `892`、`ring_inner`、`energy_depleted`；paired control
同请求为 `emergency_landed`、terminal `516`/touchdown `486`、final SOC `.2464753827`、wait
`226.8147` s。主审拒绝两候选，critic 预测改善不足以解决退化，不继续重复或延长该训练；下一步
仅准备既有轨迹的 WAIT/RELEASE 优势与概率变化诊断，尚未启动或授权新 PPO。本协议比较原结构 legacy critic 的刷新与固定
`a82` control；不使用 context10，不重新拟合旧数据，不把 value MSE 改善作为通过条件，也不自动
放行 PPO、formal 或 heldout。

## 1. 输出、起点与两臂

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_critic_refresh_comparison_20260929
```

两臂 actor 起点同为 `3fe7`。control 臂加载 `a82` critic；refreshed legacy 臂加载 critic-context probe
full 的 legacy critic checkpoint `/home/user/training_outputs/terminal/p3_task_neutral_critic_context_probe_20260929/full/legacy_critic.pt`，SHA256 为
`9c8813ebb49fc5a54f4867c883266373f54b58997a4c4c7d2d6d4011fa8da873`。refreshed 臂只使用既有
`18` episode critic fit 的结果，不用 context10，不重新拟合旧数据。两臂各自使用 fresh Adam，
臂内两批连续更新；development PPO 仅更新 gate residual adapter 与 critic，actor encoder/base gate head、
proposal、temperature 冻结，q 固定为 `3fe7`，安全边界保持不变。

## 2. 固定训练与 evaluation 范围

每臂两批、每批 `24` 条 dense24 training rollouts：env seeds `881101..881148`、policy seeds
`881201..881248`，permutation seeds `881001/881002` 配对。两臂合计 `96` training rollouts、
`2304` requests、steps 上限 `172800`。每臂仅在 batch2 后使用原四个 development evaluation
cells：env seeds `875101..875104`、policy seeds `875201..875204`；两臂合计 `8` evaluation
rollouts、`192` requests、steps 上限 `14400`。

evaluation baseline 为 source `3fe7`：`80 normal / 16 emergency / 0 failed / 0 unresolved`，
mean/p95/max wait `124.589624/281.307806/361.183776 s`，low-SOC `119.394054`，raw reward
`1810.519819`。报告分别比较 baseline、control 和 refreshed 臂的 normal/emergency/failed/
unresolved、mean/p95/max wait、low-SOC、raw reward 与 safety。

## 3. 固定配置与执行顺序

两臂使用 task-neutral、joint-team、`gamma=1`、`lambda=1`、`reward_scale=.01`、`minibatch=128`、
`lr=3e-5`、`epochs=2`、`clip=.1`、`value=.5`、`entropy=.001`、`targetrolloutKL=.01`、
`qcoeff=.1`、`maxKL=.05`、`softtarget=.02`、`gain=400`。q 固定为 `3fe7`，anchors 累计
全部 actor-world；actor 维度与任务中立语义不变。

新 wrapper 为 `run_p3_task_neutral_critic_refresh_comparison.py`。执行顺序为准备 critic-only 合并
warmup → 定向 tests → check-only → 主审通过后唯一 full，按 control、refreshed 两臂顺序运行。
失败结果保留并停止，不重跑、不选择 batch1。已排除当前新 runner 核对 docs 与 training，未发现
`881xxx` 历史引用；预算与 seed 计划保持不变。当前 check-only 已核对两臂各 `48` training/
`4` evaluation rollouts、`2×24` training、`881` paired seeds，runtime 为 `0`；source `3fe7`、
control `a82`、legacy refresh 预检 SHA `32df2985b28699c0a7e7bf3dc5266bfcff8a75db9303a58258dc05595063ccf2`，
non-critic unchanged/config/q fixed 检查均通过。

## 4. 边界与判读

该 run 是一次有限 development comparison；不因 value MSE 下降通过，单次比较不能宣称稳定收益，
表现改善仍需独立重复验证。global `PPO_ALLOWED=false`，formal/heldout/Teacher 语义保持不变；本次唯一
full 仅属已登记的 development 例外。两候选均拒绝，下一步仅准备既有轨迹的 WAIT/RELEASE 优势与概率
变化诊断，尚未启动或授权新 PPO；后续训练仍未授权。
