# P3 Task-neutral Larger-batch Development Protocol

日期：2026-09-29。状态：`COMPLETED exit0`，check-only `CHECK_ONLY_OK exit0`，d449 checkpoint rejected，
child PID `771680` 已退出。本协议是在 episodic learning full 与六轨迹 gradient-dispersion diagnostic 之后进行的
有界 development 测试；不把更大批次选择表述为根因已证，
不自动放行 formal、heldout、P4 或全局 `PPO_ALLOWED`。

## 1. 起点与固定配置

恢复 `3fe7` actor 与 episodic protocol 中的 `a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710`
初始模型。保持 task-neutral、`gamma=1`、`lambda=1`、`reward_scale=.01` 和 joint-team actor
aggregation。相对 episodic learning curve，本协议单独明确联合采样配置变化：每批 `6→24`、
`minibatch 32→128`、总 batch `6→4`；`lr`、`epochs` 和其余参数保持不变。该配置是 development
测试，不是单因素因果消融。参数包括 `lr=3e-5`、`epochs=2`、`minibatch=128`、`clip=.1`、
`value=.5`、`entropy=.001`、`targetrolloutKL=.01`、`anchorcoeff=.1`、`maxKL=.05`、`softtarget=.02`
和 `gain=400`。`learning_curve --larger-batch` 使用固定 profile、constraints 与默认兼容语义。单一
policy 与 Adam 在全 run 中持续，更新只使用当前 batch；q anchors 累计全部 actor-world，不设固定抽样上限；
既有 q、freeze、roundtrip 和安全边界保持不变，不新建框架。

输出根为：

```text
/home/user/training_outputs/terminal/p3_task_neutral_larger_batch_20260929
```

## 2. Training 与 evaluation budget

training 为四批、每批 `24` 条 dense24 rollouts：env seeds `879101..879196`、policy seeds
`879201..879296`、permutation seeds `879001..879004`。合计 `96` training rollouts、`2304`
requests，steps 上限 `172800`。每批使用当前 batch 的数据更新，保持单一 policy/Adam 连续性。

在第 `2/4` 次 update 后，分别用原四个 dense24 evaluation cells 的 env seeds `875101..875104` 与
policy seeds `875201..875204` 做 evaluation；baseline records 复用既有结果，after 为 `8` 条新
rollouts、`192` requests、steps 上限 `14400`。这些 seed 与 records 属于已登记 development 范围，
不作为 heldout qualification。

## 3. 执行顺序与结果记录

实现与定向 tests 已完成：`86` tests passed，`py_compile` 与 `diffcheck` 通过。server check-only 已
`CHECK_ONLY_OK exit0`：`96` training rollouts/`2304` requests/`172800` steps、`8` evaluation rollouts/
`192` requests/`14400` steps，`4×24`、minibatch `128`，source `3fe7+a82`、q fixed，`runtime/update/
Teacher/heldout=0`。主审已准入唯一 fresh `ROOT/full`；full launcher PID `771673`、child PID `771680`，
cwd/cmdline 已核对。首 record 为 env `879101`/policy `879201`：`711` steps、`24` requests、
`20 normal / 4 emergency / 0 failed / 0 unresolved`，terminal/technical/lifecycle/model/weights 正常，
safety 全为 `0`。20:50 UTC snapshot：batch01 已完成 `24` rollouts/`576` requests/`18560` steps，
`469 normal / 107 emergency / 0 failed`、safety `0`；首次 PPO update 为 `290` accepted/`0` rejected，
final rollout KL `.000287400221`、reference KL `.000290478027`，MC match、最大误差 `4.768e-7`，
q fixed/frozen bitwise/proposal/temp/roundtrip 全通过，checkpoint SHA256 为
`554a1a978a152054872e20c3830bc35571135210d4acb895c23e50e29b63860f`。21:05 UTC 历史快照（当时为
`batch02_collect`）：累计 `40` records：`792 normal / 167 emergency / 1 failed / 0 unresolved`，safety `0`。唯一 failed record
为 `full/training/batch_02/training/episode_14/record.json`，env `879138`/policy `879238`：`24` requests、
`19 normal / 4 emergency / 1 failed`、`1169` steps，technical/lifecycle `true`，raw reward `359.8017673`，
mean/p95/max wait `205.11084/652.722486/992.814708 s`。失败 arrival `19`、UID `227`、standard，SOC
`.7073555255→0`，request `120`、terminal `1168`、wait `992.8147076 s`（internal `890.8147076 s`、
external `102 s`），`ring_outer`、`energy_depleted`；最后保存 step `1168` 为 requested/executed WAIT，
shield/intervention `0`，安全计数全为 `0`。无该 cell source 配对，不作唯一因果归因；按 energy-failure
保留规则继续固定 run，不重跑/换 seed/扩预算，最终 evaluation 再判候选。21:20 UTC 历史中期快照（当时已进入
`batch03_collect`）；batch02 已完成 `24` rollouts/`576` requests/`18810` steps，`473 normal / 102
emergency / 1 failed`，`294` accepted/`0` rejected，final KL `.0001340259`，MC match 最大误差
`4.77e-7`，q/freeze/roundtrip 正常。累计 `48` training records：`942 normal / 209 emergency / 1 failed /
0 unresolved`，safety `0`。中期 evaluation 两臂各 `96` requests：before/after 为 `80/16/0/0→82/14/0/0`
（normal/emergency/failed/unresolved），四个 cell normal delta `[+2,-1,0,+1]`；mean/p95/max wait
`124.5896/281.3078/361.1838→131.9878/328.4212/387.0532 s`，low-SOC
`119.3941→109.7865`，raw reward `1810.5198→1814.3240`。主审记录中期指标混合（正常微升、等待恶化），
不选择 batch02、不宣称修复，继续固定 batch03/04 及 final evaluation。21:50 UTC 历史快照（当时已进入
`batch04_collect`，PID `771680` 正常）；batch03 已完成 `24` rollouts/`576` requests/`18170` steps，
`472 normal / 104 emergency / 0 failed`，`284` accepted/`0` rejected，final rollout KL `.0004067797`、
reference KL `.0003421608`、max accepted KL `.0027564398`，MC match 最大误差 `4.77e-7`，
units/q/freeze/roundtrip 全通过，checkpoint SHA256 为
`dab2f83b43a7b878057d6ec15b47f7329e7f17c698778b309e5a52a20bec9a25`。累计 `74` training records：
`1455 normal / 320 emergency / 1 failed / 0 unresolved`，安全与技术异常计数为 `0`；历史 episode14
failed 保留。中期 evaluation 不变，final evaluation 尚未完成，不宣称性能改善。22:00 UTC 历史快照（当时仍为
`batch04_collect`，累计 `3` PPO updates、`4` new evaluation rollouts（其中 `1` 个 batch02 snapshot；final
另 `4` rollouts 才构成第二个 snapshot）；累计 `83` training records：
`1635 normal / 355 emergency / 2 failed / 0 unresolved`，安全与技术异常计数为 `0`。第二个 failed
record 为 `full/training/batch_04/training/episode_10/record.json`，env `879182`/policy `879282`：
`24` requests、`21 normal / 2 emergency / 1 failed`，raw reward `403.6930894`，mean/p95/max/low-SOC
wait `161.3673/468.6840/774.8147/214.0614 s`。失败 arrival `18`、UID `164`、raw mission
`surveillance`，SOC `.5558992080→0`，request `105`、terminal `935`、wait `774.8147076 s`，
`ring_outer`、`energy_depleted`；最后 step `935` 为 requested/applied `WAIT`，proposal `null`、
shield `null`。主审保留样本继续既定 run，不作因果归因，不改训练/安全，final evaluation 尚未完成；
第二个 failed record 及前一条 failure 均保留。22:05 UTC 历史快照（当时仍为 `batch04_collect`），累计 `88` training
records：`1733 normal / 376 emergency / 3 failed / 0 unresolved`，安全与技术异常计数为 `0`。第三个
failed record 为 `full/training/batch_04/training/episode_13/record.json`，env `879185`/policy `879285`：
`19 normal / 4 emergency / 1 failed`，raw reward `385.3507390`，mean/p95/max/low-SOC wait
`150.7375/465.2719/744.0361/170.7179 s`。失败 arrival `21`、UID `92`、`standard`，SOC
`.5699187508→0`，request `150`、terminal `1014`、wait `744.0361466 s`，`ring_middle`、
`energy_depleted`；末 step `1014` 为 requested/executed `WAIT`，applied `[0,0,0]`，shield `null`。
原两条 failure 保留，继续固定 run，不作因果归因，不改训练/安全，final evaluation 尚未完成；第三条 failure
同样保留。full 终态为 `COMPLETED exit0`，child PID `771680` 已退出；四批合计
`96` training rollouts/`2304` requests/`74571` steps、`4` PPO updates，`1166` accepted
（`290+294+284+298`）、`0` rejected，训练统计 `1891 normal / 410 emergency / 3 failed / 0 unresolved`，
安全 `0`，q/freeze/MC/roundtrip 均正常。batch04 为 `19031` steps、`477 normal / 97 emergency /
2 failed`、`298` accepted/`0` rejected，final rollout KL `.0001912632`，checkpoint SHA256 为
`d44990dc7d129573d9145f169df1cbc4f91fa32bc5127b1c58d937260c02a24f`。最终 `8` 条 new evaluation
rollouts 已完成；最后 snapshot 两臂各 `96` requests，before/after 为 `80/16/0→74/22/0`
（normal/emergency/failed），failed/unresolved/safety 均为 `0`；mean/p95/max wait
`124.5896/281.3078/361.1838→132.2342/333.6133/374.5735 s`，low-SOC
`119.3941→119.5818`，raw reward `1810.5198→1733.1170`，normal delta `[-1,-2,-1,-2]`。
主审拒绝 d449 checkpoint，不选择 batch2；更大 batch 未解决退化，继续以 `3fe7` 作为开发起点，
不延长训练。heartbeat `task-neutral-larger-batch-repair` 已删除。
technical/safety 异常或 accepted update 为 `0` 时停止；
若出现 energy failure，保留结果并由主审判读，沿用 runner 语义，不新增 gate。其余失败
结果全部保留，不自动重跑、不换 seed、不选择 batch2、不因中间收益延长预算。最终报告记录整体与分批
的 normal/emergency/failed、mean/p95/max wait、等待尾部、low-SOC、raw reward 和 safety。

该 run 只检验更大独立 batch 下的 development 行为，不增加每场景无退化硬门槛；若结果支持改善，
再由主审决定是否登记独立重复验证，不能直接宣称退化已解决。task-neutral、无 Teacher、无 heldout、
`PPO_ALLOWED=false`、formal/qualification 边界均保持不变。
