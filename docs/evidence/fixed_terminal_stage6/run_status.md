# 阶段6运行状态

更新日期：2026-10-13。

- 当前状态：正式episodes运行中。阶段6实现、修复及真实服务器pilot输出已由主审核验；冻结源码提交`f1018d8aaeff03836b13475301087f83ab70a155`已推送。
- 服务器pilot第二次启动四项均exit0；首次launcher工作目录失败证据保留，详见下文。
- 正式264episode已一次启动；独立panel15 jobs尚未启动。正式矩阵尚未完成，严格summary与结果包尚未生成。
- 复用阶段5四冻结模型，不重新训练；学习排序仅使用MLP rank/gradient boost。
- 方法、场景矩阵、停止证书与接受口径已写入[方法](../../arrival_restructure/stage6_learning_assisted_scheduling.md)和[协议](../../experiments/fixed_terminal_stage6_protocol.md)。
- 已完成三批范围不同的本地检查：核心33项、评价/dispatcher13项、summary/package13项均通过；这不是一次完整联合检查。真实冻结模型fixture汇总为1episode/1generation/2panels/16rows，状态fixture_or_pilot_incomplete，不能作为正式效果证据。
- 打包修复按阶段5manifest.files纳入全部被引用的冻结模型及辅助文件；不纳入未被引用的阶段5原始结果。修复后单独运行`venv/Scripts/python.exe -m pytest tests/test_fixed_terminal_stage6_package.py -q`，6项通过（0.62 s、exit0），包含从ZIP提取全部manifest引用文件后重新通过frozen_models校验；该批与此前检查范围重叠。

## 服务器预检与pilot

服务器28 CPU。2026-10-13 23:26:07北京时间观测load为0.389/0.866/0.433，可用内存27,426,180 kB。复用阶段5冻结模型目录`/home/user/training_outputs/terminal/stage5_validation_20261013/models`。

pilot attempt1的四个任务因launcher未进入项目目录而报`ModuleNotFoundError`，全部exit1，未进行仿真。8个`logs/*.stdout-stderr.log`与`logs/*.time.txt`原始失败记录保留。修正launcher显式cwd后，attempt2于2026-10-13 23:21:39北京时间四并发启动：

| pilot | elapsed seconds | maximum RSS kB | exit |
|---|---:|---:|---:|
| beam_adaptive | 23.70 | 27,004 | 0 |
| rank_adaptive | 148.05 | 623,092 | 0 |
| full_reference | 152.89 | 29,416 | 0 |
| initial_dense panel | 135.52 | 717,644 | 0 |

三个episode各48次decision、24请求，N/E/F/C=18/0/0/6且守恒；panel在t0/t60共16行。四模型hash不变。pilot仅验证流程和提供估时，不作方法效果结论。

## 正式episode启动记录

2026-10-13 23:26:21北京时间启动，cwd为`/home/user/projects/terminal`，源码`f1018d8aaeff03836b13475301087f83ab70a155`。`/usr/bin/time` PID3431309，dispatcher PID3431310；workers=20，OMP/MKL/OPENBLAS threads均为1。

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.scripts.run_fixed_terminal_stage6 --phase episodes --models /home/user/training_outputs/terminal/stage5_validation_20261013/models --output /home/user/training_outputs/terminal/stage6_assisted_20261013 --workers 20 --execute
```

服务器输出根`/home/user/training_outputs/terminal/stage6_assisted_20261013`，Syncthing本地路径`D:\7_codespace\training_outputs\terminal\stage6_assisted_20261013`。根下`execution_episodes.json`与`progress_episodes.jsonl`记录执行状态，`dispatch_logs/launch.json`、`dispatch_logs/episodes.stdout-stderr.log`、`dispatch_logs/episodes.time.log`保留启动、输出与资源时间证据。

启动初验为0/264完成、20 running、244 pending、0 failed，并已见非空decisions/events/frozen_input中间产物；这是启动观测，不是最新进度或成功完成证明。初始T0=75 min：pilot最大约153 s×264/20≈33.7 min，再留约2.2倍高负载、并发与尾部余量，初始预计完成时间2026-10-14 00:41北京时间。

短期监控worker已结束。当前聊天heartbeat已由工具确认创建成功（automationId=6、状态ACTIVE），约2026-10-13 23:30北京时间建立71分钟间隔，预计10月14日00:41北京时间自动复查，之后按实际剩余时间重估。正常且无变化时保持安静；episode完成后自动接续panels、严格summary及交付，交付完成后删除该heartbeat。后续依次核对264任务完成与模型hash，运行不与episodes重叠的单worker panel阶段（15jobs/30states/240rows），再严格汇总、主审研究判断、更新文档、提交推送与打包。

阶段1–5历史结果保留；阶段6A/B/C判断待正式证据。
