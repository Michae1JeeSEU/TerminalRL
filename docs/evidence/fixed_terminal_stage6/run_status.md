# 阶段6运行状态

更新日期：2026-10-14。

- 当前状态：正式episodes、panels、严格summary与研究验收完成。阶段6实现、修复及真实服务器pilot输出已由主审核验；冻结源码提交`f1018d8aaeff03836b13475301087f83ab70a155`已推送。
- 服务器pilot第二次启动四项均exit0；首次launcher工作目录失败证据保留，详见下文。
- 正式episodes为264/264成功、0 failed、exit0；独立panels为15/15成功、exit0，覆盖30states/240rows；严格summary为completed、allow_incomplete=false。
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

## 正式episode完成与panel前置检查

`execution_episodes.json`记录开始时间2026-10-13 23:26:21.767140北京时间，结束时间2026-10-14 00:17:45.013521，duration=3083.246397658251 s。264/264 unique jobs全部completed、0 failed、exit0，20 workers。每job的frozen_input/decisions/events/requests/result五文件齐全、48条decision、result可读且请求守恒；model_hashes_unchanged=true。manifest.files的14个文件逐hash匹配，四模型预期hash匹配。

2026-10-14 00:42:32北京时间服务器只读复核未见stage6进程；28 CPU，load=0/0.111/3.430，可用内存27,431,296 kB，panels目录当时尚不存在。正式panels为15 jobs/30 states/240 rows、workers=1、数值库threads=1，与episodes不重叠且沿用冻结源码。初估T0=50 min，依据dense pilot 135.52 s×15×1.5。

## 正式panel启动与接续

2026-10-14 00:43:49北京时间（UTC 2026-10-13T16:43:49Z）唯一启动正式panels；`/usr/bin/time` PID3435847、dispatcher PID3435848、worker PID3435856，源码`f1018d8aaeff03836b13475301087f83ab70a155`，workers=1、OMP/MKL/OPENBLAS threads均为1。

```bash
/home/user/envs/terminal_env/bin/python -m terminal_marl.scripts.run_fixed_terminal_stage6 --phase panels --models /home/user/training_outputs/terminal/stage5_validation_20261013/models --output /home/user/training_outputs/terminal/stage6_assisted_20261013 --workers 1 --execute
```

初验0/15完成、1 running、14 pending、0 failed；`panels/low-high-6301-standard/`已有非空`frozen_input.json`、`t0/snapshot.json`及5个arm JSON。这是启动产物确认，不代表panel完成。状态与进度写入`execution_panels.json`、`progress_panels.jsonl`；独立日志为`dispatch_logs/panels.launch.json`、`dispatch_logs/panels.stdout-stderr.log`、`dispatch_logs/panels.time.log`，均在同一正式输出根内。

启动时监控记录：Luna已结束常驻监控。当前聊天heartbeat6已由工具确认更新成功、状态ACTIVE，约00:46北京时间设置48分钟后复查，预计01:34北京时间。正常且未变化时保持安静；若仍正常运行，按实际进度重估剩余时间并更新同一heartbeat；完成后核对全部产物与模型hash，严格汇总、主审研究解释、更新文档、提交推送与打包，交付完成后删除heartbeat。不重复启动任一phase。

## 正式panel、汇总与研究验收完成

panels于2026-10-14 00:59:27.788642北京时间结束，duration938.2823586342856 s，15/15成功、exit0、workers=1，覆盖30states/240rows，与episode阶段无重叠。全部期望产物齐全，14个manifest.files引用文件及四模型预期hash均未变。

严格summary实际exit0、elapsed8.60 s、最大RSS152540 kB；日志保存在正式根的`dispatch_logs/summary.*`，输出`summary/summary.json`为completed、allow_incomplete=false，计数264episodes/15generations/30panels/240rows。SHA256为`9c712b31f9792beabb668b3082504c28053f0089dac26da56889f110c6fb75d8`。

主审核验并完成[研究判断](results.md)：B未成立；相对同机制beam无学习增量计算收益，当前配置按A停止扩展，保留实验资产。学习相对full的根数与wall实际下降保留报告；C的大量验证现象不支持扩大value或长期Q路线。阶段1–5历史结果保留，不自动开始下一阶段。

交付版本、ZIP条目与校验结果由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，包链接与最终交付状态见交付回复。交付完成后删除heartbeat6检查任务；删除的实际工具结果由主对话记录，历史ACTIVE记录不代表需要继续实验。此前pilot首次cwd失败四次exit1及全部失败日志继续保留。


本轮文档交付检查：既有summary/package测试首次3 passed、11个tmpdir fixture setup错误（exit1，2.73 s），错误位置为默认`C:/Users/user/AppData/Local/Temp/pytest-of-user`的os.scandir权限；错误输出节选保留在`tmp_stage6_preflight/evidence/pytest_delivery_default_temp_failure.log`。主审授权改用已核对位于workspace且原先不存在的绝对`--basetemp D:/7_codespace/terminal/tmp_stage6_preflight/evidence/pytest_delivery_20261014`后，同两测试14 passed（1.00 s，exit0），完整重试日志`tmp_stage6_preflight/evidence/pytest_delivery_20261014.log`。没有修改生产代码或测试。
