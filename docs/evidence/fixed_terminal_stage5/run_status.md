# 阶段5运行状态：正式test完成，continuation修复与收尾进行中

更新日期：2026-10-13。源提交`4811abb805ab03997f5ce4186c6b2610f3c6b3b3`已推送；阶段1–5合并检查为101 passed in 17.21 s、exit0，日志`tmp_stage5_preflight/combined_tests.log`。当前四模型正式训练及冻结已完成，评价流程pilot已通过；新530x正式360个jobs已全部完成，0failed、exit0、四模型hash不变；独立旧4303 continuation首轮3成功6失败，旧artifact比较修复正在应用与验证，第二轮尚未运行。服务器完整产物核验和最终研究汇总仍在进行，阶段5尚未完成。本文是运行状态，不是阶段5效果报告。

## 正式训练与冻结

服务器根目录`/home/user/training_outputs/terminal/stage5_validation_20261013`，同步本地为`D:\7_codespace\training_outputs\terminal\stage5_validation_20261013`。正式产物在其`models/`，字段依据`models/manifest.json`、`models/training.json`及`training_attempt2.record.txt`/`.time.log`核对。

固定train108/val36 contexts，model seed5401，三个神经模型各300epoch、gradient boosting 100树，CPU threads1。唯一实际训练于2026-10-13 19:47:56至19:48:18（UTC+8）完成，PID3412631，exit0，记录墙钟22.092 s，maxRSS913020 KB。checkpoint按validation informative retention@TOTAL B5选择，anchors计入预算；下表仅是checkpoint选择结果，不是独立test效果。

|模型|选中迭代|validation best-key保留|SHA256|
|---|---:|---:|---|
|linear_rank|155 epoch|24/25|`6726b902f8c624ffd0ed42fc715a7be682d93903d2dbafdd9c1c601748370293`|
|mlp_rank|161 epoch|25/25|`a13239ccba4a24bbb705b129fdc599203aaffe98e9627e83c506a33527710ef9`|
|gradient_boost|100 tree|24/25|`47d03dff1beaca5b5ab143512de1546738edfe7be710e3f654a82ccbddf92eb7`|
|value|113 epoch|24/25|`7156e473d7f7a6e955dd1850e26ff17a5ff8e53222cad26293925a248db82b08`|

manifest为`status=completed`、`frozen=true`、`smoke_only=false`、`test_read=false`、`roundtrip_predictions_equal=true`。四模型已经冻结，后续不得重新训练或按test改变checkpoint。记录runtime：Python3.12.3、NumPy2.5.1、torch2.13.0+cu130、sklearn1.9.1、joblib1.6.0；正式运行使用CPU，scaler只拟合train。

成功训练的命令在`/home/user/projects/terminal`工作目录执行：

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TERMINAL_SOURCE_COMMIT=4811abb805ab03997f5ce4186c6b2610f3c6b3b3 /home/user/envs/terminal_env/bin/python -m terminal_marl.scripts.train_fixed_terminal_stage5 --stage4-root /home/user/projects/terminal/.worktrees/arrival-restructure/tmp_stage4_results --output /home/user/training_outputs/terminal/stage5_validation_20261013/models
```

## 保留的启动失败

首次training launcher未设置项目cwd，2026-10-13 19:46:55（UTC+8）、PID3412261，0.036 s后exit1，错误为`ModuleNotFoundError: No module named 'terminal_marl'`；未生成模型。原`training_run_record.txt`、`training.stderr.log`、`training.time.log`保留。修正显式cwd后才进行上述唯一实际训练；这不是重新训练或根据结果调参。

四个pilot首attempt的wrapper预建了job输出目录，`beam-med`、`full-highlow`、`rolling60-med`、`snapshot-lowhigh`均在仿真前因`FileExistsError`退出，exit1，记录墙钟分别1.632/1.610/1.621/1.526 s。原始日志保留在根目录下`pilot/<job>/pilot.record.txt`、`pilot.stderr.log`、`pilot.time.log`。这些是流程启动失败，不是正式test失败，未改变冻结算法。

## 评价pilot与正式运行

修正wrapper后用seed5001重做四项流程pilot，输出使用四个新路径：`pilot_v2/beam-med`、`pilot_v2/full-highlow`、`pilot_v2/rolling60-med`、`pilot_v2/snapshot-lowhigh`，启动日志在`pilot_launch_logs/`。主对话已核对四项退出状态与完整产物，pilot已通过，四模型hash不变；pilot不并入正式集合。

|pilot_v2任务|exit|墙钟（s）|maxRSS（KB）|产物核对|
|---|---:|---:|---:|---|
|beam-med|0|123.55|710088|48decisions|
|rolling60-med|0|637.17|711768|48decisions|
|full-highlow|0|58.13|712684|48decisions|
|snapshot-lowhigh|0|48.94|713032|t0/15/90；t0和t15各11arm真实计时面板|

主对话已授权Luna启动冻结评价：30个snapshot场景jobs与330个episode jobs共360，另9个continuation jobs；90snapshot质量、24state真实计时和9context大池的具体口径见[协议](../../experiments/fixed_terminal_stage5_protocol.md)。已定test360使用20workers、continuation9使用4workers，每进程threads1，28 logical CPU下24并发预计内存约17 GB。实际于2026-10-13 20:05:07（北京时间）启动。

|任务|nohup/time启动PID|dispatcher PID|dispatcher开始UTC|状态|
|---|---:|---:|---|---|
|test360，20workers|3416299|3416301|2026-10-13T12:05:07.413712+00:00|已完成，360成功、0失败、exit0|
|continuation9，4workers|3416300|3416302|2026-10-13T12:05:07.404932+00:00|exit1，3成功6失败|

正式test终态来源于已同步本地`execution_test.json`：`status=completed`，360/360成功，pending/running/failed均0，exit0，`model_hashes_unchanged=true`；结束时间2026-10-13T13:17:04.821508+00:00，即北京时间21:17:04.821508，duration4317.407811361365 s。Luna正在核验服务器完整产物，最终研究结论待结果汇总和主对话解释；任务全部exit0不单独代表阶段5研究验收完成。

Luna启动初验test为3/360 completed、20 running、337 pending、0 failed，并有18份非空decisions.jsonl；这是启动时进度，不是完成结果。四模型hash不变。continuation的dispatcher结束于12:05:23.289072 UTC，记录duration15.884156 s，外层time墙钟15.93 s、exit1；原始错误为`ValueError: old visible snapshot mismatch`。主对话已完成下述限定诊断；原始失败记录保留。

根目录下保留`dispatch_logs/{test,continuation}.{record.txt,stdout.log,stderr.log,time.log}`、`execution_test.json`、`execution_continuation.json`、`progress_test.jsonl`和`progress_continuation.jsonl`。dispatcher PID、开始/结束和计数来自已同步本地execution；启动PID来自record。没有覆盖失败记录或据异常改动冻结模型。

初始T0约90 min：270个非rolling episode按约2 min、60个rolling按约10.6 min、30个snapshot场景按约1 min，合计约1206 worker-min，20workers理想墙钟约60 min，再加50%余量。它是主对话依据pilot确定的初估，未把不同任务当作均匀工作量；后续按实际阶段吞吐更新。预计超过一小时，启动验证已完成。当前线程heartbeat已工具创建并view确认：id=`5`，名称“阶段5冻结评价接续”，创建于2026-10-13 20:13:57北京时间，间隔81 min，约21:35复查并自动接续已授权分析与交付，无常驻监控worker。

## continuation限定诊断与比较修复

已收集九个旧4303场景的只读重建比较，证据为根目录`dispatch_logs/continuation_replay_comparison.json`。scenario全部exact相等；low三个场景完全相等，medium三个各有2个浮点差异，high三个各有4个浮点差异。最大绝对差均为1.7763568394002505e-15，仅位于已经存储的plan segment坐标；结构、字符串、整数、布尔和其他类型差异均0，所有新提议root plans无差异。该检查未运行evaluate_root、未读正式test或模型。

主对话诊断为跨平台浮点末位表示差异触发旧JSON严格相等检查，不归因具体库。正式test已达终态，Astra正在应用已审查的旧artifact比较修复并执行26项针对性检查，结果与验收待执行报告确认。修复仅允许finite float使用绝对容差1e-12、相对容差0；config/scenario和类型、离散值、结构保持exact。审计保留`exact_match`、差异count及max绝对差。它不改变物理、安全、J、模型、episode或snapshot计算，也不重跑已完成的360个正式test任务。

九个continuation计划先将首轮全部原始证据保留到`continuation_attempt1`，再完整重放一次。**当前第二轮尚未运行，不能写为修复验证成功或continuation已完成。** 原始失败和差异证据均保留；是否已归档、应用、通过26项检查及重放结果，均待后续实际记录更新。

## 独立汇总与打包验证

原运行源码的阶段1–5合并检查为101项、17.21 s。新增summary7项和package6项另计，共13 passed in 1.10 s；不将它们写成一次114项合并运行。真实pilot产物汇总CLI exit0，状态明确为`fixture_or_pilot_incomplete`，复制的41份文件与源文件hash保持一致。记录为`tmp_stage5_preflight/summary_integration/validation.json`；核对覆盖raw final value、decision timing/work、离线质量行、panel闭合标签、rolling目标单列及finite JSON，未启动新仿真或训练，未读正式530x结果。这是汇总输出链验证，不是正式评价完成证据。

后续summary/package与文档交付不改变冻结算法。阶段4历史结论和[前置诊断](diagnosis.md)保持原范围，阶段5收益须待正式评价结果后判断。当前不得以validation的25/25保留率宣称学习优势。
