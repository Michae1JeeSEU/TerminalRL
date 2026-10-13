# 阶段5运行记录：评价与研究验收完成

更新日期：2026-10-13。源提交`4811abb805ab03997f5ce4186c6b2610f3c6b3b3`已推送；阶段1–5合并检查为101 passed in 17.21 s、exit0，日志`tmp_stage5_preflight/combined_tests.log`。当前四模型正式训练及冻结、test360、修复后continuation9、完整产物核验和严格汇总均完成，研究验收已由主对话完成。存在离线信号和部分在线收益，但尚未达到不退化系统质量并明显降算的B类门槛，也无长期价值C类证据；不笼统宣称学习完全无效或普遍无必要。结果与边界见[结果报告](results.md)。版本与交付校验信息由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，交付状态见最终交付回复。

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

正式test终态来源于已同步本地`execution_test.json`：`status=completed`，360/360成功，pending/running/failed均0，exit0，`model_hashes_unchanged=true`；结束时间2026-10-13T13:17:04.821508+00:00，即北京时间21:17:04.821508，duration4317.407811361365 s。Luna与主对话已完成完整产物核验：330episode、30snapshot jobs、90base contexts、9large contexts、24panels/264rows。严格汇总及主对话研究判断亦已完成；运行成功与研究收益按结果报告分别解释。

Luna启动初验test为3/360 completed、20 running、337 pending、0 failed，并有18份非空decisions.jsonl；这是启动时进度，不是完成结果。四模型hash不变。continuation的dispatcher结束于12:05:23.289072 UTC，记录duration15.884156 s，外层time墙钟15.93 s、exit1；原始错误为`ValueError: old visible snapshot mismatch`。主对话已完成下述限定诊断；原始失败记录保留。

正式test记录保留在根目录`dispatch_logs/test.{record.txt,stdout.log,stderr.log,time.log}`、`execution_test.json`及`progress_test.jsonl`。首轮continuation记录已归档至`continuation_attempt1/dispatch_logs/continuation.{record.txt,stdout.log,stderr.log,time.log}`、`continuation_attempt1/execution_continuation.json`及`continuation_attempt1/progress_continuation.jsonl`；根目录的execution/progress_continuation记录第二轮。dispatcher PID、开始/结束和计数来自对应execution；启动PID来自对应record。没有覆盖失败记录或据异常改动冻结模型。

初始T0约90 min：270个非rolling episode按约2 min、60个rolling按约10.6 min、30个snapshot场景按约1 min，合计约1206 worker-min，20workers理想墙钟约60 min，再加50%余量。该初估依据pilot，未把不同任务当作均匀工作量。预计超过一小时，因此完成启动验证后创建当前线程heartbeat并view确认：id=`5`，名称“阶段5冻结评价接续”，创建于2026-10-13 20:13:57北京时间，间隔81 min，约21:35复查并接续了本次已授权分析与交付，无常驻监控worker。

## continuation限定诊断与比较修复

已收集九个旧4303场景的只读重建比较，证据为根目录`dispatch_logs/continuation_replay_comparison.json`。scenario全部exact相等；low三个场景完全相等，medium三个各有2个浮点差异，high三个各有4个浮点差异。最大绝对差均为1.7763568394002505e-15，仅位于已经存储的plan segment坐标；结构、字符串、整数、布尔和其他类型差异均0，所有新提议root plans无差异。该检查未运行evaluate_root、未读正式test或模型。

主对话诊断为跨平台浮点末位表示差异触发旧JSON严格相等检查，不归因具体库。修复提交`b7233cfb58b1b7ebeb534ad095a6373516fac84c`已推送，26项正式相关检查通过（0.79 s），其他17函数AST相同。修复仅允许finite float使用绝对容差1e-12、相对容差0；config/scenario和类型、离散值、结构保持exact，审计保留exact_match、差异count及max绝对差。它不改变物理、安全、J、模型、episode或snapshot计算，未重跑已完成的360个正式test任务。

首轮全部33files/133416 bytes逐hash保留至`continuation_attempt1/`。第二轮完整重放九个continuation，于2026-10-13 21:37:28北京时间启动，time/dispatcher PID3424338/3424339；dispatcher从13:37:28.791589至13:37:38.945099 UTC，duration10.153521 s，外层time10.20 s、exit0，9/9成功、0failed。日志为`dispatch_logs/continuation_attempt2.*`，execution/progress保留第二轮状态，首轮证据没有覆盖。四模型hash不变。

第二轮虽成功，九个selected union均仅HOLD、pair_count0，不能检测排序反转或证明continuation无偏差，两个world请求分母不同；这是证据退化边界，不是追加实验理由。

## 独立汇总与打包验证

原运行源码的阶段1–5合并检查为101项、17.21 s。新增summary7项和package6项另计，共13 passed in 1.10 s；不将它们写成一次114项合并运行。真实pilot产物汇总CLI exit0，状态明确为`fixture_or_pilot_incomplete`，复制的41份文件与源文件hash保持一致。记录为`tmp_stage5_preflight/summary_integration/validation.json`；核对覆盖raw final value、decision timing/work、离线质量行、panel闭合标签、rolling目标单列及finite JSON，未启动新仿真或训练，未读正式530x结果。这是汇总输出链验证，不是正式评价完成证据。

正式严格summary随后一次完成，`status=completed`、`allow_incomplete=false`；counts为330episode/30snapshot jobs/90base contexts/9large contexts/24panels/264rows/9continuations，日志`dispatch_logs/summary.*`。源提交分开记录为4811abb805ab03997f5ce4186c6b2610f3c6b3b3（训练/正式test）与b7233cfb58b1b7ebeb534ad095a6373516fac84c（artifact修复/continuation诊断）。原101项、独立13项及修复26项分别报告，未声称一次合并140项测试。

330episode全请求守恒均true，spacing违反和uncertified总和均0，最低着陆SOC0.06170707293994795高于reserve0.06；这些不抵销真实能源失败。32上限面板实际未扩大root ID集合；其余完整结果与局限见[结果报告](results.md)。保留beamB5/B8/full工程参考与rank/GB实验性结果，下一阶段另定范围，不自动扩大网络或启动实验。

版本与交付校验信息由包内`BUNDLE_README.md`、`MANIFEST.json`及包外`validation.json`、`zip.sha256`记录，交付状态见最终交付回复。
