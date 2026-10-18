# 真实串行共同状态窗口入口

`scripts/server/run_rotating_slots_performance_windows.py` 只执行一个真实串行窗口。
当前结果始终为 `NOT_QUALIFIED / HUMAN_PENDING`。本入口没有 paired orchestration、
并行后端、外部诊断预算或硬超时 collector；不能据此宣布合同性能验收通过。

唯一子命令是 `window`。必须提供 `--corpus-manifest`、预期 canonical
`--corpus-sha256`、六个固定 ID 之一的 `--window`、`--duration {10,60}`、
`--kernel {R0,OPT}`、`--reference`、`--deployment-manifest`、
`--implementation-commit` 和全新 `--output`。`--mode {benchmark,profile}`
默认为 benchmark。只有 implementation identity 确实改变时才提供
`--migration-approval`。通过 `python -I -B` 启动脚本；脚本在导入 helper
之前也设置 `sys.dont_write_bytecode=True`。

## 输入与 implementation 迁移

入口复用 `runner.bootstrap` 进行精确 deployment 和固定 R0 导出验证，先导入
flat `performance` helper，再选择唯一 core 根目录。窗口启动脚本自身的
deployment SHA 也加入 `helper_execution_files_sha256`；execution SHA 仍按
`{kernel,core,helpers,observation_shim}` 的 canonical digest 计算。CLI 字节变化
会改变目标 implementation identity，旧 deployment manifest 会被拒绝。

corpus 必须是 `FROZEN`，六个窗口全部 `AVAILABLE`，预期 SHA 必须等于去掉
`manifest_sha256` 字段后的 canonical digest。入口重算冻结的确定性选择，检查
固定 R0、语义、每个窗口的 checkpoint 原目标、outer phase、cold restore 证据
与 60 秒已观察后缀。对于本次运行的窗口，还重新读取原 source 的 terminal
result、ledger、source identity、完整 retained index 与 sidecar，验证冻结副本
和原 checkpoint/sidecar 的 SHA。真实 payload 通过 `load_checkpoint` 读取，
checkpoint 的 manifest、physics、时间、cursor 和 stream dependency 也要匹配。
原 ReplayBundle 单独验证并核对 workload 和已观察范围。

冻结文件中的 Linux 绝对路径保持原值；本地 Windows 上仅复制 corpus 不能
替代保留原 source。源 stream 和 metadata 全部只读。恢复会验证原 stream 的
offset、prefix SHA、完整行数和 pending frame，并复制到新 attempt；运行结束
再次检查源 prefix、源元数据和冻结 pair。record/replay 从不作为 checkpoint。

source 与 runtime 的 `CheckpointIdentity` 不同时，必须消费一个主对话复核等价证据后显式登记的
JSON artifact，schema 精确为 `window_migration_approval_v1`，包含：

- `qualified: true`、`corpus_sha256`、`target_execution_sha256`；
- `target_observation_original_sha256`、`target_observation_generated_sha256`、
  `checkpoint_helper_sha256`，分别绑定目标 observer 源码和当前 `checkpoint.py` 字节；
- 非空 `supporting_evidence: [{path, sha256}, ...]`，每个绝对路径都存在且 SHA 匹配；
- `migrations` 数组。每项为 `migration_id`、完整 `source_identity`、完整
  `target_identity`、`source_observation_original_sha256` 和
  `source_observation_generated_sha256`。

所选 pair 必须唯一；两 identity 除 `implementation_sha256` 外全部严格相同。
source/target 原 observer 和生成 shim 必须完全相同。artifact 不能授权新的
shim 行为，入口不改写 payload/cursor，不支持通用 shim 迁移。只有检查全部
通过才调用 `register_equivalence_migration`，向真实 `restore_checkpoint` 传入
该 migration ID。旧 source、target、corpus、shim、checkpoint helper 或空/变动
evidence 都被拒绝。消费者不生成 approval；`qualified=true` 也不代表完整
六窗口性能、完整 episode 或最终行为等价已经验收。

## 原调用继续执行和计时

冷恢复不 advance、不裁记录、不改 canonical 政策。起点取真实 `world.time`，
通过 `observed.resume` 保留原来的 3600/5400 目标和 outer phase。名义停止时间为
起点加 10/60 秒，在原循环第一个达到该时间的自然 hook 保存完整 checkpoint
后 `SuspendAtCut`。不以窗口终点额外调用 `advance`，不切分原物理步。结果同时
记录名义终点、实际终点、deferral 和实际 sim delta。物理失败提前发生记录为
`PHYSICAL_FAILURE`，不会算作完成整个 10/60 秒窗口。

`Measurements` 单列输入/归档验证、checkpoint load、cold restore/copy、正常
checkpoint capture/save/flush、rollout、输出 ReplayBundle 验证和 finish verify。
rollout wall 与真实 `time.process_time` CPU 从恢复完成后开始，包含后续 observer
准备、正常 checkpoint、记录和 flush；窗口内部 cache 预热全部计入。total wall
与 total parent CPU 从 bootstrap 前开始。纯 serial 没有 worker。已有
`OwnedProcessMetrics` 仅采样 parent；sample peak 是下界，OS lifetime HWM 是
上界，两者不互相冒充。资源资格仍标记 `INCOMPLETE`。

恢复完成后，入口写出 `cold_cache_report.json`：world 的三个派生 cache、
`trajectory.poly_bounds`、`energy._integrable_plan/_segment_table`、
`safety._NUMERIC_CERTIFICATES/_PLAN_CERTIFICATES` 的实际大小分别记录。
配置 `_default_config` parse cache 单列，固定配置与 Gauss 表保留。入口不清除
模块 cache；若 source archive/restore 验证已预热任何数值 cache，则拒绝进入
rollout，保留报告供主对话判断，不能把非空缓存标成全冷。

每 30 sim 秒或 300 wall 秒通过原自然切点请求正常保存。末态 checkpoint 及
sidecar 保留完整续跑状态；实现失败时保留最近完整 checkpoint，尚无新 checkpoint
则明确引用原冻结 checkpoint。实现失败后的 live replay 仅作 diagnostic。
输出包含 `result.json`、`source_identity.json`、`input_identity.json`、observer
patch/原始/生成源码、独立 heartbeat、RSS samples、完整状态 prefix、ReplayBundle
与 checkpoint index。记录起终 aircraft 状态数量、driver calls、已有可靠计数和
记录字节；当前未接入的逻辑请求/真实计算/淘汰计数明确写 `MISSING`。

benchmark 不启用 cProfile。profile 只在 rollout 的真实续跑调用中使用现有
`Measurements.profile_batch`，保存 `window.prof`；cold restore 不在 profile
内。profile 结果明确禁止用于主性能资格。两模式的输入、advance、driver、
应急和物理记录语义相同。

## 有界验证

`tests/test_rotating_slots_performance_windows.py` 默认运行纯 parser、canonical
manifest、选择/后缀、source/target/shim/helper/evidence 审批负向测试，以及新 CLI
字节进入 execution SHA 的检查。该层不生成 deployment 或导入 core。

`RUN_WINDOW_INTEGRATION=1` 启用一个独立 `python -I -B` service-events 小 fixture。
fixture 从原目标 3600 的非网格自然切点生成真实 checkpoint，以真实冷恢复和
原 driver 运行完整 10 sim 秒，核对 deferral、原目标、源只读、no-overwrite、
末态再次冷恢复和不中断同原目标的完整记录一致；再比较 profile/benchmark。
该 fixture 在测试进程里替换 corpus admission 输入，执行器和 checkpoint/record/
ReplayBundle 路径保持真实；它不是冻结 traffic 语料，也不构成性能资格。

2026-10-18 本地最终执行命令（PowerShell）：

```powershell
$env:RUN_WINDOW_INTEGRATION='1'
& './venv/Scripts/python.exe' -I -B -m pytest tests/test_rotating_slots_performance_windows.py -q -p no:cacheprovider --basetemp=window_runtime_integration_20261018b
```

实际退出码 0，`30 passed in 13.25s`。完整日志：
`D:/7_codespace/terminal/window_runtime_integration_20261018b.log`。
本轮 artifact 根目录：
`D:/7_codespace/terminal/window_runtime_integration_20261018b/test_real_cold_window_fixture_0`。
其中 `fixture_evidence.json` 保存 benchmark、profile、migration 三个执行结果；
三个子目录分别保留真实 result、完整 prefix、ReplayBundle、末态 checkpoint、
heartbeat、RSS 与 observer 源码。三个分支均从 0.35 继续原目标 3600，实际停止于
10.399999999999993，实际推进 10.049999999999994 秒，deferral 为
0.049999999999993605 秒。与不中断原目标执行以及模式之间的完整状态记录相同，
源 stream 字节保持不变；末态 checkpoint 再次冷恢复的时间和原目标一致。

`migration` 分支在绑定新窗口 CLI 前，用真实 `runner.bootstrap` identity 捕获
checkpoint；绑定入口后使用新的 runtime identity。测试内的明确 test-only artifact
包含真实 SHA evidence，经过生产 `validate_approval`、登记和 `restore_checkpoint`
执行完整 10 秒；未改写 payload identity、cursor 或 shim。该 artifact 只测试迁移
执行链，不批准生产 P2 实现。`missing-approval.stdout.txt` 记录
`MISSING_APPROVAL_REJECTED`，且目标输出目录未创建。过期 launcher deployment
SHA 由真实 CLI 拒绝为 `DEPLOYMENT_BYTES_MISMATCH`，未创建输出。

`partial-failure` 注入最后完整 hook 之后的局部 time 改动和异常，结果为
`IMPLEMENTATION_FAILURE`，最终 heartbeat/result 的 `last_complete_sim_time` 均为
0.35，并引用原完整 checkpoint。它没有把局部改动后的 live time 冒充完整状态。

`benchmark/cold_cache_report.json` 实测 world 三个 cache 全部 0；
`poly_bounds`、`_integrable_plan`、`_segment_table` 的 `currsize` 全部 0；
两个 safety 证书字典均 0，配置 parse cache 的 `currsize` 为 0，
`module_caches_cleared=false`。这些是 service-events fixture 的实际观测，
不能推广为真实冻结 traffic 的缓存状态。

本轮集成仍只有测试内的 `windows.validate_inputs` 被替换为 service-events
checkpoint 输入；`load_checkpoint`、恢复、driver、observer、记录、ReplayBundle
和生产迁移校验未被替换。真实 CLI 的来源/workload 验证未放宽。真实冻结
traffic 的端到端执行尚未在 Windows 验证，其原 Linux 绝对 source 依赖继续保留。

## Linux 执行器测试与真实冻结 checkpoint 冷恢复预检

Linux 执行器测试输出为
`/home/user/training_outputs/terminal/rotating_slots_perf_v1/window_runtime_linux_tests_20261018a`，
同步本地目录为
`D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/window_runtime_linux_tests_20261018a`。
实际 `run_metadata.txt` 记录 UTC 开始时间 `2026-10-18T13:59:36.367754Z`、
结束时间 `2026-10-18T13:59:39.998062Z`、`RUN_WINDOW_INTEGRATION=1`、
`sha_match=True` 和 R0 导出总文件数 25（24 个源文件加 manifest）。
`numericexitcode.txt` 为 0，`stdout.log` 为 `30 passed in 3.42s`，没有 skip。
这里仍是上述 service-events 测试 seam：测试内替换 `windows.validate_inputs`，
不代表真实冻结 traffic 已推进 10/60 sim 秒，也不提供真实窗口速度证据。

本轮实际部署与显式迁移登记保存在
`D:/7_codespace/terminal/artifacts/rotating_slots/window_deployment_fe66ee4_20261018a`，
服务器对应目录为
`/home/user/projects/terminal/artifacts/rotating_slots/window_deployment_fe66ee4_20261018a`。
其中 `deployment.json` 绑定 implementation commit
`fe66ee4b24f7a71ce0d39d0ecf4862034d07aaa7`，`R0/preflight.json`、
`OPT/preflight.json` 记录两个独立 byte-verified runtime；
`R0_migration_approval.json`、`OPT_migration_approval.json` 分别登记两个冻结 R0
source implementation 到目标实现的精确 identity pair。冻结来源 implementation 为：

- `36562efb3e1fc1d06abc6cf98ad83109a5d2a73c4d4937ef4712a4646fbdd341`；
- `b3f2bec1a5f71d795a772d945adcbd20d93e1a534aaafcd4df42b763a968db81`。

目标 execution SHA 为：

- R0：`ebd388961bb4e021b49dcc34e72f8223a333296680885aa91845970515ad65d4`；
- OPT：`71585196c485cf3f0ab1d9c46e2e25a10605228e3bd4d45c5e6b65c97af31b98`。

这些登记只允许已经复核的 implementation 差异；checkpoint helper、原/生成
shim 和其余 `CheckpointIdentity` 字段保持一致，不迁移语义或改写原 source
绝对路径。`artifact_manifest.json` 保存输出与 supporting evidence 的 SHA、
命令、时间和限定授权范围。登记本身仍为 `NOT_QUALIFIED / HUMAN_PENDING`。

真实冻结 checkpoint 的 Linux 预检汇总为
`/home/user/training_outputs/terminal/rotating_slots_perf_v1/window_migration_preflight_fe66ee4_20261018a/summary.json`，
同步本地文件为
`D:/7_codespace/training_outputs/terminal/rotating_slots_perf_v1/window_migration_preflight_fe66ee4_20261018a/summary.json`。
批次 UTC 开始于 `2026-10-18T14:07:35.799055Z`，最后一项结束于
`2026-10-18T14:08:32.776466Z`；summary 收尾时间 `ended_utc` 为
`2026-10-18T14:08:32.776859Z`。R0 与 OPT 各覆盖 W71-A、W71-B、W71-C、
W60、W120、W180，共 12 项，全部 `exit_code=0`、`status=ok`。

12 项均报告 world、cursor、default/effective callback 和 callback shared 状态
round-trip 精确；world time、phase、outer phase、原 advance target 精确；
state stream 除新 attempt 路径外，其 committed count、offset、pending frame/
SHA、prefix SHA 和 time 全部精确；原 checkpoint、sidecar 和 stream prefix
hash 不变；指定 world 与模块数值 cache 全为 0，且 `module_caches_cleared=false`；
结束时 source/deployment verification 通过。主对话已读取全部 summary 并抽查
`OPT/W71-A/result.json` 的逐字段结果。

这 12 项只执行真实冻结 payload 的校验、显式 implementation 迁移、冷恢复及
再次捕获比对，没有调用 `world.advance`、`world.snapshot` 或 `world.to_dict`。它们提供真实
checkpoint 迁移与冷恢复证据，不是实际 10 sim 秒 rollout 或性能加速测量。
真实窗口推进、完整交通与性能资格仍待对应运行，当前状态保持
`NOT_QUALIFIED / HUMAN_PENDING`。
