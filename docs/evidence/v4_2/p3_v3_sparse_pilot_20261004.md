# V3/90D稀疏专家标注开发pilot（2026-10-04）

## 目的与范围

用户授权继续下一步，并允许长耗时任务每60分钟heartbeat检查，不要求第一时间读取结果。primary继续采用research-pragmatist/academic-experiments的最小开发验证：实现上一轮记录的“完整规则回合→事先固定pre-state离线标注”，亲自负责代码、文档、分析及准入判断；worker仅执行指定SSH、测试、监控和本地Git。

前轮892201 expert-on-policy采集在298ticks/3625.746s被截断，3N/9U，无实际安全/hash异常；固定状态25候选查询的主要成本是轨迹生成。缓存试验等价但收益不足，已撤回，诊断commit `f542ba485456e7d7892cda6986f69fdcf928e80b`。这些是preexisting_artifact，详见[前轮报告](p3_v3_i0_pilot_20261003.md)，本轮不重跑旧pilot或改写其CENSORED。

新脚本`terminal_marl/scripts/collect_p3_v3_sparse_expert.py`只做开发数据流程验证，不训练。轨迹、安全、动力学、能量和奖励核心保持原版本；沿用`p3_integrated_repair.build_config`的static/direct_c2/nominal-holder/90D配置，以及此前integrated repair验证中的名义RELEASE、speed 0/-1/+1三个候选的rule K3行为。专家标签只在私有快照中查询，不进入规则行为动作。

## 首次reset前固定的协议

| case | seed | 负载/SOC | 环境预算 | 标注预算 |
|---|---:|---|---|---|
| smoke | 892401 | 3请求/warm3，.90–1.0 | 2ticks/180s | 180s，本窗口最多1个UID |
| high | 892402 | 12请求/warm6，.90–1.0 | 完整终态或1800ticks/3600s | 1800s，最多24次query |
| mixed | 892403 | 12请求/warm6，.55–.90 | 同high | 同high |

上述seed本地configs/scripts/tests在代码写入前扫描无匹配。server历史输出检查：候选seed路径无匹配；4531个protocol/manifest/config/result/summary的JSON/YAML及小型log扫描完成，排除小数片段的精确整数匹配为0，41个大于2MiB文件未读。先前全量raw行数字扫描在55/60s上限停止，本地出现无文件路径的数字片段，不能认定是seed使用；本轮以已完成的元数据/路径核对为依据，不声称已穷尽巨型raw文件。扫描不消费环境seed。892401–892403为新development分区，永不并入train/audit/Q；原893001–893024/894001–894006/895001–895012仅预留，未使用。

采样在标签、终态结果和耗时出现前决定：1800tick划为24个75tick窗口；每窗口第一个至少有一架actor eligible的pre-state，选择**按数值UID升序**列表中的`window_index % eligible_count`对应一架。一个窗口最多一条，空窗口不补额，不按teacher gate、拒绝理由、episode成败或事后耗时选样。自然终态之后的窗口不虚构状态。固定窗口的短暂事件漏采风险按实际覆盖报告，不为凑覆盖临时追加查询。

每个query沿用原state-only teacher的25种RELEASE意图以及最后WAIT顺序，遇首个可行候选即停止。最多24×25=600个**RELEASE意图检查**。primary代码检查发现final候选可能遍历多个LZ，所以这不等于600次`_candidate_plan`或轨迹构造；新输出分别统计release_intents_checked与candidate_plan_calls，保留teacher audit中的LZ检查。不得为了满足虚假的600计划次数门槛截断原专家搜索。

所有active每tick的90D原始观测、actor_mask、规则请求和候选、pre-state hash都保留。原始`rows.jsonl`中teacher_label始终null；实际标注写入独立`labels/window_XX.json`，按decision_id关联，只有status=OK才是有效标签。未选、未执行、超时和错误的标注不能填成WAIT；forced-WAIT不能进入actor损失。后续proposal监督仅限有效teacher RELEASE。标签单机可行性与规则K3批量执行不混淆，不把teacher request当作已执行请求。

环境循环不查询专家，不因标注额度用完而停；保留完整steps/action traces/N/E/F/D/U、checked/executed与能量。首tick保存before/request/live/replay并严格检查重放；每step复用已测execute_tape检查实际安全、hash、有限obs/reward、请求与执行契约。异常留存pre-state及请求，failed_energy立即按ERROR停止；wall/tick预算到达而未自然终态保持CENSORED。环境墙钟在tick边界检查，单tick可能略越界，不事后延长预算。

正常完整终态后才开始离线标注（smoke允许规定2tick截断）。每个已选状态一个顺序子进程，parent以剩余标注总预算为timeout；超时杀死并回收该child，记录TIMEOUT并让后续项保持NOT_ATTEMPTED。标签文件、日志及快照保留，错误不得升级成功。环境完成和标注完整性分别输出；PILOT_OK只要求环境按本case完成、至少一条有效标注且所有已选标注完整，无I0覆盖/性能资格含义。若环境存在D或其他业务阻塞，由primary分析后决定，不自动启动mixed。

执行顺序：定向tests→独立smoke→primary检查smoke原始行/标签/trace→只启动high。high自然结束且无技术/安全/耗尽/未决/业务阻塞，primary分析后才启动mixed。任何失败保留原输出，完成最小必要诊断，不同目录另记后续试验；不重启覆盖。最多一个本轮父进程及其一个标注child（顺序运行），不同时开两个环境回合。

## 输出及执行

server cwd `/home/user/projects/terminal`，Python `/home/user/envs/terminal_env/bin/python`，`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。输出根`/home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004`，各case独立新目录；相邻.log/.pid/.exit保存真实进程与状态。代码/docs仅本地修改，经Syncthing核对后执行；无源码手传、无server Git。

```bash
python -u -m terminal_marl.scripts.collect_p3_v3_sparse_expert --case smoke --output /home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004/smoke
python -u -m terminal_marl.scripts.collect_p3_v3_sparse_expert --case high --output /home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004/high
python -u -m terminal_marl.scripts.collect_p3_v3_sparse_expert --case mixed --output /home/user/training_outputs/terminal/p3_v3_sparse_pilot_20261004/mixed
```

每臂运行预算最多60分钟环境阶段＋30分钟标注阶段（环境tick边界可能略超），实际预计几十分钟内。启动后必须核对真实argv和至少一份预期产物；之后按用户偏好设置60分钟heartbeat，正常推进静默。high完成后继续已授权mixed；全轮终态/必要诊断完成、记录与提交后删除heartbeat。

## 结论边界

本轮rule occupancy与上一轮expert occupancy不同，因此不能把新旧总耗时或成功率差异直接当作同分布算法改进；本轮检验的是完整周期与有界标注能否分别完成、字段是否正确。先前完整环境开发验证已通过；新fresh90D student还未训练/qualified。正式I0仍需冻结采样及覆盖协议，后续fresh BC→Safe DAgger→student-only资格后才讨论有界PPO。PPO_ALLOWED=false；本轮不自动学习、full training、held-out或硬件，不创建/覆盖checkpoint，不声称训练退化已解决。

## 执行记录

代码、测试和以上协议由primary编写。测试与独立输出检查的实际结果随后追加，未运行项目不写PASS。

### 定向测试与独立smoke（newly_run）

`tests/test_p3_v3_sparse_expert.py tests/test_p3_v3_expert_collector.py tests/test_joint_teacher_state_only.py`共17 passed in 39.23s、exit0，日志`/home/user/training_outputs/terminal/p3_v3_sparse_tests_20261004.log`。覆盖窗口选择/数值UID/不补额、90D与mask/null、环境循环不调用teacher、真实快照标签与原teacher逐项相同、原始文件不变、真实子进程CLI与JSON、环境cap/error留证、标注总预算/单child timeout及hash校验。collector SHA`e46d4f2176f186ccf0b9650845ef1885419af984bdd0a154f3b27ce8c152cf35`，test SHA`1de56ddc67430d75965aa178d99e5258291f84cd2d298721ac81253bce59e608`，local/server一致。

独立CLI smoke892401已SMOKE_OK/exit0结束，原PID2478893退出。primary亲自读取全部6行/2steps/1selected/1label：6个decision_id唯一，90D及原始teacher_label=null、规则request与tape、row/step pre-state hash均相同；3 actor行/3 forced WAIT，选中tick0 UID0。单机teacher joint13/proposal12、request[1,0,0]，1个意图/1次plan检查，ring_transfer checked/content hash相同。first_live/replay完整相同；live规则批准3次，checked/executed samples30，所有实际安全/hash异常计数0，查询未计入live候选统计。环境阶段4.9541s、标注阶段2.8732s（child query2.1093s）。terminal=false/censored=true仅为规定2tick smoke，不是完整业务结果。

smoke result为4135bytes，SHA`106284862aa5385dd0bd6f847dd28c24ca31c9be469cbadb2b1e8b0ae965fab8`；目录`.../p3_v3_sparse_pilot_20261004/smoke`及其相邻log/pid/exit保留。primary据此接受新输出路径检查，只准许接着启动预注册high一臂；mixed仍待high终态分析。
