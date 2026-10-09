# P3 邻机关系 gate 校准候选

## 证据范围与 CV 终态

本记录登记已存在的邻机关系 CV 结果（`evidence_type=preexisting_artifact`，2026-10-09 检查），不重新运行 CV。server 结果为 `COMPLETE/exit0`，共 12 个 fold-arm，完成时间记录为北京时间 `2026-10-09 14:28:52`，旧进程已退出。结果文件为 `/home/user/training_outputs/terminal/p3_neighbor_relation_cv_20261009/six_cycle/result.json`，SHA256 为 `238cf0bb0742bef07d260819b54d3cd764ed6a34b34a2cf5bc4debcec2e10aa6`。

协议固定使用现有 V2.6 `159+N50` 观测和六个完整回合 `893511/893512/893513/893601/893602/893603`，model seed `91911`，两臂为 `without_auxiliary` 与 `native_pair_auxiliary`。原 gate+RELEASE proposal CE、pair blocked/safe BCE、六回合留一划分和 marks `0/100/500/1000/2000/5000` 保持不变。验证集的逐 fold 结果如下；本记录不计算整体均值。

| held seed | arm | validation RELEASE recall | validation proposal top1 | validation joint | validation blocker top1 |
|---:|---|---:|---:|---:|---:|
| 893511 | without_auxiliary | 1.000000 | 0.652174 | 0.500000 | 0.771429 |
| 893511 | native_pair_auxiliary | 1.000000 | 0.695652 | 0.600000 | 0.771429 |
| 893512 | without_auxiliary | 0.361702 | 0.372340 | 0.841886 | 0.062500 |
| 893512 | native_pair_auxiliary | 0.425532 | 0.372340 | 0.846047 | 0.500000 |
| 893513 | without_auxiliary | 0.482759 | 0.379310 | 0.787934 | 0.322581 |
| 893513 | native_pair_auxiliary | 0.597701 | 0.344828 | 0.733090 | 0.612903 |
| 893601 | without_auxiliary | 0.777778 | 0.722222 | 0.666667 | 1.000000 |
| 893601 | native_pair_auxiliary | 0.944444 | 0.777778 | 0.722222 | 1.000000 |
| 893602 | without_auxiliary | 0.600000 | 0.453333 | 0.821366 | 0.222222 |
| 893602 | native_pair_auxiliary | 0.533333 | 0.426667 | 0.798599 | 0.481481 |
| 893603 | without_auxiliary | 0.520000 | 0.413333 | 0.768194 | 0.447368 |
| 893603 | native_pair_auxiliary | 0.520000 | 0.360000 | 0.752022 | 0.605263 |

四个 dense 留出回合的 auxiliary blocker top1 高于对应 control：`893512 .062500→.500000`、`893513 .322581→.612903`、`893602 .222222→.481481`、`893603 .447368→.605263`。对应 auxiliary proposal top1 为 `.372340/.344828/.426667/.360000`，joint 为 `.846047/.733090/.798599/.752022`。这些是逐 fold 观察，不能写成 P3 已解决或 promotion 依据；旧 qualification 仍为 false。

## 数学假设与边界

当前 pair BCE rejection scores 通过邻机/scene `max`、destination `min` 汇总，再通过 25 个候选的 first-feasible product 生成 gate/proposal 分布。若每个候选 feasibility logit 都为 `-2`，单项可行概率约为 `.1192`，则 RELEASE union 为 `1-(1-.1192)^25≈.958`，而 max gate 仍为 `.1192`。这是组合公式的行为观察，不是经验失败的已证根因；pair scores 不称 calibrated scores 或 certificates，也不把 threshold 当作安全规则。

在线边界保持不变：旧 default/runtime 不改，`159+N50` 不改，不新增 online oracle 或 online input。有限广播接收器仍未部署；后续接入前仍须满足本机传感器、实际通信或中心有限广播来源，以及范围、更新、有效期、缺报和失联处理约束。离线阻塞 UID、候选安全结果和安全证书继续只作为训练/诊断监督，不能进入在线 observation。

## 固定下一项 protocol

只增加 optional `max_candidate` gate，范围为 `[0, max candidate_logit]`；旧 proposal logits 保持。以下内容冻结：

- 参数化、初始化 `91911`、BC+pair losses、2258 原行、317 queries、splits/samplers；
- Adam `3e-4`、batch `128`、auxiliary batch `64`、clip `.5`；
- marks `0/100/500/1000/2000/5000`，每 fold `1200s` 上限；
- 仅运行 `native_pair_auxiliary` arm × 同六 fold，输出 `/home/user/training_outputs/terminal/p3_neighbor_gate_max_20261009/six_cycle`；既有 native arm 复用为 control，不重跑旧 CV。

在 prelaunch 状态记录中，本地 23 项定向 tests 已通过（4.10s，exit0）。server preflight 首次因 source hash mismatch 未创建输出；Syncthing 匹配后仅重试一次，完成 `23 PASS/1.45s/exit0`，日志为 `/home/user/training_outputs/terminal/p3_neighbor_gate_max_20261009/preflight.log`。当时 synthetic smoke 尚待启动，当前实际 supplemental smoke 见下文。评估固定 5000-step 的全部 fold，包含 WAIT false-release 与 pair metrics；不挑最好 epoch，不新增 threshold。max_candidate gate 对照不访问 Q、不 reset、teacher query 或 checkpoint；PPO/formal 仍禁。该 prelaunch 记录当时的补充 seed 尚未消费，后续授权/运行状态见下文。

此前启动阶段状态仅记录了首个 prefix；完整终态与当前状态见下文，不把该 prefix 当作 full result。

full six_cycle 曾 once-only 启动：`2026-10-09T07:57:29Z`（北京时间 `15:57:29`），wrapper `3059929`/timeout `3059930`/Python `3059931`，cwd `/home/user/projects/terminal`，输出 `/home/user/training_outputs/terminal/p3_neighbor_gate_max_20261009/six_cycle`；启动时 `exit0`、初始 `RUNNING/error=null`。首个 held `893511` 仅到 marks `[0,100]`、elapsed `18.5478617s`，2258 total BC/317 aux，six input seeds matched，`max_candidate`/native-only/`91911`，Q/query/step/checkpoint 均为0。该 prefix 不代表 full result；后续终态见下文，原 `p3-gate` heartbeat 已删除。

## 终态：max_candidate full six_cycle（2026-10-09）

本次 `max_candidate` full six-cycle 为 `COMPLETE/exit0`，六个 held seed 均完成 fixed `5000` update；当前 max_candidate gate 对照无 active job。结果与 protocol 使用以下本地同步产物，且 server/local hash 已匹配：`D:\7_codespace\training_outputs\terminal\p3_neighbor_gate_max_20261009\six_cycle\result.json`（server/local SHA256 `1bdaf461e60a9c5d54ce0177ef6374066d3f87d29a1dd9d3964ad1e70cf11a45`）和 `D:\7_codespace\training_outputs\terminal\p3_neighbor_gate_max_20261009\six_cycle\protocol.json`（server/local SHA256 `5d8eb39bc31eae199a26f29d5c2f445b310832c212046975d8650c91717099df`）。此前 wrapper/timeout/Python PIDs `3059929/3059930/3059931` 当前均 absent。local 与 server 的 23 项 tests、synthetic smoke 和 preflight 是既有检查，本次 heartbeat 未重跑。

固定 `5000` 结果与既有 `native_pair_auxiliary` control 并列如下；各数值由上述 `result.json` 与 control 的 `p3_neighbor_relation_cv_20261009/six_cycle/result.json` 直接读取。`WAIT false-release` 为 validation gate confusion 中实际 WAIT 行被预测为 RELEASE 的比例；实际 WAIT 行为 0 时记为 `null`。

| held seed | max R recall | control R recall | max WAIT false-release | control WAIT false-release | max proposal top1/top3 | control proposal top1/top3 | max joint | control joint | max blocker top1 | control blocker top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 893511 | 0.913043 | 1.000000 | 0.714286 | 0.714286 | 0.652174/0.956522 | 0.695652/0.913043 | 0.566667 | 0.600000 | 0.742857 | 0.771429 |
| 893512 | 0.404255 | 0.425532 | 0.074960 | 0.070175 | 0.287234/0.414894 | 0.372340/0.531915 | 0.834951 | 0.846047 | 0.468750 | 0.500000 |
| 893513 | 0.643678 | 0.597701 | 0.193478 | 0.184783 | 0.379310/0.574713 | 0.344828/0.540230 | 0.727605 | 0.733090 | 0.387097 | 0.612903 |
| 893601 | 0.944444 | 0.944444 | null | null | 0.888889/0.944444 | 0.777778/0.944444 | 0.833333 | 0.722222 | 1.000000 | 1.000000 |
| 893602 | 0.573333 | 0.533333 | 0.161290 | 0.133065 | 0.413333/0.533333 | 0.426667/0.600000 | 0.781086 | 0.798599 | 0.333333 | 0.481481 |
| 893603 | 0.613333 | 0.520000 | 0.206081 | 0.131757 | 0.466667/0.626667 | 0.360000/0.506667 | 0.714286 | 0.752022 | 0.526316 | 0.605263 |

固定 5000 的四个 dense fold（`893512/893513/893602/893603`）中，max candidate 的 joint 均低于 control，同时 WAIT false-release 均高于 control；本表未把 `893511/893601` 的结果外推为整体均值。由此记录的结果为 max candidate 未显示稳定 action benefit，故不采纳 max gate，不继续该 factorization 或 network variant；不访问 Q、不启用 PPO、不进入 formal qualification。上述数学段的 `-2`/25-candidate 计算仍只是组合公式的 toy property，不是经验根因证明。

## support inventory（2026-10-09）

support inventory 本身为只读检查，未启动新训练。按 held seed 记录 train 中 absent 的受影响 RELEASE 行 / held RELEASE 总数，以及 rare `≤5` 的 supported train labels held RELEASE / 总数：`893511 0/23, 0/23`；`893512 2/94, 12/94`；`893513 4/87, 14/87`；`893601 0/18, 0/18`；`893602 0/75, 4/75`；`893603 0/75, 5/75`。六 fold 合计 absent `6/372=1.61%`、rare `35/372=9.41%`；full train 计数为 `372 R/1886 W`，class12 nominal 为 `200/372`。这些 absent/rare 类别单独不能解释大多数 proposal errors；independent context/geometry generalization 仍未解决。

固定 5000 动作证据也仅是 teacher RELEASE 条件下的 proposal 分类：native control 的 class12 正确 `136/200`、non12 正确 `18/172`；max candidate 的 class12 正确 `146/200`、non12 正确 `11/172`。这些不是执行成功率；always12 reference `200/372` 不能当作策略。max 的收益集中在 nominal，不能改善 non-nominal，故拒绝该变体；类别未见率不足以单独解释错误，新的 independent traffic/midload 场景补覆盖也不保证修复。用户已授权按原协议逐回合执行：smoke `892872` → medium12_mixed `893604` → 完整终态核查 → dense24_fixed `893605` → 完整终态核查 → dense24_mixed `893606`，不增加预算或 seeds。

## supplemental smoke 终态（2026-10-09）

smoke `892872` 已消费并为 `SMOKE_OK/exit0`，输出 `/home/user/training_outputs/terminal/p3_v26_supplemental_20261009/smoke`；该 smoke 有意截断为 2 ticks，包含 6 raw rows、3 actor rows、3 forced rows、3 labels、3 resource rows 和 159D。prefix/labels/traffic exact、first-tick replay equal，安全/hash/negative-energy 异常均为 0；PIDs `3067886/3067887` 已退出。该 smoke 不代表完整业务回合；在该 smoke 快照时 3 个正式臂尚未启动，当前首个正式臂状态见下文。后续采集依原 teacher offline annotation 协议，学习、PPO 和 Q 仍禁。

## supplemental `train_01` 启动记录（2026-10-09）

用户已解除暂停并准许按原协议逐回合执行。`medium12_mixed` seed `893604` 的 `train_01` 于 `2026-10-09T10:11:01.823Z`（北京时间 `18:11:01.823`）启动，protocol 为 `p3_v26_medium_dense_supplemental_train_20261009_v1`，输出 `/home/user/training_outputs/terminal/p3_v26_supplemental_20261009/train_01`；配置为 12 requests、warm6、SOC `.55–.90`、`taskpriority=false`。wrapper/timeout/Python 为 `3068758/3068759/3068760`，cwd `/home/user/projects/terminal`。

初核时间 `2026-10-09T10:11:17.160Z` 观察到 75 steps、572 raw rows 和首条 90D raw row；最终 159D resource 重建当时尚未输出，不将该时点的 90D raw row 写成 schema failure。`result.json`、`rollout_result`、`annotation_progress` 尚未生成，exit 文件也不存在；当前不能称完成或安全全 0。launcher full-hash、preflight、no-repeat 检查均 passed。预算固定为 1800 steps、环境墙钟 3600s、离线标注墙钟 21600s、最多 2048 queries / 51200 RELEASE intents，外层 timeout 27000s。

heartbeat `p3`（`P3 补充采集逐回合监控`）已由 primary 创建并为 `ACTIVE`，每小时正常静默；旧 `p3-gate` 已删除，`p3-v2-5` 保持 `PAUSED`。seed `893604` 已消费，`893605/893606` 尚未启动；完整终态分析通过后才进入下一个，源码保持冻结。该任务是数据采集及随后离线 teacher 标注，包含采集器自身的 reset/teacher 标注流程；不写成“无 reset/teacher query”，这里只记录没有额外诊断 query。无学生训练、optimizer 更新或 PPO。
