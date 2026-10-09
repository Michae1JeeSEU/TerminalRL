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

初核时间 `2026-10-09T10:11:17.160Z` 当时观察到 75 steps、572 raw rows 和首条 90D raw row；最终 159D resource 重建尚未输出，不将该时点的 90D raw row 写成 schema failure。该初核时点的 `result.json`、`rollout_result`、`annotation_progress` 和 exit 文件均尚未生成；随后终态见下文。launcher full-hash、preflight、no-repeat 检查均 passed。预算固定为 1800 steps、环境墙钟 3600s、离线标注墙钟 21600s、最多 2048 queries / 51200 RELEASE intents，外层 timeout 27000s。

train_01 运行及其终态记录时 heartbeat `p3`（`P3 补充采集逐回合监控`）为 `ACTIVE`、每小时正常静默；train_01 终态后该 heartbeat 已删除，旧 `p3-gate` 已删除，`p3-v2-5` 保持 `PAUSED`。seed `893604` 已消费，后续 seed 状态见下文，源码保持冻结。该任务是数据采集及随后离线 teacher 标注，包含采集器自身的 reset/teacher 标注流程；不写成“无 reset/teacher query”，这里只记录没有额外诊断 query。无学生训练、optimizer 更新或 PPO。

## supplemental `train_01` 终态（2026-10-09）

`train_01` seed `893604` 为 `COLLECTION_OK/exit0`；原 wrapper/timeout/Python `3068758/3068759/3068760` 及 child 均已退出。全回合为 691 ticks、3795 raw rows（57 actor、3738 forced），57 selected/57 OK/resource159D，37R/20W；7 个 proposal 分布为 `(12:26,14:3,7:1,13:3,17:2,9:1,2:1)`。三接口 R/W 为 `o2m 12/19`、`m2i 12/1`、`i2f 13/0`；低 SOC 11、busy 51、fast-inner 11、after-retry 25，并观察到 6 次同 UID 同接口 `W→R`。

终态为 `12N/0E/0F/0D/0U`，无 censor/error；min SOC `0.3968737386`、min energy `595.3106078`，实际安全/hash/负能量/应急计数均为 0，最小实际/扫掠间距 `82.7146226m`。其中 3 个轨迹不可行 candidates 与 60 safety interventions 必须保留为候选过滤计数，不能将本回合写成全 0。环境耗时 `105.1907449s`、标注耗时 `380.1467585s`，未延长预算。

teacher 与实际行为计数为 `32 R/R`、`5 R/W`、`20 W/W`；5 个 teacher 动作均为 non-nominal timing，保留原 label、行为和执行三者分离，不能写成标签被实际否决。105 prefix/labels/traffic exact、first replay equal，full helper 已成功；分析文件为 `.tmp/p3_v26_supplemental_train01_analysis_20261009.json`。result SHA256 为 `18c8b092a79d83ec69bf736e196c90c13cac801f3a0b520adc3f893bcee7b61f`，resource SHA256 为 `c0c59aabda9292241695ff465d0280904442b1918c0e2aaff0fa24ceb8219bf9`。

全量核验还确认 train01 两端 result（完整 hash 末尾 `...7b61f`）、`annotation_result`、57 标签聚合、105 context、159 resource 与 first-replay 的 SHA 一致；57 resource 全为 159D finite，traffic N×50 全 finite，N1–10 合计 394 邻机 rows，mask 长度/hash 匹配，105 prefix/traffic/mask 不变且 57 decode 成功。JSONL 不存 UID；snapshot 为 trace-only，旧/新 UID 一致，UID 未作为输入。N50 表示每邻机 50 维，与 window50 无关；worker 早期误解已由全量复核纠正，不作为数据失败。标签聚合 SHA256 为 `4d3a4929b1d5969cf511d2c0fc75db6f37561f6089041693c081559729ca48a4`，annotation SHA256 为 `d241208db30d2a27e5b00fa4b59eec0d70593e2c2133914c44410df913bcee65`。

此前一次错误参数调用为 `exit1` 且未产生训练产物变化；有效 full-analysis case 为 `train_medium12_mixed`，正确调用为 `exit0/FULL_ANALYSIS_COMPLETE`。primary 判断为原协议采集完整，可继续 `893605`；新增中负载独立支持仍 majority nominal，不能由单回合推断 student/gate 已修复，也不在单回合上拟合。既有 `55 tests/1 warning/exit0` 仍是 earlier preflight，本次未重跑。

## supplemental `train_02` 启动记录（2026-10-09）

一次 `train_02` launcher 已 `exit0`，preflight 的 `duplicates=[]` 与源码 hash guards 全部通过。protocol 创建于 `2026-10-09T11:12:46.973932185Z`（北京时间 `19:12:46.973932185`）。`dense24_fixed` seed `893605`（24 requests、warm12、SOC `.8`、`taskpriority=false`）输出 `/home/user/training_outputs/terminal/p3_v26_supplemental_20261009/train_02`；wrapper/timeout/Python 为 `3072599/3072600/3072601`，cwd `/home/user/projects/terminal`。预算固定为环境墙钟 3600s、离线标注墙钟 21600s、steps 1800、最多 2048 queries / 51200 RELEASE intents、总 timeout 27000s。

初核 `2026-10-09T11:13:25Z`（elapsed 约 39s）观察到 progress buffer 的 1 tick、12 raw、12 selected、`36.627s`，实际 rows `120`、steps `9`；progress buffer 与行数刷新时间不同，不将其写成矛盾。`result.json`、`collection_result`、exit 尚未生成，log 为空；当前不能假定安全全 0。新 heartbeat `p3-dense-fixed-893605`（`P3 dense fixed 893605`）为 `ACTIVE`、每小时正常静默；medium 旧 `p3` 已删除，`p3-v2-5` 保持 `PAUSED`。本节仅保留启动时点；server exit/SHA 已在下方完整终态核验，`dense24_mixed`/893606 的后续启动记录见下文。后续有效 helper case 名为 `train_dense24_fixed` / `train_dense24_mixed`，不将其写成 `train_02` / `train_03`。

## supplemental `train_02` 终态与完整验收（2026-10-09）

`dense24_fixed` seed `893605` 的终态为 `COLLECTION_OK/exit0/error=null`，blockers `[]`；原 wrapper/timeout/Python `3072599/3072600/3072601` 与 child 均已退出。全回合为 1044 ticks、12401 raw rows（542 actor、11859 forced），542 selected/542 OK/resource159D，89R/453W；15 个 proposal 分布为 `(12:39,11:1,7:4,4:2,22:3,17:6,14:5,13:14,3:4,2:2,8:2,9:4,15:1,19:1,18:1)`。三接口 R/W 为 `o2m 37/174`、`m2i 29/117`、`i2f 23/162`；低 SOC 179、busy 530、fast-inner 179、after-retry 385，并观察到 39 次同 UID 同接口 `W→R`。

本地终态为 `18N/6E/0F/0D/0U`，包含 6 committed touchdowns、78 次应急尝试、132 次应急安全阻塞和 8 次 LZ 阻塞；不能将本回合写成全 0。实际 hard separation/NMAC/collision/执行 hash mismatch/negative energy 均为 0。min SOC `0.2423020905`、min energy `363.4531358`，最小实际/扫掠间距分别为 `80.1433798174m` / `80.1426289672m`。21 个候选不可行和 261 次 safety interventions 作为候选过滤计数保留，与实际安全状态分开。环境耗时 `1170.0690524s`、标注耗时 `7068.0913217s`，约 2.29h，未延长预算。

teacher 与 trace 行为计数为 `56 R/R`、`33 R/W`、`3 W/R`、`450 W/W`；normal K3 ranks 1/2/3 分别为 `34/12/11`，其余 trace 不混入 proposal 准确率；teacher 差异为 36，2 条 unranked release trace 不混入 normal K3，不由差异直接改写 teacher 标签。完整验收确认两端 protocol/result/annotation/context/resource/first-live/first-replay 逐文件 manifest 差异为 0；全部 542 resource 为 159D finite、traffic N×50 finite，N1–20 合计 7893 neighbor rows，mask 与 trace UID 对应，542 decode pass，105 prefix/traffic/labels exact，first replay equal。标签按文件名排序内容聚合 SHA256 为 `0f4b1500e3360bddf22ec98907cb7e785895a0f6ae85e1e6b36589089e86fae6`；root SHA 为 `cb8f618832b2ce64df97b4c08b38674b427cba3dd03725077df8cab3aefe62c5`，resource SHA 为 `6e3281e84c49925726c322aa62a366e7784ba0a1f070646bc6df5491694842b8`。

本地 snapshot 重建的严格 `array_equal` 对照曾见最大 `1.42e-16` 差异；resource/context JSON 精确相同，按既有官方 `atol=1e-7`/`rtol=1e-6` 的 542 条核验全部通过。未改容差、数据或源码，不将该差异写成实验错误。

该回合提供完整采集证据，但不能据此宣称泛化解决或 PPO/资格准入；`dense24_mixed` seed `893606` 已按原授权流程启动，当前启动状态见下文，终态尚未生成。三个 supplemental 回合完成后统一覆盖 9-cycle 分析，再决定是否进行有界学习；无新学习/PPO，源码保持冻结，本轮未重跑既有 tests。旧 heartbeat `p3-dense-fixed-893605` 已删除。

## supplemental `train_03` 启动记录（2026-10-09）

`dense24_mixed` seed `893606` 已按原协议启动：24 requests、warm12、SOC `.55–.90`、`taskpriority=false`，输出 `/home/user/training_outputs/terminal/p3_v26_supplemental_20261009/train_03`。protocol 创建于 `2026-10-09T14:27:43.326556838Z`（北京时间 `22:27:43.326556838`）；wrapper/timeout/Python 为 `3090327/3090328/3090329`，三者 cwd 均为 `/home/user/projects/terminal`。launcher hash、preflight、duplicate 和前臂验证均通过。预算保持 1800 steps、环境墙钟 3600s、离线标注墙钟 21600s、最多 2048 queries / 51200 RELEASE intents、外层 timeout 27000s。

启动初读的 12 raw/0 step 仅为 pre-state。随后在 `2026-10-09T14:31:01Z` 观察到 181 steps、2833 raw；progress 为 175 ticks/2700 raw/73 selected、elapsed `188.12869s`、outcomes completed `2`。进程状态为 `S/S/R`、elapsed `3:19`；`root result`/exit 尚未生成，log 为空，终态安全/error 尚未生成，不能将该前缀写成全 0。`first_before`、`request`、`live`、`replay` 均已生成，live/replay SHA256 相同，为 `6451b65eef401ab01d2bdbc6bf9ebd61eac707824c8df1808cf92edcd59c0780`。

heartbeat `p3-dense-mixed-893606`（`P3 dense mixed 893606`）为 `ACTIVE`、每小时正常静默；旧 fixed heartbeat 已删除，`p3-v2-5` 保持 `PAUSED`。该回合是数据采集及随后离线 teacher 标注，不包含新学习或 PPO；三个 supplemental 回合完成后统一覆盖 9-cycle 分析，源码保持冻结，本轮未重跑既有 tests。
