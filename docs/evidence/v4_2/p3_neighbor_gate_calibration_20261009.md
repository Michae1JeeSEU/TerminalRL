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

该回合提供完整采集证据，但不能据此宣称泛化解决或 PPO/资格准入；`dense24_mixed` seed `893606` 已按原授权流程启动，启动时点及终态见下文。三个 supplemental 回合完成后统一覆盖 9-cycle 分析，再决定是否进行有界学习；无新学习/PPO，源码保持冻结，本轮未重跑既有 tests。旧 heartbeat `p3-dense-fixed-893605` 已删除。

## supplemental `train_03` 启动记录（2026-10-09）

`dense24_mixed` seed `893606` 已按原协议启动：24 requests、warm12、SOC `.55–.90`、`taskpriority=false`，输出 `/home/user/training_outputs/terminal/p3_v26_supplemental_20261009/train_03`。protocol 创建于 `2026-10-09T14:27:43.326556838Z`（北京时间 `22:27:43.326556838`）；wrapper/timeout/Python 为 `3090327/3090328/3090329`，三者 cwd 均为 `/home/user/projects/terminal`。launcher hash、preflight、duplicate 和前臂验证均通过。预算保持 1800 steps、环境墙钟 3600s、离线标注墙钟 21600s、最多 2048 queries / 51200 RELEASE intents、外层 timeout 27000s。

启动初读的 12 raw/0 step 仅为 pre-state。随后在 `2026-10-09T14:31:01Z` 观察到 181 steps、2833 raw；progress 为 175 ticks/2700 raw/73 selected、elapsed `188.12869s`、outcomes completed `2`。进程状态为 `S/S/R`、elapsed `3:19`；`root result`/exit 尚未生成，log 为空，终态安全/error 尚未生成，不能将该前缀写成全 0。`first_before`、`request`、`live`、`replay` 均已生成，live/replay SHA256 相同，为 `6451b65eef401ab01d2bdbc6bf9ebd61eac707824c8df1808cf92edcd59c0780`。

heartbeat `p3-dense-mixed-893606`（`P3 dense mixed 893606`）在启动阶段为 `ACTIVE`、每小时正常静默；旧 fixed heartbeat 已删除，`p3-v2-5` 保持 `PAUSED`。本段仅保留启动时点，train03 终态见下文。该回合是数据采集及随后离线 teacher 标注，不包含新学习或 PPO；源码保持冻结，本轮未重跑既有 tests。

## supplemental `train_03` 终态（2026-10-09）

`dense24_mixed` seed `893606` 的终态为 `COLLECTION_OK/exit0/error=null`，blockers `[]`；原 wrapper/timeout/Python `3090327/3090328/3090329` 与 child 均已退出。自然终态为 938 ticks、9966 raw rows（292 actor、9674 forced），292 selected/292 OK/resource159D，79R/213W；13 个 proposal 分布为 `(12:41,11:3,7:2,17:7,10:1,22:2,13:11,14:3,18:2,3:3,9:2,2:1,4:1)`。三接口 R/W 为 `o2m 25/91`、`m2i 31/25`、`i2f 23/97`；低 SOC 120、busy 280、fast-inner 115、after-retry 201，并观察到 31 次同 UID 同接口 `W→R`。

本地终态为 `19N/5E/0F/0D/0U`，5 次应急提交对应 5 次 touchdowns，45 次应急尝试中有 114 次应急安全拒绝；min SOC `0.1681304216`、min energy `252.1956323`，实际 hard separation/NMAC/collision/执行 hash 失配/负能量均为 0，negative_energy_budgets 为 0，最小实际/扫掠间距为 `81.0743767503m`。9 个 trajectory infeasible candidates 与 228 次 safety interventions 作为候选过滤计数保留，不能将本回合写成所有计数为 0。环境耗时 `572.3499320s`、标注耗时 `2840.5462680s`，约 56.88min，未延长预算。

teacher 与 trace 行为计数为 `57 R/R`、`22 R/W`、`3 W/R`、`210 W/W`；normal K3 ranks 1/2/3 分别为 `37/11/10`，保留 teacher、行为和执行分离。292 resource 均为 159D finite，traffic N×50 finite，N2–19 合计 4022 neighbor rows；mask/schema/decode/prefix/traffic/labels exact 与 first replay equal 均通过。两端关键文件 SHA 相同；292 labels 按文件名排序内容 manifest SHA256 为 `ffaf61a66a9decdbb545da10d5e9bcb1eefd5aaeea350c80ea76135eb985ca9d`；root SHA 为 `f529619741c1bb6ad38b6181d845d61993be3a2c0b7e947d5e080acd0950d09a`，resource SHA 为 `cd05568a2854e00d0d1ca07d55de52ed3c27b76ca78e093dca1ec92ce38de3bd`。

joint checks 为 `1530`，corrections `0`，fail-closed `0`。本轮三个 supplemental 回合均已完成，heartbeat `p3-dense-mixed-893606` 已删除，`p3-v2-5` 保持 `PAUSED`；当前无新活动采集或学习。九周期覆盖统计及下一项已选定的数据扩充协议见下文；本段记录的采集没有学生训练或 optimizer 更新。这些记录仅构成数据证据，不能宣称 P3 已解决或 PPO 准入；existing helper/产物 checks 已沿用，未重跑源码 tests。

## nine-cycle coverage 汇总（2026-10-09，只读聚合）

汇总 artifact 为 `.tmp/p3_nine_cycle_coverage_20261009.json`，标记为 read-only aggregation（no training/query/reset）。权威输入分为 old3：`p3_v25_neutral_course_v2_20261008/train_flow6_mixed`、`train_dense24_fixed`、`train_dense24_mixed`；mid3：`p3_v26_resources_20261009/train_01/02/03`；new3：`p3_v26_supplemental_20261009/train_01/02/03`。

| seed | group | actor | R/W | N/E |
|---:|---|---:|---:|---:|
| 893511 | old3 | 30 | 23/7 | 5/1 |
| 893512 | old3 | 721 | 94/627 | 20/4 |
| 893513 | old3 | 547 | 87/460 | 14/10 |
| 893601 | mid3 | 18 | 18/0 | 6/0 |
| 893602 | mid3 | 571 | 75/496 | 18/6 |
| 893603 | mid3 | 371 | 75/296 | 19/5 |
| 893604 | new3 | 57 | 37/20 | 12/0 |
| 893605 | new3 | 542 | 89/453 | 18/6 |
| 893606 | new3 | 292 | 79/213 | 19/5 |

九个 seed 合计 `3149 actor = 577R/2572W`，全部 `F/D/U=0`；old6 为 `2258 = 372R/1886W`，new3 为 `891 = 205R/686W`。proposal 支持仍为 20 类，class12 为 `306/577`，不能称类别全覆盖。leave-one-cycle-out 中 absent 为 `3/577 = 0.52%`，rare train count `1..5` 影响 `36/577 = 6.24%`；在相同 old6 held 集合上，加入 new3 前后 absent 为 `6/372→3/372`、rare 为 `35/372→25/372`，不把不同分母的汇总误写成提升。仍 absent 的 classes 为 held `893512` 的 `6`，以及 held `893513` 的 `1/24`，均为唯一标签；其余 held 无 absent。max neighbors 为 20，新 medium max 为 10。

new3 没有现有 pair witness；已有 witness 仅覆盖 old6，共 `127 states/317 queries`，因此按既定 witness 规则补 auxiliary 后才可进行相同辅助方法的数据比较。旧三组没有 resource JSONL 是原协议预期：`p3_v26_resource_inputs.load_resource_train` 从可信 snapshot 在内存重建额外 54D，并保留 105 prefix/traffic，不属于缺失标签或失败数据。新九组 loader 的真实全量接入验收仍待实现和 tests，不提前称为通过。

该汇总确认新增中负载与独立 dense 情境已进入数据覆盖，但 absent/rare 类别仍不足以单独解释旧的大量 proposal errors；后续按已预登记的固定网络与步数运行九 CV 检验，不承诺加数据即可修复。三 supplemental 回合本身已完整验收，Q 保持隔离、`PPO_ALLOWED=false`，没有新学习或 query。

## supplemental neighbor witness 数据扩充预登记（2026-10-09，未执行）

下一项为有界的数据扩充单因素对照，协议名 `p3_supplemental_train_neighbor_witness_20261009_v1`。目标是检验补充 independent context/traffic load 后是否出现稳定 action benefit；该预登记不保证有效，不改变资格门槛。模型保持 `IntentNeighborActor` 原 `first_feasible_union` 与 `native_pair_auxiliary`，原 teacher、safety、`159+N50` 输入、model seed `91911`、aux seed `91912`、BC+pair loss 权重 `1`、scene 权重 `.25`、Adam `3e-4`、batch `128`、auxiliary batch `64`、clip `.5`、marks `0/100/500/1000/2000/5000` 和每 fold `1200s` 上限全部冻结。

九个 parent cycle 只运行 native arm；既有六 fold native fixed5000 结果复用为 control，不重跑六 CV，不重复已拒绝的 max 变体。每个旧六 fold 完整比较 RELEASE recall、WAIT false-release、proposal top1/top3、joint、blocker 及 pair 安全拒绝识别；三个 new held 结果单独报告，不挑最好 mark。该阶段仅为 training development，`Q`、qualification、PPO 均禁用。

为三个 new held seed 按原 witness 规则补 auxiliary：每 seed、每 interface×gate 最多 4 个状态，按 tick 与数值 UID 等距选取，最多 72 states；这些不是新增独立回合。原 joint13 以及 teacher 首个可行或 `W` 时固定 joint1，均限制在原 teacher 已查 prefix 内；按 canonical plan 做逐 pair supervision，scene unknown 保持 unknown，不回填未知后缀。full 上限为 864 plans/1800s；先做三 interface smoke（最多 36 plans/120s）。输出分别为 `/home/user/training_outputs/terminal/p3_neighbor_supplemental_witness_20261009/smoke` 与 `/home/user/training_outputs/terminal/p3_neighbor_supplemental_witness_20261009/three_cycle`，outer timeout 为 180s/1900s；旧 317 queries 复用，不重跑。

约束保持为 0 reset/step、原 teacher label 不改、不回填未知后缀、不新增 online input。九 CV 输出为 `/home/user/training_outputs/terminal/p3_neighbor_nine_cycle_cv_20261009/nine_cycle`，outer timeout `11400s = 9×1200+600`，仅 native 一臂。执行顺序预登记为：本地定向 tests、新增 route synthetic（9 parent×2 update）smoke、真实 loader 只读 input 检查；通过后做 Syncthing/hash/server checks，再做真实 auxiliary smoke、full auxiliary 完整验收，最后单次运行九 CV。

源码接入与定向验证已完成；本预登记尚未 launch 任何 query、采集或学习。若该因素仍无稳定收益，后续是否继续由 primary 根据具体结果决定，不自动扩 epoch 或 network variant。既有三回合产物、9-cycle 覆盖统计及本协议均不构成 P3 已解决或 PPO 准入证据。

## 源码接入与验证终态（2026-10-09 UTC / 2026-10-10 北京时间）

本轮完成四个既有模块和一个新增测试文件的 supplemental 接入：`terminal_marl/training/p3_v26_train_dataset.py` 增加 `load_supplemental_train` 与 `load_nine_cycle_train`；`p3_neighbor_witness_data.py` 增加 supplemental protocol、固定选择和 smoke/full budget；`p3_neighbor_witness_dataset.py` 接受双 protocol 并提供 `combine_witness_targets`；`terminal_marl/scripts/diagnose_p3_intent_neighbor.py` 增加 supplemental train/witness CLI route，固定 `first_feasible_union` 与 `native_pair_auxiliary`。旧六回合入口、159+N50、teacher labels、physics/safety/model 核心和预登记预算保持不变。

本地定向回归为 supplemental data/witness 21 tests、`3.94s`、exit0；`tests/test_intent_neighbor_actor.py` 独立回归 9 tests、`2.60s`、exit0。server route check 为 `28 passed in 1.65s`、exit0。synthetic supplemental route 使用 9 个 parent seed、native-only、2 updates，结果为 `COMPLETE/exit0/error=null`；两端同步 hash 为 result `d078ddc2395a21a7ec145cb270f49f9640d715ca647fd0c6fb8a6a5cf4ab9f08`、protocol `3f14a36e4f02af64f3a89cd9948a517c75f0301b445d6094c4004f9f3de58187`。

真实九周期 loader 的只读验收 artifact 为 `.tmp/p3_nine_cycle_loader_check_20261009.json`，状态 `PASS`，SHA256 `acd7a7d7a3c3dda6c4792a5c3228432d5c73042cdc5a2a6f6bd6f91a70e3c262`。九组共 `3149 actor = 577R/2572W`，`F/D/U=0`；shape 为原始 `(3149,90)`、新 ego `(3149,159)`、traffic `(3149,20,50)`、mask `(3149,20)`。labels 按 decision ID 保持，old6 的 105D prefix/traffic/mask 保持，三 supplemental 输入选中 `65` 个 state，按 seed 为 `893604:17`、`893605:24`、`893606:24`，resource/context 的 `159+N×50` 与 schema/decode 检查通过。

旧 witness artifact 仍为 `127` records、`121` 个含 destination 的 target states、`317` destinations/queries；其余 6 条为 global block，原因计数 `final_capacity=5`、`lz_unavailable=1`，这些记录的原计划调用为 0。首次 checker FAIL 已保留在 `.tmp/p3_nine_cycle_loader_check_20261009_first_failure.json`；该 FAIL 仅反映计数语义混同，后续只修 checker，未改官方 `atol=1e-7`/`rtol=1e-6`，也未改数据或核心代码。

源码接入与只读 loader 验收已完成；supplemental auxiliary smoke/full 已完成且通过，`Q`、qualification 和 PPO 保持禁用。此前自动审批曾拒绝长时新实验，SSH 当时未执行；用户随后明确批准唯一固定九 CV，当前 run 已启动并仍在运行。不重跑旧六 fold control，不重复 max-candidate 变体，不增加 seeds 或预算。

## supplemental neighbor witness 终态与九 CV 启动核对（2026-10-09 UTC / 2026-10-10 北京时间）

已执行的 supplemental witness smoke 输出为 `/home/user/training_outputs/terminal/p3_neighbor_supplemental_witness_20261009/smoke`，状态 `SMOKE_OK/exit0`，包含 `3` states、`8` plans，耗时 `1.5839914884s`；witness SHA256 为 `8e42f02884332f3a7fe3537cc6372eea2691d5f71b98aa86ba0b9dff7a61c9b8`。该 smoke 已验收，禁止重跑。

full 输出为 `/home/user/training_outputs/terminal/p3_neighbor_supplemental_witness_20261009/three_cycle`，状态 `COMPLETE/exit0/error=null`，包含 `65` states、`156` plans，耗时 `130.6318327244s`。父进程、wrapper 和 child 均已退出；server/local 三个 full 文件 SHA256 一致：`protocol.json` `1b77d8f64f3dfbf4301b53f4c346d454f87607dd1c7db399d944eaca3636521f`、`result.json` `ce38c84739c1da577d2a3676d20e37417a9b718d180c86eb69bb1f86f72c991c`、`witnesses.jsonl` `e543020f06583ae98b0e44df7ee3c73507c76c3d622abc1be043e3c22e4dcc60`。

旧 witness 的 `127` records、`121` 个 target states、`317` queries 与 supplemental 的 `65` selected states、`63` 有目标 records、`156` plans 合并后为 `473 queries`、`184 target states`；known pairs `4555`、blocked pairs `394`、scene blocked `27`。6 个旧 global reject 仍为 `final_capacity=5`、`lz_unavailable=1`，原计划调用为 0，不计作缺标。原 teacher/labels/UID/candidate 与官方 loader 保持一致，9-fold actor 与 auxiliary split 使用同一 parent 隔离规则；分析文件为 `.tmp/p3_nine_cycle_witness_analysis_20261009.json`，SHA256 `206f0b8c3fa74059f4c91fff1f6bc726d29927e8e107922523216e4a70a137bd`。

此前九 CV 尚未启动，自动审批拒绝了长时新实验；该历史保留。用户于 2026-10-10 北京时间明确批准后，唯一 run 已启动，当前不将前缀解释为终态。后续核对固定的 `first_feasible_union`、`native_pair_auxiliary`、model seed `91911`、aux seed `91912`、9-fold/5000 updates/1200s per fold、outer timeout `11400s`；不访问 Q/PPO/formal，不追加 epochs、seeds 或预算。

## approved P3 nine-cycle comparison launch（2026-10-09 UTC / 2026-10-10 北京时间）

用户于 2026-10-10 北京时间明确批准唯一固定九 CV，primary 已接受该授权。任务于 UTC `2026-10-09T16:46:37Z`（北京时间 `2026-10-10 00:46:37`）启动，输出 `/home/user/training_outputs/terminal/p3_neighbor_nine_cycle_cv_20261009/nine_cycle`；wrapper/timeout/Python 为 `3099773/3099774/3099775`，cwd 为 `/home/user/projects/terminal`。运行导入源码 commit 为 `e496bc637ca1d9e0a9a55967958115ed597e98ed`，启动前文档 HEAD 为 `26ee3ce4b42bb441424d8c3816d42cbed3560b1c`，首 protocol SHA256 为 `994882bd2909391f23becb5abf01dba319dae421e9d9ece67aa6c0e6277db4f3`。source SHA、无重复进程、输入目录和前臂验证均通过。

固定输入为 9 个 whole-parent seeds `893511/893512/893513/893601/893602/893603/893604/893605/893606`，共 `3149 actor = 577R/2572W`，复用 `473` auxiliary queries。route 为 `first_feasible_union` + `native_pair_auxiliary`，model seed `91911`、aux seed `91912`，每 fold `5000` updates / `1200s`，outer timeout `11400s`，仅 native 一臂。启动时 root 为 `RUNNING/error=null`；首 fold `893511` train/held 为 `3119/30`，aux train/held 为 `414/59`。随后前缀核对中首 fold 已到 marks `[0,100,500]`，root 仍 `RUNNING/error=null`；该前缀是非终态，不作训练完成或效果解释。

启动时 server 观察到 `28` logical CPU 与约 `25 GiB` 空闲 RAM。用户要求尽量使用多核 CPU 并行可并行实验；本次已启动的单次 CV 保持冻结、不重启，后续独立 fold/arm/只读检查按 CPU/RAM 资源并行，存在依赖的验收任务串行。该 development CV 不构成 qualification，`Q` 与 old control 隔离，`PPO_ALLOWED=false`；不访问新 Q、不追加 seeds/epochs/预算，不把候选安全 label 作为 online observation。

## 九周期 CV 终态（2026-10-10，北京时间；newly_run）

九周期 development CV 已 `COMPLETE/exit0/error=null`。输出为 `/home/user/training_outputs/terminal/p3_neighbor_nine_cycle_cv_20261009/nine_cycle`，protocol/result SHA256 分别为 `994882bd2909391f23becb5abf01dba319dae421e9d9ece67aa6c0e6277db4f3` 与 `ab1e7d7c38c52d32393d650dc389e23e77eb844eedaa760798c698036af1d76e`。固定 `first_feasible_union`、`native_pair_auxiliary`、model seed `91911`、aux seed `91912`，9 个 fold 均完成 marks `0/100/500/1000/2000/5000`；折次耗时合计 `7225.695s`，最大 `818.867s`，均低于每 fold `1200s` 与 outer `11400s` 上限。无 Q、teacher query 或 checkpoint；父子进程已退出，monitor heartbeat 已删除，`p3-v2-5` 仍为 `PAUSED`。

下表为九个 held fold 的 validation 结果。`WAIT false-release` 是实际 WAIT 行被预测为 RELEASE 的比例；`pair blocked recall` 与 `blocker top1` 来自 native pair auxiliary。`auxiliary loss` 在聚合摘要中按 query 加权，这里保留每 fold 的 validation 值。

| held seed | rows | RELEASE recall | WAIT false-release | proposal top1 | proposal top3 | joint | blocker top1 | pair blocked recall | auxiliary loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 893511 | 30 | 1.000000 | 1.000000 | 0.695652 | 0.956522 | 0.533333 | 0.771429 | 0.418605 | 1.127700 |
| 893512 | 721 | 0.542553 | 0.084530 | 0.414894 | 0.595745 | 0.837725 | 0.625000 | 0.660000 | 0.877909 |
| 893513 | 547 | 0.574713 | 0.141304 | 0.321839 | 0.517241 | 0.767824 | 0.645161 | 0.580000 | 0.901225 |
| 893601 | 18 | 1.000000 | — | 0.666667 | 0.944444 | 0.666667 | 1.000000 | 1.000000 | 0.147322 |
| 893602 | 571 | 0.426667 | 0.116935 | 0.440000 | 0.600000 | 0.812609 | 0.555556 | 0.918919 | 0.734031 |
| 893603 | 371 | 0.706667 | 0.202703 | 0.480000 | 0.693333 | 0.719677 | 0.631579 | 0.647887 | 0.727836 |
| 893604 | 57 | 0.702703 | 0.450000 | 0.594595 | 0.729730 | 0.473684 | 0.714286 | 0.600000 | 1.137350 |
| 893605 | 542 | 0.573034 | 0.165563 | 0.449438 | 0.640449 | 0.756458 | 0.666667 | 0.727273 | 0.641787 |
| 893606 | 292 | 0.379747 | 0.061033 | 0.354430 | 0.569620 | 0.756849 | 0.794118 | 0.735849 | 0.899846 |

### old6 同 held 集合的配对聚合

旧六回合 native control 与九周期结果在相同 old6 held 集合上比较；支持为 `2258 rows = 372R/1886W`。右侧 delta 为 new nine-cycle result 减 old6 control。这里的 `auxiliary loss` 是 query-weighted 摘要，其他指标按各自 stored support 聚合。

| validation metric | old6 native control | new nine-CV on old6 held folds | delta |
|---|---:|---:|---:|
| RELEASE recall | 0.567204 (56.7204%) | 0.610215 (61.0215%) | +0.043011 |
| WAIT false-release | 0.126723 (12.6723%) | 0.128844 (12.8844%) | +0.002121 |
| proposal top1 | 0.413978 (41.3978%) | 0.440860 (44.0860%) | +0.026882 |
| proposal top3 | 0.586022 (58.6022%) | 0.637097 (63.7097%) | +0.051075 |
| joint | 0.786980 (78.6980%) | 0.789637 (78.9637%) | +0.002657 |
| blocker top1 | 0.622093 (62.2093%) | 0.668605 (66.8605%) | +0.046512 |
| pair blocked recall | 0.592308 (59.2308%) | 0.650000 (65.0000%) | +0.057692 |
| auxiliary loss (query-weighted) | 1.236169 | 0.798130 | −0.438039 |

### new3 独立聚合与九周期总支持

三个 supplemental held fold 必须单独报告，支持为 `891 rows = 205R/686W`；这些数值不并入上表的 old6 配对 delta。new3 的 micro validation 为 RELEASE recall `0.521951` (`52.1951%`)、WAIT false-release `0.141399` (`14.1399%`)、proposal top1 `0.439024` (`43.9024%`)、top3 `0.629268` (`62.9268%`)、joint `0.738496` (`73.8496%`)、blocker top1 `0.724138`、pair blocked recall `0.716418`；auxiliary loss `0.856660` 为 query-weighted 摘要。

九周期全部 native fold 的描述性支持为 `3149 rows = 577R/2572W`；对应 micro RELEASE recall `0.578856`、WAIT false-release `0.132193`、proposal top1 `0.440208`、top3 `0.634315`、joint `0.775167`、blocker top1 `0.687259`、pair blocked recall `0.672589`，auxiliary loss `0.817434`（query-weighted）。该 development CV 只记录这次固定比较，不能作为资格或 PPO 准入证据。后续本轮已完成 prefix coverage 的 prepare-only 预登记，详见 [prefix coverage 修复预登记](p3_neighbor_prefix_coverage_repair_20261010.md)；真实 prefix smoke/full/CV 尚未启动。
