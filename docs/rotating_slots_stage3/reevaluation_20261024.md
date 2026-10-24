# Stage3 冻结模型编码优化重评估（2026-10-24）

状态：`COMPLETE`，20/20 个评价 job 均完成且退出 0；两闭环 arm 的共同完成边界和独立终态均为 `60 sim s`，terminal failed/invalid 均为 false。新 ledger 已暂停，`active_phase=null`。本轮仅执行已授权的一次有界在线重评估，复用原模型、标准化器、split、实际 seed、物理初态、安全规则和规划预算，不重新训练、不重新标注、不重新测量 heldout 排名信号，旧 campaign 保持只读。

编码相关局部开销确有下降：L 闭环 feature + inference 从 `77.671627187 s` 降至 `46.163317820 s`（约 `-40.566%`）。但 L 完整在线 wall 只比自身旧值下降约 `2.8773%`，本轮仍比 H 慢约 `3.8339%`，原闭环约 `13.23%` 的相对优势未保留；同状态也未观察到有意义的完整在线加速。质量与安全观测有好有坏，不能称为质量完全不变或整体在线优势已建立。整体保持 `ONLINE_BENEFIT_NOT_ESTABLISHED`；本轮是单个短时成对 pilot，不形成普遍或因果收益结论。

## 运行身份与状态

| 项目 | 本轮实际记录 |
|---|---|
| 实现提交（已推送） | `3c07585014e1b4b1ee401b5645dd56968376e58c` |
| 实际执行 source hash | `794765f57c4584fa34d2eafdba5f21b893763acc6a4c481d90c5bf421b6e5aee` |
| 源码冻结 | 76 个文件，冻结时源码 diff 为空 |
| 本地冻结目录 | `D:\7_codespace\terminal\artifacts\rotating_slots_stage3_20261024\encoding_reevaluation_source` |
| 旧输出（只读） | `/home/user/training_outputs/terminal/stage3_learning_v1_0_20261023` |
| 新输出 | `/home/user/training_outputs/terminal/stage3_encoding_reevaluation_20261024` |
| 新输出本地同步目录 | `D:\7_codespace\training_outputs\terminal\stage3_encoding_reevaluation_20261024` |
| 正式启动 | `2026-10-24T14:25:52.645012Z`（北京时间 2026-10-24 22:25:52.645012） |
| 启动 wrapper / CLI PID | `212297` / `212308` |
| 初次有效启动证据 | `s3r21_reference_full` 子进程 `212311` 已完成并发布 result；随后 heuristic 子进程为 `212423` |
| 正式结束 / CLI 退出码 | `2026-10-24T14:53:39.844975Z` / `0` |
| 最终评价与预算状态 | 20/20 job `completed` 且 exit 0；evaluation `1665.095692873001 / 3600 s`，ledger 已暂停、active phase 为空 |

上述启动 PID 是历史记录。最终状态依据 `evaluation_exit.json`、全部 job receipt、比较文件和独立 terminal observation 核对；不是从启动记录或进度百分比推定。replay 离线校验已完成，人工视觉验收仍为 `HUMAN_PENDING`。

`run_binding.generated.json` 的原 fingerprint 被明确标为 `frozen_model_contract`，仅用于原有模型兼容性检查。它保留模型训练时的源码身份；实际执行提交、source hash、文件清单及环境另存于 `execution_provenance` 和本轮 `source_freeze.json`。结果、闭环边界及 replay `code_commit` 使用本轮实际执行身份，不能把旧 fingerprint 当成本轮源码来源。

| 冻结输入 | SHA256 / 身份 |
|---|---|
| `model/model.pt` | `b565c2a98af7f30c5b8179e823a6447bb1ccecebcbf4da2dfdcb44679ce85969` |
| `model/feature_normalizer.json` | `007de6c700959a8cadc0397e8896f279aeca0070fe12bbac5441c3c5b2e63491` |
| split manifest ID | `84332c2ed23717ba61bb27857e320ab9e04008192b469e7bcc28b4613cdba6e0` |
| split manifest SHA256 | `76d6fe0f1b37baf09e7ed30fd2a4e019ff1ea0955ee8d1ee0830ea3558a75945` |
| 旧模型 contract 的源码提交 | `bcf4703d56a1fca58b14a08806ffd0910479f866` |
| 闭环 root / 实际 seed | `s3r23` / `7324` |
| 闭环初态 hash | `d4a0513ec327a7bc77d33e300d1401194a5a057bf9e1229b3836e845fcfcca79` |

模型文件、split、heldout signal、六个选定 root 的初始 checkpoint/state/manifest 均按字节复制。checkpoint identity 保留，root 仅改写文件路径；24-root 索引保留原顺序，仅物化本轮使用的六个 root，不复制大体积 raw 流。旧摘要及来源记录保存在新输出的 `prior_evidence/`。heldout signal 明确作为先前证据复用，不产生第二次 heldout 测量。

## 冻结协议与验证证据

沿用 `run_evaluation`：先执行 18 个同状态 job，case 顺序为 `s3r21,s3r22,s3r23,s3r24,s3r17,s3r18`，各有 reference/full、heuristic/reduced、learned/reduced 三 arm，按既有规则轮转顺序，每个 job 上限 `90 s`。随后在 `s3r23` 上串行执行 heuristic、learned 两 arm，目标为 `60 sim s`，保留原有等额 wall 分配与每 `5 sim s` 完成边界。使用同一物理 CPU `12`、每 arm 独立冷进程及单数值线程。

新 ledger 保留原 `3600 s` evaluation 上限、`900 s` preflight/tests 上限及其他 campaign 上限，不转移阶段剩余额度。prepare 拒绝任何非空新输出；tests/evaluate 各有一次尝试 guard，成功或异常后均暂停 ledger，不自动重试或重置预算。prepare 检查源码清单、不可变输入、checkpoint 恢复、物理/配置/合同/协议一致性、依赖版本、线程环境、CPU12 可用性和单 worker CP probe；evaluate 再查资源。

| 检查 | 已观测结果 |
|---|---|
| 本地指定项目 Python 回归 | `110 PASS`；另 2 项因本地缺少 ORTools 未通过 |
| 上述 2 项服务器补验 | `2 PASS`，`0.35 s` |
| 服务器 prepare | 退出码 `0` |
| 服务器原 targeted gate | `69 PASS`，S3-T01 至 S3-T12 共 12 项映射检查均 `PASS` |
| 服务器本轮 reevaluation + encoding 检查 | `37 PASS` |
| 服务器自动 gate 合计 | `106` 个唯一测试，全通过，无 failure/skip；tests CLI 退出码 `0` |

早先误用 Anaconda 环境的本地运行出现环境相关错误，未作为成功 gate。以上分别记录本地、服务器补验和本轮服务器 gate，不把不同批次重复相加成新的唯一测试数。自动检查也不代替人工视觉验收。

## 同状态比较（18/18 完成）

H 表示 `heuristic_reduced`，L 表示 `learned_reduced`，R 表示 `reference_full`。本轮 `same_state_comparison.json` 中 18 个 job 均为 `COMPLETE`、`certified=true`、CPU 为 `[12]`。下表为各 case 的 `costs.full_online_wall_s`，单位秒，显示到 9 位小数；完整值保留在原 JSON。该边界包含 job 内 setup/model load 至完成规划，不能只比较编码或推理耗时。

| root | 旧 R | 旧 H | 旧 L | 新 R | 新 H | 新 L |
|---|---:|---:|---:|---:|---:|---:|
| s3r21 | 32.571786950 | 27.980428287 | 28.892941128 | 35.909202304 | 27.733290260 | 31.975179461 |
| s3r22 | 32.320493835 | 26.168286851 | 29.040347943 | 32.482088647 | 25.944027785 | 25.732828196 |
| s3r23 | 31.808662713 | 22.195068064 | 22.858878487 | 34.356853251 | 26.580686895 | 24.838661288 |
| s3r24 | 36.480456376 | 33.309496002 | 34.532590339 | 35.015681195 | 30.582760367 | 30.904076447 |
| s3r17 | 31.426114529 | 29.459108193 | 28.679988308 | 32.687801561 | 27.187740273 | 27.935509716 |
| s3r18 | 33.253811111 | 27.516944311 | 28.220205744 | 32.798111063 | 31.634400000 | 30.576802998 |
| 合计 | 197.861325514 | 166.629331708 | 172.224951949 | 203.249738021 | 169.662905580 | 171.963058106 |

| 六 case 汇总 | 旧 R | 旧 H | 旧 L | 新 R/H/L |
|---|---:|---:|---:|---|
| 物理 attempts | 991 | 449 | 438 | 1125 / 618 / 589 |
| descriptor queries | 1303 | 539 | 522 | 1676 / 773 / 719 |
| planning deadline miss / timeout / fallback | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 | 三 arm 均 6 / 0 / 0 |
| planning P50（s） | 32.2928789 | 27.4775258 | 27.9451309 | 32.766855938 / 27.162084161 / 27.355638310 |
| planning P95 = max（s） | 36.4415418 | 33.2711751 | 33.7943427 | 35.877636123 / 31.603357682 / 31.389966288 |

新 L 的完整在线 wall 合计为 `171.96305810601916 s`，相对旧 L 的 `172.224951949 s` 为 `-0.15206498%`；新 H 为 `169.66290558036417 s`，相对旧 H 为 `+1.82055214%`。本轮新 L 相对新 H 为 `+1.35571916%`，同状态未观察到有意义的完整在线加速。新 H/L attempts 为 `618/589`、queries 为 `773/719`，均高于原 H/L 的 `449/438` 和 `539/522`；完整 wall 与搜索工作量必须分开报告。

旧 L 对旧 H 的完整目标向量比较为 `4 平 / 2 差 / 0 好`：`s3r17` 首个差异在第三维 risk，H/L 为 `377.7784444 / 2529.5966389`；`s3r24` 首个差异在第四维 Ehat，H/L 为 `990334714.4 / 992387792.6`。本轮新 L 对新 H 为 `2 好 / 1 平 / 3 差`，逐 case 首个差异如下：

| root | 新 L 相对新 H | 首个不同维度 | 新 H / 新 L 该维度值 |
|---|---|---|---|
| s3r21 | 更差 | 第三维 risk | 30887.594388889 / 34095.664777778 |
| s3r22 | 更好 | 第四维 Ehat | 352679060.20000005 / 349002666.90000004 |
| s3r23 | 更好 | 第四维 Ehat | 578018893.5 / 551212714.9 |
| s3r24 | 更差 | 第三维 risk | 46120.539972222 / 48689.637166667 |
| s3r17 | 完全相同 | 无 | 全五维相同 |
| s3r18 | 更差 | 第三维 risk | 29549.784972222 / 32526.821055556 |

各新 H/L 配对的前两维相同。目标使用原 Stage2 字典序，后续维度下降不能抵消前面维度劣化，本轮同状态质量非劣尚未成立。各 arm 新对旧的完整向量比较为 H `4 好 / 1 平 / 1 差`、L `5 好 / 0 平 / 1 差`、R `5 好 / 0 平 / 1 差`。H/L 共同变差 case 为 `s3r24`：原 risk `32538.0338333333` 分别变为 `46120.5399722222 / 48689.6371666667`；R 的变差 case 为 `s3r18`，risk 从 `15925.0015555556` 变为 `16629.8303611111`。

L 的六 case feature + inference 合计从 `2.149295021 s` 降至 `1.895515758 s`，但 CP 从 `38.946706223 s` 增至 `47.425469070 s`，新 prefix validation 为 `57.373281750 s`，attempts 从 438 增至 589。相同 wall 预算内完成了不同数量的工作，搜索路径变化是可行解释；这些观测不能证明同一搜索过程被等比例加速，也不能把所有 wall 或质量差异单独归因于缓存或编码优化。同状态没有 dispatch 样本，空统计不能记为零耗时。

## 成对闭环比较（共同 60 秒边界）

旧、新两轮 `closed_loop_comparison.json` 均为 `COMPLETE`，共同完成边界均为 `60 sim s`。本轮两 arm 独立 terminal observation 也均为 60 秒，未记录 failed 或 invalid；未用共同边界代替独立终态检查。

| 指标 | 旧 H | 旧 L | 新 H | 新 L |
|---|---:|---:|---|---|
| 共同完成边界（sim s） | 60 | 60 | 60 | 60 |
| 完整在线 wall（s） | 631.5473748500226 | 547.9939025760395 | 512.5745946640382 | 532.2263109689811 |
| job receipt wall（s） | 675.0836747119902 | 577.4162543540006 | 551.2853842279874 | 561.8026581830345 |
| full dispatch wall（s） | 605.291373077373 | 522.540882851463 | 490.54350675083697 | 509.4462071422022 |
| 模型 setup wall（s） | 0 | 0.673787084990181 | 0 | 0.5664047880563885 |
| 正常 / 成功着陆 | 4 / 4 | 4 / 4 | 4 / 4 | 4 / 4 |
| 应急着陆 / 能量失败 / 改航 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| admitted / airborne pending | 25 / 21 | 25 / 21 | 25 / 21 | 25 / 21 |
| 总能量（J） | 112618897.06073686 | 112530665.81757627 | 113241146.0233063 | 112314983.51112975 |
| holding time（s） | 514.57421875 | 489.587890625 | 638.4781494140625 | 571.580322265625 |
| airborne time（s） | 1403.5750761619288 | 1403.5750761619288 | 1413.5750761619288 | 1408.5750761619288 |
| upstream wait（s） | 0 | 0 | 0 | 0 |
| 物理 attempts / queries | 2839 / 5874 | 1819 / 4140 | 3237 / 6526 | 2385 / 4796 |
| callback / planning deadline miss / timeout | 36 / 36 / 0 | 36 / 36 / 0 | 35 / 35 / 0 | 32 / 32 / 0 |
| fallback / coverage 记录 / unserved | 0 / 36 / 0 | 0 / 36 / 0 | 0 / 35 / 0 | 0 / 32 / 0 |
| conservation | true | true | true | true |
| 最小间隔 lower / upper bound（m） | 82.79987429702142 / 83.7257990032708 | 82.79987429702142 / 83.7257990032708 | 82.85041888461755 / 92.88236784781643 | 80.17782389886187 / 80.23084222121737 |
| 独立 terminal time / failed / invalid | 60 / false / false | 60 / false / false | 60 / false / false | 60 / false / false |

完整在线 wall 截止完成模拟边界；job receipt wall 包含子进程生命周期及后续 replay 写入等开销，两者分别比较。full dispatch、模型 setup 和各规划分项是嵌套观测，不能与完整在线 wall 重复相加。本轮 L 相对旧 L 完整在线 wall 约 `-2.8773%`，H 相对旧 H 约 `-18.8383%`；本轮 L 相对 H 约 `+3.8339%`，旧 pilot 中 L 的约 `13.23%` 相对优势未保留。

本轮 L 对 H 的 `quality_regression_observed=false`、`lexicographic_episode_worse=false`：能量少 `926162.5121765435 J`（约 `-0.817867%`）、airborne time 少 `5 s`、holding time 少 `66.8978271484375 s`。但与各自旧结果比较，L 能量约 `-0.191665%`，holding time 却增加 `81.992431640625 s`（约 `+16.7472%`），airborne time 增加 `5 s`；H 能量约 `+0.552526%`，holding time 增加 `123.9039306640625 s`，airborne time 增加 `10 s`。不能把这些混合变化写成“质量不变”。

L 的最小间隔下界从旧 `82.79987429702142 m` 下降至 `80.17782389886187 m`，本轮上界为 `80.23084222121737 m`。下界仍高于冻结配置的硬约束 `80 m`，但观测安全余量收窄；这是距离下界观测，不能把其与碰撞或完整风险保证混为一谈。

holding energy 为执行器实际能量增量，不能用 holding time 乘固定功率代替，也不能与总能量重复相加：

| holding energy（J） | 旧 H | 旧 L | 新 H | 新 L |
|---|---:|---:|---:|---:|
| outer | 7664759.446658136 | 2743833.797816116 | 13982767.67204097 | 9588278.996318065 |
| middle | 13818138.864069458 | 16250337.952780452 | 15566935.355935685 | 13762431.038716108 |
| inner | 20158082.154972162 | 21529905.903038133 | 20874559.785077155 | 22894527.04982146 |

| 延迟（nearest-rank，s） | 旧 H P50 / P95 / max | 旧 L P50 / P95 / max | 新 H P50 / P95 / max | 新 L P50 / P95 / max |
|---|---|---|---|---|
| planning | 16.770029016 / 22.229037189 / 23.748851631 | 13.555944251 / 22.164466145 / 22.896782569 | 15.380767158 / 20.972762450 / 26.537823972 | 16.469264098 / 21.581017440 / 24.318830254 |
| dispatch | 16.770600827 / 22.229788182 / 23.750061457 | 13.557148593 / 22.165661356 / 22.897931236 | 15.381295179 / 20.973623978 / 26.538763214 | 16.469780447 / 21.581957362 / 24.319786004 |

本轮 H/L callback 数分别由旧 `36/36` 变为 `35/32`，总在线 wall 改善不能改写成每 callback 都更快。L planning P50 与 max 均高于旧值，P95 较低；本轮 H/L 的 `35/32` 次 callback 均记录 planning deadline miss，timeout 和 fallback 均为 0。这些结果不支持实时性达标结论。

规划分项按每 arm **全部** `exclusive_planning_timings` 字典逐字段汇总，分别保留 `[wall_s,cpu_s]`；旧闭环每 arm 36 份，本轮 H/L 为 35/32 份，未取列表第 0 份代替整次运行。L 的主要 wall 分项如下：

| L 分项 wall（s） | 旧 | 新 |
|---|---:|---:|
| feature | 39.155114877 | 20.733442360 |
| inference | 38.516512310 | 25.429875460 |
| feature + inference | 77.671627187 | 46.163317820 |
| candidate generation + single vehicle | 46.661820785 | 57.093665849 |
| CP solve | 136.269330166 | 181.291737142 |
| trajectory validation | 138.059516880 | 102.464750358 |
| prefix validation | 43.100088990 | 43.480140286 |

L 的 feature + inference 减少约 `40.566%`，同时 CP 增加 `45.022406975 s`；本轮 CP、trajectory validation、prefix validation 三项共约占 L 完整在线 wall 的 `61.48%`。attempts 从 `1819` 增至 `2385`、queries 从 `4140` 增至 `4796`，candidate 分项也上升。局部开销减少与 wall 预算内工作量、路径和 callback 数变化同时存在，因此既不能从局部分项直接推算总在线收益，也不能把质量差异全部归因于编码缓存。

## 离线校验与收尾

`archive_checks.json` 记录两 replay 的 hash、配置和轨迹检查均通过：H 为 88 个 states，L 为 85 个 states，两者时间范围均为 `[0,60]`，实际 `code_commit` 均为 `3c07585014e1b4b1ee401b5645dd56968376e58c`。其 `assertions=[]` 表示没有展开额外外部断言，不能解释成一组未执行断言已通过。人工视觉验收仍为 `HUMAN_PENDING`。

31 个不可变输入 hash、76 个冻结源码 hash 均有效；全部 20 个结果的 CPU12 与实际执行提交匹配。旧 same-state、closed-loop 和 budget 文件 hash 与 `original_metrics.json` 的事前记录一致，旧证据未被本轮改写。未调用通用 archive 或训练曲线重生成，复制的模型产物保持字节不变。

离线校验的精确完整 wall 未被独立测量，已依据 SSH wrapper 的 wall 观测保守取整，将 `7.0 s` 外部 credit 计入 archive；该值是保守记账额，不是精确校验耗时。evaluation 仍为 `1665.095692873001 / 3600 s`，ledger 已暂停且无 active phase。后续报告发布等小额 archive 记账以最终 `budget_ledger.json` 为准。

## 入口与证据索引

以下为本轮冻结、prepare、tests、evaluate 的入口记录，不是追加运行授权。已建立 guard 后不能再次 evaluate；不手动重置输出或 ledger。

```powershell
python scripts/stage3/reevaluate.py freeze --output artifacts/rotating_slots_stage3_20261024/encoding_reevaluation_source
```

```bash
cd ~/projects/terminal
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=/home/user/envs/terminal_stage2_v1/lib/python3.12/site-packages
PYTHON=$HOME/envs/terminal_env/bin/python
OLD=$HOME/training_outputs/terminal/stage3_learning_v1_0_20261023
NEW=$HOME/training_outputs/terminal/stage3_encoding_reevaluation_20261024
$PYTHON scripts/stage3/reevaluate.py prepare --parent "$OLD" --output "$NEW" --source-freeze artifacts/rotating_slots_stage3_20261024/encoding_reevaluation_source
$PYTHON scripts/stage3/reevaluate.py tests --output "$NEW"
$PYTHON scripts/stage3/reevaluate.py evaluate --output "$NEW"
```

新输出根的 `reevaluation_protocol.json`、`source_freeze.json`、`run_binding.generated.json` 保存协议、输入 hash 和双重身份；`targeted_checks.json`、`reevaluation_checks.json`、相应 XML 保存 gate；`evaluate.started.json`、`evaluate.finished.json`、`evaluation_exit.json` 和 `budget_ledger.json` 保存尝试、退出与预算证据；`evaluation/`、`same_state_comparison.json`、`closed_loop_comparison.json` 保存本轮结果；`archive_checks.json` 保存离线完整性与 replay 检查。仓库内 `artifacts/rotating_slots_stage3_20261024/{original_metrics,same_state_metrics_comparison,closed_loop_metrics_comparison}.json` 保存旧基线及只读提取的比较明细。旧结果详见 [campaign_20261023.md](campaign_20261023.md)，旧报告不被本轮覆盖。

局部编码优化有效，L 相对自身旧闭环总时间有小幅改善，但未保留对 H 的在线相对优势，质量变化混合。冻结权重和等价特征不保证 wall 预算下相同搜索次数、候选覆盖或质量；编码耗时变化可能改变截止前完成的工作。同状态判断仅限六 case，闭环仅限本次 seed 的 60 秒成对 pilot，不能推广为稳定、普遍或因果性的性能收益。整体保持 `ONLINE_BENEFIT_NOT_ESTABLISHED`，不追加训练或实验，人工视觉验收保留 `HUMAN_PENDING`。
