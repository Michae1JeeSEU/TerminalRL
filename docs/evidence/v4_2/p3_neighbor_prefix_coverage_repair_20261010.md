# P3 邻机前缀覆盖修复预登记（2026-10-10）

状态：prefix smoke/full validation 已完成；原 prefix coverage 九 CV 因 fold wall budget exhausted 保留为历史 `INCOMPLETE/exit2`，随后登记的 ring-reuse 九 CV 已 `COMPLETE/exit0`（9/9 fixed-5000）。当前无运行中的训练任务；prepare 阶段未执行 query、snapshot、step 或训练；本轮 full collector 与 CV 均仅为 train-only development。

来源标注：`preexisting` 表示既有产物，`read-only` 表示本轮只读核对，`newly_run` 表示本轮新执行的实验。本轮 `prepare-only` 仅生成本地准备产物，不执行 query、snapshot、step 或训练。

## blocker-ranking controlled CV 预登记（2026-10-10，本轮 prepare-only；source NOT_READY）

沿用既有 P3 修复继续授权，仅登记一次有界 blocker-ranking 假设检验；manifest、guarded launcher 和 dry-run 已准备，code worker 的限定实现已在工作树中，未启动 SSH、未创建训练输出。预定输出为 `/home/user/training_outputs/terminal/p3_neighbor_blocker_ranking_20261010/nine_cycle`，精确启动需最终协议核对、记录源码 hashes，并由 primary 接受源码冻结后另行批准。

候选保持 `IntentNeighborActor`、state_dict、`first_feasible_union`、3149 actor rows、1270 expanded targets、473 held reference、model/aux seeds `91911/91912`、固定 marks `[0,100,500,1000,2000,5000]`、每折 `1200s`、outer `4200s`、4 spawn 与 `OMP_NUM_THREADS=MKL_NUM_THREADS=1`。唯一新增 loss 是原 native pair auxiliary BCE 之外的 fixed scene BCE `0.25` 与 fixed listwise known-blocker ranking `0.25`：`logsumexp(known)-logsumexp(known_blocked)`，只在 query 同时有 known-blocked 与 known-safe 时求均值，unknown mask；31 个无 known-safe 的 query 跳过。无 architecture、输入、physics、safety、teacher、Q 或 observation 改动。

`.tmp/p3_blocker_ranking_support_20261010.json` 为 `PASS`，SHA256 `d6b9e22ab1b53173062ffd548666f12067ffc50cfce4d63752bb65c5436c9ad4`；1270 queries 中 839 个有 known-blocked、808 个同时有 known-blocked/known-safe、31 个无 known-safe，blocked query 中 unknown contender 为 `0`。该 support inventory 不证明 ranking loss 是根因。预登记 manifest `.tmp/p3_blocker_ranking_cv_proposed_20261010.json` 保持 `source_freeze.status=NOT_READY`、`execution=NOT_STARTED`（当前登记版 SHA256 `fd86c89992dc159df649fcd78056aca03412aa7ce8e32a832077b29011755efb`）；launcher `.tmp/launch_p3_blocker_ranking_cv_20261010.py` SHA256 `af1a3ff16a8ea71fcd3e84f455f07d983ba768386f676af7e7c19f15f26ccb16`，其 dry-run `.tmp/launch_p3_blocker_ranking_cv_20261010.dryrun.json` SHA256 `3aa363c3c401bc981fc8b86e9e49d80f66abbade14465e6a3ff601e670bfee37`，已验证 argv/token round-trip；`--execute` 在 source freeze READY 前硬阻断。无 hyperparameter sweep、epoch 选择、重启、预算延长或新 seed，不重跑 ring-reuse/control；仍为 train-only development，`PPO_ALLOWED=false`，不是泛化、qualification 或 PPO/formal 准入。
新增加的 blocker-ranking 定向测试为 `12` 项，合并本地定向测试为 `35` 项，均 PASS（含 synthetic 4-spawn route）；server 同一 `35` 项 PASS，exit `0`，用时 `5.23s`。该 support inventory 已完成并为 `PASS`，目标 fixed `0.25` ranking loss 已明确预登记为 `NOT_STARTED`，scope 无歧义。预登记 manifest `.tmp/p3_blocker_ranking_cv_proposed_20261010.json` 保持 `source_freeze.status=NOT_READY`、`execution=NOT_STARTED`（当前登记版 SHA256 `fd86c89992dc159df649fcd78056aca03412aa7ce8e32a832077b29011755efb`）；launcher `.tmp/launch_p3_blocker_ranking_cv_20261010.py` SHA256 `af1a3ff16a8ea71fcd3e84f455f07d983ba768386f676af7e7c19f15f26ccb16`，其 dry-run `.tmp/launch_p3_blocker_ranking_cv_20261010.dryrun.json` SHA256 `3aa363c3c401bc981fc8b86e9e49d80f66abbade14465e6a3ff601e670bfee37`，已验证 argv/token round-trip；`--execute` 在 source freeze READY 前硬阻断。无 hyperparameter sweep、epoch 选择、重启、预算延长或新 seed，不重跑 ring-reuse/control；仍为 train-only development，`PPO_ALLOWED=false`，不是泛化、qualification 或 PPO/formal 准入。

## ring-reuse 九折终态与完整对照（2026-10-10，本轮 newly_run）

用户授权的唯一 ring-reuse 九折 CV 已终态完成。新 run 输出为 `/home/user/training_outputs/terminal/p3_neighbor_prefix_ring_reuse_20261010/nine_cycle`，源 commit `1575edefe5e55bfffbef8cca8dbdf2d3ce7bb796`，actor 源码 SHA256 `305cd2ad9417400b81342010226b130e545077f329c47d2a27ae2212db0a41eb`。root `result.json` 为 `COMPLETE/error=null`，sibling `nine_cycle.exit` 为 `0`，9/9 folds 均写入固定 marks `[0,100,500,1000,2000,5000]`，每折 error 均为 `null`，所有父子进程均已退出。各折训练内部记录的 `elapsed_seconds`（不等同于 OS 进程墙钟耗时，且不含加载等阶段）为：`893511=743.908978`、`893512=651.504703`、`893513=702.575448`、`893601=722.236428`、`893602=827.161459`、`893603=738.457399`、`893604=709.281309`、`893605=733.739929`、`893606=315.212726`；最大记录值 `827.161459s < 1200s`。protocol/result mtime 差约 `1748.806027s`，仅为文件 mtime 的近似根 wall 参考，不能当精确进程计时。

终态核验清单为 `.tmp/p3_ring_reuse_terminal_check_20261010.json`，SHA256 `61b84c1393c4004951bda62984475ee7c65354239aa7fb6f637cc92588752afd`；比较 artifact 为 `.tmp/p3_ring_reuse_comparison_20261010.json`，SHA256 `1ff299fdae5160f5f04ebdb8774fdf9036431f4488b4c7ed86a7295e31e64f42`。启动清单的 19 个源码路径已在终态重新从本地和服务器读取：local/server 与 launch manifest 三方 hash 全部一致，mismatch paths 为空。新 run protocol SHA256 为 `fc9bde3b1990ddc67a259e511deeb791a019434f35429fdbd87968f91a99b451`，result SHA256 为 `51f098ae4378d59f1a9e7ab189afcc9f9ba9c0ee2ec520d90dd3683fb144a553`；本地与服务器的 protocol、result、9 个 progress、log、exit、pid SHA 均一致。

既有 control `/home/user/training_outputs/terminal/p3_neighbor_nine_cycle_cv_20261009/nine_cycle` 也核对为 `COMPLETE/error=null`、exit `0`、9/9 fixed-5000；其全部 fold curve 保存在 `result.json`，旧格式没有独立 `fold_*.progress.json` 文件。control protocol/result SHA256 为 `994882bd2909391f23becb5abf01dba319dae421e9d9ece67aa6c0e6277db4f3` / `ab1e7d7c38c52d32393d650dc389e23e77eb844eedaa760798c698036af1d76e`，log 为 `COMPLETE/folds=9/error=null`，exit 为 `0`，且没有匹配的 control 训练进程。control 的 protocol/result mtime 差约 `7226.152341s`，同样只作近似文件时间参考。此前失败 prefix CV `0/9` 与首次 argparse failure 的原始证据均保留，未与本终态混用。

比较固定为新 run 的 `validation_reference_pairs` 与 control 的 `validation_pairs`，二者均为原 held reference 支持；expanded `validation_pairs` 单独报告。指标公式使用 artifact 中记录的整数分子/分母，`WAIT false-release` 为 WAIT 支持上的误放行比例，proposal 分母为 RELEASE proposal rows，joint 分母为 actor rows。

### all-9 held-reference micro

| 指标 | control 分子/分母 | ring-reuse 分子/分母 | delta |
|---|---:|---:|---:|
| RELEASE recall | `334/577 = 57.8856%` | `369/577 = 63.9515%` | `+6.0659 pp` |
| WAIT false-release | `340/2572 = 13.2193%` | `317/2572 = 12.3250%` | `-0.8942 pp` |
| proposal top-1 | `254/577 = 44.0208%` | `252/577 = 43.6742%` | `-0.3466 pp` |
| proposal top-3 | `366/577 = 63.4315%` | `354/577 = 61.3518%` | `-2.0797 pp` |
| joint accuracy | `2441/3149 = 77.5167%` | `2474/3149 = 78.5646%` | `+1.0480 pp` |
| blocked-pair recall | `265/394 = 67.2589%` | `296/394 = 75.1269%` | `+7.8680 pp` |
| blocker top-1 | `178/259 = 68.7259%` | `159/259 = 61.3900%` | `-7.3359 pp` |

### all-9 held-reference macro

| 指标 | control macro | ring-reuse macro | delta | folds |
|---|---:|---:|---:|---:|
| RELEASE recall | `65.6231%` | `71.0598%` | `+5.4367 pp` | 9 |
| WAIT false-release | `27.7758%` | `25.4825%` | `-2.2933 pp` | 8，排除 `893601` 零 WAIT 分母 |
| proposal top-1 | `49.0835%` | `50.0270%` | `+0.9435 pp` | 9 |
| proposal top-3 | `69.4121%` | `67.6935%` | `-1.7185 pp` | 9 |
| joint accuracy | `70.2759%` | `73.2475%` | `+2.9717 pp` | 9 |
| blocked-pair recall | `69.8726%` | `74.4087%` | `+4.5361 pp` | 9 |
| blocker top-1 | `71.1533%` | `64.4438%` | `-6.7095 pp` | 9 |

### all-9 per-fold delta（ring-reuse minus control，pp）

| seed | RELEASE recall | WAIT false-release | proposal top-1 | proposal top-3 | joint | blocker top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 893511 | `0.0000` | `-28.5714` | `0.0000` | `-4.3478` | `+6.6667` | `+2.8571` |
| 893512 | `+1.0638` | `+0.7974` | `-3.1915` | `-3.1915` | `-0.4161` | `-18.7500` |
| 893513 | `-2.2989` | `-1.0870` | `-1.1494` | `-3.4483` | `+0.1828` | `-29.0323` |
| 893601 | `0.0000` | `—` | `+11.1111` | `+5.5556` | `+11.1111` | `0.0000` |
| 893602 | `+4.0000` | `-6.2500` | `+2.6667` | `-5.3333` | `+5.4291` | `0.0000` |
| 893603 | `+1.3333` | `-4.3919` | `-2.6667` | `+2.6667` | `+3.5040` | `0.0000` |
| 893604 | `+10.8108` | `+15.0000` | `+2.7027` | `-5.4054` | `+3.5088` | `0.0000` |
| 893605 | `+11.2360` | `+2.8698` | `-2.2472` | `-4.4944` | `-2.2140` | `+5.1282` |
| 893606 | `+22.7848` | `+3.2864` | `+1.2658` | `+2.5316` | `-1.0274` | `-20.5882` |

dense6 为 `893512/893513/893602/893603/893605/893606`：joint 三折上升（`893513/893602/893603`）、三折下降（`893512/893605/893606`）；WAIT false-release 在 `893512/893605/893606` 变差。该方向不构成稳定 dense 改善。

### expanded pair 支持（单独报告）

新 run expanded `validation_pairs` 有 `1270` queries、`14396` known pairs、`1469` blocked pairs；blocked recall 为 `1089/1469 = 74.1321%`，blocker top-1 为 `562/839 = 66.9845%`，pair accuracy 为 `11381/14396 = 79.0567%`，safe recall 为 `10292/12927 = 79.6163%`。新 run 的原 held reference 仍为 `473` queries、`4555` known pairs、`394` blocked pairs；expanded 分母不同，不能与 473 reference 直接作 delta。原始 3149 actor rows 在 9 个 fold 的 train pool 中重复出现，不能将 `25192` 计为独立样本。

本轮结果支持“覆盖补充后 gate/部分 joint 聚合指标改善”的有限观察，但 proposal ranking 与 blocker top-1 下降，dense 方向不稳定；不能接受为完整 P3 训练退化修复、泛化解决、qualification 或 PPO 准入。保留 prefix 数据和纯代数 ring score reuse 优化，不升格模型，不重复 CV、不延长 epoch、不变更 gate、不追加 seed。下一步仅为已在执行的只读 rank-support inventory，针对既有 `1270` targets 统计 known-blocked、known-safe、unknown contenders；不启用 optimizer、环境、teacher 或 Q，待事实返回后由 primary 决定下一有界修复。`PPO_ALLOWED=false`，原 physics/safety/state-only teacher、random priority、159+N50 来源约束继续保持。

## prepare-only 验收（2026-10-10，本轮 newly_run）

实际输入为九个 parent seed 的 old3/additional3/supplemental3 目录与 old/supplemental witness；输出路径为 `.tmp/p3_neighbor_prefix_prepare_20261010_r5`，包含 `protocol.json`、`selection.json` 和 `result.json`。结果为 `PREPARED/error=null`：`223` states、`733` selected groups，其中 `20` reuse、`713` new；new weighted plan upper bound 为 `1808`。prepare counters 均为零：`snapshot_reads=0`、`teacher_queries=0`、`plan_calls=0`、`resets=0`、`steps=0`。

本轮限定集成回归为 `53 passed in 9.63s`、exit `0`，其中 `tests/test_p3_neighbor_prefix_coverage_data.py` 的 collector 单测为 `8 passed`；指定源码、collector 与七个相关测试文件 `py_compile` exit `0`，`git diff --check` exit `0`。后续 smoke/full 已完成，九 CV 启动事实见下文。

## smoke 与 full validation（2026-10-10，本轮 newly_run）

smoke 输出为 `D:\\7_codespace\\training_outputs\\terminal\\p3_neighbor_prefix_coverage_20261010\\smoke`，状态 `SMOKE_OK/exit0`，`3` states、`5` plans、`1.5563752595s`，`3` candidate requests，`0` teacher query、reset 或 step。full 输出为 `D:\\7_codespace\\training_outputs\\terminal\\p3_neighbor_prefix_coverage_20261010\\full`，状态 `COMPLETE/exit0/error=null`，`223` states、`713` new candidates、`797` plan calls；`733` selected groups 中 `20` reuse、`713` new。

`.tmp/p3_neighbor_prefix_full_validation_20261010.json` 为 `PASS`，input/protocol、loader `159+N50`/labels、old+supplemental witness reference、9 parent split、prefix+combined witness、selection、teacher-audit reason 和 root result/witness 检查均为 `PASS`。protocol/result/witness SHA256 分别为 `e1785882cb09296de408076af379c04b4a9ad8374c1159528807946537522c7c`、`3a33b54ecfd8fd85be1f75dea818693d798a5b3cc82e73bc22acc1a4f163dea0`、`0e5f14fc6c7e8eaf2f90abdca700c2dbf887f7a2c036df5e9a11bd8be46e777b`。9 个 parent 最大耗时 `212.275s`、合计 `1050.952s`；protocol/result mtime 差约 `380.806s`，只作近似 wall-time 参考。runtime manifest 为 `.tmp/p3_neighbor_prefix_full_runtime_20261010.json`；对应 `full.exit` exit `0`、`full.log` 为 `COMPLETE`，PIDs `3127173/3127174/3127175` 均已退出。

原存档 labels、`105D` prefix/traffic/mask 与输入 SHA 保持一致；官方旧快照资源重建比对沿用 `atol=1e-7`、`rtol=1e-6`。本轮 `reset=0/step=0`，仅为 train-only development，新 Q 保持隔离且 `PPO_ALLOWED=false`。两次 checker 路径误报 artifact `.tmp/p3_neighbor_prefix_full_validation_20261010_path_fail.json` 与 `.tmp/p3_neighbor_prefix_full_runtime_20261010_path_fail.json` 保留，原数据未变。full monitor `p3-prefix` 已删除。

九 CV 启动记录（历史）为按既有 preregistration 启动，输出 `/home/user/training_outputs/terminal/p3_neighbor_prefix_coverage_20261010/cv/nine_cycle`；UTC `2026-10-10T00:40:00Z` 已确认启动并出现首批 mark `0`，cwd `/home/user/projects/terminal`，wrapper/timeout/root 为 `3129150/3129151/3129152`，4 workers 为 `3129383/3129386/3129387/3129388`，resource tracker 为 `3129382`。启动初读 root 为 `RUNNING/error=null`；held `893511/893512/893513/893601` 已到 mark `0`，其余 progress 文件不单独证明已开始训练。protocol 两端 SHA256 均为 `fc9bde3b1990ddc67a259e511deeb791a019434f35429fdbd87968f91a99b451`，17/17 source hashes 两端一致；launch artifact `.tmp/p3_neighbor_prefix_cv_launch_20261010.json` SHA256 为 `b0882bc001be55f290e2bb7296104c0baccee21615c6d1a11ec4f87892fff00`。heartbeat `p3-prefix` 已在终态后删除。该段只记录启动事实，不记录学习改善。

## 九 CV 预算终态（2026-10-10，本轮 newly_run）

来源 `.tmp/p3_neighbor_prefix_cv_budget_check_20261010.json`，运行终态为 `INCOMPLETE/exit2`，错误为 `ValueError('fold wall budget exhausted')`；server/local `nine_cycle.log` 内容一致，exit 内容均为 `2`，所有父子进程已退出。首四个 held fold `893511/893512/893513/893601` 分别耗时 `1200.203/1200.083/1200.012/1200.167s`，均只写入 marks `[0,100,500,1000,2000]`；这四个 progress 记录均为 `ValueError` 终态，另 `5` folds 仅有初始化 `RUNNING` 文件、无 curve，`0/9` folds 达到 fixed-5000 `COMPLETE`。13 个关键文件两端 SHA 一致；protocol SHA256 为 `fc9bde3b1990ddc67a259e511deeb791a019434f35429fdbd87968f91a99b451`，result SHA256 为 `4d4ec4dc70a08a8b2cd96370d7f22b1eb5f9e8f93c3f1b873f761e2529bea051`；budget-check manifest SHA256 为 `ab2196d57684bc148173139486901d69fe41d138d37b246caca277a8c4af64f1`（路径元数据修正，不改原数据）。无重启、延长预算、换 seed 或换 model，原九周期 control `9/9` fixed-5000 complete 未重跑；source 为 `842e4f0`。该次 CV 失败及 partial/2000 产物永久保留，不作动作效果或泛化结论。

该 partial/2000 结果不用于 prefix 动作效果或泛化结论；后续固定权重分解见下：四并发每折主要计算变慢并触及预算；CPU 频率/缓存/带宽等具体硬件机制未单独归因。后续仅预备一次 `0 optimizer/0 updates` 固定权重短性能计时，不自动重启 CV；新 Q 保持隔离，`PPO_ALLOWED=false`。

## 计算效率与纯代数复用（2026-10-10，本轮 newly_run）

`.tmp/p3_neighbor_runtime_probe_summary_20261010.json` 的旧 expanded 数据为单 worker `0.161371s/pass`、四 worker `0.302928s/pass`；四 worker 的总有效吞吐为 `8.4337` passes/s，单 worker 为 `4.1712` passes/s，aux forward 约占 wall `0.55%`。该计时只表明四并发时单折耗时增加并触及既定 fold wall budget；CPU 频率、cache 或 bandwidth 等具体机制未由此 probe 单独证实。

纯代数修复保持 architecture、state_dict、loss、输入和 safety 不变：`3149` 行中 `1874` 个 ring non-final 行的 six pads 已由 `bases` 置零，因此相同 query 只计算 pad0 再 expand 到 six；final 行仍计算 six。固定权重 `ring_reuse_four` 的 `64` 个 timed passes 均值为 `0.148717s/pass`，相对旧四 worker 数据降低约 `50.91%`、为 `2.037x`；该结果不能断言 `5000` 全程必过或 P3 已解决。probe model 初始/最终 SHA 均为 `398981a06e2e2f97893d317d733b4f4474ea98ba47fc46b7bd208ec2771da976`，`optimizer=0/reset=0/step=0/teacher=0/Q=false`，probe 全部父子进程均已退出；官方 loader 只读既有 snapshot 重建资源。protocol/result SHA256 分别为 `309fb4ebdfbb7a47faaee254387876f45e971d20b0828c795c637d80699fe8d4` / `c572e1998ad2b6ae84858504a9f733b0d609f15df079760f16c015395b533546`。

local `72` tests/`9.29s`、server `62` tests/`1.47s`；float32 outputs max diff `0`、parameter grad `1.19e-6`、input grad `4.47e-8`，float64 outputs `4.44e-16`、parameter grad `3.55e-15`、input grad `6.94e-17`，未放宽既有 torch tolerance。当前 monitor `p3-prefix` 已删除，`p3-v2-5` 未触碰。

## 历史后续候选登记（已由 ring-reuse 九折终态段取代）

用户于 `2026-10-10`（北京时间）明确授权按原候选协议执行一次新登记；以下为启动阶段的历史观察，终态见本文顶部 ring-reuse 终态段。观察时间为 UTC `2026-10-10T01:55:29.5700436Z`（北京时间 `09:55`，为观察时间，不伪作精确启动秒），输出 `/home/user/training_outputs/terminal/p3_neighbor_prefix_ring_reuse_20261010/nine_cycle`，cwd `/home/user/projects/terminal`；wrapper/timeout/root/tracker `3138030/3138031/3138032/3138174`，workers `3138175/3138178/3138179/3138180`。当时 root 为 `RUNNING`、`error=null`；首四个 held fold `893511/893512/893513/893601` 已到 marks `[0,100]`，其余五个仅有初始化文件，不能写成全程完成。两端 protocol SHA256 均为 `fc9bde3b1990ddc67a259e511deeb791a019434f35429fdbd87968f91a99b451`（与旧 run 相同，不能单凭 SHA 区分；以新输出路径与 source code 区分）；合法 launcher `.tmp/launch_p3_ring_reuse_cv_20261010.py` SHA256 为 `f41e36aa43fde0e9ef86f45c042b7d4fd45213717732caf71d9de6ef0a1e403d`，pycompile、逐 token 和 6 行 wrapper 验证均 PASS，已调用一次。launch manifest `.tmp/p3_neighbor_ring_reuse_cv_launch_valid_20261010.json` SHA256 为 `0ec818e13545e02d48b028b220558af1c63d09aa82072bd1261f3834c56ec754`，其中 19 项 source hashes 匹配；source code commit 为 `1575edefe5e55bfffbef8cca8dbdf2d3ce7bb796`，记录时 docs HEAD 为 `33b9e998cf63a657765dab25ca3754601ac6166e`。首次错误 shell 的 wrapper `3136102`、server sibling `nine_cycle.argparse_failure_001.log/.exit/.pid` 与 `.tmp/p3_neighbor_ring_reuse_cv_argparse_failure.json` 均保留，未生成 protocol/marks；该解析失败与当前合法新 run 分开记录。候选仍为 `3149` rows/`1270` targets/reference `473`、model/aux seeds `91911/91912`、每 fold `5000/1200s`、最多 `4` spawn、outer `4200s`；此前 prefix CV `0/9` fixed-5000 failure/partial 与旧 control 均保留，旧 control 不重跑，不新 Q/PPO/formal。monitor `p3-ring-reuse` 后续已在终态删除；`p3-v2-5` 保持 `PAUSED`。

## coverage gain（2026-10-10，本轮 newly_run）

来源 artifact 为 `.tmp/p3_neighbor_prefix_coverage_gain_20261010.json`。reference old+supplemental 为 `4555` known pairs / `394` blocked pairs；prefix 新增 `9841` / `1075`；combined 为 `14396` / `1469`。candidate 为 `301 reference + 713 new = 1014`，target rows 为 `1270`，其中 `797` 为 new targets；`100` 个新增 `final_capacity` global candidates 合法为 `0` targets。

全部 `75` 个 interface×joint 组的 candidate/known/blocked 覆盖为 reference `18/18/8`，new 与 combined `75/75/75`。两切环接口 `o2m,m2i × 25`（与 N50 特征无关）为 reference `14/14/4`，new 与 combined `50/50/50`。这些数字只说明监督覆盖缺口已补齐，不说明学习或泛化已解决；九 CV 后续已按技术预算终止为 `INCOMPLETE/exit2`。

## 既有九周期对照（preexisting）

来源：`.tmp/p3_nine_cycle_comparison_20261010.json`，本轮 `read-only` 读取。新九周期目录为 `D:\7_codespace\training_outputs\terminal\p3_neighbor_nine_cycle_cv_20261009\nine_cycle`，状态 `COMPLETE`，9 folds、native-only、fixed 5000 mark 存在；旧六周期 control 也为 `COMPLETE`。两份协议均记录 `Q_accessed=false`、`environment_steps=0`、无 checkpoint；新九周期说明 supplemental 三回合只用于 train-only development，不作 qualification。

九周期全体 native 汇总（all-9 micro，不能与 old-6 直接作 delta）：RELEASE recall `57.89%`，WAIT false-release `13.22%`，proposal top-1 `44.02%`，proposal top-3 `63.43%`，joint accuracy `77.52%`。在相同 old-6 支持 `2258=372 RELEASE+1886 WAIT` 上，old-6→new-6 micro 为：RELEASE recall `56.7204%→61.0215%`、WAIT false-release `12.6723%→12.8844%`、proposal top-1 `41.3978%→44.0860%`、proposal top-3 `58.6022%→63.7097%`、joint accuracy `78.6980%→78.9637%`。四个 dense fold 的 joint-2 方向为两升两降，未形成稳定方向。

## 已核对的原标签与 witness coverage（read-only）

来源：`.tmp/p3_neighbor_candidate_coverage_20261010.json`，本轮 `read-only` 读取。九组原 label 共 `3149` 行，RELEASE `577`、WAIT `2572`；所有 `annotation_result`/label/context join 已通过。teacher audit 共 `69,305` 条，其中 joint-0 为 `2,572`，非零 checked RELEASE prefix 为 `66,733`。

candidate key 固定为 `(seed, interface, joint, original_feasible)`。每组按 `tick`、数值 `uid`、`decision_id` 排序取 lower median，每组保留一条。既有 old 与 supplemental witness 中已有相同 `(decision_id, joint)` 的条目复用并写入 manifest，不重复安排。

| parent seed | selected groups | existing reuse | new groups | new weighted plan upper bound |
|---:|---:|---:|---:|---:|
| 893511 | 58 | 7 | 51 | 171 |
| 893512 | 100 | 3 | 97 | 242 |
| 893513 | 101 | 1 | 100 | 240 |
| 893601 | 8 | 3 | 5 | 10 |
| 893602 | 99 | 2 | 97 | 257 |
| 893603 | 95 | 0 | 95 | 245 |
| 893604 | 72 | 3 | 69 | 139 |
| 893605 | 102 | 0 | 102 | 257 |
| 893606 | 98 | 1 | 97 | 247 |
| **合计** | **733** | **20** | **713** | — |

加权上界规则为 ring interface 每个 selected group 1 plan、`i2f` 每个 selected group 6 plans；每个 seed 的 new bound 均不超过登记上限 `900`。既有 old/supplemental witness 合计 `473` plan calls，覆盖 `75` 个 teacher-audit interface×joint strata 中的 `18` 个；ring strata 中 `46/50` 没有已知 blocked-positive。原标签行和 actor 输入保持不变，本轮只扩展 auxiliary coverage。

713 个 new group 的既有 reason 计数为：`certified=153`、`predicted_swept_separation=75`、`predicted_holder_separation=7`、`predicted_separation=248`、`final_capacity=100`、`predicted_holder_reachable_tube=69`、`merge_slot_retime_conflict=28`、`trajectory_infeasible=20`、`predicted_post_plan_holder_separation=13`；其中 feasible `153`、infeasible `560`。`final_capacity` 的 100 个条目是 global capacity reason，预期 `0` plans；因此不能把全部 733 selected groups 当作已有 neighbor pair labels。九个 parent 的 new weighted bound 合计 `1808`，仍低于总登记上限 `8100`；实际 plan/pair coverage 留待 full collector 完成后核对。

## prefix collector 预算与后续九 CV

以下保留执行前冻结的 collector 预算与后续九 CV 约束；prefix smoke/full 已完成，九 CV 已按预算终止为 `INCOMPLETE/exit2`：

- 每个 parent 最多 `150` 个 new candidates、`900` plans、`1800s`；九个 parent 合计上限 `1350` candidates、`8100` plans。root timeout `5700s`，outer timeout `6000s`，上限不延长、不重跑、不补成功样本。
- collector 最多 `4` 个 parent 进程并发；并发时固定 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。prepare 阶段为 `0 snapshot / 0 query`，只生成候选 manifest 和输入校验。
- smoke 只取三个 interface 各自首个 joint 非 `1/13` 的 new candidate，最多 `3` requests、`18` plans、`120s`，outer `180s`，不训练。smoke 失败即停止。
- full 只为 `713` 个 new groups 写新 candidate 输出；20 个 reuse 组只保留 manifest 引用。实际输出为 `/home/user/training_outputs/terminal/p3_neighbor_prefix_coverage_20261010/{prepare,smoke,full}`。
- 原始 `3149` BC 行不变；新增 held auxiliary support 单独报告，不能与旧量直接作 delta。

full 完整验收（进程、exit、schema、hash、candidate IDs、reason、parent split 和上限）已通过；随后曾启动唯一同模型九周期 CV：`first_feasible_union`、`native_pair_auxiliary`、model seed `91911`、aux seed `91912`，每 fold `5000` updates / `1200s`，最多 4 个进程，固定 root outer `4200s`。已有 nine-cycle control 不重复运行；该 CV 已因 fold wall budget exhausted 终止为 `INCOMPLETE/exit2`。

该后续 CV 的每个 mark 另行输出 `validation_reference_pairs`，固定引用原 `473` plans 对应 targets；新增 held auxiliary support 单独报告，不与旧量直接作 delta。

后续 CV 仍禁止新 Q、PPO、qualification、seed、epoch 或 network sweep；失败即停止，不追加预算。prepare-only、prefix smoke/full 与限定集成回归已完成；九 CV 已按技术预算终止为 `INCOMPLETE/exit2`，后续仍需遵守本节冻结的预算和输入 guard，不报告学习改善。

## 只读命名记录与边界

当前已存在的相关 launcher 名称包括：`.tmp/run_p3_neighbor_cv.py`、`.tmp/run_p3_neighbor_gate_max_20261009.py`、`.tmp/run_p3_neighbor_supplemental_20261009.py`；实际 prefix launcher `.tmp/run_p3_neighbor_prefix_20261010.py` 已执行 prepare/smoke/full/cv。

本记录不把离线 teacher audit、blocked/safe witness 或候选安全结果当作在线 observation；不引入 teacher、Q、PPO 或真实飞行资格结论。
