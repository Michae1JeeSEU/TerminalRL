# P3 邻机前缀覆盖修复预登记（2026-10-10）

状态：prepare-only 预登记已通过。真实 query、snapshot 读取、采集和新训练尚未启动；本轮已完成确定性输入校验与候选选择，并冻结下一步协议。

来源标注：`preexisting` 表示既有产物，`read-only` 表示本轮只读核对，`newly_run` 表示本轮新执行的实验。本轮 `prepare-only` 仅生成本地准备产物，不执行 query、snapshot、step 或训练。

## prepare-only 验收（2026-10-10，本轮 newly_run）

实际输入为九个 parent seed 的 old3/additional3/supplemental3 目录与 old/supplemental witness；输出路径为 `.tmp/p3_neighbor_prefix_prepare_20261010_r5`，包含 `protocol.json`、`selection.json` 和 `result.json`。结果为 `PREPARED/error=null`：`223` states、`733` selected groups，其中 `20` reuse、`713` new；new weighted plan upper bound 为 `1808`。prepare counters 均为零：`snapshot_reads=0`、`teacher_queries=0`、`plan_calls=0`、`resets=0`、`steps=0`。

本轮限定集成回归为 `53 passed in 9.63s`、exit `0`，其中 `tests/test_p3_neighbor_prefix_coverage_data.py` 的 collector 单测为 `8 passed`；指定源码、collector 与七个相关测试文件 `py_compile` exit `0`，`git diff --check` exit `0`。真实 prefix smoke/full/CV 尚未启动。

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

## 待执行的 prefix collector 预登记

本节是执行前冻结协议，不表示已启动：

- 每个 parent 最多 `150` 个 new candidates、`900` plans、`1800s`；九个 parent 合计上限 `1350` candidates、`8100` plans。root timeout `5700s`，outer timeout `6000s`，上限不延长、不重跑、不补成功样本。
- collector 最多 `4` 个 parent 进程并发；并发时固定 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。prepare 阶段为 `0 snapshot / 0 query`，只生成候选 manifest 和输入校验。
- smoke 只取三个 interface 各自首个 joint 非 `1/13` 的 new candidate，最多 `3` requests、`18` plans、`120s`，outer `180s`，不训练。smoke 失败即停止。
- full 只为 `713` 个 new groups 写新 candidate 输出；20 个 reuse 组只保留 manifest 引用。输出预期为 `/home/user/training_outputs/terminal/p3_neighbor_prefix_coverage_20261010/{prepare,smoke,full}`。
- 原始 `3149` BC 行不变；新增 held auxiliary support 单独报告，不能与旧量直接作 delta。

full 完整验收（进程、exit、schema、hash、candidate IDs、reason、parent split 和上限）通过后，才允许唯一同模型九周期 CV：`first_feasible_union`、`native_pair_auxiliary`、model seed `91911`、aux seed `91912`，每 fold `5000` updates / `1200s`，最多 4 个进程，固定 root outer `4200s`。已有 nine-cycle control 不重复运行。

该后续 CV 的每个 mark 另行输出 `validation_reference_pairs`，固定引用原 `473` plans 对应 targets；新增 held auxiliary support 单独报告，不与旧量直接作 delta。

后续 CV 仍禁止新 Q、PPO、qualification、seed、epoch 或 network sweep；失败即停止，不追加预算。prepare-only 与限定集成回归已完成；真实 prefix smoke/full/CV 尚未启动，后续仍需遵守本节冻结的预算和输入 guard。

## 只读命名记录与边界

当前已存在的相关 launcher 名称（仅读取名称，未在本预登记中创建或执行）包括：`.tmp/run_p3_neighbor_cv.py`、`.tmp/run_p3_neighbor_gate_max_20261009.py`、`.tmp/run_p3_neighbor_supplemental_20261009.py`。prefix collector 的新 launcher 尚未登记为已创建文件。

本记录不把离线 teacher audit、blocked/safe witness 或候选安全结果当作在线 observation；不引入 teacher、Q、PPO 或真实飞行资格结论。
