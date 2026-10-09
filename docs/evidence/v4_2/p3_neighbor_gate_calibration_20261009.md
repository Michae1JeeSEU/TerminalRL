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

本地 23 项定向 tests 已通过（4.10s，exit0）。server preflight 首次因 source hash mismatch 未创建输出；Syncthing 匹配后仅重试一次，完成 `23 PASS/1.45s/exit0`，日志为 `/home/user/training_outputs/terminal/p3_neighbor_gate_max_20261009/preflight.log`。独立 synthetic smoke 已 `COMPLETE/exit0`（2 fold × 2 updates），结果为 `/home/user/training_outputs/terminal/p3_neighbor_gate_max_20261009/smoke_synthetic/result.json`；full six_cycle launch 仍待执行。评估固定 5000-step 的全部 fold，包含 WAIT false-release 与 pair metrics；不挑最好 epoch，不新增 threshold。新 protocol 不访问 Q，不 reset、teacher query、checkpoint、PPO 或 formal。补充 seeds `892872/893604–893606` 仍未消费。

当前状态：**local/server 23 tests 与各自 synthetic smoke 已完成；full six_cycle launch 待执行**。若没有稳定 action 改善，不继续延长该 factorization，回到未解决的样本/表示证据，不自动进行 formal admission。
