# 外环优先与 15 秒接入：2×2 探索的初步混合效应

研究状态为 **PRELIMINARY_MIXED_EFFECTS**，交付状态为 **BUDGET_LIMITED_FINAL_REPLAY_AUDIT_PENDING**。八臂都保存了 300 秒终点，子进程均退出 0；批次仍因累计预算超限标记 FAILED。八个新 ReplayBundle 尚未做最终物理审计，因此本轮未完整验收，也没有阶段 B 准入。

中流量的两个单因素分别增加正常着陆，组合却回到控制水平；高流量的两个单因素均低于控制，组合最低。观察到的效应随这两类负荷/种子的场景而变，组合效应不具可加性，不能把外环优先或更短 lead 设为已验证的通用改进。既有默认仍为 baseline20，没有自动替换。

## 固定范围与完成状态

科学源码为 `226c58c215cc28c93b5fff2dfd6e1d1cc2a5b3a5`。当前服务器专项检查为 **82 passed、0 skipped**，并完成独立的真实 10 秒烟雾验证；这些检查不替代本次八臂保存轨迹的完整审计。本轮没有训练、学习模型、额外八快照采集或旧 A 结果改写。

每个根分别运行 baseline20、outer20、baseline15、outer15，一次 300 秒。中流量 s3r23/seed7324 使用物理 CPU12，高流量 s3r24/seed7325 使用 CPU14；每根内串行，两根并行。名义 K8、局部上限12、mandatory/阻塞扩展/overflow 机制、搜索候选类型、物理/能源/应急/准入/全局目标保持原协议。FULL_H 不在本轮重跑。

| 场景 | baseline20 正常着陆 | outer20 | baseline15 | outer15 |
|---|---:|---:|---:|---:|
| s3r23 | 10 | 12 | 14 | 10 |
| s3r24 | 14 | 13 | 12 | 6 |

所有八臂记录的应急着陆和 F_energy_total 均为 0。**F_energy_total 是能源失败数量，不是能耗**；实际能耗读取 energy_j。尚未运行的新回放审计不能由这些零计数代替；原 A 的自动审计也不覆盖新八臂。

## 累计工作、任务均值与条件等待

完整 normal-work 使用原有记账，含 pending 任务已执行的部分和 history 一次。下表均值为累计工作除以该字段 count，分母包括不同执行/交付状态的任务，不是仅成功计划的平均值。LOCAL、OVERFLOW_FULL、UNKNOWN 仍在原 followup_observables.json 中分组，本表总计不将全量降级称为纯 K8 成本。

“最终准备后等待”仅统计 COMMITTED 且有 ready_time 的任务，等于 effective_time-ready_time；拒绝、无前缀和 pending 不补造值。

| 场景 / 臂 | 完整工作累计 s | 任务数 | 每任务均值 s | COMMITTED 等待样本数 | 等待均值 s |
|---|---:|---:|---:|---:|---:|
| s3r23 / baseline20 | 145.840646 | 16 | 9.115040 | 6 | 7.504428 |
| s3r23 / outer20 | 168.392268 | 18 | 9.355126 | 9 | 9.170207 |
| s3r23 / baseline15 | 175.279095 | 23 | 7.620830 | 12 | 6.154059 |
| s3r23 / outer15 | 190.132756 | 25 | 7.605310 | 7 | 5.284925 |
| s3r24 / baseline20 | 134.044976 | 14 | 9.574641 | 11 | 9.393407 |
| s3r24 / outer20 | 153.181976 | 15 | 10.212132 | 11 | 7.972653 |
| s3r24 / baseline15 | 174.588599 | 22 | 7.935845 | 10 | 6.615282 |
| s3r24 / outer15 | 164.097613 | 21 | 7.814172 | 7 | 5.392271 |

15 秒臂的每任务工作均值较低，但任务数较多，累计工作仍高于相应 baseline20。等待均值较短也没有在两个场景都转化为更多着陆，不能直接等同服务改善。

20 秒是接入 lead，不是调度心跳；心跳始终为 5 秒，pending 期间触发会合并。15 秒干预同时改变未来投影状态、可用搜索窗口（准备1秒/交接4秒不变时，15→10秒）和任务阻塞节奏，不是单纯减少五秒空等，也不承诺相同计划。

## 真实跨环启动、选择与终点分布

O→M、M→I 是保存执行事件中的累计真实 TRANSFER 启动数。selected outer 是多次规划中外环被选 ID 的累计次数，不是去重飞机数，各臂任务数不同，不能当作每任务因果效应。终点环计数按记录的当前/source ring，包含在途动作，不等同全部 HOLD。

| 场景 / 臂 | O→M 启动 | M→I 启动 | 累计 selected outer | 终点 inner / middle / outer | 另有 INBOUND | 总空中 pending |
|---|---:|---:|---:|---:|---:|---:|
| s3r23 / baseline20 | 3 | 3 | 36 | 1 / 8 / 20 | 1 | 30 |
| s3r23 / outer20 | 5 | 6 | 61 | 2 / 7 / 18 | 1 | 28 |
| s3r23 / baseline15 | 13 | 13 | 38 | 7 / 8 / 10 | 1 | 26 |
| s3r23 / outer15 | 12 | 9 | 59 | 7 / 11 / 11 | 1 | 30 |
| s3r24 / baseline20 | 11 | 10 | 55 | 6 / 11 / 17 | 1 | 35 |
| s3r24 / outer20 | 14 | 14 | 62 | 9 / 14 / 12 | 1 | 36 |
| s3r24 / baseline15 | 5 | 13 | 96 | 10 / 2 / 24 | 1 | 37 |
| s3r24 / outer15 | 13 | 9 | 67 | 13 / 16 / 13 | 1 | 43 |

每臂另有 1 架 ring=null 的 INBOUND，故三个环计数之和比总空中 pending 少 1；按准确保存的 300 秒状态核对，中流量为 req_000015，高流量为 req_000012。这里没有把三环小计强制当成全部空中飞机。

outer20 在中流量将 O→M/M→I 从 3/3 提高至 5/6，在高流量从 11/10 提高至 14/14；但高流量正常着陆由 14 降至 13，更多向内转移并不足以确保端到端服务改善。baseline15 在中流量变为 13/13，在高流量却变为 5/13；高流量累计 selected outer 从 55 增至 96，O→M 反而由 11 降至 5，选择次数本身不能证明执行进展。

高流量 outer15 在终点为 HOLD_INNER13、HOLD_MIDDLE16、HOLD_OUTER13、INBOUND1，没有终点 FINAL_NORMAL/TRANSFER；baseline20 则为 FINAL_NORMAL4、HOLD_INNER2、HOLD_MIDDLE11、HOLD_OUTER15、TRANSFER_O_M2、INBOUND1。组合把更多未完成对象留在内/中环，但没有改善正常服务。现有证据尚不能将原因归为 CP 额度、pad 瓶颈或某一耦合约束，未做这些因果归因。

## 能耗与 holding

| 场景 / 臂 | energy_j 换算 MJ | 累计 holding s |
|---|---:|---:|
| s3r23 / baseline20 | 568.791642 | 6110.303399 |
| s3r23 / outer20 | 573.543870 | 5754.236748 |
| s3r23 / baseline15 | 588.455452 | 4728.338685 |
| s3r23 / outer15 | 580.681147 | 5421.063172 |
| s3r24 / baseline20 | 808.276453 | 8432.009856 |
| s3r24 / outer20 | 861.086929 | 8498.033411 |
| s3r24 / baseline15 | 825.803265 | 8729.577002 |
| s3r24 / outer15 | 959.910915 | 10594.901590 |

这些能耗须与完成数及 pending 一起阅读，不单独用较低值宣称效率提升。两个根各臂只有一次，场景还同时对应不同随机种子，没有统计推广或唯一负荷归因。

## 预算异常与未完成验证

supervisor_requested aggregate cap 为 **3750 秒**，实际八个子进程 wall 累计为 **4000.9887617981294 秒**，超出 **250.9887617981294 秒**。两根并行，因此该数是子进程 wall 累计，不是整批日历耗时。matrix_finished.json 的批次状态为 FAILED，最后一个 s3r24/outer15 行也标记 failed，但其 exit_code=0、completed_boundary_s=300、timed_out=false。批次预算失败与八臂模拟推进完成是不同事实。

按约定口径，旧 A、后续本地/服务器测试、freeze、smoke 和正式运行的已测累计为 **7323.47022066447 秒**，超过原 **7200 秒**约 **123.47022066447 秒**。另有一次 launcher 失败 wall 尚未记入，所以该已测数不是完整总耗时。supervisor 没有及时约束请求的累计上限；现有持久化 argv/stdin/空日志不足以复原其确切触发条件，未编造根因，也未重置旧账本掩盖超额。

达到上限后没有新建物理审计或重试。本轮剩余验证是八个已保存 ReplayBundle 的完整 PVA/C2、连续间隔、资源和非执行归因复核；目前 **NOT_RUN**。本稿不宣称 safety PASS 或完整接受，也不授权新实验、自动重跑或学习阶段。后续只在另行明确的额度与范围内处理剩余保存产物验证。

## 直接证据

- [只读原始指标摘要](D:/7_codespace/training_outputs/terminal/rs_selective_followup_20261028/followup_raw_metrics.json)，schema=followup_raw_metrics_v1，生成于 2026-10-28T08:14:36Z。
- [批次终态与逐臂执行收据](D:/7_codespace/training_outputs/terminal/rs_selective_followup_20261028/matrix_finished.json)。
- 各臂源绑定、treatment、完成标记与原始输出：`D:/7_codespace/training_outputs/terminal/rs_selective_followup_20261028/arms/<root>/<variant>/`；执行结果和 followup_observables.json 位于其 `evaluation/<root>/RULE_LOCAL_K8/`。
- 服务器对应根为 `/home/user/training_outputs/terminal/rs_selective_followup_20261028`。原 A 合同、报告和历史账本保持不变；本次追加预算异常单独披露。

原始摘要 SHA256：`fed5df842994c33c65224a1e0f90d38c6e952d9e4f219ee7a8867a4aaf86fae2`；批次终态 SHA256：`00d95d12b84e24130d3320db919033be223b1e2bf1e21d736229601ea660adf3`。本稿只读取保存数据，没有追加测试、仿真、审计或 GUI 观看。
