# 修复后 checkpoint12 续训协议（2026-09-17，执行前预注册）

适用范围仍为静态空域 N3–N6，warm3 / flow3 / flow6；不升级高层或加密交通。
冻结配置：`terminal_marl/configs/low_v4_2/ppo_block2_transfer_repair.yaml`。
仅从原正式 checkpoint12 恢复模型和完整 optimizer（包括 Adam 状态），不从入口测试权重续训。

顺序：
1. seed732000 warm3 32步 smoke，确认模型/optimizer恢复、实际requested ratio、更新、保存恢复和输出闭环；不作为正式准入。
2. fresh seeds730001–730006 六回合 student-only qualification，前三 deterministic，后三 stochastic，场景各循环一次；不更新。沿用 I3 completion、安全、队列、shield、replay、requested logprob gates，并沿用本轮轨迹修复的顺环/径向离散误差≤0.05m检查。
3. seeds731001–731002，warm3 64步 / flow3 160步 PPO入口检查，沿用初始ratio≤1e-5、finite、KL≤0.05、checkpoint恢复，新增optimizer完整相等验证。此阶段权重不可进入正式训练。
4. 全通过才从原checkpoint12重新恢复，执行准备配置中的24回合，seeds700001–700024，场景warm3/flow3/flow6循环，900s上限，原超参数。每回合检查安全、完成、顺环几何、requested ratio、replay、有限更新、KL、模型/optimizer恢复，任一失败立即终止block，不跳gate。

资格和入口结果绑定生成器及相关PPO源文件快照。历史 R4/I1/Q2/I2/I3、首block及独立评估均只读；不重复Q2，不用Teacher动作计算PPO ratio。保留710001–710006用于后续独立评估，本次不消费。

启动前仅定向单测与smoke；代码/Git只在本地，Syncthing同步核验后服务器运行，输出位于训练输出目录。长任务启动后验证进程/中间产物，由Luna high监控：无变化静默，失败/完成/缺失产物报告，终态结束监控。

## 执行结果：全部 PASS（newly_run）

实现提交 `901da7c`；15项定向测试通过。smoke seed732000 32步全部gate通过，requested logprob误差0，两次minibatch更新，model和optimizer恢复完全一致。

服务器输出 `~/training_outputs/terminal/ppo_block2_transfer_repair_20260917`，本地同步目录 `D:/7_codespace/training_outputs/terminal/ppo_block2_transfer_repair_20260917`。服务器进程已退出，根manifest及三个阶段manifest全部PASS；文件哈希已本地/服务器交叉核对。

| 阶段 | 回合 | 完成UAV | 最大initial logprob误差 | 最大KL | Safety / shield / 逆环向 |
|---|---:|---:|---:|---:|---|
| qualify | 6 | 24/24 | 3.8147e-6 | 无更新 | 全0 |
| entry | 2短回合 | 不要求完成 | 0 | 0 | 全0 |
| train | 24 | 96/96 | 6.1989e-6 | 0.0172638 | 全0 |

资格最慢706s，训练最慢804s，均不超过900s。训练最大目标环内偏差0.017332m，低于预注册0.05m折线离散误差容限；模型/optimizer保存恢复、requested likelihood、replay、有限更新、completion和其余gate全部通过。正式训练从原checkpoint12恢复，入口测试权重未沿用，Teacher不替换行为动作。

最终checkpoint：`train/checkpoint_24.pt`，SHA256 `6420df9a8562135756819459c7682ec897d47acd64e841b75a21c697298ea14f`。
根manifest SHA256 `657f28d8a71f88fde650fb193d31bcbba1fd435877a0a5a00771b930beb2b019`。

Luna high监控因模型额度中断，主agent接管并确认终态、产物和结果；当前无训练任务继续运行。后续须先用保留710001–710006评估最终checkpoint，本轮不消费该评估集，不自动扩大课程，不将训练完成等同统计收敛或普遍性能提升。历史证据不变。无Git remote，未push，代码经Syncthing同步。
