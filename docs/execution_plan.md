# 阶段4学习辅助调度执行计划

用户[阶段4计划](fixed_terminal_stage4_learning_scheduler_plan.md)已授权；[方法](arrival_restructure/stage4_learning_scheduler.md)与[协议](experiments/fixed_terminal_stage4_protocol.md)按主线程固定设计执行。实现与最终预检已完成，正式train/val数据生成获主审授权但尚未启动；正式训练和test尚未授权。

1. **准备与冻结。** 原样复制计划、只读运行环境与既有数据覆盖；已完成。固定新seed分组、统一池、简单模型、评价预算，保留阶段3闭合/实际排序反转限制。
2. **实现与tiny预检。** 实现可见特征、train-only scaler、闭合J/Δ标签、两类MLP、top3+anchors筛选及最终独立复查。核对数据分组、信息边界、单根快路径、序列化和完整链计时；API/93维FEATURE_NAMES已按代码核对。最终10新增/67合并测试及pilot完成，主审已接受。
3. **先生成train/val，训练另行授权。** 36个数据任务已获主审授权但尚未启动，等待源码提交；正式训练尚未授权。train4101–4103、val4201，九profiles×四时刻，108/36 snapshots；后续获授权后各模型固定seed4401仅正式训练一次。按informative val top3召回、val loss、最早epoch顺序确定checkpoint。
4. **锁定checkpoint后首次test。** 才生成4301–4303的108个离线snapshot及评价，另运行27fresh场景×六方法162episode。按协议保留all/informative/degenerate和分项时间，不增加重复计时矩阵。
5. **事实汇总与主审结论。** 比较学习、同预算beam、全池参考与原规则/滚动；保留安全、失败、应急、截尾及闭合标签边界。效果不佳不改预算或删数据，结论由主线程审阅后写入。

阶段1–3已完成，其原始证据不改；[阶段3结果](evidence/fixed_terminal_stage3/results.md)不预先证明本轮有效。当前不实施复杂网络、offline RL、actor/PPO、teacher动作模仿或物理变化。状态以[CURRENT_STATUS](CURRENT_STATUS.md)为准。
