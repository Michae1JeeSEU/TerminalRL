# 切环轨迹局部修复回归协议（2026-09-17）

目的：修复 checkpoint12 演示中 UID6 的 m2i 逆环向/内切掉头，不修改历史结果，不训练模型。

实现范围限 TrajectoryGenerator.ring_transfer：合格旧 cubic 原样保留；异常曲线改为正向极坐标路径，半径/高度使用端点一、二阶导数为零的 quintic 插值。仅异常路径在原预约窗口内进行有界出发时刻搜索，速度仍由原动力学参数化器决定。inbound、final、策略88D/动作/奖励/教师/安全阈值不变。

预检查：18项几何边界测试与现有参数化/预约测试；开发 warm3 seed720000，32步输出闭环后完整回合，3/3完成、313s、安全0、exact replay通过。开发失败尝试产物保留为dev，不作为资格证据。

固定回归集（修改后第一次执行前冻结）：
- warm3 seed720001；flow3 seed720002；flow6 seed690003。
- checkpoint12 SHA256: 965c5438dfc42f9898a4059b1225527bebd94f048ae9dfe6253cb49fc0519198。
- deterministic student-only；每回合900s上限；不优化，不访问Q2，不重写原评估manifest。
- flow6/690003是已发现问题的历史回归场景，不声称全新独立测试；旧版本完成时间752s。
- 接受：各回合全部完成、I3 episode gates通过、exact replay与requested logprob通过；实际transfer采样无逆环向（vt < -0.001m/s），目标环内侵入≤0.05m（折线弦离散误差容限）；必须存在实际transfer采样。
- 保存新action tape、regression.json、validation_scene.html、generator/checkpoint哈希。报告完成时间变化，不作性能提升结论。

历史准入只证明旧生成器。修复回归不自动授予下一PPO block准入；后续须按新生成器快照补充student-only qualification/入口检查。
