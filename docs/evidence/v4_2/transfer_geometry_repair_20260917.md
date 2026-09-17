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

## 冻结回归结果

代码提交 `18377f2`；generator SHA256 `1b1f5e2ea2d9107b279a87d1b37e57469839a466bc3760c757e00c735c600024`。
下表均为本轮 newly_run；目录位于 `D:/7_codespace/training_outputs/terminal/`（服务器同名输出目录）。

| 目录 | 完成 | 仿真秒 | 实际transfer采样 | 逆环向采样 | 最大目标环内偏差 |
|---|---:|---:|---:|---:|---:|
| transfer_repair_warm3_20260917 | 3/3 | 228 | 191 | 0 | 0.014835m |
| transfer_repair_flow3_20260917 | 3/3 | 508 | 432 | 0 | 0.003695m |
| transfer_repair_flow6_20260917 | 6/6 | 707 | 965 | 0 | 0.015400m |

三回合 episode_gates、geometry_pass 全通过，碰撞/NMAC/硬间隔/未认证执行/SafetyInvariantError全0，shield全0，exact replay和requested logprob一致。UID6最小切向速度16.01879m/s，最大目标环内偏差0.001396m；旧播放中切向速度最低约-15.64m/s、进入半径约108m（目标环200m），现已消除该掉头模式。残余厘米级径向偏差来自分段线性插值的弦离散，非大幅内切。flow6旧版本752s，新版本707s；这是诊断回归，不能据此声称统计性能改善。表内均为仿真时间，不是脚本墙钟时间。

测试：`test_transfer_geometry.py` + `test_trajectory_time_parameterization.py` + `test_a67_controllability_audit.py` **28 passed**；`test_phase_a_landing.py` + `test_visualization_warm_flow.py` **10 passed, 1 failed**。唯一失败为历史V3 action schema断言，与当前V4默认配置不一致；使用修复前生成器的基线复核同样失败，本轮不修改无关断言。其余单机/多机rule-only完成测试在修复后通过。开发中被替代的失败尝试不计入最终验证，旧长测试已停止。

侵入范围：生产代码仅一个生成器文件，新增诊断工具和测试；合格旧cubic点数组完全保留。异常轨迹变化会改变后续交通状态，因此不承诺旧策略在新生成器上的逐步轨迹或统计完全相同。旧数据、checkpoint、Q2、manifest均未改写；本轮不训练。历史准入保留其版本适用范围，下一PPO block仍需新快照复核。

本地PyVista重新运行已有命令即可使用修复后的生成器；静态对照可打开 `transfer_repair_flow6_20260917/validation_scene.html`。无需服务器运行可视化。Git未配置remote，已本地提交并Syncthing同步，不声称push成功。
