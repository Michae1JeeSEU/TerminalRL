# 阶段3机制分析执行计划

用户已授权[阶段3计划](fixed_terminal_stage3_learning_problem_definition_plan.md)。阶段1/2已完成，结果见[阶段2证据](evidence/fixed_terminal_stage2/results.md)；物理规则仍由[固定问题定义](fixed_terminal_problem_definition.md)约束。本阶段不预设学习必要性。

1. **读取旧轨迹。** 核对162场承诺、holding、首差异和两例额外失败。已生成只读inventory，仅是不同策略轨迹的描述，不作单步因果解释。
2. **冻结机制样本与价值。** 按[阶段3协议](experiments/fixed_terminal_stage3_protocol.md)使用15context、共同状态、最多19根、指定continuation和无权重累计J；定义闭合估计、真实固定未来重放及可行条件边际。
3. **实现并核对工具。** 复用独立可行性与执行层，核对共同prefix、未到达信息删除、根批次/应急fallback、累计原始分量、不同世界分母及undefined状态。公开API已按实现事实写入技术规范。
4. **主审后预检与正式执行。** 初次和中间输出修订预检均保留；最终修正smoke及57项合并测试已通过，随后固定15context全部完成、exit0，共209根，不新增场景、种子或全方法矩阵。
5. **证据审查与问题定义收尾。** 分析同池短长窗排序、配对ex-post结局、条件remove-one和分项计算成本。证据核对和[学习问题定义](fixed_terminal_learning_problem_definition.md)已完成，见[阶段3结果](evidence/fixed_terminal_stage3/results.md)。局部收益、退化对照、闭合/实际排序反转和不可行边际均保留，不将预算参考称为全局最优。

阶段3已完成，学习必要性仍未证明；下一阶段待用户授权。本阶段仅定义可检验的候选或联合价值接口，不训练actor/PPO/IL/teacher，不设计网络结构或调奖励。若传统方法表现强，应如实承认；局部改善且慢不证明学习唯一必要。状态以[CURRENT_STATUS](CURRENT_STATUS.md)为准。
