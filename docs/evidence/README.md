# 实验证据索引

本目录保存 protocol、qualification、diagnostic、summary 与 reproducibility manifest。
这些文件是按当时版本冻结的证据，不因后续结果而改写，也不直接定义当前实现。

| 目录 | 内容 | 当前解释 |
|---|---|---|
| `dynamics/` | D0/D7 动力学审计、协议与报告 | 当前动力学 qualification 依据 |
| `pre_v4/` | V4 主线之前的 PPO 修复证据 | 历史对照 |
| `v4/` | 30-way joint action 阶段 | 已替代 |
| `v4_1/` | canonical WAIT + flat-26 阶段 | actor representation 已替代 |
| `v4_2/` | hierarchical gate/proposal、Observation V2.1 与当前 imitation lineage | 当前主要证据链 |

读取规则：

1. 先从 [../CURRENT_STATUS.md](../CURRENT_STATUS.md) 找到当前生效的最终决定；
2. protocol 说明“预先约定怎样判”，summary 说明“实际结果怎样”；
3. 后来的 PASS 不修改早期 FAIL，早期失败仍保留为不可变历史证据；
4. diagnostic 只能解释结果，除非有单独正式 decision，否则不能授权下一阶段；
5. 文件中的 checkpoint、dataset、cohort 使用限制持续有效。

当前 V4.2 证据链的关键入口：

- [V4.2 S0](v4_2/v4_2_s0_qualification_summary.md)
- [Observation V2.1 final freeze v2](v4_2/v4_2_v2_1_final_freeze_decision_v2.md)
- [Fresh I0-R PASS](v4_2/v4_2_v2_1_formal_i0_r_summary.md)
- [I1 formal FAIL](v4_2/v4_2_v2_1_i1_formal_qualification_summary.md)
- [I1 failure attribution](v4_2/v4_2_v2_1_i1_failure_attribution.md)
- [最新 I0-R3 FAIL](v4_2/v4_2_v2_1_i0_r3_collection_summary.md)
