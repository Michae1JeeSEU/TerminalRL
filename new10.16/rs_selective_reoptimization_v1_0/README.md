# RS-SELECT 研究修改包 v1.0

**主文件：`STAGE3R_SELECTIVE_REOPTIMIZATION_CONTRACT_v1_0.md`**

主线：A规则子集＋局部CP → B一次编码学习选组 → C动态规模，然后独立评估较短接入。

这份包提供设计和实施合同，不包含已实现的新算法。没有修改远端仓库，没有执行CP/物理仿真，没有训练模型，32项测试均为NOT_RUN。

## 文件

- 主合同：总体设计、数学与状态语义、接口、A/B/C实施和验证。
- `selective_reoptimization_config_v1_0.yaml`：主配置规范。
- `phase_A_v1_0.yaml` / `phase_B_v1_0.yaml` / `phase_C_v1_0.yaml`：分阶段入口配置。
- `dataset_plan_v1_0.yaml`：新集合标签、根级划分和采集边界。
- `feature_spec_v1_0.json`：一次全局编码所需信息与禁止输入。
- `targeted_tests_v1_0.json`：32项验收情形与预期。
- `records.schema.json` / `record_example.json`：日志结构与非实验示例。
- `SOURCE_ADAPTATION.md`：论文和公开实现借鉴范围。
- `IMPLEMENTATION_ENTRY.md`：实施顺序与阶段门。
- `validate_package.py` / `PACKAGE_VALIDATION.json`：仅文件一致性自检。
- `MANIFEST.json`：交付文件SHA256；不包含自身及自检输出，避免循环。

历史分支审查为cd9077d；实际运行必须重新绑定可复现版本。A/B共同使用20秒未来接入及现有场景；C2的10秒只在条件满足后独立测试，不默认启用。
