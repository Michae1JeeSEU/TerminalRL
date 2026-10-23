# Stage3正式实施包 v1.0

入口：[stage3_learning_initial_contract_v1_0.md](stage3_learning_initial_contract_v1_0.md)；实施顺序见[IMPLEMENTATION_ENTRY.md](IMPLEMENTATION_ENTRY.md)。

本包冻结已批准的D1—D7及首轮工程默认值。只实现学习辅助候选构造，原CP与确定性验证保留；CPU一个小模型、一次训练、新增自动计算4小时、一个60秒成对闭环。学习末端价值、RL和世界模型不在本轮。

| 文件 | 用途 |
|---|---|
| stage3_learning_config_v1_0.yaml | 训练、运行、预算与评价规范 |
| stage3_dataset_plan_v1_0.yaml | 数据根、seed、split和试验场景计划 |
| stage3_data_schema_v1_0.json | 便携记录的最小JSON Schema |
| stage3_targeted_checks_v1_0.json | 12项待实施的接口/防护检查 |
| CHANGELOG.md | 从讨论稿到正式版的变化 |
| validate_package.py | 仅检查文件内部一致性与哈希 |
| DOCUMENT_CHECK.json | 本包检查结果，不是算法验收 |
| FILES_SHA256.json | 文件完整性清单 |

校验命令：`python validate_package.py`。需Python及PyYAML；不读取GitHub、不安装依赖、不训练、不运行仿真。仓库实现后生成的resolved配置、数据和权重不包含在本包内。
