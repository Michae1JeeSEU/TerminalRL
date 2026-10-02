# P3统一修复后的分阶段效果验证（2026-10-03）

## 目的与已有结果

用户要求逐步验证修复是否解决P3退化。沿用commit `d43076b`的新V3/90D profile，先确认有负载的物理/业务闭环，再检查策略迁移和学习。上轮196测试及3请求规则/冻结source完整周期属于preexisting_artifact，不能替代本轮密集场景证据。源码和文档修改、诊断及下一步选择由primary完成；worker仅执行指定命令、读取产物和监控。

本轮读取发现warm-start已进入OutcomeLedger却遗漏request_times，使CLI generated_requests漏算warm请求。补登记原始request_time，不改变运动、调度、奖励或随机采样；在大样本运行前用定向测试和warm-start两步smoke验证。

## 顺序与停止条件（运行前确定）

采用现有 `terminal_marl.scripts.diagnose_p3_integrated_repair`，全周期最多1800 ticks。规则与冻结source使用同一V3物理、安全、动作权限、90D、入场及reward；source为 `/home/user/training_outputs/terminal/p3_normal_advantage_gate_repair_20260928/fit_smoke/terminal_gate_repair.pt`，SHA256 `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab`，仅显式迁移并冻结。所有seed是开发诊断用途，不使用历史held-out，也不将单场景称为泛化证明。

1. A：12请求、warm6、SOC .90–1.0，env seed890501/policy seed890601。先规则两步smoke；primary核对初始化、统计和输出后运行规则完整周期，再运行同场景冻结source。
2. B（A未暴露阻塞后）：24请求、warm12、SOC .90–1.0，seed890502/890602，规则/source配对。保持高SOC，检查密集构型能否疏通。
3. C（B可运行后）：B相同请求随机流/种子，仅SOC范围改为.55–.90，规则/source配对，检查能量与资源耦合。更低SOC或额外seed只在当前证据需要时另记，不自动扩充矩阵。
4. 基础闭环可接受后，决定V3/90D数据适配与student-only qualification，再预先固定有界学习更新及前后配对比较。不能把零初始化迁移称为训练，也不能用旧V2 qualification自动解锁PPO。当前仍 `PPO_ALLOWED=false`。

每臂完整保留N/E/F/D/U、是否自然终态、技术error/censor、实际安全计数和checked/executed一致性、完成时间与能量。任一技术异常、安全/执行不一致立即停止向下一难度推进并诊断；高SOC仍发生耗尽或未决时先检查可达性/阻塞，不直接开始PPO。备降、应急等业务结果保留原类别；主审结合两控制器结果判断，不把“脚本exit0”视为业务通过。

工作目录为服务器 `/home/user/projects/terminal`，Python `/home/user/envs/terminal_env/bin/python`。输出根 `/home/user/training_outputs/terminal/p3_integrated_validation_20261003/`，各臂新目录，不覆盖历史。运行前核对Syncthing源码同步；不手动传输源码。预计长周期配置heartbeat，正常推进静默，终态清理。

## 本轮结果

待运行；本段后续只追加实际结果。

### A运行前检查（newly_run）

- 两个定向测试文件：30 passed，26.90s，exit0；日志 `preflight_tests.txt`。
- `A_rule_smoke`：2ticks、CENSORED、error=null，初始6架活跃、generated_requests=6、queue=0；安全计数全0，结构化输出正常。CENSORED是预定两步结束，不是完整周期结果。
- 本地/服务器5个指定文件SHA一致，服务器无重复诊断进程。采用前述固定A参数继续规则完整周期，输出 `A_rule/result.json`；冻结source尚未启动。
