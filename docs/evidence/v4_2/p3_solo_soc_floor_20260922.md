# P3 单机 SOC 下限核验（2026-09-22）

## 目的与参数

本检查核对随机初始 SOC 的下限是否高于单架 UAV 在无拥堵、正常规则降落流程中的最低可达电量。当前参数为：

- P3 V4 随机 SOC：`Uniform(.55, .90)`；
- touchdown reserve：`SOC >= .25`；
- 能量容量：`1500` 个归一化单位；
- 当前 `80/40/10 m` 安全语义、P3R 资源语义和 rule controller；
- phase power 为工程近似，不是实测功耗标定。

## 方法

本地运行完整的合法单机 flow-from-empty 矩阵，覆盖 `gate(4) x mission(4)` 共 16 个 case：

```text
.\venv\Scripts\python.exe -B -m terminal_marl.training.diagnose_v4_2_p3_solo_soc_floor_v1 --output-dir D:\7_codespace\terminal\_tmp_solo_soc_floor_full_20260922 --max-cases 16 --seed 930002
```

命令退出码为 `0`。随后对固定初始 `SOC=.55` 的同一矩阵做 replay，检查请求绑定、完成和安全结果。

## 结果

- `SOC=1.0` 参考和 `SOC=.55` 固定下限下各 16/16 个单机 case 完成；replay exact，安全计数为零。
- 到 touchdown 的最大能量消耗为 `240.9`。按 reserve 和容量计算，所需初始 SOC 为
  `0.25 + 240.9 / 1500 = 0.4106`。
- `SOC=.55` 相对该单机名义下限有 `0.1394` 的 SOC 余量，相当于 `209.1` 个能量单位；固定 `.55` replay 仍为 16/16 完成，且请求绑定 exact。
- 默认 `ArrivalProcess` fallback 的最小值为 `.35`，低于上述 `0.4106`；因此未来协议必须显式设置 SOC 下限，不能继承该 fallback。

## 边界

该证据只覆盖当前仿真中无拥堵的单机名义 rule 流程，不能保证密集交通、等待、资源竞争或安全接管期间的能量可达性。40 m 单独测试已通过真实内环缓冲接管，保护下限不满足时 fail-closed；本 16-case 矩阵没有进入缓冲区。服务器 SSH 当前会关闭连接，本次完整诊断在本地完成，不作为服务器正式 qualification。

本检查不改变 `PPO_ALLOWED=false`、`P3_QUALIFIED=false` 或 `P4_ALLOWED=false`。

## 下一步

先运行显式设置高 SOC 下限的 paired pilot；随后冻结 fresh P1/P2 资格与效果协议，在新 MDP 下复评已有 checkpoint。该单机结果本身不授权继续旧 P3 PPO 或 checkpoint promotion。
