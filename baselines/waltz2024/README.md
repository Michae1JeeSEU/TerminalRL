# Waltz 2024 / TUD_RL 对比基线

本目录保存 Waltz、Okhrin 和 Schultz（2024）论文 *Self-organized free-flight arrival for urban air mobility* 的可复现源码快照和无训练运行包装器。论文来源为 [TR Part C DOI 10.1016/j.trc.2024.104806](https://doi.org/10.1016/j.trc.2024.104806)、[arXiv v2](https://arxiv.org/html/2404.03710v2) 和[官方 TUD_RL 仓库](https://github.com/MarWaltz/TUD_RL)。

对比设计见[基线说明](../../docs/baselines/waltz2024_comparison.md)。当前工作范围先处理环境和场景，不包含重新训练、优化器更新或“当前项目优于论文基线”的结论。由于尚未取得带来源证明的 UAM actor 权重及所需输入，Policy IV 仍处于阻塞状态。

## 来源和获取

`UPSTREAM.md` 记录了精确的源码提交和覆盖范围。运行 `bootstrap.ps1` 时，脚本从官方 GitHub codeload 获取固定提交的压缩包，在本目录内展开临时文件，并且仅当 `upstream/` 或 `upstream_dev/` 不存在时复制到对应快照目录；已存在的目录会报告并跳过，脚本不会覆盖它们，也不会创建嵌套 Git 仓库。脚本还从固定的 BlueSky 提交 `f1626f99623c1ac20eadce0acaf930cb66842848` 获取 OpenAP legacy 子树和根目录 `LICENSE`，仅补齐缺少的资产文件；会校验 23 个文件和 165,964 字节，并且不会覆盖已有资产。源码快照压缩包会随原始仓库内容保留其它任务的 `.pth` 文件；这些文件不用于 UAM，且不作为额外资产获取。脚本不安装 Python 包、不运行测试，也不向服务器上传源码。

源码快照和资产由本目录的 `.gitignore` 保持为本地数据，可通过 `bootstrap.ps1` 按固定版本重新获取。Syncthing 是项目源码同步路径；不要使用 `scp`、`rsync` 或手工压缩包上传源码。服务器运行输出写入 `~/training_outputs/terminal/waltz2024`，不写入 `~/projects/terminal`。

## 环境和安装

本地使用隔离的 Python 3.12 虚拟环境。下面的 PowerShell 命令从项目目录创建环境、安装依赖并获取固定源码和 legacy 资产：

```powershell
cd D:\7_codespace\terminal\baselines\waltz2024
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\bootstrap.ps1
```

需要从批准的 CPU Torch 索引安装 actor 推理依赖时，可以在安装 `requirements.txt` 前执行：

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<2.14"
```

当前固定依赖见 [`requirements.txt`](requirements.txt)，其中 `mycolorpy==1.5.3` 为选定版本。服务器上已核验的解释器是 `/home/user/envs/terminal_env/bin/python`，已有 Torch 2.13；额外依赖隔离在 `/home/user/envs/waltz2024_deps`，通过 `PYTHONPATH` 暴露。无图形环境运行时设置 `MPLBACKEND=Agg`，并固定 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。下面给出从服务器项目目录执行的复核命令；命令只写入指定的训练输出目录：

```bash
cd ~/projects/terminal
env -u TUD_RL_OPENAP_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/envs/waltz2024_deps MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/envs/terminal_env/bin/python baselines/waltz2024/run.py \
  --scenario wave --diagnostic-zero --max-steps 800 \
  --output "$HOME/training_outputs/terminal/waltz2024/diagnostic-wave"

env -u TUD_RL_OPENAP_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/envs/waltz2024_deps MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/envs/terminal_env/bin/python baselines/waltz2024/run.py \
  --scenario study --n 30 --entry-check --noise 100 \
  --profile upstream_dev1000 --diagnostic-zero --max-steps 800 \
  --output "$HOME/training_outputs/terminal/waltz2024/diagnostic-study"

env -u TUD_RL_OPENAP_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/envs/waltz2024_deps MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/envs/terminal_env/bin/python -m pytest -q baselines/waltz2024/tests
```

这些服务器命令按当前 `run.py`、`runner.py` 参数和 `baselines/waltz2024/tests` 路径记录；`env -u TUD_RL_OPENAP_PATH` 保持运行器使用当前默认本地资产，`PYTHONDONTWRITEBYTECODE=1` 避免向源码目录写入字节码。`bootstrap.ps1` 不负责安装 Python 环境。

Windows Python 3.12 安装配方尚未在本地执行。当前主机检测到 bundled Python 3.12.14：`C:/Users/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`；该环境只核验到 `numpy==2.3.5`、`pandas==3.0.1`、pip 26.2.1 和 `venv`，尚未装齐本基线所需 simulator、Torch、SciPy、Matplotlib、Gym 等依赖，不能视为已验证的本地运行环境。当前验证使用服务器上的隔离依赖环境，未更新共享环境。2026-10-02 的新运行证据位于服务器 `/home/user/training_outputs/terminal/waltz2024/verification_20261002`，并同步到本地 `D:/7_codespace/training_outputs/terminal/waltz2024/verification_20261002`；本地汇总校验文件为 [`checks.json`](../../../training_outputs/terminal/waltz2024/verification_20261002/checks.json)，服务器对应 `/home/user/training_outputs/terminal/waltz2024/verification_20261002/checks.json`。其中 `pytest_final.stdout` 记录 `17 passed, 14 warnings in 2.03s`、退出码 `0`；测试覆盖前 65 步的 wave/study 源码与包装器状态、动力学和 history 一致性，截断 JSON/CSV schema 与调度边界、不兼容 checkpoint 拒绝，以及真实 CLI 缺少 actor 时退出码 `2` 且不创建输出目录。该证据的 `evidence_type` 为 `newly_run`。

九个完整 `diagnostic-zero` case 均退出码为 `0`，并生成各自的输出和 chart：

| case | profile / 参数 | requested | sim time / steps | complete / truncated |
|---|---|---:|---:|---|
| `wave_upstream` | `upstream_dev1000`, noise 0 | 12 | 229 / 229 | true / false |
| `study30_off` | `upstream_dev1000`, `n=30`, entry off | 30 | 607 / 607 | true / false |
| `study30_on` | `upstream_dev1000`, `n=30`, entry on | 30 | 603 / 603 | true / false |
| `study5_noise10` | `n=5`, entry on, noise 10 | 5 | 240 / 240 | true / false |
| `study5_noise20` | `n=5`, entry on, noise 20 | 5 | 240 / 240 | true / false |
| `study5_noise100` | `n=5`, entry on, noise 100 | 5 | 240 / 240 | true / false |
| `poisson_off` | `upstream_dev1000`, entry off | 22 | 616 / 616 | true / false |
| `poisson_on` | `upstream_dev1000`, entry on | 22 | 726 / 726 | true / false |
| `wave_paper800` | `paper800`, noise 0 | 12 | 214 / 214 | true / false |

九个 case 均观察到 `active=0`、`unspawned=0`、request/spawn conservation、CSV `unique(time,uid)` 和 `sigma∈{-1,+1}`。`complete=true` 只表示有限调度已消费且没有 active vehicle，不证明 service completion、landing success、安全性或性能。所有 case 使用 constant-zero heading，输出标记为 `status=runtime_diagnostic`、`policy_verified=false`、`official_policy_claim=false`；例如 `wave_upstream` 为 12 个请求、2 个 VTOL entry 和 10 个 outside removal，`study30_off/on` 均为 30 个 outside removal。上述证据不构成论文策略、actor inference 或性能结果。

## 运行器和场景

[`runner.py`](runner.py) 是可持续使用的无训练基线入口，当前 CLI 已完成核对。它要求在 `--actor` 和 `--diagnostic-zero` 中恰好选择一个；`--diagnostic-zero` 只是运行诊断，不构成 Policy IV 证据，也不代表论文策略评估。当前参数为：

```text
--scenario {wave,study,poisson}
--n N
--noise {0,10,20,100}
--entry-check
--profile {upstream_dev1000,paper800}
--actor ACTOR_STATE_DICT | --diagnostic-zero
--max-steps STEPS
--output ABSOLUTE_PATH
```

例如，明确标记为诊断的本地命令为：

```powershell
.\.venv\Scripts\python.exe .\run.py --scenario study --n 30 --entry-check --noise 100 --profile upstream_dev1000 --diagnostic-zero --max-steps 800 --output D:\absolute\waltz2024\diagnostic-study
```

严格 actor 形式保留同样的场景参数，把 `--diagnostic-zero` 换成 `--actor D:\absolute\path\UAM_actor.pth`。在取得实际 UAM actor checkpoint 及其来源证明前，不能称为 `waltz_original` 或零样本策略评估。运行器默认从 `assets/openap_legacy/bluesky/resources/performance/OpenAP` 读取本地 legacy OpenAP 数据；只有数据位于替代位置时才设置 `TUD_RL_OPENAP_PATH` 覆盖该默认路径。actor 推理只需要 actor 权重，不需要 critic 或训练优化器；严格加载使用 `weights_only=True`，没有回退路径。

`wave` 是包装器用于源兼容验证的四个 gate，时刻为 `t=0/30/60 s`，速度为 13。`study` 使用 15 秒调度和源码验证噪声路径。`poisson` 是本地明确标注的四簇扩展：簇起始时刻为 `0/120/240/360 s`，每簇数量独立采样 `Poisson(λ=5)`，非空簇内相邻飞机间隔 10 秒；该扩展不冒充原始 TUD_RL 语义。

`upstream_dev1000` 保留源码物理出生半径和原生几何。源码 `spawn_radius=1000` 是出生/观测归一化相关参数，不是硬性的圆形 `R=800` 边界；`paper800` 将物理出生半径设为 800，同时保留论文要求的观测距离分母 1000，不修改原始快照。当前日志使用 East/North 坐标（`x=e, y=n`）；上游航向为 0° 指向北、顺时针增加。面向共同物理后端的坐标适配器仍需后续实现并通过测试，不能把该转换写成已完成的项目适配。每个 UID 的历史在有序邻居变化时保持，仅新车辆重置；保留 NaN 填充和 `ghost-neighbor` 占位语义，该占位不代替上游真实 actor。需要改用其他 OpenAP 目录时，在运行器启动前设置 `TUD_RL_OPENAP_PATH`；这只是运行时输入，不会改写源码。

包装器保持上游模块不变，并处理本地有限队列的调度、移除收集和最终 logger 行。上述 17 项测试覆盖了有限队列相关的 schema 和调度边界；九个诊断 case 也已完成有限调度运行，但这些证据不能写成完整论文复现。

## 当前可用内容和限制

已获取的官方 main 快照和 dev UAM 快照提供 UAM 环境和场景代码，但没有已确认的 UAM actor 权重；另行检查的 release v1.1 未提供可用的 UAM 权重资产。dev 快照中的另外七个二进制文件不是 UAM 策略权重，只随原始快照保留。因此当前不报告 actor 推理、共同后端零样本比较、学习曲线、四个训练策略比较或论文硬件复现。论文中的 Crazyflie/MOCAP/Crazyswarm2 现场控制器也不在现有文件中；名称相近的 Bullet 仿真属于其他资产。

尽管已有上述有限测试证据，当前仍未取得带来源证明的 UAM 权重，因此不报告 actor inference、zero-shot 迁移或性能结果；不会从 `diagnostic-zero` 路径推断任何结果。
