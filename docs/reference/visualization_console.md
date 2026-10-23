# Terminal visualization console

可视化 console 是终端区 `WorldRenderSnapshot` 的只读工作台。它提供实时场景、时间轴回放、UAV 详情、事件与遥测面板，以及 JSON/HTML 导入和 JSON 导出。界面控制独立仿真进程的启动、暂停和步进；渲染与回放不改写记录数据或安全链决策。

## 启动

在仓库根目录运行：

```powershell
.\venv\Scripts\python.exe -m terminal_marl.visualization.console_server
```

跨平台环境可以使用：

```bash
python -m terminal_marl.visualization.console_server
```

服务只绑定 `127.0.0.1`，默认端口为 `8787`，成功绑定后默认打开浏览器。需要手工打开浏览器或使用回放时可以使用：

```powershell
.\scripts\start_terminal_console.ps1 -Port 8787 -NoBrowser
.\scripts\start_terminal_console.ps1 -Replay .tmp\terminal_console\console_replay.html
```

启动器只选择仓库 `venv\Scripts\python.exe` 或 PATH 中的 `python`，不安装依赖、不修改系统设置，也不隐藏用户主动启动的终端。`--no-autostart` 只启动 HTTP 工作台；`--replay PATH` 可读取 rotating_slots ReplayBundle 目录、其中的 `replay.json`、console JSON 或支持的旧 HTML，不执行 HTML 中的 JavaScript。四份 480 s 记录的完整命令及操作见 [rotating_slots 回放教程](rotating_slots_console.md)。

## 实时运行

默认运行配置为 `fixed_rule`、`fixed_rule_warm_flow`、seed `0`、仿真上限 `600 s`、播放速度 `1×`，并在第一帧后暂停。工作台通过 loopback HTTP API 控制独立仿真进程；父进程收集最多 `6000` 帧，HTTP 请求与渲染线程分离。缓存达到上限或发生旧帧淘汰时，界面会显示 truncation，导出 metadata 也会保留 `truncated` 与 `dropped_frames`。

支持的控制模式包括 `fixed_rule`、`fixed_low_rl`、`dynamic_rule`、`learned_high_rule_low`、`learned_high_frozen_low` 和 `checkpoint`。实时开始请求包含 mode、scenario、seed、仿真秒数、max_steps、播放速度、checkpoint 路径、profile、暂停和 deterministic 选项；非法参数会返回 JSON 错误，并保留当前运行。

checkpoint 模式要求填写本机可访问的绝对 `low_checkpoint` 路径。P3 使用 `profile=p3_task_neutral`，场景选择 `flow6`、`flow12` 或 `dense24`，seed 和 deterministic 开关显式记录在运行 metadata 中。P3 adapter 沿用原 formal environment config，用户的 `max_steps` 和秒数只作为本次执行的上限，不改写 checkpoint 的环境配置；它是新诊断接口，不宣称对旧记录动作逐步精确复现。`checkpoint12`、`p2_initial_k3` 与 `p2_trained_k3` 继续服从各自 checkpoint 的 SHA、schema 和场景约束。最终 metadata 只有在子进程返回时才记录实际 checkpoint 信息，不为缺失信息生成假 SHA。

## 数据与回放

实时与导入回放都使用 plain `WorldRenderSnapshot` 字典。`time` 是仿真的绝对 simulation time；实时 `step_count` 是环境 tick，rotating_slots 则是从零开始的记录帧序号（metadata 标为 `recorded_frame_index`），并以原始 `time` 对齐。UAV 的 `arrival_id` 用于旧记录请求匹配；rotating_slots 增加原始字符串 `aircraft_id`，跨回放比较优先用它匹配，`uid` 用于当前对象显示。位置、速度、SOC、phase、环、目标 LZ、承诺轨迹、冲突 UID、累计终态事件和 telemetry 均来自快照。rotating_slots 的 `interpolation=none` 明确要求显示原始采样帧，不在两个采样姿态间作直线插值。

工作台可以导入：

- `{title, frames, metadata}` JSON；
- 旧 `snapshot_playback.html` 中的 `const data = ...`；
- 新单文件 HTML 中的 `window.__TERMINAL_REPLAY__ = ...`；
- 左栏“本地回放”填写本机 ReplayBundle 绝对目录或 `replay.json` 路径，点“打开回放”或“设为参考”。服务端会先检查所有成员的内容哈希、配置和权威轨迹，再替换当前回放；失败保留当前会话；
- 独立参考回放：按当前帧的 absolute simulation time 查找参考中的相同 sample time，再用 `arrival_id` 匹配 UAV。只有找到相同采样时间时才计算精确差值；采样时间不一致时明确显示未对齐，不自动重采样或外推。

浏览器文件导入由前端限制为最多 `256 MiB`、`100000` 帧；命令行 `--replay`/服务端 `load_replay` 限制为最多 `256 MiB`、`6000` 帧。ReplayBundle 同时检查源成员总大小及转换后 JSON 大小，超过限制时报错，不丢弃或下采样帧。实时服务父进程的有界缓存同样是 `6000` 帧，这些限制分别适用于浏览器文件导入、服务端加载和实时缓存。

`write_playback(path, frames, title)` 保持原有 Python API，但新生成的 HTML 会内联 `web/index.html`、`console.css`、`scene.js` 和 `console.js`，并嵌入 `window.__TERMINAL_REPLAY__`。它是 snapshot-only 文件，不依赖服务、Node、CDN 或外部资源，设计为复制一个 HTML 文件后离线查看；当前验证覆盖生成、读回、无外部资源和 inline 脚本语法，未在本环境实测 `file://` 浏览器打开。导出前会拒绝空帧以及几何、时间和计数等关键字段的非有限数字；reset 时不可用的 telemetry 非有限值会转为 JSON `null`。`<` 会转义以防止 replay 文本关闭脚本块。仓库中的旧 `snapshot_playback.html` 保留，用于兼容历史文件。

## 界面操作

界面提供实时跟随和历史回放两种视图，时间轴可定位到任意已缓存帧；暂停时可逐 tick 单步，播放速度只改变显示/控制节拍。场景支持透视、俯视、重置相机、选中 UAV、跟随选中 UAV、图层开关、历史尾迹、承诺轨迹、冲突高亮、航班表、终态事件表和遥测面板。

- `Space`：播放/暂停；
- `←` / `→`：历史回放向前/向后一步；
- `F`：跟随当前选中的 UAV；
- 场景拖动：旋转；`Shift` 加拖动：平移；滚轮：缩放。

服务器异常、子进程错误、停止和重建都会显示在状态区；停止或重建会清理旧进程，错误以 JSON 返回，不进入无响应循环。服务仅接受 loopback Host/Origin 和有效 token，静态文件严格限制为工作台所需文件。

## 依赖与边界

最低项目运行依赖为 `numpy`、`gymnasium`、`torch` 和 `PyYAML`；当前仓库 `venv` 已具备这些包。console 的静态启动和 JSON/HTML 回放不要求 torch；rotating_slots 回放需要 `numpy`、`PyYAML`，不加载 PyVista/VTK，也不需要 Node 构建。旧 PyVista/VTK 3D viewer 仍是可选的 live diagnostic viewer，见 `requirements-visualization.txt`。

该平台当前没有 ROS/Crazyflie 实机控制接口，不训练模型，也不改变既有 checkpoint、action/observation 或 snapshot metadata schema。它显示仿真和已有记录，不宣称实现 SUMO 的全部交通仿真功能，也不提供未经测量的浏览器性能保证。
