# rotating_slots：在本地 console 查看四份 480 s 回放

以下操作读取已同步到本机的 v6 记录；不会启动实验。先在 PowerShell 中进入 `D:\7_codespace\terminal`，使用仓库已有的 `venv\Scripts\python.exe`。

## 最容易的用法：开一个页面，按路径切换

```powershell
Set-Location D:\7_codespace\terminal
.\venv\Scripts\python.exe -m terminal_marl.visualization.console_server --no-autostart
```

浏览器会打开 [http://127.0.0.1:8787](http://127.0.0.1:8787)。页面默认收起侧栏和底部面板，优先显示大幅三维场景。点上方“场景设置”展开左栏，在“本地回放”的“绝对路径”粘贴以下任意一个目录，点“打开回放”：

```text
D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_stress_480s_20261022\energy_slack\replay_0480
D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_stress_480s_20261022\rh_cp_sat\replay_0480
D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_diagnostic_480s_20261022\energy_slack\replay_0480
D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_diagnostic_480s_20261022\rh_cp_sat\replay_0480
```

目录末尾加 `\replay.json` 也可以。加载会验证全部文件内容哈希、配置和轨迹，并转换所有原始采样帧；本机对四份记录测得约 19–29 秒，转换后约 46–72 MiB，出现加载提示时请等待，实际时间取决于机器。目录中的 `replay.json` 是成员索引，不能单独通过顶部“导入”按钮当作完整帧文件导入。顶部“导入”用于 console 导出的 JSON 或兼容 HTML。

在同一输入框换成另一个方法的目录，点“设为参考”，即可保留主回放并加载参考回放。加载后自动展开场景下方的“策略对比”面板；也可随时点上方“对比面板”展开或收起。指标位于独立底部面板中，不遮盖场景中的飞机。它按相同绝对 simulation time 和原始字符串 `aircraft_id` 匹配飞机；两个记录的采样时间不同时显示未对齐，不自动补帧。移除参考用比较面板的 `×`。

## 直接用命令打开四份记录

每条命令各占一个 PowerShell 窗口，端口互不重复。启动时先加载记录，再打开浏览器。

```powershell
.\venv\Scripts\python.exe -m terminal_marl.visualization.console_server --port 8787 --replay "D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_stress_480s_20261022\energy_slack\replay_0480"
```

```powershell
.\venv\Scripts\python.exe -m terminal_marl.visualization.console_server --port 8788 --replay "D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_stress_480s_20261022\rh_cp_sat\replay_0480"
```

```powershell
.\venv\Scripts\python.exe -m terminal_marl.visualization.console_server --port 8789 --replay "D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_diagnostic_480s_20261022\energy_slack\replay_0480"
```

```powershell
.\venv\Scripts\python.exe -m terminal_marl.visualization.console_server --port 8790 --replay "D:\7_codespace\training_outputs\terminal\stage2_rescue_opportunity_v6_diagnostic_480s_20261022\rh_cp_sat\replay_0480"
```

也可用启动器，例如 ` .\scripts\start_terminal_console.ps1 -Port 8787 -Replay "完整目录路径"`。加 `--no-browser`（启动器为 `-NoBrowser`）后手动打开对应 loopback URL。结束时在对应 PowerShell 窗口按 `Ctrl+C`；关闭浏览器标签不会关闭服务器。

## 看时间轴、飞机和资源

- 上方“场景设置”“飞机详情”分别开关侧栏；展开后侧栏与场景并排。点击飞机会自动展开详情。底部“航班表”“事件”“遥测”“资源诊断”“策略对比”标签会展开对应面板，右侧箭头收起面板。“专注场景”一次收起全部面板。窄屏侧栏排列在场景下方。
- 点击播放/暂停，或按 `Space`；拖动时间轴查看历史时刻，`←` / `→` 逐个记录帧移动。记录同时包含常规采样和事件时刻，因此帧间隔不一定等于索引的 `sample_dt_s`。
- 场景拖动旋转，`Shift` 加拖动平移，滚轮缩放；使用俯视/透视和重置相机按钮改变视角。点击飞机或航班表行查看原始 ID、状态、SOC、记录功率和救援状态；`F` 跟随选中飞机。
- “旋转槽位”图层显示 FREE、OCCUPIED、RESERVED、DEPARTING、SOFT；资源面板区分当前占用者、未来预留者、机动开始、下降、touchdown 和服务结束时钟。
- 降落 pad 使用带 `P{id}` 标识的状态圆形：绿色为空闲、红色为服务中、琥珀色为已预留、紫色为软预约（`soft_pending`）、灰色为未知。相机工具栏保留同一图例。圆形有最小屏幕尺寸以便识别，是状态示意，不表示真实物理占地；精确占用和时间仍以资源诊断中的记录为准。
- “承诺轨迹采样”每条活动机动最多显示 24 个权威轨迹点。稳定持环不附带整段未来轨迹。画面只显示记录帧的权威姿态，不在稀疏样本之间作直线插值；轨迹采样折线是显示提示。
- “在空 UAV”排除服务中、已清场和冻结失败；“服务中”表示 pad 实际占用，“已完成”是已完成服务/清场数量。touchdown 是落地事件，不能当作服务完成。上游队列是外部待准入请求，不是场景中的物理飞机。
- 隔离失败标记沿用记录中的冻结位置；该实验的隔离模型不表示事故级联模拟。分离下界/上界标注为全 episode 证据，不能解释为当前帧的即时安全证明。

## 依赖和常见错误

rotating_slots 回放需要项目 Python、`numpy` 和 `PyYAML`，不需要 PyVista、VTK、torch 或 Node 构建。优先用上述仓库虚拟环境；缺包时按项目依赖文件安装到该环境。

“replay cannot be read”通常表示路径不在服务器所在的本机，或 Syncthing 尚未同步完整；确认目录里存在 `replay.json` 和它列出的全部成员。内容哈希不一致、缺文件、配置/轨迹哈希错误会明确拒绝加载并保留原会话，不会回退到未校验读取。等待 Syncthing 完成后重试，勿手改源记录来绕过校验。

端口已占用时换一个未使用的 `--port`，并打开同一端口 URL。页面加载失败时检查 PowerShell 日志和 loopback URL；本地路径加载需要连接此服务器，离线 HTML 页面无法访问机器目录。单份服务端回放最多 6000 帧、源文件总计及转换 JSON 各最多 256 MiB；超过限制会报错，不会静默删帧。浏览器“导入”支持最多 256 MiB 的已导出帧文件。
