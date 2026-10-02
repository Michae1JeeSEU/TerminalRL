# 文献与基线综述（检索截止 2026-10-02）

本文件记录本轮已经核验的文献和代码事实，检索范围覆盖动态终端区/vertiport 到达控制、排程与容量、可执行轨迹安全、UAM 仿真和多机学习。它不是“全世界论文”的穷尽清单；预印本与其正式期刊版本按同一项工作去重。论文的实验结论只按作者报告记录。凡标为“针对本项目的适用边界（本次分析）”的内容，是本项目侧的范围判断，不是论文作者的结论。

本轮去重后记录 43 项工作，其中 41 项为研究论文/预印本，2 项为综述；R03 的预印本与正式期刊版合并为一项，R30、R37、R43 虽属同一安全过滤研究线但为不同出版物，分别记录。R38 的 arXiv 与 RA-L 版本合并为一项。其他 conference/journal 版本若内容属于同一工作则不重复计数。

当前项目处于 Phase A：landing-only dynamic terminal management，研究动态环形终端区、多 UAV 降落意图和安全过滤后的可执行轨迹。PPO_ALLOWED=false，static-geometry qualification 仍 pending；本文件中的“候选”不表示已经通过本项目 qualification，也不表示已有可用权重。

## 导航与阅读等级

| 家族 | 条目 | 阅读等级 | 代码/权重状态 |
|---|---|---|---|
| 外部目标面板 | R01 Waltz；R02 Yang–Wei；R03 Saxena | FULL_TEXT 或完整主文档 | Waltz 权重未获得；Yang、Saxena 有公开源码，运行状态仍以各自基线文档为准 |
| 终端区构型、到达和排序 | R04–R13、R18–R20、R22–R23、R33–R34、R41–R42 | FULL_TEXT / ABSTRACT_OR_SECTIONS | 多数没有作者公开的可复现 checkpoint |
| 学习型终端/降落决策 | R14、R16–R18、R39–R40 | FULL_TEXT / ABSTRACT_OR_SECTIONS | 代码可见性不等于同场景可复现 |
| 轨迹与安全过滤 | R15、R21、R29–R30、R37–R38、R43 | FULL_TEXT / ABSTRACT_OR_SECTIONS | 主要是轨迹层或安全层，不自动成为交通管理基线 |
| 仿真、网络和车队 | R24–R28 | FULL_TEXT / ABSTRACT_OR_SECTIONS | fleet/network scope 与 landing-only 主线不同 |
| 结构化 routing/interlock 与移动平台 | R31–R32 | FULL_TEXT | 不自动成为动态 geometry 基线 |
| 综述 | R35–R36 | FULL_TEXT | 用于术语和研究边界，不作为数值控制器 |

外部最终目标面板按当前项目决策记录为：R02 Yang–Wei 的非学习在线规划、R03 Saxena 的资源/时隙 MILP（预印本与 2026 期刊版合并为一条）、R01 Waltz 的 learned free-flight arrival。这个面板分别覆盖 search、optimization 和 learned free-flight 三类决策机制。R01 当前没有可用作者 checkpoint，不能用随机初始化代替。R04、R07–R10 等 queue/geometry 工作用于定位和结构化规则参照；由于未核验到作者公开可执行代码，不能把项目 fixed_rule 称作它们的精确复现。R14、R16、R17、R39–R40 等学习型终端/降落工作的代码可见性本身也不使其自动成为当前 terminal-flow 面板；R15、R21、R29–R30、R37–R38、R43 等轨迹/安全工作只有在论文最终主张需要对应 planner layer 时才是条件候选。项目内部仍保留 fixed_rule、fixed_low_rl、dynamic_rule、dynamic_hrl 以及 clean FCFS、local EDF、SOC heuristic；这些是本项目的 method controls，不冒充下列论文的原始复现。

## 逐篇记录

### R01 — Waltz, Okhrin, Schultz (2024)

**元数据与阅读级别。** *Self-organized free-flight arrival for urban air mobility*. Martin Waltz, Ostap Okhrin, Michael Schultz. *Transportation Research Part C*, 167, 104806 (2024). DOI [10.1016/j.trc.2024.104806](https://doi.org/10.1016/j.trc.2024.104806)，[出版社页面](https://www.sciencedirect.com/science/article/pii/S0968090X24003279)，[arXiv v2](https://arxiv.org/html/2404.03710v2)。FULL_TEXT。官方代码为 [MarWaltz/TUD_RL](https://github.com/MarWaltz/TUD_RL)。

**问题。** 论文研究圆形终端到达空域中的自由飞行和高密度到达。

**方法。** 每架飞机是一个 agent，使用局部信息和 recurrent LSTM-RecTD3（recurrent TD3）策略学习航向；landing-zone entry order 由一个简单 RULE 信号处理，未把整个入场排序写成端到端 RL。原始实现保留连续航向增量动作、单高度圆形空域和局部邻机历史，并报告传感器噪声、入场流变化以及小型无人机演示。

**作者报告的发现。** 作者报告未加 admission guard 时会出现 accidents；加入 300 m gate check 后，在相应实验中观察到 zero accidents 并减少 incidents。100 m sensor noise 会增加 incidents/accidents。五架 Crazyflie 在进入 vertiport 后使用 hard-coded landing positions，并通过 Crazyswarm2 做了展示；这些观察不构成一般性的 zero-safety 声明。

**作者明确的局限。** arXiv v2 第 7 节说明系统是 noncooperative、没有 intent communication，且简单 RULE 决定 landing-zone entry order；作者将 communicative autonomous sequencing 和 full-scale eVTOL 作为后续方向。

**针对本项目的适用边界（本次分析）。** 这是 learned free-flight arrival 目标面板。原始策略、迁移后的 common backend 和 OOD zero-shot 三种状态必须分开；没有 checkpoint 时不能声称复现 learned policy，也不能用随机 actor 代替。它没有被改造成三环 WAIT/RELEASE 策略，不能把改造后的结果称作原始方法。

**代码/权重。** 官方仓库可获取源码；本项目截至本记录没有获得官方 UAM actor 权重。当前没有论文性能结果或 Policy-IV 资格结论。

### R02 — Yang, Wei (2020)

**元数据与阅读级别。** *Scalable Multi-Agent Computational Guidance with Separation Assurance for Autonomous Urban Air Mobility*. Xuxi Yang, Peng Wei. *Journal of Guidance, Control, and Dynamics*, 43(8), 1473–1486 (2020). DOI [10.2514/1.G005000](https://doi.org/10.2514/1.G005000)，[论文 PDF](https://xuxiyang1993.github.io/papers/jgcd20_xuxi.pdf)，[官方代码](https://github.com/xuxiyang1993/Multi_MCTS_Guidance_Separation_Assurance)。FULL_TEXT。源码运行快照为 2c83aeab807daee6bab0561199116210d44839da。

**问题。** 论文研究自主 UAM 中多机可扩展计算制导与间隔保证。

**方法。** 去中心化顺序消息传递、多智能体 MCTS、hex/gate 状态、2-D Dubins 运动；论文原始设置包括 100 rollouts、depth 3、5 s 决策周期以及 {-5, 0, +5} 度航向动作。项目基线另保留源码配置中可见的 lite profile，不把 lite 配置误写成论文原始压力实验。

**作者报告的发现。** 在作者报告的 1000 次 encounter 统计中，LOS 小于 1%，NMAC 小于 0.1%；这些不是零风险声明。

**作者明确的局限。** 作者指出场景偏 rural/suburban、没有高度维度和高速率/高保真动力学，未来需要更高保真和战略交通流耦合。

**针对本项目的适用边界（本次分析）。** 这是非学习在线规划基线，适合与高层意图/到达排序比较；不能把其二维空域制导结果直接解释为三维动态环和 LZ service 的系统容量。

**代码/权重。** 官方源码公开；没有训练 checkpoint 依赖，当前项目已有受限 headless wrapper，正式跨方法结果仍 pending。

### R03 — Saxena, Kuri, Prabhakar, Yadav (2024/2026，去重一项)

**元数据与阅读级别。** 预印本题名为 *Vertiport Terminal Scheduling with Multiple Surface Directions along with Throughput Analysis*；正式版题名为 *Vertiport Terminal Scheduling and Throughput Analysis for Multiple Surface Directions*. 预印本作者顺序为 Ravi Raj Saxena, T. V. Prabhakar, Joy Kuri, Manogna Yadav；正式版顺序为 Ravi Raj Saxena, Joy Kuri, T. V. Prabhakar, Manogna Yadav。预印本 [arXiv:2408.01152](https://arxiv.org/html/2408.01152v1) (2024)；正式版 *The Aeronautical Journal*, 130, 2434–2460 (2026), DOI [10.1017/aer.2026.10173](https://doi.org/10.1017/aer.2026.10173)，[出版社页面](https://www.cambridge.org/core/journals/aeronautical-journal/article/abs/vertiport-terminal-scheduling-and-throughput-analysis-for-multiple-surface-directions/D16630616927A393FC08D90F6FB5AE92)。FULL_TEXT（预印本）/ABSTRACT_OR_SECTIONS（正式版摘要页）。官方代码 [rrsaxena92/Vertiport_schedule](https://github.com/rrsaxena92/Vertiport_schedule)，快照 87a8e5ba7393d622768a7695770a4fa4e09c26bd。

**问题。** 论文研究多方向表面流下的 vertiport terminal scheduling、资源占用、等待和容量。

**方法。** MILP 同时考虑资源、等待、taxi、OFV、approach/departure、gate TAT，并给出最低容量分析；公开仓库原始实现是 MATLAB Optimization Toolbox 文件。

**作者报告的发现。** 一到两个方向的测试中，作者报告约 50% 延误下降；更多方向仅在路径分离主导而非 OFV 占用主导时继续有利。

**作者明确的局限。** 作者指出确定性设定、可扩展性和 QoS；未来方向包括 rolling horizon。

**针对本项目的适用边界（本次分析）。** 这是资源/时隙调度参照，不是自由飞行轨迹策略。与本项目比较时要固定外生请求 tape、LZ/service 语义和动态资源预算；不能用一个方法的资源容量归一化结果替代共同的 mu_ref。

**代码/权重。** 官方仓库公开 MATLAB 文件，无公开 checkpoint；本项目的 Python/SciPy/HiGHS port 已实现并通过 7 tests/7 tiny fixtures。MATLAB cross-solver parity 和 2026 正式版的完整 formulation 仍未核验；仓库未见 LICENSE。

### R04 — Bertram, Wei (2020)

**元数据与阅读级别。** *An Efficient Algorithm for Self-Organized Terminal Arrival in Urban Air Mobility*. Joshua R. Bertram, Peng Wei. AIAA SciTech 2020. DOI [10.2514/6.2020-0660](https://doi.org/10.2514/6.2020-0660)，[作者 PDF](https://bpb-us-w2.wpmucdn.com/web.seas.gwu.edu/dist/9/15/files/2019/12/scitech20-josh-terminal.pdf)。FULL_TEXT。

**问题。** 自组织 UAM terminal arrival 的计算制导和多机到达间隔。

**方法。** 两环、FIFO、FastMDP/pseudo-6DOF 风格的到达/间隔模拟。

**作者报告的发现。** 作者报告 15 aircraft、两环、10 次运行中 0 NMAC；45 aircraft、四环的单次压力实验中报告 9 NMAC、0 collisions。

**作者明确的局限。** VTOL threshold entry 后 aircraft 被移除；垂直下降和地面 service 未建模，未来需要 speed resolution 等扩展。

**针对本项目的适用边界（本次分析）。** 是结构化终端到达先例和强排程参照，不能写成当前项目的完整 landing/service 基线，也不能把 FIFO 结构当成本项目 fixed_rule 的精确复现。

**代码/权重。** 在该论文的作者主页面和 PDF 中未找到官方代码或 checkpoint。

### R05 — Kleinbekman, Mitici, Wei (2020)

**元数据与阅读级别。** *Rolling-Horizon Electric Vertical Takeoff and Landing Arrival Scheduling for On-Demand Urban Air Mobility*. Imke C. Kleinbekman, Mihaela Mitici, Peng Wei. *Journal of Aerospace Information Systems*, 17(3), 150–159 (2020；online 2019). DOI [10.2514/1.I010776](https://doi.org/10.2514/1.I010776)，[作者页面](https://research.tudelft.nl/en/publications/rolling-horizon-electric-vertical-takeoff-and-landing-arrival-sch/)，[作者 PDF](https://cpb-us-w2.wpmucdn.com/web.seas.gwu.edu/dist/9/15/files/2020/01/jais19-imke.pdf)。FULL_TEXT。

**问题。** 多 pad vertiport 的 rolling-horizon eVTOL arrival schedule，同时考虑 route、RTA 和 SOC。

**方法。** MILP rolling horizon，route/RTA/SOC 约束，Houston 两 pad 场景。

**作者报告的发现。** 论文报告高峰平均延误约 50 s、非高峰小于 10 s；5/10/15/20 min horizon 的计算时间约为 5.7/76/190/510 s（论文条件）。

**作者明确的局限。** 未联合建模 turnaround/departure，极端 demand 和 contingency 未覆盖。

**针对本项目的适用边界（本次分析）。** 可作为 resource-aware rolling MILP 参照；不能把其两 pad、时间窗语义直接与三环自由飞行的 actor 比较。

**代码/权重。** 未在作者 primary 页面找到官方代码或 checkpoint。

### R06 — Pradeep, Wei (2020)

**元数据与阅读级别。** *Heuristic Approach for Arrival Management of Aircraft in On-Demand Urban Air Mobility*. Priyank Pradeep, Peng Wei. *Journal of Aerospace Information Systems*；DOI [10.2514/1.I010758](https://doi.org/10.2514/1.I010758) 本轮标为 DOI metadata UNVERIFIED；[作者 accepted PDF](https://bpb-us-w2.wpmucdn.com/web.seas.gwu.edu/dist/9/15/files/2020/07/jais_priyank_arrivalscheduling.pdf)。FULL_TEXT。accepted PDF 中另见 eVTOLs 的题名措辞变体；以下按已核验发表题名记录。

**问题。** 研究 on-demand UAM 的 arrival sequencing、landing management 和 pad/window 资源。

**方法。** ILS+MILP，与 Time Advance (TA) 比较；winged/wingless 先分开再在 metering 合并，优化 makespan 和 local window/pad 约束。

**作者报告的发现。** 作者报告 k=3 时 250 aircraft 的 schedule 小于 10 s；100 aircraft 的 k=5 与 k=3 分别约 39.68 s 和 1.37 s，体现解质量/计算时间权衡。

**作者明确的局限。** 本轮未核验完整作者 limitations 段落。

**针对本项目的适用边界（本次分析）。** 可作为小规模排程参照；其具体 ConOps/约束和排程语义需与本项目分开记录，不是 multi-UAV continuous trajectory 或 dynamic geometry baseline。

**代码/权重。** 未找到官方代码或 checkpoint。

### R07 — Song, Yeo, Moon (2021)

**元数据与阅读级别。** *Approach Control Concepts and Optimal Vertiport Airspace Design for Urban Air Mobility (UAM) Operation*. Kyowon Song, Hwasoo Yeo, Jung-Ho Moon. *International Journal of Aeronautical and Space Sciences*, 22, 982–994. DOI [10.1007/s42405-020-00345-9](https://doi.org/10.1007/s42405-020-00345-9)，[Springer](https://link.springer.com/article/10.1007/s42405-020-00345-9)，[作者页面](https://pure.kaist.ac.kr/en/publications/approach-control-concepts-and-optimal-vertiport-airspace-design-f/)。ABSTRACT_OR_SECTIONS。

**问题。** 研究 vertiport approach-control concept、holding point/circle 和空域半径设计。

**方法。** 比较 SBA 与 BQA，讨论 holding geometry 和 VTCA 控制点。

**作者报告的发现。** 摘要报告 SBA 适合空间受限条件，BQA 更强调安全。

**作者明确的局限。** accessible preview 中未找到明确 limitation 段落，不能据此声称全文没有限制。

**针对本项目的适用边界（本次分析）。** 是结构化 arrival geometry 先例；不等于本项目三环动态构型算法，尤其不能把它写成 author code 已复现。

**代码/权重。** 未找到官方代码或 checkpoint。

### R08 — Song, Yeo (2021)

**元数据与阅读级别。** *Development of optimal scheduling strategy and approach control model of multicopter VTOL aircraft for urban air mobility (UAM) operation*. Kyowon Song, Hwasoo Yeo. *Transportation Research Part C*, 128, 103181. DOI [10.1016/j.trc.2021.103181](https://doi.org/10.1016/j.trc.2021.103181)，[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0968090X21001972)，[作者 PDF](https://intellectum.unisabana.edu.co/server/api/core/bitstreams/45ed2510-2685-467d-a606-132c46293243/content)。ABSTRACT_OR_SECTIONS；索引中可读到部分全文段落，直接下载返回 403。

**问题。** 研究 vertiport holding point、arrival scheduling 和多机 approach control。

**方法。** FCFS、AETA、GA，以及 BQA、SBA、SBAM approach models。

**作者报告的发现。** 作者报告 100 schedules/scenarios、600 simulations、36,000 vehicle records；一项比较中 SBA+GA OTP 85.9%，BQA 没有 LOS risk。

**作者明确的局限。** 本轮无法访问完整 limitations/future-work 段落。

**针对本项目的适用边界（本次分析）。** 是 scheduling/structured approach 家族背景，不应替代 common backend，也不能把新项目 fixed_rule 直接标成 SBA/BQA。

**代码/权重。** 未找到官方代码或 checkpoint。

### R09 — Song (2023)

**元数据与阅读级别。** *Optimal Vertiport Airspace and Approach Control Strategy for Urban Air Mobility (UAM)*. Kyowon Song. *Sustainability*, 15(1), 437. DOI [10.3390/su15010437](https://doi.org/10.3390/su15010437)，[MDPI](https://www.mdpi.com/2071-1050/15/1/437)，[可读 PDF 镜像](https://www.wisdomlib.org/uploads/journals/mdpi-sust/2023-volume-15-issue-1--2071-1050-15-1-437-.pdf)。FULL_TEXT（PDF 镜像）。

**问题。** 研究多分支 vertiport airspace 和 approach control。

**方法。** BBQA 基于 BQA，允许调整 landing order；full search 选择 holding-circle radius、holding-point arrangement 和 branch count，并比较 FCFS/GA 及 BQA/SBA/SBAM。

**作者报告的发现。** BBQA 被报告为保持 BQA safety、提高相对 SBA 的 OTP；一项报告的 BBQA OTP 为 86.3%。

**作者明确的局限。** 机型 operating specifications 尚未准确组织；未来需要 mixed aircraft types 和 multi-vertiport。

**针对本项目的适用边界（本次分析）。** 是 static/structured geometry 与排序参照，不能直接提供本项目动态三环的 common executable trajectory。

**代码/权重。** 未找到官方代码或 checkpoint。

### R10 — Shao, Shao, Lu (2021)

**元数据与阅读级别。** *Terminal area control rules and eVTOL adaptive scheduling model for multi-vertiport system in urban air Mobility*. Quan Shao, Mengxue Shao, Yang Lu. *Transportation Research Part C*, 132, 103385. DOI [10.1016/j.trc.2021.103385](https://doi.org/10.1016/j.trc.2021.103385)，[ScienceDirect 摘要](https://www.sciencedirect.com/science/article/abs/pii/S0968090X21003843)，[作者页面](https://faculty.nuaa.edu.cn/ly/en/lwcg/16599/content/35552.htm)。ABSTRACT_OR_SECTIONS；全文需购买。

**问题。** 多 vertiport terminal area 的 control rule、路径和分布式 sequencing。

**方法。** multi-ring 结构、基于 backpressure 的 junction control、考虑 power constraint 和 dynamic priority 的 distributed scheduling。

**作者报告的发现。** 摘要报告大流量下相对于传统方法的效率/安全优势；density 比 volume 更重要；低于 120 flights/hour 时，效率随数量/密度变化约在 10.3% 内。

**作者明确的局限。** 由于全文受限，作者限制段落未核验；不能推断其在线 geometry、代码或硬件不存在。

**针对本项目的适用边界（本次分析）。** 是 multi-vertiport backpressure/queue 先例；不能把本项目规则控制器冒充为 Shao 的精确实现。

**代码/权重。** primary/author 页面未找到官方代码或 checkpoint。

### R11 — Zheng et al. (2025)

**元数据与阅读级别。** *Optimized Airspace Structures and Sequencing Method for Urban Logistics Droneport*. Yuan Zheng, Die Li, Zhou Shen, Chenglong Li, Zhaoxuan Zhang. *Aerospace*, 12(3), 257 (2025). DOI [10.3390/aerospace12030257](https://doi.org/10.3390/aerospace12030257)，[MDPI](https://www.mdpi.com/2226-4310/12/3/257)。ABSTRACT_OR_SECTIONS。

**问题。** 城市物流 droneport 的上升/下降空域结构和排序。

**方法。** 分离升/降区、Hungarian cost、battery/priority 约束。

**作者报告的发现。** 论文摘要/索引报告总起降时间下降 34.8%、低电量等待下降 47.3%（相对于其建模 funnel baseline）。

**作者明确的局限。** 这些数值来自特定 droneport/funnel 场景，并非一般化安全或容量保证；作者完整限制段落本轮未完全读取。

**针对本项目的适用边界（本次分析）。** 可作空域结构和低 SOC 排序的主题邻居；其 logistics/droneport 语义需要与本项目 passenger/landing/service 语义分开。

**代码/权重。** 未核验到作者公开代码或 checkpoint。

### R12 — Chang et al. (2026)

**元数据与阅读级别。** *Vertiport arrival airspace design, analysis, and capacity assessment: Integrating the point merge system into UAM*. Zhening Chang, Minghua Hu, Jianan Yin, Hua Xie, Shuce Wang. *Aerospace Science and Technology*, article 113051. DOI [10.1016/j.ast.2026.113051](https://doi.org/10.1016/j.ast.2026.113051)。出版社将其列入 2026 年 11 月 issue，早期在线日期未核验。ABSTRACT_OR_SECTIONS。

**问题。** 研究将 point merge system 融入 UAM vertiport arrival airspace design、分析和 capacity assessment。

**方法。** 摘要层面可见 vPMS/Multi-vPMS geometry/topology rules，以及 demand-aware differential angular sizing 与 PMS/arrival geometry 的比较。

**作者报告的发现。** 摘要/索引报告：相对于统一扩大角度，demand-aware differential sizing 在相近 footprint 下更有利；对 heterogeneous wake fleet mix，balanced geometry 的 capacity 最低。

**作者明确的局限。** 本轮只读到摘要/索引，完整 limitation 段落未核验；不能据此宣称“首次”或补写作者未声明的限制。

**针对本项目的适用边界（本次分析）。** 是 demand-aware geometry 邻居；不能把动态 angular design 等同于本项目三环 online reconfiguration，也不能把本项目固定 geometry 结果归因于该方法。

**代码/权重。** 未找到官方代码或 checkpoint。

### R13 — Wei, Xiao, Zhao, Yuan (2025)

**元数据与阅读级别。** *Optimization Methods for Unmanned eVTOL Approach Sequencing Considering Flight Priority and Traffic Flow Imbalance*. Zhiqiang Wei, Xinlong Xiao, Xiangling Zhao, Jie Yuan. *Drones*, 9(6), 396. DOI [10.3390/drones9060396](https://doi.org/10.3390/drones9060396)，[MDPI](https://www.mdpi.com/2504-446X/9/6/396)。ABSTRACT_OR_SECTIONS；索引段落可读，直接 publisher page 在本轮受限。

**问题。** 研究优先级和流量不均衡下的 eVTOL approach sequencing。

**方法。** AIR、SRMA、ARAM 结构，0–1 multi-objective optimization，包含 inflow/outflow、continuity、flow balance 和 conflict avoidance。

**作者报告的发现。** 作者报告 ARAM 相对 BQA/AIR 在 balanced flow 下平均 holding time 分别减少 8.4%/7.6%，在 imbalanced flow 下相对 BQA 减少 6.5%。

**作者明确的局限。** 实际 airside/safety interval 标准不可得，多个参数为假设；求解时间和复杂度也依赖其场景。

**针对本项目的适用边界（本次分析）。** 是 structured sequencing 邻居；当前主课程 task_priority_enabled=false，不能把其 priority-aware 结果直接与中性任务课程混合。

**代码/权重。** 未找到作者公开代码或 checkpoint。

### R14 — Deniz et al. (2025)

**元数据与阅读级别。** *Deep reinforcement learning approach to autonomous landing of eVTOL vehicles for advanced air mobility*. Sabrullah Deniz, Ali Barzegar, Fatma Yamac Sagirli, Katherine Vasiloff, Zhenbo Wang. *Control Engineering Practice*, 164, 106519. DOI [10.1016/j.conengprac.2025.106519](https://doi.org/10.1016/j.conengprac.2025.106519)，[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0967066125002813)。ABSTRACT_OR_SECTIONS。

**问题。** eVTOL single/multi-aircraft autonomous landing。

**方法。** 三个同心多区圆形空域、DQN/DDPG/Actor-Critic、AirSim/Unreal，以及小型无人机 HIL 展示。

**作者报告的发现。** 摘要报告 collision avoidance、landing sequencing、simulation 和 sim-to-real 展示。

**作者明确的局限。** 作者明确要求对大型 operational eVTOL、高保真动力学和更严格 safety/regulation 谨慎解释。

**针对本项目的适用边界（本次分析）。** 是 multi-zone landing 主题邻居；不能把其三圈语义强制映射到原始 Waltz，也不能把底层 landing policy 当成当前 terminal flow baseline。

**代码/权重。** 未找到官方代码或 checkpoint。

### R15 — Iriarte et al. (2026)

**元数据与阅读级别。** *Speed-Based Tactical Deconfliction of Multiple Aircraft Around a Vertiport Through a Conservative Airspace Discretization Algorithm and Constraint Programming*. Imanol Iriarte, Estela Nieto Ramos, Iñaki Iglesias, Josu Del Río, Joseba Lasa, Santi Vilardaga, Sergi Lucas, Basilio Sierra. *Aerospace*, 13(6), 519. DOI [10.3390/aerospace13060519](https://doi.org/10.3390/aerospace13060519)，[MDPI](https://www.mdpi.com/2226-4310/13/6/519)，[作者/机构页](https://scienceportal.tecnalia.com/en/publications/speed-based-tactical-deconfliction-of-multiple-aircraft-around-a-/)。ABSTRACT_OR_SECTIONS。

**问题。** vertiport 周边多机 tactical deconfliction。

**方法。** 3-D discretized airspace、相邻 cell 检查、high-level plan 到 synthetic trajectory、OR-Tools CP-SAT；主要动作是 speed reduction。

**作者报告的发现。** Rule B 在测试场景中报告 100% conflict detection 且无 false negative；Rule A 在 25-aircraft case 漏检冲突。测试规模到 25 aircraft。

**作者明确的局限。** 没有 intrinsic feasibility guarantee；head-on、持续平行、同 cell simultaneous entry、throughput saturation 和 minimum-speed saturation 时 speed-only 方法可能不可行。

**针对本项目的适用边界（本次分析）。** 是 tactical trajectory/execution layer；如果接入共同 safety backend，必须报告其 speed-only authority、feasibility 和 timeout 行为。

**代码/权重。** 未找到公开作者代码或 checkpoint；primary supplementary material 可见视频。

### R16 — Li, Gao, Qu (2026)

**元数据与阅读级别。** *DeepSet-Enhanced Edge Reinforcement Learning for UAV Autonomous Landing and Takeoff at Portable Vertiports*. Zhirun Li, Tiying Gao, Chengyi Qu. IEEE ICNC 2026, 775–780. DOI [10.1109/ICNC68183.2026.11416848](https://doi.org/10.1109/ICNC68183.2026.11416848)，[官方 PDF](https://www.conf-icnc.org/2026/papers/p775-li.pdf?title=PDF+file)，[作者页面](https://scholarscommons.fgcu.edu/esploro/outputs/conferenceProceeding/DeepSet-Enhanced-Edge-Reinforcement-Learning-for-UAV/99385932896006570)，[官方代码](https://github.com/SECNetLabUNM/UAV_Landing_Takeoff_Control)。ABSTRACT_OR_SECTIONS。

**问题。** portable vertiport 中多 UAV landing/takeoff 的 edge inference 与安全。

**方法。** CTDE、permutation-invariant DeepSet encoder、safety-constrained PPO-GAE、edge/MEC 部署。

**作者报告的发现。** 论文报告 98% mission success、超过 90% collision reduction 和 100 ms edge-to-UAV latency budget；可见结果还报告不同 UAV 数量下的安全/成功率。

**作者明确的局限。** future work 包括 CBF、现实网络条件、federated learning、transformer encoder 和更强 baselines。

**针对本项目的适用边界（本次分析）。** 是 learned terminal coordination/landing 邻居，但 portable 5×5×5 m 场景和 acceleration actions 与当前 dynamic ring flow 不同。

**代码/权重。** 官方仓库可见；README 的 How-to-use/results 区域仍标为待完成，未在可见文件中核验 pretrained checkpoint。

### R17 — KrisshnaKumar et al. (2023)

**元数据与阅读级别。** *Fast Decision Support for Air Traffic Management at Urban Air Mobility Vertiports Using Graph Learning*. Prajit KrisshnaKumar, Jhoel Witter, Steve Paul, Hanvit Cho, Karthik Dantu, Souma Chowdhury. *2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 1580–1585 (2023). DOI [10.1109/IROS55552.2023.10341398](https://doi.org/10.1109/IROS55552.2023.10341398)，[作者记录](https://researchconnect.buffalo.edu/en/publications/fast-decision-support-for-air-traffic-management-at-urban-air-mob/)，[arXiv preprint](https://arxiv.org/abs/2308.09075)。FULL_TEXT（arXiv HTML；publisher metadata verified）。

**问题。** UAM Vertiport Schedule Management 的实时 landing/takeoff/hover/charge 决策。

**方法。** 两个图分别表示 vertiport spots 和 aircraft；GCN+MLP/PPO；AirSim/Unreal，4 架 scaled-down vehicle，比较 graph RL、vanilla RL、random 和 FCFS。

**作者报告的发现。** 作者报告 graph RL 在 50 episodes 中相对 vanilla RL/FCFS 的 collision、delay 和 battery 指标表现；一个摘要可见的 collision 均值约为 0.3，并非零碰撞保证。

**作者明确的局限。** 模拟始终维持 4 架，departure aircraft 离开后由另一架接替；环境假设无通信损失和显式不确定性。

**针对本项目的适用边界（本次分析）。** 是 conditional learned terminal coordination baseline；其 actions 是 landing/takeoff/hover/charge allocation，而不是本项目 ring geometry + landing intent。不得在没有作者权重时重训后冒称原始基线。

**代码/权重。** 本轮未找到作者公开代码或 checkpoint。

### R18 — Chen, Evans, Brittain, Wei (2024)

**元数据与阅读级别。** *Integrated Conflict Management for UAM With Strategic Demand Capacity Balancing and Learning-Based Tactical Deconfliction*. Shulu Chen, Antony D. Evans, Marc Brittain, Peng Wei. *IEEE Transactions on Intelligent Transportation Systems*, 25(8), 10049–10061 (2024). DOI [10.1109/TITS.2024.3351049](https://doi.org/10.1109/TITS.2024.3351049)，[arXiv v1 HTML](https://arxiv.org/html/2305.10556v1)，[arXiv DOI](https://doi.org/10.48550/arXiv.2305.10556)。FULL_TEXT（arXiv HTML）。公开仓库 [bluesky-DCB](https://github.com/Shulu-Chen/bluesky-DCB)。

**问题。** 将 strategic demand-capacity balancing 与 tactical conflict deconfliction 结合用于 UAM。

**方法。** Section III 为多资源 DCB MILP/heuristic；Section IV 为 CTDE shared PPO tactical speed policy 与 game-theoretic grouping。

**作者报告的发现。** 作者报告 DCB conditioning 相比 MARL-only/rule 条件改善 density-conditioned safety/efficiency。

**作者明确的局限。** deterministic ETA 可能被 tactical speed changes 破坏；未来需要 stochastic ETA 处理。

**针对本项目的适用边界（本次分析）。** 公开仓库只能作为 partial public DCB+rule artifact；不得声称 MARL tactical 已发布或已复现。若未来纳入，只能单独作为 strategic conditioning 比较并明确 public subset。

**代码/权重。** README 明确写明只包含 DCB 和 rule-based tactical deconfliction，RL tactical methods 尚未公开。sequencing/compute_DCB.py 依赖 Gurobi；supports/generate_scn_files.py 生成 hybird 场景；rule_based_tactical.py 是原生 headless tactical entry。未发现 paper RL checkpoint。

### R19 — Chen et al. (2024)

**元数据与阅读级别。** *Integrated Arrival and Departure Management for Urban Air Mobility Vertiport Operations in the New York City Airspace*. Shulu Chen, Peng Wei, Paul Krois, Joseph Block, Paul Cobb, Andre Louis-Ferdinand, Jeff Hyman. ICNS 2024. DOI [10.1109/ICNS60906.2024.10550765](https://doi.org/10.1109/ICNS60906.2024.10550765)（本轮按 primary PDF/ICNS program 交叉核验，仍标作 DOI metadata pending），[primary PDF](https://bpb-us-w2.wpmucdn.com/web.seas.gwu.edu/dist/9/15/files/2024/04/shulu_ICNS24_vertiport-20b5f57d17169f7b.pdf)。FULL_TEXT。

**问题。** NYC airspace 中 vertiport arrival/departure integrated management。

**方法。** V-HATT MILP、STA slot assignment、FATO 150 s interval、energy earliest/latest bounds、arrival priority、tactical speed holding 与 human operator。

**作者报告的发现。** 120 aircraft 计算约 53.33 s；高需求时 average air delay 约 130.9 s，最大约 1018 s。

**作者明确的局限。** future work 包括 surface taxi/parking integration 和 HIL；高需求 delay 仍然显著。

**针对本项目的适用边界（本次分析）。** 是 integrated arrival/departure scheduling 邻居；当前项目 Phase A 只做 landing-only，不能把该 paper 的 departure/parking 语义提前混入主任务。

**代码/权重。** 未找到官方代码或 checkpoint。

### R20 — Espejo-Diaz, Alfonso-Lizarazo, Montoya-Torres (2023)

**元数据与阅读级别。** *A heuristic approach for scheduling advanced air mobility aircraft at vertiports*. Julian Alberto Espejo-Diaz, Edgar Alfonso-Lizarazo, Jairo R. Montoya-Torres. *Applied Mathematical Modelling*, 123, 871–890 (2023). DOI [10.1016/j.apm.2023.07.009](https://doi.org/10.1016/j.apm.2023.07.009)，[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0307904X23003062)，[作者记录](https://espace2.etsmtl.ca/id/eprint/29941/)。ABSTRACT_OR_SECTIONS。

**问题。** vertiport scheduling 中 throughput 和 target-deviation 的优化。

**方法。** 两个 MILP，显式 TLOF/taxi/gate blocking/separation resource；heuristics 处理较大规模。

**作者报告的发现。** 论文摘要/索引报告在其测试中最多约 10 aircraft 可求最优，并展示大规模 heuristic。

**作者明确的局限。** 只有摘要与索引段落可访问，作者完整 limitations 未核验。

**针对本项目的适用边界（本次分析）。** 是资源调度参照，不能作为 dynamic airspace geometry 或 executable free-flight 基线。

**代码/权重。** 未找到官方代码或 checkpoint。

### R21 — Taye et al. (2024)

**元数据与阅读级别。** *Safe and Scalable Real-Time Trajectory Planning Framework for Urban Air Mobility*. Abenezer G. Taye, Roberto Valenti, Akshay Rajhans, Anastasia Mavrommati, Pieter J. Mosterman, Peng Wei. *Journal of Aerospace Information Systems*, 21(8), 641–650 (2024). DOI [10.2514/1.I011381](https://doi.org/10.2514/1.I011381)，[primary PDF](https://abenezertaye.com/papers/safe-and-scalable-real-time-trajectory-planning-framework-for-urban-air-mobility.pdf)，[作者页](https://abenezertaye.com/research)，[官方文档](https://chapter-1.readthedocs.io/en/latest/index.html)，[官方代码](https://github.com/Abenezergirma/Chapter-1)。FULL_TEXT。

**问题。** UAM 多机实时轨迹规划的可扩展性和安全执行。

**方法。** MDP/FastMDP、data-driven reachable overapproximation、DryVR、shield/reward shaping；fixed-wing 6-DOF、25–70 m/s、3-D free-flight。

**作者报告的发现。** 在作者报告的最多 32 aircraft 场景，NMAC mean 的 baseline/shield/reward-shaping 数值分别约 9.96/2.61/1.48；这些不是 zero-safety 证明。

**作者明确的局限。** hard constraints 作为 reward 时仍可能违反；shield 可能 deadlock，需要激进 maneuver，作者给出最高约 70 度 roll/gamma 条件。

**针对本项目的适用边界（本次分析）。** 这是 conditional trajectory-layer safety baseline，不是 traffic-flow/geometry actor；若使用必须固定相同 dynamics/resource budget 并分开报告 shield intervention。

**代码/权重。** 官方代码和文档公开；本轮未执行，不能写成已通过项目 qualification。

### R22 — Vascik, Hansman (2019)

**元数据与阅读级别。** *Development of Vertiport Capacity Envelopes and Analysis of Their Sensitivity to Topological and Operational Factors*. Parker D. Vascik, R. John Hansman. AIAA SciTech 2019. DOI [10.2514/6.2019-0526](https://doi.org/10.2514/6.2019-0526)，[MIT primary record](https://dspace.mit.edu/entities/publication/d81d7c70-4c4f-4ab7-91ee-7baca6977cc6)。ABSTRACT_OR_SECTIONS。

**问题。** vertiport topology/operation 对 capacity envelope 的影响。

**方法。** IP capacity envelopes，考虑 pad/taxi/gate/parking、service 和 arrival/departure ratios。

**作者报告的发现。** 摘要说明不同 topology、service ratio 和 arrival/departure pair 会改变容量 envelope。

**作者明确的局限。** PDF 下载在本轮返回 405，作者 limitation 段落未读取；static capacity 分析而不是 policy performance 是本次分析的分类，不标作作者明确限制。

**针对本项目的适用边界（本次分析）。** 是 capacity envelope 背景，用于固定 footprint 和 resource capacity 参照，不是动态 geometry actor。

**代码/权重。** 未找到官方代码或 checkpoint。

### R23 — Guerreiro et al. (2020)

**元数据与阅读级别。** *Capacity and Throughput of Urban Air Mobility Vertiports with a First-Come, First-Served Vertiport Scheduling Algorithm*. Nelson M. Guerreiro, George E. Hagen, Jeffrey M. Maddalon, Ricky W. Butler. AIAA Aviation 2020. DOI [10.2514/6.2020-2903](https://doi.org/10.2514/6.2020-2903)，[NASA NTRS record](https://ntrs.nasa.gov/citations/20205001421)，[NASA PDF](https://ntrs.nasa.gov/api/citations/20205001421/downloads/AIAA_Vertiport_Analysis_final_20200507.pdf)。ABSTRACT_OR_SECTIONS（正文可下载但本轮未逐页全文阅读）。

**问题。** FCFS vertiport resource reservation 下的 capacity/throughput。

**方法。** pad、parking space 和 timeline reservation 的 fast-time model，并比较 queueing/simulation。

**作者报告的发现。** 多数测试 configuration 的 peak throughput/capacity 至少 80%；FCFS 在资源数增加时出现资源使用 inefficiency。

**作者明确的局限。** 正文本轮未逐页阅读，作者明确 limitation 未核验。

**针对本项目的适用边界（本次分析）。** clean FCFS 是本项目内部强控制参照的来源之一；不能宣称当前 fixed_rule 就是该论文的完整复现。

**代码/权重。** 未找到官方代码或 checkpoint。

### R24 — Onat et al. (2024)

**元数据与阅读级别。** *Evaluating eVTOL Network Performance and Fleet Dynamics through Simulation-Based Analysis*. Emin Burak Onat, Vishwanath Bulusu, Anjan Chakrabarty, Mark Hansen, Raja Sengupta, Banavar Sridhar. AIAA SciTech 2024 Forum. DOI [10.2514/6.2024-0336](https://doi.org/10.2514/6.2024-0336)，[arXiv record](https://arxiv.org/abs/2312.02505)，[VertiSim code](https://github.com/eminb61/vertisim)。ABSTRACT_OR_SECTIONS。

**问题。** 多 vertiport network 中 fleet、passenger、energy flow 和 infrastructure tradeoff。

**方法。** VertiSim multi-agent discrete-event simulator；同时模拟 passenger/aircraft/energy，包含 heuristic dispatch/charge policy。

**作者报告的发现。** 19 个场景、每日 2,834 passengers；fleet size 增加可减少 passenger delay，但会增加 repositioning/operational expense；距离改变 utilization 和 delay。

**作者明确的局限。** 摘要/sections 本轮未核验作者 limitation 段落。

**针对本项目的适用边界（本次分析）。** fleet/network/charging/profit 研究与当前 landing-flow 主线不同；不纳入最终交通控制 panel。

**代码/权重。** 官方 VertiSim 仓库公开；README 的单配置 runner 可见，批量 runner 标为未维护；未见训练 checkpoint。

### R25 — Yedavalli et al. (2022)

**元数据与阅读级别。** *SimUAM: A Comprehensive Microsimulation Toolchain to Evaluate the Impact of Urban Air Mobility in Metropolitan Areas*. Pavan Yedavalli, Emin Burak Onat, Xin Peng, Raja Sengupta, Paul Waddell, Vishwanath Bulusu, Min Xue. AIAA Aviation 2022 Forum. DOI [10.2514/6.2022-3837](https://doi.org/10.2514/6.2022-3837)，[NASA primary PDF](https://ntrs.nasa.gov/api/citations/20220006884/downloads/20220006884_Xue_Aviation2022_manuscript_final.pdf)，[UC Berkeley record](https://escholarship.org/uc/item/5709d8vr)。FULL_TEXT。

**问题。** metropolitan UAM ground-air travel behavior、network value 和 deployment assessment。

**方法。** MANTA ground traffic、VertiSim vertiport/pedestrian、Fe3 aerial microsimulation 的组合 toolchain。

**作者报告的发现。** ground-air interface 纳入后，UAM market 相对于 static transfer-time assumptions 在各 network design 均下降；通过 transfer-time demand balancing 可改善 benefited trips。

**作者明确的局限。** 本轮未在可访问文本中核验作者 limitation 段落；system-level multi-modal scope 是本次适用边界判断。

**针对本项目的适用边界（本次分析）。** 是系统仿真背景，不是同一粒度的交通控制基线。

**代码/权重。** 未找到 SimUAM 全部 toolchain 的官方可复现实验仓库或 checkpoint；VertiSim 有独立公开仓库（R24）。

### R26 — Paul, Witter, Chowdhury (2024)

**元数据与阅读级别。** *Graph Learning-based Fleet Scheduling for Urban Air Mobility under Operational Constraints, Varying Demand & Uncertainties*. Steve Paul, Jhoel Witter, Souma Chowdhury. ACM SAC 2024, 638–645. DOI [10.1145/3605098.3635976](https://doi.org/10.1145/3605098.3635976)，[arXiv](https://arxiv.org/abs/2401.04851)，[作者/出版记录](https://researchconnect.buffalo.edu/en/publications/graph-learning-based-fleet-scheduling-for-urban-air-mobility-unde/)。FULL_TEXT（arXiv）。

**问题。** 多 vertiport fleet 的在线 schedule/destination，在变动 demand、capacity、weather route closure 和 downtime 下运行。

**方法。** graph capsule conv-nets、transformer demand/fare encoding、multi-head attention decoder、PPO。

**作者报告的发现。** 8 vertiports/40 aircraft unseen scenarios 中，作者报告 daily averaged profit 优于 random 和 genetic-algorithm-derived solutions，并声称执行速度约快 1000 倍。

**作者明确的局限。** 摘要没有给出可直接引用的 limitation 段落；fleet scheduling/profit scope 归入本次适用边界。

**针对本项目的适用边界（本次分析）。** fleet/network/profit 与当前 Phase A 不同，排除最终 traffic baseline；不能把 graph fleet policy 当作 terminal intent policy。

**代码/权重。** 本轮未找到作者公开代码或 checkpoint。

### R27 — Onat et al. (2024，SSRN)

**元数据与阅读级别。** *Urban Air Mobility Fleet Management Under Uncertainty: A Deep Reinforcement Learning Approach*. Emin Burak Onat, Shangqing Cao, Raja Sengupta, Mark Hansen. SSRN 5072017 (posted 2024-12-25). DOI [10.2139/ssrn.5072017](https://doi.org/10.2139/ssrn.5072017)，[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5072017)，[官方代码](https://github.com/eminb61/RL-UAM-Framework)。ABSTRACT_OR_SECTIONS。

**问题。** UAM fleet dispatch、routing、charging 的 stochastic passenger demand 管理。

**方法。** PPO+LSTM，VertiSim，联合 dispatch/routing/charging。

**作者报告的发现。** 摘要报告 two-vertiport case 的 optimality gap 约 12.5%；hub/8-spoke/2,000 passenger/35 aircraft scenario 中 spill rate 约 3.67% 对比 greedy 12.27%。

**作者明确的局限。** 公开摘要没有完整 limitations 段落。

**针对本项目的适用边界（本次分析）。** 明确属于 network/fleet/charging/profit，排除最终 terminal-flow panel。

**代码/权重。** 官方代码公开；本轮未执行，未核验 checkpoint。

### R28 — Murthy, Neogi, Bharadwaj (2022)

**元数据与阅读级别。** *Scheduling for Urban Air Mobility using Safe Learning*. Surya Murthy, Natasha A. Neogi, Suda Bharadwaj. EPTCS 371, 86–102 (2022). DOI [10.4204/EPTCS.371.7](https://doi.org/10.4204/EPTCS.371.7)，[arXiv](https://arxiv.org/abs/2209.15457)，[官方代码](https://github.com/suryakmurthy/UAM_scheduler)。FULL_TEXT（arXiv）。

**问题。** UAM scheduling 在 hard/soft trip deadlines 下的安全学习。

**方法。** model-based safe learning、MCTS、EDF、non-preemptive scheduling，与 VI/random MCTS 比较。

**作者报告的发现。** 论文报告在其 scheduling model 中的 safe-learning 对照结果。

**作者明确的局限。** 本轮未单独摘录作者 limitation 段落；hard-request scheduling scope 与 pad motion/3-D trajectory 的区分属于本次适用边界。

**针对本项目的适用边界（本次分析）。** 作为安全调度背景可保留，不作为 terminal executable controller。

**代码/权重。** 官方仓库公开；本轮未运行，未核验 checkpoint。

### R29 — Mesbah, Roshanian, Ginchev (2026)

**元数据与阅读级别。** *Obstacle Avoidance for Multirotor Urban Air Mobility via Prediction-Based Control Barrier Functions*. Ali Mesbah, Jafar Roshanian, Dimitar Ginchev. *Engineering Proceedings*, 121(1), 30 (2026). DOI [10.3390/engproc2025121030](https://doi.org/10.3390/engproc2025121030)，[MDPI primary](https://www.mdpi.com/2673-4591/121/1/30)。FULL_TEXT（primary 摘要/正文索引）。

**问题。** multirotor UAM 在动态/不确定障碍环境中的 collision avoidance。

**方法。** Prediction-Based Control Barrier Function，通过 escape-path prediction 构造 safety margin，配合高 relative-degree、input-constrained model。

**作者报告的发现。** 作者用仿真 obstacle-avoidance scenario 展示 PB-CBF 的安全过滤作用和 prediction mechanism。

**作者明确的局限。** 采用 3-DoF planar simplified UAM model；结果是 simulation safety-filter evidence，不是多机 terminal capacity。

**针对本项目的适用边界（本次分析）。** 只在论文主张需要 CBF/PB-CBF 的 trajectory-layer comparison 时作为 safety filter 参照；同一 shield 语义必须共享给所有 traffic methods。

**代码/权重。** 未找到官方代码或 checkpoint。

### R30 — Safaoui, Vinod et al. (2024)

**元数据与阅读级别。** *Safe multi-agent motion planning under uncertainty for drones using filtered reinforcement learning*. Sleiman Safaoui, Abraham P. Vinod, Ankush Chakrabarty, Rien Quirynen, Nobuyuki Yoshikawa, Stefano Di Cairano. *IEEE Transactions on Robotics*, 40, 2529–2542 (2024). DOI [10.1109/TRO.2024.3387010](https://doi.org/10.1109/TRO.2024.3387010)，[MERL primary record](https://www.merl.com/publications/TR2024-048)，[arXiv preprint](https://arxiv.org/abs/2311.00063)。FULL_TEXT（MERL PDF/arXiv 前置版本）。

**问题。** uncertain、cluttered workspace 中多无人机安全 motion planning。

**方法。** single-agent RL 产生 reference plan，再由 convex optimization、chance constraints、set-based constrained control safety filter 修正；讨论 state/control constraints、recursive feasibility 和概率安全。

**作者报告的发现。** 作者报告 numerical simulations 和 quadrotor experiments，强调过滤器可实时执行并在不确定性下提供高概率碰撞避免。

**作者明确的局限。** 作者指出 centralized coordination 可能增加 communication/computation burden，并将 decentralized setting、nonlinear dynamics 和更多 multi-agent RL evaluation 列为 future work。

**针对本项目的适用边界（本次分析）。** 是 executable trajectory/safety-filter 家族，非 terminal flow/geometry policy；若纳入必须报告 requested/reference/filter-applied action 与 intervention。

**代码/权重。** 本轮未找到该 TRO 论文的公开作者 checkpoint 或可直接复现实验仓库。其较早的 ICRA 版本另列为 R37；不得把 R37 的 deterministic/known-workspace 结果当作本条的 stochastic extension。

### R31 — Bae, Lee, Lee (2022)

**元数据与阅读级别。** *Marker-Based 3D Position-Prediction Algorithm of Mobile Vertiport for Cabin-Delivery Mechanism of Dual-Mode Flying Car*. Hyansu Bae, Jeongwook Lee, Kichang Lee. *Electronics*, 11, 1837 (2022). DOI [10.3390/electronics11121837](https://doi.org/10.3390/electronics11121837)，[PDF](https://pdfs.semanticscholar.org/78f3/e443b456b15d594f28fad337275aa2bbbfe1.pdf)。FULL_TEXT。

**问题。** mobile vertiport 与 dual-mode flying car 的 landing-site localization、position correction 和 cabin docking。

**方法。** marker/depth-camera relative localization、position prediction、docking correction；论文报告 50 次移动与 20 次 docking 实验。

**作者报告的发现。** 作者报告 mobile station 与 AEV 相对定位/对接误差落在其机械可纠正范围内。

**作者明确的局限。** 本轮未核验作者 limitation 段落；移动站/dual-mode cabin delivery 与 UAM arrival traffic scheduling 的区分属于本次适用边界。

**针对本项目的适用边界（本次分析）。** 属于 moving-platform/landing infrastructure 主题，和当前固定 footprint 的 terminal flow 不同；排除最终比较。

**代码/权重。** 未找到官方代码或 checkpoint。

### R32 — Goudarzi et al. (2024)

**元数据与阅读级别。** *Airspace Routing and Automation for Vertiports*. Hirad Goudarzi, Mickey Li, Duncan Hine, Thomas Richardson, Arthur G. Richards. EuroGNC 2024. DOI [10.82124/CEAS-GNC-2024-016](https://doi.org/10.82124/CEAS-GNC-2024-016)，[primary PDF](https://eurognc.ceas.org/archive/EuroGNC2024/pdf/CEAS-GNC-2024-016.pdf)。FULL_TEXT。

**问题。** vertiport airspace routing、route interlock 和 automated separation。

**方法。** fixed route、block reservation、railway-signalling 类 interlock、handcrafted/counter-threaded helix；含 simulation 和 multi-drone physical separation。

**作者报告的发现。** 论文展示 route interlock 对 vertiport routing/physical separation 的作用。

**作者明确的局限。** 本轮未核验作者 limitation 段落；固定 route/interlock 和 human-in-the-loop 是方法描述，不标作作者明确限制。

**针对本项目的适用边界（本次分析）。** 可作为结构化空域控制背景，不能作为动态 geometry learned baseline。

**代码/权重。** 未找到官方代码或 checkpoint。

### R33 — Wille, Reichmuth, Borille (2026)

**元数据与阅读级别。** *Determining vertiport airside capacity using discrete event simulation*. Eva Wille, Johannes Reichmuth, Giovanna Miceli Ronzani Borille. *Journal of the Air Transport Research Society*, 6, 100109 (2026). DOI [10.1016/j.jatrs.2026.100109](https://doi.org/10.1016/j.jatrs.2026.100109)，[出版社记录](https://www.jatrs.org/)。ABSTRACT_OR_SECTIONS。

**问题。** vertiport airside practical capacity、TLOF/stand/taxi interaction 和 delay。

**方法。** discrete-event simulation，比较 stands/TLOFs、taxi process 和 queueing policy。

**作者报告的发现。** taxi 在 TLOF-limited 条件下显著影响效率；priority-based queue 可降低 arrival delay 而不降低 practical capacity。

**作者明确的局限。** 作者指出 evolving aircraft specifications/procedures 带来的不确定性。

**针对本项目的适用边界（本次分析）。** 是 capacity/resource background；不提供当前 3-D executable trajectory policy。

**代码/权重。** 未找到官方代码或 checkpoint。

### R34 — Nagrare, Lieb (2026)

**元数据与阅读级别。** *Throughput and Capacity Analysis of a Vertiport with Taxing and Parking Levels*. Samiksha Rajkumar Nagrare, Joonas Lieb. *Aerospace*, 13(1), 109 (2026). DOI [10.3390/aerospace13010109](https://doi.org/10.3390/aerospace13010109)，[DLR record](https://elib.dlr.de/222339/)。ABSTRACT_OR_SECTIONS。

**问题。** taxi/parking levels、charging 和 vertiport throughput/capacity。

**方法。** simulation-based capacity analysis，考虑 hourly demand、parking spots、charging 和 scheduling。

**作者报告的发现。** charging time 和 parking spots 对 capacity/scheduling 有显著影响；priority scheduling 可改变 delay/throughput tradeoff。

**作者明确的局限。** ABSTRACT_OR_SECTIONS 阅读未核验完整 limitation 段落；参数/动态 geometry 的区分属于本次适用边界。

**针对本项目的适用边界（本次分析）。** 用作 resource-service 语义背景，不能与 free-flight policy 直接做同任务性能结论。

**代码/权重。** 未找到官方代码或 checkpoint。

### R35 — Schweiger, Preis (2022)

**元数据与阅读级别。** *Urban Air Mobility: Systematic Review of Scientific Publications and Regulations for Vertiport Design and Operations*. Karolin Schweiger, Lukas Preis. *Drones*, 6(7), 179 (2022). DOI [10.3390/drones6070179](https://doi.org/10.3390/drones6070179)，[MDPI](https://www.mdpi.com/2504-446X/6/7/179)。FULL_TEXT。

**问题。** 系统梳理 UAM ground infrastructure、vertiport design/operation 和 regulation。

**方法。** 对 2016–2021 的 49 篇 Scopus-listed publications 做 systematic review，并按 airspace、design、network、capacity、ground operation、safety 等分类。

**作者报告的发现。** airspace operation、vertiport design 和 throughput/capacity 是主要主题；ground infrastructure 与 airside/ground/service streams 的耦合仍分散。

**作者明确的局限。** 文献截止至 2021；非正常约束和 contingency 在多数研究中覆盖不足。

**针对本项目的适用边界（本次分析）。** 用于术语和 gap 定位，不作为实验 baseline。

**代码/权重。** 不适用；未发现控制器代码。

### R36 — Sengupta et al. (2025)

**元数据与阅读级别。** *Urban Air Mobility Research Challenges and Opportunities*. Raja Sengupta, Vishwanath Bulusu, Chams Eddine Mballo, Emin Burak Onat, Shangqing Cao. *Annual Review of Control, Robotics, and Autonomous Systems*, 8, 407–431 (2025). DOI [10.1146/annurev-control-022823-031353](https://doi.org/10.1146/annurev-control-022823-031353)，[Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev-control-022823-031353)。FULL_TEXT。

**问题。** 综述 aircraft、vertiport、network、ATM/autonomy 和 UAM cross-layer challenges。

**方法。** 综述不同 UAM research areas，讨论 deterministic/stochastic capacity、gate/parking bottleneck 和 high-density autonomy。

**作者报告的发现。** 综述指出 vertiport capacity 同时受 TLOF、gate/parking、charging 和随机 operation 影响，且高密度 ATM 的 scalability 是持续问题。

**作者明确的局限。** 本文是综述；本轮未将综述中的研究范围描述改写成单一作者实验限制。

**针对本项目的适用边界（本次分析）。** 用作研究定位和交叉层 gap 背景，不计入 controller performance panel。

**代码/权重。** 不适用。

### R37 — Vinod et al. (2022)

**元数据与阅读级别。** *Safe multi-agent motion planning via filtered reinforcement learning*. Abraham P. Vinod, Sleiman Safaoui, Ankush Chakrabarty, Rien Quirynen, Nobuyuki Yoshikawa, Stefano Di Cairano. *IEEE International Conference on Robotics and Automation (ICRA)*, 7270–7276 (2022). DOI [10.1109/ICRA46639.2022.9812259](https://doi.org/10.1109/ICRA46639.2022.9812259)，[MERL primary record](https://www.merl.com/publications/TR2022-053)。FULL_TEXT（作者记录/PDF）。

**问题。** cluttered environment 中多 agent motion planning 的硬安全约束。

**方法。** 用 single-agent RL 产生到达目标的 motion plan，再用 convex optimization filter 处理 agent-agent 和 static-obstacle collision avoidance、state/control constraints；作者记录了 deterministic、known-workspace 的早期版本语义。

**作者报告的发现。** 作者报告该过滤器可实时执行，并通过数值仿真和六架 Crazyflie 硬件实验展示多机安全规划效果。

**作者明确的局限。** 本轮未在 R37 primary record 中单独核验完整 limitation 段落；R30 后续论文把该研究线描述为 deterministic/known-workspace preliminary work，本记录据此区分假设，不把后续 stochastic guarantee 倒写入本条。

**针对本项目的适用边界（本次分析）。** 是 trajectory-layer safety-filter 参照，不是 terminal geometry 或 landing-flow policy；若接入，必须让所有 traffic methods 使用同一安全执行和 intervention log。

**代码/权重。** MERL 论文记录未提供可直接用于本项目的作者 checkpoint 或 terminal-flow harness。

### R38 — Pan et al. (2025 arXiv / 2026 RA-L，去重一项)

**元数据与阅读级别。** *Robust Trajectory Generation and Control for Quadrotor Motion Planning with Field-of-View Control Barrier Certification*. Lishuo Pan, Mattia Catellani, Lorenzo Sabattini, Nora Ayanian. *IEEE Robotics and Automation Letters*, 11(2), 1682–1689 (2026). DOI [10.1109/LRA.2025.3630888](https://doi.org/10.1109/LRA.2025.3630888)，[作者/机构 primary record](https://iris.unimore.it/handle/11380/1395608)，[arXiv preprint](https://arxiv.org/abs/2502.01009)，[官方代码 FoVMPC](https://github.com/LishuoPan/fovmpc)。FULL_TEXT（arXiv v2；arXiv 与正式 RA-L 版按同一工作合并）。

**问题。** 通信丢失和状态估计不确定条件下，多机器人如何维持有限 field of view 中的邻机可见性与安全导航。

**方法。** continuous-time robust trajectory/control 与 CLF-CBF 认证，提出 MPC-CBF 离散近似；以高阶 CBF 线性 surrogate 和 sequential quadratic programming 求解。

**作者报告的发现。** 作者在 10 robots 仿真和 2 架定制 UAV 实验中展示了该方法的 trajectory/control 行为，并报告其对 temporary tracking loss 的处理。

**作者明确的局限。** 本轮未从全文中摘录可直接引用的独立 limitation 段落；官方仓库列出 CPLEX 12.10 与 ROS1 依赖，这是运行条件事实，不改写成作者局限。

**针对本项目的适用边界（本次分析）。** 是可执行轨迹/视场安全层候选，不能替代 terminal flow、LZ service 或 dynamic geometry policy；若接入需冻结同一 sensing、communication 和 solver budget。

**代码/权重。** 官方仓库公开，README 列出 CPLEX 12.10、ROS1；本轮未运行，也未核验 pretrained checkpoint。

### R39 — KrisshnaKumar et al. (2023, SciTech)

**元数据与阅读级别。** *Graph Learning Based Decision Support for Multi-Aircraft Take-Off and Landing at Urban Air Mobility Vertiports*. Prajit KrisshnaKumar, Jhoel Witter, Steve Paul, Karthik Dantu, Souma Chowdhury. *AIAA SciTech 2023 Forum*. DOI [10.2514/6.2023-1848](https://doi.org/10.2514/6.2023-1848)，[作者/出版记录](https://researchconnect.suny.edu/en/publications/graph-learning-based-decision-support-for-multi-aircraft-take-off/)，[arXiv preprint](https://arxiv.org/abs/2302.05849)。ABSTRACT_OR_SECTIONS。该条与 R17 的 IROS 2023 出版物分开记录。

**问题。** UAM vertiport 中多 aircraft take-off/landing 的实时 schedule decision support。

**方法。** UAM-VSM 用两个图分别表示 vertiport physical spots 和 aircraft，使用 GCN/graph RL 生成 hover/cruise、idle/take-off、land 等决策，并与基础 RL/随机选择对照。

**作者报告的发现。** 作者在缩比多旋翼仿真中按 delay、collision 和 battery consumption 评价 graph policy，并报告其相对于所列基础/随机对照的适用性。

**作者明确的局限。** 本轮可访问文本未单独核验作者 limitation 段落。

**针对本项目的适用边界（本次分析）。** 是 learned terminal scheduling 邻居；代码或权重可见性不足以把它自动纳入当前 landing-flow 面板，且不能与 R17 或本项目 fixed_rule 混写为同一算法。

**代码/权重。** 本轮未找到作者公开的完整实现或 checkpoint。

### R40 — Goldschmid, Ahmad (2024)

**元数据与阅读级别。** *Reinforcement Learning based Autonomous Multi-Rotor Landing on Moving Platforms*. Pascal Goldschmid, Aamir Ahmad. *Autonomous Robots*, 48, article 13 (2024). DOI [10.1007/s10514-024-10162-8](https://doi.org/10.1007/s10514-024-10162-8)，[Springer full text](https://link.springer.com/article/10.1007/s10514-024-10162-8)，[官方代码](https://github.com/robot-perception-group/rl_multi_rotor_landing)。FULL_TEXT。

**问题。** 多旋翼自主降落到二维移动平台，并降低训练时间、提高成功率以及向真实硬件迁移。

**方法。** 将纵向/横向控制拆为两个 1-D Double Q-learning agent，采用由平台运动学推导的 state discretization 和 sequential curriculum/transfer；使用 Gazebo 11、RotorS、ROS Noetic，并做真实硬件展示。

**作者报告的发现。** 在无噪声观测的 simulation comparison 中，每个 evaluation scenario 使用 150 次 landing trials；作者报告相对 DDPG baseline 的训练时间约降低 80%，RPM 0.4 的成功率为 99%（高 8 个百分点），RPM 1.2 的成功率为 99%（高 26 个百分点）。两个比较场景与 cascaded PI 表现相当，另有独立的真实硬件统计；上述百分比仅指无噪声仿真比较。

**作者明确的局限。** 未来工作包括控制 vertical movement 和 yaw，以及 inclined/flying platform；作者还讨论 sim-to-real 中质量/惯量、低层控制器、状态估计和 ground effect 的影响。

**针对本项目的适用边界（本次分析）。** 代码可复现的是 moving-platform landing controller，属于低层/落点任务；它不处理本项目的 terminal geometry、multi-UAV arrival flow 或 LZ service，排除最终 traffic panel。

**代码/权重。** 官方 ROS/Gazebo 仓库和实验数据入口公开；本轮未运行，未把其 agent 权重迁移到本项目。

### R41 — Pruekprasert, Nakadai (2025)

**元数据与阅读级别。** *Safe Arrival Scheduling at Constraint Waypoints in UAM Corridors*. Sasinee Pruekprasert, Shinji Nakadai. *AIAA SciTech 2025 Forum*. DOI [10.2514/6.2025-2232](https://doi.org/10.2514/6.2025-2232)，[arXiv version](https://arxiv.org/abs/2605.31243)。ABSTRACT_OR_SECTIONS。

**问题。** 在 UAM corridor 的 constrained waypoints (CWPs) 交换/批准 arrival schedule，以支持纵向 self-separation。

**方法。** 计算每对 vehicle 在 CWP 所需的 minimum arrival-time gap；分别基于 NMAC separation rule 和 Responsibility-Sensitive Safety (RSS) rule。

**作者报告的发现。** NMAC-based gap 在遵守 speed limit 的正常情形可防止 collision；RSS-based gap 可处理超速 emergency，但正常情形需要更大的 gap、会降低 flow。作者用 numerical simulations 验证这些差异。

**作者明确的局限。** 作者明确说明 NMAC 方法在超速时不保证安全，RSS 方法在正常情形的更大 gap 会牺牲 traffic flow。

**针对本项目的适用边界（本次分析）。** 是 corridor waypoint scheduling rule，适合用作 arrival-gap/structured-rule 主题参照；它不是 dynamic circular geometry、LZ service 或 executable 3-D terminal policy。

**代码/权重。** 本轮未找到官方代码或 checkpoint。

### R42 — Pruekprasert, Nakadai, Nishinari (2026)

**元数据与阅读级别。** *Safety-Assured Arrival Scheduling in Sequential UAM Corridor Sections under Speed and Separation Constraints*. Sasinee Pruekprasert, Shinji Nakadai, Katsuhiro Nishinari. arXiv:2605.23333 (2026)，[arXiv primary](https://arxiv.org/abs/2605.23333)。论文标注为 accepted IFAC；本轮未核验正式出版 DOI。ABSTRACT_OR_SECTIONS。

**问题。** 异质 speed limits 的 sequential corridor sections 中，如何计算满足纵向 separation 的 ETA gap。

**方法。** 根据各 section 的 speed bounds、travel times 和 separation distance，推导 sufficient ETA-gap condition，并以 decreasing-speed corridor numerical simulation 验证。

**作者报告的发现。** 在规定的 section travel times 和 speed limits 下，计算 gap 可维持所有 sections 的安全间隔；仿真中相对于 unscheduled operation 改善 spacing 和 traffic flow。

**作者明确的局限。** 结论以 prescribed section travel times、speed limits 和纵向 corridor separation 为前提；本轮未核验其最终出版版本的完整 limitation 段落。

**针对本项目的适用边界（本次分析）。** 是可解释的 ETA/queue safety rule 邻居，不能直接替代本项目的 dynamic ring geometry、multi-LZ service 或 3-D trajectory execution。

**代码/权重。** 本轮未找到官方代码或 checkpoint。

### R43 — Vinod, Safaoui, Summers, Yoshikawa, Di Cairano (DOI 2024 / record 2025，时间未决)

**元数据与阅读级别。** *Decentralized, Safe, Multi-agent Motion Planning for Drones Under Uncertainty via Filtered Reinforcement Learning*. Abraham P. Vinod, Sleiman Safaoui, Tyler Summers, Nobuyuki Yoshikawa, Stefano Di Cairano. *IEEE Transactions on Control Systems Technology*, 32(6), 2492–2499；MERL record lists January 2025, while the DOI is [10.1109/TCST.2024.3433229](https://doi.org/10.1109/TCST.2024.3433229). [MERL primary record](https://www.merl.com/publications/TR2024-136)。ABSTRACT_OR_SECTIONS；record year/issue timing remains unresolved in this review. This is a separate TCST work and is not arXiv:2311.00063/R30.

**问题。** 在 stochastic uncertainty 下为多无人机生成 decentralized、safe、real-time motion plans。

**方法。** 将 single-agent RL 与 distributionally robust convex optimization、buffered Voronoi cells 和 temporal discounting 结合；作者摘要称对 mean trajectories 给出 recursive-feasibility/probabilistic-safety guarantees，并通过 drone experiments 验证。

**作者报告的发现。** 摘要报告仿真和 physical experiments 中的安全与 performance 结果，并强调相对于直接安全过滤的 conservativeness mitigation。

**作者明确的局限。** 本轮只读到 MERL abstract/record，未核验完整 limitation 段落。

**针对本项目的适用边界（本次分析）。** 是 decentralized trajectory/safety-layer 参照，不能替代 terminal geometry、LZ admission 或 landing-flow scheduler；R43 不应与 R30 或 R37 的假设和数值混用。

**代码/权重。** 本轮未确认官方代码或作者 checkpoint。

## 代码与可复现实验状态

| 记录 | 官方代码 | 权重/运行事实 | 当前状态 |
|---|---|---|---|
| R01 Waltz | [TUD_RL](https://github.com/MarWaltz/TUD_RL) | 官方 UAM actor 权重尚未获得 | 原始源码静态核验和 diagnostic 已有；Policy-IV/performance 未验证 |
| R02 Yang | [Multi_MCTS_Guidance_Separation_Assurance](https://github.com/xuxiyang1993/Multi_MCTS_Guidance_Separation_Assurance) | 无训练权重依赖 | 有受限 headless wrapper；正式 common-harness 结果 pending |
| R03 Saxena | [Vertiport_schedule](https://github.com/rrsaxena92/Vertiport_schedule) | MATLAB 原始文件；Python/SciPy/HiGHS port 已实现并通过 7 tests/7 fixtures；无 checkpoint | MATLAB cross-solver parity 与 2026 正式版完整 formulation pending |
| R16 Li | [UAV_Landing_Takeoff_Control](https://github.com/SECNetLabUNM/UAV_Landing_Takeoff_Control) | visible tree 未核验 checkpoint | README 的 results/how-to-use 仍 incomplete |
| R18 DCB | [bluesky-DCB](https://github.com/Shulu-Chen/bluesky-DCB) | 只有 DCB+rule；RL tactical 未公开 | 只能称 public partial artifact |
| R21 Taye | [Chapter-1](https://github.com/Abenezergirma/Chapter-1) | 轨迹层代码公开；未运行 | conditional trajectory-layer baseline |
| R24 VertiSim | [vertisim](https://github.com/eminb61/vertisim) | 配置 runner 可见；未执行 | fleet/network simulator |
| R27 Onat fleet | [RL-UAM-Framework](https://github.com/eminb61/RL-UAM-Framework) | checkpoint 未核验 | fleet/charging scope |
| R28 Murthy | [UAM_scheduler](https://github.com/suryakmurthy/UAM_scheduler) | 未运行 | safe scheduling background |
| R38 Pan | [fovmpc](https://github.com/LishuoPan/fovmpc) | 官方仓库公开；未运行，checkpoint 未核验 | conditional trajectory/safety layer |
| R40 Goldschmid | [rl_multi_rotor_landing](https://github.com/robot-perception-group/rl_multi_rotor_landing) | 官方 ROS/Gazebo 仓库公开；未运行 | moving-platform landing scope |
| R43 TCST | 未确认官方代码 | checkpoint 未确认 | separate conditional trajectory/safety layer |
| 其余条目 | 未在本轮 primary/author source 找到官方实现 | 无 checkpoint 事实 | 文献参照或条件候选 |

当前项目已有的 native diagnostic 证据只证明 wrapper/source-level execution 和 schema/conservation 等检查；“complete”表示有限 schedule 被消费且没有 active/unspawned，而不表示 service completion、safety、capacity 或 learned performance。本文不把 smoke/diagnostic 结果写成论文性能。

三套 native label 也不能直接互换：Yang 的 `goal/empty` 是其搜索终止语义，Saxena 的 `feasible` 是 MILP schedule 语义，Waltz 的 `full schedule consumed` 是有限输入被消费语义；它们都不等同于共同 backend 的 safety 或 service completion。

已观察到的 `verification_20261002` 证据为：服务器端 `pytest_final.stdout` 显示 17 passed、14 warnings、2.03 s、exit 0；九个有限诊断案例均 exit 0，`complete=true`、`truncated=false`、`active=0`、`unspawned=0`，并通过 conservation 检查及 CSV 的 `(time, uid)` 唯一性和 `sigma∈{−1,+1}` 检查。案例为 `wave_upstream12requests`（12 requests/229 ticks）、`study30_off`（30/607）、`study30_on`（30）、`study5_noise10`、`study5_noise20`、`study5_noise100`（各 5）、`poisson_off`（22/616）、`poisson_on`（22/726）和 `wave_paper800`（12/214）。这些诊断使用 constant-zero heading；spawn/removal、service completion、landing safety 和 performance 均未由此得到结论。证据文件位于 `D:/7_codespace/training_outputs/terminal/waltz2024/verification_20261002/checks.json`；Policy-IV、官方 actor checkpoint 和 common-harness publication performance 仍 pending。

同一日期的 native source/fixture checks 还观察到：Yang wrapper 的 9 tests PASS，使用 isolated Shapely 2.0.6；`crossing_natural` 生成 10/10 goals、0 active，在 774 steps 和 1172 MCTS decisions 后自然终止（13.46 s）；`hex_original` 为 120 steps、156 decisions、12 generated、0 goals、12 active、`truncated=true`（0.60 s）；`hex_two_stage_original` 为 120 steps、138 decisions、12 generated、0 goals、12 active、`truncated=true`（0.54 s）。Saxena Python port 的 7 tests 和 7 tiny fixtures PASS，最大 row residual 为 `3.84e-12`、最大 integrality residual 为 `9.46e-13`；七个 artifact 名称及 objective 分别为 `arrival_final2=20.02794117647059`、`departure_final2=18.027941176470588`、`tat_final2=52.055882352941175`、`mixed_final3=166.52720588235294`、`legacy_final3=166.52720588235297`、`capacity3_final2=225.22352941176393`、`capacity1_two_final=161.31176470588258`。其中 `capacity1_two_final` 使用 2 TAT requests、最大 virtual-gate occupancy 为 1，row residual 为 `3.13e-13`、integrality residual 为 0；旧的 `capacity1_final` 使用 4 TAT requests 并返回 infeasible，不能与上述 optimal fixture 混写。运行环境观察到 Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4；native MATLAB/Octave 不可用，2026 正式模型尚未转录。Yang/Saxena 结果是 wrapper/fixture validation，不等于 publication performance、共同场景服务完成或 safety。输出分别位于 `D:/7_codespace/training_outputs/terminal/yang2020_mcts/verification_20261002/` 和 `D:/7_codespace/training_outputs/terminal/saxena2024_schedule/verification_20261002/`；MATLAB parity、common adapter 和跨方法 publication performance 仍 pending。

## 面向本项目的公平比较协议

### 1. 三种执行标签分开

- native_original：保留论文原始 observation、action decoder、geometry、termination 和 resource semantics。
- adapted_common：为共同 backend 做必要接口适配，并明确列出新增信息、动作映射和 shield intervention。
- ood_zero_shot：原始 checkpoint 不重训，直接进入共同场景；结果只表示迁移行为，不能单独证明同任务 learned-algorithm superiority。

适配登记先写清保留项与新增项，原始包和 common adapter 不混称：

| 方法 | 原始语义保留 | 明确的 adapted extension |
|---|---|---|
| Saxena MILP | 保留 MILP resource order、slot objective；Phase A 先用 arrival-only view | node time 映射为 common RELEASE target/WAIT-until-slot，再由 common trajectories recheck feasibility；加入 SOC/deadline、service `30/120/10/60 s` 和 LZ compatibility 属于适配扩展 |
| Yang–Wei MCTS | 保留 sequential MCTS、forward rollouts、decision cadence 和 search budget | heading proposal 经过 common motion bounds/shield；source 2-D/hex route 与 project 3-D 不同，先做 matched single-altitude bridge；重写成 ring WAIT/RELEASE state/action 后只能称 MCTS-based adaptation |
| Waltz Policy IV | 保留 actor observation normalization/history、continuous heading 和 rule-based entry signal | 不新增 feature 或改变 neural action 后宣称 zero-shot；one-LZ bridge 优先，多 pad/SOC assignment 通过相同外部 scheduler 属于适配；需要官方 actor weights |

本表是 adapter/registration 约束；本轮没有实现 common adapter，也没有把 native 包或诊断运行写成正式跨方法实验。`fixed_best`、`fixed_largest`、rule 以及 geometry-only/flow-only/joint controls 用于检验 dynamic benefit 的来源。

不得把 Waltz heading output 映射成项目的 WAIT/RELEASE，也不得把原始 free-flight 强制为三环后仍命名为 original。原始单高度 circular airspace 与本项目三维 ring architecture 的差异要单独标记；若可执行，应增加共同 altitude/geometry ablation，而不是直接归因于意图学习。

### 2. 共同外生输入和物理预算

所有方法使用同一份预生成 request tape：request id、arrival time、gate、speed、SOC、service type、service duration。planner 的随机数消耗不得改变下一架飞机的外生样本；每 UID 的 exogenous RNG 与 planner RNG 分离。不得丢弃 blocked request、teleport gate 或按方法改变 arrival cohort。外部 queue/admission 规则共同固定，reassigned gate 单独记录。

两种方法共享同一 airspace footprint envelope、允许高度区间、dt/physics substeps、Vmax、turn/acceleration/jerk/downwash 和 delay budget。物理横向加速度按 a_lat = v × omega = v² × kappa 计算，其中 omega 是角速度、kappa 是曲率；不可为某个方法单独放宽。信息预算也要冻结：forecast horizon、邻机可见性、resource availability、SOC/service/type 字段和 prediction budget 不能只给 proposed method；不得给一个方法 full future tape，除非明确标为 oracle lower bound。

当前项目自身的比较记录还要固定这些已定义的语义：主 reset 流程为 `flow_from_empty`，不能只用 `warm_start_rings`；requirements 19.3 已列出 tuned fixed-geometry、strong backpressure-minslack 和 rolling MILP-MPC 对照。two-ring 对照需同时匹配 storage、footprint 和 service resource，并分别记录 geometry-only、flow-only、joint 变体，避免把资源变化混入几何或流量效应。当前 design 21.1 的 `footprint_seconds` 是逐 tick 的 `sum(actual R²) / sum(initial R²)` 成本代理，不是物理 land area 或 3-D volume；energy 仍是未校准到 EH216 实际 Wh 的代理量，也没有 joint recursive-feasibility theorem。`OutcomeLedger` 区分 `touchdown_time` 与 service-complete 的 `completion_time`，比较终点应使用后者。

### 3. 两层场景

**Bridge tier。** 一个 LZ、60 s uniform service、充足 energy、task-neutral；相同 R=800 circular footprint、四个 gate、匹配 dynamics/altitude budget，用来隔离方法接口和基础 arrival 行为。Waltz source profile 的 physical spawn radius 是 1000、observation normalization denominator 也是 1000；paper profile 使用 physical R=800 但保留 denominator=1000。只改 spawn radius 并让 native denominator 自动变成 800 会破坏论文 normalization，因此这两个量要显式解耦。原始策略的 single-altitude/single-LZ 语义保留，不能用 one-LZ 结果声称已经验证项目三环服务调度。

**Research tier。** 共用真实项目场景：LZ/service 30/120/10/60 s，mission probabilities 0.45/0.25/0.10/0.20，task_priority_enabled=false，SOC/energy guard 共用。该 tier 的最终 YAML/config 必须先冻结实际值；rate 0.5/0.8/1.1 mu_ref、低/中/过载以及 1200 s above + 1200 s below burst 是 design suggestions，不是已经注册的 protocol。持续峰值窗口必须覆盖 geometry dwell，短于 dwell 的 burst 不能作为 sustained capacity 证据。mu_ref 应来自同一个 independent fixed-rule measured reference capacity，不能按方法归一化。

### 4. 运行单位、指标和统计

有限 cohort 要显式记录 completed、failed、diverted、unresolved、censored；连续流要在相同 warm-up 后报告 service completions/hour、backlog change 和稳定区间。service completion 以 service release 为终点，不以进入 200 m 或圆圈 entry 代替。

共同报告 collision 10 m、项目标准默认 separation threshold 80 m、protected floor/NMAC 40 m、最小 separation sweep、forbidden entry/resource conflict、normal/emergency/failure/diversion/unresolved、request-to-service delay、external/internal wait、p95/max、makespan、service utilization、energy、shield intervention 和 solver latency/timeout/feasible fallback。当前项目标准没有把 100 m 作为另一个默认 threshold；任何额外阈值都必须作为单独冻结的 sweep 记录。不能比较跨环境 raw reward，不能只报 endpoint minimum distance，也不能 cherry-pick 成功 episode。

使用 paired CRN episode statistics；30 个独立 evaluation episodes 仅是当前 design suggestion，不是已注册 protocol，也不宣称代表文献共同规模。single training seed 的 checkpoint 不支持 training-seed variability claim；development seeds 和后续 held-out/preregistered evaluation 必须分开。所有结论在 root protocol 完成前标为 pending。

## 方法家族与 Citation-to-Claim Map

| Claim family | 证据条目 | 可支持的事实 |
|---|---|---|
| 自组织/在线多机到达 | R01、R02、R04 | shared free-flight policy、MCTS guidance、结构化 terminal arrival 的不同控制边界 |
| 资源/排序/容量 | R03、R05、R06、R08–R13、R19–R20、R22–R23、R33–R34、R41–R42 | MILP/heuristic/FCFS、pad/gate/TLOF/parking、density/priority/flow imbalance 对 capacity/等待的影响 |
| learned terminal/landing decisions | R14、R16–R18、R39–R40 | DRL landing、DeepSet/GCN/PPO、partial DCB+rule、moving-platform landing；均有不同 scope/资源语义 |
| trajectory safety | R15、R21、R29–R30、R37–R38、R43 | speed-only deconfliction、reachable/shield、PB-CBF、filtered RL；不能自动替代 terminal flow policy |
| structured routing/interlock | R32 | fixed route、block reservation、interlock 和 physical separation 的结构化参照 |
| moving-platform/infrastructure landing | R31、R40 | mobile landing-site localization 或 moving-platform controller；不等同于 fixed-footprint terminal flow |
| network/fleet/charging | R24–R28 | passenger/aircraft/energy/fleet/network scheduling；与 landing-only 主线分开 |
| geometry/capacity review | R07、R09–R12、R22、R35–R36 | vertiport geometry、capacity envelope、reviewed gaps；不等同于 executable common baseline |

## 当前边界与未决事实

- 当前 final target panel 仍是 R01/R02/R03；R01 checkpoint、R03 cross-solver parity 和所有 common-harness publication performance 仍 pending。
- R18 的公开部分明确是 DCB+rule；不能把 paper 中未公开的 MARL tactical 当作已获得实现。
- R24–R28 的 fleet/network/charging/profit 代码和 R31、R40 的 moving-platform/landing 研究不匹配当前 Phase A landing-flow claim；它们保留为背景，不进入同任务最终面板。
- R15、R21、R29/R30、R37/R38、R43 是 safety/trajectory layer；若接入交通比较，必须给所有交通方法同一 common shield，并同时记录 requested、accepted、executed 和 intervention。
- Graph GCN/PPO、CBF/MPC、FastMDP 等只有在论文主张与当前比较问题一致、且接口/权重/执行语义可核验时，才可被列为条件候选；没有 checkpoint 时不得重训后冒充原论文结果。
- 本轮没有对上述文献重新训练任何模型，也没有产生新的跨论文性能结论。
