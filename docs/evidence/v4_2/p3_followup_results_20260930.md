# P3 follow-up 修复与复核事实（2026-09-30）

## 范围与集成检查

本页记录本轮冻结评估、被动诊断 artifact 和测试事实；本轮未训练、未运行 optimizer、未使用 held-out，也未进行 qualification。回归测试和 control full 均不构成 P3 性能改善结论。

## Frozen always-RELEASE gate control 终态

这是 `newly_run` 的冻结、离策略 gate-control full，不是训练、PPO、Teacher、optimizer、held-out 或 formal qualification。它复用 `9` 个 `preexisting_artifact` reference scenes（`6` 个环境 profile、`9` 个 fixed keyed pairs），不是 `216` 个独立请求实验；source `3fe7+a82`、88D、K=3、legacy eligibility、真实 proposal `q`、物理配置和 native safety 保持不变，仅将 legacy `actor_mask=true` 的 gate 固定为 RELEASE，其他行保持 canonical WAIT。artifact 为 `.tmp/p3_gate_control_full_proof.json`，server full SHA256 为 `fa856f3980be848c8cb2fd5971115188a9e2f5303c37d8699b44631e57a58403`，`FULL_OK`、`9/9 PAIR_OK`、9/9 natural、technical/safety/source-RNG checks 均通过；`PPO_ALLOWED=false`，held-out 未使用。

逐 pair 的控制结果与相对既有 legacy reference 的 delta 如下。`N/E/F/D/U` 依次为 normal/emergency/failed/diverted/unresolved；括号内为 `control/reference`。wait、request elapsed 和 energy 均为 control 减 reference，unknown 不填零。

| pair | N/E/F/D/U control/reference | Δwait s | Δelapsed s | Δenergy |
|---:|---|---:|---:|---:|
| 1 | 22/2/0/0/0 / 23/1/0/0/0 | -239.352168125554 | -98 | -100.978648204367 |
| 5 | 22/2/0/0/0 / 20/4/0/0/0 | +18.339647352861 | +143 | +40.892863295369 |
| 9 | 22/2/0/0/0 / 20/4/0/0/0 | -66.722642900639 | -11 | -26.622650716003 |
| 13 | 20/4/0/0/0 / 20/3/1/0/0 | -1873.686726820392 | -1683 | -1724.690972830649 |
| 17 | 23/1/0/0/0 / 21/3/0/0/0 | -433.175318573591 | -507 | -564.411033581651 |
| 21 | 22/2/0/0/0 / 20/4/0/0/0 | -345.431682372297 | -270 | -285.375711314852 |
| 14 | 19/5/0/0/0 / 19/5/0/0/0 | -526.172317326245 | -574 | -621.337151864353 |
| 15 | 19/4/1/0/0 / 21/3/0/0/0 | +951.717922176038 | +650 | +815.757062620658 |
| 16 | 21/3/0/0/0 / 16/7/1/0/0 | -1590.833149623624 | -1315 | -1436.236666501356 |

Aggregate control/reference values were `N=190/180`, `E=25/34`, `F=1/2`, `D=0/0`, `U=0/0`; admitted was `216/216`, successful completed was `215/214`, and resolved was `216/216`. Wait was `25124.700815134504 / 29230.017251347945 s`, request elapsed was `79035 / 82700 s`, and energy was `64612.22804454819 / 68515.23095364538`. Raw reward excluding rejection penalty was `4315.902567907697 / 4132.633739495724` (delta `+183.26882841197312`); the rejection penalty was `-1140.55 / -165.8` (delta `-974.75`), so total reward was `3175.3525679076965 / 3966.8337394957234` (delta `-791.4811715880269`). These are observed return and business tradeoffs, not a root-cause attribution.

The registered acceptance review rejected the candidate: pair `15` introduced a new failed arrival `21`, despite aggregate normal increasing `180→190`, failed decreasing `2→1`, successful completed increasing `214→215`, wait decreasing to `25124.700815134504 s`, request elapsed decreasing to `79035 s`, and energy decreasing to `64612.22804454819`. The retained new failure row was UID `108`, mission `cargo`, initial SOC `0.5758906587575819`, request time `150`, terminal time `1013`, phase `ring_outer`, outcome `failed_energy`, and wait `807.8147075862395 s`. In the `env877104` cross-keyed traces, arrival `13` was `emergency_landed` under all four control keyed runs; the two previous legacy runs that failed are retained as comparison evidence, not as an all-scene feasibility claim.

The control tests were `27` unique cases (`6` gate-control, `11` opportunity, `10` forced-WAIT); code/docs commit `3a3e1ba` was not pushed because no remote is configured. The `p3-gate` heartbeat was deleted after the terminal full artifact, and no active background process remains.

## Passive pair-15 failure replay 终态 (`REPLAY_OK`)

The root-approved passive exact replay ran once for each arm of fixed pair `15` (`env877104`, keyed `9303103`) with source runner `CF8C08AC204BAFC2BFE6D8E3A661FFD779BC7ADCF4DBBFC58F7F51B6ED8F9794`, source checkpoints `3fe7+a82`, unchanged configuration, seeds, and 1800-step cap. Server PID `2097784` exited; the terminal snapshot was `2026-09-30T16:54:53Z`. Legacy ran `724` ticks and control ran `1077` ticks. Request identity, outcome counts, terminal rows, energy, wait, reward, model/RNG invariants, and hard-safety checks were exact for both arms; the replay artifact is `/home/user/training_outputs/terminal/p3_followup_20260930/gate_failure_replay_20261001/failure_replay.json` with local proof `.tmp/p3_gate_failure_replay_launch_proof.json` and status `REPLAY_OK`. `PPO_ALLOWED=false`, held-out was unused, and no formal promotion occurred.

Target arrival `21` was cargo: legacy became `emergency_landed` at terminal `606` after `ring_outer→ring_middle→ring_inner`, while control remained in `ring_outer` for `777` waiting frames and ended as `failed_energy` at terminal `1013`. Legacy had one positive emergency attempt, which succeeded at tick `438` with SOC `0.39723546555721995`; control had `120` actual attempt-positive emergency calls, with first safety-block evidence at tick `414`, SOC `0.39966843653535883`, and first energy-block evidence at tick `664`, SOC `0.23300176986869217`. The control blocker counts were `58` safety-block calls and `69` energy-block calls; these overlap and must not be summed. Native call totals were `336` legacy and `899` control, including early returns and retries; only attempt-positive deltas count as attempts. These are observations from this pair's causal traces only, not a global root-cause conclusion.

The first launch proof's postprocessing parser incorrectly read the release field path and reported the fields missing; the collector and replay were unchanged. Corrected raw verification used `trace.steps[*].targets[str(arrival_id)].after.release_decision`: control target `21` in `ring_outer` has `777` release frames, `51` `safety_reject` + `661` `safety_wait` + `64` `emergency_energy_guard` + `1` terminal-null; legacy has `15` release frames, `10` `safety_reject` + `4` `safety_wait` + `1` `approved`, with the approved step `312`, proposal `10`, shortlist `[10,11,12]`. The prior proof v1 is retained. Because a new live server hash verification was not available, this correction is sourced from the synchronized mirror proof and is not a new server hash check. `last_safety_decision` labels are retained as stale labels and are not interpreted as the immediate failure reason. The final exact regression command over replay, gate-control, opportunity, and forced-WAIT tests produced `32 passed in 4.07s` (`5` new replay tests plus the original `27`), without duplicate counting; native fixture v3 also passed with no optimizer. Commits `d153f9e` and `99b6bd9` are local only, with no remote/upstream; the replay heartbeat was deleted and no background process remains.
随后完成 server 直读 cross-check：更新后的 proof `.tmp/p3_gate_failure_replay_launch_proof.json` SHA256 为 `914a4b235916a64f165ebf8073ea5f74828b18cf2af12f1b45e00bb414e651ac`；replay summary 为 `REPLAY_OK`、mismatch `0`、model/RNG/safety checks 为 `true`，server trace SHA 为 legacy `d1b71ef6abfe32feb84c009961b974e369935377eef85db1dd01c10dcd607745`、control `1b185b39f7e26cd0f8d534e3573001bf70bbbd04afb760516f1bf7af8364ec1b`。上段关于首次 parser 的 mirror 来源是直读前的历史 provenance 记录；本次 server 直读复核与之不矛盾。

## Proposal-support control full 终态 (`FULL_OK`; development candidate)

冻结 proposal-support control 的 full 比较已经完成。固定 9 个场景全部自然终态，0 error、0 censor；每个 pair 的 24 个请求、source/model-caller RNG、ledger、energy、wait、request identity 和 hard-safety 检查均通过。该运行只评估冻结 off-policy proposal-support control，没有训练、optimizer、teacher、PPO 或 heldout。

服务器产物为 `.tmp/p3_proposal_support_full_proof.json`、`.tmp/p3_proposal_support_full_perpair.csv`；full source SHA 为 `0178f6cef9083babab207d1fbd0e71cc50b4ebd90ffc2e4601caaeb07a8668b5`，PID 2102330 已于 2026-10-01 01:49:36（Asia/Shanghai）退出。9 个场景来自 6 个固定环境 profile 和 9 个 keyed 场景，不能视作 216 个独立样本。

| pair | scene | ticks | control N/E/F/D/U | reference N/E/F/D/U | Δwait (s) | Δelapsed (s) | Δenergy |
|---:|---|---:|---|---|---:|---:|---:|
| 1 | 877101/9303101 | 722/694 | 22/2/0/0/0 | 23/1/0/0/0 | +252.53942549349722 | +282 | +324.9373252078476 |
| 5 | 877102/9303101 | 799/795 | 20/4/0/0/0 | 20/4/0/0/0 | +261.4727465143619 | +363 | +322.5319456248826 |
| 9 | 877103/9303101 | 711/668 | 20/4/0/0/0 | 20/4/0/0/0 | +367.11554965350433 | +351 | +397.4807547823866 |
| 13 | 877104/9303101 | 758/863 | 19/5/0/0/0 | 20/3/1/0/0 | -578.3562047072569 | -374 | -359.9870170506765 |
| 17 | 877105/9303101 | 685/828 | 21/3/0/0/0 | 21/3/0/0/0 | -504.171103904288 | -497 | -542.7246012361056 |
| 21 | 877106/9303101 | 836/752 | 19/5/0/0/0 | 20/4/0/0/0 | -176.0746084480652 | -154 | -150.76281530593496 |
| 14 | 877104/9303102 | 742/742 | 19/5/0/0/0 | 19/5/0/0/0 | -246.58688277428746 | -274 | -276.1631260163076 |
| 15 | 877104/9303103 | 685/724 | 21/3/0/0/0 | 21/3/0/0/0 | -66.14638229183583 | -64 | -61.602128748916584 |
| 16 | 877104/9303104 | 738/863 | 22/2/0/0/0 | 16/7/1/0/0 | -688.2406166132155 | -511 | -584.9103552780334 |

聚合控制侧为 N/E/F/D/U = 183/33/0/0/0，admitted/completed/resolved = 216/216/216；reference 为 180/34/2/0/0，successful completed = 214。wait 为 27851.56917427036 对 29230.017251347945，Δ = -1378.448077077585（-4.7158647%）；elapsed 为 81822 对 82700，Δ = -878（-1.0617%）；energy 为 67584.03093562453 对 68515.23095364538，Δ = -931.20001802085（-1.3591%）。

raw reward（排除 rejection penalty）为 control 4250.50923752765、reference 4132.63373949572；rejection penalty 分别为 -156 和 -165.8。两类数值分开报告，不能把结果归结为 penalty 变化。

按预注册规则，该结果是 development candidate PASS only。它不是 PPO 准入、默认配置晋升、formal qualification，也不能单独证明训练退化的根因或已经解决 P3。

## 2026-10-01 Balanced proposal-support full 终态 (`FULL_OK`; development candidate)

balanced proposal-support control full 已完成。服务器 PID `2105971` 已退出，full mtime 为 `2026-10-01 02:37:12`（Asia/Shanghai）；artifact proof 为 `.tmp/p3_proposal_support_balanced_full_proof.json`，同步 full JSON 为 `D:/7_codespace/training_outputs/terminal/p3_followup_20260930/proposal_support_balanced_20261001/full.json`，SHA256=`f5cd9041f14c86f3b2e6ebde504a4e98f0eafcc4ee4c404db0939db38b2bb48d`。独立核对 proof 为 `.tmp/p3_proposal_support_balanced_full_proof.json`（SHA256=`02ad0f032a369fc412121e887830da71618092150ffbd08481333a7c05ae8634`），per-pair CSV 为 `.tmp/p3_proposal_support_balanced_full_perpair.csv`（SHA256=`dfdfe17476b01eda3ae3c8c96000fcb60572fabf35fc038878937e01ca3fae8f`）；ALL24 与六 profile 汇总均在 `1e-8` 内，24 control/reference 匹配、reference hashes、reuse9/new15 IDs 和 all checks 均通过。固定 24 个 pair 全部自然终态，0 censor/error；source/model RNG、request identity、ledger、energy、hard-safety checks 全通过，未出现 new failed/unresolved/diverted arrival IDs。范围为复用 9 个既有 control/reference 场景并新增 15 个 control 场景，不能视作 576 个独立样本。

| pair | scene | ticks C/R | C N/E/F/D/U | R N/E/F/D/U | Δwait | Δelapsed | Δenergy |
|---:|---|---:|---|---|---:|---:|---:|
| 1 | 877101/9303101 | 722/694 | 22/2/0/0/0 | 23/1/0/0/0 | 252.53942549349722 | 282 | 324.9373252078476 |
| 2 | 877101/9303102 | 713/713 | 23/1/0/0/0 | 23/1/0/0/0 | 0 | 0 | 9.094947017729282E-13 |
| 3 | 877101/9303103 | 768/655 | 20/4/0/0/0 | 23/1/0/0/0 | 685.5275409802998 | 642 | 646.8438997043531 |
| 4 | 877101/9303104 | 698/725 | 21/3/0/0/0 | 23/1/0/0/0 | 260.5767619747626 | 302 | 339.0913124511708 |
| 5 | 877102/9303101 | 799/795 | 20/4/0/0/0 | 20/4/0/0/0 | 261.4727465143619 | 363 | 322.5319456248826 |
| 6 | 877102/9303102 | 801/791 | 18/6/0/0/0 | 19/5/0/0/0 | -202.24211122579618 | -251 | -247.63182639344632 |
| 7 | 877102/9303103 | 735/786 | 20/4/0/0/0 | 20/4/0/0/0 | -169.61955018254366 | -121 | -173.83825824859105 |
| 8 | 877102/9303104 | 809/824 | 22/2/0/0/0 | 19/5/0/0/0 | -465.464473863537 | -397 | -443.5336065149195 |
| 9 | 877103/9303101 | 711/668 | 20/4/0/0/0 | 20/4/0/0/0 | 367.11554965350433 | 351 | 397.4807547823866 |
| 10 | 877103/9303102 | 831/713 | 18/6/0/0/0 | 21/3/0/0/0 | 241.61708810662867 | 113 | 197.95249503380546 |
| 11 | 877103/9303103 | 702/770 | 21/3/0/0/0 | 20/4/0/0/0 | -428.18800897632354 | -440 | -399.874432841023 |
| 12 | 877103/9303104 | 751/797 | 19/5/0/0/0 | 19/5/0/0/0 | -478.6270830262347 | -570 | -532.4805304688753 |
| 13 | 877104/9303101 | 758/863 | 19/5/0/0/0 | 20/3/1/0/0 | -578.3562047072569 | -374 | -359.9870170506765 |
| 14 | 877104/9303102 | 742/742 | 19/5/0/0/0 | 19/5/0/0/0 | -246.58688277428746 | -274 | -276.1631260163076 |
| 15 | 877104/9303103 | 685/724 | 21/3/0/0/0 | 21/3/0/0/0 | -66.14638229183583 | -64 | -61.602128748916584 |
| 16 | 877104/9303104 | 738/863 | 22/2/0/0/0 | 16/7/1/0/0 | -688.2406166132155 | -511 | -584.9103552780334 |
| 17 | 877105/9303101 | 685/828 | 21/3/0/0/0 | 21/3/0/0/0 | -504.171103904288 | -497 | -542.7246012361056 |
| 18 | 877105/9303102 | 706/706 | 22/2/0/0/0 | 22/2/0/0/0 | 0 | 0 | -9.094947017729282E-13 |
| 19 | 877105/9303103 | 742/865 | 22/2/0/0/0 | 19/5/0/0/0 | -944.2728093343212 | -802 | -886.4363900023491 |
| 20 | 877105/9303104 | 818/776 | 22/2/0/0/0 | 22/2/0/0/0 | -148.2594080823701 | -223 | -245.13615725234376 |
| 21 | 877106/9303101 | 836/752 | 19/5/0/0/0 | 20/4/0/0/0 | -176.0746084480652 | -154 | -150.76281530593496 |
| 22 | 877106/9303102 | 748/748 | 20/4/0/0/0 | 20/4/0/0/0 | 0 | 0 | 9.094947017729282E-13 |
| 23 | 877106/9303103 | 759/755 | 19/5/0/0/0 | 20/4/0/0/0 | 476.3792891837884 | 485 | 469.0832526346667 |
| 24 | 877106/9303104 | 799/757 | 21/3/0/0/0 | 21/3/0/0/0 | 276.4580103747421 | 151 | 211.86047995051922 |

| profile | pair IDs | C normal | R normal | C wait (s) | R wait (s) | Δwait (s) |
|---:|---|---:|---:|---:|---:|---:|
| 877101 | 1,2,3,4 | 86 | 92 | 12260.308761386932 | 11061.665032938374 | 1198.6437284485582 |
| 877102 | 5,6,7,8 | 80 | 78 | 13760.424366494517 | 14336.277755252031 | -575.853388757514 |
| 877103 | 9,10,11,12 | 78 | 80 | 13243.684865547717 | 13541.767319790142 | -298.08245424242523 |
| 877104 | 13,14,15,16 | 81 | 76 | 12144.420913491927 | 13723.750999878524 | -1579.330086386597 |
| 877105 | 17,18,19,20 | 87 | 84 | 12382.405353223297 | 13979.108674544275 | -1596.703321320978 |
| 877106 | 21,22,23,24 | 79 | 81 | 13516.843074574452 | 12940.080383463985 | 576.7626911104671 |

聚合 control N/E/F/D/U=`491/85/0/0/0`，reference=`491/83/2/0/0`；generated/admitted/resolved 均为 `576/576/576`，completed 为 control/reference=`576/574`。wait 为 `77308.08733471885` 对 `79582.65016586734`，Δ=`-2274.5628311484907`；request elapsed 为 `221361` 对 `223350`，Δ=`-1989`；energy 为 `183268.3495482137` 对 `185263.6493281816`，Δ=`-1995.2997799678997`。raw reward（排除 rejection penalty）为 control `11371.746812675045`、reference `11251.448813678655`；rejection penalty 分别为 `-403.90000000000003` 和 `-426.05`。

root review criterion 为 development candidate PASS only：technical/source/model-RNG/request identity/ledger/energy/hard-safety checks 全通过，normal 不低于 reference、wait 不高于 reference、successful completed 不低于 reference，strict business gain 为 true。该结果不是 formal qualification、默认配置晋升、PPO 准入或训练 remedy；`PPO_ALLOWED=false`，未使用 heldout。六个 profile 中 `877101` 的 normal 为 `86/92` 且 wait 增加 `1198.6437284485582 s`，`877106` 的 normal 为 `79/81` 且 wait 增加 `576.7626911104671 s`，因此不记录为普遍收益。

balanced full 终态后 heartbeat `p3-balanced-proposal-support` 已删除，无 active background job。下一步由 root 选择 actual trained checkpoint validation；本轮不启动训练。

## 2026-10-01 P3 trained proposal-support pilot 终态（full running）

这是新的 frozen trained proposal-support development comparison，不是训练、qualification、default promotion 或 PPO。`preexisting_artifact` 包含原始 2-second/3-arm fixture 的 raw proof；该 fixture 的 evaluator classification 不再作为生产终态，新的 runner 只将 nonterminal exact-replay `IncompleteLifecycle` 保留为 lifecycle qualification metadata，并生成 safety-snapshot fallback summary。修正后的 3-arm contract 为 valid/censored，`generated=12`、`admitted=12`、`pending=12`，full metrics 保持 null；没有 physical rerun。

`newly_run` 代码 commit 为 `9b66049`，28 个定向测试和 compile 通过。server no-env preflight 为 `PREFLIGHT_OK`，proof SHA 前缀为 `0911e9...fe38`；范围是原始 4 cells（`875101..875104` / `875201..875204`），每 cell 3 arms，native stochastic epsilon=`0.10` only support。该 preflight 不作 perfect keyed-CRN claim；PPO、teacher、heldout 和 default promotion 均关闭。

pilot 服务器 PID `2112900` 已自然退出；server pilot SHA=`b9133fdbe1c94b7e2a2d5225c052c103339bb55ca2d59d6df3e885a871a55cda`，与 preflight SHA `0911e9...fe38` 不同。3 arms 均 natural，0 censor/error，safety、source/model、caller-RNG、24-request identity、replay、ledger 和 energy checks 全通过。pilot cell 1 的 observed metrics 如下：

| arm | N/E/F | wait (s) | energy | raw reward excl. rejection penalty | rejection penalty |
|---|---|---:|---:|---:|---:|
| before_legacy | 19/5/0 | 3327.4996213135246 | 7757.917108755452 | 451.32887937331253 | -19.55 |
| after_legacy | 18/6/0 | 3046.1997149195568 | 7457.839771374826 | 450.31157694902555 | -20.950000000000003 |
| after_support | 18/6/0 | 3749.379758593257 | 8176.11847746592 | 434.348464181481 | -24.700000000000003 |

该 pilot 首 cell 的 before normal=`19`、after legacy/support normal=`18`，与历史首 cell 计数一致；support arm 的首 cell wait 为 `3749.379758593257 s`。这些是该 pilot cell 的 observed metrics，不形成 global acceptance 或 root-cause 结论。root 已授权 fixed full：复用 pilot cell 1，加上 3 cells/9 arms。以下启动记录属于历史状态：PID `2113754`，cwd `/home/user/projects/terminal`，module phase 为 `full`，server env/executable checks 通过；首个新 arm `cell_02_before_legacy.json` 已生成，大小 `71978` bytes，server mtime ns=`1790799508572507832`。full 随后已完成，终态见下方；启动时的 heartbeat `p3-trained-proposal-support` 已在终态后删除。

## 2026-10-01 P3 trained proposal-support full 终态（development candidate rejected）

trained proposal-support full 已完成。服务器 PID `2113754` 已退出，full mtime ns=`1790799958139098485`；full artifact SHA=`3f65446f49474452d31c3e1498c8019f04f4250cad3e8594c75107a123525919`，independent proof `.tmp/p3_trained_support_full_proof.json` SHA=`6e3cd91c472cdaf522020f39d7f8180e2802dfd035332187c21b4204a79f15b8`。4 cells × 3 arms 全部 natural、0 censor/error、hard-safety clear；每个 arm generated/admitted/resolved=`96/96/96`，source/model/caller-RNG、24-request identity、replay、ledger 和 energy checks 全通过。

| cell | env | arm | N/E/F/D/U | simulation s | request elapsed s | wait s | energy | raw reward excl penalty | rejection penalty |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 875101 | before_legacy | 19/5/0/0/0 | 761 | 9050 | 3327.4996213135246 | 7757.917108755452 | 451.32887937331253 | -19.55 |
| 1 | 875101 | after_legacy | 18/6/0/0/0 | 689 | 8737 | 3046.1997149195568 | 7457.839771374826 | 450.31157694902555 | -20.950000000000003 |
| 1 | 875101 | after_support | 18/6/0/0/0 | 755 | 9504 | 3749.379758593257 | 8176.11847746592 | 434.348464181481 | -24.700000000000003 |
| 2 | 875102 | before_legacy | 20/4/0/0/0 | 726 | 9133 | 3234.6369068496506 | 7677.921091787695 | 465.58665490460396 | -16.75 |
| 2 | 875102 | after_legacy | 18/6/0/0/0 | 727 | 9232 | 3292.5856932730585 | 7750.015446572095 | 443.8305713428898 | -20.650000000000002 |
| 2 | 875102 | after_support | 20/4/0/0/0 | 726 | 9159 | 3189.316530514645 | 7679.415958554219 | 463.8215531471326 | -19.450000000000003 |
| 3 | 875103 | before_legacy | 22/2/0/0/0 | 707 | 8369 | 2667.140927166041 | 7098.867749409129 | 497.63654049060835 | -14.3 |
| 3 | 875103 | after_legacy | 21/3/0/0/0 | 776 | 9194 | 3521.746147520452 | 7950.49929366206 | 471.0605408998054 | -17.55 |
| 3 | 875103 | after_support | 20/4/0/0/0 | 833 | 9186 | 3573.5243270983137 | 8005.473731383905 | 462.88896723234376 | -15.350000000000001 |
| 4 | 875104 | before_legacy | 19/5/0/0/0 | 720 | 8712 | 2731.326453726862 | 7034.847039351723 | 460.9677443897178 | -14.4 |
| 4 | 875104 | after_legacy | 17/7/0/0/0 | 726 | 8795 | 2833.9475529832466 | 7139.991726676687 | 442.2643044437435 | -15.200000000000001 |
| 4 | 875104 | after_support | 17/7/0/0/0 | 757 | 9443 | 3345.4609609007143 | 7798.724737635735 | 438.3307463573676 | -14.100000000000001 |

聚合 before_legacy/after_legacy/after_support 分别为 normal=`80/74/75`、emergency=`16/22/21`、failed/diverted/unresolved=`0/0/0`。wait 分别为 `11960.603909056077`、`12694.479108696314`、`13857.68157710693` s；request elapsed 分别为 `35264`、`35958`、`37292` s；energy 分别为 `29569.552989304`、`30298.346238285667`、`31659.73290503978`。raw reward（排除 rejection penalty）分别为 `1875.5198191582426`、`1807.4669936354644`、`1799.3897309183249`；rejection penalty 分别为 `-65.0`、`-74.35000000000001`、`-73.60000000000001`。

before→after_legacy 的 observed delta 为 normal `-6`、wait `+733.875199640237 s`、request elapsed `+694 s`、energy `+728.7932489816667`。after_support→after_legacy 的 observed delta 为 normal `+1`，但 wait `+1163.2024684106163 s`（`+9.16305788091587%`）、request elapsed `+1334 s`、energy `+1361.3866667541115`（`+4.493270543703249%`）。按既有 aggregate criterion，trained fixed-epsilon support 被拒绝为 development candidate；不进行 epsilon retune 或 default promotion。

证据边界：该 full reproduces the four-cell trained regression；proposal encoder/head、gate head、temperature bitwise frozen，允许变化范围仅为 gate adapter/critic，因此不把本结果写成 newly trained proposal-head collapse。coverage 结果不足以单独建立 fix；本轮不作 team-credit、entropy、clipping root-cause 或 P3 solved 结论。原 balanced24 的 `491/85/0` 对 `491/83/2` 结果仍是 separate preexisting cohort/source/sampler，不能与本 full 合并。

full heartbeat `p3-trained-proposal-support` 已删除，无 active job。下一步仅登记 matched-state/offline training-mechanism diagnostic preparation；未批准新训练，`PPO_ALLOWED=false`，teacher/optimizer/heldout 均未使用。

## 可执行动作机会诊断预注册与终态

这是冻结策略下的被动机制诊断，不是训练、checkpoint 选择、性能 pilot 或 qualification。固定参考为 9 个 legacy arms：pair `1/5/9/13/17/21`（`env=877101..877106`、keyed policy `9303101`），以及 `pair 14/15/16`（`env=877104`、keyed policy `9303102/9303103/9303104`）。源为 `3fe7+a82`，保持 88D、K=3、legacy mask 和同一物理配置；`PPO_ALLOWED=false`，held-out、optimizer 和 teacher 均未使用。

每个参考场景在 tick `100–149`、`300–349`、`500–549` 中按顺序取第一个满足预注册条件的合法状态，最多 `27` 个 state；每个 state 固定执行 `WAIT + 25` proposal 的 `26` 个 one-step 分支，共 `25` 个已选 state、`650` 个分支。分支使用真实策略 proposal 概率计算 `C` 与 `pC`，只记录 native decision 和即时 reward difference；不计算 `D` 或 global value，不取 best proposal、不删除失败分支、不重抽样。完整条件、source replay 对齐和停止规则见 [P3 executable-opportunity coverage protocol](p3_opportunity_coverage_protocol_20260930.md)。9 个参考 arm 对应固定场景，不把其 `216` 个请求事件当作 `216` 个独立环境 profile。

证据标记如下：

- `preexisting_artifact`：既有 frozen-WAIT `full.json` 的 `FULL_OK/pair_count=24` 结果、reference arms 和该协议。
- `newly_run`：local preflight 为 `PREFLIGHT_OK`，机会 runner 的 focused suite 为 `11 passed`，combined recheck 为 `37 passed in 8.15s`；服务器只读基础快照 exit `0`，115-file runtime manifest 的 SHA mismatch count 为 `0`，source `3fe7` 与 warmup `a82` checkpoint SHA 与预注册值一致。V2 preflight、V1 pilot 只读导入 guard 和 full 均已完成，`full.json` 为 `FULL_OK`。

V1 server pilot 已完成：纠正项目 cwd 后，`PID 2085294` 的 `pilot.json` 为 `PILOT_OK/PAIR_OK`，env `877101`、keyed policy `9303101`，`694` ticks、`24` requests、`23 normal/1 emergency/0 failed`，自然终止且无 censor/error；source/model/caller-RNG、请求 identity、ledger、energy、wait 和 safety checks 均通过。3 个窗口共 `78` 个分支，均成功、0 error；`C=0`、`pC=0`，没有 `D`。这个新 legacy replay 与原 legacy reference 的 `reward_total` 差和 `release_rejection_penalty` 差均为 `0`，排除 penalty 的 raw reward 差也为 `0`。该 pilot 是冻结机会覆盖事实，不形成性能或因果结论；历史 frozen-WAIT 对照中的 `+2.5` 仍只在下文自己的对照段中记录。

首个 launcher 在 phase 初始化前记录了 `NameError: name 'subproces' is not defined`，原始片段保存在 `.tmp/p3_opportunity_launcher_v1.stderr.txt`；修正后的 cwd/subprocess 启动 guard 完成了上述 V1 pilot。V1/V2 同时保留两种 model digest 约定：paired execution digest `20afdf4b60babc63a0a64eba6702d6f9db3891b537168a5e5346639f148b1d23`，canonical source-binding digest `a8a133863bc03627452814535aa938ed062fe0496b35678d0fe9333524503f07`；二者来自同一加载模型，只是摘要算法不同，不表示权重变化。V2 full artifact 为 `/home/user/training_outputs/terminal/p3_followup_20260930/opportunity_coverage_v2/full.json`，本地紧凑核对为 `.tmp/p3_opportunity_coverage_v2_evidence.json`。

V2 的 9/9 pair 均为 `PAIR_OK`、reference exact；25/27 registered states 有结果，缺失的是 pair `14` 与 `15` 的 window `03`（`500–549`），两行均保留为 unknown，不写成 `C=0`。650/650 selected-state branches 状态为 `ok`，branch error 与 true truncation 均为 `0`。9 个 arm 的 source/model/caller-RNG、配置绑定、request identity、ledger、energy、wait 和 safety checks 均通过：source/model digest unchanged、24 个 request identity 唯一、energy coverage 与 ledger conservation valid、wait/elapsed finite、hard-safety violation false；共同 config snapshot SHA 为 `e6e8535e65a7452cb4644c43d13d6bbf3b15cdd621d7294dfdfdeaf4b79a3b97`，preflight reference SHA 为 `fe7954aac75f9e947298ad7de8c8bdec74bc84c3bb18bb59ab7c40860d9e6267`。V2 new legacy arm 的逐 arm N/E/F/D/U、generated/admitted/completed/resolved、wait、request elapsed 和 energy 如下；`elapsed` 为 24 个请求 elapsed 的和，energy 为 24 个唯一 arrival 的累计消耗。

| pair | env/keyed | ticks | N/E/F/D/U | generated/admitted/completed/resolved | wait s | elapsed s | energy |
|---:|---|---:|---|---|---:|---:|---:|
| 1 | 877101/9303101 | 694 | 23/1/0/0/0 | 24/24/24/24 | 2727.950535124339 | 8377 | 7069.867720034218 |
| 5 | 877102/9303101 | 795 | 20/4/0/0/0 | 24/24/24/24 | 3181.211230903982 | 9296 | 7545.658308610789 |
| 9 | 877103/9303101 | 668 | 20/4/0/0/0 | 24/24/24/24 | 2849.7762184481476 | 8903 | 7274.853470463249 |
| 13 | 877104/9303101 | 863 | 20/3/1/0/0 | 24/24/23/24 | 4195.360899337695 | 9957 | 8410.643286521372 |
| 17 | 877105/9303101 | 828 | 21/3/0/0/0 | 24/24/24/24 | 3298.552351300552 | 9332 | 7830.513734136947 |
| 21 | 877106/9303101 | 752 | 20/4/0/0/0 | 24/24/24/24 | 3448.775915692401 | 9621 | 7845.565803633816 |
| 14 | 877104/9303102 | 742 | 19/5/0/0/0 | 24/24/24/24 | 3028.2335629926934 | 8991 | 7427.784515825126 |
| 15 | 877104/9303103 | 724 | 21/3/0/0/0 | 24/24/24/24 | 2464.0124355070643 | 8412 | 6793.041730387722 |
| 16 | 877104/9303104 | 863 | 16/7/1/0/0 | 24/24/23/24 | 4036.144102041071 | 9811 | 8317.302384032148 |

上述 9 个 V2 new legacy arm 的累计 wait 为 `29230.017251347945 s`，request elapsed 为 `82700 s`，energy 为 `68515.23095364538`；这是机会覆盖的 legacy arm 汇总，不是 24-pair frozen-WAIT 对照的 48-run 汇总。

逐 state 结果（window `01/02/03` 分别为 `100–149/300–349/500–549`）如下。`p` 是 `p_release`；approved indices 是即时 native decision 通过的 proposal index。pair `15`, window `01` 的原始 `C=1.0000000000000053` 保留，显示为约 `1` 只是 rounding，未 clip。

| pair | window | tick | C | pC | p | qmax | approved indices |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 01 | 100 | 0.0 | 0.0 | 0.0034812598023563623 | 0.9501171708106995 | `[]` |
| 1 | 02 | 300 | 0.0 | 0.0 | 0.0008625637274235487 | 0.7396410703659058 | `[]` |
| 1 | 03 | 500 | 0.0 | 0.0 | 0.27430734038352966 | 1.0 | `[]` |
| 5 | 01 | 100 | 0.0 | 0.0 | 0.0010812501423060894 | 0.999981164932251 | `[]` |
| 5 | 02 | 300 | 0.0 | 0.0 | 0.05724853277206421 | 0.9999755620956421 | `[]` |
| 5 | 03 | 500 | 0.0 | 0.0 | 0.042119335383176804 | 1.0 | `[]` |
| 9 | 01 | 100 | 0.0 | 0.0 | 0.025524627417325974 | 0.9999926090240479 | `[]` |
| 9 | 02 | 300 | 2.0926044087687933e-18 | 2.078339034892979e-18 | 0.9931829571723938 | 1.0 | `[4]` |
| 9 | 03 | 500 | 0.0 | 0.0 | 0.9081240296363831 | 0.999997615814209 | `[]` |
| 13 | 01 | 116 | 0.0 | 0.0 | 0.0011279478203505278 | 0.837800145149231 | `[]` |
| 13 | 02 | 300 | 0.0 | 0.0 | 0.00023066700669005513 | 0.9706255197525024 | `[]` |
| 13 | 03 | 500 | 0.0 | 0.0 | 8.028752063182765e-07 | 0.9999547004699707 | `[]` |
| 17 | 01 | 100 | 0.0 | 0.0 | 0.9985920786857605 | 0.9945176243782043 | `[]` |
| 17 | 02 | 300 | 0.0 | 0.0 | 0.0014054110506549478 | 0.9165021181106567 | `[]` |
| 17 | 03 | 500 | 0.0 | 0.0 | 0.8001037240028381 | 0.9831014275550842 | `[]` |
| 21 | 01 | 100 | 0.0 | 0.0 | 0.0020757101010531187 | 0.9999845027923584 | `[]` |
| 21 | 02 | 300 | 0.0 | 0.0 | 0.862338662147522 | 0.5847722291946411 | `[]` |
| 21 | 03 | 500 | 0.0 | 0.0 | 0.969641923904419 | 0.9991999268531799 | `[]` |
| 14 | 01 | 116 | 0.0 | 0.0 | 0.00012697969214059412 | 0.8391591310501099 | `[]` |
| 14 | 02 | 300 | 0.0 | 0.0 | 0.009739718399941921 | 0.945147693157196 | `[]` |
| 14 | 03 | unknown | unknown | unknown | unknown | unknown | unknown（未选出） |
| 15 | 01 | 134 | 1.0000000000000053 | 0.0003375054511707294 | 0.0003375054511707276 | 1.0 | `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]` |
| 15 | 02 | 346 | 0.0 | 0.0 | 0.40018537640571594 | 0.5672844052314758 | `[]` |
| 15 | 03 | unknown | unknown | unknown | unknown | unknown | unknown（未选出） |
| 16 | 01 | 116 | 0.0 | 0.0 | 0.00024265077081508934 | 0.9968762397766113 | `[]` |
| 16 | 02 | 300 | 0.0 | 0.0 | 2.664043449840392e-06 | 0.9775197505950928 | `[]` |
| 16 | 03 | 500 | 0.0 | 0.0 | 9.466992923989892e-05 | 0.9971941709518433 | `[]` |

其中 pair `9`/window `02`/tick `300` 是 q-starvation witness：`C=2.0926044087687933e-18`、`pC=2.078339034892979e-18`、`p=0.9931829571723938`、`qmax=1`，只有 proposal `[4]` 被记录为 approved。pair `15`/window `01`/tick `134` 是即时可执行但低 RELEASE 概率、当步未请求的 witness：`C≈1`（原值见表）、`p=0.0003375054511707276`、`pC=0.0003375054511707294`、`qmax=1`，25 个 proposal 均进入 approved index。这不证明等待有害；当前 WAIT 可能有未来收益，是否有害要看完整周期 gate control。多数 selected states 的 proposal 被 native safety/capacity/priority/energy guards 拒绝；这不是 PPO 归因。上述 q-starvation、低 `p` gate 和旧 source16 failure 是分开的机制观察，旧 source16 failure 未并入这 9 个 arm 的表。`C=0` 只表示该 one-step coverage 观测为零，不能推出 full-cycle 无作用；本轮没有产生 full-cycle `D` 或 `m`，也没有把结果写成 strong-oracle 不可行性。

env `877104` 的 arrival `13` 四条实际 trace 摘要如下；trace 来自每个 legacy arm 的 `target13_trace`，不是独立 profile：

| pair/keyed | ticks | active phase counts | actual decision counts | first/last normal release attempt | terminal row |
|---|---:|---|---|---|---|
| 13/9303101 | 863 | inbound 57; ring_outer 747 | None 59; not_requested 734; safety_reject 1; safety_wait 28; emergency_energy_guard 41 | tick 161 / 161, p 0.024594346061348915, qmax 1, safety_reject | UID 46, failed_energy, t=862, wait 776.8147075862394, energy 833.1666666666666, final SOC 0 |
| 14/9303102 | 742 | inbound 57; ring_outer 217; ring_middle 1; transfer_o2m 55; emergency_landing 87 | None 325; not_requested 407; safety_reject 8; approved 1; plan_active 1 | tick 164 reject p 0.17946651577949524 → tick 332 approved p 0.894378662109375, qmax 0.7798160910606384 | UID 46, emergency_landed, t=506, wait 246.0261539121092, energy 458.2358821167139 |
| 15/9303103 | 724 | inbound 57; ring_outer 88; transfer_o2m 62; ring_middle 1; transfer_m2i 61; ring_inner 16; emergency_landing 54 | None 385; not_requested 333; safety_reject 4; approved 2 | tick 142 reject p 0.9835757613182068 → tick 358 reject p 0.9883878827095032, qmax 0.9975574016571045 | UID 72, emergency_landed, t=446, wait 151.21199255359232, energy 374.5005100650064 |
| 16/9303104 | 863 | inbound 57; ring_outer 747 | None 59; not_requested 729; safety_reject 1; safety_wait 30; emergency_energy_guard 44 | tick 267 / 267, p 0.7881418466567993, qmax 0.973076581954956, safety_reject | UID 46, failed_energy, t=862, wait 776.8147075862394, energy 833.1666666666666, final SOC 0 |

依据上述 witness，已完成同一 `9` 个 reference pairs 的单因素 frozen always-RELEASE gate control；结果已在本文顶部的终态段报告。随后已完成 pair `15` 的一次 passive exact replay，结果为 `REPLAY_OK`，但不构成 P3 根因结论或 formal qualification。该 replay 未训练、未 promote，`PPO_ALLOWED=false`、held-out 未使用。

本地唯一集成命令为：

```text
D:\7_codespace\venv\Scripts\python.exe -m pytest -p no:cacheprovider -q \
  tests/test_paired_gate_diagnostic.py tests/test_service_context_migration_v4_2.py \
  tests/test_high_cost_conservation.py tests/test_research_high_level_repairs.py \
  tests/test_a6_effectiveness.py tests/test_a6_dynamic_high.py \
  tests/test_forced_wait_protocol.py tests/test_research_low_level_repairs.py \
  tests/test_student_rollout_v4_2.py tests/test_p3_followup_diagnostics.py
```

结果为 `82 passed in 45.59s`、exit `0`。`--collect-only -q` 同样 exit `0`，收集 `82` 个唯一 nodeids。原始日志位于 `.tmp/p3_followup_review/integration.log` 和 `.tmp/p3_followup_review/collect_only.log`。

| 模块 | nodeids |
|---|---:|
| `test_paired_gate_diagnostic.py` | 16 |
| `test_service_context_migration_v4_2.py` | 11 |
| `test_high_cost_conservation.py` | 5 |
| `test_research_high_level_repairs.py` | 7 |
| `test_a6_effectiveness.py` | 2 |
| `test_a6_dynamic_high.py` | 10 |
| `test_forced_wait_protocol.py` | 6 |
| `test_research_low_level_repairs.py` | 8 |
| `test_student_rollout_v4_2.py` | 9 |
| `test_p3_followup_diagnostics.py` | 10 |

## 已确认的实现修复

- [`paired_gate_diagnostic.py`](../../../terminal_marl/training/paired_gate_diagnostic.py) 的 `restore_diagnostic_snapshot` 在 `TerminalParallelEnv` 深拷贝后重新绑定该 copy 的 `_resolve` 与 `arrivals.requests_until`。真实隔离测试 [`test_paired_gate_diagnostic.py`](../../../tests/test_paired_gate_diagnostic.py) 检查 branch hook 的 `__self__`、未来请求读取和 source ledger/RNG 不变。
- [`student_rollout_v4_2.py`](../../../terminal_marl/training/student_rollout_v4_2.py) 在 opt-in `pre_action_forced_wait` 下将 forced rows 的 gate/joint logprob 记录为 `0`；proposal 继续为 `-1/NaN`，metadata 继续为 `None`。legacy 模式和 actor 可行动 rows 保持原请求/RNG/动作结果。相关 6 个测试包含 25 个真实环境 proposal、affected-interface、正 SOC 低电量和 PPO masked-row 检查。
- high-level cost metadata、600 秒 sustained-low shrink regression、queue history/unresolved overlap 和 V4.2 adapter fail-fast 均在集成套件中通过；这些是代码回归事实，不是 A6 qualification。

## 服务上下文迁移

[`service_context_migration_v4_2.py`](../../../terminal_marl/algo/service_context_migration_v4_2.py) 与 [`test_service_context_migration_v4_2.py`](../../../tests/test_service_context_migration_v4_2.py) 的 11 个测试通过。迁移绑定的 source 是 `phaseA_low_obs_v2_1_retry_state` 88D，target 是 V2.4 90D；新增输入权重列初始化为 0，实际 V2.4 观测列按 UAV/LZ 状态计算。V2.3 只有在调用方显式传入该 schema 时被拒绝，不能据 88D 维度推断所有 schema 语义等价。迁移保留已有状态、mode、gradient 和 CPU RNG；fresh optimizer required，qualification status 仍为 experimental only。

## Paired gate helper 与新 runner

[`paired_gate_diagnostic.py`](../../../terminal_marl/training/paired_gate_diagnostic.py) 当前集成测试为 16 passed。每个 repeat 的 25 个 proposal 分支按实际请求和权重记录；branch error/censor 时不产生 qualified `delta_Q`，raw branch report 和 progress 仍保留。continuation 使用同一个 frozen policy；diagnostic 不调用 optimizer，也不改变 model、gradient、state 或 RNG。

[`run_p3_followup_diagnostics.py`](../../../terminal_marl/training/run_p3_followup_diagnostics.py) 是新的独立诊断入口，配合 credit observer；其 runner tests 为 10 passed。代码登记的输入契约是 6 episodes、每 episode 24 requests、总计 4,497 steps；selection test 固定 24 个 occurrence、无放回且顺序可复现。sharding 后的 focused runner suite 为 `10 passed`，不计入上述 `82` 个已记录集成 cases。本轮只验证代码和契约，没有把 runner phase 输出写成性能结果。

追加事实：此前 Python `-m` 运行时的 `__main__` entry file 被 name filter 遗漏；当前仅在 package root 内纳入该 entry file。该修复的 focused suite 为 `11 passed in 2.35s`，重复测试不计入新的总数。当前运行中的 preflight 未重跑或覆盖；修复适用于后续 preflight，既有 hash 和 startup observation 保持原边界。

## 既有 episode-gradient replay artifact

实际服务器运行使用的旧模块 [`diagnose_p3_task_neutral_episode_gradients.py`](../../../terminal_marl/training/diagnose_p3_task_neutral_episode_gradients.py) 是本次 replay 的 artifact source；它不是本提交新增的自包含入口。`runtime_sources.json` 记录其 SHA256：`c5e35752f78e33698510c59aa36ed9dd714606c07a80d785aa76c49d42aa19bf`。

- preflight 为 `CHECK_ONLY_OK`：6 episodes、4,497 steps、144 requests，`runtime_optimizer_calls=0`、`runtime_replay_calls=0`、`runtime_env_reset=false`。
- legacy replay 为 `COMPLETED`：6 episodes、4,497 steps、144 requests、6 replay calls；`replay_exact`、`mc_suffix_match_all`、`aggregate_gradient_matches_sum`、`model_unchanged_after_replay_and_gradient` 均为 true，`training_called=false`、`optimizer_used=false`、`PPO_ALLOWED=false`。
- artifact paths：`/home/user/training_outputs/terminal/p3_followup_20260930/runtime_sources.json`、`legacy_preflight/report.json`、`legacy_replay/report.json`、`legacy_replay/summary.json`。

## pilot artifact

给定 pilot artifact `D:/7_codespace/training_outputs/terminal/research_review_repairs_20260930/p3_followup/pilot.json`（服务器路径：`/home/user/training_outputs/terminal/research_review_repairs_20260930/p3_followup/pilot.json`）状态为 `COMPLETED`。它只包含 episode 1、step 278、UID 5、continuation seed `9303101`：26 个 branch 均 `natural_terminal`，branch steps 范围 `410..427`，总计 `10677` ticks，无 censor 和 safety event；prefix 410 valid，model/source snapshot、RNG 和 `q_unchanged` 均为 true。

该状态的 paired 值为 `C=0`、`p=0.7176065445`、`p*C=0`、`D=-1.4162728544`。对实际 batch-01 after model 的 gate 对齐为 `0.7176070213→0.7267845273`，`delta=0.0091775060`，linear approximation `0.0093069952`，residual `-0.0001294892`，`local_m=-0.0129978526`。

该条件状态的一次续跑显示更新方向不利；一个状态一个seed不支持认定credit、entropy或clip为退化根因，也不构成性能修复证据。

历史计算记录：`C` 固定为 `24×4`，由 4 个进程按 modulo 分片；这只是计算调度，不增加实验 factor、不重抽样、不删除失败 branch。此前 4 个分片进程 PID 为 `2007505`、`2007601`、`2007551`、`2007650`；timestamp snapshot 曾显示 `14/24`，该快照不是终态，最终结果见下节。`PPO_ALLOWED=false`，未消费 held-out。此前 `82` 个集成测试只作为已记录 raw count，不在本节重复累计。

## C 初步结果

raw artifact `causal/shard_0/diagnose/state_0001.json` 当前包含同一首状态的 `104` 个分支，来自 `4` 个 continuation seeds，均为 `natural_terminal`；model、environment 和 caller RNG 均保持不变。该状态的 `C=0`、`p=0.7176065`、平均 `D=6.56333546`、平均 `m=0.06023505`；四个 seed 的 `D` 为 `[-1.41627285, -0.05, 11.62504392, 16.09457077]`。

同一首状态四种续跑的符号混合，单seed pilot负值不能用于根因判定；C=0仅表示即时覆盖为零，不能据此认定全周期动作无效或直接强制WAIT。该段是早期 pilot/timestamp snapshot；完整 24 状态的最终 named-group 结果见下节。失败机制三状态、312 个 branch 已完成，不构成性能改善证据。

## C 全部 24 状态最终结果

本节读取四个 causal shard 的 named group JSON（`summary.json`、`report.json`、`diagnose.json` 及其列出的 state report JSON），不把 raw branch 文件作为汇总输入。四个 shard 的状态均为 `COMPLETED`，各含 6 个 state、624 个 branch；全局共 24 个 state、2,496 个 branch（每个 state 为 4 个 continuation seed、每个 seed 26 个 WAIT/RELEASE 分支）。四个 shard 的 `diagnose.json` 修改时间分别为：shard 0 `19:12:51`、shard 1 `18:23:26`、shard 2 `19:19:08`、shard 3 `18:14:25`。

24/24 个 state 的 D 与 m 均有值；2,496/2,496 个 branch 为 `natural_terminal`，error 与 censor 均为 0。所有 state 的 model、environment 和 caller RNG 不变标记均为 `true`，四个 shard 的 `PPO_ALLOWED=false`、held-out=false、optimizer calls=0、teacher=false。source checkpoint `3fe7dd1e01d032227521f6fc5c8464fe8d00d98fb0e9f0dbd13411123a27d4ab` 与 warmup `a82ed7a287035fb1fc868fc955e056c0481de75bcb99880d5c79f35588dde710` 固定；after batch-01 checkpoint 的 SHA256 为 `a7bcf3d1e6f4bda82d0d051ec25966a18f267309f1d6030689b52d90e63aa366`。D 与四个 seed 差分均值的最大绝对误差为 `7.1054273576010019e-14`。

表中 `C` 是 `C_i`，`p` 是 gate 的 `p_release`，`D` 是 qualified full-return difference，`m` 是 alignment 的 per-row `local_m`。`Dseed` 顺序为 continuation seeds `9303101`、`9303102`、`9303103`、`9303104`；表中标注“浮点近零”的 D 保留原始科学计数值，不改写成 0。

| selection | step / UID | C | p | D | m | Dseed |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 278 / 5 | 0.0000000000000000e+00 | 7.1760654449462891e-01 | 6.5633354576678471e+00 | 6.0235050345867361e-02 | `-1.4162728544092147e+00, -4.9999999999954525e-02, 1.1625043919265863e+01, 1.6094570765814922e+01` |
| 2 | 319 / 97 | 0.0000000000000000e+00 | 6.8654189817607403e-06 | -5.0000000000011369e-02 | 1.8383116184854403e-08 | `-5.0000000000011369e-02, -5.0000000000125056e-02, -5.0000000000011369e-02, -4.9999999999954525e-02` |
| 3 | 354 / 237 | 0.0000000000000000e+00 | 1.9145017722621560e-03 | -2.4999999999948841e-02 | -4.5512570068147030e-08 | `2.8421709430404007e-14 (浮点近零), 5.6843418860808015e-14 (浮点近零), -4.9999999999926104e-02, -5.0000000000011369e-02` |
| 4 | 361 / 298 | 0.0000000000000000e+00 | 5.5721051289543363e-16 | -5.0000000000011369e-02 | 1.5885547060386379e-18 | `-4.9999999999954525e-02, -4.9999999999954525e-02, -5.0000000000011369e-02, -5.0000000000011369e-02` |
| 5 | 96 / 2 | 0.0000000000000000e+00 | 2.2396884858608246e-02 | -9.7227664391866142e+00 | -1.5937326957307082e-02 | `-5.0000000000068212e-02, -4.6285767845538089e+01, 4.7968038411779617e+00, 2.6478982476137958e+00` |
| 6 | 131 / 2 | 0.0000000000000000e+00 | 3.4216021682631492e-10 | -4.9999999999897682e-02 | 2.6690594179202613e-13 | `-4.9999999999954525e-02, -4.9999999999897682e-02, -5.0000000000011369e-02, -4.9999999999840838e-02` |
| 7 | 155 / 18 | 0.0000000000000000e+00 | 6.3217245042324066e-02 | -3.7499999999965894e-02 | -3.2444112002820074e-05 | `-5.6843418860808015e-14 (浮点近零), -5.0000000000011369e-02, -5.0000000000068212e-02, -5.0000000000011369e-02` |
| 8 | 187 / 0 | 0.0000000000000000e+00 | 7.6557704620228151e-11 | -5.0000000000068212e-02 | 2.2415402867212491e-13 | `-5.0000000000011369e-02, -5.0000000000011369e-02, -5.0000000000068212e-02, -5.0000000000011369e-02` |
| 9 | 312 / 5 | 0.0000000000000000e+00 | 1.6058335217117303e-10 | -5.0000000000011369e-02 | 1.1853712456047414e-13 | `-4.9999999999954525e-02, -5.0000000000011369e-02, -4.9999999999954525e-02, -5.0000000000011369e-02` |
| 10 | 335 / 53 | 0.0000000000000000e+00 | 9.3492984888143837e-05 | -5.0000000000068212e-02 | 6.0744823713404112e-07 | `-5.0000000000068212e-02, -5.0000000000068212e-02, -5.0000000000068212e-02, -5.0000000000068212e-02` |
| 11 | 373 / 0 | 0.0000000000000000e+00 | 5.8644739910960197e-03 | -5.0000000000039790e-02 | 4.6765222214199566e-05 | `-4.9999999999982947e-02, -5.0000000000039790e-02, -5.0000000000068212e-02, -5.0000000000039790e-02` |
| 12 | 417 / 74 | 0.0000000000000000e+00 | 1.6107469524850870e-11 | -5.0000000000011369e-02 | 8.5018016932078455e-14 | `-5.0000000000011369e-02, -5.0000000000039790e-02, -4.9999999999982947e-02, -5.0000000000068212e-02` |
| 13 | 440 / 74 | 0.0000000000000000e+00 | 8.1910187699991218e-16 | -4.9999999999954525e-02 | -1.1249656330710945e-20 | `-4.9999999999982947e-02, -4.9999999999926104e-02, -4.9999999999982947e-02, -4.9999999999982947e-02` |
| 14 | 497 / 32 | 0.0000000000000000e+00 | 3.7413606150948908e-06 | 4.1246121364927717e+00 | -1.5179809506855633e-06 | `4.9905124224302710e+00, 5.3624389315122869e+00, 4.1015253646625638e-01, 5.7353446555621872e+00` |
| 15 | 2 / 11 | 0.0000000000000000e+00 | 1.5535855008859903e-18 | -4.9999999999954525e-02 | 4.4498181052249653e-22 | `-5.0000000000068212e-02, -4.9999999999897682e-02, -4.9999999999897682e-02, -5.0000000000011369e-02` |
| 16 | 297 / 211 | 0.0000000000000000e+00 | 1.9081822710473317e-14 | -4.9999999999954525e-02 | 8.8530528394223347e-17 | `-4.9999999999954525e-02, -4.9999999999954525e-02, -4.9999999999954525e-02, -5.0000000000011369e-02` |
| 17 | 41 / 5 | 0.0000000000000000e+00 | 3.7640047328979520e-11 | -5.0000000000011369e-02 | 1.0240315539980058e-13 | `-5.0000000000068212e-02, -5.0000000000011369e-02, -4.9999999999954525e-02, -5.0000000000068212e-02` |
| 18 | 328 / 0 | 0.0000000000000000e+00 | 1.4511273801326752e-01 | -5.0000000000068212e-02 | -1.0516569018378300e-04 | `-5.0000000000125056e-02, -5.0000000000011369e-02, -5.0000000000125056e-02, -5.0000000000125056e-02` |
| 19 | 42 / 8 | 0.0000000000000000e+00 | 6.0440950619522482e-05 | -5.0000000000011369e-02 | -2.7586356736725883e-07 | `-5.0000000000011369e-02, -5.0000000000011369e-02, -4.9999999999897682e-02, -5.0000000000068212e-02` |
| 20 | 238 / 69 | 0.0000000000000000e+00 | 1.8143804726572961e-17 | -5.0000000000011369e-02 | 1.1942999120165839e-20 | `-5.0000000000011369e-02, -4.9999999999954525e-02, -5.0000000000011369e-02, -5.0000000000011369e-02` |
| 21 | 274 / 265 | 2.0938614857594729e-14 | 2.0441766083240509e-01 | 5.6843418860808015e-14 (浮点近零) | -1.2761017218507024e-15 | `3.9790393202565610e-13 (浮点近零), -5.6843418860808015e-14 (浮点近零), 0.0000000000000000e+00 (浮点近零), -1.1368683772161603e-13 (浮点近零)` |
| 22 | 328 / 307 | 0.0000000000000000e+00 | 7.5787720543729620e-10 | -5.0000000000011369e-02 | -2.0763946118056711e-13 | `-5.0000000000011369e-02, -5.0000000000039790e-02, -4.9999999999982947e-02, -5.0000000000068212e-02` |
| 23 | 418 / 264 | 0.0000000000000000e+00 | 2.8082777280360460e-03 | -5.0000000000011369e-02 | -3.2498734071858168e-06 | `-5.0000000000011369e-02, -5.0000000000011369e-02, -4.9999999999982947e-02, -4.9999999999982947e-02` |
| 24 | 616 / 69 | 0.0000000000000000e+00 | 3.8778381422162056e-03 | -4.9999999999982947e-02 | 7.0226495154178037e-06 | `-4.9999999999954525e-02, -4.9999999999982947e-02, -4.9999999999997158e-02, -4.9999999999982947e-02` |

`m` 是每个 state 的 local alignment 项，不将 24 个 m 相加解释为 global gain。named group JSON 中若报告 baseline branch outcome aggregate，则它是重复 counterfactual 场景的累加计数，不是请求总体样本量，也不作为总体比例。failure-mechanism 仍保持独立的 source16 / batch-02 episode-24 记录，不并入本表。

## P3 energy-failure mechanism completed

服务器 artifact `/home/user/training_outputs/terminal/p3_followup_20260930/failure_mechanism/diagnose.json` 已报告 `COMPLETED`：3 个状态共 `312` 个 branch，全部为 `natural_terminal` 且 uncensored；model、environment 和 caller RNG 均保持不变。

| state | step | SOC | C | p | D | m | full D by continuation seed |
|---|---:|---:|---:|---:|---:|---:|---|
| F01 | 280 | 0.398656 | 0 | 8.28291e-5 | -0.05 | +2.7650e-8 | `[-0.05, -0.05, -0.05, -0.05]` |
| F02 | 445 | 0.288656 | 1.1869090464568813e-14 | 0.000650799135 | +2.30523854878933 | -0.0002574839567 | `[0.593587408892546, -0.05, -0.05, 8.72736678626487]` |
| F03 | 726 | 0.110987 | 0 | 2.21449e-5 | -0.05 | +1.58285e-7 | `[-0.05, -0.05, -0.05, -0.05]` |

Full-return `D` is `paired.expected_release_undiscounted_return_by_repeat[i] - wait[i].undiscounted_return`; the mean check was within `1e-8`. Target arrival `14` was evaluated under WAIT and true q-weighted RELEASE. RELEASE outcomes by seed were F01: `9303201`, `9303202`, `9303204` `emergency_landed`, `9303203` `failed_energy`; F02: all four `emergency_landed`; F03: all four `failed_energy`. The F02 raw `C` is `1.1869090464568813e-14`; the earlier `C=0` display was rounded. Proposals `22`, `23`, and `24` were approved, each with K=3 runtime rank `1`; proposal `11` carried approximately `99.999999986%` of the true q mass. The new blocker first-step record agrees with the old paired first-step record; F02 is not described as 25 proposals all rejected.

第726步的一次gate干预不能在冻结后续策略下挽救该机；第445步续跑仍可应急落地，需要核对等待期资源/安全/能量拒绝原因，尚未选定训练修复。one-step check 不产生 `D`。该诊断未使用 PPO 或 held-out。

## Bounded blocker telemetry completed

历史启动记录：初次 PID `2025086` 因工作目录缺少项目包产生
`ModuleNotFoundError: No module named 'terminal_marl'`；该次真实 phase 未开始，
没有产生 phase outputs。`blockers_v1.log` 保留该旧错误；纠正 project cwd 后，
同一 blocker phase 产生了以下完成产物，旧错误不指向当前完成结果。

`firststep_blockers.json` 报告 `78/78` 首步分支成功、`0` 个错误，三个固定
状态各 `26` 个分支。`original_auto_emergency_trace.json` 报告 exact `984`
步、`24` requests 和 terminal rows，保留 failed UID `57`、time `892`，
source unchanged，target calls `878`、trace errors `0`。`manifest.json` 和
`progress.json` 均为 `COMPLETED`。

原始 trace 的 `878` 个 target calls 中有 `102` 次实际 attempt。pad-attempt
reason counts 为 `blocked_lz=5`、`blocked_safety=96`、`blocked_energy=226`、
`blocked_dynamics=0`、`committed=0`；这些是 pad 尝试的原因次数，不能求和当作
调用次数或全局收益。首次 attempt 在 step `278`、SOC `0.399989`、energy
`599.984`，compatible-pad availability 为 `4/0`；其中
`blocked_safety=2` 的 reason 包含 `predicted_holder_reachable_tube`。首次
energy rejection 在 step `656`、SOC `0.157654`、energy `236.481`，
compatible-pad availability 为 `5/3`，reason counts 为 `energy=1`、
`safety=1`。所有 native calls 均未成功；UID `57` 在 `892` 仍为
`failed_energy`。`attempt=false` 不计作 reject。

源码和 tests 的本地/server SHA 一致，focused suite 为 `12 passed`（commit
`e3e13d7`）。全局 `PPO_ALLOWED=false`，未使用 held-out；该 phase 不改变
physics、guard、model 或 optimizer，也不宣称修复或新增因果归因。

原始失败回放显示安全拒绝先发生，后期出现能量拒绝；step `445` 仍存在极低
q 的可执行 K=3 分支，因此不将该时刻描述为不可行。这只是单次 failure 加
frozen continuation，尚未选择训练修复，P3 仍未解决。

## Frozen-WAIT 对照 pilot 与完整运行状态

本轮 frozen-WAIT 对照的代码版本为 `9aa7970`，配套文档提交为
`b2aa6f8`。新增 focused tests 为 `10` 个；既有相关 `36` 个测试和
forced-WAIT `6` 个测试保持通过，不重复累计。服务器源码 SHA 与 preflight
校验通过；本轮没有新增环境改动。`preflight_v1.log` 保留了初始工作目录错误：
phase 开始前出现 `ModuleNotFoundError` 且没有 phase outputs；纠正项目 cwd 后，
`preflight_v2` 报告 `PREFLIGHT_OK`，两份记录均保留。

固定 pilot pair 为 environment seed `877101` 与 keyed policy seed `9303101`，
状态为 `PILOT_OK/PAIR_OK`：两臂均运行 `694` ticks、自然终止且未 censor；两臂
的 `24` 个请求一致，ledger 全部 resolved，model、caller RNG、source、safety、
energy 和 wait checks 均通过。两臂 outcome 均为 `23 normal / 1 emergency /
0 failed / 0 diverted / 0 unresolved`；wait 为 `2727.950535124339 s`，
external 为 `335 s`，internal 为 `2392.950535124339 s`，full-scene time 为
`694 s`。raw reward 的差值为 `+2.5`，全部来自 rejection penalty；排除该
penalty 后差值为 `0`。该 pilot 没有观测到 business gain，不形成 performance claim。

完整运行使用显式项目 cwd 启动，`full.json` 完成时间为 `20:50:31`；PID
`2063491` 已退出，状态为
`FULL_OK/pilot_reused`。固定的 `24` 个 complete-scene pairs 全部为
`PAIR_OK`；两臂共 `48` 个 run 均自然终止，censor 与 error 均为 `0`。
safety、source/model/caller-RNG、config/runtime、preflight、24 个请求的
identity、ledger、energy 和 wait checks 均通过。P3 heartbeat 已删除，授权的
后台终端已无继续运行的长任务。

这些是 `6` 个 environment seeds × `4` 个 keyed policy seeds 的 `24` 个 scene
events，合计每臂 `576` 个 request events，不是 `576` 个 distinct environment
 profiles。每臂在 24 个场景中的累计 generated/admitted 均为 `576`，successful
completion 为 `574`，resolved 为 `576`；单场景 generated/admitted 均为 `24`。
business outcomes 汇总为 `491 normal / 83 emergency / 2 failed_energy / 0 diverted /
0 unresolved`。两臂每个 pair 的
request identity、business outcome、wait、elapsed 和 energy 均一致；全体 request
wait 为 `79582.65016586734 s`（每 request `138.1643232046308 s`），elapsed
为 `223350 s`（每 request `387.7604166666667 s`），model energy 为
`185263.64932818164`（每 request `321.6382800836487`）。

| pair | env | keyed policy | outcome N/E/F/D/U | request wait (s) | Δ reward | Δ raw reward excluding penalty |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 877101 | 9303101 | 23/1/0/0/0 | 2727.950535124339 | 2.5 | 0 |
| 2 | 877101 | 9303102 | 23/1/0/0/0 | 2971.032914830321 | 2.24999999999989 | -1.13686837721616e-13 |
| 3 | 877101 | 9303103 | 23/1/0/0/0 | 2389.289311552574 | 2.5 | 0 |
| 4 | 877101 | 9303104 | 23/1/0/0/0 | 2973.392271431140 | 2.85000000000002 | 5.6843418860808e-14 |
| 5 | 877102 | 9303101 | 20/4/0/0/0 | 3181.211230903982 | 2.79999999999995 | -5.6843418860808e-14 |
| 6 | 877102 | 9303102 | 19/5/0/0/0 | 4085.035316034269 | 3.10000000000014 | 1.70530256582424e-13 |
| 7 | 877102 | 9303103 | 20/4/0/0/0 | 3209.871293396141 | 2.00000000000017 | 1.70530256582424e-13 |
| 8 | 877102 | 9303104 | 19/5/0/0/0 | 3860.159914917639 | 3.0999999999998 | -2.27373675443232e-13 |
| 9 | 877103 | 9303101 | 20/4/0/0/0 | 2849.776218448148 | 2.75 | 0 |
| 10 | 877103 | 9303102 | 21/3/0/0/0 | 3072.733442276747 | 2.19999999999999 | 0 |
| 11 | 877103 | 9303103 | 20/4/0/0/0 | 3611.372152201485 | 2.69999999999993 | -5.6843418860808e-14 |
| 12 | 877103 | 9303104 | 19/5/0/0/0 | 4007.885506863763 | 3.24999999999989 | -1.13686837721616e-13 |
| 13 | 877104 | 9303101 | 20/3/1/0/0 | 4195.360899337695 | 3.80000000000018 | 1.70530256582424e-13 |
| 14 | 877104 | 9303102 | 19/5/0/0/0 | 3028.233562992693 | 2 | 0 |
| 15 | 877104 | 9303103 | 21/3/0/0/0 | 2464.012435507064 | 2.5 | 0 |
| 16 | 877104 | 9303104 | 16/7/1/0/0 | 4036.144102041071 | 3.65000000000032 | 2.8421709430404e-13 |
| 17 | 877105 | 9303101 | 21/3/0/0/0 | 3298.552351300552 | 3.55000000000007 | 1.13686837721616e-13 |
| 18 | 877105 | 9303102 | 22/2/0/0/0 | 3055.202866579842 | 2.50000000000011 | 1.13686837721616e-13 |
| 19 | 877105 | 9303103 | 19/5/0/0/0 | 4149.664203967385 | 2.69999999999993 | -5.6843418860808e-14 |
| 20 | 877105 | 9303104 | 22/2/0/0/0 | 3475.689252696497 | 2.04999999999995 | -5.6843418860808e-14 |
| 21 | 877106 | 9303101 | 20/4/0/0/0 | 3448.775915692401 | 3.15000000000009 | 5.6843418860808e-14 |
| 22 | 877106 | 9303102 | 20/4/0/0/0 | 3312.577657439126 | 2.44999999999993 | -5.6843418860808e-14 |
| 23 | 877106 | 9303103 | 20/4/0/0/0 | 2977.679794638645 | 2.60000000000002 | 5.6843418860808e-14 |
| 24 | 877106 | 9303104 | 21/3/0/0/0 | 3201.047015693814 | 2.65000000000009 | 1.13686837721616e-13 |

Legacy total reward was `10825.398813678652` with release rejection penalty
`-426.05`; forced-WAIT total reward was `10890.998813678656` with penalty
`-360.45`. Excluding the penalty, both arms were `11251.448813678655`; the total
reward gain `65.6` came entirely from the rejection penalty. The maximum per-pair
absolute difference after excluding that penalty was
`2.8421709430404007e-13`, treated as floating-point noise. Physical wait, elapsed,
energy and outcome deltas were all `0`.

Failure pairs `13` and `16` are the same request event under environment `877104`
and keyed policy seeds `9303101`/`9303104`: request `13`, UID `46`, `standard`,
initial SOC `0.5549178989494378`, final SOC `0`, request time `30`, terminal time
`862`, `ring_outer`, `energy_depleted`, external wait `29 s`, internal wait
`747.8147075862394 s`, total wait `776.8147075862394 s`, consumed energy
`833.1666666666666`. The event is identical in both arms. These are repeated
request events in the 24-scene comparison, not evidence that the scene is
infeasible or a root-cause attribution for P3. The source before update was
`3fe7+a82`; the separate old source16 true failure with UID `57` is not mixed into
this result.

The frozen-WAIT mechanical and probability semantics passed this protocol. The
mode is not promoted to formal/default behavior, and the result does not verify
learning efficacy or qualification. Joint resource-energy reachability and
executable-candidate coverage evidence, followed by qualification, remain required
before PPO; no new training was approved or launched.

## 边界

pilot 与完整 frozen-WAIT 对照均已完成；尚无学习 efficacy 或 qualification 结论。P3 退化仍未证解决；strong baselines、sensitivity、joint recursive feasibility、完整 V4.2 high adapter 和 qualification 仍是未关闭事实。`C` 的完整 `24×4` 诊断已 `COMPLETED`：24/24 state、2,496/2,496 branch，自然终止且无 error/censor；`PPO_ALLOWED=false` 保持。forced-WAIT 仅作为已完成的实验性 mechanical/probability 对照，不提升为 formal/default，也未批准或启动新训练。下一步需补 joint resource-energy reachability、executable-candidate coverage 和 qualification 证据。
