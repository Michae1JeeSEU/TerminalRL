"""Export the completed same-v2 36-job geometry report without running jobs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from itertools import product
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("radius250", "half6", "radius350")
METHODS = ("energy_slack", "rh_cp_sat")
RATES, SEEDS = (90, 180), (11, 23, 71)
KEYS = ("profile", "rate_per_hour", "seed", "method")
OUTCOMES = ("normal_touchdown", "emergency", "failure", "rerouted", "upstream_pending", "airborne_pending")
JOINT = "normal_within_sla_and_reserve_ge_025"
QOS = "ontime_reserve_fraction_all_offered"
SOURCE = "afa8b9fb780b2f63f1bfe1f08564798dc43f72e379137a7b0f5361d561271db9"


def read_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    coverage, rows = report["coverage"], report["per_job"]
    if coverage.get("all_completed") is not True or any(coverage.get(k) != 36 for k in ("expected_jobs", "complete_jobs")):
        raise ValueError("export requires a completed 36/36-job geometry study")
    if len(rows) != 36 or {tuple(row[k] for k in KEYS) for row in rows} != set(product(PROFILES, RATES, SEEDS, METHODS)):
        raise ValueError("geometry per-job grid mismatch")
    spec = report["spec"]
    for name, values in (("profiles", PROFILES), ("rates_per_hour", RATES), ("seeds", SEEDS), ("methods", METHODS)):
        if len(spec[name]) != len(values) or set(spec[name]) != set(values):
            raise ValueError("geometry spec grid mismatch: " + name)
    for field, value in dict(warmup_s=600, measurement_s=1800, drain_s=600, sla_delay_s=600,
                             progress_interval_s=60, demand_kind="poisson").items():
        if spec[field] != value:
            raise ValueError("geometry protocol mismatch: " + field)
    if {row["source_hash"] for row in rows} != {SOURCE} or report["source_hash"] != SOURCE:
        raise ValueError("geometry export requires the same repaired v2 source for all jobs")
    for row in rows:
        if row["offered"] != sum(row[name] for name in OUTCOMES) or row[QOS] != row[JOINT] / row["offered"]:
            raise ValueError("stored full-offered accounting mismatch")
    paired = report["paired"]
    if paired["unavailable"] or len(paired["method_cp_minus_es"]) != 18 or len(paired["factors_minus_half6"]) != 24:
        raise ValueError("incomplete stored same-seed pair coverage")
    return report


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(report):
    grouped = defaultdict(list)
    for row in report["per_job"]:
        grouped[(row["profile"], row["rate_per_hour"], row["method"])].append(row)
    summary = []
    for profile, rate, method in product(PROFILES, RATES, METHODS):
        rows = grouped[(profile, rate, method)]
        summary.append(dict(profile=profile, rate_per_hour=rate, method=method, n=len(rows),
            offered_sum=sum(row["offered"] for row in rows), joint_sum=sum(row[JOINT] for row in rows),
            joint_equal_seed_mean=mean(row[QOS] for row in rows), joint_min=min(row[QOS] for row in rows),
            joint_max=max(row[QOS] for row in rows), passing_090_count=sum(row[QOS] >= .90 for row in rows),
            passing_095_count=sum(row[QOS] >= .95 for row in rows), passing_099_count=sum(row[QOS] >= .99 for row in rows),
            normal_delay_p95_seed_mean_s=mean(row["normal_delay_p95_s"] for row in rows),
            normal_delay_p95_min_s=min(row["normal_delay_p95_s"] for row in rows),
            normal_delay_p95_max_s=max(row["normal_delay_p95_s"] for row in rows),
            pad_busy_utilization_seed_mean=mean(row["pad_busy_utilization"] for row in rows),
            all_admitted_inner_holding_seed_mean_aircraft_s=mean(row["audit_inner_holding_s"] for row in rows),
            **{field: rows[0][field] for field in ("inner_radius_m", "inner_design_K", "inner_stable_holding_power_w", "touchdown_lower_s")},
            pad_upper_zero_loss_per_h=rows[0]["loss_0.00_offered_rate_upper_per_h"],
            pad_upper_epsilon_005_per_h=rows[0]["loss_0.05_offered_rate_upper_per_h"]))
    fields = (*KEYS, "offered", *OUTCOMES, JOINT, QOS, "normal_delay_p95_s", "pad_busy_utilization",
              "audit_inner_holding_s", "finite_integer_count_upper", "finite_bound_observed_joint_count")
    per_seed = [{name: row[name] for name in fields} for row in sorted(report["per_job"], key=lambda r: tuple(r[k] for k in KEYS))]
    paired = []
    deltas = ("offered", JOINT, QOS, *OUTCOMES, "normal_delay_p95_s", "pad_busy_utilization", "audit_inner_holding_s")
    for kind, name in (("method", "method_cp_minus_es"), ("factor", "factors_minus_half6")):
        for row in report["paired"][name]:
            paired.append(dict(kind=kind, profile=row["profile"], rate_per_hour=row["rate_per_hour"],
                seed=row["seed"], method=row.get("method", "rh_cp_sat_minus_energy_slack"),
                difference=row["difference"], **{field + "_delta": row[field + "_delta"] for field in deltas}))
    return summary, per_seed, paired


def plot_delay(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    colors, markers = ("#484878", "#A8A8A8"), ("o", "s")
    upper = 50 * (int(max(row["normal_delay_p95_s"] for row in rows) / 50) + 1)
    for panel, (axis, rate) in enumerate(zip(axes, RATES)):
        for index, profile in enumerate(PROFILES):
            for m, method in enumerate(METHODS):
                selected = sorted((row for row in rows if (row["profile"], row["rate_per_hour"], row["method"]) ==
                                   (profile, rate, method)), key=lambda row: row["seed"])
                y = [row["normal_delay_p95_s"] for row in selected]
                x = index + (-.12 if m == 0 else .12)
                axis.errorbar(x, mean(y), yerr=[[mean(y) - min(y)], [max(y) - mean(y)]],
                    fmt=markers[m], markersize=7.5, markerfacecolor="white", markeredgewidth=1.5,
                    color=colors[m], capsize=4, elinewidth=1.4, zorder=3)
                axis.scatter([x + jitter for jitter in (-.035, 0, .035)], y, marker=markers[m],
                    s=14, color=colors[m], edgecolors="#333333" if m else colors[m], linewidths=.4, zorder=4)
        axis.set(xticks=(0, 1, 2), xticklabels=("250", "300", "350"), xlim=(-.4, 2.4), ylim=(0, upper),
                 xlabel="Inner-ring radius R (m)")
        axis.set_title(f"({chr(97 + panel)}) Nominal offered rate {rate} aircraft/h", loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", color="#E6E6E6", linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Normal-touchdown delay p95 (s)\nMeasurement cohort; normal outcomes only")
    handles = [Line2D([], [], color=colors[m], marker=markers[m], markerfacecolor="white", linestyle="none",
                      markersize=7, label=method) for m, method in enumerate(METHODS)]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.53, 1.), fontsize=10)
    fig.text(.53, .018, "Small dots: 3 seeds; large symbols: equal-seed means; bars: observed min–max (not confidence intervals).",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.99, top=.84, bottom=.18, wspace=.08)
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix("." + suffix), dpi=300)
    plt.close(fig)


def results_markdown(report, summary):
    rows = report["per_job"]
    totals = {field: sum(row[field] for row in rows) for field in ("offered", JOINT, *OUTCOMES)}
    nontrivial = sum(row["finite_integer_count_upper"] < row["offered"] for row in rows)
    factors = [row for row in report["paired"]["factor_difference_summary"] if row["rate_per_hour"] == 180]
    text = rf"""# 同版本几何研究：半径、延迟与联合服务质量

修复后的几何研究已完成 **36/36 jobs**：inner 半径 250/300/350 m × 标称 Poisson 到达率 90/180 aircraft/h × seeds 11/23/71 × 两方法 energy_slack（ES）与 rh_cp_sat（CP）。warmup/measurement/drain=600/1800/600 s，接地 SLA=600 s，reserve 要求为接地 SOC≥0.25。所有组合均为 6 pads、相同半服务时间分布。三个 source-v2 campaigns 均完成，已有[研究报告](../../artifacts/full_capacity_20261025/geometry_study_report/report.json)完整覆盖；本导出仅读取报告，不运行仿真或重新分析 jobs。

所有 jobs 使用 source `{report['source_hash']}`，方法配对 18 个、相对 R=300 m 的 factor 配对 24 个，unavailable=0。旧 v1 的 24 个几何 jobs 全部排除，包括其中 4 个完成项；v1 [非几何研究](full_capacity_nongeometry_results.md)与本 v2 几何研究保持分开，不把跨版本 half6 拼接为同质对照。当前预定 screening 的两个子研究均已完成；进一步阈值验证仍待开展，不能据此宣告完整研究目标已完成。

## 从必要界到有限时域观测

几何同时改变静态储存量、稳定等待功率和最早接地时间。静态槽位关系为 \(K(R)=\lfloor\pi/\arcsin[d/(2R)]\rfloor\)；正常能量可行性要求 \(Q(s_i-0.25)\ge E_{{move,i}}+\sum_rP_rW_{{ir}}\)。final 阶段的必要时间下界至少为径向位移/速度、垂直位移/速度及消除初始水平速度所需时间三者的最大值。相应[理论笔记](terminal_service_capacity.md)与[必要界补充](full_capacity_bounds_derivation.md)解释这些约束的对象与单位。

| Inner R（m） | 设计 K | 稳定等待功率（W） | 接地时间必要下界（s） | 零损失 pad 上界（/h） | ε=0.05 offered 上界（/h） |
|---:|---:|---:|---:|---:|---:|
"""
    for profile in PROFILES:
        row = next(row for row in summary if row["profile"] == profile)
        text += (f"| {row['inner_radius_m']:.0f} | {row['inner_design_K']} | {row['inner_stable_holding_power_w']:.1f} | "
                 f"{row['touchdown_lower_s']:.1f} | {row['pad_upper_zero_loss_per_h']:.4f} | {row['pad_upper_epsilon_005_per_h']:.4f} |\n")
    text += """
所有半径的 pad 数和类别工作量分布相同，因此零损失与允许 5% 非正常质量的 pad 必要界均相同。它们约束长期 offered-rate 的必要资源工作量，不保证能量、分离、SLA 或具体策略可达；不能从更大的 K 或更低等待功率直接推出更低延迟或更高服务容量。

联合服务质量的分母为测量 cohort 的**全部外部 offered 请求**；分子要求正常接地、在 600 s SLA 内且接地 SOC≥0.25。drain 延长跟踪，不改变分母。下面 joint/offered 为三个 seed 的计数之和，联合均值对 seed 等权；二者口径不同，均保留。pass 列依次给出联合比例≥0.90、≥0.95、≥0.99 的 seed 数，每项分母均为 3；0.95 为主要阈值，0.90/0.99 为协议规定的次要阈值。**p95 仅针对该测量 cohort 中正常接地者**，不含应急与失败，因此不能替代全 offered 的联合服务质量。

| R（m） | 标称率 /h | 方法 | joint/offered | 联合均值 [min,max] | pass90,95,99（各 /3） | 正常延迟 p95 均值 [min,max]（s） | pad util 均值 | inner 等待均值（aircraft-s） |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
"""
    for row in summary:
        text += (f"| {row['inner_radius_m']:.0f} | {row['rate_per_hour']} | {'ES' if row['method']=='energy_slack' else 'CP'} | "
                 f"{row['joint_sum']}/{row['offered_sum']} | {row['joint_equal_seed_mean']:.6f} [{row['joint_min']:.6f},{row['joint_max']:.6f}] | "
                 f"{row['passing_090_count']},{row['passing_095_count']},{row['passing_099_count']} | {row['normal_delay_p95_seed_mean_s']:.3f} "
                 f"[{row['normal_delay_p95_min_s']:.3f},{row['normal_delay_p95_max_s']:.3f}] | "
                 f"{row['pad_busy_utilization_seed_mean']:.4f} | {row['all_admitted_inner_holding_seed_mean_aircraft_s']:.3f} |\n")
    text += f"""
全部 36 jobs 的联合比例均≥0.95；90/h 下全部为 1。180/h 下，R=300 m 的 ES/CP 均值为 0.995935/0.995935，R=250 m 为 0.995935/0.987805，R=350 m 为 0.991870/0.987805。当前两载荷点未定位任何半径的服务质量阈值，不能由这些接近 1 的比例作容量排名。

inner 等待量来自事件重建，统计到仿真结束的**全部已准入飞机，包括 warmup 到达**，单位为 aircraft-s；它与 p95 的测量 cohort 不是同一群体，不能把其总量差直接分解为 p95 的原因。

## 同 seed 延迟差分及其解释范围

以下取已有配对结果，在标称 180/h 下比较同 seed、同方法的 profile−R300 正常延迟 p95 差分；正数表示该 profile 的 p95 更高。各范围是三个配对 seed 的实际 min–max，不是置信区间。

| 半径相对 R300 | 方法 | p95 差分均值（s） | 配对 min–max（s） |
|---|---|---:|---:|
"""
    for row in sorted(factors, key=lambda row: (row["profile"], row["method"])):
        text += (f"| {row['profile']}−half6 | {'ES' if row['method']=='energy_slack' else 'CP'} | "
                 f"{row['normal_delay_p95_s_delta_mean']:+.3f} | "
                 f"[{row['normal_delay_p95_s_delta_min']:+.3f},{row['normal_delay_p95_s_delta_max']:+.3f}] |\n")
    text += f"""
R350 相对 R300 的 p95 差分均值在 ES/CP 下分别为 +27.868/+39.301 s；ES 三个差分均为正，CP 包含负值。R250 的 ES 差分均为负，CP 的范围则跨越 0。这些观测未建立普遍半径排序，也未独立识别因果机制。增大半径同时扩大缓冲槽位、降低稳定等待功率，并改变运动路径、相位/冲突关系及固定搜索预算覆盖的候选集合；更大的缓冲与更低的功率不保证端到端延迟下降。CP 的有限候选求解结果也不构成物理全局最优或容量证书。

## 数量界与审计口径

有限 cohort LP 的整数数量上界在 {nontrivial}/36 jobs 小于 offered，即本次全部 {36-nontrivial} 项均为平凡 N；没有一项违例。该放松界并未在本 screening 中给出有效的紧容量区分。全部 jobs 的 offered 计数合计为 {totals['offered']}，正常及联合完成均为 {totals[JOINT]}，应急为 {totals['emergency']}，失败、rerouted、upstream pending 与 airborne pending 均为 0。这些数跨方法和配置重复计入相同随机输入，仅用于结果核算，不能解释为 {totals['offered']} 个独立样本。

配置、manifest、存储/重算 cohort、cohort/world 请求守恒检查均为 true；所有 selected results 的 terminated_invalid=false，审计遗漏飞机为 0。能量恒等式最大绝对残差为 {max(row['energy_identity_max_abs_residual_j'] for row in rows):.16g} J，等待重建最大绝对残差为 {max(row['holding_time_max_abs_residual_s'] for row in rows):.16g} s。`derived_move_energy` 是总能耗减稳定等待能耗的残差，不是独立积分；这些数值检查不验证物理 surrogate 的真实性。有效仿真内的 emergency/failure 结局与实现无效的 INVALID 是不同概念。

本结果只有 3 seeds，均值与范围均为描述性观测；不作显著性、长期稳定性或精确服务容量阈值主张。进一步阈值验证及完整研究进度见[状态文档](full_capacity_validation_status.md)。

## 图与机器导出数据

![三种半径的正常接地延迟 p95](figures/full_capacity_geometry_delay.png)

图注：面板 (a)/(b) 分别为标称 90/180 aircraft/h。三个离散半径各显示两方法的错位标记；小标记为 3 个 seed，空心大标记为等权均值，竖线为观测 min–max（非 CI）。延迟只针对测量 cohort 的正常接地者；所有 36 jobs 的全 offered 联合比例另列于上表。半径点之间不连线，不作平滑或插值拟合。

Figure Contract：以完整 36-job report 为主要证据，采用两面板 quantitative grid / discrete point-range，分别显示两个载荷点的正常延迟分布。核心观察是增加 K、降低等待功率并未保证延迟下降；小样本及正常结局条件化是主要解释边界。CSV 源数据与图共同交付，图不承担容量阈值或独立因果机制证明。

QA Contract：12 行 summary / 36 行 per-seed / 42 行 paired 数据全部导出；n=3、正常结局筛选与 min–max 非 CI 均已标注。两方法沿用非几何图的圆/方标记与颜色，y 轴从 0 起；SVG 文字可编辑，PNG 为 300 dpi。自动 SVG QA 已通过（21 个可编辑文字元素、无禁用色板、灰度检查通过）；主对话已于北京时间 2026-10-27 完成当前图稿的视觉与数据复核。未指定 venue，不宣称特定投稿规范认证。

数据：[summary](tables/full_capacity_geometry_summary.csv)、[per-seed](tables/full_capacity_geometry_per_seed.csv)、[paired](tables/full_capacity_geometry_paired.csv)；图：[SVG](figures/full_capacity_geometry_delay.svg)、[PNG](figures/full_capacity_geometry_delay.png)。paired CSV 仅选取已有的 18 个同 seed 方法差分（CP−ES）和 24 个同 source 的 profile−half6 差分，不构造新配对。

重建命令（只读已有报告）：

```powershell
.\\venv\\Scripts\\python.exe scripts/analysis/export_full_capacity_geometry_results.py
```
"""
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/full_capacity_20261025/geometry_study_report/report.json")
    args = parser.parse_args(argv)
    report = read_report(args.report)
    summary, per_seed, paired = tables(report)
    table_dir, figure_dir = ROOT / "docs/research/tables", ROOT / "docs/research/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", summary), ("per_seed", per_seed), ("paired", paired)):
        write_csv(table_dir / f"full_capacity_geometry_{name}.csv", data)
    plot_delay(report["per_job"], figure_dir / "full_capacity_geometry_delay")
    (ROOT / "docs/research/full_capacity_geometry_results.md").write_text(results_markdown(report, summary), encoding="utf-8")
    print(json.dumps(dict(summary_rows=len(summary), per_seed_rows=len(per_seed), paired_rows=len(paired),
        offered_sum=sum(row["offered"] for row in per_seed), joint_sum=sum(row[JOINT] for row in per_seed)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
