"""Export the completed 72-job nongeometry study; never run or reanalyze jobs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from itertools import product
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("half6", "original6", "pads3half", "low_soc", "deterministic66", "high_variance66")
METHODS = ("energy_slack", "rh_cp_sat")
RATES = (90, 180)
SEEDS = (11, 23, 71)
KEYS = ("profile", "rate_per_hour", "seed", "method")
OUTCOMES = ("normal_touchdown", "emergency", "failure", "rerouted", "upstream_pending", "airborne_pending")
JOINT = "normal_within_sla_and_reserve_ge_025"
QOS = "ontime_reserve_fraction_all_offered"


def read_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    coverage = report["coverage"]
    if coverage.get("all_completed") is not True or any(coverage.get(k) != 72 for k in ("expected_jobs", "complete_jobs")):
        raise ValueError("export requires a completed 72/72-job study")
    expected = set(product(PROFILES, RATES, SEEDS, METHODS))
    rows = report["per_job"]
    if len(rows) != 72 or {tuple(row[k] for k in KEYS) for row in rows} != expected:
        raise ValueError("nongeometry per-job grid mismatch")
    spec = report["spec"]
    for name, values in (("profiles", PROFILES), ("rates_per_hour", RATES), ("seeds", SEEDS), ("methods", METHODS)):
        if len(spec[name]) != len(values) or set(spec[name]) != set(values):
            raise ValueError("nongeometry spec grid mismatch: " + name)
    for field, value in dict(warmup_s=600, measurement_s=1800, drain_s=600, sla_delay_s=600,
                             progress_interval_s=60, demand_kind="poisson").items():
        if spec[field] != value:
            raise ValueError("nongeometry protocol mismatch: " + field)
    for row in rows:
        if row["offered"] != sum(row[name] for name in OUTCOMES) or row[QOS] != row[JOINT] / row["offered"]:
            raise ValueError("stored full-offered accounting mismatch")
    if len({row["source_hash"] for row in rows}) != 1:
        raise ValueError("mixed source versions cannot be exported as one study")
    paired = report["paired"]
    if paired["unavailable"] or len(paired["method_cp_minus_es"]) != 36 or len(paired["factors_minus_half6"]) != 60:
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
            joint_max=max(row[QOS] for row in rows), passing_095_count=sum(row[QOS] >= .95 for row in rows),
            normal_delay_p95_seed_mean_s=mean(row["normal_delay_p95_s"] for row in rows),
            pad_busy_utilization_seed_mean=mean(row["pad_busy_utilization"] for row in rows),
            pad_upper_zero_loss_per_h=rows[0]["loss_0.00_offered_rate_upper_per_h"],
            pad_upper_epsilon_005_per_h=rows[0]["loss_0.05_offered_rate_upper_per_h"]))
    fields = (*KEYS, "offered", *OUTCOMES, JOINT, QOS, "normal_delay_p95_s", "pad_busy_utilization",
              "finite_integer_count_upper", "finite_bound_observed_joint_count")
    per_seed = [{name: row[name] for name in fields} for row in sorted(report["per_job"], key=lambda r: tuple(r[k] for k in KEYS))]
    paired = []
    deltas = ("offered", JOINT, QOS, *OUTCOMES, "normal_delay_p95_s", "pad_busy_utilization")
    for kind, name in (("method", "method_cp_minus_es"), ("factor", "factors_minus_half6")):
        for row in report["paired"][name]:
            paired.append(dict(kind=kind, profile=row["profile"], rate_per_hour=row["rate_per_hour"],
                seed=row["seed"], method=row.get("method", "rh_cp_sat_minus_energy_slack"),
                difference=row["difference"], **{field + "_delta": row[field + "_delta"] for field in deltas}))
    return summary, per_seed, paired


def plot_qos(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    colors, markers = ("#484878", "#A8A8A8"), ("o", "s")
    labels = ("half6", "original6", "pads3half", "low_soc", "deterministic\n66", "high variance\n66")
    for panel, (axis, rate) in enumerate(zip(axes, RATES)):
        for index, profile in enumerate(PROFILES):
            for m, method in enumerate(METHODS):
                values = sorted((row for row in rows if row["profile"] == profile and row["rate_per_hour"] == rate
                                 and row["method"] == method), key=lambda row: row["seed"])
                y = [row[QOS] for row in values]
                x = index + (-.16 if m == 0 else .16)
                axis.errorbar(x, mean(y), yerr=[[mean(y) - min(y)], [max(y) - mean(y)]],
                    fmt=markers[m], markersize=7.5, markerfacecolor="white", markeredgewidth=1.5,
                    color=colors[m], capsize=4, elinewidth=1.4, zorder=3)
                axis.scatter([x + jitter for jitter in (-.045, 0, .045)], y, marker=markers[m],
                    s=14, color=colors[m], edgecolors="#333333" if m else colors[m], linewidths=.4, zorder=4)
        axis.axhline(.95, color="#606060", linewidth=.9, linestyle="--", zorder=1)
        axis.set(xticks=range(len(PROFILES)), xticklabels=labels, xlim=(-.5, 5.5), ylim=(0, 1.03),
                 yticks=(0, .2, .4, .6, .8, 1.), xlabel="Profile")
        axis.set_title(f"({chr(97 + panel)}) Nominal offered rate {rate} aircraft/h", loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", color="#E6E6E6", linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=8.5)
    axes[0].set_ylabel("Joint on-time normal touchdown with reserve\nFraction of all offered requests")
    handles = [Line2D([], [], color=colors[m], marker=markers[m], markerfacecolor="white", linestyle="none",
                      markersize=7, label=method) for m, method in enumerate(METHODS)]
    handles.append(Line2D([], [], color="#606060", linestyle="--", linewidth=.9, label="0.95 reference"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.53, 1.), fontsize=10)
    fig.text(.53, .018, "Small dots: 3 seeds; large symbols: equal-seed means; bars: observed min–max (not confidence intervals).",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.99, top=.84, bottom=.20, wspace=.08)
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix("." + suffix), dpi=300)
    plt.close(fig)


def results_markdown(report, summary):
    lookup = {(row["profile"], row["rate_per_hour"], row["method"]): row for row in summary}
    rows = report["per_job"]
    def value(profile, rate, method, field="joint_equal_seed_mean"):
        return lookup[(profile, rate, method)][field]
    def comparison(profile, rate):
        return "/".join(f"{value(profile, rate, method):.4f}" for method in METHODS)
    def outcomes(profile, rate, method):
        selected = [row for row in rows if (row["profile"], row["rate_per_hour"], row["method"]) == (profile, rate, method)]
        return "/".join(str(sum(row[field] for row in selected)) for field in ("normal_touchdown", "emergency", "failure"))
    nontrivial = [row for row in rows if row["finite_integer_count_upper"] < row["offered"]]
    totals = {field: sum(row[field] for row in rows) for field in ("offered", JOINT, *OUTCOMES)}
    text = rf"""# 非几何因素：完整终端服务质量结果

本研究的非几何子集已完成 **72/72 jobs**：6 profiles × 2 标称 Poisson 到达率（90/180 aircraft/h）× 3 seeds（11/23/71）× 2 methods。inner 半径固定为 300 m、设计槽位数 K=18；除 pads3half 使用 3 pads 外均为 6 pads。warmup/measurement/drain 为 600/1800/600 s，接地 SLA 为 600 s，reserve 要求为接地 SOC≥0.25。联合服务质量以测量 cohort 的**全部外部 offered 请求**为分母，正常接地、满足 SLA 与 reserve 三条件同时成立才计入分子；drain 继续跟踪该 cohort，不改变分母。

结果读取自已有 [report.json](../../artifacts/full_capacity_20261025/nongeometry_study_report/report.json)，未运行新仿真或重分析原 jobs。原 96-job campaign 的 24 个旧几何 jobs 全部排除（含 4 个旧完成项），保留原始状态；其 20 个 INVALID 使父 campaign 的 `run.exit=1`，与非几何子集完整性并不矛盾。子集沿用 source `{report['source_hash']}`；36 个同 seed 方法配对和 60 个 profile 对 half6 配对均已存在于报告，unavailable=0。本文完成非几何结果交付；截至本次导出，修复后的同版本几何研究仍为 **35/36，1 job pending**，完整研究尚未完成，见[状态文档](full_capacity_validation_status.md)。

## 观测服务质量与必要资源界

pad 工作量守恒给出零损失必要上界 \(3600m/E[13.5+S]\)。允许非正常比例 ε 时，\(C^{{UB}}_{{offered}}(ε)=3600m/\min_{{0≤q≤p,\sum q≥1-ε}}\sum_cq_c(13.5+S_c)\)，乐观允许优先舍弃较长服务类别。推导见[必要界补充](full_capacity_bounds_derivation.md)。这些是长期 offered-rate 的必要上界，不保证 SLA、SOC、空域分离或某策略可达，也不能与有限 cohort 接地吞吐混作同一对象。

| Profile | 零损失 pad 上界（/h） | ε=0.05 offered 上界（/h） |
|---|---:|---:|
"""
    for profile in PROFILES:
        s = lookup[(profile, 90, "energy_slack")]
        text += f"| {profile} | {s['pad_upper_zero_loss_per_h']:.4f} | {s['pad_upper_epsilon_005_per_h']:.4f} |\n"
    text += f"""
下表由 [24 行 summary CSV](tables/full_capacity_nongeometry_summary.csv) 自动生成；方法缩写 ES=energy_slack、CP=rh_cp_sat。联合均值对 3 个 seed 等权，范围为实际最小–最大值；joint/offered 是三个 seed 的计数之和，**其比值不替代等权 seed 均值**。pass 是联合比例≥0.95 的 seed 数。p95 列是各 seed 正常接地延迟 p95 的均值，pad util 是测量窗口物理 pad 占用比例的 seed 均值。

| Profile | 标称率 /h | 方法 | joint/offered | 联合均值 [min,max] | pass /3 | 正常延迟 p95 均值 s | pad util 均值 |
|---|---:|---|---:|---:|---:|---:|---:|
"""
    for row in summary:
        text += (f"| {row['profile']} | {row['rate_per_hour']} | {'ES' if row['method']=='energy_slack' else 'CP'} | "
                 f"{row['joint_sum']}/{row['offered_sum']} | {row['joint_equal_seed_mean']:.4f} "
                 f"[{row['joint_min']:.4f},{row['joint_max']:.4f}] | {row['passing_095_count']}/3 | "
                 f"{row['normal_delay_p95_seed_mean_s']:.3f} | {row['pad_busy_utilization_seed_mean']:.4f} |\n")
    text += f"""
### 相同均值服务时间不决定有限时域表现

half6、deterministic66 与 high_variance66 均为 6 pad、E[S]=66 s，其零损失 pad 上界相同（{value('half6',90,'energy_slack','pad_upper_zero_loss_per_h'):.4f}/h），但在标称 180/h 下 ES 联合均值依次为 {value('half6',180,'energy_slack'):.4f}、{value('deterministic66',180,'energy_slack'):.4f}、{value('high_variance66',180,'energy_slack'):.4f}，CP 为 {value('half6',180,'rh_cp_sat'):.4f}、{value('deterministic66',180,'rh_cp_sat'):.4f}、{value('high_variance66',180,'rh_cp_sat'):.4f}。这些样本不支持“服务方差越大，表现必然越差”的单调规律；CP 在 half6 与 deterministic66 上较高，在 high_variance66 上较低，也未显示对所有配置一致占优。相同均值只保证该零损失工作量界相同；ε=0.05 的选择性界还依赖类别分布，deterministic66 的值为 {value('deterministic66',90,'energy_slack','pad_upper_epsilon_005_per_h'):.4f}/h。

### 能量与 pad 数共同限制实现服务质量

low_soc 的 pad 资源界与 half6 不变，但标称 180/h 的 ES/CP 联合均值降为 {comparison('low_soc',180)}，两方法均为 0/3 seeds 达到 0.95；正常/应急/失败计数分别为 ES {outcomes('low_soc',180,'energy_slack')}、CP {outcomes('low_soc',180,'rh_cp_sat')}。这显示被测策略对初始 SOC 的敏感性，说明仅用 pad 工作量界不能解释完整服务质量。当前必要 pad 界不含 SOC，并非计算错误。上游等待不消耗 SOC；能量约束从正式准入后作用，不能把外部释放后的全部等待直接解释为 SOC 消耗。不同准入决策仍可能减少空中等待，因此该结果不证明最优物理容量随低 SOC 下降。

pads3half 的零损失与 ε=0.05 offered 上界为 {value('pads3half',90,'energy_slack','pad_upper_zero_loss_per_h'):.4f}/h 和 {value('pads3half',90,'energy_slack','pad_upper_epsilon_005_per_h'):.4f}/h。在 90/h（低于这两个资源上界）时 ES/CP 均值已为 {comparison('pads3half',90)}，两方法均仅 1/3 seeds 达到 0.95；180/h 时为 {comparison('pads3half',180)}。必要工作量条件不是充分条件；测量窗口 pad busy utilization 小于 1，也不能证明仍有可被当前受限到达、能量和冲突过程利用的闲余容量。

original6 的服务时间较长，零损失上界为 {value('original6',90,'energy_slack','pad_upper_zero_loss_per_h'):.4f}/h，ε=0.05 为 {value('original6',90,'energy_slack','pad_upper_epsilon_005_per_h'):.4f}/h。90/h 下两方法所有 seeds 联合比例均为 1；180/h 下 ES/CP 联合均值为 {comparison('original6',180)}。界与实际结果之间的差距同时包含有限时域、能量、运动学、分离和策略等因素，不能解释为纯调度损失或策略最优性差距。

### 有限 cohort 界与核算边界

有限 cohort LP 数量上界仅在 {len(nontrivial)}/72 jobs 小于 offered，其余 {72-len(nontrivial)} 项均为平凡 N。两个非平凡项均为 pads3half、seed 71、180/h：

| 方法 | offered | 有限 cohort 整数上界 | 观测联合完成 |
|---|---:|---:|---:|
"""
    for row in nontrivial:
        text += f"| {row['method']} | {row['offered']} | {row['finite_integer_count_upper']} | {row[JOINT]} |\n"
    text += f"""
该界仅将 82 个 offered 请求的乐观正常按时完成上界收紧至 81；观测联合完成分别为 2 和 4，并未与上界接近。LP 放松了 SOC 等约束，故不能据此单独归因于调度器。

全部 72 jobs 共有 {totals['offered']} 个 offered 计数和 {totals[JOINT]} 个联合完成计数；六类结局合计为正常 {totals['normal_touchdown']}、应急 {totals['emergency']}、失败 {totals['failure']}、rerouted {totals['rerouted']}、upstream pending {totals['upstream_pending']}、airborne pending {totals['airborne_pending']}。这里跨配置与方法重复计入相同随机输入，仅用于逐 job 核算，不能解释为 {totals['offered']} 个独立样本。`failure` 是有效仿真内的运行结局，与实现无效的 `INVALID` 不同。

配置、manifest、存储 cohort 与重算 cohort、cohort/world 请求守恒检查均为 true；所有 selected results 的 terminated_invalid=false，机制审计遗漏飞机为 0。能量恒等式最大绝对残差为 {max(row['energy_identity_max_abs_residual_j'] for row in rows):.16g} J，等待重建最大绝对残差为 {max(row['holding_time_max_abs_residual_s'] for row in rows):.16g} s。`derived_move_energy` 是总能耗减去稳定等待能耗得到的残差，不是独立积分；数值核算一致性不验证能耗或轨迹物理 surrogate 的真实性。

只有 3 seeds，均值与 min–max 是描述性统计而非置信区间，本文未作显著性检验。90/180 两个标称率不足以定位 half6 的服务质量阈值，也不足以证明长期稳定容量；不插值拟合阈值。

## 图与可复用数据

![非几何 profile 的联合服务质量](figures/full_capacity_nongeometry_qos.png)

图注：两个面板分别显示标称 90 与 180 aircraft/h。每个类别包含两方法的横向错位标记，细点为 seeds 11/23/71，较大空心标记为等权 seed 均值，竖线为观测 min–max（非置信区间）。分母为全部 offered 请求；虚线是 0.95 参考线，类别之间不连线。

Figure Contract：核心证据是相同资源必要界下的有限时域服务质量仍随 profile 和方法变化；证据来自完整 72-job report 的逐 seed 结果。采用两面板 quantitative grid / categorical point-range，分别回答 90/h 与 180/h 的配置差异；主要风险为小样本范围被误作置信区间或容量阈值，故逐 seed 全部显示、不作插值。未指定投稿 venue，当前为通用论文全宽版式。

QA Contract：CSV 覆盖 24 summary / 72 per-seed / 96 paired 行；全量数据、完整 offered 分母、n=3、min–max 非 CI 已明示。两方法同时用颜色和圆/方标记区分；y 轴为 0–1.03；SVG 保留可编辑文字，PNG 为 300 dpi。自动 SVG QA 已通过（32 个可编辑文字元素、无禁用色板、灰度检查通过）；主对话已于北京时间 2026-10-27 完成当前图稿的视觉与数据复核。未指定目标 venue，以上不构成特定投稿规范的最终认证。

导出文件：[summary](tables/full_capacity_nongeometry_summary.csv)、[per-seed](tables/full_capacity_nongeometry_per_seed.csv)、[paired](tables/full_capacity_nongeometry_paired.csv)、[SVG](figures/full_capacity_nongeometry_qos.svg)、[PNG](figures/full_capacity_nongeometry_qos.png)。paired CSV 直接选取报告已有的同 seed 方法差分（CP−ES，36 行）及 profile−half6 差分（60 行），不构造跨版本或新增配对。

重建命令（读取现有报告，不运行仿真）：

```powershell
.\\venv\\Scripts\\python.exe scripts/analysis/export_full_capacity_nongeometry_results.py
```
"""
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/full_capacity_20261025/nongeometry_study_report/report.json")
    args = parser.parse_args(argv)
    report = read_report(args.report)
    summary, per_seed, paired = tables(report)
    table_dir, figure_dir = ROOT / "docs/research/tables", ROOT / "docs/research/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", summary), ("per_seed", per_seed), ("paired", paired)):
        write_csv(table_dir / f"full_capacity_nongeometry_{name}.csv", data)
    plot_qos(report["per_job"], figure_dir / "full_capacity_nongeometry_qos")
    (ROOT / "docs/research/full_capacity_nongeometry_results.md").write_text(results_markdown(report, summary), encoding="utf-8")
    print(json.dumps(dict(summary_rows=len(summary), per_seed_rows=len(per_seed), paired_rows=len(paired),
        offered_sum=sum(row["offered"] for row in per_seed), joint_sum=sum(row[JOINT] for row in per_seed)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
