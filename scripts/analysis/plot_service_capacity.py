"""Plot verified reduced inner-entry queue results without changing raw data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENARIOS = (
    "baseline6half", "original6", "pads3half", "low_soc6half",
    "same_mean_deterministic", "same_mean_high_variance", "inner_radius250", "inner_radius350",
)
RATES = (60., 90., 120., 150., 180., 210., 240., 270., 300.)
SEEDS = (11, 23, 47, 71, 101)
PANELS = (
    ("Infrastructure", ("baseline6half", "original6", "pads3half")),
    ("Equal mean ground service (66 s)", ("baseline6half", "same_mean_deterministic", "same_mean_high_variance")),
    ("SOC distribution", ("baseline6half", "low_soc6half")),
    ("Inner-radius counterfactuals", ("inner_radius250", "baseline6half", "inner_radius350")),
)
# Baseline is identical across panels; variants additionally differ by line and marker.
STYLES = {
    "baseline6half": ("6 pads; half service", "#484878", "o", "-"),
    "original6": ("6 pads; original service", "#7C9B98", "s", "--"),
    "pads3half": ("3 pads; half service", "#B58B74", "^", "-."),
    "low_soc6half": ("Lower SOC", "#7C9B98", "s", "--"),
    "same_mean_deterministic": ("Deterministic 66 s", "#7C9B98", "s", "--"),
    "same_mean_high_variance": ("Higher variance", "#B58B74", "^", "-."),
    "inner_radius250": ("250 m", "#7C9B98", "s", "--"),
    "inner_radius350": ("350 m", "#B58B74", "^", "-."),
}


def validate_data(payload, allow_smoke=False):
    """Require the full pre-specified study unless smoke mode was explicit."""
    if payload.get("model_kind") != "reduced_model_inner_entry_fcfs":
        raise ValueError("input must be reduced_model_inner_entry_fcfs results")
    rows = payload.get("rows", [])
    counts = payload.get("task_counts", {})
    if not rows or payload.get("failed_tasks"):
        raise ValueError("input has no rows or contains failed tasks")
    if any(counts.get(k) != 0 for k in ("failed", "missing", "duplicate", "duplicate_requested")):
        raise ValueError("task counts report failed, missing or duplicate tasks")
    if counts.get("expected") != len(rows) or counts.get("completed") != len(rows):
        raise ValueError("expected/completed task counts do not match row count")
    keys, warmups, durations = set(), set(), set()
    group_seeds = defaultdict(set)
    for row in rows:
        scenario, seed, rate = row["scenario"], row["seed"], float(row["offered_rate_per_h"])
        if scenario not in SCENARIOS or row.get("model_kind") != payload["model_kind"]:
            raise ValueError("unknown scenario or inconsistent row model_kind")
        key = (scenario, seed, rate)
        if key in keys:
            raise ValueError("duplicate scenario/seed/load row")
        keys.add(key)
        group_seeds[(scenario, rate)].add(seed)
        warmups.add(float(row["warmup_s"]))
        durations.add(float(row["cohort_duration_s"]))
        if not math.isfinite(rate) or rate <= 0 or rate > 310:
            raise ValueError("load lies outside positive plotted range [0, 310]")
        if row["arrivals"] != sum(row[k] for k in ("normal", "reject_buffer", "reject_deadline")):
            raise ValueError("per-row arrival conservation failed")
        if any(not isinstance(row[k], int) or row[k] < 0 for k in ("arrivals", "normal", "reject_buffer", "reject_deadline")):
            raise ValueError("invalid arrival/outcome counts")
        if row["arrivals"] <= 0:
            raise ValueError("normal fraction undefined for zero-arrival cohort")
        fraction = row["normal_fraction"]
        if not math.isfinite(fraction) or not 0 <= fraction <= 1 or not math.isclose(fraction, row["normal"] / row["arrivals"], abs_tol=1e-12):
            raise ValueError("normal fraction disagrees with cohort counts")
        if (row.get("drained_waiting_backlog") != 0 or row.get("drained_in_service") != 0
                or row["calendar_drain_end_s"] < row["cohort_end_s"]):
            raise ValueError("cohort calendar has not drained")
    if len(warmups) != 1 or len(durations) != 1:
        raise ValueError("all rows must use identical warmup and cohort duration")
    warmup, duration = next(iter(warmups)), next(iter(durations))
    if not math.isfinite(warmup) or warmup < 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid cohort horizon")
    if not allow_smoke:
        expected = {(s, seed, rate) for s in SCENARIOS for seed in SEEDS for rate in RATES}
        if keys != expected or len(rows) != 360 or warmup != 3600. or duration != 14400.:
            raise ValueError("formal plot requires all 360 pre-specified tasks and 1 h + 4 h horizons; use --allow-smoke explicitly for previews")
    seed_counts = sorted({len(s) for s in group_seeds.values()})
    return {"warmup_s": warmup, "cohort_duration_s": duration, "seed_counts": seed_counts,
            "row_count": len(rows), "smoke": bool(allow_smoke)}


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], float(row["offered_rate_per_h"]))].append(row["normal_fraction"])
    return [{"scenario": scenario, "offered_rate_per_h": rate,
             "mean": statistics.mean(values), "min": min(values), "max": max(values), "n": len(values)}
            for (scenario, rate), values in sorted(grouped.items())]


def plot(input_path, output, allow_smoke=False):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    verified = validate_data(payload, allow_smoke)
    data = aggregate(payload["rows"])
    output.mkdir(parents=True, exist_ok=True)
    stem = "service_capacity_smoke" if allow_smoke else "service_capacity"
    svg, png = output / f"{stem}.svg", output / f"{stem}.png"
    exports = [svg, png, output / "plot_data.csv", output / "figure_contract.json", output / "qa_brief.json"]
    if input_path.resolve() in {path.resolve() for path in exports}:
        raise ValueError("output must not overwrite raw input results")
    seed_label = "/".join(map(str, verified["seed_counts"]))
    caption = (f"Poisson, FCFS, {seed_label} seeds per load; "
               f"{verified['warmup_s'] / 3600:g} h warmup + {verified['cohort_duration_s'] / 3600:g} h arrival cohort + drain.\n"
               "Bands: seed min-max (descriptive, not CI); not full-terminal capacity. 95% / 99%: reference only.")
    title = "Reduced inner-entry queue: normal service fraction"
    if allow_smoke:
        title = "SMOKE PREVIEW — " + title
    contract = {"core_conclusion": "Compare controlled factors affecting cohort normal service fraction in the reduced queue.",
                "evidence_hierarchy": {"primary": str(input_path.resolve()), "context": "95% and 99% reference lines only"},
                "figure_archetype": "quantitative grid; trends with descriptive seed ranges",
                "panel_mapping": [{"panel": chr(97+i), "question": title, "scenarios": list(names)} for i, (title, names) in enumerate(PANELS)],
                "target_venue": "unspecified; research review figure", "axes": {"x": [0, 310], "y": [0, 1.02]},
                "reviewer_risk": "Reduced FCFS results do not bound optimized full-terminal capacity; bands are not confidence intervals.",
                "statistics": "Unweighted seed mean and observed seed min/max; no significance test or inferential claim.",
                "caption": caption, "verification": verified,
                "export_bundle": [p.name for p in exports]}
    (output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9,
                         "axes.titlesize": 10, "legend.fontsize": 8, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8, "xtick.direction": "out", "ytick.direction": "out"})
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.86, wspace=.18, hspace=.32)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=.97)
    fig.text(.5, .925, "Cohort outcomes after drain; separate one-factor controls", ha="center", fontsize=9, color="#505050")
    for i, (ax, (panel_title, names)) in enumerate(zip(axes.flat, PANELS)):
        for reference, linestyle in ((.95, ":"), (.99, "--")):
            ax.axhline(reference, color="#B0B0B0", linewidth=.7, linestyle=linestyle, zorder=0)
        for name in names:
            rows = [row for row in data if row["scenario"] == name]
            if not rows:
                continue
            label, color, marker, linestyle = STYLES[name]
            if i == 3 and name == "baseline6half":
                label = "300 m (baseline)"
            x = [row["offered_rate_per_h"] for row in rows]
            ax.fill_between(x, [row["min"] for row in rows], [row["max"] for row in rows], color=color, alpha=.13, linewidth=0)
            ax.plot(x, [row["mean"] for row in rows], label=label, color=color,
                    marker=marker, linestyle=linestyle, linewidth=1.5, markersize=4,
                    markerfacecolor="white", markeredgewidth=1.)
        ax.set_title(panel_title, loc="left", pad=10)
        ax.text(-.12, 1.055, f"({chr(97+i)})", transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 310)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 60, 120, 180, 240, 300])
        ax.set_yticks([0, .2, .4, .6, .8, 1.])
        ax.grid(axis="y", color="#E5E5E5", linewidth=.5, zorder=0)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, loc="lower left")
        else:
            ax.text(.5, .5, "No scenarios in smoke input", ha="center", transform=ax.transAxes)
    fig.supxlabel("Offered arrival load (jobs/h)", y=.105, fontsize=10)
    fig.supylabel("Normal service fraction", x=.02, fontsize=10)
    fig.text(.5, .035, caption, ha="center", va="center", fontsize=8, linespacing=1.6)
    fig.savefig(svg)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    with (output / "plot_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario", "offered_rate_per_h", "mean", "min", "max", "n"])
        writer.writeheader()
        writer.writerows(data)
    editable_text = "<text" in svg.read_text(encoding="utf-8")
    qa = {"data_integrity": "pass", "rows_verified": verified["row_count"],
          "source_data_export": "plot_data.csv", "statistics": "seed mean/min/max; no significance testing",
          "sample_count": seed_label, "axes_zero_origin": "pass", "editable_svg_text": "pass" if editable_text else "fail",
          "color_and_grayscale": "three muted colors plus distinct markers/line styles; visual review pending",
          "png_dpi": 150, "png_dpi_basis": "explicit task requirement; SVG is vector master",
          "readability_and_core_conclusion": "primary visual review pending", "caption": caption,
          "status": "generated_for_primary_review"}
    (output / "qa_brief.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    if not editable_text:
        raise ValueError("SVG contains no editable text nodes")
    print(json.dumps({"output": str(output.resolve()), "verification": verified, "svg": str(svg), "png": str(png)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true", help="Explicitly permit incomplete preview data and label it SMOKE")
    args = parser.parse_args(argv)
    try:
        plot(args.input, args.output, args.allow_smoke)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
