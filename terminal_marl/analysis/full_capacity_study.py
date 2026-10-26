"""Select an explicit, same-source study grid from immutable campaign outputs.

Campaign failures outside the selected grid remain visible in the report. They
neither fill missing study cells nor invalidate independently verified cells.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
import json
from pathlib import Path

from terminal_marl.analysis.full_capacity_report import (
    KEY_FIELDS, aggregate_rows, analyze_campaign, paired_differences, write_report,
)
from terminal_marl.analysis.full_service_capacity import digest, validate_spec

PROTOCOL_FIELDS = ("warmup_s", "measurement_s", "drain_s", "sla_delay_s",
                   "progress_interval_s", "demand_kind")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _key(params):
    return tuple(params[name] for name in KEY_FIELDS)


def _expected_spec(value):
    if "spec" in value:
        if value.get("purpose") != "analysis_only" or value.get("automatic_launch") is not False:
            raise ValueError("study envelope must explicitly prohibit automatic launch")
        raw = value["spec"]
    else:
        raw = value
    if not all(field in raw for field in ("profiles", "rates_per_hour", "seeds", "methods")):
        raise ValueError("study requires an explicit expected grid")
    spec = validate_spec(raw)
    count = len(set(product(spec["profiles"], spec["rates_per_hour"], spec["seeds"], spec["methods"])))
    if "spec" in value and value.get("expected_jobs") != count:
        raise ValueError("study envelope expected_jobs differs from its explicit grid")
    return spec


def _environment(campaign):
    source, runtime, policy = (campaign[name] for name in ("source", "runtime", "policy"))
    if not source.get("sha256") or source["sha256"] != digest(source["files"]):
        raise ValueError("campaign source hash does not match archived source file hashes")
    # Exact runtime equality includes Python, interpreter path, package versions,
    # platform, numerical-thread and CP-worker policy; no cross-environment pool.
    required = ("python", "interpreter", "platform", "packages", "numerical_threads", "cp_workers")
    if any(name not in runtime for name in required):
        raise ValueError("campaign runtime identity is incomplete")
    return source["sha256"], runtime, policy


def _summary_matches(row, directory):
    # Syncthing preserves the jobs/<name> suffix across Linux/Windows roots.
    parts = str(row.get("directory", "")).replace("\\", "/").rstrip("/").split("/")
    return parts[-2:] == ["jobs", directory.name]


def analyze_study(campaign_dirs, expected_spec):
    """Analyze only explicit expected keys; never infer coverage from inputs.

    Incompatible campaign source/runtime/protocol identities raise ValueError.
    Missing, failed, duplicate or unreadable selected outputs produce an
    incomplete report, whose CLI exit status is nonzero.
    """
    spec = _expected_spec(expected_spec)
    expected = set(product(spec["profiles"], spec["rates_per_hour"], spec["seeds"], spec["methods"]))
    campaigns, records, unreadable = [], defaultdict(list), []
    reference = None
    for path in campaign_dirs:
        path = Path(path)
        raw = _load(path / "campaign.json")
        identity = _environment(raw)
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("campaign source/runtime/policy mismatch; separate versions and environments")
        parent_spec = validate_spec(raw["spec"])
        for field in PROTOCOL_FIELDS:
            if parent_spec[field] != spec[field]:
                raise ValueError("campaign protocol differs from study: " + field)
        for field in ("methods", "rates_per_hour"):
            if not set(parent_spec[field]) <= set(spec[field]):
                raise ValueError("campaign " + field + " incompatible with study")
        parent = analyze_campaign(path)
        summary = _load(path / "summary.json") if (path / "summary.json").exists() else None
        summary_rows = summary.get("jobs", []) if summary else []
        by_directory = {row["row"]["job_directory"]: row for row in parent["jobs"]}
        campaign_record = dict(campaign_directory=str(path.resolve()), spec=parent_spec,
            source=raw["source"], runtime=raw["runtime"], policy=raw["policy"],
            coverage=parent["coverage"], issues=parent["issues"], summary=summary)
        campaigns.append(campaign_record)
        for directory in sorted((path / "jobs").glob("*")):
            if not directory.is_dir():
                continue
            record = dict(campaign_directory=str(path.resolve()), directory=str(directory.resolve()))
            try:
                job_identity = _load(directory / "identity.json")
                key = _key(job_identity["job"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                unreadable.append(dict(record, status="UNREADABLE_IDENTITY", error=str(exc)))
                continue
            record.update(key=list(key), identity=job_identity,
                          summary_rows=[row for row in summary_rows if _summary_matches(row, directory)])
            try:
                result = _load(directory / "results.json") if (directory / "results.json").exists() else None
                record.update(status=result.get("status") if result else "NO_RESULT", result=result)
            except (OSError, ValueError, TypeError) as exc:
                record.update(status="UNREADABLE_RESULT", error=str(exc), result=None)
            record["analysis"] = by_directory.get(str(directory.resolve()))
            errors = []
            if key in expected:
                if any(job_identity.get(name) != raw[name] for name in ("source", "runtime", "policy")):
                    errors.append("job source/runtime/policy differs from parent campaign")
                if any(job_identity["job"].get(field) != spec[field] for field in PROTOCOL_FIELDS):
                    errors.append("job protocol differs from study")
                if record["analysis"] is None:
                    errors.append("no verified completed analysis in parent campaign")
                else:
                    audited = record["analysis"]["row"]
                    for field in ("config_hash_verified", "manifest_hash_verified", "stored_cohort_matches",
                                  "population_conservation", "world_population_conservation"):
                        if audited.get(field) is not True:
                            errors.append("selected analysis audit is not true: " + field)
                    if audited.get("audit_omitted_aircraft") != 0:
                        errors.append("selected analysis has omitted aircraft in mechanism audit")
                if (record.get("result") or {}).get("safety", {}).get("terminated_invalid") is not False:
                    errors.append("selected result safety.terminated_invalid is not false")
                matching = record["summary_rows"]
                if len(matching) != 1 or matching[0].get("status") != "COMPLETED":
                    errors.append("selected job requires exactly one COMPLETED parent summary row")
                if summary is None or summary.get("finished_jobs") != len(summary_rows):
                    errors.append("parent summary finished_jobs/rows inconsistent or absent")
                if summary is not None and summary.get("expected_jobs") != parent["coverage"]["expected_jobs"]:
                    errors.append("parent summary expected_jobs differs from campaign grid")
            record["validation_errors"] = errors
            records[key].append(record)
    if not campaigns:
        raise ValueError("at least one campaign is required")
    analyses, selected, excluded = [], [], []
    issues = dict(missing=[], failed=[], incomplete=[], duplicates=[], unreadable=unreadable)
    for key in sorted(expected):
        entries = records.get(key, [])
        selected.extend(entries)
        if not entries:
            issues["missing"].append(list(key))
        elif len(entries) != 1:
            issues["duplicates"].append(dict(key=list(key), directories=[row["directory"] for row in entries]))
        else:
            entry = entries[0]
            if entry["status"] == "COMPLETED" and not entry["validation_errors"]:
                analyses.append(entry["analysis"])
            else:
                target = "incomplete" if entry["status"] in ("NO_RESULT", "RUNNING", "PENDING") else "failed"
                issues[target].append(dict(key=list(key), directory=entry["directory"], status=entry["status"],
                    validation_errors=entry["validation_errors"], error=entry.get("error"),
                    exception=(entry.get("result") or {}).get("exception")))
    for key in sorted(set(records) - expected):
        excluded.extend(dict(entry, exclusion_reason="not_in_selected_study") for entry in records[key])
    # Analyses already appear once in jobs; provenance records preserve raw
    # identity/result/summary history without repeating the large audit tables.
    for entry in selected + excluded:
        entry.pop("analysis", None)
    rows = [analysis["row"] for analysis in analyses]
    coverage = dict(expected_jobs=len(expected), complete_jobs=len(rows), missing_jobs=len(issues["missing"]),
        failed_jobs=len(issues["failed"]), incomplete_jobs=len(issues["incomplete"]),
        duplicate_keys=len(issues["duplicates"]),
        duplicate_jobs=sum(len(item["directories"]) - 1 for item in issues["duplicates"]),
        unreadable_jobs=len(unreadable), excluded_jobs=len(excluded))
    coverage["all_completed"] = len(rows) == len(expected) and not any(issues.values())
    return dict(schema="full_capacity_study_v1", spec=spec, coverage=coverage,
        source_hash=reference[0], source_campaigns=campaigns, selected_job_records=selected,
        excluded_job_records=excluded, selection_rule="Only explicit expected-spec keys; exactly one verified result and matching completed parent summary row per key. All other keys retain not_in_selected_study history. No cross-version pooling.",
        summary_directory_matching="jobs/<name> suffix, permitting Syncthing Linux/Windows root differences",
        jobs=analyses, per_job=rows, per_profile_rate_method=aggregate_rows(rows), paired=paired_differences(rows),
        issues=issues, interpretation=["Seed mean/min/max are descriptive observations, not confidence intervals.",
            "Excluded parent failures remain in source campaign and excluded-job history.",
            "Selected-study completion does not assert that every source campaign completed.",
            "Finite-cohort and steady offered-rate bounds retain their original distinct denominators."])


__all__ = ["analyze_study", "write_report"]
