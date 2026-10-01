from __future__ import annotations

import json
from pathlib import Path

import pytest

from terminal_marl.training import run_p3_before_prefix_fallback_20261001 as runner


def test_scope_restores_base_module_after_exception():
    base = runner._base
    original = {
        name: getattr(base, name)
        for name in (
            "__file__", "STAGE", "AFTER_CHECKPOINT", "AFTER_CHECKPOINT_SHA256",
            "CANDIDATE_ARM", "REFERENCE_ARM_NAME", "DEPENDENCY_FILES",
            "_candidate_binding", "_historical_checkpoint_binding", "_pair_summary",
            "_business_review", "_build_preflight", "_validate_preflight",
        )
    }
    with pytest.raises(RuntimeError, match="scope fixture"):
        with runner._before_base_scope():
            assert base.STAGE == runner.STAGE
            assert base.AFTER_CHECKPOINT == runner.BEFORE_CHECKPOINT
            assert base.CANDIDATE_ARM == runner.CANDIDATE_ARM
            assert base.REFERENCE_ARM_NAME == runner.REFERENCE_ARM_NAME
            raise RuntimeError("scope fixture")
    for name, value in original.items():
        assert getattr(base, name) is value or getattr(base, name) == value


def test_before_candidate_and_history_bindings_have_distinct_roles(monkeypatch):
    captured = {}

    def fake_policy_binding(path, digest, _config, label):
        captured["candidate"] = (path, digest, label)
        return {"path": str(path), "sha256": digest, "label": label}

    monkeypatch.setattr(runner.support, "_policy_binding", fake_policy_binding)
    candidate = runner._candidate_binding(object())
    assert captured["candidate"][:2] == (runner.BEFORE_CHECKPOINT, runner.BEFORE_CHECKPOINT_SHA256)
    assert captured["candidate"][2] == "before_candidate"
    assert candidate["label"] == "before_candidate"

    monkeypatch.setattr(runner._base, "_resolve_checkpoint", lambda path, digest: path)
    monkeypatch.setattr(runner._base, "_sha256", lambda _path: "bound-sha")
    history = runner._historical_checkpoint_binding()
    assert set(history) == {"after_existing_comparison", "warmup"}
    assert history["after_existing_comparison"]["role"] == "existing_comparison_only"
    assert history["warmup"]["role"] == "historical_input_binding_only"


def test_baseline_guard_reads_native_before_arm_schema(monkeypatch, tmp_path: Path):
    root = tmp_path / "baseline"
    (root / "arms").mkdir(parents=True)
    cells = [dict(cell) for cell in runner.CELL_SPECS]
    metrics = {"censored": False, "complete_scene_metrics_known": True, "normal": 1}
    full = {
        "status": "FULL_OK",
        "pair_count": 4,
        "pair_summaries": [
            {"cell": cell, "arms": {"before_legacy": {"metrics": metrics}}}
            for cell in cells
        ],
    }
    full_path = root / "full.json"
    runner._write_json(full_path, full)
    expected_full_sha = runner._sha256(full_path)
    monkeypatch.setattr(runner, "BASELINE_ROOT", root)
    monkeypatch.setattr(runner, "BASELINE_FULL_SHA256", expected_full_sha)
    monkeypatch.setattr(runner.support, "_record_metrics", lambda _arm: metrics)
    for cell in cells:
        arm = {
            "status": "completed",
            "cell": cell,
            "seed": cell["env_seed"],
            "policy_seed": cell["policy_seed"],
            "model_unchanged": True,
            "caller_rng_unchanged": True,
            "safety_nonzero": False,
            "checkpoint_sha256": runner.BEFORE_CHECKPOINT_SHA256,
            "metrics": metrics,
        }
        (root / "arms" / f"cell_{cell['cell_id']:02d}_before_legacy.json").write_text(
            json.dumps(arm, sort_keys=True) + "\n", encoding="utf-8"
        )
    metadata, arms = runner._baseline_guard()
    assert metadata["full_sha256"] == expected_full_sha
    assert sorted(arms) == [1, 2, 3, 4]
    assert all(arm["seed"] == runner.CELL_SPECS[arm["cell"]["cell_id"] - 1]["env_seed"] for arm in arms.values())
    assert all(arm["policy_seed"] == runner.CELL_SPECS[arm["cell"]["cell_id"] - 1]["policy_seed"] for arm in arms.values())
    assert all("env_seed" not in arm and "keyed_policy_seed" not in arm for arm in arms.values())
    assert all(arm["metrics"]["censored"] is False for arm in arms.values())


def test_pair_serialization_uses_before_comparison_labels(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_BASE_PAIR_SUMMARY",
        lambda *_args: {
            "candidate_minus_after_legacy": {"known": True, "values": {"normal": 1.0}},
            "costs": {"wait_seconds": {"candidate": 1.0, "after_legacy": 2.0, "delta": -1.0}},
        },
    )
    summary = runner._pair_summary({}, {}, {})
    assert "candidate_minus_after_legacy" not in summary
    assert summary["candidate_minus_before_legacy"]["values"]["normal"] == 1.0
    assert summary["costs"]["wait_seconds"]["before_legacy"] == 2.0
    assert "after_legacy" not in summary["costs"]["wait_seconds"]


def test_business_review_renames_reference_aggregate(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_BASE_BUSINESS_REVIEW",
        lambda _rows: {"after_legacy_aggregate": {"normal": 80}, "criterion_pass": False},
    )
    review = runner._business_review([])
    assert review["before_legacy_aggregate"] == {"normal": 80}
    assert "after_legacy_aggregate" not in review


def test_dependency_set_binds_wrapper_base_and_immutable_outputs(tmp_path: Path, monkeypatch):
    base_full = tmp_path / "base_full.json"
    after_full = tmp_path / "after_prefix_full.json"
    base_full.write_text("base\n", encoding="utf-8")
    after_full.write_text("after\n", encoding="utf-8")
    dependencies = dict(runner.DEPENDENCY_FILES)
    dependencies["immutable_base_full"] = base_full
    dependencies["after_prefix_fallback_full"] = after_full
    monkeypatch.setattr(runner, "DEPENDENCY_FILES", dependencies)
    assert runner.DEPENDENCY_FILES["before_prefix_fallback_wrapper"] == runner.WRAPPER_PATH
    assert runner.DEPENDENCY_FILES["prefix_fallback_base_runner"] == runner.BASE_RUNNER_PATH
    assert runner.DEPENDENCY_FILES["immutable_base_full"] == base_full
    assert runner.DEPENDENCY_FILES["after_prefix_fallback_full"] == after_full
    with runner._before_base_scope():
        deps = runner._base._runtime_dependency_shas()
        assert deps["before_prefix_fallback_wrapper"] == runner._sha256(runner.WRAPPER_PATH)
        assert deps["prefix_fallback_base_runner"] == runner._sha256(runner.BASE_RUNNER_PATH)
        assert deps["immutable_base_full"] == runner._sha256(base_full)
        assert deps["after_prefix_fallback_full"] == runner._sha256(after_full)


def test_preflight_payload_adds_auxiliary_after_binding(monkeypatch):
    monkeypatch.setattr(runner, "_BASE_BUILD_PREFLIGHT", lambda: {"stage": runner.STAGE})
    monkeypatch.setattr(
        runner,
        "_auxiliary_after_prefix_binding",
        lambda: {"path": "aux/full.json", "sha256": "aux-sha", "role": "existing_comparison_only", "loaded_for_candidate": False},
    )
    payload = runner._build_preflight()
    assert payload["auxiliary_after_prefix_fallback"]["sha256"] == "aux-sha"


def test_main_preflight_roundtrip_uses_aux_and_source_hash_guards(monkeypatch, tmp_path: Path):
    auxiliary = {
        "path": "aux/full.json",
        "sha256": "aux-sha",
        "role": "existing_comparison_only",
        "loaded_for_candidate": False,
    }
    expected_runner_sha = runner._sha256(runner.WRAPPER_PATH)
    expected_dependencies = {
        "before_prefix_fallback_wrapper": expected_runner_sha,
        "prefix_fallback_base_runner": runner._sha256(runner.BASE_RUNNER_PATH),
    }

    def fake_build_preflight():
        return {
            "stage": runner.STAGE,
            "phase": "preflight",
            "status": "PREFLIGHT_OK",
            "runner_sha256": expected_runner_sha,
            "runtime_dependency_shas": expected_dependencies,
        }

    def fake_validate_preflight(payload):
        runner._base._require(
            payload.get("runner_sha256") == expected_runner_sha,
            "wrapper source changed",
        )
        runner._base._require(
            payload.get("runtime_dependency_shas") == expected_dependencies,
            "base source changed",
        )

    monkeypatch.setattr(runner, "_BASE_BUILD_PREFLIGHT", fake_build_preflight)
    monkeypatch.setattr(runner, "_BASE_VALIDATE_PREFLIGHT", fake_validate_preflight)
    monkeypatch.setattr(runner, "_auxiliary_after_prefix_binding", lambda: dict(auxiliary))
    monkeypatch.setattr(runner._base, "_build_inputs", lambda: ("config", {}))
    monkeypatch.setattr(
        runner._base,
        "_baseline_guard",
        lambda: ({}, {index: {} for index in range(1, 5)}),
    )
    output = tmp_path / "before_prefix"

    assert runner.main(["--phase", "preflight", "--output-dir", str(output)]) == 0
    payload = runner._read_json(output / "preflight.json")
    assert payload["auxiliary_after_prefix_fallback"] == auxiliary
    phase_payload, config, arms = runner._phase_inputs(output)
    assert phase_payload == payload
    assert config == "config"
    assert sorted(arms) == [1, 2, 3, 4]

    missing_aux = dict(payload)
    missing_aux.pop("auxiliary_after_prefix_fallback")
    (output / "preflight.json").write_text(json.dumps(missing_aux), encoding="utf-8")
    with pytest.raises(runner.PrefixFallbackRunnerError, match="auxiliary binding"):
        runner._phase_inputs(output)

    wrong_aux = dict(payload)
    wrong_aux["auxiliary_after_prefix_fallback"] = dict(auxiliary, sha256="wrong")
    (output / "preflight.json").write_text(json.dumps(wrong_aux), encoding="utf-8")
    with pytest.raises(runner.PrefixFallbackRunnerError, match="auxiliary binding"):
        runner._phase_inputs(output)

    wrong_source = dict(payload, runner_sha256="wrong")
    (output / "preflight.json").write_text(json.dumps(wrong_source), encoding="utf-8")
    with pytest.raises(runner.PrefixFallbackRunnerError, match="wrapper source changed"):
        runner._phase_inputs(output)


def test_phase_delegation_restores_scope_and_keeps_grid(monkeypatch, tmp_path: Path):
    seen = {}

    def fake_run_full(output):
        seen["output"] = output
        seen["stage"] = runner._base.STAGE
        seen["candidate"] = runner._base.CANDIDATE_ARM
        return {"stage": runner.STAGE, "pair_count": len(runner.CELL_SPECS), "status": "FULL_OK"}

    monkeypatch.setattr(runner._base, "run_full", fake_run_full)
    result = runner.run_full(tmp_path)
    assert result["pair_count"] == 4
    assert seen == {"output": tmp_path, "stage": runner.STAGE, "candidate": runner.CANDIDATE_ARM}
    assert runner._base.STAGE != runner.STAGE


def test_parser_exposes_only_registered_phases():
    parser = runner.build_parser()
    assert parser.parse_args(["--phase", "preflight"]).phase == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "train"])
