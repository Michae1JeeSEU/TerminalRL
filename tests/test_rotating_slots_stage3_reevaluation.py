"""Synthetic, bounded reevaluation preparation; never runs online evaluation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stage3 import reevaluate as ree
from terminal_marl.core.rotating_slots.stage2_campaign import atomic_json, digest, load_json
from terminal_marl.core.rotating_slots.stage3_learning import campaign, data, model, evaluation


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    physical_source = repo / "physical.py"
    physical_source.write_text("physical unchanged\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "ROOT", repo)
    monkeypatch.setattr(campaign, "source_files", lambda: [physical_source])
    hashes = {"physical.py": ree.file_hash(physical_source)}
    frozen = dict(source_files=hashes, source_hash=digest(hashes), source_commit="actual-new-commit",
                  workspace_diff_hash="clean", workspace_status="", branch="test")
    freeze_dir = tmp_path / "freeze"
    atomic_json(freeze_dir / "source_freeze.json", frozen)
    parent = tmp_path / "parent"
    cfg = campaign.contract_config()
    from terminal_marl.core.rotating_slots.stage2_planning import PlanningBudget, HORIZON
    from terminal_marl.core.rotating_slots.config import load_service_profile
    config = load_service_profile("service_half_v1")
    rows = []
    for i in range(1, 25):
        rows.append(dict(root_id=f"s3r{i:02d}", ancestry_root_id=f"s3r{i:02d}",
                         manifest_lineage_id=f"manifest:{i}", checkpoint_parent_id=None,
                         split="train" if i <= 16 else "validation" if i <= 20 else "test",
                         seed=7301 + i, scenario_profile="medium24_rate150_v1" if i == 23 else "diagnostic18_rate120_v1"))
    split = dict(roots=deepcopy(rows))
    split["split_manifest_id"] = digest(split)
    atomic_json(parent / "split_manifest.json", split)
    fp = dict(source_commit="old-training-commit", split_manifest_id=split["split_manifest_id"],
              evaluation_protocol_id=digest(cfg["evaluation"]), physical_config_hash=config.config_hash)
    versions = {name: "observed-version" for name in ("numpy", "scipy", "PyYAML", "ortools", "torch", "pytest")}
    old = dict(fingerprint=fp, evaluation_cpu=12, environment=dict(versions=versions),
               source_files=hashes, source_hash="original-source", resolved_budgets=cfg["compute"],
               planning_budget=PlanningBudget.for_profile("quality_v5").settings(), horizon_s=HORIZON,
               inputs=dict(physical_config_sha256=config.raw_hash),
               contract_hash=ree.file_hash(campaign.PACKAGE / cfg["contract_file"]))
    atomic_json(parent / "run_binding.generated.json", old)
    atomic_json(parent / "final_status.json", dict(evaluation="COMPLETE"))
    for name in ("heldout_signal.json", "same_state_comparison.json", "closed_loop_comparison.json", "heldout_access.json", "source_freeze.json"):
        atomic_json(parent / name, dict(prior=name))
    (parent / "model").mkdir()
    (parent / "model/model.pt").write_bytes(b"immutable synthetic model")
    atomic_json(parent / "model/feature_normalizer.json", dict(immutable="normalizer"))
    monkeypatch.setattr(ree, "MODEL_HASH", ree.file_hash(parent / "model/model.pt"))
    monkeypatch.setattr(ree, "NORMALIZER_HASH", ree.file_hash(parent / "model/feature_normalizer.json"))
    atomic_json(parent / "model/model_card.json", dict(model_frozen=True, model_hash=ree.MODEL_HASH, fingerprint=fp))
    worlds = {}
    for row in rows:
        directory = parent / "roots" / row["root_id"]
        initial = dict(aircraft=[], slots=dict(root=row["root_id"]), pads={})
        manifest = dict(seed=row["seed"])
        identity = dict(source_hash="original-checkpoint-source", manifest_sha256=digest(manifest))
        row.update(checkpoint_path=str(directory / "initial.checkpoint.json"), raw_snapshot_path=str(directory / "initial_state.json"),
                   initial_state_hash=digest(initial), manifest_hash=digest(manifest), identity=identity)
        payload = dict(time=0., identity=identity, sink=None)
        atomic_json(directory / "initial.checkpoint.json", dict(payload=payload, sha256=digest(payload)))
        atomic_json(directory / "initial_state.json", initial)
        atomic_json(directory / "manifest.json", manifest)
        atomic_json(directory / "root.json", row)
        worlds[row["root_id"]] = SimpleNamespace(aircraft={}, slots=SimpleNamespace(to_dict=lambda initial=initial: initial["slots"]),
                                                 pads=SimpleNamespace(to_dict=lambda: {}))
    atomic_json(parent / "raw_snapshot_index.json", dict(complete=True, roots=rows))
    calls = []
    class Advisor:
        def __init__(self, path, *, fingerprint, evaluation):
            calls.append(dict(path=str(path), fingerprint=fingerprint, evaluation=evaluation))
        def _load(self):
            pass
    monkeypatch.setattr(model, "LearnedAdvisor", Advisor)
    monkeypatch.setattr(data, "restore_root", lambda row, scheduler: worlds[row["root_id"]])
    monkeypatch.setattr(ree, "probe_environment", lambda old: dict(free_physical_cores=[12], versions=versions))
    return SimpleNamespace(parent=parent, output=tmp_path / "new", freeze=freeze_dir, frozen=frozen, old=old,
                           calls=calls, physical_source=physical_source)


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): ree.file_hash(p) for p in path.rglob("*") if p.is_file()}


def test_prepare_separate_exact_inputs_honest_provenance_and_no_parent_writes(prepared_inputs):
    f = prepared_inputs
    before = tree_hashes(f.parent)
    binding = ree.prepare(f.parent, f.output, f.freeze)
    assert tree_hashes(f.parent) == before
    assert binding["fingerprint"] == binding["frozen_model_contract"] == f.old["fingerprint"]
    assert binding["execution_provenance"]["source_commit"] == "actual-new-commit"
    assert binding["fingerprint"]["source_commit"] == "old-training-commit"
    assert binding["source_hash"] == f.frozen["source_hash"] != f.old["source_hash"]
    assert f.calls == [dict(path=str(f.parent / "model"), fingerprint=f.old["fingerprint"], evaluation=True)]
    for name in ("model/model.pt", "model/feature_normalizer.json", "model/model_card.json", "split_manifest.json", "heldout_signal.json"):
        assert (f.output / name).read_bytes() == (f.parent / name).read_bytes()
    original = load_json(f.parent / "raw_snapshot_index.json")["roots"]
    copied = load_json(f.output / "raw_snapshot_index.json")["roots"]
    assert [r["root_id"] for r in copied] == [r["root_id"] for r in original]
    for old, new in zip(original, copied):
        assert {k: v for k, v in new.items() if k not in ("checkpoint_path", "raw_snapshot_path")} == {k: v for k, v in old.items() if k not in ("checkpoint_path", "raw_snapshot_path")}
        assert str(f.output) in new["checkpoint_path"]
    assert set(p.name for p in (f.output / "roots").iterdir()) == set(ree.CASE_IDS)
    assert not (f.output / "raw").exists() and not (f.output / "labels").exists()
    for rid in ree.CASE_IDS:
        assert (f.output / "roots" / rid / "initial.checkpoint.json").read_bytes() == (f.parent / "roots" / rid / "initial.checkpoint.json").read_bytes()
    protocol, _, ledger = ree.verify_prepared(f.output)
    assert protocol["heldout_signal_policy"] == "BYTE_IDENTICAL_PRIOR_EVIDENCE_NO_NEW_MEASUREMENT"
    assert ledger.data["paused_at"] is not None
    assert ledger.remaining("evaluation") == 3600. and ledger.data["phase_used"] == {}


@pytest.mark.parametrize("mutation,match", [("model", "model/normalizer"), ("source", "frozen source"), ("split", "split identity"), ("checkpoint", "checkpoint identity"), ("initial", "initial state"), ("caseorder", "six-case order")])
def test_prepare_rejects_invalid_frozen_inputs_before_output(prepared_inputs, mutation, match):
    f = prepared_inputs
    if mutation == "model":
        (f.parent / "model/model.pt").write_bytes(b"changed")
    elif mutation == "source":
        f.physical_source.write_text("changed", encoding="utf-8")
    elif mutation == "split":
        path = f.parent / "split_manifest.json"
        value = load_json(path); value["roots"][0]["seed"] += 1; atomic_json(path, value)
    elif mutation == "checkpoint":
        path = f.parent / "roots/s3r21/initial.checkpoint.json"
        value = load_json(path); value["payload"]["time"] = 1.; atomic_json(path, value)
    elif mutation == "initial":
        atomic_json(f.parent / "roots/s3r21/initial_state.json", {})
    else:
        path = f.parent / "raw_snapshot_index.json"
        value = load_json(path); value["roots"][-1], value["roots"][-2] = value["roots"][-2], value["roots"][-1]; atomic_json(path, value)
    with pytest.raises(ValueError, match=match):
        ree.prepare(f.parent, f.output, f.freeze)
    assert not f.output.exists()


def test_prepare_refuses_same_nested_nonempty_and_repeated_output(prepared_inputs):
    f = prepared_inputs
    for output in (f.parent, f.parent / "new", f.parent.parent):
        with pytest.raises(ValueError, match="separate"):
            ree.prepare(f.parent, output, f.freeze)
    ree.prepare(f.parent, f.output, f.freeze)
    before = tree_hashes(f.output)
    with pytest.raises(ValueError, match="nonempty"):
        ree.prepare(f.parent, f.output, f.freeze)
    assert tree_hashes(f.output) == before


def test_unknown_physical_drift_rejected_even_with_valid_new_freeze(prepared_inputs):
    f = prepared_inputs
    f.physical_source.write_text("new physical semantics", encoding="utf-8")
    value = deepcopy(f.frozen)
    value["source_files"]["physical.py"] = ree.file_hash(f.physical_source)
    value["source_hash"] = digest(value["source_files"])
    atomic_json(f.freeze / "source_freeze.json", value)
    with pytest.raises(ValueError, match="physical/protocol/config source drift"):
        ree.prepare(f.parent, f.output, f.freeze)


@pytest.mark.parametrize("relative", ["model/model.pt", "heldout_signal.json", "roots/s3r21/initial.checkpoint.json", "roots/s3r21/root.json", "raw_snapshot_index.json"])
def test_prepared_inputs_cannot_change(prepared_inputs, relative):
    f = prepared_inputs
    ree.prepare(f.parent, f.output, f.freeze)
    (f.output / relative).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        ree.verify_prepared(f.output)


def test_ledger_cap_or_missing_ledger_cannot_reset(prepared_inputs):
    f = prepared_inputs
    ree.prepare(f.parent, f.output, f.freeze)
    ledger_path = f.output / "budget_ledger.json"
    original = load_json(ledger_path)
    changed = deepcopy(original); changed["phase_caps"]["evaluation"] = 7200.; atomic_json(ledger_path, changed)
    with pytest.raises(ValueError, match="caps changed"):
        ree.verify_prepared(f.output)
    ledger_path.unlink()
    with pytest.raises(ValueError, match="existing ledger required"):
        ree.verify_prepared(f.output)
    assert not ledger_path.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_evaluate_single_attempt_unchanged_runner_and_always_pauses(prepared_inputs, monkeypatch, fail):
    f = prepared_inputs
    ree.prepare(f.parent, f.output, f.freeze)
    atomic_json(f.output / "targeted_checks.json", dict(all_automated_pass=True))
    atomic_json(f.output / "reevaluation_checks.json", dict(passed=True))
    calls = []
    def evaluate(output, ledger):
        calls.append(ledger.remaining("evaluation"))
        assert ledger.data["active_phase"]["phase"] == "evaluation"
        if fail:
            raise RuntimeError("observed synthetic failure")
    monkeypatch.setattr(evaluation, "run_evaluation", evaluate)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            ree.run_stage(f.output, "evaluate")
    else:
        ree.run_stage(f.output, "evaluate")
    ledger = campaign.CampaignLedger(f.output / "budget_ledger.json")
    assert ledger.data["paused_at"] is not None and ledger.data["active_phase"] is None
    assert 3599. < calls[0] <= 3600.
    assert ledger.data["phase_caps"]["evaluation"] == 3600.
    with pytest.raises(ValueError, match="already attempted"):
        ree.run_stage(f.output, "evaluate")
    assert len(calls) == 1


def test_prior_heldout_signal_reused_without_model_or_label_access(prepared_inputs, monkeypatch):
    f = prepared_inputs
    binding = ree.prepare(f.parent, f.output, f.freeze)
    evaluation.open_test_gate(f.output, load_json(f.output / "model/model_card.json"))
    def forbidden(*args, **kwargs):
        raise AssertionError("new heldout measurement attempted")
    monkeypatch.setattr(model, "LearnedAdvisor", forbidden)
    assert evaluation.measure_heldout_signal(f.output, binding) == load_json(f.parent / "heldout_signal.json")


def test_freeze_rejects_uncommitted_sources_and_does_not_seed_scan(tmp_path, monkeypatch):
    root = tmp_path / "repo"; root.mkdir()
    source = root / "source.py"; source.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "source_files", lambda: [source])
    def forbidden(*args, **kwargs):
        raise AssertionError("seed scan invoked")
    monkeypatch.setattr(campaign, "freeze_local", forbidden)
    def git(command, **kwargs):
        action = command[1]
        return {"diff": b"", "ls-files": b"source.py\n", "rev-parse": b"new-commit\n", "status": b" M unrelated.md\n", "branch": b"branch\n"}[action]
    monkeypatch.setattr(ree.subprocess, "check_output", git)
    value = ree.freeze(root / "artifacts/freeze")
    assert value["source_commit"] == "new-commit" and value["source_files"] == {"source.py": ree.file_hash(source)}
    assert not (root / "artifacts/freeze/split_manifest.json").exists()
    monkeypatch.setattr(ree.subprocess, "check_output", lambda *args, **kwargs: b"dirty")
    with pytest.raises(ValueError, match="commit all"):
        ree.freeze(root / "artifacts/other")
