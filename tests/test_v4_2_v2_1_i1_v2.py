import json
from pathlib import Path

import pytest
import torch
import yaml

from terminal_marl.training.run_v4_2_v2_1_i1_v2 import protocols, fit, reserve_once, validate_r4


def test_new_protocol_keeps_learning_thresholds_and_fresh_cohort():
    p, q = protocols()
    old = yaml.safe_load(Path("terminal_marl/configs/low_v4_2/i1_v2_1_qualification_protocol.yaml").read_text())
    assert p["per_seed_hard_gates"] == old["per_seed_hard_gates"]
    assert p["fresh_actor_seeds"] == [6201, 6202, 6203]
    assert p["training_rows"] == 7576
    assert q["cycles"] * 9 == q["episodes"] == 45
    assert q["seed_start"] == 430001
    assert set(q["smoke_seeds"]).isdisjoint(range(430001, 430046))


def test_qualification_attempt_cannot_be_reused(tmp_path):
    marker = tmp_path / "once.json"
    reserve_once(marker, {"attempt": 1})
    with pytest.raises(FileExistsError):
        reserve_once(marker, {"attempt": 2})
    assert json.loads(marker.read_text()) == {"attempt": 1}


def test_unknown_r4_manifest_cannot_authorize_i1(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"I0_R4": "PASS"}))
    with pytest.raises(ValueError, match="mismatch"):
        validate_r4(path)


def test_bc_kernel_runs_only_on_supplied_synthetic_training_arrays():
    torch.manual_seed(1)
    observations = torch.randn(8, 88)
    gates = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    proposals = torch.tensor([-1, -1, -1, -1, 10, 11, 12, 13])
    joint = torch.tensor([0, 0, 0, 0, 11, 12, 13, 14])
    policy, log = fit(9, (observations, gates, proposals, joint), {"epochs": 1, "batch_size": 4, "learning_rate": .0003})
    assert policy.observation_dim == 88
    assert len(log) == 1 and log[0]["finite_gradients"]
    assert not hasattr(policy, "update")


def test_training_freeze_checkpoint_reload_and_single_evaluation(tmp_path, monkeypatch):
    import terminal_marl.training.run_v4_2_v2_1_i1_v2 as runner
    protocol, cohort = protocols()
    protocol = {**protocol, "epochs": 1, "training_rows": 8}
    monkeypatch.setattr(runner, "protocols", lambda: (protocol, cohort))
    monkeypatch.setattr(runner, "validate_r4", lambda _: {})
    q_root = tmp_path / "q2"
    q_root.mkdir()
    rows = [{"observation_v2_1": [0.] * 88, "teacher_gate": int(i >= 4),
             "teacher_proposal": i + 6 if i >= 4 else None,
             "teacher_joint_index": i + 7 if i >= 4 else 0} for i in range(8)]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    (tmp_path / "training.jsonl").write_text(payload)
    (q_root / "q2.jsonl").write_text(payload)
    runner._write(q_root / "manifest.json", {"Q2_FROZEN": True,
        "freeze_gates": {"synthetic_test": True}, "protocol_sha256": runner.PROTOCOL_SHA,
        "dataset_sha256": runner._sha(q_root / "q2.jsonl"), "rows": 8})
    calls = []
    def evaluate(policy, q_rows, p, finite):
        assert (tmp_path / "out" / "training_freeze.json").exists()
        assert (q_root / "evaluation_consumed.json").exists()
        assert len(q_rows) == 8 and finite
        calls.append(policy)
        return {"PASS": True, **dict.fromkeys(("WAIT_recall", "RELEASE_recall",
            "gate_macro_F1", "proposal_top1", "proposal_top3", "decoded_accuracy"), 1.)}
    monkeypatch.setattr(runner, "_evaluate_seed", evaluate)
    result = runner.train_and_qualify(tmp_path / "out", tmp_path / "manifest.json", q_root)
    assert len(calls) == 3 and result["I1_V2"] == "PASS"
    assert result["Q_EVALUATION_COUNT"] == 1 and not result["PPO_ALLOWED"]
    assert not result["Q_DATASET_ACCESS_DURING_TRAINING"]
    with pytest.raises(FileExistsError):
        runner.train_and_qualify(tmp_path / "second", tmp_path / "manifest.json", q_root)
