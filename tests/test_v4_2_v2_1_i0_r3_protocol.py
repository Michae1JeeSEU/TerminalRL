from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "terminal_marl/configs/low_v4_2/i0_r3_repair_protocol.yaml"


def _protocol() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


def test_r3_protocol_freezes_radius_coverage_and_diagnostic_purity() -> None:
    p = _protocol()
    q = p["support_qualification"]
    assert p["protocol"] == "V2_1_I0_R3_REPAIR_PROTOCOL_V2"
    assert q["radius_based_coverage_gate"] == "HARD"
    assert q["knn_purity_role"] == "DIAGNOSTIC_ONLY"
    assert q["error_manifold_hard_gates"]["shared_gate_FN_same_gate_RELEASE"] == {
        "coverage_at_0_25_min": 0.75, "coverage_at_1_00_min": 0.90,
    }
    assert q["error_manifold_hard_gates"]["shared_proposal_error_same_proposal_RELEASE"] == {
        "coverage_at_0_25_min": 0.75, "coverage_at_1_00_min": 0.90,
    }


def test_r3_protocol_freezes_proposal14_traffic_support() -> None:
    minima = _protocol()["proposal14_traffic_minima"]
    assert [minima[key]["independent_teacher_release_lifecycles"] for key in ("N5", "N6", "N12")] == [15, 15, 30]
    assert all(minima[key]["independent_episodes"] == 5 for key in ("N5", "N6", "N12"))
    assert not minima["discard_episode_when_teacher_not_proposal14"]


def test_r3_protocol_keeps_lineage_and_learning_locks() -> None:
    p = _protocol()
    assert p["lineage"]["strategy"] == "fresh_standalone_complete_expert_episodes"
    assert p["lineage"]["training_seed_range"] == [410001, 419999]
    assert p["lineage"]["future_q2_seed_range_reserved"] == [430001, 439999]
    assert p["authorization"] == {
        "I0_R3_PROTOCOL_PREREGISTERED": True,
        "I0_R3_COLLECTION_ALLOWED": True,
        "observation_revision_required": False,
        "V4_2_V2_1_I0_R2": "FAIL",
        "V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED": False,
        "V4_2_V2_1_I2_DAGGER_PREFLIGHT_ALLOWED": False,
        "PPO_ALLOWED": False,
    }
