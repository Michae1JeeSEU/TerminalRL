from __future__ import annotations

import copy
from pathlib import Path

import pytest

from terminal_marl.training import evaluate_v4_2_p2_v4 as evaluator


def _training_payload() -> dict:
    gates = {
        "safety": True,
        "replay": True,
        "requested_logprob": True,
        "geometry": True,
    }
    records = [
        {
            "seed": 769001 + index,
            "gates": dict(gates),
            "checkpoint": f"/fabricated/checkpoint_{index + 1}.pt",
        }
        for index in range(8)
    ]
    return {
        "stage": "train",
        "PASS": True,
        "protocol_sha256": evaluator.TRAINING_PROTOCOL_SHA,
        "formal": True,
        "records": records,
        "source_hashes": evaluator.p2_source_hashes(),
    }


def _checkpoint_payload() -> dict:
    return {
        "stage": evaluator.CANDIDATE_STAGE,
        "formal": True,
        "protocol_sha256": evaluator.TRAINING_PROTOCOL_SHA,
        "source_hashes": evaluator.p2_source_hashes(),
        "model": {"fabricated_weight": 1},
    }


def _episode_record(
    controller: str,
    seed: int,
    *,
    completed: int,
    requests: int,
    seconds: float,
    wait: float,
    request_to_touchdown: float,
    gates: dict[str, bool] | None = None,
) -> dict:
    return {
        "controller": controller,
        "scenario": "flow6",
        "result": {
            "seed": seed,
            "requests": requests,
            "completed": completed,
            "seconds": seconds,
            "weighted_wait_seconds": wait,
            "request_to_touchdown_seconds": request_to_touchdown,
            "safety": {"collisions": 0, "nmac": 0},
            "shield": 1,
            "shield_actor": 1,
            "actor_decisions": 10,
            "gates": gates or {"complete": True, "safety": True, "replay": True},
        },
    }


def test_protocol_freezes_formal_schedule_and_smoke_boundary():
    protocol = evaluator.protocol()

    assert evaluator._sha(evaluator.PROTOCOL) == evaluator.PROTOCOL_SHA
    assert protocol["formal"]["seeds"] == list(evaluator.FORMAL_SEEDS)
    assert protocol["formal"]["scenarios"] == list(evaluator.FORMAL_SCENARIOS)
    assert protocol["formal"]["scenario_cycle"] == list(evaluator.FORMAL_SCENARIOS) * 2
    assert protocol["formal"]["controllers"] == list(evaluator.CONTROLLERS)
    assert protocol["formal"]["deterministic"] is True
    assert protocol["formal"]["updates"] is False
    assert protocol["formal"]["teacher_enabled"] is False
    assert protocol["formal"]["collect_labels"] is False
    assert set(evaluator.FORMAL_SEEDS).isdisjoint({evaluator.SMOKE_SEED})
    assert protocol["smoke"] == {
        "seed": evaluator.SMOKE_SEED,
        "scenario": evaluator.SMOKE_SCENARIO,
        "steps": evaluator.SMOKE_STEPS,
        "reserves_seeds": False,
    }


def test_validate_training_manifest_payload_accepts_frozen_shape():
    payload = evaluator.validate_training_manifest_payload(_training_payload())

    assert payload["stage"] == "train"
    assert payload["PASS"] is True
    assert len(payload["records"]) == 8
    assert Path(payload["records"][-1]["checkpoint"]).name == "checkpoint_8.pt"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", "entry"),
        ("PASS", False),
        ("protocol_sha256", "wrong-protocol"),
        ("formal", False),
        ("records", []),
        ("source_hashes", {}),
    ],
)
def test_validate_training_manifest_payload_rejects_frozen_metadata(field, value):
    payload = _training_payload()
    payload[field] = value

    with pytest.raises(ValueError):
        evaluator.validate_training_manifest_payload(payload)


def test_validate_training_manifest_payload_rejects_failed_record_and_checkpoint_name():
    failed = _training_payload()
    failed["records"][3]["gates"]["geometry"] = False
    with pytest.raises(ValueError):
        evaluator.validate_training_manifest_payload(failed)

    wrong_checkpoint = _training_payload()
    wrong_checkpoint["records"][-1]["checkpoint"] = "/fabricated/checkpoint_7.pt"
    with pytest.raises(ValueError):
        evaluator.validate_training_manifest_payload(wrong_checkpoint)


def test_validate_checkpoint_payload_accepts_frozen_lineage():
    payload = evaluator.validate_checkpoint_payload(
        _checkpoint_payload(),
        expected_stage=evaluator.CANDIDATE_STAGE,
        expected_formal=True,
        expected_protocol_sha=evaluator.TRAINING_PROTOCOL_SHA,
        expected_source_hashes=evaluator.p2_source_hashes(),
    )

    assert payload["stage"] == evaluator.CANDIDATE_STAGE
    assert payload["model"]


@pytest.mark.parametrize("change", ["stage", "formal", "protocol", "source", "model"])
def test_validate_checkpoint_payload_rejects_lineage_or_model(change):
    payload = _checkpoint_payload()
    if change == "stage":
        payload["stage"] = "V4_2_P2_ENTRY"
    elif change == "formal":
        payload["formal"] = False
    elif change == "protocol":
        payload["protocol_sha256"] = "wrong-protocol"
    elif change == "source":
        payload["source_hashes"] = {}
    else:
        payload["model"] = {}

    with pytest.raises(ValueError):
        evaluator.validate_checkpoint_payload(
            payload,
            expected_stage=evaluator.CANDIDATE_STAGE,
            expected_formal=True,
            expected_protocol_sha=evaluator.TRAINING_PROTOCOL_SHA,
            expected_source_hashes=evaluator.p2_source_hashes(),
        )


def test_aggregate_records_uses_completed_weight_and_reports_safety_shield():
    records = [
        _episode_record("candidate", 755001, completed=2, requests=3, seconds=100.0, wait=10.0, request_to_touchdown=20.0),
        _episode_record("candidate", 755002, completed=1, requests=3, seconds=200.0, wait=20.0, request_to_touchdown=40.0),
    ]

    aggregate = evaluator.aggregate_records(records, "candidate")

    assert aggregate["completed"] == 3
    assert aggregate["requests"] == 6
    assert aggregate["mean_seconds"] == pytest.approx(150.0)
    assert aggregate["weighted_wait_seconds"] == pytest.approx(40.0 / 3.0)
    assert aggregate["request_to_touchdown_seconds"] == pytest.approx(80.0 / 3.0)
    assert aggregate["all_gates"] is True
    assert aggregate["safety"] == {"collisions": 0, "nmac": 0}
    assert aggregate["safety_all_zero"] is True
    assert aggregate["shield"] == 2
    assert aggregate["shield_actor_fraction"] == pytest.approx(2.0 / 20.0)

    failed = copy.deepcopy(records)
    failed[1]["result"]["gates"]["replay"] = False
    assert evaluator.aggregate_records(failed, "candidate")["all_gates"] is False


def test_pair_deltas_are_candidate_minus_baseline_and_rule():
    records = [
        _episode_record("candidate", 755001, completed=3, requests=3, seconds=100.0, wait=10.0, request_to_touchdown=20.0),
        _episode_record("baseline", 755001, completed=3, requests=3, seconds=110.0, wait=12.0, request_to_touchdown=22.0),
        _episode_record("rule", 755001, completed=3, requests=3, seconds=120.0, wait=14.0, request_to_touchdown=24.0),
    ]

    deltas = evaluator.pair_deltas(records)

    assert len(deltas) == 1
    delta = deltas[0]
    assert (delta["seed"], delta["scenario"]) == (755001, "flow6")
    assert delta["candidate_minus_baseline"] == {
        "seconds": -10.0,
        "weighted_wait_seconds": -2.0,
        "request_to_touchdown_seconds": -2.0,
    }
    assert delta["candidate_minus_rule"] == {
        "seconds": -20.0,
        "weighted_wait_seconds": -4.0,
        "request_to_touchdown_seconds": -4.0,
    }
    assert delta["wait_seconds_delta"] == {"baseline": -2.0, "rule": -4.0}


def test_smoke_semantics_has_no_reservation_and_cannot_pass(monkeypatch, tmp_path):
    def fake_policy_loader(path, **kwargs):
        return object(), {"stage": kwargs["expected_stage"]}

    def fake_episode(policy=None, config=None, seed=None, output=None, **kwargs):
        output.mkdir(parents=True, exist_ok=False)
        return {
            "seed": seed,
            "requests": 6,
            "completed": 6,
            "terminal": True,
            "seconds": 10.0,
            "safety": {"collisions": 0},
            "replay_exact": True,
            "geometry": {"samples": 1, "reverse_samples": 0, "max_inner_excursion_m": 0.0},
            "weighted_wait_seconds": 1.0,
            "request_to_touchdown_seconds": 2.0,
            "shield": 0,
            "shield_actor": 0,
            "actor_decisions": 1,
            "student_likelihood_exact": True,
            "gates": {"complete": True, "safety": True, "replay": True},
        }

    def fake_rule(config=None, seed=None, output=None, **kwargs):
        return fake_episode(seed=seed, output=output, **kwargs)

    monkeypatch.setattr(evaluator, "load_evaluation_policy", fake_policy_loader)
    monkeypatch.setattr(evaluator, "evaluate", fake_episode)
    monkeypatch.setattr(evaluator, "evaluate_rule", fake_rule)
    monkeypatch.setattr(evaluator, "p2_config", lambda count: object())
    monkeypatch.setattr(evaluator, "validate_training_manifest", lambda path: pytest.fail("smoke validated training"))
    monkeypatch.setattr(evaluator, "historical_registry", lambda *args: pytest.fail("smoke scanned history"))
    monkeypatch.setattr(evaluator, "reserve_once", lambda *args: pytest.fail("smoke reserved seeds"))

    checkpoint = tmp_path / "checkpoint_8.pt"
    baseline = tmp_path / "baseline.pt"
    result = evaluator.run(
        tmp_path / "smoke",
        checkpoint,
        baseline,
        tmp_path / "history",
        smoke=True,
    )

    assert result["smoke"] is True
    assert result["PASS"] is False
    assert len(result["records"]) == 3
    assert result["smoke_semantics"] == {
        "single_seed": True,
        "flow6": True,
        "steps": True,
        "no_reservation": True,
    }
    assert not (tmp_path / "p2_v4_heldout_consumed.json").exists()
    assert evaluator.smoke_semantics(
        smoke=False,
        seeds=evaluator.FORMAL_SEEDS,
        scenarios=evaluator.FORMAL_SCENARIOS,
        steps=0,
    )["no_reservation"] is False
