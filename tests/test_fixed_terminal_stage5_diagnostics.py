"""Small saved-label fixtures; no simulator, training, or new counterfactuals."""
import json
from pathlib import Path
import pytest

from terminal_marl.scripts import analyze_fixed_terminal_stage5 as diagnostics


def fixture_context():
    roots = [{"root_id": name, "plans": [] if name == "hold" else [{"aircraft_id": name+str(i), "wait_cycles": [0]} for i in range(n)],
              "labels": ["HOLD"] if name == "hold" else []} for name, n in (("hold", 0), ("a", 1), ("b", 2), ("c", 3))]
    values = []
    for normal in (0, 1, 2, 3):
        j = [0, normal, 0, 0, -10., -2., -.1]
        values.append({"J": j, "components": dict(zip(diagnostics.COMPONENTS, (0, normal, 0, 0, 10., 2., .1)))})
    return {"metadata": {"load": "low", "soc": "medium", "seed": 4301, "root_time": 0.}, "path": "fixture/t0",
            "roots": roots, "values": values, "targets": [[0, i, 0, 0, 0., 0., 0.] for i in range(4)],
            "features": [[float(i)]*93 for i in range(4)],
            "pool_metadata": {"beam_order": ["a", "b", "c"], "anchors": ["hold", "c"]}}


def predictions():
    return {"value_topk": [[0, i, 0, 0, 0., 0., 0.] for i in (0, 3, 2, 1)],
            "rank_topk": [[0.], [3.], [2.], [1.]]}


def test_anchors_regret_and_selected_batch_size():
    row = diagnostics.analyze_context(fixture_context(), predictions())
    before = row["methods"]["rank_topk"]["topk"]["1"]["before_anchors"]
    after = row["methods"]["rank_topk"]["topk"]["1"]["after_anchors"]
    assert not before["best_key_hit"] and before["batch_size"] == 1
    assert before["raw_J_regret_full_minus_chosen"] == [0, 2, 0, 0, 0., 0., 0.]
    assert before["raw_component_regret_full_minus_chosen"]["normal_landed"] == 2
    assert after["best_key_hit"] and after["batch_size"] == 3
    assert after["selected_count"] == 3 and set(after["root_ids"]) == {"hold", "a", "c"}
    assert row["methods"]["beam_topk"]["original_order"] == ["a", "b", "c"]
    assert row["methods"]["beam_topk"]["complete_ranking"] == ["a", "b", "c", "hold"]
    assert row["methods"]["beam_topk"]["topk"]["all"]["before_anchors"]["selected_count"] == 4
    summary = diagnostics.summarize_contexts([row])
    assert summary["groups"]["rank_topk"]["1"]["after_anchors"]["batch_size"]["3"]["contexts"] == 1


def test_key_rounding_uses_action_tie_and_preserves_raw_regret():
    context = fixture_context()
    context["values"][2]["J"] = [0, 3, 0, 0, -10.+1e-12, -2., -.1]
    chosen = diagnostics.selected_value(context, ["b", "c"])
    assert chosen["chosen_root_id"] == "b" and chosen["best_key_hit"]
    # Full best also follows action_key, not the insignificant float difference.
    assert chosen["full_best_root_id"] == "b"


def test_error_distributions_hold_and_rounding():
    context = fixture_context()
    prediction = [r[:] for r in context["targets"]]
    prediction[0][0] = .5
    prediction[1][1] = 1.6
    report = diagnostics.value_diagnostics([context], [prediction])
    assert report["all_candidates"]["MAE"][0] == .125
    assert report["rounded_first_four"]["MAE"][0] == 0
    assert report["rounded_first_four"]["MAE"][1] == .25
    assert report["HOLD"]["MAE"][0] == .5
    assert report["true_integer_distributions"]["delta_J"]["1"] == {"0": 1, "1": 1, "2": 1, "3": 1}


def test_exact_feature_equality_not_approximate():
    context = fixture_context()
    context["features"][1] = context["features"][0][:]
    context["features"][2] = context["features"][0][:]
    context["features"][2][0] = 1e-15
    result = diagnostics.identical_feature_labels([context])
    assert result["duplicate_groups"] == 1
    assert result["different_labels"]["delta_J"]["unequal_label_pairs"] == 1
    assert result["unique_feature_vectors"] == 3


def test_plan_key_order_independent_and_wait_sensitive():
    plans = [{"aircraft_id": "b", "wait_cycles": [0, 1]}, {"aircraft_id": "a", "wait_cycles": [2]}]
    assert diagnostics.plan_key(plans) == diagnostics.plan_key(plans[::-1])
    assert diagnostics.plan_key(plans) != diagnostics.plan_key([{**plans[0], "wait_cycles": [1, 0]}, plans[1]])


def test_cli_finite_serialization_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "analyze", lambda root: {"contexts": [diagnostics.analyze_context(fixture_context(), predictions())], "episodes": {"episodes": 0}})
    path = tmp_path/"result.json"
    assert diagnostics.main(["--stage4-root", str(tmp_path), "--output", str(path)]) == 0
    assert len(json.loads(path.read_text())["contexts"]) == 1
    with pytest.raises(FileExistsError):
        diagnostics.main(["--stage4-root", str(tmp_path), "--output", str(path)])


def test_saved_offline_models_files_schema(tmp_path, monkeypatch):
    from hashlib import sha256
    context = fixture_context()
    (tmp_path/"models").mkdir()
    (tmp_path/"summary").mkdir()
    files = {}
    for kind in ("value", "rank"):
        path = tmp_path/"models"/f"{kind}.pt"
        path.write_bytes(kind.encode())
        files[path.name] = sha256(path.read_bytes()).hexdigest()
    analyzed = diagnostics.analyze_context(context, predictions())
    old = {}
    for method in diagnostics.METHODS:
        row = analyzed["methods"][method]
        chosen = row["topk"]["3"]["after_anchors"]
        old[method] = {"contexts": [{"context": "low-medium-4301/t0", "filtered_chosen": chosen["chosen_root_id"],
                                     "pair_correct": row["pair_correct"], "raw_J_regret_full_minus_filtered": chosen["raw_J_regret_full_minus_chosen"]}]}
    (tmp_path/"summary"/"offline_test.json").write_text(json.dumps({"contexts": 1, "models": {"path": "models", "files": files}, "methods": old}))
    monkeypatch.setattr(diagnostics, "load_contexts", lambda *args: [context])
    class FrozenFixture:
        @staticmethod
        def load(path):
            class Score:
                def predict(self, features):
                    return predictions()[path.stem+"_topk"]
            return Score()
    monkeypatch.setattr(diagnostics, "Scorer", FrozenFixture)
    monkeypatch.setattr(diagnostics, "episode_diagnostics", lambda root: {"episodes": 0})
    result = diagnostics.analyze(tmp_path)
    assert result["models_sha256"] == files
    assert result["saved_offline_top3_reproduction"] is True
