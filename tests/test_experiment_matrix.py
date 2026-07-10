from terminal_marl.experiments.experiment_matrix import (
    export_experiment_matrix,
    recommended_experiment_matrix,
)


def test_experiment_matrix_contains_paper_combinations(tmp_path) -> None:
    rows = recommended_experiment_matrix()
    combos = {(row["config"], row["policy"]) for row in rows}
    assert ("fixed_rule", "rule-only") in combos
    assert ("fixed_residual", "random-policy") in combos
    assert ("dynamic_rule", "dynamic-rule") in combos
    assert ("dynamic_hrl", "high-level-ppo") in combos
    assert ("dynamic_hrl", "high-level-random") in combos
    assert ("dynamic_hrl", "high-level-static") in combos
    assert ("full_hrl_no_safety", "high-level-ppo") in combos
    path = export_experiment_matrix(tmp_path / "matrix.csv")
    assert path.is_file()
    assert "simulation" in path.read_text(encoding="utf-8")

