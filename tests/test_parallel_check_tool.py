from terminal_marl.training.check_parallel import profile_parallel


def test_parallel_profile_reports_dummy_and_handles_subproc(tmp_path) -> None:
    rows = profile_parallel(
        config_name="dynamic_hrl",
        demand_level="medium",
        n_envs_values=[1, 2],
        vec_env_values=["dummy", "subproc"],
        high_level_steps=1,
        output_dir=tmp_path,
    )
    assert rows
    dummy = [row for row in rows if row["vec_env"] == "dummy"]
    assert {row["n_envs"] for row in dummy} == {1, 2}
    assert all(row["steps_per_sec"] > 0 for row in dummy)
    assert (tmp_path / "parallel_profile.json").is_file()
