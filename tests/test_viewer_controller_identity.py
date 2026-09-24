from terminal_marl.visualization.rollout_viewer import _controller_identity


def test_controller_identity_formats_all_rollout_modes():
    cases = {
        "fixed_rule": {
            "low_level_controller=rule",
            "low_checkpoint=none",
            "high_checkpoint=none",
            "no_trained_checkpoint=true",
        },
        "fixed_low_rl": {"low_level_controller=learned", "low_checkpoint=low.pt"},
        "dynamic_rule": {"low_level_controller=rule", "no_trained_checkpoint=true"},
        "learned_high_rule_low": {"low_level_controller=rule", "high_checkpoint=high.pt"},
        "learned_high_frozen_low": {
            "low_level_controller=learned",
            "low_checkpoint=low.pt",
            "high_checkpoint=high.pt",
        },
    }

    for mode, expected_fields in cases.items():
        line = _controller_identity(
            mode,
            low_checkpoint="low.pt",
            high_checkpoint="high.pt",
        )
        assert f"selected_mode={mode}" in line
        for field in expected_fields:
            assert field in line


def test_rule_low_level_modes_do_not_claim_learned_low_level():
    for mode in ("fixed_rule", "dynamic_rule", "learned_high_rule_low"):
        line = _controller_identity(mode, high_checkpoint="high.pt")
        assert "low_level_controller=rule" in line
        assert "low_level_controller=learned" not in line


def test_checkpoint_paths_are_only_shown_when_used():
    fixed_low = _controller_identity("fixed_low_rl", low_checkpoint="low.pt")
    learned_high_rule_low = _controller_identity(
        "learned_high_rule_low", high_checkpoint="high.pt"
    )

    assert "low_checkpoint=low.pt" in fixed_low
    assert "high_checkpoint=" not in fixed_low
    assert "high_checkpoint=high.pt" in learned_high_rule_low
    assert "low_checkpoint=" not in learned_high_rule_low
