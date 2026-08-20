from terminal_marl.training.a616_o2m_selection_audit import PPO_AUTHORIZATION, SELECTED


def test_a616_snapshot_set_covers_both_seeds_dynamic_and_static_controls():
    assert {row[0] for row in SELECTED} == {0, 1}
    assert {row[1] for row in SELECTED} >= {"static_outer_plus1", "nominal", "dynamic_t240", "dynamic_t300"}


def test_a616_cannot_authorize_ppo():
    assert PPO_AUTHORIZATION == "STOP"
