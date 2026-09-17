from terminal_marl.training.run_v4_2_curriculum_pilot import config_for
import pytest

@pytest.mark.parametrize('n,horizon', [(6,1210.),(8,1330.),(12,1570.)])
def test_p2_changes_request_count_without_relaxing_safety(n,horizon):
    c=config_for(n)
    assert c.total_requested_uavs==n and len(c.fixed_arrival_schedule)==n
    assert c.fixed_arrival_schedule[0]==10. and c.fixed_arrival_schedule[-1]==10+60*(n-1)
    assert c.max_simulation_seconds==horizon
    assert c.reset_mode=='flow_from_empty' and c.warm_start_uavs==0
    assert c.safe_distance==80 and c.shield_enabled
    assert not c.teacher_enabled and not c.high_level_enabled
    assert c.initial_soc==.8 and c.initial_priority==1
