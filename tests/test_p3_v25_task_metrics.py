from types import SimpleNamespace as NS

import pytest

from terminal_marl.training.p3_v25_task_metrics import completed_metrics,reward_parts


def item(uid,priority,initial,wait,outcome):
    return NS(uid=uid,arrival_id=uid,base_priority=priority,mission_type=NS(value='standard'),
        energy_capacity=1000.,energy_remaining=(initial-.2)*1000,energy_consumed=200.,soc=initial-.2,
        terminal_outcome=NS(value=outcome),emergency_landing=outcome=='emergency_landed',
        request_time=10.,touchdown_time=110.,completion_time=140.,
        external_wait_time=wait/4,internal_wait_time=wait*3/4)


def test_priority_weighted_and_initial_soc_cohorts_keep_emergencies():
    env=NS(completed_uavs=[item(2,5,.85,40,'emergency_landed'),item(1,1,.60,10,'completed')])
    report=completed_metrics(env)
    assert report['completed_count']==2 and report['normal_count']==report['emergency_count']==1
    assert report['priority_weighted_wait_seconds']==pytest.approx(35.)
    assert report['low_soc_count']==1 and report['low_soc_wait_mean']==10.
    assert report['priority_weighted_request_to_touchdown_seconds']==100.
    assert report['priority_weighted_request_to_completion_seconds']==130.
    assert report['low_soc_request_to_touchdown_mean']==100.
    assert report['initial_soc_min']==pytest.approx(.6)
    assert report['completed_rows'][0]['arrival_id']==1
    assert report['completed_rows'][1]['request_to_touchdown_seconds']==100.
    assert report['lifecycle_distributions']['wait_seconds']==dict(count=2,mean=25.,p95=38.5,max=40.)
    assert report['lifecycle_distributions']['request_to_touchdown_seconds']==dict(count=2,mean=100.,p95=100.,max=100.)
    empty=completed_metrics(NS(completed_uavs=[]))
    assert empty['completed_count']==0 and empty['priority_weighted_wait_seconds'] is None
    assert empty['priority_weighted_request_to_touchdown_seconds'] is None
    assert empty['lifecycle_distributions']['wait_seconds']==dict(count=0,mean=None,p95=None,max=None)


def test_reward_parts_exclude_per_actor_total_and_include_queue():
    info=dict(reward_components={'1':dict(time=-.01,urgency=-.02,total=-.03),
                                 '2':dict(touchdown=25.,total=25.)},
              team_reward_components=dict(external_queue=-.1),team_reward=24.87)
    parts=reward_parts(info)
    assert sum(parts.values())==pytest.approx(24.87) and 'total' not in parts
    with pytest.raises(ValueError):reward_parts(dict(info,team_reward=24.9))
