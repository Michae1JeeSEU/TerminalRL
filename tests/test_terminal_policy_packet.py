from dataclasses import replace

import numpy as np
import pytest

from terminal_marl.core.terminal_policy_packet import (
    LocalPolicySample,PublishedNeighbor,TerminalPolicyPublication,ObservationUnavailable,
    LOCAL_INDICES,CENTER_INDICES,receive_policy_observation)


def packet():
    ego=np.arange(159,dtype=np.float32)/200.
    local=LocalPolicySample(3,10.,tuple(ego[list(LOCAL_INDICES)]))
    neighbor=PublishedNeighbor(11,'terminal-A',10.,tuple(np.arange(50,dtype=np.float32)/100.))
    report=TerminalPolicyPublication('terminal-A',3,10.,tuple(ego[list(CENTER_INDICES)]),(11,),(neighbor,),True)
    return ego,local,report


def test_current_fields_reconstruct_from_received_sources_without_env():
    ego,local,report=packet()
    obs=receive_policy_observation(local,report,terminal_id='terminal-A',now=10.)
    assert np.array_equal(obs.ego,ego) and obs.traffic.shape==(1,50) and obs.traffic_uids==(11,)
    assert obs.traffic_mask.tolist()==[True]
    with pytest.raises(ObservationUnavailable):receive_policy_observation(object(),report,terminal_id='terminal-A',now=10.)


@pytest.mark.parametrize('change',[
    dict(surveillance_complete=False),dict(expected_neighbor_uids=(11,12)),
    dict(neighbors=()),dict(recipient_uid=9),dict(terminal_id='another-terminal'),
    dict(timestamp=7.),dict(timestamp=11.),dict(expected_neighbor_uids=(11,11)),
])
def test_bad_publication_is_unavailable_not_silently_filtered(change):
    _,local,report=packet()
    with pytest.raises(ObservationUnavailable):
        receive_policy_observation(local,replace(report,**change),terminal_id='terminal-A',now=10.)


def test_neighbor_age_scope_capacity_and_nonfinite_are_fail_closed():
    _,local,report=packet();neighbor=report.neighbors[0]
    for changed in (replace(neighbor,report_timestamp=7.),replace(neighbor,terminal_id='outside'),
                    replace(neighbor,values=(float('nan'),)*50)):
        with pytest.raises(ObservationUnavailable):
            receive_policy_observation(local,replace(report,neighbors=(changed,)),terminal_id='terminal-A',now=10.)
    with pytest.raises(ObservationUnavailable):
        receive_policy_observation(local,replace(report,neighbors=(replace(neighbor,report_timestamp=8.5),)),
            terminal_id='terminal-A',now=11.)
    neighbors=tuple(replace(neighbor,uid=100+i) for i in range(32))
    full=replace(report,expected_neighbor_uids=tuple(n.uid for n in neighbors),neighbors=neighbors)
    assert len(receive_policy_observation(local,full,terminal_id='terminal-A',now=10.).traffic)==32
    extra=replace(neighbor,uid=200)
    with pytest.raises(ObservationUnavailable):
        receive_policy_observation(local,replace(full,expected_neighbor_uids=full.expected_neighbor_uids+(200,),
            neighbors=neighbors+(extra,)),terminal_id='terminal-A',now=10.)
