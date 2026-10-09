import pytest
from terminal_marl.configs.p3_v25_course import SCENARIOS,build_config
from terminal_marl.training import p3_v26_supplemental_data as data
from terminal_marl.training import p3_v26_course_data as original


def test_medium_is_explicit_without_changing_original_three_conditions():
    assert SCENARIOS==('flow6_mixed','dense24_fixed','dense24_mixed')
    config=build_config('medium12_mixed')
    assert config.total_requested_uavs==12 and config.warm_start_uavs==6
    assert len(config.fixed_arrival_schedule)==12 and config.fixed_arrival_schedule[:6]==(0.,)*6
    assert config.fixed_arrival_schedule[6:]==(15.,30.,45.,60.,75.,90.)
    assert config.max_simulation_seconds==1800
    assert not config.task_priority_enabled and config.low_level_priority_time_weight==0
    assert config.initial_soc_mode=='uniform' and config.initial_soc_min==.55 and config.initial_soc_max==.9
    assert config.nominal_holder_separation_mode=='swept_curve_v1'


def test_supplement_has_separate_seed_identity_and_uses_same_frozen_resource_mechanics(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(data,'collect_resource_case',lambda output,**kw:calls.append(kw) or kw)
    for index,(seed,scenario) in enumerate(data.TRAIN_CASES,1):
        result=data.collect(tmp_path/str(index),index)
        assert result==dict(index=index,seed=seed,scenario=scenario,smoke=False,protocol_name=data.PROTOCOL)
    smoke=data.collect(tmp_path/'smoke',1,smoke=True)
    assert smoke['seed']==892872 and smoke['scenario']=='medium12_mixed'
    assert not {s for s,_ in data.TRAIN_CASES}&{s for s,_ in original.TRAIN_CASES}
    for index in (0,4,True):
        with pytest.raises(ValueError):data.collect(tmp_path/'bad',index)
    with pytest.raises(ValueError):data.collect(tmp_path/'badsmoke',2,smoke=True)
