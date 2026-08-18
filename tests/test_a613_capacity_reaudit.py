from terminal_marl.core.config import RingRole
from terminal_marl.training.a611_causal_audit import DemandProfile
from terminal_marl.training.a613_capacity_reaudit import repair_config


def test_a613_keeps_repair_scheduler_enabled_and_static_capacity_is_derived():
    profile = DemandProfile("test", ((0., 10., .1),), 10., 1.)
    nominal = repair_config(profile, dynamic=False)
    assert nominal.repair_aware_scheduler
    for role in RingRole:
        static = repair_config(profile, dynamic=False, role=role)
        assert static.repair_aware_scheduler
        assert static.rings[role].radius > nominal.rings[role].radius
