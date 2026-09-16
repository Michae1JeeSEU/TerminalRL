import json
from dataclasses import replace

import pytest

from terminal_marl.training.run_v4_2_v2_1_formal_i0 import _sha
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r3 import FixedCollectionSchedule, _collect_partition
from terminal_marl.training.run_v4_2_v2_1_formal_i0_r4 import (
    EXPECTED_PROTOCOL_SHA, PROTOCOL_PATH, collection_strata, historical_registry,
)


def test_r4_freeze_and_real_scenario_schedule():
    assert _sha(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA
    strata = collection_strata()
    assert len(strata) == 26
    assert len({s.name for s in strata}) == 26
    assert sum(s.spec.warm_start_uavs == 5 and not s.spec.dynamic for s in strata) == 4
    assert all(s.spec.phases == ("ring_outer", "ring_middle", "ring_inner") for s in strata)


def test_registry_checks_jsonl_seeds_before_reset(tmp_path):
    (tmp_path / "events.jsonl").write_text(json.dumps({"seed": 510001, "lifecycle_event_id": "prior"}) + "\n")
    identities = historical_registry(tmp_path, tmp_path / "new")
    assert 510001 in identities["seeds"]
    assert "prior" in identities["lifecycle_ids"]


def test_fixed_episode_budget_cannot_silently_run_partial_schedule(tmp_path):
    plan = FixedCollectionSchedule(collection_strata(), 8, 12000, "test")
    with pytest.raises(ValueError, match="exactly one seed"):
        _collect_partition(tmp_path, "training", [1], {}, schedule=plan)
    with pytest.raises(ValueError, match="positive"):
        replace(plan, cycles=0)
