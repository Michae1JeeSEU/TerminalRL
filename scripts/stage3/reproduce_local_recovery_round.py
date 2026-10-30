"""Read an archived checkpoint and reproduce one bounded normal callback.

This never restores/open-writes an archived state sink and never advances the
simulation or publishes an action. A snapshot hash mismatch is reported as a
nearby physical-state reproduction, not an exact historical dispatch replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from terminal_marl.core.rotating_slots.stage2_campaign import Stage2Codec,digest
from terminal_marl.core.rotating_slots.stage2_planning import NormalSchedulerCallback
from terminal_marl.core.rotating_slots.planning_latency import PlanningTiming,project_visible_snapshot,_FuturePort
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld,SchedulerPort


def reproduce(checkpoint,*,expected_source_id=None,inspect_only=False):
    checkpoint=Path(checkpoint)
    raw=checkpoint.read_bytes();envelope=json.loads(raw)
    payload=envelope['payload']
    if envelope['sha256']!=digest(payload): raise ValueError('checkpoint content hash mismatch')
    if payload['schema']!='ft_rs_stage2_campaign_checkpoint_v1': raise ValueError('unsupported checkpoint schema')
    codec=Stage2Codec()
    # Decode only: restore_checkpoint can reopen/truncate the archived sink.
    world=object.__new__(RotatingSlotWorld)
    world.__dict__.update(codec.decode(payload['world']))
    world.state_sink=None;world.decision_callback=None;world.admission_callback=None
    world._scheduler_port=SchedulerPort(world)
    source=world.build_planning_snapshot()
    saved=codec.decode(payload['scheduler']);state=saved['callback']
    timing=PlanningTiming.from_lead(source.time,saved['lead_seconds'],saved['preparation_margin'])
    projected=project_visible_snapshot(source,timing.effective_time,timing=timing)
    binding=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        checkpoint_payload_sha256=envelope['sha256'],checkpoint_time=payload['time'],
        source_snapshot_id=source.version,source_time=source.time,effective_snapshot_id=projected.version,
        effective_time=projected.time,expected_source_snapshot_id=expected_source_id,
        source_binding_exact=expected_source_id is not None and source.version==expected_source_id,
        reproduction_kind=('exact_source_snapshot' if expected_source_id is not None and source.version==expected_source_id
            else 'nearby_physical_state'),archived_sink_opened=False,simulation_advanced=False,
        action_published=False,checkpoint_identity=payload['identity'])
    if inspect_only: return dict(schema='local_recovery_round_reproduction_v1',binding=binding,inspect_only=True)
    selective=state.get('selective_replanning',{})
    callback=NormalSchedulerCallback(state['method'],budget_profile=state['budget_profile'],
        planning_profile=state['planning_profile'],evaluation_profile=state['evaluation_profile'],
        online_policy=state.get('online_policy'),prefix_reuse=state.get('prefix_reuse'),
        selective_method=selective.get('method'),selective_rule_variant=selective.get('rule_variant','baseline'),
        opportunity_seed_offset=selective.get('opportunity_seed_offset',0),search_policy=selective.get('search_policy'))
    callback.load_state_dict(state)
    if callback.selective_method is not None:
        events=world._scheduler_port.normal_maneuver_start_events(callback.selective_history.event_cursor)
        callback.observe_execution_snapshot(source,source.time,events=events)
    started=time.perf_counter()
    ready=started+(timing.ready_deadline-timing.snapshot_time)-max(0.,saved.get('service_until',source.time)-source.time)
    reserve=saved['search_handoff_reserve_seconds']
    future=_FuturePort(projected,source,ready,search_wall_deadline=ready-reserve if reserve else None,
        planning_wall_origin=started,search_handoff_reserve_seconds=reserve)
    callback(future)
    from terminal_marl.core.rotating_slots.replay import _plain
    return _plain(dict(schema='local_recovery_round_reproduction_v1',binding=binding,
        callback_wall_s=time.perf_counter()-started,callback=callback.records[-1],
        deferred_proposal_hashes=[p.trajectory_hash for p in future.proposal or ()],
        search_handoff=future.search_handoff))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--expected-source-id')
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError('fresh diagnostic output required: '+str(output))
    result=reproduce(args.checkpoint,expected_source_id=args.expected_source_id,inspect_only=args.inspect_only)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output.resolve()),binding=result['binding']),ensure_ascii=False))


if __name__=='__main__': main()
