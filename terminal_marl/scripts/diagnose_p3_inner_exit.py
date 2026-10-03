"""Read trusted P3 replay snapshots and measure unchanged exit candidates.

This is a static diagnostic: no environment step, training or safety override.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace
import json
from pathlib import Path
import pickle
import time

import numpy as np

from terminal_marl.core.config import RingRole
from terminal_marl.core.merge_scheduler import LowLevelIntent, RepairCandidate
from terminal_marl.core.safety_filter import SafetyFilter
from terminal_marl.envs.parallel_env import WorldSnapshot, TerminalParallelEnv
from terminal_marl.configs.p3_integrated_repair import build_config
from terminal_marl.core.trajectory_generator import TrajectoryPoint, TrajectoryPlan
from terminal_marl.scripts.diagnose_p3_integrated_repair import write_json


def steady_holder_positions(uav, ring, times, now):
    """Independent analytic reference, restricted to steady nominal holders."""
    angle = float(uav.ring_progress)
    theta = angle + float(uav.speed)/float(ring.radius)*(times-float(now))
    return np.column_stack((ring.radius*np.cos(theta), ring.radius*np.sin(theta),
                            np.full_like(theta, ring.height)))


def measure_candidate(env, uav, plan):
    snapshot = WorldSnapshot(env.now, env.active, env.airspace)
    safety = SafetyFilter(env.config)
    full = safety.filter_plan(plan, {}, snapshot)
    solo = SafetyFilter(env.config).filter_plan(
        plan, {}, WorldSnapshot(env.now, {uav.uid:uav}, env.airspace))
    times = np.arange(plan.start_time, plan.end_time, env.config.physics_dt)
    times = np.append(times, plan.end_time)
    positions = np.asarray(plan.sample_positions(times))
    pairs = []
    departure = float(plan.metadata['scheduled_departure'])
    for other in env.active.values():
        if other.uid == uav.uid:
            continue
        ring = env.airspace.actual_ring(other.ring_role)
        reference = steady_holder_positions(other, ring, times, env.now)
        distances = np.linalg.norm(positions-reference, axis=1)
        index = int(np.argmin(distances))
        unsafe = np.flatnonzero(distances < env.config.safe_distance)
        first = None if not len(unsafe) else int(unsafe[0])
        lower, upper = safety.holder_reachable_tube(other, snapshot, times)
        nominal = np.asarray(safety._holder_positions(other, snapshot, times, 0.0))
        pairs.append(dict(other_uid=other.uid, minimum_m=float(distances[index]),
            minimum_at=float(times[index]), minimum_during_ring_prefix=bool(times[index] < departure),
            first_unsafe_at=None if first is None else float(times[first]),
            first_unsafe_during_ring_prefix=None if first is None else bool(times[first] < departure),
            candidate_radius_at_minimum=float(np.linalg.norm(positions[index,:2])),
            candidate_altitude_at_minimum=float(positions[index,2]),
            tube_width_max=float(np.max(upper-lower)),
            holder_reference_max_error=float(np.max(np.linalg.norm(reference-nominal,axis=1)))))
    nearest = min(pairs, key=lambda row:row['minimum_m']) if pairs else None
    return dict(full_approved=full.approved, full_reason=full.reason,
        solo_approved=solo.approved, solo_reason=solo.reason,
        dynamics_errors=env.generator.dynamics.validate_trajectory(plan),
        execution_boundary=env._validate_v3_execution_boundary(uav,plan),
        duration=float(plan.end_time-plan.start_time), departure_delay=departure-env.now,
        entry_speed=plan.metadata.get('actual_entry_speed_mps'),
        required_braking=plan.metadata.get('required_braking_duration_seconds'),
        energy_required=env.energy.emergency_required_energy(uav,plan),
        remaining_energy=float(uav.energy_remaining), nearest=nearest, pairs=pairs)


def probe(snapshot_path, output, *, kind, limit=0):
    # These files are produced by this project's trusted local replay only.
    with snapshot_path.open('rb') as stream:
        env=pickle.load(stream)
    if env.config.holding_speed_contract != 'nominal_until_release':
        raise ValueError('requires nominal holder contract')
    for u in env.active.values():
        if u.ring_role is not RingRole.INNER or u.committed_plan is not None:
            raise ValueError('probe requires only uncommitted inner holders')
        nominal=env.airspace.actual_ring(u.ring_role).nominal_speed+env.airspace.speed_profile_bias
        if abs(u.speed-nominal)>1e-7:
            raise ValueError('analytic probe requires steady nominal speed')
    started=time.monotonic()
    rows=[]
    for u in sorted(env.active.values(),key=lambda u:(u.soc,u.uid)):
        for lz_id in env.lz_manager.rank_available_zones(u,env.now):
            for residual in ((0.,-1.,1.) if kind=='normal' else (None,)):
                row=dict(time=env.now,uid=u.uid,lz_id=lz_id,kind=kind,residual=residual)
                try:
                    if kind=='normal':
                        intent=LowLevelIntent.from_array(np.asarray([1.,residual,0.]))
                        _,plan,reason=env._candidate_plan(u,RepairCandidate('static_probe',intent),
                            'final',WorldSnapshot(env.now,env.active,env.airspace),lz_id)
                        row['candidate_reason']=reason
                    else:
                        plan=env.generator.emergency_landing(u,env.lz_manager.zones[lz_id],env.airspace,env.now)
                    row.update(plan_missing=True) if plan is None else row.update(measure_candidate(env,u,plan))
                except Exception as exc:
                    row['error']=dict(type=type(exc).__name__,message=str(exc))
                rows.append(row)
                write_json(output/'progress.json',dict(rows=len(rows),elapsed=time.monotonic()-started,last=row))
                if limit and len(rows)>=limit:
                    break
            if limit and len(rows)>=limit:break
        if limit and len(rows)>=limit:break
    result=dict(snapshot=str(snapshot_path),time=env.now,kind=kind,limit=limit,
        safe_distance=env.config.safe_distance,rows=rows,
        reasons=dict(Counter(row.get('full_reason',row.get('candidate_reason','error')) for row in rows)),
        technical_errors=sum('error' in row for row in rows),elapsed=time.monotonic()-started,
        PPO_ALLOWED=False,trained=False)
    write_json(output/'result.json',result)
    return result


def probe_transfer_boundaries():
    """Six original initial-state candidates; no reset, step or intervention."""
    env=TerminalParallelEnv(build_config(requests=12,warm_start=6,high_soc=True),seed=890501)
    rows=[]
    for uid,target in [(0,RingRole.MIDDLE),(4,RingRole.INNER)]:
     for residual in (0.,-1.,1.):
      u=env.active[uid]
      _,p,reason=env._candidate_plan(u,RepairCandidate('boundary_probe',LowLevelIntent.from_array(np.array([1.,residual,0.]))),target,WorldSnapshot(env.now,env.active,env.airspace),None)
      row=dict(uid=uid,residual=residual,reason=reason)
      if p is not None:
       row['dynamics_errors']=env.generator.dynamics.validate_trajectory(p)
       diffs=[]
       for i,seg in enumerate(p.polynomial_segments):
        for name,t,point in [('start',seg.start_time,p.points[i]),('end',seg.end_time,p.points[i+1])]:
         observed=seg.sample(t)
         delta=[float(np.max(np.abs(a-b))) for a,b in zip(observed,(point.position,point.velocity,point.acceleration))]
         if max(delta)>1e-6:
          diffs.append(dict(segment=i,side=name,time=t,duration=seg.duration,pva_max_errors=delta,expected=[x.tolist() for x in (point.position,point.velocity,point.acceleration)],observed=[x.tolist() for x in observed]))
       row['endpoint_differences']=diffs
       row['minimum_segment_duration']=min(x.duration for x in p.polynomial_segments)
      rows.append(row)
    return rows


def probe_spiral_candidate(env):
    """One fixed diagnostic counterfactual, not a production action mapping.

    The 100m construction circle belongs only to the candidate builder. The
    existing unchanged full-world SafetyFilter certifies the combined plan.
    Speed/timing proposal semantics and multi-aircraft scheduling still need
    separate implementation and validation before this can be integrated.
    """
    u=min(env.active.values(),key=lambda u:(u.soc,u.uid))
    ring=env.airspace.actual_ring(u.ring_role)
    assert u.uid==6 and abs(env.now-600.)<1e-9 and abs(u.speed-16.)<1e-9
    lz_id=env.lz_manager.rank_available_zones(u,env.now)[0]
    assert lz_id==1
    T=60.; goal_radius=100.; dr=ring.radius-goal_radius; omega=u.speed/ring.radius
    points=[]
    for t in np.linspace(0.,T,int(round(T/env.config.physics_dt))+1):
     s=t/T; q=10*s**3-15*s**4+6*s**5
     q1=30*s**2-60*s**3+30*s**4; q2=60*s-180*s**2+120*s**3
     r=ring.radius-dr*q; r1=-dr*q1/T; r2=-dr*q2/T**2
     theta=u.ring_progress+omega*t
     er=np.array([math.cos(theta),math.sin(theta),0.]); et=np.array([-math.sin(theta),math.cos(theta),0.])
     p=r*er+np.array([0.,0.,ring.height]); v=r1*er+r*omega*et
     a=(r2-r*omega**2)*er+2*r1*omega*et
     points.append(TrajectoryPoint(env.now+t,p,v,a))
    shadow=deepcopy(u);end=points[-1]
    shadow.position=end.position.copy();shadow.velocity=end.velocity.copy();shadow.acceleration=end.acceleration.copy()
    shadow.speed=float(np.linalg.norm(end.velocity[:2]));shadow.ring_progress=u.ring_progress+omega*T
    inner=replace(ring,radius=goal_radius)
    construction_airspace=SimpleNamespace(actual_ring=lambda role:inner)
    reservation=SimpleNamespace(merge_time=end.t,reservation_id='static_spiral_probe')
    tail=env.generator.final_approach(shadow,env.lz_manager.zones[lz_id],reservation,end.t,airspace=construction_airspace)
    metadata=dict(tail.metadata);metadata.pop('trajectory_hash',None)
    metadata.update(scheduled_departure=env.now,diagnostic_only=True,spiral_duration=T,spiral_target_radius=goal_radius)
    plan=TrajectoryPlan('static_spiral_probe',u.uid,'final_approach',tuple(points[:-1])+tail.points,env.now,tail.end_time,metadata=metadata)
    result=dict(uid=u.uid,lz_id=lz_id,variant='diagnostic_only_constant_angular_rate_inward',
        production_modified=False,PPO_ALLOWED=False)
    result.update(measure_candidate(env,u,plan))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=('normal','emergency','spiral','transfer'),required=True)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.limit<0:parser.error('limit must be nonnegative')
    if args.kind!='transfer' and args.snapshot is None:parser.error('--snapshot is required for this kind')
    if args.kind in ('spiral','transfer') and args.limit:parser.error('fixed probes do not use --limit')
    if args.output.exists():parser.error('preserve existing output')
    args.output.mkdir(parents=True)
    if args.kind in ('normal','emergency'):
        result=probe(args.snapshot,args.output,kind=args.kind,limit=args.limit)
    else:
        started=time.monotonic()
        if args.kind=='transfer':
            rows=probe_transfer_boundaries()
        else:
            with args.snapshot.open('rb') as stream:env=pickle.load(stream)
            rows=[probe_spiral_candidate(env)]
        result=dict(kind=args.kind,rows=rows,snapshot=str(args.snapshot),
            reasons=dict(Counter(row.get('full_reason',row.get('reason','')) for row in rows)),
            technical_errors=0,elapsed=time.monotonic()-started,PPO_ALLOWED=False,trained=False)
        write_json(args.output/'result.json',result)
    print(json.dumps({key:result[key] for key in ('kind','reasons','technical_errors','elapsed')}))
    return int(result['technical_errors']>0)


if __name__=='__main__':
    raise SystemExit(main())
