"""P3 service-complete lifecycle metrics from the existing simulator ledger."""
import numpy as np


def completed_metrics(env):
    rows=[]
    for uav in sorted(env.completed_uavs,key=lambda u:u.arrival_id):
        rows.append(dict(arrival_id=int(uav.arrival_id),uid=int(uav.uid),
            base_priority=int(uav.base_priority),mission=uav.mission_type.value,
            initial_soc=float((uav.energy_remaining+uav.energy_consumed)/uav.energy_capacity),
            final_soc=float(uav.soc),outcome=uav.terminal_outcome.value,
            emergency_landing=bool(uav.emergency_landing),request_time=float(uav.request_time),
            touchdown_time=float(uav.touchdown_time),completion_time=float(uav.completion_time),
            request_to_touchdown_seconds=float(uav.touchdown_time-uav.request_time),
            request_to_completion_seconds=float(uav.completion_time-uav.request_time),
            external_wait_seconds=float(uav.external_wait_time),
            internal_wait_seconds=float(uav.internal_wait_time),
            wait_seconds=float(uav.external_wait_time+uav.internal_wait_time),
            energy_consumed=float(uav.energy_consumed)))
    def mean(group,key):
        return float(np.mean([r[key] for r in group])) if group else None
    priorities=sorted({r['base_priority'] for r in rows})
    # Initial SOC defines the cohort, not an outcome-dependent final SOC.
    low=[r for r in rows if r['initial_soc']<=.65]
    def priority_mean(key):
        return (float(np.average([r[key] for r in rows],weights=[r['base_priority'] for r in rows]))
                if rows else None)
    def distribution(key):
        values=[r[key] for r in rows]
        return dict(count=len(values),mean=float(np.mean(values)) if values else None,
            p95=float(np.percentile(values,95)) if values else None,
            max=max(values) if values else None)
    return dict(completed_rows=rows,completed_count=len(rows),
        wait_definition='external queue plus uncommitted ring holding; excludes committed trajectories',
        lifecycle_distributions={key:distribution(key) for key in
            ('wait_seconds','request_to_touchdown_seconds','request_to_completion_seconds')},
        priorities_observed=priorities,initial_soc_min=min((r['initial_soc'] for r in rows),default=None),
        initial_soc_max=max((r['initial_soc'] for r in rows),default=None),
        priority_weighted_wait_seconds=priority_mean('wait_seconds'),
        priority_weighted_request_to_touchdown_seconds=priority_mean('request_to_touchdown_seconds'),
        priority_weighted_request_to_completion_seconds=priority_mean('request_to_completion_seconds'),
        low_soc_count=len(low),low_soc_wait_mean=mean(low,'wait_seconds'),
        low_soc_request_to_touchdown_mean=mean(low,'request_to_touchdown_seconds'),
        by_priority={str(p):dict(count=sum(r['base_priority']==p for r in rows),
            wait_mean=mean([r for r in rows if r['base_priority']==p],'wait_seconds')) for p in (1,3,5)},
        normal_count=sum(r['outcome']=='completed' for r in rows),
        emergency_count=sum(r['outcome']=='emergency_landed' for r in rows))


def reward_parts(info):
    """Count each physical reward once, including upstream queue costs."""
    parts={}
    for row in info['reward_components'].values():
        for key,value in row.items():
            if key!='total':parts[key]=parts.get(key,0.)+float(value)
    for key,value in info.get('team_reward_components',{}).items():
        parts[key]=parts.get(key,0.)+float(value)
    if not np.isfinite(list(parts.values())).all() or not np.isclose(sum(parts.values()),
            float(info['team_reward']),atol=1e-7,rtol=0):
        raise ValueError('team reward components do not sum to the recorded reward')
    return parts
