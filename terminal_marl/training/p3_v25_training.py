"""Current-state critic context and detached V2.5 rollout records.

No execution/learning is started here. Context contains public simulator state,
not the teacher, a sampled future action, a future outcome, or random draws.
"""
import numpy as np

from ..algo.trajectory_set_ppo import CENTRAL_CONTEXT_DIM, SetStep
from ..core.config import RingRole

CENTRAL_FEATURES = (
    'time','arrivals_remaining','next_arrival_delay','arrivals_exhausted',
    'queue_count','queue_wait_mean','queue_wait_max','queue_soc_min',
    'lz_free','lz_reserved','lz_occupied','service_time_mean','service_time_max',
    'reservation_time_mean','reservation_time_max','active_count','unresolved_count',
    'outer_radius','outer_height','outer_speed','middle_radius','middle_height','middle_speed',
    'inner_radius','inner_height','inner_speed',
)
assert len(CENTRAL_FEATURES)==CENTRAL_CONTEXT_DIM


def central_context(env):
    if env.config.total_requested_uavs is None:
        raise ValueError('initial V2.5 training supports finite cohorts')
    total=max(1,env.config.total_requested_uavs)
    queue=list(env.admission.queue.items)
    waits=[q.external_wait_time for q in queue]
    def mean(values): return float(np.mean(values)) if values else 0.
    def maximum(values): return max(values,default=0.)
    def bounded(value): return value/(value+150.)
    zones=env.lz_manager.zones
    statuses=[z.status.value for z in zones]
    services=[float(z.service_remaining) for z in zones if z.status.value=='occupied']
    reservations=[max(0.,z.expected_release_at-env.now) for z in zones
                  if z.status.value=='reserved' and z.expected_release_at is not None]
    exhausted=env.arrivals.exhausted()
    values=[env.now/(env.now+1800.),max(0,total-env.arrivals.generated)/total,
            0. if exhausted else bounded(max(0.,env.arrivals.next_time-env.now)),float(exhausted),
            len(queue)/total,bounded(mean(waits)),bounded(maximum(waits)),min((q.soc for q in queue),default=0.),
            *[statuses.count(s)/max(1,len(zones)) for s in ('free','reserved','occupied')],
            bounded(mean(services)),bounded(maximum(services)),
            bounded(mean(reservations)),bounded(maximum(reservations)),
            len(env.active)/total,len(env.ledger.unresolved)/total]
    for role in RingRole:
        ring=env.airspace.actual_ring(role)
        values.extend((ring.radius/1000.,ring.height/1000.,ring.nominal_speed/25.))
    result=np.asarray(values,dtype=np.float32)
    if result.shape!=(CENTRAL_CONTEXT_DIM,) or not np.isfinite(result).all():
        raise ValueError('invalid critic context')
    return result


def step_record(request, context, *, value, next_value, reward, terminated, episode_end,
                stochastic_behavior=True):
    if request.uids:
        ego,traffic,mask=(t.detach().cpu().numpy().copy() for t in request.tensors)
        gl,pl=request.logits
    else:
        ego=np.zeros((0,105),np.float32);traffic=np.zeros((0,0,50),np.float32);mask=np.zeros((0,0),bool)
        gl=np.zeros((0,2),np.float32);pl=np.zeros((0,25),np.float32)
    gp=np.asarray([request.likelihoods[u]['gate'] for u in request.uids],np.float32)
    pp=np.asarray([np.nan if request.likelihoods[u]['proposal'] is None else request.likelihoods[u]['proposal']
                   for u in request.uids],np.float32)
    return SetStep(ego,traffic,mask,np.asarray(context,dtype=np.float32).copy(),
                   request.gates.copy(),request.proposals.copy(),gp,pp,gl.copy(),pl.copy(),
                   request.actor_mask.copy(),float(value),float(next_value),float(reward),
                   bool(terminated),bool(episode_end),stochastic_behavior=stochastic_behavior)
