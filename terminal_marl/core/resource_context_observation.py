"""V2.6 augments traffic context with pre-action spatial pad resources.

The center already observes each pad's occupancy and announced release ETA.
No candidate plan, safety predicate, teacher label or future action is queried.
"""
import math

import numpy as np

from .observations import FeatureSpec, ObservationSchema, build_center_broadcast
from .trajectory_context_observation import (
    EGO_SCHEMA as TRAFFIC_EGO_SCHEMA, POSITION_SCALE, TIME_SCALE,
    TrajectoryContextCache, TrajectoryContextObservation,
)

VERSION='phaseA_low_obs_v2_6_spatial_pad_context_set'
PAD_FEATURES=('relative_x','relative_y','relative_z','free','reserved','occupied',
              'compatible','eta_known','eta_scaled')
PAD_COUNT=6
EGO_SCHEMA=ObservationSchema(TRAFFIC_EGO_SCHEMA.features+(
    FeatureSpec('landing_zones_spatial_resources',PAD_COUNT*len(PAD_FEATURES)),))
assert EGO_SCHEMA.dimension==159


class ResourceContextCache(TrajectoryContextCache):
    def __init__(self,env):
        super().__init__(env)
        self.center=build_center_broadcast(env)
        self.zones=tuple(sorted(env.lz_manager.zones,key=lambda z:z.lz_id))
        if [z.lz_id for z in self.zones]!=list(range(PAD_COUNT)):
            raise ValueError('V2.6 requires the six declared physical pad slots')

    def build(self,ego):
        original=super().build(ego)
        angle=math.atan2(ego.position[1],ego.position[0])
        basis=np.asarray(((math.cos(angle),math.sin(angle),0.),
                          (-math.sin(angle),math.cos(angle),0.),(0.,0.,1.)))
        broadcasts={z.lz_id:z for z in self.center.landing_zones}
        mission=ego.mission_type.value
        resources=[]
        for zone in self.zones:
            broadcast=broadcasts[zone.lz_id]
            eta=broadcast.eta_seconds
            known=math.isfinite(eta)
            if known and eta<0:raise ValueError('negative pad availability ETA')
            relative=(np.asarray(zone.position)-ego.position) @ basis.T / POSITION_SCALE
            resources.extend((*relative,float(broadcast.status=='free'),
                float(broadcast.status=='reserved'),float(broadcast.status=='occupied'),
                float(mission in broadcast.compatible_missions),float(known),
                float(eta/(eta+TIME_SCALE)) if known else 0.))
        values=np.concatenate((original.ego,np.asarray(resources,dtype=np.float32))).astype(np.float32)
        if values.shape!=(159,) or not np.isfinite(values).all():
            raise ValueError('invalid V2.6 resource observation')
        return TrajectoryContextObservation(values,original.traffic,original.traffic_mask,
            original.traffic_uids,original.target_uids,original.target_reference_seconds)


def build_low_observation_v2_6(uav,env,*,cache=None):
    return (cache or ResourceContextCache(env)).build(uav)
