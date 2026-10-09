"""Finite source-bound receiving interface for cooperative terminal policies.

This module cannot access an environment, teacher, candidate generator or
future traffic. The publisher owns surveillance/intent forecasts; the UAV
receives only this declared finite report. Ideal fresh-report simulation does
not establish robustness to real sensing error, delay or untracked intruders.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .trajectory_context_observation import TrajectoryContextObservation

CONTRACT='cooperative_terminal_policy_report_v1'
MAX_NEIGHBORS=32
PUBLICATION_PERIOD_SECONDS=1.
MAX_AGE_SECONDS=2.
FORECAST_HORIZON_SECONDS=150.
MAX_NUMERIC_PAYLOAD_BYTES=8192

# Local state + locally known assignment/retry/service. release_eligible (17)
# is formed by the local controller using the same publication's freeze flag.
LOCAL_INDICES=tuple(range(27))+(87,88)
CENTER_INDICES=tuple(range(27,87))+tuple(range(89,159))
assert len(LOCAL_INDICES)==29 and len(CENTER_INDICES)==130


class ObservationUnavailable(ValueError):
    """Receiver must keep its hold/fail-safe path; do not query the actor."""


@dataclass(frozen=True)
class LocalPolicySample:
    uid: int
    timestamp: float
    values: tuple[float,...]  # 29 named self/controller fields above


@dataclass(frozen=True)
class PublishedNeighbor:
    uid: int
    terminal_id: str
    report_timestamp: float
    values: tuple[float,...]  # existing 50D phase/status/current/announced forecast


@dataclass(frozen=True)
class TerminalPolicyPublication:
    terminal_id: str
    recipient_uid: int
    timestamp: float
    center_values: tuple[float,...]  # 130 existing center/derived fields
    # Current registered service-area traffic, not future arrivals or all-world
    # truth. Header IDs detect dropped reports and never enter model tensors.
    expected_neighbor_uids: tuple[int,...]
    neighbors: tuple[PublishedNeighbor,...]
    surveillance_complete: bool
    contract: str=CONTRACT


def _check(condition,message):
    if not condition:raise ObservationUnavailable(message)


def _vector(values,width):
    result=np.asarray(values,dtype=np.float32)
    _check(result.shape==(width,) and np.isfinite(result).all(),'invalid finite report vector')
    return result


def receive_policy_observation(local,publication,*,terminal_id,now):
    """Only explicitly received local + terminal data may form the observation.

    The existing V2.6 vector is reassembled unchanged. No privileged fallback,
    stale-neighbor removal, online safety-oracle query or silent truncation.
    """
    _check(isinstance(local,LocalPolicySample) and isinstance(publication,TerminalPolicyPublication),
           'typed local/terminal reports required')
    _check(publication.contract==CONTRACT and publication.terminal_id==terminal_id,
           'wrong terminal publication')
    _check(local.uid==publication.recipient_uid,'wrong report recipient')
    _check(all(math.isfinite(float(t)) for t in (now,local.timestamp,publication.timestamp)),
           'invalid publication clock')
    _check(0.<=float(now)-publication.timestamp<=MAX_AGE_SECONDS,'publication stale or from future')
    _check(abs(local.timestamp-publication.timestamp)<=1e-6,
           'self and relative-track reference epochs differ')
    _check(publication.surveillance_complete,'terminal surveillance is incomplete')
    expected=publication.expected_neighbor_uids
    received=tuple(t.uid for t in publication.neighbors)
    _check(len(expected)<=MAX_NEIGHBORS and len(received)<=MAX_NEIGHBORS,'terminal report capacity exceeded')
    _check(len(set(expected))==len(expected) and len(set(received))==len(received)
           and set(expected)==set(received) and local.uid not in expected,'missing/duplicate/self neighbor report')
    ego=np.empty(159,dtype=np.float32)
    ego[list(LOCAL_INDICES)]=_vector(local.values,len(LOCAL_INDICES))
    ego[list(CENTER_INDICES)]=_vector(publication.center_values,len(CENTER_INDICES))
    ordered=sorted(publication.neighbors,key=lambda r:r.uid)
    traffic=[]
    for report in ordered:
        _check(report.terminal_id==terminal_id,'neighbor report outside subscribed terminal')
        _check(math.isfinite(report.report_timestamp) and
            0.<=publication.timestamp-report.report_timestamp and
            float(now)-report.report_timestamp<=MAX_AGE_SECONDS,'neighbor report stale or from future')
        traffic.append(_vector(report.values,50))
    traffic=np.asarray(traffic,dtype=np.float32).reshape(-1,50)
    # Fixed-width numeric payload estimate includes both local and center
    # vectors, timestamps, IDs and a 64-byte header. Transport framing is not
    # implemented here; this is not a measured radio throughput claim.
    n=len(traffic);payload_bytes=ego.nbytes+traffic.nbytes+8*(2+n)+4*(1+n)+64
    _check(payload_bytes<=MAX_NUMERIC_PAYLOAD_BYTES,'numeric payload exceeds declared report budget')
    return TrajectoryContextObservation(ego,traffic,np.ones(n,dtype=bool),
        tuple(r.uid for r in ordered),(None,None),0.)
