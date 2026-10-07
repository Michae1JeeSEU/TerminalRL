"""Strict numeric loader for reconstructed V2.5 train observations; no learner."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..core.trajectory_context_observation import (
    VERSION, EGO_SCHEMA, TRAFFIC_SCHEMA, TrajectoryContextObservation, pad_trajectory_context,
)

DATA_SCHEMA = 'p3_v3_trajectory_context_reconstruction_v1'


@dataclass(frozen=True)
class ContextBatch:
    observations: tuple[TrajectoryContextObservation, ...]
    gate_target: np.ndarray
    proposal_target: np.ndarray
    decision_ids: tuple[str, ...]

    def padded(self):
        return pad_trajectory_context(self.observations)


def decode_row(row):
    if row['schema'] != DATA_SCHEMA or row['observation_schema'] != VERSION or row['partition'] != 'train':
        raise ValueError('V2.5 requires reconstructed train data')
    if row['ego_feature_hash'] != EGO_SCHEMA.feature_hash or row['traffic_feature_hash'] != TRAFFIC_SCHEMA.feature_hash:
        raise ValueError('V2.5 feature hash mismatch')
    if row['actor_mask'] is not True:
        raise ValueError('only original eligible actors can be reconstructed')
    ego = np.asarray(row['ego'], dtype=np.float32)
    traffic = np.asarray(row['traffic'], dtype=np.float32)
    if traffic.shape == (0,):
        traffic = traffic.reshape(0, 50)
    mask = np.asarray(row['traffic_mask'])
    if mask.shape == (0,) and len(traffic) == 0:
        mask = mask.astype(bool)
    if (ego.shape != (105,) or traffic.ndim != 2 or traffic.shape[1] != 50
            or mask.shape != (len(traffic),) or mask.dtype != np.bool_ or not mask.all()
            or not np.isfinite(ego).all() or not np.isfinite(traffic).all()):
        raise ValueError('invalid V2.5 tensors')
    gate, proposal = row['teacher_gate'], row['teacher_proposal']
    if type(gate) is not int or gate not in (0, 1):
        raise ValueError('invalid original teacher gate')
    if (gate == 0 and proposal is not None) or (gate == 1 and (type(proposal) is not int or not 0 <= proposal < 25)):
        raise ValueError('invalid original teacher proposal')
    # IDs/trace metadata are intentionally not retained in numeric observations.
    return TrajectoryContextObservation(ego, traffic, mask, (), (None, None), 0.), gate, -1 if proposal is None else proposal


def load_context_dataset(path, *, allowed_train_seeds):
    path = Path(path)
    result = json.loads((path/'result.json').read_text(encoding='utf8'))
    if result['status'] != 'RECONSTRUCTION_COMPLETE' or result['mode'] != 'full':
        raise ValueError('only complete full reconstruction can be loaded for future training')
    observations, gates, proposals, ids = [], [], [], []
    for line in (path/'observations.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        if row['seed'] not in allowed_train_seeds:
            raise ValueError('seed outside allowed train inputs')
        expected_id = f"p3_v3_90d_sparse_expert_v1:{row['seed']}:{row['tick']}:{row['uid']}"
        if row['decision_id'] != expected_id or row['decision_id'] in ids:
            raise ValueError('invalid or duplicate decision ID')
        obs, gate, proposal = decode_row(row)
        observations.append(obs); gates.append(gate); proposals.append(proposal); ids.append(row['decision_id'])
    if len(ids) != result['rows'] or not ids:
        raise ValueError('incomplete V2.5 rows')
    return ContextBatch(tuple(observations), np.asarray(gates, dtype=np.int64),
                        np.asarray(proposals, dtype=np.int64), tuple(ids))
