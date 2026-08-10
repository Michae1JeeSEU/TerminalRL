"""Deterministic A4 baseline which emits only the canonical 3-D intention."""
from __future__ import annotations
import numpy as np
class RuleLowLevelPolicy:
    def act(self, observations, snapshot=None):
        ids = observations.keys() if isinstance(observations,dict) else []
        return {str(uid):np.array([1.,0.,0.],dtype=np.float32) for uid in ids}
