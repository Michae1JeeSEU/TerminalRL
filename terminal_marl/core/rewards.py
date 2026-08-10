"""A5 low-level reward components.  Touchdown, never service completion, terminates agent reward."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass
class LowLevelReward:
    time:float=0.;progress:float=0.;release:float=0.;touchdown:float=0.;failure:float=0.
    @property
    def total(self):return self.time+self.progress+self.release+self.touchdown+self.failure
    def as_dict(self):return {"time":self.time,"progress":self.progress,"release":self.release,"touchdown":self.touchdown,"failure":self.failure,"total":self.total}
