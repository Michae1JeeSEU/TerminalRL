"""Macro-step environment for canonical 10-D dynamic terminal control."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from ..core.config import RingRole, SimConfig
from ..core.reconfiguration_safety import ReconfigurationSafetyCoordinator
from ..core.uav import TerminalOutcome
from ..policies import RuleLowLevelPolicy
from .parallel_env import TerminalParallelEnv

HIGH_LEVEL_OBSERVATION_DIM = 29

class LowLevelPolicy(Protocol):
    def act(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

class RuleLowLevelAdapter:
    def __init__(self) -> None:
        self.policy = RuleLowLevelPolicy(); self.checkpoint_hash: str | None = None
    def act(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.policy.act(observations)

class FrozenLowLevelPolicy:
    """Strict A5 adapter: missing/invalid weights are an error, never a fallback."""
    def __init__(self, checkpoint: str | Path, config: SimConfig, expected_hash: str | None = None) -> None:
        if config.action_schema_version == "low_action_schema_v4_2_hierarchical":
            raise ValueError(
                "FrozenLowLevelPolicy does not support V4.2 hierarchical gate+K3 actions; "
                "the SharedPPO adapter is limited to the qualified 3-output low-level interface"
            )
        path = Path(checkpoint)
        if not path.is_file(): raise FileNotFoundError(f"frozen low-level checkpoint does not exist: {path}")
        self.checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if expected_hash is not None and self.checkpoint_hash != expected_hash.upper(): raise ValueError("frozen low-level checkpoint SHA256 mismatch")
        import torch
        # Evaluation invokes a tiny variable-N forward pass every simulator
        # step; one CPU thread avoids OpenMP launch overhead without changing
        # the frozen weights or the policy computation.
        torch.set_num_threads(1)
        from ..core.observations import observation_schema_for_version
        from ..algo.shared_ppo import SharedPPO
        from ..core.checkpoints import load_checkpoint
        self.policy = SharedPPO(observation_schema_for_version(config.observation_schema_version).dimension)
        load_checkpoint(path, self.policy, config); self.policy.model.eval()
    def act(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not observations: return {}
        identifiers = list(observations); rows = np.stack([observations[key] for key in identifiers])
        actions, _, _ = self.policy.act(rows, rows, deterministic=True)
        if actions.shape != (len(identifiers), 3) or not np.all(np.isfinite(actions)):
            raise FloatingPointError("frozen low-level policy returned invalid action")
        return {key: action for key, action in zip(identifiers, actions)}

@dataclass(frozen=True)
class HighLevelReward:
    service_completion: float = 0.0; external_queue: float = 0.0; delay: float = 0.0
    safety: float = 0.0; energy: float = 0.0; reconfiguration: float = 0.0; footprint: float = 0.0
    @property
    def total(self) -> float: return sum(getattr(self, name) for name in self.__dataclass_fields__)
    def as_dict(self) -> dict[str, float]: return {name: float(getattr(self, name)) for name in self.__dataclass_fields__} | {"total": self.total}

class DynamicHighLevelWrapper:
    """Runs one 180-second high decision through a fixed low-level controller."""
    action_dim = 10; observation_dim = HIGH_LEVEL_OBSERVATION_DIM
    def __init__(self, config: SimConfig, low_level: LowLevelPolicy | None = None) -> None:
        if not config.high_level_enabled: raise ValueError("DynamicHighLevelWrapper requires high_level_enabled=True")
        self.config=config; self.low_level=low_level or RuleLowLevelAdapter(); self.base=TerminalParallelEnv(config); self.coordinator=ReconfigurationSafetyCoordinator(self.base)
        self._initial_radius_sq = float(sum(float(config.rings[role].radius) ** 2 for role in RingRole))
        self.action_history: list[dict] = []
        self._previous_completed=0
        self._previous_energy=0.0
        self._low_demand_since: float | None = None
        self._energy_ledger: dict[int, float] = {}
    def _energy_total(self) -> float:
        return float(sum(self._energy_ledger.values()))

    @staticmethod
    def _energy_arrival_id(state) -> int:
        if hasattr(state, "arrival_id"):
            return int(state.arrival_id)
        return int(state.request.arrival_id)

    def _energy_object_refs(self) -> list[object]:
        return (
            list(self.base.active.values())
            + list(self.base.service_uavs.values())
            + list(self.base.completed_uavs)
            + list(self.base.admission.queue.items)
        )

    def _observe_energy(self, previous_refs: list[object]) -> None:
        """Retain the maximum observed consumption for every arrival."""
        states = previous_refs + self._energy_object_refs()
        observed: dict[int, float] = {}
        for state in states:
            arrival_id = self._energy_arrival_id(state)
            consumed = float(state.energy_consumed)
            observed[arrival_id] = max(observed.get(arrival_id, 0.0), consumed)
        for row in self.base.energy_outcomes:
            arrival_id = row.get("arrival_id")
            consumed = row.get("energy_consumed")
            if arrival_id is None or consumed is None:
                continue
            arrival_id = int(arrival_id)
            observed[arrival_id] = max(observed.get(arrival_id, 0.0), float(consumed))
        for arrival_id, consumed in observed.items():
            self._energy_ledger[arrival_id] = max(self._energy_ledger.get(arrival_id, 0.0), consumed)
    def _context(self) -> dict:
        airspace=self.base.airspace; occupancy=[airspace.occupancy(role) for role in RingRole]; capacity=[airspace.capacity(role) for role in RingRole]
        final=sum(u.phase.value=="final" for u in self.base.active.values()); occupied=sum(zone.status.value!="free" for zone in self.base.lz_manager.zones)
        return {"external_queue":len(self.base.admission.queue.items),"occupancy":occupancy,"capacity":capacity,"occupancy_ratio":[n/max(1,c) for n,c in zip(occupancy,capacity)],"final_pressure":final/max(1,self.config.final_queue_capacity),"lz_utilization":occupied/len(self.base.lz_manager.zones),"transition_active":self.base.airspace.transition is not None}

    @staticmethod
    def _is_congested(context: dict) -> bool:
        return bool(
            context["external_queue"] > 0
            or max(context["occupancy_ratio"], default=0.0) >= 0.65
            or context["final_pressure"] > 0.0
            or context["lz_utilization"] >= 0.80
        )

    @staticmethod
    def _is_low_demand(context: dict) -> bool:
        return bool(
            context["external_queue"] == 0
            and max(context["occupancy_ratio"], default=0.0) < 0.25
            and context["final_pressure"] == 0.0
            and context["lz_utilization"] < 0.50
        )

    def _update_low_demand_state(self) -> None:
        context = self._context()
        if self._is_low_demand(context):
            if self._low_demand_since is None:
                self._low_demand_since = float(self.base.now)
        else:
            self._low_demand_since = None

    def geometry_gate_reason(self) -> str:
        context = self._context()
        if self._is_congested(context):
            return "congestion"
        if (
            self._is_low_demand(context)
            and
            self._low_demand_since is not None
            and self.base.now - self._low_demand_since >= self.config.geometry_min_dwell_seconds
        ):
            return "sustained_low_demand"
        return "none"

    def geometry_event_gate(self) -> bool:
        return self.geometry_gate_reason() != "none"
    def observation(self) -> np.ndarray:
        context=self._context(); airspace=self.base.airspace; actual=airspace.normalized_state_vector()
        target_raw=np.asarray([airspace.target_ring(role).radius for role in RingRole]+[airspace.target_ring(role).height for role in RingRole],np.float32)
        lo=np.asarray([self.config.radius_min]*3+[self.config.height_min]*3,np.float32); hi=np.asarray([self.config.radius_max]*3+[self.config.height_max]*3,np.float32)
        target=2.*(target_raw-lo)/(hi-lo)-1.; remain=[zone.service_remaining for zone in self.base.lz_manager.zones]
        vector=np.asarray(list(actual)+list(target)+context["occupancy_ratio"]+[min(1.,context["external_queue"]/20.),context["final_pressure"],context["lz_utilization"],min(1.,max(remain,default=0.)/120.),float(self.geometry_event_gate()),float(context["transition_active"]),airspace.transition_progress(self.base.now),min(1.,self.base.now/max(1.,self.config.max_simulation_seconds)),min(1.,self.base.telemetry["safety_interventions"]/50.),min(1.,self.base.telemetry["collisions"]/5.)],np.float32)
        if vector.shape!=(self.observation_dim,): raise AssertionError(f"high-level observation must be ({self.observation_dim},), got {vector.shape}")
        return vector
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.base.reset(seed)
        self.coordinator=ReconfigurationSafetyCoordinator(self.base)
        self.action_history=[]
        self._previous_completed=0
        self._energy_ledger.clear()
        self._observe_energy([])
        self._previous_energy=self._energy_total()
        self._low_demand_since = None
        self._update_low_demand_state()
        return self.observation(),{"low_level_checkpoint_hash":getattr(self.low_level,"checkpoint_hash",None)}
    def _done(self) -> bool:
        return bool(self.base.arrivals.exhausted() and not self.base.active and not self.base.admission.queue.items and not self.base.service_uavs)
    def step(self, action: np.ndarray) -> tuple[np.ndarray,float,bool,bool,dict]:
        requested=np.asarray(action,np.float32)
        if requested.shape!=(self.action_dim,) or not np.all(np.isfinite(requested)): raise ValueError("high-level action must be finite shape (10,)")
        gate_reason=self.geometry_gate_reason(); gate=gate_reason != "none"; shield=self.coordinator.coordinate(requested,geometry_gate=gate); update=shield.update; before=dict(self.base.telemetry); before_completed=self._previous_completed; before_energy=self._previous_energy; start_time=self.base.now; low_reward=0.; latest_info={}
        external_queue_seconds=0.0
        unresolved_request_seconds=0.0
        footprint_seconds=0.0
        for _ in range(max(1,round(self.config.high_level_decision_interval_seconds/self.config.dt))):
            deferred = self.coordinator.advance()
            requested_low_actions = self.low_level.act(self.base._observations())
            # A proposed student deployment never routes routine low-level
            # decisions through the legacy high-level repair helper.
            low_actions = requested_low_actions if self.config.experiment_mode == "dynamic_hrl_student" else self.coordinator.apply_speed_repairs(requested_low_actions)
            tick_start = float(self.base.now)
            previous_refs = self._energy_object_refs()
            queue_counter_before = float(self.base.external_queue_uav_seconds)
            _,rewards,_,_,latest_info=self.base.step(low_actions)
            tick_end = float(self.base.now)
            external_queue_seconds += float(self.base.external_queue_uav_seconds) - queue_counter_before
            for request_id, request_time in self.base.request_times.items():
                resolved_at = self.base.resolution_times.get(request_id)
                right = tick_end if resolved_at is None else min(tick_end, float(resolved_at))
                left = max(tick_start, float(request_time))
                if right > left:
                    unresolved_request_seconds += right - left
            actual_radius_sq = sum(float(self.base.airspace.actual_ring(role).radius) ** 2 for role in RingRole)
            footprint_seconds += actual_radius_sq / self._initial_radius_sq * (tick_end - tick_start)
            self._observe_energy(previous_refs)
            self._update_low_demand_state()
            low_reward+=float(sum(rewards.values()))
            if self._done() or self.base.now>=self.config.max_simulation_seconds: break
        completed=sum(outcome is TerminalOutcome.COMPLETED for outcome in self.base.ledger.outcomes.values()); energy=self._energy_total()
        shield_cost=shield.repair_magnitude+.1*shield.projection_magnitude+(.5 if shield.outcome in {"delayed","rejected"} else 0.)
        reward=HighLevelReward(service_completion=8.*(completed-before_completed),external_queue=-.05*external_queue_seconds,delay=-.002*unresolved_request_seconds,safety=-2.*(self.base.telemetry["safety_interventions"]-before["safety_interventions"])-25.*(self.base.telemetry["collisions"]-before["collisions"])-shield_cost,energy=-.0005*max(0.,energy-before_energy),reconfiguration=-.05*float(update.structural_applied),footprint=-.0001*footprint_seconds)
        self._previous_completed,self._previous_energy=completed,energy
        raw_cost_integrals={"external_queue_uav_seconds":float(external_queue_seconds),"unresolved_request_seconds":float(unresolved_request_seconds),"footprint_seconds":float(footprint_seconds)}
        record={"time":start_time,"requested_action":list(shield.requested_action),"applied_action":list(shield.applied_action),"structural_mask":list(update.structural_mask),"geometry_gate":gate,"geometry_gate_reason":gate_reason,"actual_geometry":self.base.airspace.state_vector()[:6].tolist(),"target_geometry":[self.base.airspace.target_ring(role).radius for role in RingRole]+[self.base.airspace.target_ring(role).height for role in RingRole],"capacity":self._context()["capacity"],"headways":dict(self.base.airspace.headways),"transition_active":self.base.airspace.transition is not None,"shield_outcome":shield.outcome,"execution_outcome":shield.execution_outcome,"request_id":shield.request_id,"original_request_time":shield.original_request_time,"actual_application_time":shield.actual_application_time,"application_delay_seconds":shield.application_delay_seconds,"operational_effective":shield.operational_effective,"structural_effective":shield.structural_effective,"capacity_effective":shield.capacity_effective,"shield_reason_code":shield.reason_code,"traffic_repair":shield.repair_speed_residuals,"traffic_repair_magnitude":shield.repair_magnitude,"geometry_projection_magnitude":shield.projection_magnitude,"deferred_until":shield.deferred_until,"safety_reason":shield.safety_reason,"requested_radius_delta":list(update.requested_radius_delta),"projected_radius_delta":list(update.projected_radius_delta),"applied_radius_delta":list(update.applied_radius_delta),"capacity_before":list(update.capacity_before),"capacity_after":list(update.capacity_after),"structural_reason_codes":list(update.structural_reason_codes),"high_policy_decisions":len(self.action_history)+1,"coordinator_attempts":self.coordinator.telemetry["coordinator_attempts"],"pending_retries":self.coordinator.telemetry["pending_retries"],"effective_updates":self.coordinator.telemetry["effective_updates"],"pending_state":self.coordinator.pending_state.value,"reward_schema":"high_reward_v2_time_integrated","footprint_definition":"normalized_sum_ring_disk_area_proxy","raw_cost_integrals":raw_cost_integrals}
        self.action_history.append(record); terminated=self._done(); truncated=self.base.now>=self.config.max_simulation_seconds and not terminated
        info={"reward_components":reward.as_dict(),"low_level_reward":low_reward,"high_action":record,"context":self._context(),"telemetry":dict(self.base.telemetry)|dict(self.coordinator.telemetry),"outcomes":dict(self.base.ledger.outcomes),"conflict_events":list(self.base.conflict_events),"low_level_checkpoint_hash":getattr(self.low_level,"checkpoint_hash",None),"latest_low_info":latest_info,"reward_schema":"high_reward_v2_time_integrated","footprint_definition":"normalized_sum_ring_disk_area_proxy","raw_cost_integrals":raw_cost_integrals}
        return self.observation(),reward.total,terminated,truncated,info
