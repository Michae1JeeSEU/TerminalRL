"""Runnable, no-training wrapper for the official TUD_RL UAM validation code.

The physics, observation construction, history update, and recurrent actor are
loaded from the pinned ``upstream_dev`` snapshot.  This module only supplies
scenario scheduling, strict actor loading, and auditable output files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream_dev"
UPSTREAM_COMMIT = "2395f35438571e274dcb233b4cdf4f7d692ff798"
OPENAP_DEFAULT = ROOT / "assets" / "openap_legacy" / "bluesky" / "resources" / "performance" / "OpenAP"


@dataclass(frozen=True)
class Arrival:
    scheduled_t: int
    gate: int
    noise: bool
    cluster: Optional[int] = None


@dataclass(frozen=True)
class ScenarioPlan:
    name: str
    target_n: int
    arrivals: tuple[Arrival, ...]
    cluster_counts: tuple[int, ...] = ()


def _namespace_package(name: str, path: Path) -> types.ModuleType:
    """Install a real namespace package without executing upstream scanners."""
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = [str(path)]
    return module


_UPSTREAM_CACHE: Optional[dict[str, Any]] = None


def load_upstream() -> dict[str, Any]:
    """Load only the UAM modules needed by this wrapper.

    ``tud_rl.envs`` and ``tud_rl.agents`` in the official tree perform broad
    import scans.  Namespace package shims keep the package path while avoiding
    imports of unrelated environments and agents.
    """
    global _UPSTREAM_CACHE
    if _UPSTREAM_CACHE is not None:
        return _UPSTREAM_CACHE
    if not UPSTREAM.is_dir():
        raise FileNotFoundError(f"missing upstream snapshot: {UPSTREAM}")
    root = str(UPSTREAM)
    if root not in sys.path:
        sys.path.insert(0, root)

    openap_path = Path(os.environ.get("TUD_RL_OPENAP_PATH", str(OPENAP_DEFAULT))).expanduser().resolve()
    required_openap = (
        openap_path / "synonym.dat",
        openap_path / "rotor" / "aircraft.json",
        openap_path / "fixwing" / "aircraft.json",
        openap_path / "fixwing" / "engines.csv",
        openap_path / "fixwing" / "dragpolar.csv",
    )
    missing_openap = [str(p) for p in required_openap if not p.is_file()]
    if missing_openap:
        raise FileNotFoundError(
            "OpenAP legacy assets are incomplete; set TUD_RL_OPENAP_PATH to a "
            f"full performance/OpenAP directory. Missing: {missing_openap}"
        )
    import bluesky as bs

    bs.settings.perf_path_openap = str(openap_path)
    import tud_rl  # noqa: F401  # normal logger and registration initialization

    env_path = UPSTREAM / "tud_rl" / "envs"
    agents_path = UPSTREAM / "tud_rl" / "agents"
    env_pkg = _namespace_package("tud_rl.envs", env_path)
    agent_pkg = _namespace_package("tud_rl.agents", agents_path)
    sys.modules["tud_rl.envs"] = env_pkg
    sys.modules["tud_rl.agents"] = agent_pkg
    setattr(sys.modules["tud_rl"], "envs", env_pkg)
    setattr(sys.modules["tud_rl"], "agents", agent_pkg)

    val_mod = importlib.import_module("tud_rl.envs._envs.UAM_Modular_Validation")
    mod_mod = importlib.import_module("tud_rl.envs._envs.UAM_Modular")
    nets_mod = importlib.import_module("tud_rl.common.nets")
    _UPSTREAM_CACHE = {
        "validation": val_mod.UAM_Modular_Validation,
        "base": mod_mod.UAM_Modular,
        "gate": val_mod.Gate,
        "nets": nets_mod,
        "ed": mod_mod.ED,
        "bng_rel": mod_mod.bng_rel,
        "dtr": mod_mod.dtr,
        "angle_to_pi": mod_mod.angle_to_pi,
        "cpa": mod_mod.cpa,
        "commit": UPSTREAM_COMMIT,
    }
    return _UPSTREAM_CACHE


def build_plan(
    scenario: str,
    n: int = 20,
    poisson_seed: int = 43,
) -> ScenarioPlan:
    """Build deterministic arrival schedules without touching global RNG state."""
    if scenario == "wave":
        arrivals = tuple(
            Arrival(t, gate, False)
            for t in (0, 30, 60)
            for gate in range(4)
        )
        return ScenarioPlan("wave", 12, arrivals)
    if scenario == "study":
        n = int(n)
        if n < 1:
            raise ValueError("study n must be at least 1")
        # The source samples gates at run time; times and the target count are
        # recorded here for output accounting.
        arrivals = tuple(Arrival(0 if i == 0 else 15 * i, -1, True) for i in range(n))
        return ScenarioPlan("study", n, arrivals)
    if scenario == "poisson":
        rng = np.random.default_rng(int(poisson_seed))
        arrivals: list[Arrival] = []
        counts: list[int] = []
        for cluster, base in enumerate((0, 120, 240, 360)):
            count = int(rng.poisson(5))
            counts.append(count)
            for k in range(count):
                arrivals.append(Arrival(base + 10 * k, 2, True, cluster))
        arrivals.sort(key=lambda a: (a.scheduled_t, a.cluster if a.cluster is not None else -1))
        return ScenarioPlan("poisson", len(arrivals), tuple(arrivals), tuple(counts))
    raise ValueError(f"unknown scenario: {scenario}")


class ZeroTurnPolicy:
    """Explicit diagnostic policy.  It is never a paper-policy evaluation."""

    policy_verified = False
    status = "runtime_diagnostic"

    def select_action(self, s, s_hist, a_hist, hist_len):  # noqa: D401, ANN001
        return np.zeros(1, dtype=np.float32)


class StrictActorPolicy:
    """The official LSTMRecActor loaded from an actor-only state dictionary."""

    policy_verified = True
    status = "checkpoint_evaluation"

    def __init__(self, actor_path: Path):
        import torch

        mods = load_upstream()
        self.actor_path = Path(actor_path).expanduser().resolve()
        if not self.actor_path.is_file():
            raise FileNotFoundError(f"actor checkpoint does not exist: {self.actor_path}")
        self.actor = mods["nets"].LSTMRecActor(
            1, 3, 6, use_past_actions=False, device="cpu"
        )
        payload = torch.load(
            self.actor_path, map_location="cpu", weights_only=True
        )
        state = payload
        if isinstance(payload, Mapping) and "state_dict" in payload:
            state = payload["state_dict"]
        if not isinstance(state, Mapping) or not state:
            raise TypeError(
                "actor checkpoint must be a non-empty bare state_dict or "
                "a {'state_dict': ...} container"
            )
        self.actor.load_state_dict(state, strict=True)
        self.actor.eval()
        self._torch = torch

    def select_action(self, s, s_hist, a_hist, hist_len):  # noqa: D401, ANN001
        torch = self._torch
        state = np.array(s, dtype=np.float32, copy=True)
        history = np.array(s_hist, dtype=np.float32, copy=True)
        current = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        past = torch.tensor(history, dtype=torch.float32).reshape(1, history.shape[0], -1)
        length = torch.tensor([int(hist_len)], dtype=torch.long)
        with torch.inference_mode():
            action, _ = self.actor(current, past, None, length)
        return action.detach().cpu().numpy()[0].astype(np.float32, copy=False)


def _stable_uid(index: int) -> str:
    return f"uav-{index:04d}"


class LocalValidationEnv:
    """Validation wrapper preserving upstream dynamics and state semantics."""

    def __init__(
        self,
        plan: ScenarioPlan,
        seed: int,
        noise: float = 0.0,
        entry_check: bool = False,
        profile: str = "upstream_dev1000",
    ):
        mods = load_upstream()
        self.plan = plan
        self.seed = int(seed)
        self.profile = profile
        self.distance_norm_m = 1000.0
        self._next_arrival = 0
        self._study_next_t = 15
        self._blocked_attempts = 0
        self._spawn_events: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._trajectory: list[dict[str, Any]] = []
        self._entered_count = 0
        self._outside_count = 0
        self._unauthorized_count = 0
        self._accident_pair_steps = 0
        self._incident_pair_steps = 0
        self._min_separation = None
        validation = mods["validation"]
        sim_study = plan.name in ("study", "poisson")
        super_obj = validation.__new__(validation)
        # Calling the official constructor is intentional: it creates the
        # OpenAP performance model, spaces, destination, and constants.
        validation.__init__(
            super_obj,
            situation=1,
            sim_study=sim_study,
            sim_study_N=max(2, plan.target_n) if sim_study else None,
            safe_number=self.seed,
            entry_check=bool(entry_check),
            noise=float(noise),
        )
        self._impl = super_obj
        self.__dict__.update(super_obj.__dict__)
        self._source_validation_type = validation
        self.entry_check = bool(entry_check)
        self.noise = float(noise)
        self.distance_norm_m = 1000.0
        if profile == "paper800":
            self.dest.spawn_radius = 800.0
            self.dest.respawn_radius = 1200.0
            self.dest.radius = 200.0
            self.dest.restricted_area = 200.0
            Gate = mods["gate"]
            self.gates = [
                Gate(self.dest, "N"),
                Gate(self.dest, "E"),
                Gate(self.dest, "S"),
                Gate(self.dest, "W"),
            ]
        elif profile != "upstream_dev1000":
            raise ValueError(f"unknown profile: {profile}")
        self.reset()

    # Delegate the unchanged source operations through the wrapper instance.
    # Keeping these calls here makes the local scheduler the only code that
    # changes episode timing and collection handling.
    def _high_level_control(self):
        return self._source_validation_type._high_level_control(self)

    def _add_noise(self):
        return self._source_validation_type._add_noise(self)

    def _set_state(self):
        return self._source_validation_type._set_state(self)

    def _calculate_reward(self, a0: float):
        return self._source_validation_type._calculate_reward(self, a0)

    def _get_state(self, i: int) -> np.ndarray:
        mods = load_upstream()
        p = self.planes[i]
        pn = getattr(p, "noisy_n", p.n)
        pe = getattr(p, "noisy_e", p.e)
        rel_bng_goal = mods["bng_rel"](
            N0=pn,
            E0=pe,
            N1=self.dest.N,
            E1=self.dest.E,
            head0=mods["dtr"](p.hdg),
            to_2pi=False,
        ) / np.pi
        d_goal = mods["ed"](
            N0=pn, E0=pe, N1=self.dest.N, E1=self.dest.E
        ) / self.distance_norm_m
        s_i = np.array([rel_bng_goal, d_goal, p.fly_to_goal])
        ts_info = []
        for j, other in enumerate(self.planes):
            if i == j:
                continue
            other_n = getattr(other, "noisy_n", other.n)
            other_e = getattr(other, "noisy_e", other.e)
            v_r = (other.tas - p.tas) / (2 * self.delta_tas)
            bng = mods["bng_rel"](
                N0=pn,
                E0=pe,
                N1=other_n,
                E1=other_e,
                head0=mods["dtr"](p.hdg),
                to_2pi=False,
            ) / np.pi
            d = mods["ed"](N0=pn, E0=pe, N1=other_n, E1=other_e) / self.distance_norm_m
            c_t = mods["angle_to_pi"](mods["dtr"](other.hdg - p.hdg)) / np.pi
            dcpa, tcpa = mods["cpa"](
                NOS=pn,
                EOS=pe,
                NTS=other_n,
                ETS=other_e,
                chiOS=np.radians(p.hdg),
                chiTS=mods["dtr"](other.hdg),
                VOS=p.tas,
                VTS=other.tas,
            )
            ts_info.append([d, bng, v_r, c_t, dcpa / 100.0, tcpa / 60.0])
        if not ts_info:
            ts_info.append([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
        ts_info = np.hstack(sorted(ts_info, key=lambda x: x[0], reverse=True)).astype(np.float32)
        desired_length = self.obs_per_TS * (self.N_agents_max - 1) + self.obs_per_TS * self.N_cutters_max
        ts_info = np.pad(ts_info, (0, desired_length - len(ts_info)), "constant", constant_values=np.nan).astype(np.float32)
        return np.concatenate((s_i, ts_info))

    def _reset_trackers(self):
        self._next_arrival = 0
        self._study_next_t = 15
        self._blocked_attempts = 0
        self._spawn_events = []
        self._events = []
        self._trajectory = []
        self._entered_count = 0
        self._outside_count = 0
        self._unauthorized_count = 0
        self._accident_pair_steps = 0
        self._incident_pair_steps = 0
        self._min_separation = None
        self._spawned_count = 0
        self._unspawned_count = 0

    def reset(self):
        self._reset_trackers()
        self.step_cnt = 0
        self.sim_t = 0
        self.planes = []
        self.dest.reset()
        self.N_planes = 0
        self.id_counter = 0
        self.unique_ids = [_stable_uid(i) for i in range(max(1, self.plan.target_n))]

        if self.plan.name == "poisson":
            self._spawn_due(initial=True)
        elif self.plan.name == "wave":
            for gate in range(4):
                self._spawn_one(gate, noise=False, scheduled_t=0, cluster=None)
            self._next_arrival = 4
        else:
            self._spawn_one(
                int(np.random.choice(4)), noise=True, scheduled_t=0, cluster=None
            )
            self._next_arrival = 1
        self._high_level_control()
        self._add_noise()
        self._set_state()
        self.state_init = self.state
        for event in self._events:
            event.setdefault("destination_open_before", bool(self.dest.is_open))
            event.setdefault("destination_open_after", bool(self.dest.is_open))
        self._record_trajectory()
        return self.state

    def _spawn_one(
        self,
        gate: int,
        noise: bool,
        scheduled_t: int,
        cluster: Optional[int],
    ) -> Any:
        p = self._source_validation_type._spawn_plane(self, gate=gate, noise=noise)
        p.id = self.unique_ids[self.id_counter]
        self.planes.append(p)
        self.id_counter += 1
        self._spawned_count += 1
        self.N_planes = len(self.planes)
        data = {
            "time": int(self.sim_t),
            "uid": p.id,
            "event_type": "spawn",
            "scheduled_time": int(scheduled_t),
            "gate": int(gate),
            "cluster": cluster,
            "entry_open": False,
            "entry_closed": False,
            "signaled": False,
            "destination_open_before": bool(self.dest.is_open),
            "destination_open_after": bool(self.dest.is_open),
            "authorized_vtol_entry": False,
            "outside_respawn_removal": False,
            "outcome": "active",
        }
        self._spawn_events.append(data)
        self._events.append(dict(data))
        return p

    def _gate_is_free(self, gate):
        ed = load_upstream()["ed"]
        return all(ed(N0=p.n, E0=p.e, N1=gate.N, E1=gate.E) >= 300 for p in self.planes)

    def _select_study_gate(self) -> Optional[int]:
        gates = [0, 1, 2, 3]
        if not self.entry_check:
            return int(np.random.choice(4))
        while gates:
            gate = int(np.random.choice(gates))
            if self._gate_is_free(self.gates[gate]):
                return gate
            gates.remove(gate)
        return None

    def _spawn_due(self, initial: bool = False):
        if self.plan.name == "poisson":
            while self._next_arrival < len(self.plan.arrivals):
                arrival = self.plan.arrivals[self._next_arrival]
                if arrival.scheduled_t > self.sim_t:
                    break
                if self.entry_check and not self._gate_is_free(self.gates[arrival.gate]):
                    self._blocked_attempts += 1
                    self._events.append({
                        "time": int(self.sim_t), "uid": "", "event_type": "unspawned",
                        "scheduled_time": int(arrival.scheduled_t), "gate": int(arrival.gate),
                        "cluster": arrival.cluster, "entry_open": False,
                        "entry_closed": False, "signaled": False,
                        "authorized_vtol_entry": False, "outside_respawn_removal": False,
                        "outcome": "gate_occupied",
                    })
                    # Keep the request pending.  The eventual spawn event
                    # retains scheduled_time, making any delay auditable.
                    break
                self._next_arrival += 1
                self._spawn_one(arrival.gate, arrival.noise, arrival.scheduled_t, arrival.cluster)
            return
        if self.plan.name == "wave":
            while self._next_arrival < len(self.plan.arrivals):
                arrival = self.plan.arrivals[self._next_arrival]
                if arrival.scheduled_t > self.sim_t:
                    break
                self._next_arrival += 1
                if arrival.scheduled_t == 0 and not initial:
                    continue
                self._spawn_one(arrival.gate, False, arrival.scheduled_t, None)
            return
        if self.sim_t >= self._study_next_t and self.id_counter < self.plan.target_n:
            if self.sim_t % 15 == 0:
                gate = self._select_study_gate()
                if gate is None:
                    self._blocked_attempts += 1
                else:
                    self._spawn_one(gate, True, self.sim_t, None)
                self._study_next_t += 15

    def _record_trajectory(self, removed: Optional[Iterable[Any]] = None):
        for p in self.planes:
            self._trajectory.append({
                "time": int(self.sim_t), "uid": p.id,
                "x": float(p.e), "y": float(p.n), "heading": float(p.hdg),
                "speed": float(p.tas), "sigma": float(getattr(p, "fly_to_goal", 0.0)),
                "groundtruth_n": float(p.n), "groundtruth_e": float(p.e),
                "active": True,
            })
        for p in (removed or []):
            self._trajectory.append({
                "time": int(self.sim_t), "uid": p.id,
                "x": float(p.e), "y": float(p.n), "heading": float(p.hdg),
                "speed": float(p.tas), "sigma": float(getattr(p, "fly_to_goal", 0.0)),
                "groundtruth_n": float(p.n), "groundtruth_e": float(p.e),
                "active": False,
            })

    def _pair_metrics(self):
        ed = load_upstream()["ed"]
        for i, pi in enumerate(self.planes):
            for pj in self.planes[i + 1 :]:
                d = float(ed(N0=pi.n, E0=pi.e, N1=pj.n, E1=pj.e))
                self._min_separation = d if self._min_separation is None else min(self._min_separation, d)
                if d <= self.accident_dist:
                    self._accident_pair_steps += 1
                if d <= self.incident_dist:
                    self._incident_pair_steps += 1

    def _complete(self):
        if self.plan.name == "study":
            schedule_done = self.id_counter >= self.plan.target_n
        else:
            schedule_done = self._next_arrival >= len(self.plan.arrivals)
        return bool(schedule_done and len(self.planes) == 0)

    def step(self, policy):
        mods = load_upstream()
        self.step_cnt += 1
        self.sim_t += self.dt
        event_start = len(self._events)
        a0 = 0.0
        for i, p in enumerate(list(self.planes)):
            if p.role == "RL":
                act = np.asarray(policy.select_action(p.s, p.s_hist, None, p.hist_len), dtype=np.float32)
                p.upd_dynamics(a=act, discrete_acts=False, perf=self.perf, dest=None)
                if i == 0:
                    a0 = float(act[0])
            else:
                p.upd_dynamics(perf=self.perf, dest=None)
        for p in self.planes:
            p.n, p.e, _ = importlib.import_module("tud_rl.envs._envs.HHOS_Fnc").to_utm(lat=p.lat, lon=p.lon)
            p.D_dest_old = float(p.D_dest)
            p.D_dest = float(mods["ed"](N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e))
        self._pair_metrics()
        open_before = bool(self.dest.is_open)
        entered_close, entered_open, just_opened = self.dest.step(self.planes)
        if self.planes:
            self._calculate_reward(a0)
        else:
            self.r = np.zeros((0, 1), dtype=np.float32)
        removed: list[Any] = []
        survivors: list[Any] = []
        for i, p in enumerate(self.planes):
            signaled = p.fly_to_goal == 1.0
            authorized = bool(entered_open[i] and signaled)
            outside = bool((not authorized) and p.D_dest >= self.dest.respawn_radius)
            zone_entry = bool(entered_open[i] or entered_close[i])
            if zone_entry and not authorized:
                self._unauthorized_count += 1
            if authorized or outside:
                removed.append(p)
                if authorized:
                    self._entered_count += 1
                if outside:
                    self._outside_count += 1
                self._events.append({
                    "time": int(self.sim_t), "uid": p.id, "event_type": "removal",
                    "scheduled_time": None, "gate": None, "cluster": None,
                    "entry_open": bool(entered_open[i]), "entry_closed": bool(entered_close[i]),
                    "signaled": bool(signaled), "authorized_vtol_entry": authorized,
                    "outside_respawn_removal": outside,
                    "outcome": "entered_vtol" if authorized else "outside_respawn",
                })
            else:
                survivors.append(p)
                if zone_entry:
                    self._events.append({
                        "time": int(self.sim_t), "uid": p.id, "event_type": "entry_observation",
                        "scheduled_time": None, "gate": None, "cluster": None,
                        "entry_open": bool(entered_open[i]), "entry_closed": bool(entered_close[i]),
                        "signaled": bool(signaled), "authorized_vtol_entry": False,
                        "outside_respawn_removal": False,
                        "outcome": "unauthorized_entry" if entered_open[i] else "closed_entry",
                    })
        self.planes = survivors
        self.N_planes = len(self.planes)
        self._spawn_due()
        self.N_planes = len(self.planes)
        if just_opened:
            self._high_level_control()
        self._add_noise()
        self._set_state()
        self._record_trajectory(removed)
        self._unspawned_count = max(0, self.plan.target_n - self._spawned_count)
        for event in self._events[event_start:]:
            event.setdefault("destination_open_before", open_before)
            event.setdefault("destination_open_after", bool(self.dest.is_open))
        done = bool(self._complete() or self.step_cnt >= self._max_episode_steps)
        return self.state, 0.0, done, {
            "destination_open_before": open_before,
            "destination_open_after": bool(self.dest.is_open),
            "just_opened": bool(just_opened),
            "a0": a0,
        }


def _runtime_versions() -> dict[str, Optional[str]]:
    result: dict[str, Optional[str]] = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "torch", "gym", "bluesky"):
        try:
            mod = importlib.import_module(name)
            result[name] = getattr(mod, "__version__", None)
        except Exception:
            result[name] = None
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_chart(env: LocalValidationEnv, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in env._trajectory:
        by_uid.setdefault(str(row["uid"]), []).append(row)
    for uid, rows in by_uid.items():
        ax.plot([r["x"] for r in rows], [r["y"] for r in rows], linewidth=0.8, label=uid)
    ax.scatter([env.dest.E], [env.dest.N], marker="x", color="black", label="destination")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title(f"TUD_RL UAM {env.plan.name}")
    if len(by_uid) <= 12:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _output_path(requested: Optional[str], scenario: str) -> Path:
    if requested:
        out = Path(requested).expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path.home() / "training_outputs" / "terminal" / "waltz2024" / f"{stamp}_{scenario}"
    project_root = ROOT.parents[1]
    forbidden = [project_root, Path.cwd().resolve(), Path.home() / "projects" / "terminal"]
    if any(out == p.resolve() or p.resolve() in out.parents for p in forbidden):
        raise ValueError("output must be outside the project source directories")
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_one(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    plan = build_plan(args.scenario, args.n, args.poisson_seed)
    env = LocalValidationEnv(
        plan=plan,
        seed=args.seed,
        noise=args.noise,
        entry_check=args.entry_check,
        profile=args.profile,
    )
    policy = ZeroTurnPolicy() if args.diagnostic_zero else StrictActorPolicy(Path(args.actor))
    cap = args.max_steps
    if cap is None:
        cap = 2000 if args.scenario == "wave" else 100000
    steps = 0
    done = False
    while not done and steps < cap:
        _, _, done, _ = env.step(policy)
        steps += 1
    complete = bool(env._complete())
    truncated = bool(not complete)
    trajectories = output / "trajectories.csv"
    events = output / "events.csv"
    _write_csv(
        trajectories,
        env._trajectory,
        ["time", "uid", "x", "y", "heading", "speed", "sigma", "groundtruth_n", "groundtruth_e", "active"],
    )
    _write_csv(
        events,
        env._events,
        ["time", "uid", "event_type", "scheduled_time", "gate", "cluster", "entry_open", "entry_closed", "signaled", "destination_open_before", "destination_open_after", "authorized_vtol_entry", "outside_respawn_removal", "outcome"],
    )
    if args.chart:
        _write_chart(env, output / "trajectory.png")
    summary = {
        "status": policy.status,
        "policy_verified": bool(policy.policy_verified),
        "scenario": args.scenario,
        "profile": args.profile,
        "seed": int(args.seed),
        "poisson_seed": int(args.poisson_seed),
        "noise_m": float(args.noise),
        "entry_check": bool(args.entry_check),
        "source_commit": UPSTREAM_COMMIT,
        "actor_path": str(policy.actor_path) if isinstance(policy, StrictActorPolicy) else None,
        "steps": int(steps),
        "sim_time_s": int(env.sim_t),
        "max_steps": int(cap),
        "complete": complete,
        "truncated": truncated,
        "spawned": int(env._spawned_count),
        "entered_vtol": int(env._entered_count),
        "outside_respawn": int(env._outside_count),
        "requested": int(env.plan.target_n),
        "unspawned": int(env._unspawned_count),
        "blocked_attempts": int(env._blocked_attempts),
        "active": int(len(env.planes)),
        "final_service_timer_s": float(env.dest.t_nxt_open),
        "dt_s": float(env.dt),
        "turn_limit_deg": 5.0,
        "landing_closure_s": float(env.dest.t_close),
        "physical_spawn_radius_m": float(env.dest.spawn_radius),
        "v_tol_radius_m": float(env.dest.radius),
        "respawn_radius_m": float(env.dest.respawn_radius),
        "distance_norm_m": float(env.distance_norm_m),
        "min_groundtruth_separation_m": env._min_separation,
        "accident_pair_step_count": int(env._accident_pair_steps),
        "incident_pair_step_count": int(env._incident_pair_steps),
        "unauthorized_entry_observations": int(env._unauthorized_count),
        "conservation": {
            "requested_eq_spawned_plus_unspawned": bool(env.plan.target_n == env._spawned_count + env._unspawned_count),
            "spawned_eq_entered_plus_outside_plus_active": bool(env._spawned_count == env._entered_count + env._outside_count + len(env.planes)),
        },
        "evidence_type": "newly_run",
        "checkpoint_provenance_verified": False,
        "official_policy_claim": False,
        "runtime_packages": _runtime_versions(),
        "command": " ".join(sys.argv),
        "output_sha256": {},
    }
    for p in (trajectories, events):
        summary["output_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, allow_nan=False)
    return summary


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run official TUD_RL UAM scenarios without training.")
    p.add_argument("--scenario", choices=("wave", "study", "poisson"), default="wave")
    p.add_argument("--n", type=int, default=20, help="study aircraft target count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--poisson-seed", type=int, default=43)
    p.add_argument("--noise", type=float, choices=(0.0, 10.0, 20.0, 100.0), default=0.0)
    p.add_argument("--entry-check", action="store_true")
    p.add_argument("--profile", choices=("upstream_dev1000", "paper800"), default="upstream_dev1000")
    p.add_argument("--actor", help="path to an official actor state_dict")
    p.add_argument("--diagnostic-zero", action="store_true", help="explicit zero-turn runtime diagnostic")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--output")
    p.add_argument("--chart", action="store_true")
    p.add_argument("--suite", action="store_true", help="run wave, study, and poisson in separate directories")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if bool(args.actor) == bool(args.diagnostic_zero):
        parser.error("specify exactly one of --actor or --diagnostic-zero")
    if args.actor and not Path(args.actor).expanduser().is_file():
        parser.error(f"actor checkpoint does not exist: {Path(args.actor).expanduser()}")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.n < 1:
        parser.error("--n must be positive")
    if args.suite:
        root = _output_path(args.output, "suite")
        for scenario in ("wave", "study", "poisson"):
            child = argparse.Namespace(**vars(args))
            child.scenario = scenario
            run_one(child, root / scenario)
        return 0
    out = _output_path(args.output, args.scenario)
    run_one(args, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
