"""Explicit method adaptations for the common disk, never native-paper scores."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import math
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class MissingCheckpointError(FileNotFoundError):
    pass


class RulePolicy:
    name = "fcfs_rule"

    def __init__(self, config, seed=0):
        self.config, self.seed = config, seed
        self.calls, self.solver_records = [], []
        self.last_time, self.last_ids, self.last_grant = -math.inf, (), None
        self.turns = {}
        self.metadata = {"policy": self.name, "execution_label": "adapted_common",
                         "future_arrivals_visible": False, "learning": False,
                         "decision_period": config.decision_period}

    def select_grant(self, env):
        a = min(env.available_to_controller().values(), key=lambda a: (a.request.request_time, a.uid))
        return a.uid, env.now, None

    def calculate_turns(self, env):
        return {uid: env.guidance_turn(a) for uid, a in env.available_to_controller().items()}

    def act(self, env):
        ids = tuple(sorted(env.available_to_controller()))
        if not ids:
            return {}
        needs_grant = env.pad_uid is None and env.grant_uid is None
        if (env.now - self.last_time >= self.config.decision_period - 1e-8
                or ids != self.last_ids or env.grant_uid != self.last_grant or needs_grant):
            start = time.perf_counter()
            if needs_grant:
                uid, not_before, touchdown = self.select_grant(env)
                env.set_grant(uid, not_before=not_before, planned_touchdown=touchdown)
            self.turns = self.calculate_turns(env)
            elapsed = time.perf_counter() - start
            late = elapsed > self.config.planner_deadline
            if late:
                self.turns = RulePolicy.calculate_turns(self, env)
            self.calls.append({"time": env.now, "visible_uids": list(ids), "seconds": elapsed,
                               "deadline_fallback": late})
            self.last_time, self.last_ids, self.last_grant = env.now, ids, env.grant_uid
        return dict(self.turns)


def _load_yang_search(config):
    root = ROOT / "baselines/yang2020_mcts/upstream"
    paths = {"config_vertiport": root / "Simulators/config_vertiport.py",
             "common": root / "MCTS/common.py", "nodes_multi": root / "MCTS/nodes_multi.py",
             "search_multi": root / "MCTS/search_multi.py"}
    saved = {key: sys.modules.get(key) for key in paths}
    loaded, hashes = {}, {}
    try:
        for key, path in paths.items():
            raw = path.read_text(encoding="utf-8")
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            if key == "nodes_multi":
                old = "if self.state.init_action == 'random':"
                if raw.count(old) != 1:
                    raise ValueError("upstream NumPy compatibility patch no longer matches")
                raw = raw.replace(old, "if isinstance(self.state.init_action, str) and self.state.init_action == 'random':")
            module = ModuleType(key)
            module.__file__ = str(path)
            sys.modules[key] = module
            with redirect_stdout(io.StringIO()):
                exec(compile(raw, str(path), "exec"), module.__dict__)
            loaded[key] = module
        cfg = loaded["config_vertiport"].Config
        cfg.scale = 1.0
        cfg.window_width = cfg.window_height = 2 * config.radius + 100
        cfg.minimum_separation = config.safety_distance
        cfg.goal_radius = 40.0
        cfg.min_speed = cfg.max_speed = config.cruise_speed
        cfg.speed_sigma = 0.0
        cfg.d_heading = math.radians(config.max_turn_deg_s)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return loaded, hashes


class YangMCTSPolicy(RulePolicy):
    name = "yang_mcts_adapted"

    def __init__(self, config, seed=0):
        super().__init__(config, seed)
        self.modules, hashes = _load_yang_search(config)
        self.rng_state = np.random.RandomState(seed).get_state()
        self.metadata.update(source_hashes=hashes, simulations=config.mcts_simulations,
                             search_depth=config.mcts_depth,
                             adaptations=["SI dynamics constants", "common decision cadence",
                                          "FCFS pad grant and soft holding target", "V3 execution"])

    def calculate_turns(self, env):
        ids = sorted(env.available_to_controller())
        offset = self.config.radius + 50
        rows = []
        for uid in ids:
            a = env.arrivals[uid]
            goal = env.reference_goal(a)
            rows.append([*(a.position[:2] + offset), *a.velocity[:2], a.speed,
                         a.heading, *(goal + offset)])
        state = np.asarray(rows)
        actions = np.ones(len(ids), dtype=np.int32)
        saved = np.random.get_state()
        try:
            np.random.set_state(self.rng_state)
            for index in range(len(ids)):
                node_state = self.modules["nodes_multi"].MultiAircraftState(state, index, actions.copy())
                node = self.modules["nodes_multi"].MultiAircraftNode(node_state)
                selected = self.modules["search_multi"].MCTS(node).best_action(
                    self.config.mcts_simulations, self.config.mcts_depth)
                actions[index] = selected.state.prev_action[index]
            self.rng_state = np.random.get_state()
        finally:
            np.random.set_state(saved)
        return {uid: (int(actions[i]) - 1) * math.radians(self.config.max_turn_deg_s)
                for i, uid in enumerate(ids)}


class SaxenaMILPPolicy(RulePolicy):
    name = "saxena_milp_adapted"

    def __init__(self, config, seed=0):
        super().__init__(config, seed)
        from baselines.saxena2024_schedule.model import ScheduleModel, build_flight_set
        self.model_class, self.build_set = ScheduleModel, build_flight_set
        self.metadata.update(implementation="Python/HiGHS source port; MATLAB parity unverified",
                             adaptations=["causal arrival-only replanning", "arrival equality to earliest time",
                                          "physical ETA and common pad occupancy", "shared heading guidance"])

    def build_model(self, env):
        aircraft = env.available_to_controller()
        requests = [{"id": str(uid), "type": "arr", "class": "Small", "gate": "G1",
                     "direction": ("N", "E", "S", "W")[a.request.gate], "arr_req_time": 0.0}
                    for uid, a in sorted(aircraft.items())]
        model = self.model_class(self.build_set(requests, preprocess=False), formulation="paper")
        model.build()
        for row in model.builder.rows:
            if row.family == "ARAPR_i":
                row.upper = math.inf
        descent = max(40.0, 2 * self.config.capture_radius / self.config.cruise_speed,
                      2 * self.config.altitude / 5.0)
        for uid, a in aircraft.items():
            eta = max(0.0, np.linalg.norm(a.position[:2]) - self.config.capture_radius) / self.config.cruise_speed + descent
            model.builder.row(f"common_eta[{uid}]", "common_eta", {model.t_vars[(str(uid), "R2")]: 1}, lower=eta)
        big_m = model.params.big_m(len(aircraft))
        for uid, a in aircraft.items():
            for other in aircraft:
                if uid == other:
                    continue
                coeffs = {model.t_vars[(str(other), "R2")]: 1, model.t_vars[(str(uid), "R2")]: -1,
                          model.y_vars[("R2", str(uid), str(other))]: -big_m}
                model.builder.row(f"common_pad[{uid},{other}]", "common_pad", coeffs,
                                  lower=a.request.service_seconds + descent - big_m)
        return model, descent

    def select_grant(self, env):
        start = time.perf_counter()
        model, descent = self.build_model(env)
        result = model.solve(time_limit=self.config.milp_time_limit)
        elapsed = time.perf_counter() - start
        violation = max((r["violation"] for r in result.constraint_checks), default=math.inf)
        integral = result.summary.get("integrality_max_abs_error")
        accepted = (bool(result.node_times) and violation <= 1e-6 and integral is not None
                    and integral <= 1e-6 and elapsed <= self.config.planner_deadline)
        self.solver_records.append({"time": env.now, "status": result.status, "accepted": accepted,
                                    "seconds": elapsed, "max_violation": violation,
                                    "objective": result.objective, "gap": result.mip_gap,
                                    "node_times_relative": result.node_times,
                                    "integrality_error": integral})
        if not accepted:
            return super().select_grant(env)
        key = min(result.node_times, key=lambda uid: (result.node_times[uid]["R2"], int(uid)))
        touchdown = env.now + result.node_times[key]["R2"]
        return int(key), max(env.now, touchdown - descent), touchdown


def make_policy(name, config, seed=0, checkpoint=None):
    if name == "waltz_actor_adapted":
        if checkpoint is None or not Path(checkpoint).is_file():
            raise MissingCheckpointError("Waltz trained actor absent; no random/zero policy substitution")
        return WaltzPolicy(config, seed, checkpoint)
    return {"fcfs_rule": RulePolicy, "yang_mcts_adapted": YangMCTSPolicy,
            "saxena_milp_adapted": SaxenaMILPPolicy}[name](config, seed)


class WaltzPolicy(RulePolicy):
    name = "waltz_actor_adapted"

    def __init__(self, config, seed, checkpoint):
        super().__init__(config, seed)
        if config.decision_period != 1:
            raise ValueError("Waltz history requires a one-second decision period")
        from baselines.waltz2024.runner import StrictActorPolicy, load_upstream
        self.actor = StrictActorPolicy(Path(checkpoint))
        self.upstream = load_upstream()
        self.histories = {}
        self.metadata.update(checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                             adaptations=["common disk/dynamics/service", "nearest pad grant", "V3 execution"],
                             original_distance_normalization=1000.0)

    def select_grant(self, env):
        uid = min(env.arrivals, key=lambda uid: (np.linalg.norm(env.arrivals[uid].position[:2]), uid))
        return uid, env.now, None

    def calculate_turns(self, env):
        ids = sorted(env.arrivals)
        planes = [SimpleNamespace(n=env.arrivals[uid].position[1], e=env.arrivals[uid].position[0],
                                  hdg=math.degrees(math.pi / 2 - env.arrivals[uid].heading) % 360,
                                  tas=env.arrivals[uid].speed, fly_to_goal=1 if uid == env.grant_uid else -1)
                  for uid in ids]
        raw = SimpleNamespace(planes=planes, dest=SimpleNamespace(N=0.0, E=0.0, spawn_radius=1000.0),
                              delta_tas=3.0, obs_per_TS=6, N_agents_max=max(2, self.config.requests), N_cutters_max=0)
        turns = {}
        for index, uid in enumerate(ids):
            state = np.asarray(self.upstream["base"]._get_state(raw, index), dtype=np.float32)
            history, length = self.histories.setdefault(uid, (np.zeros((2, len(state)), dtype=np.float32), 0))
            action = np.asarray(self.actor.select_action(state, history, None, length)).reshape(-1)
            if len(action) != 1 or not np.isfinite(action).all():
                raise ValueError("invalid Waltz actor action")
            turns[uid] = -float(np.clip(action[0], -1, 1)) * math.radians(self.config.max_turn_deg_s)
            if length < 2:
                history[length] = state
            else:
                history[:-1] = history[1:]
                history[-1] = state
            self.histories[uid] = history, min(2, length + 1)
        return turns
