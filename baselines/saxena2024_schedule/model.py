"""Transparent SciPy/HiGHS port of the official arrival/departure model.

The source formulation is ``upstream/test_10_vert_schedule_ArrDep.m`` at the
snapshot recorded in :mod:`UPSTREAM.md`.  This module deliberately keeps the
linear rows visible: every generated row has a family and a source-style name
and is exported by ``run.py`` for residual inspection.

This is a new Python port.  It does not claim numerical parity with MATLAB
``solve`` until both solvers have been run on the same input.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


CLASS_NAMES = ("Small", "Medium", "Jumbo", "Super", "Ultra")
CLASS_TO_ID = {name.lower(): i + 1 for i, name in enumerate(CLASS_NAMES)}


class InputError(ValueError):
    """Raised for malformed or unsupported flight requests."""


@dataclass
class Parameters:
    """Physical and formulation values copied from ``test_10``."""

    vehicle_diagonals: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0)
    taxi_speed: float = 6.0
    vertical_climb_speed: float = 8.0
    slant_climb_speed: float = 17.0
    tot: tuple[float, ...] = (2.0, 2.0, 4.0, 4.0, 6.0)
    tat: tuple[float, ...] = (90.0, 120.0, 150.0, 210.0, 300.0)
    different_direction_sep: float = 6.5
    extra_arrival_delay: float = 0.0
    gate_capacity: int = 3
    capacity_nodes: int = 1
    min_time: float = 0.0

    def __post_init__(self) -> None:
        if len(self.vehicle_diagonals) != 5 or len(self.tot) != 5 or len(self.tat) != 5:
            raise InputError("vehicle_diagonals, tot, and tat must each have five classes")
        if self.taxi_speed <= 0 or self.vertical_climb_speed <= 0 or self.slant_climb_speed <= 0:
            raise InputError("all speeds must be positive")
        if self.gate_capacity not in (1, 3):
            raise InputError("gate_capacity must be 1 or 3, matching the source branches")

    @property
    def gate_edge_length(self) -> float:
        return 2.0 * self.vehicle_diagonals[-1]

    @property
    def taxi_edge_length(self) -> float:
        return 3.0 * self.vehicle_diagonals[-1]

    @property
    def vertical_edge_length(self) -> float:
        return 5.0 * self.vehicle_diagonals[-1]

    @property
    def direction_edge_length(self) -> float:
        return 20.0 * self.vehicle_diagonals[-1]

    @property
    def d_sep_taxi(self) -> np.ndarray:
        d = np.asarray(self.vehicle_diagonals, dtype=float)
        return np.maximum.outer(d, d)

    @property
    def d_sep_fix(self) -> np.ndarray:
        return 15.0 * self.d_sep_taxi

    @property
    def twake(self) -> np.ndarray:
        return self.d_sep_taxi / self.taxi_speed

    @property
    def f_separation(self) -> np.ndarray:
        # This table is created by the MATLAB script even though the selected
        # C20 rows use d_sep_fix directly.  Retaining it makes input metadata
        # auditable against the source constants.
        m = math.ceil(float(self.twake[-1, -1]))
        fdt = self.direction_edge_length / (3.0 * m)
        fractions = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 2, 3, 4, 5],
                [3, 3, 3, 4, 5],
                [4, 4, 4, 4, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=float,
        ) / 5.0
        return self.twake * (fractions * fdt)

    @property
    def weights(self) -> dict[str, float]:
        return {
            "approach": 7.0,
            "landing": 10.0,
            "taxi_arrival": 8.0,
            "gate_departure": 2.0,
            "taxi_departure": 8.0,
            "takeoff": 10.0,
            "climb": 7.0,
            "gate_queue": 1.0,
        }

    def big_m(self, num_flights: int) -> float:
        return float(math.ceil(num_flights / 10.0 + 1.0) * 2000.0)

    def to_json(self) -> dict[str, Any]:
        return {
            "vehicle_diagonals": list(self.vehicle_diagonals),
            "taxi_speed": self.taxi_speed,
            "vertical_climb_speed": self.vertical_climb_speed,
            "slant_climb_speed": self.slant_climb_speed,
            "tot": list(self.tot),
            "tat": list(self.tat),
            "different_direction_sep": self.different_direction_sep,
            "extra_arrival_delay": self.extra_arrival_delay,
            "gate_capacity": self.gate_capacity,
            "capacity_nodes": self.capacity_nodes,
            "edge_lengths": {
                "gate": self.gate_edge_length,
                "taxi": self.taxi_edge_length,
                "vertical": self.vertical_edge_length,
                "direction": self.direction_edge_length,
            },
            "weights": self.weights,
        }


@dataclass(frozen=True)
class Topology:
    """The ``topo_1_arr_dep_dir_4.m`` network in deterministic Python order."""

    gates: tuple[str, ...] = ("G1", "G2", "G3", "G4")
    taxi: tuple[str, ...] = ("a", "b", "c")
    tlof: tuple[str, ...] = ("R2",)
    overflight: tuple[str, ...] = ("X",)
    directions: tuple[str, ...] = ("N", "E", "W", "S")

    @property
    def gate_capacity_nodes(self) -> tuple[str, ...]:
        return tuple(f"{g}c{k}" for g in self.gates for k in range(2))

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.gates + self.gate_capacity_nodes + self.taxi + self.tlof + self.overflight + self.directions

    @property
    def gate_edges(self) -> tuple[str, ...]:
        return ("G1-a", "G3-a", "G2-b", "G4-b", "a-G1", "a-G3", "b-G2", "b-G4")

    @property
    def taxi_edges(self) -> tuple[str, ...]:
        return ("a-b", "b-c", "b-a", "c-b")

    @property
    def tlof_edges(self) -> tuple[str, ...]:
        return ("c-R2", "R2-c")

    @property
    def overflight_edges(self) -> tuple[str, ...]:
        return ("R2-X", "X-R2")

    @property
    def direction_edges(self) -> tuple[str, ...]:
        return ("X-N", "N-X", "X-E", "E-X", "X-W", "W-X", "X-S", "S-X")

    @property
    def edges(self) -> tuple[str, ...]:
        return self.gate_edges + self.taxi_edges + self.tlof_edges + self.overflight_edges + self.direction_edges

    @property
    def non_tlof_edges(self) -> tuple[str, ...]:
        return tuple(e for e in self.edges if e not in self.tlof_edges)

    def branch(self, gate: str) -> str:
        if gate in ("G1", "G3"):
            return "a"
        if gate in ("G2", "G4"):
            return "b"
        raise InputError(f"unsupported gate {gate!r}")

    def arrival_nodes(self, gate: str, direction: str) -> tuple[str, ...]:
        self._validate_direction(direction)
        branch = self.branch(gate)
        if branch == "a":
            return (direction, "X", "R2", "c", "b", "a", gate)
        return (direction, "X", "R2", "c", "b", gate)

    def departure_nodes(self, gate: str, direction: str) -> tuple[str, ...]:
        return tuple(reversed(self.arrival_nodes(gate, direction)))

    @staticmethod
    def _validate_direction(direction: str) -> None:
        if direction not in ("N", "E", "W", "S"):
            raise InputError(f"unsupported fix direction {direction!r}")

    def edge_lengths(self, params: Parameters) -> dict[str, float]:
        out: dict[str, float] = {}
        out.update({e: params.gate_edge_length for e in self.gate_edges})
        out.update({e: params.taxi_edge_length for e in self.taxi_edges})
        out.update({e: params.vertical_edge_length for e in self.overflight_edges})
        out.update({e: params.direction_edge_length for e in self.direction_edges})
        return out


@dataclass
class Request:
    request_id: str
    kind: str
    class_id: int
    gate: str
    direction: str
    arr_req_time: float | None = None
    dep_req_time: float | None = None
    tat: float | None = None
    original_arr_req_time: float | None = None
    original_dep_req_time: float | None = None

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in {"arr", "dep", "tat"}:
            raise InputError(f"{self.request_id}: kind must be arr, dep, or tat")
        if not 1 <= int(self.class_id) <= 5:
            raise InputError(f"{self.request_id}: class_id must be in 1..5")
        self.class_id = int(self.class_id)
        if self.original_arr_req_time is None:
            self.original_arr_req_time = self.arr_req_time
        if self.original_dep_req_time is None:
            self.original_dep_req_time = self.dep_req_time
        if self.kind in {"arr", "tat"} and self.arr_req_time is None:
            raise InputError(f"{self.request_id}: arrival and turnaround requests need arr_req_time")
        if self.kind == "dep" and self.dep_req_time is None:
            raise InputError(f"{self.request_id}: departure requests need dep_req_time")
        if self.kind == "tat" and self.tat is None:
            raise InputError(f"{self.request_id}: turnaround requests need tat")

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id - 1]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "type": self.kind,
            "class": self.class_name,
            "class_id": self.class_id,
            "gate": self.gate,
            "direction": self.direction,
            "arr_req_time": self.arr_req_time,
            "dep_req_time": self.dep_req_time,
            "tat": self.tat,
            "original_arr_req_time": self.original_arr_req_time,
            "original_dep_req_time": self.original_dep_req_time,
        }


@dataclass
class Leg:
    record_id: str
    parent_id: str
    kind: str
    class_id: int
    gate: str
    direction: str
    nodes: tuple[str, ...]
    edges: tuple[str, ...]
    tlof: str
    req_time: float
    original_req_time: float
    tot: float
    tat: float

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["nodes"] = list(self.nodes)
        data["edges"] = list(self.edges)
        data["class"] = CLASS_NAMES[self.class_id - 1]
        return data


@dataclass
class Turnaround:
    record_id: str
    parent_id: str
    class_id: int
    gate: str
    direction: str
    nodes: tuple[str, ...]
    tat: float
    arr_leg_id: str
    dep_leg_id: str

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["nodes"] = list(self.nodes)
        data["class"] = CLASS_NAMES[self.class_id - 1]
        return data


@dataclass
class FlightSet:
    requests: list[Request]
    records: list[Leg | Turnaround]
    arrivals: list[Leg]
    departures: list[Leg]
    turnarounds: list[Turnaround]
    topology: Topology
    params: Parameters
    preprocessing: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "requests": [r.to_json() for r in self.requests],
            "records": [r.to_json() for r in self.records],
            "arrivals": [r.to_json() for r in self.arrivals],
            "departures": [r.to_json() for r in self.departures],
            "turnarounds": [r.to_json() for r in self.turnarounds],
            "preprocessing": self.preprocessing,
        }


def class_id(value: int | str) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key.isdigit():
            value = int(key)
        else:
            if key not in CLASS_TO_ID:
                raise InputError(f"unknown class {value!r}")
            return CLASS_TO_ID[key]
    try:
        answer = int(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"invalid class {value!r}") from exc
    if not 1 <= answer <= 5:
        raise InputError(f"class_id must be in 1..5, got {answer}")
    return answer


def request_from_mapping(value: Mapping[str, Any]) -> Request:
    request_id = value.get("id", value.get("request_id"))
    if not request_id:
        raise InputError("each request needs id")
    return Request(
        request_id=str(request_id),
        kind=str(value.get("type", value.get("kind", "arr"))),
        class_id=class_id(value.get("class_id", value.get("class", 1))),
        gate=str(value.get("gate", "G1")),
        direction=str(value.get("direction", "N")).upper(),
        arr_req_time=_as_float_or_none(value.get("arr_req_time", value.get("arrival_request_time"))),
        dep_req_time=_as_float_or_none(value.get("dep_req_time", value.get("departure_request_time"))),
        tat=_as_float_or_none(value.get("tat", value.get("turnaround_time"))),
    )


def _as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        answer = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"time value {value!r} is not numeric") from exc
    if not math.isfinite(answer):
        raise InputError("times must be finite")
    return answer


def time_on_edge(edge: str, params: Parameters, topology: Topology) -> float:
    lengths = topology.edge_lengths(params)
    if edge not in lengths:
        raise InputError(f"edge {edge!r} has no source length")
    if edge in topology.direction_edges:
        return lengths[edge] / params.slant_climb_speed
    if edge in topology.overflight_edges:
        return lengths[edge] / params.vertical_climb_speed
    return lengths[edge] / params.taxi_speed


def calc_dep_req_time(arr: Leg, params: Parameters, topology: Topology) -> float:
    # Literal translation of calcDepReqTime in test_10.  TLOF edge time is
    # excluded from the taxi list because it is absent from Edges.taxi/gate.
    fix_time = time_on_edge(arr.edges[0], params, topology) - 5.0
    ovf_time = time_on_edge(arr.edges[1], params, topology) - 5.0
    taxi_time = 0.0
    for edge in arr.edges:
        if edge in topology.taxi_edges or edge in topology.gate_edges:
            taxi_time += max(time_on_edge(edge, params, topology) - 5.0, 1.0)
    return fix_time + ovf_time + arr.tot + arr.tat + taxi_time


def preprocess_arrival_requests(
    arrivals: list[Leg],
    requests: list[Request],
    params: Parameters,
    topology: Topology,
) -> dict[str, Any]:
    """Apply the source request-time spacing pass and retain both time sets.

    MATLAB adjusts the later request in each pair until same-fix separation or
    the configured different-direction separation is met.  The Python port
    uses a stable time/id order for the same operation so equal request times
    have deterministic behavior.
    """

    before = {leg.record_id: float(leg.req_time) for leg in arrivals}
    # The nested loops intentionally follow the MATLAB f1/f2 order.  In
    # particular, equal request times enter the ``else`` branch and shift f1;
    # sorting the requests first changes the source transformation.
    max_passes = max(1, len(arrivals) * len(arrivals) * 1000)
    converged = False
    for _ in range(max_passes):
        changed = False
        for first_index, first in enumerate(arrivals):
            for second_index, second in enumerate(arrivals):
                if first_index == second_index:
                    continue
                if first.direction == second.direction:
                    direction_time = topology.edge_lengths(params)[first.edges[0]] / params.slant_climb_speed
                    actual_speed = topology.edge_lengths(params)[first.edges[0]] / (direction_time - 5.0)
                    required = params.d_sep_fix[first.class_id - 1, second.class_id - 1] / actual_speed + params.extra_arrival_delay
                else:
                    required = params.different_direction_sep
                time_diff = abs(first.req_time - second.req_time)
                if round(required - time_diff, 2) > 0:
                    shift = float(required - time_diff)
                    if second.req_time > first.req_time:
                        second.req_time += shift
                    else:
                        first.req_time += shift
                    changed = True
        if not changed:
            converged = True
            break
    if not converged:
        raise InputError("arrival request preprocessing did not converge within its explicit guard")

    by_parent = {leg.parent_id: leg for leg in arrivals}
    request_by_id = {request.request_id: request for request in requests}
    for leg in arrivals:
        request = request_by_id[leg.parent_id]
        request.arr_req_time = float(leg.req_time)
        if request.kind == "tat":
            # Source updates turnaround departure request after every arrival
            # request shift.  Recompute it from the effective arrival.
            computed = float(request.arr_req_time) + calc_dep_req_time(leg, params, topology)
            if request.original_dep_req_time is None:
                request.original_dep_req_time = computed
            request.dep_req_time = computed

    return {
        "enabled": True,
        "same_direction_rule": "D_sep_fix[class_i,class_j] / actual_speed + extra_arrival_delay",
        "different_direction_sep": params.different_direction_sep,
        "extra_arrival_delay": params.extra_arrival_delay,
        "original_arrival_request_times": before,
        "effective_arrival_request_times": {leg.record_id: float(leg.req_time) for leg in arrivals},
        "changed": {k: float(v) for k, v in before.items() if not math.isclose(v, {leg.record_id: leg.req_time for leg in arrivals}[k])},
    }


def build_flight_set(
    requests: Iterable[Mapping[str, Any] | Request],
    *,
    params: Parameters | None = None,
    topology: Topology | None = None,
    preprocess: bool = True,
) -> FlightSet:
    params = params or Parameters()
    topology = topology or Topology()
    parsed = [x if isinstance(x, Request) else request_from_mapping(x) for x in requests]
    if not parsed:
        raise InputError("at least one request is required")
    if len({r.request_id for r in parsed}) != len(parsed):
        raise InputError("request ids must be unique")
    request_by_id = {r.request_id: r for r in parsed}
    arrivals: list[Leg] = []
    departures: list[Leg] = []
    turnarounds: list[Turnaround] = []

    for request in parsed:
        topology.branch(request.gate)
        topology._validate_direction(request.direction)
        tot = params.tot[request.class_id - 1]
        tat = params.tat[request.class_id - 1] if request.tat is None else float(request.tat)
        if request.kind in {"arr", "tat"}:
            assert request.arr_req_time is not None
            nodes = topology.arrival_nodes(request.gate, request.direction)
            leg = Leg(
                record_id=request.request_id if request.kind == "arr" else f"{request.request_id}_Arr",
                parent_id=request.request_id,
                kind="arr",
                class_id=request.class_id,
                gate=request.gate,
                direction=request.direction,
                nodes=nodes,
                edges=tuple(f"{u}-{v}" for u, v in zip(nodes, nodes[1:])),
                tlof="R2",
                req_time=float(request.arr_req_time),
                original_req_time=float(request.arr_req_time),
                tot=float(tot),
                tat=float(tat),
            )
            arrivals.append(leg)
            if request.kind == "tat":
                computed = float(request.arr_req_time) + calc_dep_req_time(leg, params, topology)
                if request.original_dep_req_time is None:
                    request.original_dep_req_time = computed
                request.dep_req_time = computed
        if request.kind in {"dep", "tat"}:
            if request.kind == "dep":
                assert request.dep_req_time is not None
            nodes = topology.departure_nodes(request.gate, request.direction)
            leg_id = request.request_id if request.kind == "dep" else f"{request.request_id}_Dep"
            req = float(request.dep_req_time if request.dep_req_time is not None else 0.0)
            dep_leg = Leg(
                record_id=leg_id,
                parent_id=request.request_id,
                kind="dep",
                class_id=request.class_id,
                gate=request.gate,
                direction=request.direction,
                nodes=nodes,
                edges=tuple(f"{u}-{v}" for u, v in zip(nodes, nodes[1:])),
                tlof="R2",
                req_time=req,
                original_req_time=req,
                tot=float(tot),
                tat=float(tat),
            )
            departures.append(dep_leg)
        if request.kind == "tat":
            assert request.arr_req_time is not None
            turnarounds.append(
                Turnaround(
                    record_id=request.request_id,
                    parent_id=request.request_id,
                    class_id=request.class_id,
                    gate=request.gate,
                    direction=request.direction,
                    nodes=(f"{request.gate}c0", f"{request.gate}c1"),
                    tat=float(tat),
                    arr_leg_id=f"{request.request_id}_Arr",
                    dep_leg_id=f"{request.request_id}_Dep",
                )
            )

    # Keep the route-level objects and turnaround records separate, while the
    # combined order mirrors the source sets used by each family.
    if preprocess and arrivals:
        prep = preprocess_arrival_requests(arrivals, parsed, params, topology)
    else:
        prep = {
            "enabled": False,
            "original_arrival_request_times": {leg.record_id: leg.req_time for leg in arrivals},
            "effective_arrival_request_times": {leg.record_id: leg.req_time for leg in arrivals},
            "changed": {},
        }

    # Synchronize turnaround departure request after preprocessing, and use
    # effective request values in the corresponding leg records.
    for leg in departures:
        request = request_by_id[leg.parent_id]
        if request.kind == "tat":
            leg.req_time = float(request.dep_req_time)
            leg.original_req_time = float(request.original_dep_req_time if request.original_dep_req_time is not None else leg.req_time)
    for leg in arrivals:
        request = request_by_id[leg.parent_id]
        request.arr_req_time = float(leg.req_time)

    records: list[Leg | Turnaround] = [*arrivals, *departures, *turnarounds]
    return FlightSet(
        requests=parsed,
        records=records,
        arrivals=arrivals,
        departures=departures,
        turnarounds=turnarounds,
        topology=topology,
        params=params,
        preprocessing=prep,
    )


@dataclass
class ConstraintRow:
    name: str
    family: str
    coeffs: dict[int, float]
    lower: float
    upper: float


class LinearBuilder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.rows: list[ConstraintRow] = []

    def var(self, name: str, *, lower: float = 0.0, upper: float = math.inf, integer: bool = False) -> int:
        if name in self.names:
            raise InputError(f"duplicate variable {name}")
        index = len(self.names)
        self.names.append(name)
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.integrality.append(1 if integer else 0)
        return index

    def row(
        self,
        name: str,
        family: str,
        coeffs: Mapping[int, float],
        *,
        lower: float = -math.inf,
        upper: float = math.inf,
    ) -> None:
        clean = {int(k): float(v) for k, v in coeffs.items() if abs(float(v)) > 1e-12}
        self.rows.append(ConstraintRow(name=name, family=family, coeffs=clean, lower=float(lower), upper=float(upper)))

    def matrix(self) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
        matrix = csr_matrix(
            (
                [value for row in self.rows for value in row.coeffs.values()],
                (
                    [r for r, row in enumerate(self.rows) for _ in row.coeffs],
                    [c for row in self.rows for c in row.coeffs],
                ),
            ),
            shape=(len(self.rows), len(self.names)),
        )
        lower = np.asarray([row.lower for row in self.rows], dtype=float)
        upper = np.asarray([row.upper for row in self.rows], dtype=float)
        return matrix, lower, upper


@dataclass
class ScheduleResult:
    status: str
    solver_status: int
    message: str
    objective: float | None
    raw_objective: float | None
    mip_gap: float | None
    lower_bound: float | None
    runtime_seconds: float
    formulation: str
    variable_count: int
    constraint_count: int
    node_times: dict[str, dict[str, float]]
    integer_values: dict[str, float]
    constraint_checks: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "solver_status": self.solver_status,
            "message": self.message,
            "objective": self.objective,
            "raw_objective": self.raw_objective,
            "mip_gap": self.mip_gap,
            "lower_bound": self.lower_bound,
            "runtime_seconds": self.runtime_seconds,
            "formulation": self.formulation,
            "variable_count": self.variable_count,
            "constraint_count": self.constraint_count,
            "node_times": self.node_times,
            "integer_values": self.integer_values,
            "constraint_checks": self.constraint_checks,
            "summary": self.summary,
        }


class ScheduleModel:
    """Build and solve the sparse MILP for one :class:`FlightSet`."""

    def __init__(self, flight_set: FlightSet, *, formulation: str = "paper") -> None:
        if formulation not in {"paper", "upstream-legacy"}:
            raise InputError("formulation must be paper or upstream-legacy")
        self.flight_set = flight_set
        self.topology = flight_set.topology
        self.params = flight_set.params
        self.formulation = formulation
        self.builder = LinearBuilder()
        self.t_vars: dict[tuple[str, str], int] = {}
        self.y_vars: dict[tuple[str, str, str], int] = {}
        self.y1_vars: dict[tuple[str, str], int] = {}
        self.objective = np.zeros(0, dtype=float)
        self.objective_constant = 0.0
        self._built = False

    @property
    def records(self) -> list[Leg | Turnaround]:
        return self.flight_set.records

    def _t(self, record: Leg | Turnaround, node: str) -> int:
        key = (record.record_id, node)
        if key not in self.t_vars:
            raise InputError(f"record {record.record_id} does not use node {node}")
        return self.t_vars[key]

    def _y(self, node: str, left: str, right: str) -> int:
        key = (node, left, right)
        if key not in self.y_vars:
            raise InputError(f"no y variable for shared node {node}, {left}, {right}")
        return self.y_vars[key]

    def _add_big_m_lower(self, name: str, family: str, lhs: Mapping[int, float], binary: int, m: float) -> None:
        coeffs = dict(lhs)
        coeffs[binary] = coeffs.get(binary, 0.0) - m
        self.builder.row(name, family, coeffs, lower=-m)

    def _add_ordered_big_m(self, name: str, family: str, before: int, after: int, binary: int, m: float) -> None:
        # after >= before - (1-binary)M
        self._add_big_m_lower(name, family, {after: 1.0, before: -1.0}, binary, m)

    def build(self) -> "ScheduleModel":
        if self._built:
            return self
        # MATLAB computes M from the requested physical-flight count before
        # expanding turnaround requests into arrival/departure/base records.
        n = len(self.flight_set.requests)
        m = self.params.big_m(n)

        # Source optimvar t_iu is nonnegative.  Unused unconstrained cells are
        # intentionally omitted; all route and gate cells referenced below are
        # retained.
        for record in self.records:
            for node in record.nodes:
                self.t_vars[(record.record_id, node)] = self.builder.var(f"t[{record.record_id},{node}]")

        # Directed order variables are present for each shared node and pair.
        for left, right in itertools.combinations(self.records, 2):
            shared = sorted(set(left.nodes).intersection(right.nodes))
            for node in shared:
                self.y_vars[(node, left.record_id, right.record_id)] = self.builder.var(
                    f"y[{node},{left.record_id},{right.record_id}]", upper=1.0, integer=True
                )
                self.y_vars[(node, right.record_id, left.record_id)] = self.builder.var(
                    f"y[{node},{right.record_id},{left.record_id}]", upper=1.0, integer=True
                )

        # Auxiliary y1 variables are only created when source GateCapacity3 is
        # active for a gate with more than three turnaround records.
        for gate in self.topology.gates:
            group = [r for r in self.flight_set.turnarounds if r.gate == gate]
            if self.params.gate_capacity == 3 and len(group) > 3:
                for left, right in itertools.permutations(group, 2):
                    self.y1_vars[(left.record_id, right.record_id)] = self.builder.var(
                        f"y1[{left.record_id},{right.record_id}]", upper=1.0, integer=True
                    )

        self._add_request_rows(m)
        self._add_speed_rows(m)
        self._add_turnaround_rows()
        self._add_ytime_and_complement_rows(m)
        self._add_overtake_rows()
        self._add_collision_rows()
        self._add_taxi_and_fix_rows(m)
        self._add_tlof_rows(m)
        self._add_wake_rows(m)
        self._add_gate_rows(m)
        self._build_objective()
        self._built = True
        return self

    def _add_request_rows(self, m: float) -> None:
        for leg in self.flight_set.arrivals:
            self.builder.row(
                f"ARAPR_i[{leg.record_id}]",
                "ARAPR_i",
                {self._t(leg, leg.nodes[0]): 1.0},
                lower=leg.req_time,
                upper=leg.req_time,
            )
        for leg in self.flight_set.departures:
            self.builder.row(
                f"gateOutC1[{leg.record_id}]",
                "gateOutC1",
                {self._t(leg, leg.nodes[0]): 1.0},
                lower=leg.req_time,
            )

    def _add_speed_rows(self, m: float) -> None:
        lengths = self.topology.edge_lengths(self.params)
        for leg in [*self.flight_set.arrivals, *self.flight_set.departures]:
            for edge in leg.edges:
                if edge in self.topology.tlof_edges:
                    continue
                u, v = edge.split("-", 1)
                travel = time_on_edge(edge, self.params, self.topology)
                delay = 0.0
                if u in self.topology.directions and leg.kind == "arr":
                    delay = 0.0  # source global descentDelay is zero in test_10
                minimum = max(travel - 5.0, 1.0)
                maximum = min(travel + 5.0 + delay, m)
                self.builder.row(
                    f"minspeed[{leg.record_id},{edge}]",
                    "minspeed",
                    {self._t(leg, v): 1.0, self._t(leg, u): -1.0},
                    lower=minimum,
                )
                self.builder.row(
                    f"maxspeed[{leg.record_id},{edge}]",
                    "maxspeed",
                    {self._t(leg, v): 1.0, self._t(leg, u): -1.0},
                    upper=maximum,
                )

    def _add_turnaround_rows(self) -> None:
        for record in self.flight_set.turnarounds:
            self.builder.row(
                f"TAT[{record.record_id}]",
                "TAT",
                {self._t(record, record.nodes[1]): 1.0, self._t(record, record.nodes[0]): -1.0},
                lower=record.tat,
            )

    def _add_ytime_and_complement_rows(self, m: float) -> None:
        for left, right in itertools.combinations(self.records, 2):
            for node in sorted(set(left.nodes).intersection(right.nodes)):
                yl = self._y(node, left.record_id, right.record_id)
                yr = self._y(node, right.record_id, left.record_id)
                self.builder.row(
                    f"y7[{node},{left.record_id},{right.record_id}]",
                    "y7",
                    {yl: 1.0, yr: 1.0},
                    lower=1.0,
                    upper=1.0,
                )
                if node not in self.topology.gates:
                    self._add_ordered_big_m(
                        f"ytime[{node},{left.record_id},{right.record_id}]",
                        "ytime",
                        self._t(left, node),
                        self._t(right, node),
                        yl,
                        m,
                    )
                    self._add_ordered_big_m(
                        f"ytime[{node},{right.record_id},{left.record_id}]",
                        "ytime",
                        self._t(right, node),
                        self._t(left, node),
                        yr,
                        m,
                    )

    def _add_overtake_rows(self) -> None:
        excluded = set(self.topology.overflight_edges) | set(self.topology.tlof_edges)
        legs = [*self.flight_set.arrivals, *self.flight_set.departures]
        for left, right in itertools.combinations(legs, 2):
            common_edges = [e for e in left.edges if e in right.edges and e not in excluded]
            for edge in common_edges:
                u, v = edge.split("-", 1)
                self.builder.row(
                    f"Overtake[{edge},{left.record_id},{right.record_id}]",
                    "Overtake",
                    {self._y(u, left.record_id, right.record_id): 1.0, self._y(v, left.record_id, right.record_id): -1.0},
                    lower=0.0,
                    upper=0.0,
                )

    def _add_collision_rows(self) -> None:
        # One representative of every opposite directed edge pair, matching
        # the source UniqueEdges/ReverseEdges construction.
        seen: set[frozenset[str]] = set()
        for edge in self.topology.edges:
            u, v = edge.split("-", 1)
            pair = frozenset((edge, f"{v}-{u}"))
            if pair in seen:
                continue
            seen.add(pair)
            reverse = f"{v}-{u}"
            legs = [*self.flight_set.arrivals, *self.flight_set.departures]
            for left, right in itertools.combinations(legs, 2):
                opposite = (edge in left.edges and reverse in right.edges) or (reverse in left.edges and edge in right.edges)
                if opposite:
                    self.builder.row(
                        f"Collision[{edge},{left.record_id},{right.record_id}]",
                        "Collision",
                        {self._y(u, left.record_id, right.record_id): 1.0, self._y(v, left.record_id, right.record_id): -1.0},
                        lower=0.0,
                        upper=0.0,
                    )

    def _add_taxi_and_fix_rows(self, m: float) -> None:
        lengths = self.topology.edge_lengths(self.params)
        taxi_or_gate = set(self.topology.taxi_edges) | set(self.topology.gate_edges)
        direction_edges = set(self.topology.direction_edges)
        legs = [*self.flight_set.arrivals, *self.flight_set.departures]
        for left, right in itertools.combinations(legs, 2):
            common_taxi = [e for e in left.edges if e in right.edges and e in taxi_or_gate]
            for edge in common_taxi:
                u, v = edge.split("-", 1)
                sep_left = self.params.d_sep_taxi[left.class_id - 1, right.class_id - 1]
                sep_right = self.params.d_sep_taxi[right.class_id - 1, left.class_id - 1]
                frac_left = sep_left / lengths[edge]
                frac_right = sep_right / lengths[edge]
                yl = self._y(u, left.record_id, right.record_id)
                yr = self._y(u, right.record_id, left.record_id)
                self._add_big_m_lower(
                    f"taxiSeparation1[{edge},{left.record_id},{right.record_id}]",
                    "taxiSeparation1",
                    {
                        self._t(right, u): 1.0,
                        self._t(left, u): -1.0 + frac_left,
                        self._t(left, v): -frac_left,
                    },
                    yl,
                    m,
                )
                self._add_big_m_lower(
                    f"taxiSeparation1[{edge},{right.record_id},{left.record_id}]",
                    "taxiSeparation1",
                    {
                        self._t(left, u): 1.0,
                        self._t(right, u): -1.0 + frac_right,
                        self._t(right, v): -frac_right,
                    },
                    yr,
                    m,
                )
            common_fix = [e for e in left.edges if e in right.edges and e in direction_edges]
            for edge in common_fix:
                u, v = edge.split("-", 1)
                sep = self.params.d_sep_fix[left.class_id - 1, right.class_id - 1] / lengths[edge]
                self._add_big_m_lower(
                    f"fixSeparation1[{edge},{left.record_id},{right.record_id}]",
                    "fixSeparation1",
                    {
                        self._t(right, u): 1.0,
                        self._t(left, u): -1.0 + sep,
                        self._t(left, v): -sep,
                    },
                    self._y(u, left.record_id, right.record_id),
                    m,
                )
                sep = self.params.d_sep_fix[right.class_id - 1, left.class_id - 1] / lengths[edge]
                self._add_big_m_lower(
                    f"fixSeparation1[{edge},{right.record_id},{left.record_id}]",
                    "fixSeparation1",
                    {
                        self._t(left, u): 1.0,
                        self._t(right, u): -1.0 + sep,
                        self._t(right, v): -sep,
                    },
                    self._y(u, right.record_id, left.record_id),
                    m,
                )

    def _add_tlof_rows(self, m: float) -> None:
        for leg in self.flight_set.arrivals:
            self.builder.row(
                f"TLOFexitArr[{leg.record_id}]",
                "TLOFexitArr",
                {self._t(leg, leg.nodes[3]): 1.0, self._t(leg, leg.tlof): -1.0},
                lower=leg.tot,
            )
        for left, right in itertools.permutations(self.flight_set.arrivals, 2):
            if left.tlof == right.tlof:
                self._add_ordered_big_m(
                    f"TLOFClearArr[{left.record_id},{right.record_id}]",
                    "TLOFClearArr",
                    self._t(left, left.nodes[3]),
                    self._t(right, right.nodes[1]),
                    self._y(left.tlof, left.record_id, right.record_id),
                    m,
                )
        for leg in self.flight_set.departures:
                self.builder.row(
                    f"TLOFenterDep[{leg.record_id}]",
                    "TLOFenterDep",
                    {self._t(leg, leg.tlof): 1.0, self._t(leg, leg.nodes[-4]): -1.0},
                    lower=leg.tot,
                )
        for left, right in itertools.permutations(self.flight_set.departures, 2):
            if left.tlof == right.tlof:
                self._add_ordered_big_m(
                    f"TLOFenter2Dep[{left.record_id},{right.record_id}]",
                    "TLOFenter2Dep",
                    self._t(left, left.nodes[-2]),
                    self._t(right, right.nodes[-4]),
                    self._y(left.tlof, left.record_id, right.record_id),
                    m,
                )
        for arr in self.flight_set.arrivals:
            for dep in self.flight_set.departures:
                if arr.tlof == dep.tlof:
                    self._add_ordered_big_m(
                        f"TLOFpadSingle[{arr.record_id},{dep.record_id}]",
                        "TLOFpadSingle",
                        self._t(arr, arr.nodes[3]),
                        self._t(dep, dep.nodes[-4]),
                        self._y(arr.tlof, arr.record_id, dep.record_id),
                        m,
                    )

    def _add_wake_rows(self, m: float) -> None:
        legs = [*self.flight_set.arrivals, *self.flight_set.departures]
        for left, right in itertools.combinations(legs, 2):
            if left.tlof != right.tlof:
                continue
            pairs = [(left, right)] if self.formulation == "upstream-legacy" else [(left, right), (right, left)]
            for first, second in pairs:
                sep = self.params.twake[first.class_id - 1, second.class_id - 1]
                # second >= first + sep - (1-y)M.
                self.builder.row(
                    name=f"wake[{first.record_id},{second.record_id},{first.tlof}]",
                    family="wake",
                    coeffs={
                        self._t(second, second.tlof): 1.0,
                        self._t(first, first.tlof): -1.0,
                        self._y(first.tlof, first.record_id, second.record_id): -m,
                    },
                    lower=sep - m,
                )

    def _add_gate_rows(self, m: float) -> None:
        for tat in self.flight_set.turnarounds:
            arr = next(x for x in self.flight_set.arrivals if x.record_id == tat.arr_leg_id)
            dep = next(x for x in self.flight_set.departures if x.record_id == tat.dep_leg_id)
            self.builder.row(
                f"GateQ[{tat.record_id},entry]",
                "GateQ",
                {self._t(arr, arr.nodes[-1]): 1.0, self._t(tat, tat.nodes[0]): -1.0},
                lower=0.0,
                upper=0.0,
            )
            self.builder.row(
                f"GateQ[{tat.record_id},exit]",
                "GateQ",
                {self._t(dep, dep.nodes[0]): 1.0, self._t(tat, tat.nodes[1]): -1.0},
                lower=0.0,
                upper=0.0,
            )

        for left, right in itertools.combinations(self.flight_set.turnarounds, 2):
            if left.gate != right.gate:
                continue
            arr_l = next(x for x in self.flight_set.arrivals if x.record_id == left.arr_leg_id)
            arr_r = next(x for x in self.flight_set.arrivals if x.record_id == right.arr_leg_id)
            dep_l = next(x for x in self.flight_set.departures if x.record_id == left.dep_leg_id)
            dep_r = next(x for x in self.flight_set.departures if x.record_id == right.dep_leg_id)
            self.builder.row(
                f"GateQ_ygenij[{left.record_id},{right.record_id}]",
                "GateQ_ygenij",
                {
                    self._y(left.gate, arr_l.record_id, arr_r.record_id): 1.0,
                    self._y(left.nodes[0], left.record_id, right.record_id): -1.0,
                },
                lower=0.0,
                upper=0.0,
            )
            self.builder.row(
                f"GateQ_ygexij[{left.record_id},{right.record_id}]",
                "GateQ_ygexij",
                {
                    self._y(left.gate, dep_l.record_id, dep_r.record_id): 1.0,
                    self._y(left.nodes[1], left.record_id, right.record_id): -1.0,
                },
                lower=0.0,
                upper=0.0,
            )

        for gate in self.topology.gates:
            group = [x for x in self.flight_set.turnarounds if x.gate == gate]
            if len(group) < 2:
                continue
            for left, right in itertools.combinations(group, 2):
                if self.params.gate_capacity == 1:
                    self._add_gate_capacity_pair(left, right, gate, m, None)
                elif len(group) > 3:
                    self._add_gate_capacity_pair(left, right, gate, m, self.y1_vars)
            if self.params.gate_capacity == 3 and len(group) > 3:
                for left in group:
                    others = [x for x in group if x.record_id != left.record_id]
                    for trio in itertools.combinations(others, 3):
                        coeffs = {self.y1_vars[(left.record_id, other.record_id)]: 1.0 for other in trio}
                        self.builder.row(
                            f"GateCapacity2[{gate},{left.record_id},{','.join(x.record_id for x in trio)}]",
                            "GateCapacity2",
                            coeffs,
                            lower=1.0,
                        )

    def _add_gate_capacity_pair(
        self,
        left: Turnaround,
        right: Turnaround,
        gate: str,
        m: float,
        aux: dict[tuple[str, str], int] | None,
    ) -> None:
        def row(first: Turnaround, second: Turnaround, suffix: str) -> None:
            coeffs = {
                self._t(first, first.nodes[0]): 1.0,
                self._t(second, second.nodes[1]): -1.0,
                self._y(first.nodes[0], second.record_id, first.record_id): -m,
            }
            lower = -m
            if aux is not None:
                coeffs[aux[(first.record_id, second.record_id)]] = -m
                lower = -2.0 * m
            self.builder.row(
                f"GateCapacity1[{gate},{first.record_id},{second.record_id},{suffix}]",
                "GateCapacity1",
                coeffs,
                lower=lower,
            )

        row(left, right, "forward")
        row(right, left, "reverse")

    def _build_objective(self) -> None:
        c = np.zeros(len(self.builder.names), dtype=float)
        w = self.params.weights
        constant = 0.0
        for leg in self.flight_set.arrivals:
            c[self._t(leg, leg.nodes[1])] += w["approach"]
            constant -= w["approach"] * leg.req_time
            c[self._t(leg, leg.nodes[3])] += w["landing"]
            c[self._t(leg, leg.nodes[1])] -= w["landing"]
            c[self._t(leg, leg.nodes[-1])] += w["taxi_arrival"]
            c[self._t(leg, leg.nodes[3])] -= w["taxi_arrival"]
        for leg in self.flight_set.departures:
            c[self._t(leg, leg.nodes[0])] += w["gate_departure"]
            constant -= w["gate_departure"] * leg.req_time
            c[self._t(leg, leg.nodes[-4])] += w["taxi_departure"]
            c[self._t(leg, leg.nodes[0])] -= w["taxi_departure"]
            c[self._t(leg, leg.nodes[-2])] += w["takeoff"]
            c[self._t(leg, leg.nodes[-4])] -= w["takeoff"]
            c[self._t(leg, leg.nodes[-1])] += w["climb"]
            c[self._t(leg, leg.nodes[-2])] -= w["climb"]
        for tat in self.flight_set.turnarounds:
            c[self._t(tat, tat.nodes[1])] += w["gate_queue"]
            c[self._t(tat, tat.nodes[0])] -= w["gate_queue"]
        self.objective = c / 10.0
        self.objective_constant = constant / 10.0

    def solve(self, *, time_limit: float | None = 60.0, mip_rel_gap: float | None = None) -> ScheduleResult:
        self.build()
        matrix, lower, upper = self.builder.matrix()
        options: dict[str, Any] = {}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        if mip_rel_gap is not None:
            options["mip_rel_gap"] = float(mip_rel_gap)
        start = time.perf_counter()
        result = milp(
            c=self.objective,
            integrality=np.asarray(self.builder.integrality, dtype=int),
            bounds=Bounds(np.asarray(self.builder.lower), np.asarray(self.builder.upper)),
            constraints=LinearConstraint(matrix, lower, upper),
            options=options,
        )
        elapsed = time.perf_counter() - start
        if result.x is None:
            status = "infeasible" if int(result.status) == 2 else "no_solution"
            checks: list[dict[str, Any]] = []
            node_times: dict[str, dict[str, float]] = {}
            integer_values: dict[str, float] = {}
            objective = None
            raw_objective = None
        else:
            status = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded"}.get(int(result.status), "solver_error")
            checks = self.constraint_checks(result.x)
            node_times = self.node_times(result.x)
            integer_values = {
                name: float(result.x[index])
                for index, name in enumerate(self.builder.names)
                if self.builder.integrality[index]
            }
            raw_objective = float(result.fun) if result.fun is not None else None
            objective = raw_objective + self.objective_constant if raw_objective is not None else None
        mip_gap = getattr(result, "mip_gap", None)
        raw_dual_bound = getattr(result, "mip_dual_bound", None)
        lower_bound = None if raw_dual_bound is None else float(raw_dual_bound) + self.objective_constant
        return ScheduleResult(
            status=status,
            solver_status=int(result.status),
            message=str(result.message),
            objective=objective,
            raw_objective=raw_objective,
            mip_gap=float(mip_gap) if mip_gap is not None and math.isfinite(float(mip_gap)) else None,
            lower_bound=lower_bound,
            runtime_seconds=elapsed,
            formulation=self.formulation,
            variable_count=len(self.builder.names),
            constraint_count=len(self.builder.rows),
            node_times=node_times,
            integer_values=integer_values,
            constraint_checks=checks,
            summary={
                "source_snapshot": "87a8e5ba7393d622768a7695770a4fa4e09c26bd",
                "evidence_type": "newly_run",
                "native_matlab_run": False,
                "formulation": self.formulation,
                "constraint_rows_by_family": self._row_family_counts(),
                "integrality_max_abs_error": self._integrality_error(result.x) if result.x is not None else None,
            },
        )

    def _row_family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.builder.rows:
            counts[row.family] = counts.get(row.family, 0) + 1
        return counts

    def _integrality_error(self, values: np.ndarray) -> float:
        errors = [abs(float(values[i]) - round(float(values[i]))) for i, flag in enumerate(self.builder.integrality) if flag]
        return max(errors, default=0.0)

    def node_times(self, values: np.ndarray) -> dict[str, dict[str, float]]:
        return {
            record.record_id: {node: float(values[index]) for node in record.nodes for index in [self._t(record, node)]}
            for record in self.records
        }

    def constraint_checks(self, values: np.ndarray, tolerance: float = 1e-6) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        for row in self.builder.rows:
            actual = sum(values[index] * coefficient for index, coefficient in row.coeffs.items())
            violation = max(row.lower - actual, actual - row.upper, 0.0)
            checks.append(
                {
                    "name": row.name,
                    "family": row.family,
                    "actual": float(actual),
                    "lower": None if not math.isfinite(row.lower) else row.lower,
                    "upper": None if not math.isfinite(row.upper) else row.upper,
                    "violation": float(violation),
                    "satisfied": bool(violation <= tolerance),
                }
            )
        return checks


def fixture(name: str) -> list[dict[str, Any]]:
    """Return small explicit inputs; these are not MATLAB rng(26) samples."""

    name = name.lower()
    fixtures: dict[str, list[dict[str, Any]]] = {
        "arrival": [
            {"id": "A1", "type": "arr", "class": "Small", "gate": "G1", "direction": "N", "arr_req_time": 10.0}
        ],
        "departure": [
            {"id": "D1", "type": "dep", "class": "Medium", "gate": "G2", "direction": "E", "dep_req_time": 5.0}
        ],
        "tat": [
            {"id": "T1", "type": "tat", "class": "Medium", "gate": "G1", "direction": "N", "arr_req_time": 10.0, "tat": 120.0}
        ],
        "mixed": [
            {"id": "A1", "type": "arr", "class": "Small", "gate": "G1", "direction": "N", "arr_req_time": 10.0},
            {"id": "A2", "type": "arr", "class": "Medium", "gate": "G3", "direction": "N", "arr_req_time": 45.0},
            {"id": "D1", "type": "dep", "class": "Jumbo", "gate": "G2", "direction": "E", "dep_req_time": 5.0},
            {"id": "D2", "type": "dep", "class": "Super", "gate": "G4", "direction": "W", "dep_req_time": 35.0},
            {"id": "D3", "type": "dep", "class": "Small", "gate": "G4", "direction": "N", "dep_req_time": 5.0},
            {"id": "T1", "type": "tat", "class": "Medium", "gate": "G1", "direction": "S", "arr_req_time": 0.0, "tat": 120.0},
        ],
        "gate_capacity3": [
            {"id": f"T{i}", "type": "tat", "class": "Small", "gate": "G1", "direction": ("N", "E", "W", "S")[i % 4], "arr_req_time": 100.0 + i * 20.0, "tat": 90.0}
            for i in range(4)
        ],
    }
    if name not in fixtures:
        raise InputError(f"unknown fixture {name!r}; choose arrival, departure, tat, mixed, or gate_capacity3")
    return json.loads(json.dumps(fixtures[name]))


def load_requests(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("requests")
    if not isinstance(value, list):
        raise InputError("input JSON must contain a requests list")
    return value
