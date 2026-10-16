# FT-RS-S1-1.0 schema and scheduling API

This document describes the implemented interfaces. Use the code revision in
each run's provenance to identify its source. It is an API reference, not a qualification result or
a claim that stage 1 or researcher visual acceptance is complete. The
[stage 1 entry points](README.md), [migration notes](migration.md), and
[source classifications and limits](sources.md) provide the surrounding scope.
The archived `core/fixed_terminal` interfaces and checkpoints are separate.

## Canonical configuration, coordinates, and vehicle

[config.py](../../terminal_marl/core/rotating_slots/config.py) exports
`load_config(path=None) -> Config`. With no path it loads and memoizes the
[canonical YAML](../../new10.16/fixed_terminal_rotating_slots_stage1_config_v1_0.yaml).
An explicit path is read and validated each time. The loader checks the frozen
semantic payload, required fields, finite numbers, and exact ring geometry;
it does not accept a parameter variant under the same contract identity.

`Config` is frozen. Its nested `Section` mappings and tuple sequences are
immutable, support mapping/attribute access, and expose `to_dict()` copies.
`raw_hash` is SHA-256 of the original YAML bytes; `config_hash` hashes the
canonical sorted JSON payload; `profile_hash` hashes the `vehicle`, `energy`,
and `coordinates` sections. `path` identifies the loaded file.

Positions use local ENU metres with flat-terrain AGL height; time is seconds,
velocity m/s, acceleration m/s², and jerk m/s³. The fixed rings are outer
600 m/110 m/37 slots, middle 450 m/70 m/28 slots, and inner 300 m/30 m/18 slots
(radius/height/count). All rotate clockwise at 0.035 rad/s. The boundary radius
is 900 m, the height ceiling 120 m, four entry gates lie on the boundary, and
six shared pads lie on a 100 m circle. Slot counts constrain reservations;
they are not a total airborne-aircraft cap.

[geometry.py](../../terminal_marl/core/rotating_slots/geometry.py) provides:

```text
slot_phase(ring, slot, t, config=None)  # absolute, unwrapped phase in radians
slot_state(ring, slot, t, config=None) # KinematicState
pad_position(pad, config=None)        # (east, north, height)
```

Ring IDs are `"outer"`, `"middle"`, `"inner"`; slots and pads are zero based.
Slot phase uses the absolute clock, not elapsed time since planning. The frozen
vehicle uses fixed mass 620 kg, 16 rotors of diameter 1.575 m, horizontal speed
≤25 m/s, horizontal acceleration norm and lateral limit 1.5 m/s², vertical
acceleration ≤1 m/s², vertical speed ≤2 m/s, horizontal/vertical jerk limits
1/0.5 m/s³. The complete constraints remain in the YAML and validator.
`VehicleProfile` and `EnergyProfile` in
[energy.py](../../terminal_marl/core/rotating_slots/energy.py) are frozen metadata
records; they do not replace canonical configuration validation.

## Authoritative trajectories and candidate generation

[models.py](../../terminal_marl/core/rotating_slots/models.py) defines frozen
`KinematicState(p, v=(0.,0.,0.), a=(0.,0.,0.), jerk=(0.,0.,0.))`.
Each field is a finite three-float tuple. `position`, `velocity`, and
`acceleration` alias `p`, `v`, and `a`.

[trajectory.py](../../terminal_marl/core/rotating_slots/trajectory.py) defines
frozen `TrajectoryPlan` and `Segment` records. A plan carries `aircraft_id`,
`kind`, `start_time`, `end_time`, source/target ring and slot, optional `pad_id`,
`source_version`, `profile_hash`, `config_hash`, `trajectory_hash`, and a tuple
of segments. A segment stores `start_time`, `duration`, `kind`, and polynomial
coefficient tuples `r`, `theta`, `z` in normalized segment time. These parameters
are the checked and executed trajectory, not rendering control points.

`plan.state(t) -> KinematicState` evaluates analytic derivatives at the requested
absolute time. A plan with a source ring continues that source slot before its
start; a target-ring plan continues its target slot after its end; a completed
pad plan continues at the ground endpoint with zero derivatives. Inbound
pre-admission intervals are undefined to `plan.intervals(...)` and must not be
treated as an already admitted flight. `intervals(start, end)` includes the
physical continuation used by energy/safety checks; `bounds(start, end,
derivative=0)` returns coordinate intervals for derivative 0–3.

`plan.to_dict()` includes all parameters and its hash.
`TrajectoryPlan.from_dict(data, config=None)` rebuilds segments and checks the
payload hash; passing `config` additionally checks `profile_hash`.
`validate_plan` and replay loading also check the configuration identity.
The trajectory hash is the canonical JSON digest of all plan fields except
the hash itself. Changing a coefficient, time, resource, version, or profile
changes the identity. A frozen record alone does not certify physical validity.

The minimum continuity is C2: position, velocity, and acceleration match at
segment junctions and the source/target slot boundaries. It does not require
continuous jerk. Normal/emergency finals include the final-capture segment
and the complete prescribed 13.5 s descent atomically; the validator checks
their exact descent structure and touchdown state.

The actual public signatures are:

```text
generate_inbound(aircraft_id, gate_id, target_slot, start_time,
                 source_version=0, config=None, durations=None)
generate_transfer(aircraft_id, source_ring, source_slot, target_slot, start_time,
                  source_version=0, config=None, durations=None)
generate_final(aircraft_id, source_ring, source_slot, pad_id, start_time,
               emergency=False, source_version=0, config=None, durations=None)
make_stable_plan(aircraft_id, ring, slot, start_time=0., end_time=240.,
                 source_version=0, config=None)
make_ground_plan(aircraft_id, pad_id, start_time=0., end_time=240., config=None)
```

The three generators return `CandidateResult(status, plan, reason, details)`.
`accepted` means exactly `status == "FEASIBLE"`; only that result supplies an
accepted plan. Other current generator statuses are `NOT_FOUND` and
`INVALID_INPUT`. The default inbound/transfer duration grid is 20–180 s in
5 s increments; final generation also searches 185–300 s. Explicit `durations`
replaces that finite grid. Normal transfers are adjacent inward edges; normal
finals require the inner ring; emergency finals can originate at any ring.

`NOT_FOUND` records exhaustion of the selected duration/phase candidates in
`details["attempts"]`. It is not proof that every physically possible path is
infeasible. Candidate generation validates individual motion and power-domain
constraints; it does not grant admission, a pad, a slot, traffic separation,
normal touchdown reserve, or cumulative peak-budget permission. Those are
checked by world commands against current state and the whole affected batch.

This read-only planning example uses the same API pattern as the motion tests:

```python
from terminal_marl.core.rotating_slots import (
    load_config, generate_transfer, TrajectoryPlan, validate_plan,
    evaluate_power, integrate_energy,
)

config = load_config()
candidate = generate_transfer("demo", "outer", 0, 0, 0., config=config)
if candidate.accepted:
    plan = candidate.plan
    rebuilt = TrajectoryPlan.from_dict(plan.to_dict(), config=config)
    assert rebuilt.trajectory_hash == plan.trajectory_hash
    state = rebuilt.state(0.137)
    validation = validate_plan(rebuilt, config=config)
    power = evaluate_power(state.v, state.a, config=config)
    energy = integrate_energy(rebuilt, plan.start_time, plan.end_time,
                              max_dt=.2, config=config)
    print(validation.status, power.battery_w, energy.joules)
else:
    print(candidate.status, candidate.reason)  # no infeasibility conclusion
```

## Validation and energy results

[safety.py](../../terminal_marl/core/rotating_slots/safety.py) exports:

```text
validate_plan(plan, config=None, use_cache=True)
validate_separation(plan_a, plan_b, start, end, config=None,
                    max_depth=32, min_interval=1e-7)
certified_minimum_distance(plan_a, plan_b, start, end, config=None,
                           tolerance_m=.001, max_intervals=100000)
```

The first two return `ValidationResult(accepted, status, reason, details)`;
`certificate` aliases `details`. The validator statuses are `SAFE`, `UNSAFE`,
`INVALID`, and `UNKNOWN`. `UNKNOWN` has `accepted=False`: unresolved interval
refinement is rejected, never rounded into safe acceptance. A witness may
establish rejection, but sampled clearances do not establish continuous safety.
The hard rule is true Euclidean centre distance ≥80 m, with no allowed
undershoot. This is separate from the numerical endpoint tolerances used for
algebraic/C2 matching.

Separation checks can use exact/structural stable-slot certificates or general
continuous interval bounds. The `lower_bound_m`/`upper_bound_m` of the fast
separation validator are interval evidence, not necessarily a tightly refined
global minimum. `certified_minimum_distance` separately returns a dictionary
with `status` (`CERTIFIED` or `UNKNOWN`), bounds, `witness_time`,
`witness_distance_m`, `intervals`, and `method`. Exhausting its refinement
budget does not certify the apparent sampled minimum.

`evaluate_power(v, a, config=None) -> PowerResult` uses the shared appendix-A
surrogate. Outputs are battery and shaft power in watts, battery current in
amperes, induced velocity in m/s, dimensionless induced residual, and axial
descent ratio. `status` is `VALID`, `POWER_LIMIT`, or `ENERGY_MODEL_DOMAIN`;
`valid` is true only for `VALID`. Invalid-domain results can contain NaN power
values; they are not clipped to zero or interpreted as free flight.

`integrate_energy(plan, start, end, max_dt=.2, config=None) -> EnergyResult`
returns `joules`, `soc_used` (charge fraction, not percent), `peak_seconds`,
`valid`, `status`, `reason`, and `error_j` (quadrature error estimate in joules).
Invalid intervals or `max_dt` outside `(0,.2]` raise `ValueError`; an invalid
plan/domain may instead return `valid=False`. Ground segments consume no
energy in this model. Battery capacity is 252 Ah at an equivalent constant
85 V: 21.42 kWh or 77,112,000 J. SOC consumption is energy divided by that
equivalent capacity. The elapsed peak measure is time above **92.8 kW shaft
power** (16×5.8 kW), and the world retains the 120 s cumulative terminal-visit
budget across maneuvers. Battery power is not shaft power, and emergency cost
does not multiply either quantity.

## Demand, snapshots, and the scheduler boundary

[demand.py](../../terminal_marl/core/rotating_slots/demand.py) exposes frozen
`ArrivalRequest(id, release_time, gate_id, initial_soc, service_class,
vehicle_profile_id="eh216s_reference_pm16_v1_0")` and
`ArrivalManifest(requests=(), seed=None, rate_per_hour=None,
arrival_window=3600.0)`. Requests are sorted by `(release_time, id)` and IDs must
be unique. `ArrivalManifest.generate(seed=11, rate_per_hour=60.0,
arrival_window=3600.0)` accepts the frozen seeds 11/23/47/71/101 and rates
60/120/180 per hour. It produces the fixed Poisson, gate, truncated-normal SOC,
and service-class workload. `to_dict()` is the offline/audit serialization.

The manifest owner can inspect the full pre-generated workload offline. The
scheduler receives only revealed state through
[world.py](../../terminal_marl/core/rotating_slots/world.py):

```text
RotatingSlotWorld(config=None, manifest=None, physics_dt=.2,
                  scheduler_period=5., cache_enabled=True, state_sink=None)
world.advance(until, decision_callback=None)  # -> WorldSnapshot
world.snapshot()                            # -> WorldSnapshot
```

`WorldSnapshot` is frozen and contains `time`, tuple `aircraft`, tuple
`requests`, tuple `slots`, tuple `pads`, immutable nested `metrics`, and `failed`.
`to_dict()` returns serializable copies. Aircraft are frozen `AircraftState`
records. Relevant fields are identity/status, `soc`, `p/v/a/jerk`,
`service_class`, `ring/slot/pad_id`, `version`, `emergency`, `trajectory_hash`,
`energy_j`, cumulative `peak_seconds`, `release_time`, `touchdown_time`,
`service_end`, `admitted`, `fixture`, elapsed phase times, and
`battery_power_w`. A version change matters when preparing a later plan.

`snapshot.requests` contains revealed requests, including ones already acted
on; it is not an external waiting queue or the unreleased manifest. Check
whether an ID is already in `port.aircraft` before issuing admission again.
`world.snapshot()`, the callback snapshot, `port.snapshot()`, and
`port.metrics()` omit `planned_manifest` and `unreleased` cardinalities.
Raw `world.metrics()` and `world.to_dict()` are owner/audit interfaces retaining
those totals; they are not exposed through the scheduler port.

The callback signature is **`callback(snapshot, port)`**, where `port` is a
`SchedulerPort`, not the raw world. Its read properties are `time`, immutable
`config`, and read-only copied mappings `aircraft`, `active`, `soft`, and
`requests`. Mapping values are the frozen state/plan/request records. It has
no public `manifest`, event history, recording history, or raw-world forwarding.
This is an API visibility boundary, not a Python security sandbox.

Callbacks occur after same-time settlement/eligibility processing at decision
events and scheduler heartbeats. The default scheduler period is 5 s;
execution uses ≤0.2 s steps plus exact intermediate events. Automatic
deterministic emergency handling runs before the callback when
`world.auto_emergency` is true (the default). Callback return values are ignored.
A request left undecided after its release callback is rerouted at that
timestamp; ordinary demand does not wait unadmitted at the boundary.

## Commands and resource dependencies

The port delegates these public operations to the executor:

| Command | Normal return and effect |
|---|---|
| `submit(plans, *, cancel=(), simulated_latency=0.)` | `ValidationResult` with `ACCEPTED`/`REJECTED`; atomically validates replacements/cancellations, source/version, energy, resource calendars, and affected traffic. Plans due now start; future plans remain soft. |
| `cancel(aircraft_id)` | `ValidationResult`; equivalent to `submit((), cancel=(aircraft_id,))`. Started maneuvers and protected release dependencies cannot simply be removed. |
| `admit(request_id, plan=None)` | `ValidationResult`; admission must occur at the revealed request's release time. With no plan it searches inbound candidates; with a plan it validates that candidate. It includes a single-aircraft normal free-flow reserve screen and traffic/resource checks. Exhausting the candidates reroutes the request and returns the last rejection. |
| `reroute(request_id, reason="NO_PLAN_FOUND")` | `ValidationResult`; accepts only a revealed request not already represented in aircraft state, creates its `REROUTED` outcome, and charges its fixed cost. |
| `try_final(aircraft, emergency=False)` | `bool`; deterministic six-pad attempt using the given `AircraftState`, generation, and ordinary joint `submit` checks. |
| `reject(reason, aircraft_id=None, **details)` | Rejected `ValidationResult`; records a diagnostic candidate rejection. It does not execute the candidate. |

World command results use the same `ValidationResult` class as safety results,
but command status strings are `ACCEPTED`/`REJECTED`. Inspect `accepted`,
`reason`, and `details`; generation feasibility and command acceptance are
different decisions. `simulated_latency` is a diagnostic injection that first
advances time and can make a stale command invalid; production latency is zero.
Commands/execution can also raise the explicit exceptions described below.

[resources.py](../../terminal_marl/core/rotating_slots/resources.py) defines
`SlotCalendar`, `PadCalendar`, and frozen `Reservation(resource, aircraft_id,
start, end=inf, kind="target", plan_hash=None, committed=False,
dependencies=())`. Slot resources are `(ring, slot)` tuples; pad resources are
integer IDs. Intervals are half open `[start,end)`. `reservations` returns a
tuple; `active(resource,time)` and `conflicts(...)` return matching tuples.
Calendars themselves are mutable owner-side objects. `clone`, `reserve`,
`release`, `mark_committed`, `remove_plan`, and rollback `transaction()` are
available; overlapping reservations raise `ResourceConflict`, and protected
removal raises `ResourceDependencyError`. Serialized infinite ends are `null`.

The world derives source release and target capture times from continuous
protection checks, rather than freeing a source slot at maneuver start.
Reservation kinds include `occupied`, `source`, `target`, `service`, `final`,
and `emergency`. `active`/`soft` distinguish the current immutable commitment
from an unstarted editable plan. Started target users record the release-plan
hashes on which their safety depends. Removing such a release is rejected;
a replacement may redirect the dependency only when it preserves the release
source and clears no later, with the replacement batch revalidated.

Before binding a future release to a new dependent admission/start or a
protected replacement, the executor traverses that release's future support
chain. It forecasts SOC using the exact remaining committed-plan energy and
holding energy, then applies the production `qE`/`qHold` eligibility check at
the current active plan's target time (when present) and the scheduled start.
A predictable eligibility trigger rejects the binding with
`FUTURE_EMERGENCY_DISPOSITION`; its witness records the supporting aircraft,
trajectory hash, checkpoint/time, predicted SOC, `q_e`, and `q_hold`.
Unprotected soft plans remain editable and preemptible. This is a screen at
those explicit timestamps, not a continuous proof of future emergency
ineligibility. The dependency-conflict guard below remains a software stop,
not an observed physical failure.

Normal finals reserve the pad from final capture/descent start through
touchdown plus service, including future availability. Emergency finals require
a currently free, uncommitted pad and reserve from actual final start through
service completion. Unstarted ordinary soft reservations can be preempted only
when dependency integrity permits. Started maneuvers and ground service are
not preempted. Short/standard/long service lasts 60/120/240 s, including ground
clearance; there is no independent emergency pad.

This small command example uses actual release/admission APIs without revealing
future requests to its callback:

```python
from terminal_marl.core.rotating_slots.demand import ArrivalManifest, ArrivalRequest
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots import generate_inbound

manifest = ArrivalManifest((ArrivalRequest("r", 0., 0, .85, "short"),), seed=11)
world = RotatingSlotWorld(manifest=manifest)
decisions = []

def on_decision(snapshot, port):
    for request in snapshot.requests:
        if request.id in port.aircraft:
            continue
        candidate = generate_inbound(request.id, request.gate_id, 1, port.time,
                                     config=port.config, durations=[55.])
        result = (port.admit(request.id, candidate.plan) if candidate.accepted
                  else port.reroute(request.id, "EXAMPLE_CANDIDATE_NOT_FOUND"))
        decisions.append((request.id, result.accepted, result.reason))

snapshot = world.advance(0., on_decision)
print(snapshot.time, decisions)
```

## Emergency, physical failure, and software stops

`EmergencyStatus` fields are `eligible`, `declared`, `pending`,
`declaration_time/ring`, `start_time/ring`, `charged`, `state`, `q_e`,
`reference_missing`, and `deadline_missed`. Eligibility is checked from SOC,
the dynamically feasible direct-rescue energy reference, and ten seconds of
holding energy. A missing reference is recorded; it is not an immediate safe
landing certificate. During a committed maneuver an alert is pending and
disposition is evaluated at the endpoint. Declaration is latched. The permitted
immediate inner normal-final exception is handled before emergency declaration.

The emergency cost is charged once at the **actual** emergency-final start,
using its start ring: inner/middle/outer 1200/2400/3600 equivalent delay seconds.
Eligibility, declaration, soft planning, and candidate rejection alone do not
charge that cost. Actual power and joules are unchanged by the cost. Normal
touchdown requires SOC ≥0.25; emergency touchdown requires physical energy
feasibility and may be reported below that reserve.

Airborne statuses are `INBOUND`, `HOLD_OUTER/MIDDLE/INNER`, `TRANSFER_O_M/M_I`,
`FINAL_NORMAL`, `FINAL_EMERGENCY`, and `VERTICAL_DESCENT`. `LANDED_SERVICE` and
`CLEARED` both mean physically landed and are counted once. Actual energy
failure or executed hard separation violation stops the episode, gives the
affected aircraft `ENERGY_FAILURE` or
`EXECUTED_HARD_SEPARATION_VIOLATION`, and changes remaining airborne aircraft to
`CENSORED_DUE_TO_FAILURE`. Frozen positions remain physical bodies in replay.
Unreleased manifest demand stays unreleased; it is not silently admitted or
counted as landed. Failure costs and remaining truncated cost are reported as
such, with a lower-bound flag. An execution safety `UNKNOWN` raises an
`EXECUTION_SAFETY_UNKNOWN` runtime error rather than executing an uncertified
interval.

The current `EmergencyDependencyConflict(RuntimeError)` guard is distinct from
physical failure. When emergency disposition cannot cancel/replace a soft
release on which a started maneuver depends, the executor preserves the old
reservation, records `emergency_dependency_conflict`, records the state, and
raises with `aircraft_id`, `reason`, and `details` (including dependent started
plans). There is no general recovery handler inside `advance`. A downstream
runner must treat this exception as a software stop and preserve its evidence;
it must not silently violate the dependency or label the exception itself an
observed energy/separation failure. This describes the guard's behavior, not
a claim that its occurrence is impossible in ordinary callback operation.

## Replay schema and bounded recording

[replay.py](../../terminal_marl/core/rotating_slots/replay.py) exports:

```text
ReplayBundle.write(path, *, world=None, record=None, config_path=DEFAULT_CONFIG,
                   code_commit, assertions=(), command="", seed=None, eager=None)
ReplayBundle.load(path)      # eager immutable local replay
ReplayBundle.validate(path)  # lightweight integrity-checked ReplayArchive
BufferedStateJSONLSink(path)
```

The writer creates a new directory and never overwrites an existing archive.
Supply either a world or its exported record. `replay.json` has schema
`FT-RS-S1-1.0-ReplayBundle`, code commit, command, seed, semantic config hash,
raw config SHA-256, trajectory hash list, member hashes, actual sample interval,
pose/telemetry source labels, and optional state count. The members are:

| Member | Meaning |
|---|---|
| `config.yaml` | Exact original configuration bytes. |
| `manifest.json` | Offline full manifest, not scheduler observation. |
| `trajectories.json` | Authoritative plan parameters keyed by verified hash. |
| `states.jsonl` | Complete sampled states plus exact-event states; one final frame per timestamp. |
| `events.jsonl`, `rejections.jsonl` | Actual event records and separately identified diagnostic rejections. |
| `metrics.json`, `assertions.json` | Episode/audit metrics and supplied automatic assertion records. |
| `visual_acceptance.json` | Initial researcher acceptance `PENDING`; automation is not observation. |
| `screenshots/` | Viewer screenshot destination; initially empty. |

Events are dictionaries with `time`, `type`, `aircraft_id`, and `details`.
The world retains them as an audit list; there is no separately exported
`EventLog` class. The scheduler port does not expose that audit history.

`BufferedStateJSONLSink` is callable with a frame, exposes `path` and
`frame_count`, and supports reentrant `flush()`, `close()`, and context-manager
use. It buffers one timestamp; subsequent updates at that same timestamp replace
the pending/latest flushed line. A forward timestamp finalizes the previous
frame. With a sink, the world keeps only its latest complete frame in `records`.
`world.to_dict()` flushes and includes the full `states_jsonl` path and count;
generic callbacks without a readable history path are not accepted as complete
streamed replay input. Keep the spool outside the new archive directory.

Streamed writing copies and hashes state bytes in bounded chunks and normally
returns `ReplayArchive(path, index, state_count, time_bounds)`. `validate()`
checks integrity/configuration/plans and state ordering without eager history
loading; `archive.open()` or `load()` opts into eager local observation.
Nonstreamed writing defaults to a loaded `ReplayBundle`; `eager` can select the
return behavior explicitly. Loaded bundle data are immutable nested mappings
and tuples; the files remain provenance artifacts and should not be edited in
place. `sampled_state(t)` returns the last recorded state at/before the query,
`plan(hash)` reconstructs its authoritative plan, and `curves(id)` returns
logged speed/acceleration/jerk/power/SOC values. Arbitrary-time replay poses come
from `TrajectoryPlan.state(t)`, with no smoothing or physics regeneration.

An empty-world recording example avoids fixture seeding or an implicit policy.
Both output paths must be new; replace `CODE_COMMIT` with the actual frozen
revision used to run it:

```python
from pathlib import Path
from terminal_marl.core.rotating_slots.world import RotatingSlotWorld
from terminal_marl.core.rotating_slots.replay import BufferedStateJSONLSink, ReplayBundle

output = Path("artifacts/rotating_slots/api_example")
with BufferedStateJSONLSink(output / "states_source.jsonl") as sink:
    world = RotatingSlotWorld(state_sink=sink)
    world.advance(.4)
    archive = ReplayBundle.write(output / "replay", world=world,
        code_commit="CODE_COMMIT", command="API empty-world example", seed=11)
    print(archive.state_count, archive.time_bounds)
```

## Diagnostic entry points and model boundary

`seed_hold(aircraft_id, ring="outer", slot=0, soc=.75,
service_class="standard", emergency=False)` and
`seed_service(aircraft_id, pad_id=0, remaining=60., service_class="short", soc=.5)`
are **named test-fixture entry points** returning `AircraftState`. They are
restricted to time zero before initial settlement and perform separation/
resource checks. They are not production admission and cannot justify spawning
ordinary requests directly into slots or manufacturing outcomes. Fixture seeds
are explicitly counted separately from manifest arrivals in audit metrics.

[DeterministicDriver](../../terminal_marl/core/rotating_slots/driver.py) and
`run_deterministic(world, until=5400.)` are deterministic acceptance/diagnostic
drivers, not a selected research baseline or a stage-2 optimizer. The public
snapshot, candidate, validation, and command interfaces above are the available
integration boundary; no training result or optimizer superiority follows from
using them.

The model is a deterministic performance-constrained point-mass surrogate:
flat terrain, zero wind/noise, no full 6DOF/contact dynamics, and no OEM-calibrated
EH216-S energy prediction. The equivalent-voltage energy model omits voltage
sag, SOH, temperature, auxiliary loads, regeneration, charging, and ground-service
energy. Service classes, demand, initial SOC, emergency trigger/costs, and several
motion limits are frozen research assumptions. Finite candidate search and
minimum-distance refinement can return unresolved outcomes. See
[sources.md](sources.md) for REF/LIT-PROXY/DESIGN distinctions and the reconstructed
appendix reference-program limitation. Acceptance must be read from actual
automatic evidence and a separate researcher observation record.
