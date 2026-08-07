# Event-gated 180s high-level control protocol

This phase adopts **periodic high-level decision-making with event-gated
structural airspace reconfiguration**.

The method remains a fixed-period SB3-compatible high-level MDP:

- metrics sampling: 20 s;
- high-level decision interval: 180 s;
- geometry minimum dwell: 600 s;
- episode duration for PPO pilots: 3600 s, i.e. at least 20 high-level steps.

The high-level policy is still called every 180 s. It is not an asynchronous
event-triggered SMDP. The event gate decides only whether a structural geometry
change is allowed to be applied at that decision boundary.

## Action grouping

The eight-dimensional `DynamicAirspace` action remains incremental and bounded:

| Index | Meaning | Group |
|---:|---|---|
| 0 | outer ring radius residual | structural geometry |
| 1 | inner ring radius residual | structural geometry |
| 2 | outer ring height residual | structural geometry |
| 3 | inner ring height residual | structural geometry |
| 4 | outer ring speed residual | operational |
| 5 | inner ring speed residual | operational |
| 6 | ring headway residual | operational |
| 7 | final headway residual | operational |

When the geometry gate is closed, indices 0--3 are converted to an explicit
masked no-op. Indices 4--7 are not frozen by the geometry gate; they continue
through their existing rate limits, dwell logic, and safety filter.

Rejected geometry requests are not cached. Logs distinguish requested and
applied actions so reward and diagnostics can be computed from actual applied
changes.

## Geometry event gate

`terminal_marl/control/geometry_event_gate.py` owns the gate logic. Gate states
are logged as:

- `NORMAL`;
- `PENDING_TRIGGER`;
- `TRIGGERED`;
- `DWELL_LOCKED`;
- `RECOVERY_ELIGIBLE`.

Inputs are updated every 20 s and include normalized congestion, backlog ratio,
occupancy pressure, backlog slope, normalized conflict, recent arrivals/recent
landings, arrival-service imbalance, and active-count pressure. The gate uses a
transparent pressure score with persistence, hysteresis, and minimum geometry
dwell:

- trigger on: `score > theta_on` for `K_on` consecutive samples;
- trigger off: `score < theta_off` for `K_off` consecutive samples;
- geometry may apply only when event/periodic mode allows it and the 600 s dwell
  is satisfied;
- nominal recovery can become eligible after stable normal traffic when geometry
  deviates from nominal.

## Experimental modes

The event-gated study compares:

| Mode | Geometry permission logic | Low-level behavior |
|---|---|---|
| `fixed_rule` | fixed nominal airspace | rule-only |
| `event_rule` | same event gate, rule chooses direction/magnitude | rule-only |
| `periodic_hrl` | 180 s high-level decision; geometry allowed by dwell only | rule-only/placeholder low-level |
| `event_gated_hrl` | 180 s high-level decision; geometry allowed by event gate and dwell | rule-only/placeholder low-level |

The PPO pilots for `periodic_hrl` and `event_gated_hrl` must start from scratch.
Do not resume 20 s PPO checkpoints after changing the decision interval.

## Training demand mixture

New event-gated PPO presets use an episode-level demand mixture:

```yaml
demand_mixture:
  medium: 0.40
  high: 0.30
  bursty: 0.30
```

`demand_level: medium` remains as a compatibility fallback, but the trainer uses
`demand_mixture` when present.

## Server workflow

Code is edited and committed only in `D:\7_codespace\terminal`. Syncthing keeps
that directory synchronized to `~/projects/terminal` on `lab-server`; do not
manually zip, scp, upload, or unzip source code. Server-side commands should
only run tests, simulations, training, evaluation, and log inspection. All
outputs for this phase must be under:

```text
$HOME/training_outputs/terminal/event_gated_hrl
```

This protocol remains simulation-only and does not connect to ROS2,
Crazyswarm2, or real Crazyflie hardware.
