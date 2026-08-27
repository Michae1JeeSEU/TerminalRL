# D0 Vehicle Dynamics Audit

Status: D0 complete; C0 and low-level PPO remain blocked pending D7.

| Location | Historical behaviour | Required dynamics-v2 change | Risk addressed |
|---|---|---|---|
| `core/config.py` | Generic/legacy limits and 600/500/350/200 m vertical geometry | Fixed EH216-S research profile and 110/100/70/40 m AGL geometry | Profile ambiguity and altitude-envelope violation |
| `core/trajectory_generator.py` | Piecewise linear samples; duration based on average speed | Geometric paths converted to executable position/velocity/acceleration samples | Instantaneous turn/speed changes |
| `envs/parallel_env.py` | Ring speed changed immediately on an intention | D4: 0.2 s plan substeps and shared finite-acceleration holder execution | Policy intent confused with vehicle actuation |
| `core/safety_filter.py` | Independent segment speed/rate checks | D5: authoritative dynamics and plan-content-hash verification before separation | Checked path may be dynamically infeasible |
| `core/runtime_traffic_safety.py` and `core/reconfiguration_safety.py` | Duplicated constant-speed holder prediction | D5: `VehicleDynamics.ring_motion_step` reused by runtime and geometry prediction | Check/execution mismatch |
| `core/merge_scheduler.py` | Geometric duration surrogate | Conservative EH216-S acceleration-aware lower bound; pending exact generator reachability check | Unreachable merge reservation |
| `core/observations.py` | Some scale-normalized features, velocity only as `speed` | D6: `speed` is explicitly horizontal physical speed normalized only by the active profile; checkpoints bind dynamics/profile schemas | Future scaled transfer leakage |

## EH216-S migration report

| Historical canonical value | Frozen replacement | Implementation impact |
|---|---|---|
| 36 m/s max speed | 25.0 m/s operational; 29.17 m/s V_NE; 36.11 m/s design metadata | `VehiclePerformanceProfile` is authoritative |
| 4 m/s² acceleration | 1.5 m/s² horizontal/lateral operational | Parameterizer and ring dynamics |
| 5/7 m/s climb/descent | 4/6 m/s | Vertical profile validation |
| ring speeds 28/24/20 m/s | 22/20/16 m/s | Nominal geometry config |
| gate/rings 600/500/350/200 m | 110/100/70/40 m AGL | AGL envelope validation |
| SOC reserve/emergency .15/.08 | .25/.20 (warning .35) | Normal demand lower bound and urgency semantics |

The only main-simulation profile is `research_evtol`, model `EHang EH216-S`, version `eh216s_research_v1`.  Crazyflie values are intentionally absent.
