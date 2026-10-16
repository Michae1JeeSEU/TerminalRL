# FT-RS-S1-1.0 executable qualification

The normative catalogue in `new10.16/acceptance_catalog_v1_0.json` contains 94 requirements. It is not a result file. The runner executes concrete fixtures against `terminal_marl.core.rotating_slots`; unselected cases remain `NOT_RUN`, missing prescribed fixtures are `BLOCKED`, and assertion failures are `FAIL`.

## Commands

Run from the repository root with the project Python environment:

```powershell
venv\Scripts\python.exe -m terminal_marl.scripts.run_stage1_tests --output artifacts/rotating_slots/catalogue_run --workers 4
venv\Scripts\python.exe -m terminal_marl.scripts.run_stage1_tests --output artifacts/rotating_slots/selected_run --cases A01,D01,D02,D04,G08 --workers 2
venv\Scripts\python.exe -m terminal_marl.scripts.run_stage1_scenario full_lifecycle --output artifacts/rotating_slots/lifecycle_run
venv\Scripts\python.exe -m terminal_marl.scripts.run_stage1_diagnostics --output artifacts/rotating_slots/diagnostic_pilot --workers 2 --pilot traffic --pilot-window 0.4 --pilot-drain 0.2
```

Use a fresh output directory for each invocation. Existing replay archives are never overwritten. The complete catalogue includes expensive high occupancy benchmarks; do not mistake it for a quick unit test.

Server execution uses `ssh lab-server`, the synchronized project directory and an output directory below `$HOME/training_outputs/terminal`. Supply the local commit with `--code-commit COMMIT` or `TERMINAL_CODE_COMMIT`. The server runner does not invoke Git. Each job hashes the actual source files and frozen configuration, using platform independent relative paths.

## Evidence and status

Every executed case has a directory containing:

- `fixture.json`: seed, ordered production operations, concrete inputs and candidate plans.
- `assertions.json`: every executed assertion, expected value, actual value and result.
- `observations.json`, `execution.log` and any exception traceback.
- `environment.json`: interpreter, CPU, thread settings, command, commit and source/configuration hashes.
- `result.json`: case status and absolute evidence paths.
- Content verified replay bundles containing authoritative trajectories, exact events, recorded states, metrics, assertions and a pending human acceptance record.

Analytic negative fixtures are explicitly marked as fixtures rather than physical flight outcomes; unsupported power/SOC telemetry is absent. Positive stable analytic fixtures use production constant power. Runtime fixtures archive the actual world's plans and telemetry.

The aggregate `results.json` always lists all 94 IDs. `progress.json` records completed cases; long simulation phases also expose per-case progress. Automatic, adapter and researcher statuses are separate. Automatic checks cannot set `STAGE1_ACCEPTED`.

Use `run_stage1_tests --output DIR --merge SERVER_RESULTS_JSON LOCAL_RESULTS_JSON` to aggregate existing reports. The merge accepts executed rows only, rejects conflicting duplicates, requires the canonical configuration and identical source hash/commit across executed rows, preserves original evidence paths, and leaves absent IDs `NOT_RUN`.

`V01`–`V11` execute adapter/data assertions and create replays for the existing PyVista platform. Their `PASS` applies to the automated portion. The aggregate label is `ADAPTER_CHECKS_COMPLETE`; researcher observation remains `PENDING`, and `V12` remains `HUMAN_PENDING`. Each visual case writes `viewer_instructions.json` with the existing `open_stage1_replay` command for top, side and 3D views. No replacement website is created.

Named scenarios: `full_lifecycle`, `all_ring_emergencies`, `six_pads`, `busy_release`, `busy_failure`, `threshold_crossing`, `three_laps_controls`. The lifecycle includes four gate requests and all three service classes. Threshold curves are labelled unexecuted diagnostic candidates. Three laps use exact stable trajectories with 0.2 s constant-power/SOC records.

## Frozen diagnostics

Without `--pilot`, diagnostics run:

- Rates 60/120/180 per hour, seeds 11/23/47/71/101, a 3600 s arrival window and 1800 s drain, initially empty, using the deterministic acceptance driver.
- Stable counts 10/30/60/83, five repeats each, plus 20 additional certified inflight benchmarks. Unavailable inflight fixtures are blocked; rejected attempts do not count as accepted inflight measurements.
- The same committed final and ground entity at `dt=0.2/0.1/0.05`, reporting energy, positions, outcomes, exact events and certified continuous pair minimum.
- Deterministic policy sensitivity at scheduler periods 1/5/10 s, reporting decision counts, CPU, timing and actual outcomes.

`--pilot traffic|benchmark|sweeps|all` is labelled `PILOT_NOT_FORMAL_MATRIX`. Pilot window, drain, count, repeat and duration options validate outputs and estimate resources; they cannot silently replace the formal matrix.

`--workers N` uses independent processes with one BLAS/OpenMP thread each. Jobs have separate directories, and aggregation detects missing, duplicate, unexpected and failed jobs. Pipeline tests compare serial/parallel manifests, metrics, decisions and state counts. States stream to JSONL; replay writing and integrity validation keep RAM bounded.

Measurements include wall time, process CPU, CPU model, dependencies, thread settings, process lifetime peak RSS and simulated seconds per wall second. Summaries report medians and nearest-rank P95 over actual repeats. No absolute performance PASS threshold is invented. `cProfile` uses the process CPU clock; exclusive function times are classified into generation, motion/power, separation, resources, state update, serialization, emergency and other work. Nested work belongs to its callee's category. Timings explicitly include profiler overhead. The viewer is off throughout measurement.

The N83 inflight benchmark uses the demonstrated release chain: Inner0→pad5 at t=0 with 35+13.5 s duration; Middle1→Inner0 at 0.001 s with 40 s duration; Outer1→Middle1 at 0.002 s with 40 s duration; gate0→Outer1 inbound at t=0 with 55 s duration. All 83 initial slots are occupied. Production joint, calendar and continuous checks approve the chain and actual additional admission.

To avoid duplicate benchmark execution, pass `run_stage1_tests --cases G09 --diagnostic-report PATH/diagnostics.json`. G09 verifies the formal report, exactly 40 expected count/mode/repeat IDs, 20 actual accepted inflight rows, source/configuration/commit agreement, measured timings and verified replay archives. The source report path and SHA256 are saved. Missing or stale evidence cannot pass.

Job progress records PID, simulation time, target and metrics. Physical failures remain observed outcomes with truncated/lower-bound metrics; software exceptions and unavailable fixtures have separate statuses. No runner trains a model or chooses an optimizer.

`run_stage1_tests --audit BUNDLE --output DIR` invokes the separate read-only episode audit and saves its report. Its declared scope and issue list determine what was checked; archive integrity alone is not episode acceptance.

F11 includes completed rescues and separate failed episodes: a distinct low-SOC aircraft fails after a valid emergency start. The rescue is correctly censored/unresolved, its one-time fee remains charged, and physical energy is unweighted. This does not claim that an approved deterministic rescue developed an unmodelled midflight power fault.
