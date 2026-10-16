# FT-RS-S1-1.0 stage 1

This is the active implementation track for the frozen rotating-slot contract.
The canonical inputs are the unchanged files in `new10.16/`: the operating
contract, YAML configuration and 94-case acceptance catalog. A normative catalog
is not a test result. Automation, viewer integration and researcher acceptance
are distinct; `visual_acceptance.json` remains **PENDING** until actual observation.

The implementation lives in `terminal_marl/core/rotating_slots/`. The old
`terminal_marl/core/fixed_terminal/` model, its checkpoints, prior staged research
and old viewer launchers are archived research assets, incompatible with this
contract and never automatically loaded by the new stage 1 entry points.
See [migration](migration.md) and [sources and model limits](sources.md).

From the repository root, install `requirements-rotating-slots.txt` in the local
environment. On this Windows checkout use `venv\Scripts\python.exe` in place of
`python` if Python is absent from PATH. The documented module entry points are:

```powershell
python -m terminal_marl.scripts.run_stage1_tests --help
python -m terminal_marl.scripts.run_stage1_scenario --help
python -m terminal_marl.scripts.run_stage1_diagnostics --help
python -m terminal_marl.scripts.open_stage1_replay --help
python -m pytest tests/test_rotating_slots_replay.py -q
python -m terminal_marl.scripts.open_stage1_replay PATH_TO_BUNDLE
python -m terminal_marl.scripts.open_stage1_replay PATH_TO_BUNDLE --time 15 --off-screen --screenshots PATH_TO_BUNDLE/screenshots
```

The test/scenario/diagnostic entry points are authored alongside this adapter;
their generated reports record the actual command, seed, case, checks and artifacts.
No historical test report counts as evidence for this track. Diagnostic outputs
on the CPU server go under `$HOME/training_outputs/terminal`, never the synced
source checkout. Source changes happen locally and reach `ssh lab-server` through
Syncthing; no manual source archives are uploaded.

The actual existing platform is
`terminal_marl.visualization.pyvista_viewer.Terminal3DViewer`, previously fed by
`terminal_marl.visualization.snapshot.build_world_render_snapshot` and immutable
`schema.WorldRenderSnapshot`. The new `RotatingSlotsReplayViewer` subclasses that
same renderer, extends its data adapter, actor overlays and interaction controls,
and opens through `open_stage1_replay`. It does not use the old live interpolation
observer. Every requested aircraft pose comes from the archived
`TrajectoryPlan.state(t)`; sampled telemetry curves remain logged observations.
No smoothing filter, physics stepping or trajectory re-generation occurs on seek.

Controls: pause checkbox/Space, execution/event sample step with Right, previous/
next event with p/n, time slider for seek, 0.25/1/4/10 speed slider, right-click an
aircraft for selection, f to follow, t to cycle all/selected/hidden trajectories,
1/2/3 for top/side/3D, and l to cycle selected-ring/all/hidden slot labels.
All slot states remain visible as coloured points. `set_view(..., vertical_exaggeration=...)` labels the
display scale; minimum distances always use true ENU coordinates. Protection
balls have radius **40 m each**, representing an **80 m centre separation**.
Rejected candidates are explicitly diagnostic and never interpreted as executed.

A ReplayBundle preserves exact config bytes with SHA-256 and semantic hash,
code commit, manifest, authoritative serialized plans and verified plan hashes,
states at the execution sample interval plus exact events, events, rejected
candidate diagnostics, metrics, automatic assertions, screenshot directory and
a separate human observation record. The `replay.json` index hashes core files.
Snapshots and archive objects are immutable. Service/clearance and failure
states come from the recorded lifecycle; no new outcome is inferred by the UI.

Programmatic recording:

```python
ReplayBundle.write(output_dir, world=world, code_commit=commit,
                   assertions=assertion_rows, command=command, seed=seed)
```

The interactive replay itself is a repeatable observation artifact. Screenshots
support inspection but do not establish researcher visual acceptance. Add the
observer, observation time, bundle/commit, viewed events and explicit decisions
to the human record only after the researcher has used the existing platform.
The indexed original bundle record remains immutable; retain a separate signed
observation file alongside it, referring to the bundle hashes, until an explicit
acceptance-record update is requested. Do not hand-edit a hashed archive member.
