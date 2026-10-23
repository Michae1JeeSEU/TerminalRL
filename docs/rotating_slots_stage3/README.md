# Stage3 bounded learning pilot

The canonical numeric settings and dataset plan are the sealed files under
`new10.16/rotating_slots_stage3_learning_v1_0`. The experiment overlay selects
the actual Stage2 `downstream_service_v8`, `rescue_need_v7`, `service_half_v1`,
and `quality_v5` interfaces. Importing these modules never starts a campaign.

Freeze source bytes and the 24-root split locally, after implementation edits:

```powershell
python scripts/stage3/run.py freeze --output artifacts/stage3_source_freeze
```

The source-freeze directory must be visible through the existing Syncthing
chain. On the server, use its existing environment and an output directory
under `$HOME/training_outputs/terminal`. Server code never invokes Git. It
verifies local source hashes before every job. It records actual dependencies,
free physical CPU topology, other pinned jobs, memory, and thread variables;
it does not install missing dependencies or guess versions.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python scripts/stage3/run.py probe --output "$HOME/training_outputs/terminal/STAGE3_RUN"
python scripts/stage3/run.py preflight --output "$HOME/training_outputs/terminal/STAGE3_RUN" --credit-id preflight_external --credit-wall ACTUAL_MEASURED_SECONDS
python scripts/stage3/run.py all --output "$HOME/training_outputs/terminal/STAGE3_RUN"
```

The ledger is created before automatic preflight/testing. External automated
checks use an immutable credit ID and actual wall duration. `pause` and `resume`
explicitly exclude human editing intervals. A restart retains used wall and
phase caps; an interrupted active interval includes downtime. Stage caps are
900/7200/1800/3600/900 seconds, global 14400, compute stop 13500. Independent roots
use up to two fresh worker processes on distinct verified free physical cores,
with one numerical thread and hard parent timeouts. Parallel CPU accounting is
measured once for the whole process tree batch to avoid overlap double counting. Every
evaluation arm uses the same selected free physical core and cold caches. A
completed job receipt prevents replaying that job on resume. A partial raw
label trace requires review before retry, preserving the original evidence.

Dedicated `collect`, `label`, `train`, `evaluate`, `archive`, `credit`, `pause`,
and `resume` stages are available through the same CLI. Targeted checks and a
compatible server binding are prerequisites for collection and training.
Failed or uncovered targeted checks stop `all` before data generation.
After collecting the frozen 24 roots, `label --root-id s3r01` produces one
serialization pilot for inspection. A later `label` reuses its completed receipt
and processes the remaining frozen roots. This does not allocate new roots.

Initial roots use the existing fixture's unchanged SOC/service distributions,
geometry, pad state and safety rules, with independent seed-derived SOC and
service streams. They are marked `SYNTHETIC_INITIAL`. Source metadata are scanned
for seed collisions before outcomes; 24 unique roots are frozen in 16/4/4
splits. Identical t0 states with different manifests cannot count as independent.
Connected root, manifest, and checkpoint ancestry cannot cross splits. Seed71
remains development-only. There is no fabricated historical commitment or ledger.

Raw initial checkpoints, descriptors, pre-query features and facts are separate
from derived labels. The offline probe uses the frozen seed and one in five
opportunities; it records its selected stratum and conditional probability.
There is no live exploration. Each finite reference uses one common materialized
catalog, independent Slack/CP checks, and original prefix validation. Conditional
solves restrict a target aircraft to generated descendants, retaining the original
downstream evaluator, catalog objective and background. UNKNOWN, unsupported
valid branches, ties and uncertified resolves cannot produce negative ranking
labels. Timed auxiliary cost explicitly covers generation plus observed energy,
excluding joint validation. Missing pairs return `DATA_INSUFFICIENT`.
Only the saved maximum eight distinct reference solutions contribute support;
the separately bounded conditional attempts contribute comparable strict pairs.

Training input physically omits test/development contexts. Test labels remain
sealed until a frozen model and normalizer open the one-model evaluation gate.
At most six frozen snapshots rotate three arms; the two reduced arms share
coverage and budget. The serial 60-second pilot uses the first frozen compatible
medium test root (s3r23). Every five simulated seconds it publishes a checkpoint
and boundary metrics; comparison uses the latest common completed boundary if
the hard wall ends early. Observational energy deltas at the unchanged executor
integration boundary record actual holding joules by ring.
Each arm also preserves its actual terminal time, failure/invalid flags, and
metrics separately from the matched completed window; earlier common boundaries
cannot hide a later observed failure. Interrupted unpublished terminal state is
reported as unknown.

The final report separates infrastructure, learned signal and online benefit.
No accuracy statistic alone establishes online benefit. Quality remains a vector
without an authorized degradation tolerance. Missing models/data create explicit
skipped artifacts. `targeted_checks.json` maps actual pytest node names; replay
integration can pass automatically while researcher visual acceptance remains
`HUMAN_PENDING`.
The gate additionally requires every merged executed testcase to pass; a later
run may supersede the same pytest classname/name. Latency summaries use actual
callback `wall_s` and dispatch `dispatch_wall_s`, nearest-rank P50/P95/max, explicit
timeouts, and separate deadline misses. Coverage counts distinguish proposals,
queries, physical attempts, single-valid evidence, and certified support by ring
and construction stage. Full online time includes scheduler and model setup,
also reported separately. Archive renders the actual completed epoch losses to
`model/training_curve.png` when Matplotlib is available; missing history or plot
dependencies receive an explicit status. `DATA_INSUFFICIENT` creates no curve or
weights.

Open a generated replay through the existing viewer:

```bash
python -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController('PATH_TO_REPLAY')).run()"
```

The archive and HUD distinguish computation proposals, physical attempts, and
actual current commits. Proposal records never enter the authoritative trajectory
map or resource calendars. Human observation must be recorded separately.
