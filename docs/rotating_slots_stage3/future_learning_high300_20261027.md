# Frozen future-state model: high-traffic 300 s pair

Status: implementation accepted; 21 local tests and five new server tests passed.
Server tests took 1.185 s and preparation exited zero. The single scientific
H-then-L pair completed 300 s in both arms and archive exited zero. Primary
review finds one extra L normal landing, but higher overall planning/dispatch
work and no stable quality advantage. The timeline is readable with an overlap
of final-task status labels: PRIMARY_VIEWED_WITH_LAYOUT_LIMITATION, not
publication-ready. Human replay acceptance remains HUMAN_PENDING.

The wrapper was committed and pushed as `0c04d08`; its byte-frozen SHA256 is
`b1749ac1eeef91d52b8fe2c04ebb17069409ac4d06dd1197540a530bcadc5495`.
Preparation records execution scope
`be3cca075c098aaff8cff2d0a67d7b8db4fe1adc26d26593fb33278262697343`.
No driver or test bytes were changed during the run.

An initial shell launch at 2026-10-27 07:27:30 UTC (PID 291376) failed before
entering the driver because the redirection target's `logs` directory was
missing. There was no `run.started` marker, scientific child or active run
ledger from that attempt. A subsequent Popen preflight-expression error also
occurred before launch and had no execution effects. These were launch-wrapper
failures, not model or simulation failures. Original launch/PID evidence is
preserved alongside `startup_redirection_failure.json`.

The corrected Popen started at 2026-10-27 07:29:24.711183 UTC: main PID 291508,
H PID 291510, pinned to CPU12. Startup verification observed the 50 s boundary
after approximately 25 s wall. Evidence is retained in
`run_resume_launch.log` and `run_resume_main.pid`. This is one scientific
H-then-L run; L is started only by that parent, with no manual additional arm.
The initial estimate was T0 = 900 s, with the first normal progress query planned
for 07:33:09 UTC. Both arms and saved reporting subsequently completed; the
launch-wrapper failures did not create an additional scientific run.

This is one H/L pair using the already trained future-state model, without
retraining, tuning or a model-selection pass. The new output is
`/home/user/training_outputs/terminal/stage3_future_learning_high300_20261027`.
All earlier outputs remain read-only.

The physical input is the original `s3r24`, seed 7325,
`stress36_rate180_v1`, from `stage3_learning_v1_0_20261023`. Its original bytes
contain 36 airborne aircraft (14 outer, 12 middle, 10 inner) and three ground
aircraft, 180 requests/hour, and 167 manifest requests over 3600 s. Thirteen
manifest release times fall at or before 300 s. The driver copies the original
manifest, initial state and t0 checkpoint without changing them; relocated root
metadata retains the original physical identity and digests. Geometry, targets,
admission, emergency handling and energy/safety rules are unchanged.

The execution checkout is explicitly
`/home/user/projects/terminal/.worktrees/future-learning-repair-20261027`:

- Core commit: `79fc55c81c28a2b5e59e28c939d99cf40b438d1b`.
- Core source hash: `52852f54bd7372a44f4ff7053ef94dcf67522b05779e03f9643c7167f9ccd524`, 106 frozen files.
- Frozen model SHA256: `29d942674fad65805bd33598aecd880add6d9cad223b1fd8f11d2b90c6ac292d`.
- Normalizer SHA256: `6df1cc091824ac4bf066ad8ff8039bd542310a14e6f3e46630a39481384abeb0`.

`scripts/stage3/run_future_learning_high300.py` imports the frozen existing
runner through that explicit checkout. It changes only the imported runner's
case, horizon, step, checkpoint interval, CPU and job-cap constants. It calls
the existing `run_job` with its own high-traffic execution verifier. The wrapper
hash is bound separately from the original core source hash and original model
fingerprint. No frozen checkout or original algorithm source is edited.

The source inventory is checked against the fixed core identity and its file
hashes. Preparation verifies the model in its original parent directory, where
its exclusive training guard points to the original training output. The model,
normalizer, card, binding, source freeze and guard are copied byte-for-byte.
The copied guard is provenance, not a claim of training in this new directory;
its `output` field is never rewritten. The new scope says
`FROZEN_NO_TRAINING`. Numerical package versions must match the model parent's
record and all five numerical thread environment variables must equal one
before Python starts.

H then L run serially on the same verified-free physical CPU12. Serial execution
keeps the actual wall-latency comparison on the same core; there are no independent
cases to parallelize in this two-arm comparison. Lead/heartbeat/preparation
margin/search reserve remain 20/5/1/4 s. Candidate-active/CP/hint/soft/watchdog
limits remain 6/8/3/30/60 s, attempts remain 32 per aircraft and 1024 total,
and ranking remains `comparable_group_lazy_v2`. The network is the same frozen
119555-parameter SCHEMA_V2 model with 66 query values and three heads.

Each arm advances at 5 s saved boundaries through exactly 300 simulated seconds,
with checkpoints at 0, 60, 120, 180, 240 and 300. `settle_only_at_end=True`
preserves endpoint censoring: dispatch/delivery exactly at 300 is pending and
is not executed by an extra step. There is no additional 600 s rollout.

The existing CampaignLedger is reused. Total wall allowance is 4205 s:
preflight 300, evaluation 3605 and archive 300, with compute stop at 3905 s.
Each H/L subprocess has a fixed 1800 s watchdog. Before each arm, including the
second, at least 1801 s of evaluation allowance must remain; the cap cannot be
silently shortened. Preparation time is credited once; review pauses exclude
human review time. There are no retries or automatic stage transitions.
Initial estimated total runtime is 900 s: the previous medium 120 s pair took
about 198 s wall, and `198 × 2.5 × 1.5 ≈ 743 s`, with additional margin.
Actual progress will replace this estimate; this estimate does not extend caps.

Stages are separate primary-reviewed invocations:

```text
python scripts/stage3/run_future_learning_high300.py prepare --checkout <frozen-checkout>
python scripts/stage3/run_future_learning_high300.py run --checkout <frozen-checkout>
python scripts/stage3/run_future_learning_high300.py archive --checkout <frozen-checkout>
```

Internal `job` is used only by the parent run. Preparing does not start H, and
completing L does not start archive. Git stays local and source synchronization
uses the established Syncthing path; this driver runs no Git or transfer command.

Archive uses the existing saved-report collector, replay P/V/A and C2 checks,
CSV exporters and timeline figures. Its two legacy root-identity comparisons
are explicitly recomputed from the actual s3r24 root, both saved results and
the new scope; no fake s3r23 alias is created. Summary/title/endpoint wording
uses 300 s. The old contact sheet sampled only 0/60/120, so it is omitted;
the 300 s task timeline and full saved-replay P/V/A checking remain. The primary
viewed the numeric/time/landing trends: they are readable, while the right-side
status labels for final tasks overlap. Image review is
PRIMARY_VIEWED_WITH_LAYOUT_LIMITATION; no publication-ready claim or redraw is
made. Replay remains HUMAN_PENDING. One pair supports descriptive findings only, with no broad
generalization, deployment or isolated training-update causal claim.

Validation before launch consists of small fixtures for JSON protocol readback,
case identity, unchanged raw/model/guard copies, a 300 s frozen-runner boundary
loop, second-arm cap preservation and the actual frozen saved-report
collector/text/figures on a small fake saved pair. They invoke no scientific
simulation, solve, inference or fit. The real local s3r24 input is read-only
checked. Original guard-path validation must occur on the server because its
recorded Linux absolute path differs from the local synchronized path.

## Completed pair and primary interpretation

The frozen model was not retrained or tuned. Core/model hashes, 20/5/1/4 timing,
CPU12 and budgets stayed unchanged. Both arms admitted 49 aircraft: the initial
36 airborne aircraft plus 13 new requests. All normal landings came from the
initial cohort; all 13 new requests remained airborne at 300 s. Neither arm had
an emergency landing or failure.

| Saved result | H | L |
| --- | ---: | ---: |
| Normal landings / airborne pending | 14 / 35 | 15 / 34 |
| All-task normal planning work (s) | 189.537665689 | 196.209438064 |
| Dispatch work (s) | 208.654763247 | 227.677272771 |
| Finished-task P95 (s; samples) | 15.9125 (13) | 15.8024 (14) |
| Joined / finished nonempty groups | 8 / 11 | 8 / 11 |
| Pending nonempty groups / submitted plans | 1 / 58 | 1 / 53 |
| Terminal energy (J) | 849682830.4738 | 860493251.5005 |
| Holding (aircraft s) | 7754.51167 | 8040.67272 |
| Physical attempts | 829 | 707 |
| CP solve (s) | 24.9756 | 17.8549 |
| Trajectory validation (s) | 53.4995 | 58.5525 |
| Prefix validation (s) | 13.8051 | 18.0888 |

L has one additional normal landing, but all-task normal planning work is 3.52%
higher and dispatch work 9.12% higher: there is no overall online acceleration.
Its slightly lower P95 uses 14 completed samples versus H13. Terminal energy is
1.27% higher and holding 3.69% higher. Normal landings at 60/120/180/240/300 s
are H 0/6/8/11/14 and L 0/6/6/11/15. The lead changes and does not establish a
stable quality advantage. `normal_lifecycle` all-task work is the planning
measure; `full_online_wall_s` includes setup/checkpoint I/O.

Lower attempts and CP work do not establish efficient candidate identification.
L feature/inference work is 4.01086 + 18.79356 = 22.8044 s, with 499 rankings and
zero cache hits, and greater trajectory/prefix validation work. Its two scheduler
fallbacks are NO_SOLVER_TIME/HINT_BUDGET; the model fallback map is empty.

Each arm has two ready-deadline rejections. H additionally has one
ENERGY_BOUND_CHANGED and L one TRAFFIC_CHANGED_AFTER_DEADLINE, with no packet
expiry. One late H task and two late L tasks already have search certificates but incur online-service
delays of about H9.90/L12.01 and L8.68 s before preparation. They cannot be
attributed solely to CP or the model. Revalidation is H4 rounds (3 certified,
1 late), about 0.1321 s, and L2 certified rounds, about 0.1590 s; it is not a
dominant total-time term.

The primary checked the shared emergency-service semantics: `handle_emergencies`
includes rescue-opportunity search/checking for low-SOC or warning/eligible
aircraft even when `required=False` is returned. Its measured H10.6117/L21.4542 s
is shared rescue search/checking overhead, not an emergency-landing count.
Neither arm declared, triggered or left unresolved an emergency. Terminal
`rescue_warning` H1/L3 are state counts, not cumulative events. The additional
10.84 s of service work is observed under different physical states and does
not isolate a model effect. No emergency rule changed.

Saved replay validation, hashes and P/V/A checks pass: H16906/L16362 samples,
H58/L53 C2 checks and zero start-time mismatches. Source/model/manifest/initial
identity checks agree; different executed H/L trajectory sets are expected and
are not an identity failure. Human replay acceptance remains HUMAN_PENDING.

`artifacts/finalize_future_learning_high300_20261027.py` applies this supplied
post-archive primary interpretation to derived FINAL_REPORT/README/analysis/QA
metadata, preserving original report copies and the historical archive receipt.
It is idempotent, creates `primary_review.json`, and changes no raw result,
replay, driver or test. No additional experiment or policy change is authorized.
