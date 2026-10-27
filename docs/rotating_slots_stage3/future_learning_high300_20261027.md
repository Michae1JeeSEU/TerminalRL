# Frozen future-state model: high-traffic 300 s pair

Status: driver and small fixtures prepared for primary review. The new high-traffic
rollouts, archive, result interpretation, image review and replay acceptance are
PENDING. No high-traffic scientific run has been launched by this implementation.

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
the 300 s task timeline and full saved-replay P/V/A checking remain. Image
review is PENDING_PRIMARY_VIEW_IMAGE and replay remains HUMAN_PENDING until
review. One pair supports descriptive findings only, with no broad
generalization, deployment or isolated training-update causal claim.

Validation before launch consists of small fixtures for JSON protocol readback,
case identity, unchanged raw/model/guard copies, a 300 s frozen-runner boundary
loop, second-arm cap preservation and the actual frozen saved-report
collector/text/figures on a small fake saved pair. They invoke no scientific
simulation, solve, inference or fit. The real local s3r24 input is read-only
checked. Original guard-path validation must occur on the server because its
recorded Linux absolute path differs from the local synchronized path.
