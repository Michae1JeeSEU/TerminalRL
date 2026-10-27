# Future-state learning: 2026-10-27

Status: implementation and small fixtures only. New acquisition, labels, formal fit,
120 s H/L evaluation, replay review and figure review are **PENDING**. No new
performance result is asserted here.

The accepted search-handoff records motivate this bounded update. In
`artifacts/search_progress_analysis_20261027.csv`, closed-loop H has six callbacks
and L has five. First nonempty certificates occur at H 6.721631–7.531727 s and
L 6.625655–7.295599 s; returns occur at H 12.167859–12.882911 s and
L 11.357687–12.354470 s. Within the same first catalog, four of six H and two of
five L callbacks subsequently improve the certified objective. Normal-landing
layers often improve, so these records do not support uniformly reducing CP time.
The two H NEW_TRAFFIC rejections occur after preparation and exceed the
revalidation deadline; they are not attributed to the learned model. Recorded L
feature work is 2.315718473 s and inference is 8.815807075 s, with 334 ranking
calls and nine cache hits. The old data contain nine contribution records and
zero contribution preference pairs. These are preexisting observations, not
results from this campaign.

The execution entry is `scripts/stage3/run_future_learning.py`. Each of `prepare`,
`collect`, `train`, `evaluate`, and `archive` is a separate invocation and stops
for primary review. Internal `job` invocations are watchdog children of those
stages, not an automatic next-stage workflow. The new output is
`/home/user/training_outputs/terminal/stage3_future_learning_20261027`.
The old online-v2 and learning-v1 archives are read-only. The latest accepted
search-handoff execution source freeze is the physical-code baseline, because
the older model source predates already accepted latency-interface changes.
All execution imports use the explicit checkout and primary-reviewed source
freeze. No server Git operation or manual source transfer is part of this flow.

The normal-planning lead/heartbeat/margin/search reserve stay 20/5/1/4 s. Both
H and L share candidate-active/CP/hint/soft/watchdog limits of 6/8/3/30/60 s,
32 attempts per aircraft and 1024 total. The network stays at 119555 parameters,
SCHEMA_V2, 66 query values and three heads. Runtime ranking explicitly uses
`comparable_group_lazy_v2`; the old ranking scope and old model-hash verifier
remain the default in the original latency runner.

The campaign reuses `CampaignLedger`, subprocess watchdogs, stage guards and the
exclusive formal-training guard. Its total wall cap is 7200 s: acquisition and
labels 3600, training 900, evaluation 1800, archive 300, preflight 600. Preparation
is measured once against preflight; review pauses exclude human review time.
Failed or insufficient stages retain their evidence and cannot be retried to
obtain more collection or training time.

Collection uses exactly these original roots:

| Root | Seed | Split | Profile |
| --- | ---: | --- | --- |
| s3r01 | 7302 | train | diagnostic18_rate120_v1 |
| s3r02 | 7303 | train | medium24_rate150_v1 |
| s3r17 | 7318 | validation | medium24_rate150_v1 |

Each root runs one contiguous H latency-aware rollout to 90 physical seconds,
advancing first to 30 and then to 90. Both physical checkpoints are saved before
offline labeling. This preserves ordinary arrivals and existing emergency
handling, and prevents label wall time from affecting acquisition timestamps.
Cold label roots discard normal scheduler search/pending-task state while
retaining the world checkpoint and all physical commitments. They project only
the already visible approved trajectories with
`PlanningTiming.from_lead(source.time, 20, 1)` and `project_visible_snapshot`.
Thus source times 30/90 become effective states 50/110. The labels retain source
time/version, effective time/version, visible aircraft IDs, the projection
protocol and an explicit no-future-manifest-access flag. A small fixture rejects
additional projected aircraft and makes manifest access fail.

There are six labels under `bounded_joint_contribution_v2`. The unchanged label
limits remain 180 s per snapshot and eight conditional solves. Each complete
root acquisition-plus-label job has a 900 s watchdog. CPU18 and CPU22 execute two
waves (s3r01/s3r02, then s3r17); the primary checks actual server resources before
launch. There is no supplementary acquisition on missing rank labels.

Dataset construction selects each descriptor's first feature observation and
joins facts by the exact `features_before_query_ref` (`FIRST_QUERY_EXACT_REF`).
Root ancestry fixes the split. New contribution preferences replace old
conditional-quality preferences for the same unordered descriptor pair; opposed
targets cannot coexist. Four distinct inclusion/removal solve references,
content identities, intervention bindings, objective differences and comparison
scope are rechecked before contribution pairs enter training.

Only the old dataset's train split is migrated through the existing migration
tool; numeric V2 features and fact values/references are asserted unchanged.
Originally complete costs are retained only when they already carry
`cost_complete` and `GENERATOR_ENERGY_TERMINAL_VALIDATION_V2`; old partial costs
remain masked. Original fingerprint and earlier provenance chains are preserved.
Old contexts explicitly carry `CURRENT_STATE_V2`; fresh contexts carry
`planning_effective_state_v1`. The model binding records
`MIXED_TRAIN_FUTURE_TARGET_V1` plus the runtime projection version. Old validation
is excluded; only new projected s3r17 contexts select the model.
`dataset_index.json` records old-current train, new-future train and new-future
validation counts, pair sources, contribution masks and insufficiency reasons.
READY requires at least one new future train contribution pair and at least one
new future validation rank pair. Otherwise the campaign stops DATA_INSUFFICIENT.
s3r23 never enters training.

A read-only check against the actual saved 497369093-byte online-v2 dataset
completed locally without fitting or solving. The original 1223 contexts/203
pairs become 968 train contexts/165 pairs; `_groups` accepts all 968 groups.
There are 583 preserved originally complete cost facts and 5510 masked partial
facts. All 165 inherited pairs use the conditional-objective source. The source
file SHA256 is unchanged before/after:
`53eb66305a74c20b1db70b5e2c526edd603ce40d834d70aac26aba6f9966798c`.

Training invokes the original guarded `train_model` once: seed 20261023,
AdamW settings unchanged, at most 30 epochs, patience five, same validation rank
selection, no tuning. The completed unique model card records actual model and
normalizer SHA256 values plus the dataset SHA256. Evaluation verifies that card,
fingerprint, seed, architecture size and exclusive training guard; it never
borrows the previous model's hashes.

Evaluation reuses `run_latency_interface.run_job` and its complete reporting
fields: cost clocks, first certificate/improvement/return, preparation,
revalidation, expiry/lateness, actual joins, landings, energy and replay.
It runs only s3r23, seed 7324, for 120 simulated seconds, H then L serially on
CPU12, with equal 900 s watchdogs inside the 1800 s evaluation phase. This is
DEVELOPMENT_SEEN evidence with no timing sweep or generalization claim.
Arm receipts and both run/evaluate terminal markers are retained.

After review, invoke each stage separately, supplying the reviewed checkout and
freeze paths (placeholders below are intentionally not launch commands):

```text
python <checkout>/scripts/stage3/run_future_learning.py prepare --checkout <checkout> --source-freeze <reviewed-source-freeze.json> --baseline-execution-freeze <accepted-search-handoff-source-freeze.json>
python <checkout>/scripts/stage3/run_future_learning.py collect --checkout <checkout>
python <checkout>/scripts/stage3/run_future_learning.py train --checkout <checkout>
python <checkout>/scripts/stage3/run_future_learning.py evaluate --checkout <checkout>
python <checkout>/scripts/stage3/run_future_learning.py archive --checkout <checkout>
```

Archive uses saved artifacts only: summary, existing
`report_latency_interface.py`, replay hash and P/V/A checks, task timelines,
quality CSVs, PNG/SVG figures and saved-position contact sheets. It does not run
a simulator. Missing arm results are reported as an incomplete pair rather than
invented. A training curve may be derived later from the saved history after
primary review. Exported image visual QA and replay human acceptance remain
PENDING_PRIMARY_VIEW_IMAGE and HUMAN_PENDING.
