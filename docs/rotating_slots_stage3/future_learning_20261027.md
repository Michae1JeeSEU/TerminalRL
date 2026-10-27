# Future-state learning: 2026-10-27

Status: acquisition, the single guarded pilot fit, both 120 s H/L rollouts and
saved-artifact archive completed. The primary reviewed the results, timeline,
contact sheet and actual training curve. This one DEVELOPMENT_SEEN pair shows a modest
complete-planning cost reduction without a normal-landing throughput gain.
All three image visual reviews are PASSED_PRIMARY_VIEW_IMAGE; replay human
acceptance remains HUMAN_PENDING. No additional fit, evaluation or timing change
was made.

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

The original `2eaf9d98a3793d974214323af459fceb463c37a0` collection startup
(PID 287699) exited during protocol verification: the in-memory root tuple
differed from its JSON list readback. This occurred before any root job or
simulation. Its original log and launch bytes remain archived under
`implementation_repair_precollect`. The implementation repair changes only the
driver's root container to an explicit list and adds JSON roundtrip/readback
tests; serialized protocol values and bytes stay unchanged. Metadata was
rebound to the repaired freeze using the same paused ledger, without resetting
the budget or replacing the failure evidence. This was a startup implementation
failure, not a data or model failure.

The actual collection main process was PID 288529, using repaired commit
`79fc55c81c28a2b5e59e28c939d99cf40b438d1b` and source hash
`52852f54bd7372a44f4ff7053ef94dcf67522b05779e03f9643c7167f9ccd524`.
The earlier model archive remains unchanged.

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
For the actual six labels, acquisition source-time checks pass 6/6 and the
projected-snapshot interface is covered by code tests. Raw `SnapshotStart`
records do not contain a version field; that raw-record version is retained as
unknown rather than inferred or backfilled.

There are six labels under `bounded_joint_contribution_v2`. The unchanged label
limits remain 180 s per snapshot and eight conditional solves. Each complete
root acquisition-plus-label job has a 900 s watchdog. CPU18 and CPU22 execute two
waves (s3r01/s3r02, then s3r17); the primary checks actual server resources before
launch. There is no supplementary acquisition on missing rank labels.

Actual collection on CPU18/22 retained both checkpoints per root and completed
all six labels:

| Case | Source → effective time (s) | Facts | Complete-cost facts | New pairs |
| --- | --- | ---: | ---: | ---: |
| s3r01_t30 | 30 → 50 | 174 | 134 | 3 |
| s3r01_t90 | 90 → 110 | 210 | 129 | 6 |
| s3r02_t30 | 30 → 50 | 211 | 188 | 0 |
| s3r02_t90 | 90 → 110 | 233 | 186 | 1 |
| s3r17_t30 | 30 → 50 | 223 | 222 | 3 |
| s3r17_t90 | 90 → 110 | 235 | 173 | 6 |

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

The assembled dataset has 1093 contexts and 184 pairs: 968 inherited current-state
train contexts with 165 conditional-quality pairs, 83 new future-state train
contexts with 10 contribution pairs, and 42 new future-state validation contexts
with nine pairs. All 19 new pairs have source
`observed_bounded_joint_contribution_preference`; there are 18 valid masked-in
contribution records. UNKNOWN/censored observations contribute zero negative
labels. Test labels were not read, and inherited validation does not participate
in model selection. The ledger is paused after collection; the primary accepted
these observations before authorizing the single fit.

A read-only comparison of the 19 saved new pairs finds the first differing
delta layer at Ehat for all 10 train pairs, and at risk/Ehat for five/four
validation pairs. These are finite-budget relative comparisons. For example,
all three s3r01_t30 comparisons first differ at Ehat and compare negative deltas:
the preferred value can be −1616072.9 against −17861922.9, meaning the observed
bounded difference is less adverse, not a positive contribution. No pair is
claimed to identify a global oracle or a causal improvement.

A read-only check against the actual saved 497369093-byte online-v2 dataset
completed locally without fitting or solving. The original 1223 contexts/203
pairs become 968 train contexts/165 pairs; `_groups` accepts all 968 groups.
There are 583 preserved originally complete cost facts and 5510 masked partial
facts. All 165 inherited pairs use the conditional-objective source. The source
file SHA256 is unchanged before/after:
`53eb66305a74c20b1db70b5e2c526edd603ce40d834d70aac26aba6f9966798c`.

The authorized training invoked the original guarded `train_model` once: seed 20261023,
AdamW settings unchanged, at most 30 epochs, patience five, same validation rank
selection, no tuning. The completed unique model card records actual model and
normalizer SHA256 values plus the dataset SHA256. Evaluation verifies that card,
fingerprint, seed, architecture size and exclusive training guard; it never
borrows the previous model's hashes.

The unique fit exited zero with subprocess wall time 55.573466533 s. It retained
119555 parameters and seed 20261023, completed 15 epochs and selected epoch 10
at validation rank loss 0.640831053257. The fit used 142 eligible training
groups (including cost-only groups), 175 training pairs, two validation rank
groups and nine validation pairs. Its model SHA256 is
`29d942674fad65805bd33598aecd880add6d9cad223b1fd8f11d2b90c6ac292d`;
normalizer SHA256 is
`6df1cc091824ac4bf066ad8ff8039bd542310a14e6f3e46630a39481384abeb0`.
No tuning or repeat fit followed.

On the saved cost subset, 395 targets have MAE 0.959902894210 versus a
training-mean baseline of 1.046058055366, both in log1p(milliseconds). The MAE
is not 0.96 milliseconds. Cost adjustment changes ordering in six of 37
comparable contexts without clipping. This is a subset diagnostic rather than
evidence of online benefit. The new train contribution pairs distinguish only
the Ehat layer; validation covers risk/Ehat, with nine pairs in only two rank
groups. These limits constrain the supported learning claim.

Evaluation reuses `run_latency_interface.run_job` and its complete reporting
fields: cost clocks, first certificate/improvement/return, preparation,
revalidation, expiry/lateness, actual joins, landings, energy and replay.
It runs only s3r23, seed 7324, for 120 simulated seconds, H then L serially on
CPU12, with equal 900 s watchdogs inside the 1800 s evaluation phase. This is
DEVELOPMENT_SEEN evidence with no timing sweep or generalization claim.
Arm receipts and both run/evaluate terminal markers are retained.

Both arms actually reached 120 s. Authoritative all-task planning work includes
six tasks per arm, including the pending partial task; finished-task P95 uses
five tasks per arm and the nearest-rank rule.

| Saved result | H | L |
| --- | ---: | ---: |
| All-task normal planning work (s) | 81.887650033 | 75.625175985 |
| Finished-task P95 (s) | 15.107181865 | 13.344389986 |
| Dispatch work (s) | 85.044952253 | 78.949730081 |
| Joined / finished nonempty groups | 2 / 4 | 5 / 5 |
| No-prefix / pending tasks | 1 / 1 | 0 / 1 |
| READY_DEADLINE_MISSED | 2 | 0 |
| Normal landings | 6 | 6 |
| Terminal airborne pending | 23 | 23 |
| Terminal energy (J) | 224822657.780 | 223225679.462 |
| Holding (aircraft s) | 1559.741459 | 1885.241459 |
| Physical attempts / descriptor queries | 449 / 528 | 207 / 207 |

There are zero emergency landings and zero energy failures in this pair.
Complete-planning work is 7.6476% lower for L, much less than the candidate-work
reduction. Normal landing throughput is unchanged; terminal energy is slightly
lower with more holding. `normal_lifecycle.full_normal_planning_work` is the
main planning clock. `costs.full_online_wall_s` includes setup and checkpoint I/O
and is not pure planning work; final `all_callbacks` supplies aggregate phases,
attempts and queries. CP work is H/L 12.971925/14.331302 s, prefix validation
10.7964/7.6153 s and trajectory validation 17.0082/14.4363 s. L feature and
inference work is approximately 2.92416 + 10.20343 s.

Both H deadline rejections occurred after previous readiness. At source time
37.587890625, previous readiness was 50.515828962 before deadline 56.587890625;
new-traffic revalidation started at 56.566670201 and completed at 56.593427098,
about 0.005536473 s late after 0.026607508 s of work. At source 83.5, previous
readiness was 97.000200374 before deadline 102.5. req_000002 recertified at
99.728308864, but req_000003 revalidation started at 102.654980060, already after
the deadline. L at source 95 recertified req_000004 at 109.970703801 before
deadline 114 with 0.061253889 s of work. These observations expose late
new-traffic/service scheduling rather than an initial CP search miss. The
supplement explicitly supersedes the generic report's broader deadline wording.

Saved first nonempty certificates occur at H 7.53–10.68 s and L 6.487–7.258 s;
returns occur at H 11.97–13.45 s and L 11.59–12.58 s. Later search still improves
the landing objective in L at source 0 (negative-L 0 to −5) and 95 (0 to −1).
Some callbacks plateau between first certificate and return: L50 for about
5.52 s, H60 for 4.27 s and H83.5 for 4.16 s. This mixed pattern does not support
a blanket CP reduction. H25 returns an empty prefix with a better recorded
Ehat objective after its nonempty certificate; it is retained as objective
selection rather than reclassified as budget failure.

The six distinct L Coverage snapshot IDs, deduplicated across boundary records
and final costs, match six callbacks; all seven lazy-counter fields occur in
all six snapshots. Totals are 406 scored groups, 7364 scored descriptor inputs,
406 consumed batches, zero deferred groups/items and zero cache hits. Misses
are 386 CANDIDATES_CHANGED and 20 NO_BATCH. The run therefore provides no
observed evidence that lazy deferral or batch reuse reduced scoring work.
Historical L has five tasks versus six here, including pending work, and there
is no isolated old-model ablation. Neither this mechanism nor this training
update receives causal credit for the 7.65% pair difference. Repeated ranking
dependency granularity, separate late-new-traffic deadline accounting and
selective plateau analysis remain discussion points only; no follow-up
implementation or experiment was run.

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
invented. The training curve was derived from the saved history and passed
primary visual review. The archive's saved replay checks pass hash/index validation,
4060/3887 H/L P/V/A samples and 25/22 C2 checks, with zero normal-start time
mismatches; actual normal starts match acknowledgements plus the initial state.
The primary viewed `planning_timeline.png` and `saved_xy_contact.png` and found
readable layouts without clipping, consistent with two/five joins and six/six
normal landings. These two images have PASSED_PRIMARY_VIEW_IMAGE in the
supplement. The primary also viewed the actual `training_curve.png`: all 15
epochs are readable, the selected epoch 10 aligns with minimum validation rank
loss, and no clipping was observed. All three images now have
PASSED_PRIMARY_VIEW_IMAGE. Replay acceptance remains HUMAN_PENDING.

`artifacts/report_future_learning_20261027.py` is a standalone saved-output
supplement entry, outside the frozen algorithm inventory. Report generation
completed with exit code zero: it exported the actual training history as
PNG/SVG, compact metric JSON and a 12-row per-callback quality/compute CSV,
preserved `FINAL_REPORT.md` byte-for-byte as
`FINAL_REPORT.generic.original.md`, then appended the clearly labeled primary
interpretation. Six distinct Coverage snapshots were confirmed per arm where
the new lazy counters apply. It invoked no fitting, simulation or new evaluation. Contact
markers follow the existing plot code: circles are HOLD, downward triangles
are FINAL or VERTICAL_DESCENT, squares are other non-airborne statuses and
right-pointing triangles are remaining airborne statuses; H is grey and L blue.

`artifacts/future_learning_primary_review_20261027.json` records the final
PASSED_BOUNDED_PILOT primary decision, exact image hashes and HUMAN_PENDING
replay acceptance. This completed pair is newly run evidence; historical
search-handoff analysis is preexisting evidence. The decision makes no
generalization or deployment claim and authorizes no further fit or evaluation.
