# Stage3 v1.0 runtime interface mapping

Contract: `FT-RS-S3-INITIAL-1.0`, supplied in
`new10.16/rotating_slots_stage3_learning_v1_0`. Implementation inspection started
on local branch `codex/arrival-restructure`, HEAD `f480e40`; the pre-edit
`stage2_planning.py` blob starts `228d7f9`. These are inspection facts, not the
eventual data/model binding. The campaign must bind the actual committed source,
workspace difference, inputs and environment at preflight.

## Source ownership and deployment boundary

| Interface | Authoritative source / semantics |
|---|---|
| Snapshot, committed source states | Existing `PlanningSnapshot`, `_node()` and world executor; no second calendar or emergency policy |
| Proposal family | Existing Stage2 `_offset_cursors()`, `_next_descriptor()`, geometric screen and reference target/duration grid |
| Materialization | `ReferenceProvider.generate()` wraps its original generator, cache and exact attempt accounting |
| Motion and energy | Existing generator and `ReferenceProvider.move_energy()` / `_move_energy()`; normal reserve/peak rules are unchanged |
| Candidate construction and retention | Existing `_column()`, `check_single()`, `_retain_column()`; KEEP/WAIT and full/partial/target/delayed retention remain authoritative |
| Joint objective | Existing `objective_schema_id(planning_profile)`, `stage2_downstream` and CP/Slack; Stage3 adds no coefficients or objective |
| Execution | Existing `check_commit_prefix()` and `commit_validated_prefix()`; a ranking result is never accepted by these APIs |

Stage2 defaults remain `progress_tail_v7` / its existing budget and emergency
configuration. The campaign's explicitly selected `downstream_service_v8`,
`rescue_need_v7` and `service_half_v1` must be supplied by the campaign. They are
not silently made global defaults. Version identifiers include
`work_descriptor_v1_0`, `candidate_context_sets_v1_0` and
`pre_materialization_only_v1_0`. The actual candidate/objective, policy and source
hashes remain mandatory run-binding facts.

## Public APIs

```python
WorkDescriptor(...)                         # frozen exact contract fields
descriptor.as_dict()                        # portable record_type included
RankResult(ordered_ids, model_metadata={}, scores={})
FeatureEncoder.encode(snapshot, descriptors, observed_search_state=None)
AdvisorRuntime(mode='OFF', advisor=None, feature_encoder=None,
               fingerprint=None, observer=None, ancestry_root_id=None,
               evaluation=False, trace_limit=8192, offline_probe=None)
CandidateProvider(..., advisor_runtime=None)
NormalSchedulerCallback(..., advisor_runtime=None, budget_scale=1.0)
scale_candidate_budget(existing_budget, scale)
```

The feature encoder and learned advisor are separate modules. The runtime lazily
creates an encoder only when enabled. Importing Stage2 or the Stage3 public
interfaces does not import torch. `observer` is a callable receiving a frozen
portable dictionary; pass an object's `.record` method if needed. `traces` is a
bounded in-memory event list; observers can stream the complete event sequence.
Observer errors propagate, so a failed recording operation cannot be represented
as successful evidence collection.

`observed_search_state` supplies actual `nodes[source_node_id]` and
`parents[parent_prefix_id]` records containing `materialized`, `soc`, `ring`,
`slot`, `end_time`, `depth`, `peak`, `error_j`, `peak_error_s`, `source_node_id`
and `trajectory_hashes`. It also supplies actual `attempts`, `materialized`,
`coverage` and `remaining_budget_fraction`. Downstream learned features remain
explicitly missing unless the common Stage2 interface supplies them. No feature
query is permitted to call generation, energy integration, collision validation
or CP.

Descriptors use offsets relative to snapshot time and the actual source node's
geometry/SOC. Descendant prefixes must be present in the generated/reused plan
ledger and exactly match a computed successor's SOC, peak use, numerical error,
version and epoch. A generated reference prefix describes an energy-evaluated
normal reference; it is not necessarily executable at current SOC or compatible
with other traffic. References become catalog candidates only through the
original Stage2 replay and validation. Exact cached/previous chains have
explicit `ReuseEvidence`, with source epochs checked and no translated plans.

## Ordering and coverage

| Path | Before-query integration |
|---|---|
| Immediate progress | At most 64 already enumerated primitive heap entries, before `generate()` |
| Normal reference | At most 64 unscreened-as-invalid targets in the current traditional duration family, before `generate()` |
| Alternative multistep | At most 64 ready TEMPLATE entries whose STATE was actually generated and energy-evaluated; structural enumeration stays Stage2-owned |
| Upstream delayed primitive | Same ready-heap mechanism, classified `alternative` |
| Delayed full-reference regeneration | Factual descriptors and costs, traditional order; no learned reorder in v1.0 |
| Previous-chain reuse and compatibility `_successors()` | Explicit reuse / factual observation, traditional order; no learned reorder in v1.0 |

Only one selected work item is consumed per ranking call, below the maximum of
eight. Every next item refreshes ready features after prior attempts/parents or
coverage changes. Ranking is a permutation of ready IDs; unknown/duplicate/new
IDs and nonfinite scores fall back to the inherited order. No low score removes
an item, alters time families, or skips common physics checks.

Both guided learning and the reduced heuristic control use the same initial
coverage pass. It retains Stage2's outer/middle/inner round-robin order, with
energy slack and stable ID ordering within each ring. SCREEN yields do not count
as an opportunity: the pass advances to an actual query or exhaustion. Unserved
aircraft are recorded. After initial coverage, three materialization
opportunities use the advisor and the fourth uses deterministic order. The final
20% of the shared candidate wall/work allowance uses deterministic order.

The geometry screen may already have paid substantial physics cost before the
ready pool exists. It retains its own `geometry_template_screen` timing and
screen counters. It is never counted as a cheap feature or as a learned saving.
Coverage reports include actual attempted stages, actual learned selections by
stage, unserved aircraft, and explicitly unguided paths. These measurements do
not imply that the whole candidate front end is guided.

`RECORD` retains the original candidate order and stage sequencing. Enabled
logging can change wall-cutoff timing, so parity is defined under fixed work
budgets and wall effects are reported separately. `OFF` detaches runtime hooks
and does not instantiate features or a model. `SHADOW_OFFLINE` alone may accept
`offline_probe(descriptors, inherited_selected, opportunity_index)` returning a
ready descriptor and probability/stratum metadata; the full ready pool remains
available. The callback is forbidden for online guided modes.

## Budgets, trace and labels

Reduced budgets scale exactly `candidate_seconds`, `max_per_aircraft` and
`max_attempts` by 0.5, with integer floor and a minimum of one. Solver, hint,
soft/watchdog clocks, refinement limits and catalog retention are unchanged.
Scaling mutates the original budget and never resets time or attempts. Repeated
same-scale application is idempotent; conflicting reapplication raises. Callback
checkpoints record/assert opt-in mode, scale and fingerprint. Ordinary OFF/full
checkpoints retain the existing representation.

Events are `SnapshotStart`, `WorkDescriptor`, `Rank`, `OfflineProbe`,
`FeaturesBeforeQuery`, `FactualResult`, `SingleVehicleEvidence`, `ReuseEvidence`,
`ColumnAncestry`, `Coverage` and `PlanningSelection`. Feature references identify
the exact pre-query observation. Mutable search dictionaries are copied before
logging so later descendants cannot leak into earlier features. Stable column
IDs hash the original signature and survive objective replacement/retention.
Column records include ancestor descriptors for all plan legs, original
validation status/reason, retention and current objective. The final callback
record distinguishes selected columns from actually committed trajectory hashes.

`generate()` returning `None` is UNKNOWN, never a certified negative. A generated
plan alone is UNKNOWN pending energy checks. Only a finite, explicitly evaluated
energy violation yields `SINGLE_INVALID_CERTIFIED`; nonfinite/uncertain energy is
UNKNOWN with its mask false. Single-vehicle success still does not certify
background/joint safety. Rejected columns retain the original validator reason;
they are not relabeled as permanent single-vehicle negatives. Unselected valid
columns remain unselected facts. The campaign owns the separate bounded-label
layer and must require certified combinations for objective preferences.

Factual materialization wall/CPU measures the real generator/cache boundary;
energy events separately report energy cache and measured miss costs. Full
candidate/callback clocks also contain descriptor/trace work, with exclusive
feature/inference/fallback timings in `PlanningBudget.measure()`. These nested
measurements must not be blindly added. The source fingerprint is supplied by
preflight; an empty development fingerprint is not a valid formal dataset.

## Verification status at authoring

`tests/test_rotating_slots_stage3_runtime.py` contains focused OFF/RECORD,
budget, immutable descriptor, real parent, fallback, interleave/finish, pre-query
observation, initial coverage, checkpoint binding, offline-probe and original
commit-conflict checks. Automatic tests are intentionally not executed by this
implementation worker; the campaign owner accounts for their wall time. This
document does not declare infrastructure, learning signal or online benefit
verified.
