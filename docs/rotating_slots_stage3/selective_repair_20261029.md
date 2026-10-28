# Selective repair protocol — 2026-10-29

## Versioned contract

This revision adds `opportunity_cooperative_v2`; `baseline` and
`outer_priority_v1` retain their selection rules. The corresponding executor
policy is explicitly `coalesced_restart_v2`, including the new FULL_H control.
The default executor policy is `legacy`; missing checkpoint policy fields mean
legacy, and loading a checkpoint under a different bound policy fails.

K8/max12, the 20 s lead, 5 s heartbeat, 1 s preparation margin, 4 s search
handoff reserve, existing physical/energy/safety/pad rules, common emergency
handling, and the complete global downstream objective remain fixed. Selection
does not change background safety, resource, energy, or downstream evaluation.
The complete steps 1–4 version repairs normal-task continuation, selective
membership, cooperative opportunity ordering, and progressive candidate
publication/selection. It adds no learning entry point, dynamic K, lead scan,
or solver budget.

## Normal-task continuation

`LatencyAwareScheduler(..., restart_policy="coalesced_restart_v2")` records a
restart request only when a completed normal task has a coalesced dirty trigger.
The world drains the old task's boundary transitions and due activations, then
calls `restart_coalesced` with the real current snapshot at the same simulation
time. `_finish` never recursively invokes the planner. The scheduler persists
the request and the last normal-start/restart times; repeated calls at the same
timestamp cannot start another normal task. Starting a task clears the consumed
dirty trigger, so a NOOP or rejection cannot create a self-triggering loop.

New task work and public boundary service consume the existing measured virtual
CPU timeline. Prepared packets retain their original effective time. The
world's existing `settle_only_at_end` check precedes dispatch and therefore
preserves endpoint truncation. Lifecycle telemetry links the previous finish to
the next start and records `COALESCED_AFTER_<status>` and the elapsed simulation
gap.

## V2 mandatory evidence and age handling

Energy/peak urgency starts the hard set. Only current definite required resource
dependencies reached from those hard aircraft join its hard closure. An
unselected source's dependency cannot independently make its target hard.
Hard counts above 12 cause the existing explicit full fallback, with the same
shared budget and deadline. All hard aircraft remain represented.

Waiting is recorded in the real snapshot clock. `current_adjustment_time` is
the maximum of the current task's effective time and the aircraft's immutable
commitment endpoint. `deferred_adjustment_time` is the maximum of that endpoint
and the current effective time plus the current task's lead. This is an estimate
of the next task's adjustment boundary if the aircraft is omitted; an early
NOOP can complete sooner. Holding cost from the already projected node time to
that deferred boundary enters urgency. Projection and fixed-commitment energy
already charged up to the node time are never subtracted a second time. Masks
record both times, the timing source, total waiting, already charged waiting,
and additional charged SOC. UNKNOWN reference information never certifies safe
postponement; a lower-bound deficit can still provide separate urgency evidence.

Review age 60 s is diagnostic debt, not hard membership. A continuous priority
bonus begins at 40 s. After the hard set, at most two remaining nominal seats
are offered to fairness review, including bounded access for unknown ages.
Any necessary dependency closure is considered only when its source is chosen;
optional groups that cannot fit are skipped with a reason. No ring quota is
used. A review clock resets only on delivery of a completed explicit evaluation
whose normal continuation is known and whose source dependencies still apply.
UNKNOWN evaluations do not reset it. In v2, progress resets only on an actual
revealed `maneuver_started` event for TRANSFER or FINAL_NORMAL. Event consumption
uses a checkpointed cursor; imported missing evidence remains unknown.

## Near-term cooperative opportunities

`near_term.py` reads finite slot/pad release calendars, exact applicable cached
path references, and cheap minimum arrival estimates within a 240 s summary
window. It never reads FOLLOW tail completion times or changes the frozen
global evaluation frame. Potential owner releases are explicitly estimated,
uncertified ordering information. Each relation names the specific target slot
or pad; occupying some other slot in the same ring is not a required dependency.

Groups can follow a downstream anchor's specific source resource in reverse to
one suitable upstream follower and then another follower. Connected groups are
retained before unrelated singleton anchors when producing up to three distinct
diagnostic seeds. Hard and fairness membership precede opportunity filling;
only definite valid dependency closure can expand nominal K8 to max12.
Approximate competition and potential release estimates cannot make mandatory
membership. Output includes opportunity source/unknown masks, group membership,
selection/skipping reasons, fixed-background reasons, and overdue review debt.

`RuleSelector.select(..., opportunity_seed_offset=0)` and
`NormalSchedulerCallback(..., opportunity_seed_offset=0)` accept only 0, 1, or 2.
They expose deterministic same-state diagnostic alternatives, preserving the
same hard/fair rules. Closed-loop runs must bind zero. Alternative seeds are
not selected from experimental outcomes.

## Local checks for steps 1–3

The two new focused suites cover commit/reject/NOOP continuation requests,
endpoint truncation, checkpoint guards, overdue age without overflow, both
projected deficit and positive-but-insufficient deferred-wait margins, UNKNOWN
masks/reviews, actual-event progress, exact resource grouping, competing inner
singletons with a connected inner–middle–outer group, conditional hard closure,
approximate competition, and preservation of all unselected background IDs.
Step 4 below completes the candidate-phase repair in the same version; campaign
verification checks the combined steps 1–4 implementation.

## Step 4: v2 progressive generation and two shared-budget solves

The v2 callback opts into `LocalPlanner(..., progressive_v2=True)` and
`CandidateProvider.iter_progressive_columns(..., early_publish=True)` for both
local and FULL_H controls. The generator reuses only exact applicable previous
chains, visits the initial per-aircraft progress pass, and publishes immediately
after that pass. It keeps KEEP/WAIT and explicitly lists initial unvisited IDs
if the original attempt/time budget ends before coverage. Legacy publication
continues to use its existing (2,4,8) batches.

V2 makes at most two `scheduler.plan` calls and performs at most one actual
background-blocker scope expansion. Its first call caps CP allocation at 2 s;
the next call receives only the remaining original 8 s allocation. Both calls
share the original candidate-active 6 s, search deadline and preparation
deadline; no budget or clock is recreated. After the initial call, explicit
current resource blockers can cause expansion up to max12. Otherwise observed
coverage gaps prioritize the existing progress/reference/delayed/alternative
families in the same generator. Coverage records distinguish aircraft visits,
targets, immediate moves, delayed moves, and complete chains. Generation stops
after the second solve.

An earlier certified choice is mapped to the current catalog, rechecked as a
batch and commit prefix, and rescored with the same complete global frame before
comparison. Expanded scopes add fixed KEEP choices for newly opened IDs to this
mapping. If exact mapping/revalidation is missing or UNKNOWN, the original
certified catalog/result is retained. A worse or UNKNOWN result cannot replace
it. An objective-improving WAIT remains an allowed final result.

Strict descriptor pruning is limited to proved overlap with finite immutable
committed resource occupancy. It runs before trajectory construction. A
selected aircraft's open-ended/current target occupancy does not establish that
proof, so joint departure remains searchable. Existing exact source/version,
epoch, configuration and energy checks still govern previous-chain reuse;
UNKNOWN or shifted chains are not promoted into reusable certificates.

`EvaluationContext.check_batch_conflicts` records only explicit selected–selected
pair conflicts in `joint_conflict_observations`, with aircraft IDs, reason,
resource where known, snapshot/scope and round context, and
`source="SELECTED_SELECTED_BATCH"`. UNKNOWN outcomes and single-candidate
background rejections do not create this record. Local/callback telemetry also
records each catalog's coverage, generation/solve wall time, shared remaining
time, first certified result, best objective, and actual fallback. Existing
per-aircraft `move_candidates` / `only_keep_wait` fields describe unavailable
choices. Geometry cache counters are recorded as before/after differences, and
the existing cold-worker reset clears the new pure evidence cache as well.

The real-publication integration fixture supplements lifecycle stubs: a legal
FINAL remains prepared through endpoint truncation at effective time 20 (and
300), then publication starts the maneuver once and the same-boundary restart's
real source contains the active trajectory and committed pad reservation.

## Pure geometry evidence

`geometry_evidence.py` caches only pure SAFE separation evidence keyed by the
complete immutable trajectory/configuration identities and the checked absolute
time interval. The cache uses bounded LRU retention; UNKNOWN outcomes are never
cached. Reusing this evidence does not authorize a scheduling decision or reuse
a complete ACCEPTED result: current resource, energy, dependency and commit
checks still provide fresh final authorization.
