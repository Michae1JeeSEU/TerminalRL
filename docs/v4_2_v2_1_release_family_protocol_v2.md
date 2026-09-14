# V4.2 / V2.1 predicate-defined RELEASE family protocol

## Frozen semantics

`collection_source` is provenance only. It cannot affect classification. Each active UAV receives a continuous per-episode lifecycle ID and a decision-ledger entry containing seed, tick, UID, decision-event ID, full-state hash, observation hash, Teacher gate/proposal, exact WAIT rationale, retry/commit/interface/resource/conflict/holder-tube state, interface release time, headway requirement, headway age, and the previous semantic event.

The classifier is pure and source-independent. RELEASE rows may have multiple flags:

| Flag | Frozen predicate |
| --- | --- |
| `clear_outer` | Teacher RELEASE, outer holder, retry inactive, no commit, open interface, valid resource, no blocking conflict |
| `middle_interface` | Teacher RELEASE in middle holder with retry inactive, no commit, open interface, valid resource |
| `post_wait` | Same UID and same lifecycle; preceding semantic gate WAIT; current Teacher gate RELEASE |
| `post_conflict` | `post_wait`, preceding WAIT rationale geometry/headway/holder-tube/candidate conflict, and conflict now cleared |

UID replacement, lifecycle change, or reset boundary invalidates the transition predicate. The unit suite covers true/false outer and middle predicates, post-WAIT/post-conflict, overlap, source-name invariance, UID replacement, lifecycle break, and reset boundary.

## Predicate recipes

Each recipe records declared predicate values, actual instantiated values, reset recipe/hash, and pre-decision state hash. A witness is admitted only when the two predicate dictionaries are byte-for-byte equal.

- `clear_outer`: 3-UAV legal static warm start; outer holder release context.
- `middle_interface`: 6-UAV legal static warm start; middle-holder context.
- `post_WAIT`: 12-UAV legal dynamic-airspace transition; same UAV must move from a genuine WAIT to a later RELEASE.
- `post_conflict`: same dynamic construction, with a preceding `geometry_conflict` WAIT and a later conflict-cleared RELEASE.

All actions used during the diagnostic are unchanged Teacher-decoded requested actions. The recipes neither fabricate an observation nor modify Teacher, SafetyFilter, action catalogue, ranking, class weights, or historical rows.
