# V4.2 / Observation V2.1 I0-R3 Repair Protocol — Preregistered V2

## Status and scope

This prospective protocol is frozen before any R3 simulator reset. It only
authorizes a fresh standalone expert-data recollection after the Markdown,
YAML, collector implementation and reproducibility hashes agree.

```text
ROOT_CAUSE = MIXED_SUPPORT_AND_METRIC_FAILURE
V4_2_V2_1_I0_R2 = FAIL
I0_R3_PROTOCOL_PREREGISTERED = true
I0_R3_COLLECTION_ALLOWED = true

I1 / Q2 / DAgger / PPO = forbidden in this stage
```

R2 remains a `READ_ONLY_FAILED_QUALIFICATION_ARTIFACT`. The old Q remains a
`CONSUMED_DIAGNOSTIC_REFERENCE`. Neither may contribute an R3 training row or
become the future untouched Q2.

## Evidence basis

| Evidence | Type | Frozen role |
|---|---|---|
| R2 training, SHA `c2f0f947...` | `preexisting_artifact` | Failed-dataset support reference |
| Consumed Q, SHA `8d1e80bb...` | `preexisting_artifact` | Diagnostic reference only |
| Attribution summary, SHA `de450593...` | `preexisting_artifact` | Mixed root-cause evidence |
| Failed-state coverage, SHA `6910e573...` | `preexisting_artifact` | Defines 52/30 residual error manifold |
| This Markdown/YAML | `newly_run` | Prospective protocol freeze |

At radius 0.25, R2 covered only 34.62% of the 52 shared gate false negatives
and 50.00% of the 30 shared proposal errors with a same-proposal RELEASE.
Conversely, Q-intrinsic kNN RELEASE medians were only 0.50 for post-WAIT and
0.40 for transition/retry and post-conflict. R3 therefore repairs both true
residual support and the misaligned coverage metric.

## Frozen semantics

Observation V2.1 remains the 88D `phaseA_low_obs_v2_1_retry_state`; action
factorization remains V4.2 hierarchical WAIT/RELEASE plus conditional 25-way
proposal. Teacher, ranking, scheduler, trajectory generation, SafetyFilter and
PredictiveShield are unchanged. No architecture, loss, reward, threshold or
Safety modification is authorized.

## Fresh lineage and seeds

R3 is not `R2 + patch`. All rows arise from fresh complete expert episodes.
Before the first reset, the collector must scan every historical training,
qualification and diagnostic registry and freeze a collision-free registry.

```text
training seeds       = 410001..419999
internal audit       = 420001..429999
future Q2 reserved   = 430001..439999
```

Training and internal audit must be disjoint in seed, episode, UAV/lifecycle,
decision event and full-state hash. Observation hashes are audited for exact
opposite-gate conflicts but are not used to discard a row.

## Residual-context collection

The fixed complete-episode round-robin schedule is encoded verbatim in the
YAML. Scenario recipes may set only legal initial/environment variables such
as traffic count, spacing, ring occupancy, phase, progress, headway age,
reservation context, arrival timing, retry expiry, recovery state and resource
pressure. A stratum name is not a requested label: episodes and rows remain
admissible when the unchanged Teacher chooses a different action.

The 34 uncovered shared-FN states and 15 uncovered shared-proposal states are
frozen into nine coarse deployment cells rather than copied observations. The
largest cells are N=12 outer/o2m proposal-11 multi-family, N=12 middle/m2i
proposal-14 multi-family, inner/i2f transition, normal middle/m2i, N=5/6
proposal-14, N=8 post-WAIT, and an N=12 middle/m2i proposal-13 cell. Each cell
has independent lifecycle and episode minima in the YAML.

## Proposal-14 traffic repair

R2 proposal 14 contained only N=12 states. R3 freezes the following natural
Teacher-outcome minima:

| Traffic cell | Independent RELEASE lifecycles | Episodes |
|---|---:|---:|
| N=5 | 15 | 5 |
| N=6 | 15 | 5 |
| N=12 | 30 | 5 |

N=5/N=6 strata include outer, middle/interface, post-WAIT and post-conflict
recipes. No episode is discarded when Teacher does not select proposal 14.
TV and its bootstrap uncertainty remain reported but are diagnostic only;
the explicit traffic-cell minima are the hard distribution gate.

## Radius-based support gates

Coverage is separated from class purity. Distances use the unchanged
normalized 88D Euclidean definition and report radii 0.05, 0.10, 0.25 and
1.00.

Hard error-manifold gates:

| Reference set | r=0.25 | r=1.00 |
|---|---:|---:|
| 52 shared gate FN → nearest same-gate RELEASE | ≥0.75 | ≥0.90 |
| 30 shared proposal errors → nearest same-proposal RELEASE | ≥0.75 | ≥0.90 |

The same 0.75/0.90 coverage targets apply to post-WAIT and transition/retry
same-gate RELEASE, post-conflict same-family RELEASE, and proposal 11/14
same-proposal RELEASE. These thresholds are prospective engineering targets,
frozen before collection; they do not revise R2. The 0.75 near-radius target
requires at least three quarters of each reference error manifold to have a
close witness, while 0.90 at radius 1.00 prevents a small near-covered subset
from hiding a broad support tail. Both are frozen here before any reset.

## Purity and prior diagnostics

```text
KNN_PURITY = DIAGNOSTIC_ONLY
absolute kNN RELEASE minimum = none
purity non-inferiority hard gate = none
```

R3 still reports k=10 R3-to-reference RELEASE fraction, intrinsic Q
leave-one-out fraction, local gate entropy, global/context WAIT and RELEASE
counts, WAIT/RELEASE ratio and class-prior-adjusted score. None can substitute
for a failed radius gate or retroactively pass R2. This avoids requiring a
WAIT-heavy training manifold to be more RELEASE-pure than the reference
manifold itself.

## Nominal manifold and stopping

R3 retains ordinary, retry-active, committed, interface-frozen and resource
WAIT; clear-outer and middle/interface RELEASE; and proposals 11–14. Exact
minima are frozen in YAML. It cannot stop by merely increasing aggregate
post-WAIT/post-conflict counts.

Training targets 3,000–4,000 semantic rows with ceiling 4,200. Independent
dataset audit targets 800–1,200 with ceiling 1,400. Stop is allowed only after
a complete stratum cycle and every context, radius, nominal, provenance,
replay, aliasing and Safety gate passes. A ceiling hit fails closed.

## Provenance, replay and Safety

Every row retains reconstructable 88D sources, retry state, reset recipe, RNG
hashes, complete UAV/environment/dynamic-airspace/scheduler/resource state,
transition ledger, Teacher gate/proposal/rationale, 25-proposal audit sidecar,
and config/state/observation/post-state hashes.

Formal passage requires 100% schema/retry provenance, deterministic Teacher,
100% ActionTape exact replay, zero exact opposite-gate observation-hash
conflicts, partition/historical independence, and zero:

```text
collision / NMAC / hard violation / SafetyInvariantError / uncertified execution
```

## Formal decision and future order

Only if every YAML gate passes may a future collector declare:

```text
V4_2_V2_1_I0_R3 = PASS
V4_2_V2_1_I1_V2_PROTOCOL_PREREGISTRATION_ALLOWED = true
```

I2 and PPO remain false. After R3 PASS the mandatory order is: preregister I1
Qualification Protocol V2, collect and freeze untouched Q2, train fresh
multi-seed actors, and evaluate Q2 exactly once.

## Protocol risks

- Consumed Q defines diagnostic error cells and radius reference points, so it
  cannot later serve as untouched qualification evidence.
- Euclidean 88D distance is descriptive rather than causal; semantic cells and
  replay/provenance gates remain conjunctive safeguards.
- Traffic-targeted episode recipes change occupancy distributions by design;
  nominal anchors prevent a boundary-only dataset.
- Proposal outcomes are Teacher-dependent. If a frozen cell is unreachable by
  the hard ceiling, R3 fails; no label-conditioned discard or post-reset
  threshold revision is allowed.
