# V4 I0 Structured Expert Dataset

Implementation: `terminal_marl.training.low_v4_imitation:collect_i0_dataset`.

Each fresh sample is `observation -> teacher_joint_index`, with decoded action,
source, scenario class, seed, UAV ID, decision-event ID, state hash, time and
lifecycle phase. It is explicitly not `requested action -> repair action`.

The collector supports `expert_rollout`, `safe_warm_start`, `from_empty`,
`fixed_anchor`, `targeted_release_opportunity`, `targeted_wait_conflict`, and
the reserved `student_failure_state` source. It applies bounded same-UAV/
same-target persistence sampling, without changing or copying labels.

It emits raw-tick/event/streak/duplication statistics; WAIT/RELEASE and joint
histograms; source/scenario histograms; and a disjoint fixed decision-state
cohort containing the state snapshots and Teacher targets used by I1/I2/I3.

## Formal collection result — 2026-09-10

Fresh current-V4 collection completed at
`runs/v4_i0_structured_expert_dataset_20260910_r3` with 1,200 rows. It did
not read old V3/V4 L0 labels and did not run PPO.

- Dataset SHA256: `d868380597c2c146aa84758d1fff57a42a739a937546ea95629eb4e1c8f7d66c`.
- Frozen independent cohort SHA256: `83d28cfcd132abfa306948babf235a51edf3109ce87794ed167525ca7622590f`.
- Safety: collision/NMAC/hard/SafetyInvariantError/uncertified = `0/0/0/0/0`.
- Teacher determinism: PASS for all eight independent source/seed probes.
- Sampler: `event_start_plus_bounded_temporal_subsample`; raw ticks=4,283,
  retained decision samples=1,200, temporal duplication factor=3.569,
  same-UAV target streak mean/P50/P95/max = 19.24/15/48/72.
- Raw Teacher labels: WAIT/RELEASE=`4,225/58`; retained labels=`1,155/45`.
  RELEASE covers joint indices `5,15,16,17,18,19,24,25`; it therefore has
  enough count for the frozen numeric gate but is still highly imbalanced.
- Required sources, including interface/headway conflict and resource
  constraint, were collected; the cohort has 93 WAIT and 27 RELEASE labels,
  snapshots, Observation V2, Teacher target, source, seed and state hash.

The formal gate is **FAIL** because retained WAIT labels cover only speed
variant/index `2`; the required multi-variant WAIT speed coverage is absent.
The event-ID namespace was subsequently corrected in code to include source
and seed; this completed artifact remains historical failed evidence and will
not be reused as a passing I0 dataset.

Current evidence state:

```text
I0_STRUCTURED_EXPERT_DATASET = FAIL (WAIT speed coverage)
I1_JOINT_BC_ALLOWED = false
PPO_ALLOWED = false
```

## V4.1 status

The V4 failed dataset remains read-only evidence.  Its failure root cause was
the now-superseded WAIT-speed catalog requirement, not a reason to fabricate
WAIT variants.  V4.1 replaces that schema with one canonical WAIT and 25
RELEASE classes; it does **not** reuse this dataset as a V4.1 I0 source.
Following V4.1 S0 schema qualification, a fresh V4.1 I0 recollection is
allowed by its revised decision-relevant-family coverage gate, but has not
been run in this change.
