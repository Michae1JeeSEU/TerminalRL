# P2 Conditional-Proposal Target Chain

> Date: 2026-09-19  
> Evidence types: `newly_run` formal support/diagnostic cohorts and offline diagnostic probes  
> Current boundary: no proposal repair, PPO, P3, curriculum promotion, or checkpoint promotion is authorized

## Why the target changed

Fresh flow12 feasibility attribution showed that 66/89 rejected RELEASE events occurred when the
Teacher also selected RELEASE but the student selected a different proposal. The first response was
therefore to collect a fresh exact-Teacher-proposal dataset and test whether the unchanged 25-way
proposal head could learn that target without losing the old D1 semantics.

That exact-label route did not pass. A follow-up full-catalogue diagnostic then showed that the
Teacher's single selected proposal is only one member of a much larger certified set in many states.
The next data target is consequently the set of all SafetyFilter-certified RELEASE proposals, not a
single catalogue tie-break choice.

## Exact-label support V1

Proposal support V1 used fresh, split-disjoint flow12 seeds: train `790001–790009` and dev
`791001–791006`. All 15 episodes completed all 180 requests with safety, geometry, replay,
requested-likelihood, provenance, and split-isolation gates passing. The frozen datasets contained
540 train and 455 dev Teacher-RELEASE rows.

The train/dev sets contained 15/16 distinct exact proposal labels. Proposal 12 remained dominant
at `71.67%` of train and `60.44%` of dev. The formal support result was PASS, authorizing only an
offline learnability audit.

- protocol SHA256: `f68cefb0fc1f8c929b04342ad945ed86793b6566fd3fdc8716152219e56e3a75`
- manifest SHA256: `48c0b1cd0b92bd746bd52f18090bbffd8facf59457eb82d503802eb4fc2c512b`
- train/dev SHA256: `7aa8adaf805b22fd58aaaf258111e1b457cf5159241135d8ca262d32a5d11fa2` /
  `2ea5bbfe66d6a688a486039a6c12461356ea51abce3c778cc85e8b6b05589415`

## Exact-label learnability audit V1

The frozen candidate achieved dev Top-1/Top-3 `0.3648/0.4659`, while retaining near-perfect D1
proposal accuracy (`0.9947/0.9993`). Three anchored residual probes improved dev Top-1 only to
about `0.433` and Top-3 to `0.576–0.578`; D1 Top-1 remained about `0.962`. None passed the frozen
fresh-dev and D1-retention gates, so the audit terminated FAIL and did not authorize repair.

- protocol SHA256: `fd05c7b7928474bd649520b6bec7261df997e7d04ec14972955ae3335e99d196`
- manifest SHA256: `87bdf9b5ce0b9d9cbf967d858669a6f17679c0cf271d83c8c39e7f4fa4e52e95`

## Failure attribution

Failure attribution separated frozen-logit saturation from target identifiability. The candidate had
a negative target margin on `63.52%` of dev rows, with median margin `-18.06`, confirming severe
old-head saturation. However, independent probes also failed to establish the exact target as a
reliably learnable state function:

- nearest-neighbour target agreement was `0.5253` from dev to train and `0.4879` from dev to D1;
- linear probes reached Top-1 `0.530–0.541` and Top-3 `0.791–0.800`;
- MLP probes reached Top-1 `0.580–0.600` and Top-3 `0.712–0.732`;
- o2m remained the weakest interface, with MLP Top-1 `0.486–0.535`.

No exact observation-hash conflicts were found, but that does not establish identifiability in the
continuous state space. The formal classification was
`observation_or_label_learnability_limit_supported`; all policy-repair and training authorizations
remained false.

- protocol SHA256: `2b39758e62af4aed58c7e5dadc0172e96aa920523d29db2819e0ebc5f6dd8f43`
- manifest SHA256: `3c8e0980f4f970bb66ca207ceb3305dc3fbe9e3eb79e9e7a827a1b086acf11f5`

## Fresh feasible-set diagnostic V1

The next diagnostic evaluated the full 25-proposal Teacher catalogue only at actor-requested
RELEASE states on fresh seeds `794001–794003`. All three flow12 episodes completed 36/36 requests;
all safety, geometry, replay, likelihood, catalogue-completeness, and Teacher-consistency gates
passed. There were 166 requested-RELEASE states.

| Metric | Overall | i2f | m2i | o2m |
|---|---:|---:|---:|---:|
| events | 166 | 52 | 53 | 61 |
| no feasible RELEASE | 27 | 15 | 7 | 5 |
| policy Top-1 in feasible set | 107 | 37 | 34 | 36 |
| policy Top-3 intersects feasible set | 123 | 37 | 41 | 45 |
| Teacher selected proposal feasible | 139 | 37 | 46 | 56 |
| execution accepted | 108 | 36 | 36 | 36 |

The feasible-set size had min/median/mean/max `0/25/16.90/25`. Among execution outcomes, 106 events
were both candidate-feasible and accepted, 57 were neither, and only three crossed those labels.
This near-alignment supports feasible-set membership as the relevant conditional-proposal target.
The three cross-label events expose a protocol risk: catalogues are evaluated against one pre-step
world while multiple UAV actions may be committed sequentially within the same environment step.
They are therefore not joint multi-agent feasibility certificates.

- protocol SHA256: `dbe387cc134741f87cde8caa2b8910410e42e5ad756dcb197bc2c04c208eabca`
- manifest SHA256: `957bc2535fbef767a90fbf6cd299ea5aa38cd2f680099a3c5f79aff97fe4821f`
- catalogue rows SHA256: `c7dbbf203084d1b90db8662f42514b8c3f45a5e94b2d7e9e706ad80143784242`

## Proposal-set support V1

The separately frozen support collection evaluated all 25 RELEASE proposals on every actor state
where the label-only Teacher selected RELEASE. Fresh train seeds `795001–795009` and dev seeds
`796001–796006` produced 15/15 terminal episodes and 180/180 completed requests. All episode,
split, safety, geometry, replay, requested-likelihood, catalogue, and isolation gates passed.

| Split | Rows | i2f / m2i / o2m | Unique sets | Mean set size | Top-1 / Top-3 feasible | Late fraction |
|---|---:|---:|---:|---:|---:|---:|
| train | 463 | 113 / 159 / 191 | 132 | 19.469 | 351 / 400 | 0.5616 |
| dev | 372 | 94 / 116 / 162 | 105 | 19.169 | 265 / 325 | 0.5296 |

The formal support result was PASS and authorized only the offline set-valued learnability audit.

- protocol SHA256: `fa2d764604e05526262549559c718291caeef3ca23202bce38d20d3e038d2972`
- manifest SHA256: `d057d0f39a3a99f6641c331eecd1a127a2b639b5046bf78d39b55d1b8caf7f37`
- train/dev SHA256: `f5511da554f21614e399b6384609bac93a41e9fdef94b8735eb2ecd6cb5d3f14` /
  `dd07e8f2b877bd7f08db113aa0fc8988ed8d106223d799284b3e1f8b6324d17a`

## Proposal-set learnability audit V1

The frozen candidate had dev Top-1/Top-3 set hit `0.7124/0.8737`; on partial sets these were
`0.3270/0.7044`, with partial-set NLL `15.3519`. Three zero-output residual probes used set-mass
NLL on the fresh support set plus exact-label CE and KL anchoring on the old D1 cohort. They reduced
partial-set NLL by `53.3–54.3%`, passed Top-3 and every D1-retention gate, but did not pass Top-1:

| Probe seed | Overall Top-1 / Top-3 | Partial Top-1 / Top-3 | D1 Top-1 / Top-3 |
|---|---:|---:|---:|
| 797101 | 0.7796 / 0.9328 | 0.4843 / 0.8428 | 0.9827 / 0.9967 |
| 797102 | 0.7796 / 0.9382 | 0.4843 / 0.8553 | 0.9827 / 0.9967 |
| 797103 | 0.7769 / 0.9409 | 0.4780 / 0.8616 | 0.9827 / 0.9967 |

All probes failed the frozen overall Top-1/gain, partial-set Top-1, and o2m Top-1 gates. The audit
is terminal FAIL; no adapter or checkpoint was saved.

- protocol SHA256: `9dcb54d0b23dc81ad5fabacfb97762e127b3169f7908736540b9b8275e28868f`
- manifest SHA256: `a1b76fa16427cecdc0c70e0d38c6247d8e6e6d943063aeeddd00a8cede28e937`

## Proposal-set failure attribution V1

The attribution trained fresh standalone linear and MLP probes with set-mass NLL and no D1 anchor
or frozen candidate logits. Exact observation hashes produced no conflicting feasible sets, and the
nearest train set intersected the dev set in `98.92%` of states; mean nearest-set Jaccard was
`0.7271`. This indicates substantial local overlap, but not a reliable single executable Top-1
choice.

Linear probes reached overall Top-1 `0.8683–0.8737` and partial-set Top-1 `0.6918–0.7044`; only one
of three met every frozen gate. Larger standalone MLPs reached overall Top-1 `0.8306–0.8575` and
partial-set Top-1 `0.6038–0.6667`; all three failed the partial-set Top-1 gate, although every
interface aggregate and Top-3 gate passed. The frozen classification is therefore
`observation_or_set_target_learnability_limit_supported`, not an authorization to remove D1
anchoring or repair the policy.

- protocol SHA256: `f2e145734c267742ca13093c39df4e1534c6f7855413e07fff5d94ba35bbaa26`
- manifest SHA256: `4a1d6d9d9b594c1d680203e8c8b728e08101ec1fdfc26aa64028be9f800c5e10`

## Evidence boundary and next gate

The exact-label support, audit, failure attribution, and feasible-set diagnostic artifacts are
immutable evidence. Their seeds and rows may not be reused for training or threshold tuning. The
candidate checkpoint remained byte-identical throughout.

Exact-label and set-valued repair are both blocked. The set support rows are frozen and may not be
used for policy repair because their learnability audit failed. The candidate checkpoint remained
byte-identical throughout, and all optimizer activity was confined to unsaved diagnostic probes.

As a final narrow check, a proposal-conditioned audit expanded each frozen state into 25
`[88D observation, speed residual, merge-time residual]` pairs and learned binary feasibility with
train-split class balancing. It also failed all three MLP seeds. Feasible recall was
`0.897–0.910`, but infeasible recall was only `0.454–0.462`, macro-F1 `0.688–0.701`, ROC-AUC
`0.793–0.810`, and state Top-1 feasibility `0.858–0.887`; o2m failed in every run. Thus the older
closure-bank result that proposal-conditioned certification was separable does not generalize to
the present natural-flow flow12 distribution.

- protocol SHA256: `68694cefa2e8907b3d4beb77ca3abc2daf569bc9b5ffab6b46b1d70b9888a896`
- manifest SHA256: `4b37952e8b624284f1d45280043f89aaad9ef97be4e1f33e690644c03827067f`

The next scientifically supported task is therefore a privileged observation/target closure study
focused on partial-set o2m states: collect the scheduler reservation book, per-interface pending
commit order, candidate merge slot, and multi-UAV same-step competition that the current 88D
observation omits or compresses. Choosing between Observation V2.2, a joint-feasibility target, or a
runtime catalogue-search boundary is an architecture decision; it is not authorized by these
diagnostics. Any chosen route must be independently specified, recollected on fresh seeds, and
qualified before proposal repair. Until then `PROPOSAL_REPAIR_ALLOWED=false`, `PPO_ALLOWED=false`,
and `P3_ALLOWED=false`.
