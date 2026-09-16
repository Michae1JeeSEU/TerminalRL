# I0-R4 formal result

Verified 2026-09-17 against local Syncthing artifacts and server checksums.
All 12 preregistered gates PASS. PPO remains disallowed; this result permits
I1 Qualification Protocol V2 registration and fresh Q2 collection only.

| Partition | Episodes | Rows | Safety violations |
|---|---:|---:|---:|
| Training | 208 | 7,576 | 0 |
| Internal audit | 52 | 1,886 | 0 |

All 52 recorded ActionTapes replay exactly. All 9,462 rows pass schema/retry
provenance; Teacher determinism and historical/partition independence pass.
There are zero exact observation-hash opposite-gate conflicts. Internal
audit remains NEVER_TRAIN/NEVER_MODEL_SELECT and is not Q2.

Training C01–C09 lifecycle counts: 80, 32, 48, 256, 32, 32, 96, 72, 128.
Proposal-14 N=5/N=6/N=12 lifecycle counts: 32/32/80. All nominal anchors and
stable support pass. This repairs I0 support under the prospective R4
protocol; it does not retrospectively upgrade R3 or establish actor learning.

Artifacts: `~/training_outputs/terminal/v4_2_i0_r4_formal_20260916`.

- Manifest SHA256: `b4291ddd3540a5a048cb645fb6b741b5379ad15220e1be865ce6993ae6d6fb2d`
- Training SHA256: `c0a9e63e380051273460d13a51abc06313d9c2f9d09e202859380ebd77edb797`
- Audit SHA256: `b8bd5c5e926f38932575cfb1890ab089eb7f20dd968b482eb28355af06b15daa`
