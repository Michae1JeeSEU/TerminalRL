# First formal PPO block — PASS

Evidence type: newly_run, 2026-09-17. Runtime code `1470229`;
curriculum review `457738f`. Output:
`~/training_outputs/terminal/v4_2_ppo_initial_block_20260917`.

Completed all 12 episodes (seeds680001–680012), all 48 requested UAVs.
warm3/flow3/flow6 each occurred four times. Every recorded gate passed:
completion/terminal, zero safety counts, zero actor shield interventions,
exact replay, finite PPO updates, bounded KL and checkpoint restoration.
Maximum approximate KL=.00874680 (<.05); maximum recomputed requested
logprob error=9.536743e-6 (<=1e-5). No Teacher action substitution occurred.

Final checkpoint: `checkpoint_12.pt`, SHA256
`965c5438dfc42f9898a4059b1225527bebd94f048ae9dfe6253cb49fc0519198`.
Manifest SHA256:
`3a0461b663538af26dd7b2ca06a74c6ac203c0ac41c856535357c39c7783022a`.
`stage=FORMAL_PPO_BLOCK`, `PASS=true`. The manifest's `PPO_ALLOWED=false`
is how this runner labels a formal-block result; it does not revoke the
separate initialization admission or authorize another block.

These are training trajectories with changing policies and different seeds.
They cannot establish improvement versus I3 or a rule baseline. The final
checkpoint was saved after the last update, so it still requires fresh
student-only evaluation before promotion or continuation. This block is
neither convergence evidence nor authorization for larger N or dynamic airspace.
The current entry deliberately initializes from admitted DAgger and rejects
reused formal seeds; continuation needs an explicit new protocol/lineage.

Luna high monitored the process through exit; no monitor remains active.
Only documentation changed this turn; actual instantiated server scenarios
were inspected and the existing source-bound admission validated on launch.
The previous 16 targeted implementation tests remain applicable. No source
was manually transferred; Git has no remote and no push was performed.
