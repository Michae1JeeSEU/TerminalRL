# checkpoint_12 independent evaluation — PASS

Six fresh episodes690001–690006 completed24/24 UAVs. All original I3 gates
passed: terminal/queue drained, safety, shield burden, exact replay and
requested likelihoods. No training or Q2 access occurred.
Manifest SHA256: `fe6973c8689886a1f9f344bbd8481a525c810b051017c33af5b9e869657dd46f`.

| Scenario / behavior | checkpoint12 seconds | DAgger seconds |
|---|---:|---:|
| warm3 deterministic | 234 | 234 |
| flow3 deterministic | 469 | 469 |
| flow6 deterministic | 752 | 750 |
| warm3 stochastic | 222 | 222 |
| flow3 stochastic | 601 | 601 |
| flow6 stochastic | 742 | 855 |

This supports retention of independent closed-loop competence. Four ties,
one2s regression and one113s improvement do not establish general PPO
superiority. Continue training in the same scope before increasing load.

## Local visualization

Use the existing local Python environment with project dependencies and
`requirements-visualization.txt` installed. No server GUI is needed.

```powershell
cd D:\7_codespace\terminal
python -m terminal_marl.visualization.ppo_checkpoint_viewer --checkpoint D:\7_codespace\training_outputs\terminal\v4_2_ppo_initial_block_20260917\checkpoint_12.pt --scenario flow6 --seed 690003 --playback-speed 10
```

This uses the existing PyVista window and exactly the evaluation's88D,
Teacher-disabled, deterministic requested-action path. Expected completion:
6 UAVs at752 simulated seconds. The first arrival is at10s; approach flight
precedes ring decisions. Use pause/speed controls and right-click UAVs to
inspect SOC, phase and committed trajectories. `--scenario warm3 --seed
690001` starts with3 visible aircraft; flow6 is the preselected validation
demonstration. Closing the window ends the local demonstration.

Optional generated browser replay remains in the evaluation output as
`validation_scene.html`; it is not required for the local PyVista command.
The local GUI itself was not launched by the agent; its shared policy path
is validated headlessly. No claim of local display verification is made.

## Next training preparation

`ppo_block2_preparation.yaml` reserves24episodes seeds700001–700024,
samewarm3/flow3/flow6 and optimizer parameters, initializing model AND
optimizer from this evaluated checkpoint. Fresh evaluation seeds710001–
710006 are reserved. No training started. Before launch implement/validate
explicit continuation lineage and a restore smoke: the old first-block
entry intentionally initializes from DAgger and cannot resume checkpoint12.
Do not run its old command and claim continuation. Final selection remains
the last checkpoint; evaluation data do not enter optimization.
