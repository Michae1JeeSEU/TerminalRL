# checkpoint_12 independent evaluation

Preregistered 2026-09-17 before fresh evaluation resets. Authority:
`terminal_marl/configs/low_v4_2/ppo_checkpoint12_evaluation.yaml`.
Freeze formal checkpoint SHA965c5438…519198 and training manifest
SHA3a0461b6…83022a. No optimizer updates or Q2 access.

Six fresh seeds690001–690006 use warm3/flow3/flow6 in order twice, first
deterministic then stochastic, with exactly the I3 physical configuration,
900s deadline and unchanged episode gates (all requested complete, terminal
queue drained, zero safety, <=5% actor shield, exact replay/likelihood,
nonempty actor decisions). Both gates must appear across the candidate suite.
All six must pass before preparation of a same-scope continuation.

Evaluate the frozen pre-PPO DAgger policy on the same seeds/configurations
and behavior modes for paired descriptive comparison. Comparisons are not
used to select a checkpoint or redefine success after seeing results. This
small suite cannot prove statistical superiority or justify larger loads.
Never discard failures. An exclusive marker consumes the evaluation attempt.

Preselect deterministic flow6 seed690003 for the visual demonstration; never
choose a prettier trajectory after seeing outcomes. During exact ActionTape
replay capture the existing WorldRenderSnapshot every2steps, including real
positions, phases, resources and committed trajectories. Export a portable
browser playback with timeline, view rotation and per-UAV state. It extends
the existing renderer-neutral interface, avoiding optional PyVista/VTK
dependencies absent from the server. Visualization does not affect rollout.

Smoke seed699001 uses2steps and tests output/paired evaluation/playback, not
performance qualification. After PASS prepare a separately registered next
training block from the evaluated checkpoint, without starting it under this
evaluation command or reusing previous training seeds.
