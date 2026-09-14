# V4.2 I2 Safe DAgger Preflight — PASS

## Experiment evidence

- Status / evidence type: `newly_run`.
- Workdir: `~/projects/terminal` on `lab-server`.
- Student input: the fresh V4.2 I1 checkpoint, SHA-256
  `b40bdf0568ef46a9ec259207da5ca89e2fc484a63a13e70fe82c1bbfd03389e1`.
- Dataset parent: the read-only V4.2 derived I0 target view, SHA-256
  `681119824a739c6ec08b7900d821e4969eadfa85a9a9a307d0e5f618dfcf61fe`.
- Protocol: two fixed seeds (9701, 9702), six closed-loop steps each, student
  sampled gate/proposal actions, Teacher V4.1 semantic labels queried twice
  from each visited pre-action state, and Teacher label-only execution mode.
- Forbidden operations: no DAgger retraining, PPO, reward/Observation/Teacher/
  safety modification, weighting, or oversampling.

## Closed-loop and provenance result

The student was the sole source of every requested action:

```text
student sample → decoded requested action → ExecutionScheduler
→ TrajectoryGenerator → SafetyFilter → PredictiveShield → environment
```

Teacher labels were captured only after each student-visited state was fixed
and were never passed to execution. The preflight recorded 96 new,
decision-event-unique student-visited labels. D1 contains those records plus
the 1,500 parent derived-I0 rows (1,596 total). Its SHA-256 is
`f9ff72350e298317b5053fe913725938adc92d18806f1786a46e68db7433b624`.

Student diagnostics: 84 WAIT / 12 RELEASE; Teacher diagnostics: 73 WAIT / 23
RELEASE. There were 23 gate disagreements and 18 proposal disagreements on
Teacher-RELEASE states. Student RELEASE proposals used five values (`0, 10,
12, 13, 14`), so neither gate nor proposal collapsed. Completion and
throughput were both zero in this deliberately short preflight; queue stayed
zero and the shield was not used. These are diagnostics, not performance gates.

Both complete six-step ActionTapes replayed exactly. Each replay verified
reset/configuration, every observation, requested gate/proposal, deterministic
Teacher target, decoded/executed action, post-state, and a direct recomputation
of the saved requested joint log-probability.

Safety counters were all zero: collision, NMAC, hard violation,
SafetyInvariantError, and uncertified.

```text
V4_2_I2_DAGGER_PREFLIGHT = PASS
I2_DAGGER_ROUND1_ALLOWED = true
PPO_ALLOWED = false
```

## Protocol risks

This is a two-seed, 12-step internal engineering preflight rather than an
independent safety/generalization evaluation. It permits a separately
authorized DAgger Round 1 but does not establish throughput performance and
does not authorize PPO.
