# V4.1 Action Schema S0 Qualification

Qualification artifact: `runs/v4_1_s0_qualification/v4_1_action_schema_qualification.json`.

- Catalog encode/decode: PASS for all 26 classes; all 25 RELEASE decodes equal their former V4 physical semantics.
- Teacher determinism: PASS (`same full state → same joint target`); canonical WAIT has no residual-ranking term.
- Execution boundary and ActionTape exact replay: PASS. The tape records action schema, requested joint index/decode/log-probability, Teacher/execution/shield action fields, observation hash and post-state hash.
- Fresh actor: PASS with `Categorical(26)`, finite log-probabilities and entropy. V4 30-way checkpoint loading was rejected as incompatible.
- Safety integration: collision/NMAC/hard/SafetyInvariantError/uncertified = `0/0/0/0/0` over the S0 real-shield catalog probes.
- I0 coverage gate: PASS as a policy definition; it now requires sufficient decision-relevant expert target families, not all 26 action indices.

```text
V4_1_ACTION_SCHEMA = PASS
V4_1_I0_RECOLLECTION_ALLOWED = true
I0 = NOT_RUN
I1_JOINT_BC = NOT_RUN
I2_DAGGER_PREFLIGHT_ALLOWED = false
PPO_ALLOWED = false
```

The `true` recollection permission authorizes a future, separate I0 request;
this S0 run did not recollect I0 data.
