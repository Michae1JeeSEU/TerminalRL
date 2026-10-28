# 2026-10-29 selective repair v2 results

Status: pending execution and primary review. This file records the new protocol separately from the original Stage A contract and the 2026-10-28 exploratory follow-up.

The fixed treatment is `opportunity_cooperative_v2` with `coalesced_restart_v2`, K8/max12, lead20, heartbeat5, preparation margin1 and handoff reserve4. Formal offset0 is fixed before snapshot diagnostics. No learning, dynamic K, lead scan, physics change, objective change or emergency-policy change is authorized by this repair runner.

The formal scope is one 300-second K8 arm for each original root (`s3r23`, seed7324, CPU12; `s3r24`, seed7325, CPU14) from the original Stage A t0. Numerical libraries use one thread. Each arm has a 1200-second child wall limit. Output is `/home/user/training_outputs/terminal/rs_selective_repair_20261029`.

The new 5400 child-wall-second allowance includes prior audits82.8723364593 seconds and all new tests, smokes, snapshots, formal runs, reporting and audits. Historical cumulative usage7323.47022066447 seconds is retained separately. `repair_budget_ledger.json` includes settled and actively running child wall time; its parent watchdog stops work during execution. Failed IDs are never automatically retried. Previously executed checks must be recorded before starting new work.

The two fixed diagnostic states are old follow-up baseline20 `checkpoint_0040.json` for the two roots. Each individual command restores the same state with cold normal caches and lead20. Offsets0/1/2 and same-version FULL_H share the global frame identity. Duplicate selected sets count as one distinct set. Preparation uses the existing counterfactual preparation function, discards its token, and never publishes or advances physical state. A missing fixed checkpoint stops the action.

The old audited baseline20 is an overall version before/after reference. It is not a fully fair control comparison. Saved candidate CP suffixes remain predictions. Actual publication acknowledgements, executor start events and physical states define execution.

## Evidence to fill after execution

| Evidence | Status | Path |
|---|---|---|
| Reviewed source freeze and targeted checks | Pending | To record |
| 10-second smoke | Pending | `smokes/<root>/` |
| 40-second fixed-state diagnostics | Pending | `snapshots/<root>/*/result.json` |
| Two 300-second formal arms | Pending | `arms/<root>/evaluation/<root>/RULE_LOCAL_K8/` |
| Independent replay/resource/PVA/proposal audits | Pending | Per-arm `audit_result.json` |
| Saved accounting report | Pending | `repair_analysis_summary.json`, `REPAIR_REPORT.md` |
| Human replay viewing | HUMAN_PENDING | Record actual observations separately |

The saved report carries normal/emergency/failure/pending counts, per-ring holding/energy, fairness ages and unknowns, hard overflow causes/K/times, separate task and aircraft admission denominators, complete lifecycle work including partial pending computation, exclusive candidate/trajectory/prefix/CP timing, feature/shared-public timing, and per-task certificate/ready/publication/start/next-snapshot timestamps. Missing evidence remains null or UNKNOWN. Automated audit status remains pending until the audit has actually run.

## Individual server entrypoints

Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to1 before Python starts. Commands below are separate requested actions, not a sequence launched by the script. Replace `PYTHON` and `FREEZE` with reviewed absolute paths.

```bash
PYTHON scripts/stage3/run_selective_reoptimization_repair.py smoke --root s3r23 --cpu 12 --source-freeze FREEZE
PYTHON scripts/stage3/run_selective_reoptimization_repair.py snapshot --root s3r23 --cpu 12 --method RULE_LOCAL_K8 --offset 0 --source-freeze FREEZE
PYTHON scripts/stage3/run_selective_reoptimization_repair.py snapshot --root s3r23 --cpu 12 --method FULL_H --offset 0 --source-freeze FREEZE
PYTHON scripts/stage3/run_selective_reoptimization_repair.py arm --root s3r23 --cpu 12 --source-freeze FREEZE
PYTHON scripts/stage3/run_selective_reoptimization_repair.py report
```

For separately authorized tests and saved-replay audits, the mechanical wrapper imports `run_bounded_job(output, unique_job_id, exact_command, cap_s=...)` from the repair runner. This helper charges the same active-child ledger and does not choose or launch a matrix. The report entrypoint is saved-data-only and does not launch audits. An audit wrapper may invoke the existing `report_selective_reoptimization_a.audit_job` against one new formal result directory.
