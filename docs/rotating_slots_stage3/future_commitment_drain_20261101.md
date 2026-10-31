# R4 continuation and admitted-aircraft drain

The runner resumes each original completed R4 arm at absolute time 320. It restores the saved world and latency wrapper together, including pending computation, absolute future reservations, reference templates, previous predictions, selector history, and all caches that the original checkpoint actually stores. It compares their serialized representations before advancing. The original provider validation cache was never serialized; its reconstruction and subsequent wall times cannot be claimed identical to an uninterrupted resident process. No cache clearing, scheduler cold initialization, reseeding, forced replanning, or algorithm changes are added.

Traffic and shared FIFO admission continue through the exact settled time 600. At that boundary, the runner keeps the already released manifest prefix, retains the original seed and rate, sets its arrival window to 600, and binds the resulting manifest digest in subsequent checkpoints. The cutoff receipt retains the original digest/window/count and every suppressed request ID. Admission is disabled. Released requests still waiting outside remain in the request table and are listed as excluded external pending requests, without becoming aircraft failures. This intervention is reapplied after every post-cutoff restore because the existing restoration API installs FIFO admission by default.

The cutoff airborne cohort includes INBOUND, holding, transfer, normal final, emergency final, and vertical descent states. All admitted historical aircraft outcomes are reported separately; initial ground fixtures and rerouted external requests are excluded. Drain completion requires no global airborne aircraft and a known terminal outcome for every admitted cohort member. Touchdown resolves an aircraft in LANDED_SERVICE or CLEARED without waiting for pad service. Battery depletion and peak-budget exhaustion retain the existing physical failure archive reasons. Invalid execution, unknown admitted outcomes, and exceptions remain errors, never successful completion. Stale reservation-registry entries are reported separately from physical aircraft state.

The runner advances on five-second stable boundaries, publishes compact progress on each boundary, and saves durable checkpoints approximately every 60 simulated seconds, at restore, at cutoff, and at segment termination. A wall segment cap requests a pause only after a stable boundary; it changes no physical horizon. Explicit `--resume` accepts only PAUSED_CHECKPOINT and restores its full state into a new segment stream copy. No error retry is automatic. Original checkpoints, histories, results, and ledgers remain read-only. Streamed checkpoint sources are copied before reopening; in-memory checkpoint histories remain in memory and are preserved in full.

Each segment writes a complete, unique replay archive including the original history and initializer record provenance. The final replay is passed to the existing continuous energy/separation, resource, PVA/C2, and rejected-proposal audit through the separate `audit` command. Replay export and audit failures retain error or incomplete status; automated output never supplies human acceptance.

## Commands

Commands run on the server from `~/projects/terminal`; Syncthing supplies the locally reviewed source. `<original-freeze>` is the existing e3ae9a96 R4 source freeze and `<new-commit>` is the reviewed continuation commit. Algorithm and configuration hashes are checked against the old freeze; historical documentation changes do not invalidate that algorithm check. The new binding records hashes of this runner, its tests, and directly used runner/audit helpers.

```bash
python scripts/stage3/run_future_commitment_drain.py smoke \
  --source-output "$HOME/training_outputs/terminal/rs_future_commitment_20261031" \
  --output "$HOME/training_outputs/terminal/rs_future_commitment_drain_20261101" \
  --source-freeze <original-freeze> --code-commit <new-commit> \
  --root s3r23 --method FULL_H --cpu 16

python scripts/stage3/run_future_commitment_drain.py run \
  --source-output "$HOME/training_outputs/terminal/rs_future_commitment_20261031" \
  --output "$HOME/training_outputs/terminal/rs_future_commitment_drain_20261101" \
  --source-freeze <original-freeze> --code-commit <new-commit> \
  --root s3r23 --method FULL_H --cpu 16 --segment-wall-s 1800
```

The four fixed root/method/CPU tuples are s3r23/FULL_H/16, s3r23/RULE_LOCAL_K8/17, s3r24/FULL_H/18, and s3r24/RULE_LOCAL_K8/19. Each command operates one explicit arm. Restore-only smoke writes `restore_checks/<root>/<method>` without creating a formal arm or calling physics or the solver. To continue a normally paused formal segment, repeat its run command with `--resume`. Complete or erroneous arms cannot resume through that option.

```bash
python scripts/stage3/run_future_commitment_drain.py audit \
  --output "$HOME/training_outputs/terminal/rs_future_commitment_drain_20261101" \
  --root s3r23 --method FULL_H
python scripts/stage3/run_future_commitment_drain.py report \
  --output "$HOME/training_outputs/terminal/rs_future_commitment_drain_20261101"
```

The caller bounds the separate audit process. `report` reads saved arm results and audits into `analysis_summary.json`; missing or paused arms remain explicit. No server experiment was launched during implementation.

## Local checks

`tests/test_future_commitment_drain.py` uses small real checkpoint fixtures for streamed and in-memory histories, pending latency and future absolute reservations, cutoff release/admission behavior, post-cutoff resume, normal/emergency/battery/peak classification, initial-ground exclusion, finite JSON output, segmented replay completion, smoke isolation, and invalid-run rejection. Its synthetic advance fixture verifies runner bookkeeping without invoking the expensive normal optimizer. Actual four-arm restore smoke and physical continuation are separate server executions.

Local verification: `venv/Scripts/python.exe -m pytest tests/test_future_commitment_drain.py -q --basetemp=future_drain_units_20261101a` completed with 13 passed in 2.95 seconds. An earlier `.tmp` basetemp attempt returned WinError 5 before fixture setup; the listed top-level workspace basetemp was writable. The runner's `--help` command completed successfully. No server smoke, physical rollout, or saved-replay safety audit is included in these local test results.
