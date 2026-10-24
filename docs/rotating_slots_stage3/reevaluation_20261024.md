# Frozen encoding reevaluation — 2026-10-24

This is one newly authorized bounded online evaluation of the encoding optimization. It uses the original frozen model, normalizer, split, seeds, physical t0 states, safety rules, planning budgets and cold-process CPU12 regime. It performs no training, label collection, seed allocation, or new heldout rank measurement. The original campaign is read-only.

Parent: `/home/user/training_outputs/terminal/stage3_learning_v1_0_20261023`.
New output: `/home/user/training_outputs/terminal/stage3_encoding_reevaluation_20261024`.

Frozen model SHA256: `b565c2a98af7f30c5b8179e823a6447bb1ccecebcbf4da2dfdcb44679ce85969`.
Frozen normalizer SHA256: `007de6c700959a8cadc0397e8896f279aeca0070fe12bbac5441c3c5b2e63491`.

The original fingerprint is retained and labeled `frozen_model_contract` for the unchanged model compatibility gate. Actual execution commit, source hashes and environment are separately recorded in `execution_provenance`; replay `code_commit` uses that actual commit. Checkpoint identity remains original. Only root file paths are rewritten. Model files, split, heldout signal, selected initial checkpoints/state/manifest files are byte copies. Baseline summaries are copied under `prior_evidence/`. The 24-root index order remains original; only the six selected cases are materialized. Large raw streams are not copied.

The unchanged `run_evaluation` runs 18 same-state jobs: test roots `s3r21..s3r24`, then validation roots `s3r17,s3r18`, with the existing three-arm rotation and 90 s per-job cap. The paired closed-loop evaluation uses `s3r23`, heuristic then learned, requested 60 simulated seconds, existing equal wall allocation and completed 5 s boundaries. Its new ledger has the original 3600 s evaluation cap, original 900 s test cap and original campaign caps; unused phase budget is not reallocated. Each tests/evaluate stage has a one-attempt receipt and always pauses the ledger on completion or exception. Prepare rejects any nonempty output. No retry or reset is exposed.

Commit and push implementation first, then freeze locally (source files must be committed; unrelated dirty files may remain):

```powershell
python scripts/stage3/reevaluate.py freeze --output artifacts/rotating_slots_stage3_20261024/encoding_reevaluation_source
```

After Syncthing carries the commit and freeze, on the server use the existing Python environment and set all thread variables before starting Python:

```bash
cd ~/projects/terminal
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=/home/user/envs/terminal_stage2_v1/lib/python3.12/site-packages
PYTHON=$HOME/envs/terminal_env/bin/python
OLD=$HOME/training_outputs/terminal/stage3_learning_v1_0_20261023
NEW=$HOME/training_outputs/terminal/stage3_encoding_reevaluation_20261024
$PYTHON scripts/stage3/reevaluate.py prepare --parent "$OLD" --output "$NEW" --source-freeze artifacts/rotating_slots_stage3_20261024/encoding_reevaluation_source
$PYTHON scripts/stage3/reevaluate.py tests --output "$NEW"
$PYTHON scripts/stage3/reevaluate.py evaluate --output "$NEW"
```

Prepare verifies current source hashes and rejects all changes against the parent source inventory except the reviewed encoding/model implementation, replay/controller projection, reevaluation/provenance entrypoint and specific tests/benchmark additions. Physical/config/contract and protocol files must match. It verifies unchanged numerical dependency versions, thread environment, a working single-worker CP probe and CPU12 availability. Model loading still uses the original strict fingerprint check. Evaluate repeats the resource probe. A failed preparation leaves any partial new output intact and cannot be reset by prepare; return to the primary agent for disposition.

Status: implementation protocol only; this document does not claim that the formal reevaluation has started or completed. A performance claim requires the new observed output and comparison against prior summaries, with quality and safety observations retained.
