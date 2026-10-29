# v3 checkpoint diagnostic entrypoint

`scripts/stage3/run_selective_reoptimization_repair_v3.py` accepts one explicit
`snapshot`, `smoke`, or `arm` action. No action starts a matrix or retries.
The v3 ledger starts at zero, caps aggregate active child wall time at 5400 s,
and cites the unchanged v2 repair ledger cost 1477.5308457711415 s and earlier
campaign cost 7323.47022066447 s. Snapshot and probe work uses that same ledger.
The old runner retains its default accounting policy.
`credit_observed_receipt(output, job_id, path)` imports an explicitly measured
JSON receipt's `wall_ms / 1000` once, including nonzero exits. It retains the
receipt and path, rejects reused job IDs or identical receipt references, and
does not add historical campaign totals to the new ledger.

Snapshot requires explicit `--checkpoint`, `--expected-source-time`, unique
`--case`, `--variant`, root, physical CPU, and reviewed `--source-freeze`.
Different-time boundaries additionally require `--allow-boundary-difference`.
The entrypoint restores physical active/soft trajectories, reservations, and
saved selector history/blockers. It leaves candidate caches cold, does not run
the old callback/pending proposal, does not advance physics, and cannot publish.
Selector history is the checkpoint history, not a reconstructed historical
callback-before history. Cold catalogs differ from the historical warm cache.

R4 at historical source 60/effective 80 has a nearby boundary at 65; the cold
diagnostic therefore uses source 65/effective 85. R12 at historical source
206.03508481464814/effective 226.03508481464814 has boundary 210 and uses
source 210/effective 230. Both checkpoints have RESULT pending tasks and no
prepared packets. They are common boundary comparisons, not exact historical
solver-input replays or evidence for attributing the historical result.

Prepared packets elsewhere do serialize preparation-stage source/projected
snapshots, but source times differ from original callback source times: R1
11.759665801888332 versus 0; R8 136.88349333964288 versus 127.00476574490312;
R10 177.33075895067304 versus 167.00476574490312; P2 243.6739776940085 versus
233.65165837993845. Their packet resources are in a separate registry; physical
snapshot reservations come from world slots/pads. The entrypoint records packet
metadata and retains committed physical resources without resuming packets.

`actual_diagnostic.json` is saved before extra validation. It contains the
complete retained actual catalog/selection/global frame, full callback records
(including bounded per-call candidate observations and CP constraints), explicit
observation truncations, checkpoint identity, source/effective times, and costs.
Infinite reservation ends use an explicit `$nonfinite_float` JSON object.

`--probe-immediate` independently evaluates at most three combinations: one
existing immediate MOVE for `initial_middle_01`, one for `initial_middle_02`,
and both. Immediate means first start equals projected effective time; each UID's
first hash-sorted existing candidate is used. Missing candidates are recorded.
Every unspecified selected choice and the complete background remain fixed.
Probes perform no CP search, create no new columns, consume no formal CP budget,
and report their independent validation cost. `--probe-seconds` is at most 10.

`opportunity_cooperative_v2` is snapshot-only. Smoke and formal arm require
`opportunity_executable_v3`, original t0 inputs and seeds 7324/7325, fixed physical
CPU 12/14, K=8/max=12, lead 20 s, heartbeat 5 s, and unchanged physical/safety
objectives. Smoke lasts 10 simulated seconds; one formal arm lasts 300 simulated
seconds and has a 1200 s child wall cap. Formal smoke/arms share a frozen source
hash; each non-overwritable snapshot case records its own source hash.
