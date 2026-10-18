# Worker lifecycle integration

The bounded integration applies the reviewed lifecycle foundation to
performance/parallel.py and corrects resource-statistics claims in
performance/process_metrics.py. It changes no candidate query, pad order,
candidate arithmetic, IPC payload, canonical three-ring x 24-bin policy or
authoritative world/checkpoint field. It does not integrate the separate
service_job/systemd artifact branch or add a resident supervisor.

Each initializer receives the constructor's explicit parent PID and sets
sys.dont_write_bytecode=True. On Linux it loads the real source-verified
performance.deadline package helper and calls bind_owned_worker_to_parent before
its core-import statements. The helper checks the parent PID before and after
PR_SET_PDEATHSIG. Fork may already contain inherited core modules; their count is
reported without claiming they were absent. Windows marks PDEATHSIG unsupported.
The helper identity includes deadline.py alongside parallel.py/process_metrics.py.

The same multiprocessing context constructs the executor and W-party Barrier.
Exactly W ready tasks wait with a 30-second bound; the parent requires observed
startup PIDs to equal the registered live-worker set with exactly W entries.
Startup metadata records parent/PID identity, isolated and dont-write-bytecode
flags, native libraries, cache reset observations and canonical bin count. No new
helper process/thread or process group is introduced. Existing Windows spawn and
Linux fork choice remains; Linux process inventory proof requires actual Linux
execution and cannot be inferred from Windows tests.

Only named derived accelerators are cleared before a worker's first task:
trajectory.poly_bounds, energy._integrable_plan/_segment_table, two safety
certificate dictionaries and optional root/peak/Bernstein clear/stats pairs.
Bernstein metadata is observed as trajectory._BERNSTEIN_METADATA. A missing pair
is explicitly absent, including on R0; a partial pair is rejected. Entries must
be empty after reset, while enabled/policy/max_entries/max_bytes fields remain
unchanged. Parent caches are not cleared. Immutable config/Gauss tables and
canonical policy are retained.

ReferencePool accepts a kernel label defaulting to R0 and allowing R0/OPT. It
changes summary metadata only; the verified root/manifest still determines
which source executes. Candidate/search/reduction and task identity contents
remain unchanged. Actual worker cache/work counts remain observations rather
than required matches to serial internal work.

RSS is reported as sequential sweep sums and a separately named lifetime-HWM
estimate. Neither is a proven strict RSS bound, and rss_limit_gate is report-only.
Legacy field names remain aliases with an explicit estimate note. External
launchers/paired collectors are outside pool accounting and require separate
process/CPU/RSS reporting. A pidfd becoming readable proves worker exit, not
reaping of orphaned grandchildren by this test, nor their complete timeout CPU.
No performance speed, relative CPU or RSS pass target is introduced.

tests/test_rotating_slots_performance_worker_lifecycle.py covers 1/2/4/6 distinct
startups, inherited warm cache reset without parent mutation, retained policy
flags, reconstruction against an independent fixed-R0 process, source identity
and Linux-only parent death/strict process inventory. The unrelated sentinel
must survive the real Linux owner-death test. All fresh test drivers use -I -B
and set dont_write_bytecode before imports. Existing real pad equivalence,
exception/death/cache/profile tests remain in test_rotating_slots_performance_parallel.py;
primitive kernel deadline tests remain in test_rotating_slots_performance_deadline.py.

The local Windows invocation ran parallel, worker_lifecycle and deadline tests
through an isolated -I -B pytest parent with bytecode disabled before imports.
Result: 40 passed, 5 skipped in 60.25s, exit 0. The skips were the real Linux
worker parent-death/process-inventory test and four Linux kernel deadline tests.
All seven recorded source/test hashes matched before/after execution.

Observed 1/2/4/6-worker startup sets contained exactly 1/2/4/6 distinct registered
PIDs. All named caches were empty after reset, enabled/policy metadata was
retained, and parent warm-cache snapshots stayed unchanged. This Windows run
observed fresh spawn caches; inherited warm fork-cache clearing and strict Linux
1+W process/PDEATH proof remain pending actual Linux execution. Two fresh R0
pool reconstructions matched the independent byte-verified R0 oracle's full
plan and SOC hex 0x1.0a8da8bfe9528p-4. The trajectory hash was
987a943eadd92fbc19cb36c7253e1caea7a7cd9e476ece3ce927318fe2bbe7bf.

Actual argv, stdout/stderr, numeric exit, before/after source digests, startup
JSON and independent oracle/reconstruction files are retained under
artifacts/rotating_slots/worker_lifecycle_tests_20261019a; observed_summary.json
links their exact paths and hashes. Full traffic, six-window performance,
Linux parent-death and synchronous RSS evidence are separate from these local
tests. Test counts do not establish full P4 acceptance; performance figures
remain reported observations without a numerical pass target.
