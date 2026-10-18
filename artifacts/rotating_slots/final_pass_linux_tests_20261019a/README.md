# Prepared final Linux joint verification

Preparation only. No SSH, pytest or server job was executed while creating these
files. The three files are a direct synced Python driver, a small PowerShell
launcher and this README. There is no giant embedded copy of the test driver.

Local readiness check (no SSH, pytest, output creation or Linux proof):

```powershell
pwsh -NoProfile -File artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_linux_tests.ps1 -LocalCheckOnly
```

After primary review, all relevant changes committed and sources frozen, the
prepared invocation is:

```powershell
pwsh -NoProfile -File artifacts/rotating_slots/final_pass_linux_tests_20261019a/run_linux_tests.ps1
```

The launcher computes local HEAD, current immediate core/performance Python
source hashes, the three CLI hashes, the 14 exact test-file hashes and its own
and run_tests.py hashes in that same invocation. LocalCheckOnly is provisional
while sources change; formal execution recalculates everything. The SHA-shaped
HEAD label is not a claim that a dirty checkout matches its Git blobs. No server
Git command runs. The tiny Base64 gate verifies the actual synced driver bytes
before runpy executes it. Shell arguments are single-quoted; the gate/request
use only Base64 characters and the final command must be below 30,000 characters.
The outer Python bootstrap has no inner quotes: exec(base64.b64decode(sys.argv[1])).

The Linux driver uses the fixed environment Python -I -B, sets dont_write_bytecode
before imports, native numerical threads to one and nice exactly 10 (without
adding another ten), then enables RUN_WINDOW_INTEGRATION=1. Its inherited CPU
affinity is recorded, not changed. Existing fresh -I -B test children remain in
place. It invokes pytest once, with -q -x -p no:cacheprovider and basetemp under:

/home/user/training_outputs/terminal/rotating_slots_perf_v1/final_pass_linux_tests_20261019a

The directory must be new. No output is overwritten. The test list is exactly
Bernstein precompute, root reuse, peak reuse, energy queries, P2, Cartesian
component, execution reuse, windows, paired, parallel, worker lifecycle, deadline,
runner and rotating-slots physics. Collection counts/nodeids come from the real
pytest hook, not an invented expected case count. Missing named files reject
before launch.

Before pytest, the existing provenance.implementation_fingerprint uses explicit
extras and modules={}, then verify_deployment performs the initial check. After
pytest, provenance._source_files with the same explicit extras and modules={}
is compared exactly with that initial file map and its canonical digest, while
the named launch hash map and exact R0 export are independently verified again,
including after a test failure. This test-harness file-inventory comparison
does not claim production mixed-core/module-origin proof; pytest/conftest imports
do not enlarge the fixed inventory. Production provenance code is unchanged.
The driver records deployment/reference data, source hashes, actual argv,
UTC/monotonic duration, stdout/stderr (Python and OS file descriptors), numeric
exit, actual collection, load/meminfo, affinity, nice and thread environment.
The existing frozen R0 export is read, never regenerated. A source-verification
failure and pytest's own exit are retained separately. Local SSH argv/request,
stdout/stderr/UTC/exit are preserved in a new local_launch_UUID artifact directory
only when the approved remote branch is actually invoked.

Estimated duration is roughly 3–5 minutes based on prior bounded local test
times; Linux integration timing is uncertain. There is no new hard timeout and
no long-running monitor in this harness. A failed suite stops at its first
failure. It does not send signals to unrelated jobs, discover/kill processes
recursively, alter the test matrix or claim complete accounting after an
external interruption. This is functional/numerical/lifecycle verification;
performance values remain observations without a numerical pass threshold.
