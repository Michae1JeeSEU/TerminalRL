# Linux wall deadline primitive (P1, unintegrated)

Acceptance: **NOT_QUALIFIED**. This isolated primitive does not complete performance
contract §7.4, CPU accounting, process-tree orchestration, or hard-timeout
qualification. The existing runner, parallel implementation, measurements and
checkpoint implementation do not import or enable it. It has no P2 changes.

`LinuxWallDeadline(budget_seconds, evidence_directory)` is a one-use context
manager for a Linux main process on its main thread. It rejects a nonfinite or
nonpositive budget, a preexisting real-time timer, any nondefault SIGALRM
disposition (including ignored signals), and a blocked SIGALRM. Signal/timer
ownership must not be changed concurrently or inside the context. It creates no
thread or process. A subprocess launched directly by an external task launcher
counts as its own main process; multiprocessing children cannot enter the guard.

Before arming `setitimer(ITIMER_REAL)`, it writes a uniquely named
`deadline_<guard_id>.json` using the existing fsync-and-replace JSON writer. The
ARMED record contains its own PID, UTC and monotonic start, monotonic deadline,
budget and expected expiry signal. Evidence persistence consumes budget; the
timer uses the remaining duration. If persistence consumes the whole budget,
the timer is armed for the minimum positive interval of one microsecond. This
is a kernel timer subject to OS scheduling, not a certified exact wall bound.

Expiry uses the default SIGALRM disposition and kills the process without a
Python callback, so it does not rely on the Python main loop or a GIL-acquiring
watchdog. On ordinary or exceptional context exit, it disarms the timer,
restores its owned signal disposition, and records DISARMED with exit kind.
Expiry winning a race with context cleanup still terminates the process.
An ARMED record alone is not proof the timer actually armed: abrupt failure
can occur between evidence persistence and the kernel call. A reported arming
failure records ARM_FAILED where possible.

The dead process **cannot write PERFORMANCE_TIMEOUT**. A future external
launcher/exit collector must observe the exact owned process exit by SIGALRM,
match PID/run identity to this guard's ARMED metadata, classify the external
budget event as PERFORMANCE_TIMEOUT, preserve the last complete checkpoint,
and keep it out of physical failure rates and normal completion means. An ARMED
record without exact termination evidence does not establish timeout. This
module does not write checkpoints, classify arbitrary exits, or guarantee that
all descendants terminate. SIGALRM blocked or overridden after entry defeats
the intended contract and is outside the primitive's ownership precondition.

`bind_owned_worker_to_parent(expected_parent_pid)` is an explicit worker
initializer helper. On Linux it checks that the calling child's parent is the
known expected PID, registers `prctl(PR_SET_PDEATHSIG, SIGKILL)`, then checks the
parent again. A mismatch kills only the calling child. Registration errors
raise before task work. It never signals another PID or a process group.
Future runner integration must call it in every owned worker before task work;
the helper is currently unintegrated. Normal parent shutdown must still join
workers. It does not bind grandchildren, survive privilege transitions that
clear the Linux setting, or make unrelated process ownership claims. Linux
parent-death semantics refer to the creating thread, so owned workers must be
created by the parent main thread that owns the guarded task.

The focused tests include deterministic validation/ownership checks plus Linux
sacrificial subprocess checks for cancellation, native computation termination,
parent-death worker shutdown, and an unrelated sentinel that remains alive.
All Python children use `-I -B` and disable bytecode writes before project
imports. Test owners bound waits and clean up their own processes; the orphan
check uses pidfds and accepts either absent or zombie state as nonexecuting.
Windows executes applicable deterministic tests and explicitly skips Linux
kernel integration. These tests do not qualify the complete performance path.

Local Windows verification on 2026-10-18:

```text
venv/Scripts/python.exe -B -m pytest tests/test_rotating_slots_performance_deadline.py -q --basetemp=artifacts/rotating_slots/deadline_tests_20261018a -p no:cacheprovider
15 passed, 4 skipped in 10.36s; exit 0
```

The cache provider was disabled to avoid rewriting existing pytest cache files.
The four Linux subprocess checks were skipped, not qualified by this run.

Linux verification on 2026-10-18 was executed separately on `lab-server`.
The primary agent read the server logs and exit artifact; the execution worker
recorded matching local/server SHA values for the three new primitive, test,
and documentation files before the run. The focused invocation used Python
`-I -B`, `sys.dont_write_bytecode=True`, numerical-library threads set to one,
and pytest arguments:

```text
tests/test_rotating_slots_performance_deadline.py -q -p no:cacheprovider --basetemp=/home/user/training_outputs/terminal/rotating_slots_perf_v1/linux_deadline_tests_20261018a/tmp
19 passed in 4.10s; exit 0
```

Evidence directory:
`/home/user/training_outputs/terminal/rotating_slots_perf_v1/linux_deadline_tests_20261018a`.
`stdout.log` contains the pytest result, captured stderr was empty, and
`exitcode.txt` contains `0`. `started_utc.txt` records
`2026-10-18T10:41:31.385017086Z`; execution finished at
`2026-10-18T10:41:35.645585600Z`. The measured wrapper wall time was
`4.260568514` seconds.

Thus the four subprocess checks skipped on Windows actually passed on Linux;
the Linux result is 19 passed, rather than a reinterpretation of the Windows
skips. This validates the isolated tested primitive only. Runner integration,
external PERFORMANCE_TIMEOUT classification, complete owned-process lifecycle
handling and full-system hard-timeout qualification remain outside this primitive;
acceptance remains **NOT_QUALIFIED**.
