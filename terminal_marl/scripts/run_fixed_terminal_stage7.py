"""Bounded stage7 dispatch: episodes in parallel, formal panels serially."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from time import perf_counter

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"

from terminal_marl.configs.fixed_terminal_stage7 import scenario_jobs, panel_scenario_jobs, ONLINE_ARMS
from terminal_marl.scripts.evaluate_fixed_terminal_stage7 import frozen_models, source_commit


def utc():
    return datetime.now(timezone.utc).isoformat()


def build_jobs(phase, models, output, source_stage4=None):
    output = Path(output)
    jobs = []
    if phase == "data":
        for split in ("train", "val"):
            for scenario in scenario_jobs(split):
                stem = f"{scenario['load']}-{scenario['soc']}-{scenario['seed']}"
                name = split + "-" + stem
                jobs.append({**scenario, "split": split, "mode": "generate", "name": name,
                             "output": str(output / "data" / split / stem)})
    elif phase == "episodes":
        for scenario in scenario_jobs("test"):
            stem = f"{scenario['load']}-{scenario['soc']}-{scenario['seed']}-{scenario['variant']}"
            for arm in ONLINE_ARMS:
                name = arm["name"] + "-" + stem
                jobs.append({**scenario, "mode": "episode", "arm": arm["name"], "name": name,
                             "output": str(output / "episodes" / name)})
    elif phase == "panels":
        for scenario in panel_scenario_jobs("test"):
            stem = f"{scenario['load']}-{scenario['soc']}-{scenario['seed']}-{scenario['variant']}"
            jobs.append({**scenario, "mode": "panels", "name": "panels-" + stem,
                         "output": str(output / "panels" / stem)})
    else:
        raise ValueError("unknown stage7 phase")
    for job in jobs:
        if phase == "data":
            command = [sys.executable, "-m", "terminal_marl.scripts.generate_fixed_terminal_stage7",
                       "--split", job["split"], "--load", job["load"], "--soc", job["soc"],
                       "--seed", str(job["seed"]), "--output", job["output"]]
            if source_stage4 is not None:
                command += ["--source-stage4", str(source_stage4)]
            job["expected_artifacts"] = [str(Path(job["output"]) / name) for name in ("generation.json", "frozen_input.json")]
            job["expected_artifacts"] += [str(Path(job["output"]) / f"t{time:g}" / (name + ".json"))
                for time in (0., 5., 15., 90.) for name in ("snapshot", "context", "trace", "features", "labels", "leaves")]
        else:
            command = [sys.executable, "-m", "terminal_marl.scripts.evaluate_fixed_terminal_stage7",
                       "--mode", job["mode"], "--load", job["load"], "--soc", job["soc"],
                       "--seed", str(job["seed"]), "--variant", job["variant"], "--output", job["output"],
                       "--models", str(models) if models is not None else "<FROZEN_MODELS>"]
            if "arm" in job:
                command += ["--arm", job["arm"]]
            required = ("result.json", "decisions.jsonl", "events.jsonl", "requests.jsonl", "frozen_input.json") if phase == "episodes" else ("generation.json", "frozen_input.json")
            job["expected_artifacts"] = [str(Path(job["output"]) / name) for name in required]
            if phase == "panels":
                job["expected_artifacts"] += [str(Path(job["output"]) / f"t{time:g}" / (name + ".json"))
                    for time in (0., 60.) for name in (*[a["name"] for a in ONLINE_ARMS], "snapshot", "summary", "full_values")]
        job["command"] = command
    return jobs


def atomic_json(path, data):
    temporary = path.with_name(path.name+".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    os.replace(temporary, path)


def run_phase(phase, models, output, workers=None, execute=False, source_stage4=None):
    workers = (1 if phase == "panels" else 20) if workers is None else workers
    if phase == "panels" and workers != 1:
        raise ValueError("formal panels require exactly one worker for isolated sequential timing")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output).resolve()
    models = Path(models).resolve() if models is not None else None
    source_stage4 = Path(source_stage4).resolve() if source_stage4 is not None else None
    jobs = build_jobs(phase, models, output, source_stage4)
    description = {"phase": phase, "workers": workers, "tasks": len(jobs), "jobs": jobs,
                   "execution_order": "frozen_scenario_order_then_arm; panels_sequential"}
    if not execute:
        return {"status": "dry_run", **description}
    if models is None and phase != "data":
        raise ValueError("frozen model directory required")
    if phase == "data" and source_stage4 is None:
        raise ValueError("formal data generation requires the stage4 train/val source")
    commit = source_commit(False)
    before = frozen_models(models, smoke=False) if phase != "data" else None
    state_path = output/f"execution_{phase}.json"
    progress_path = output/f"progress_{phase}.jsonl"
    if state_path.exists() or progress_path.exists():
        raise FileExistsError("phase already has state; no automatic resume/retry: "+str(state_path))
    for job in jobs:
        for path in (Path(job["output"]), output/"logs"/(job["name"]+".stdout.log"),
                     output/"logs"/(job["name"]+".stderr.log")):
            if path.exists():
                raise FileExistsError("job output/log already exists: "+str(path))
    output.mkdir(parents=True, exist_ok=True)
    (output/"logs").mkdir(exist_ok=True)
    phase_lock = output / ".phase.lock"
    # One output-root lock prevents races between all dispatcher phases.
    with phase_lock.open("x", encoding="utf-8") as stream:
        json.dump({"phase": phase, "pid": os.getpid(), "started_utc": utc()}, stream)
    for previous in output.glob("execution_*.json"):
        if json.loads(previous.read_text(encoding="utf-8")).get("status") == "running":
            phase_lock.unlink()
            raise ValueError("another phase is recorded as running: " + str(previous))
    started = perf_counter()
    state = {"schema": "fixed-terminal-stage7-dispatch-v1", **description, "status": "running",
             "state_path": str(state_path),
             "pid": os.getpid(), "started_utc": utc(), "finished_utc": None,
             "cwd": str(Path(__file__).resolve().parents[2]), "source_commit": commit,
             "thread_env": {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
             "frozen_models_before": before, "frozen_models_after": None,
             "model_check_scope": "single_frozen_stage7_model_and_manifest_dependencies_start_end",
             "counts": {"total": len(jobs), "pending": len(jobs), "running": 0, "completed": 0, "failed": 0},
             "peak_running": 0, "jobs": [{**job, "status": "pending", "pid": None, "returncode": None,
                                          "started_utc": None, "finished_utc": None, "duration_seconds": None}
                                         for job in jobs]}
    # Exclusive creation makes repeated execute fail before any subprocess starts.
    with state_path.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.flush()
    with progress_path.open("x", encoding="utf-8"):
        pass
    lock = Lock()
    environment = {**os.environ, **state["thread_env"]}

    def work(index):
        row = state["jobs"][index]
        begin = perf_counter()
        process = None
        try:
            if Path(row["output"]).exists():
                raise FileExistsError("job output appeared before launch: "+row["output"])
            stdout_path = output/"logs"/(row["name"]+".stdout.log")
            stderr_path = output/"logs"/(row["name"]+".stderr.log")
            with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
                with lock:
                    row.update(status="running", started_utc=utc(), stdout=str(stdout_path), stderr=str(stderr_path))
                    state["counts"]["pending"] -= 1
                    state["counts"]["running"] += 1
                    state["peak_running"] = max(state["peak_running"], state["counts"]["running"])
                    process = subprocess.Popen(row["command"], stdout=stdout, stderr=stderr, env=environment, cwd=str(Path(__file__).resolve().parents[2]))
                    row["pid"] = process.pid
                    atomic_json(state_path, state)
                    print(json.dumps({"event": "started", "name": row["name"], "pid": process.pid}), flush=True)
                code = process.wait()
            error = None
            if code == 0:
                missing = [name for name in row["expected_artifacts"] if not Path(name).is_file()]
                if missing:
                    error = "expected artifacts missing: " + repr(missing)
                elif phase in ("data", "panels"):
                    generation = json.loads((Path(row["output"]) / "generation.json").read_text(encoding="utf-8"))
                    expected_count = 4 if phase == "data" else 2
                    contexts = generation.get("contexts", [])
                    if generation.get("status") != "completed" or len(contexts) != expected_count or any(c.get("status") != "completed" for c in contexts):
                        error = "generation completion/context inventory mismatch"
        except Exception as exc:
            code, error = process.wait() if process is not None else None, repr(exc)
        with lock:
            state["counts"]["running" if row["status"] == "running" else "pending"] -= 1
            status = "completed" if code == 0 and error is None else "failed"
            state["counts"][status] += 1
            row.update(status=status, returncode=code, finished_utc=utc(), duration_seconds=perf_counter()-begin)
            if error is not None:
                row["error"] = error
            event = {"event": "finished", "name": row["name"], "pid": row["pid"], "status": status,
                     "returncode": code, "finished_utc": row["finished_utc"], "duration_seconds": row["duration_seconds"]}
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(json.dumps(event, allow_nan=False)+"\n")
                progress.flush()
            atomic_json(state_path, state)
            print(json.dumps(event), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(work, i) for i in range(len(jobs))]
        for future in as_completed(futures):
            future.result()
    try:
        after = frozen_models(models, smoke=False) if phase != "data" else None
        state["frozen_models_after"] = after
        state["model_hashes_unchanged"] = before == after
    except Exception as exc:
        state["model_hashes_unchanged"] = False
        state["model_check_error"] = repr(exc)
    state.update(status="completed" if state["counts"]["failed"] == 0 and state["model_hashes_unchanged"] else "failed",
                 finished_utc=utc(), duration_seconds=perf_counter()-started)
    state["exit_code"] = 0 if state["status"] == "completed" else 1
    atomic_json(state_path, state)
    phase_lock.unlink()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("data", "episodes", "panels"), required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--source-stage4", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = run_phase(args.phase, args.models, args.output, args.workers, args.execute, args.source_stage4)
    summary = ({key: result[key] for key in ("status", "phase", "counts", "exit_code", "duration_seconds", "state_path")}
               if args.execute else result)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return result.get("exit_code", 0)


if __name__ == "__main__":
    raise SystemExit(main())
