"""Scenario-grouped observed-cohort value data; labels and features separated."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
from .baselines import apply_scheduler
from .learning_features import FEATURE_NAMES, extract_features
from .learning_scheduler import make_pool
from .value_analysis import visible_snapshot, evaluate_root, delta_value
from .scenarios import generate_scenario
from terminal_marl.envs.fixed_terminal_env import FixedTerminalEnv
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage4 import SNAPSHOT_TIMES, validate_split


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def generate_context(env, output, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"snapshot.json", visible_snapshot(env))
    roots, pool_meta = make_pool(env)
    write_json(output/"roots.json", {"roots": [r.serializable() for r in roots], "pool_metadata": pool_meta})
    tick = perf_counter()
    features = [{"root_id": r.root_id, "values": extract_features(env, r.plans)} for r in roots]
    feature_seconds = perf_counter()-tick
    write_json(output/"features.json", {"feature_names": FEATURE_NAMES, "rows": features})
    values = {}
    for root in roots:
        evaluation = evaluate_root(env, root.plans, (env.config.horizon,), realized_future=False)
        write_json(output/"rollouts"/f"{root.root_id}.json", evaluation)
        if evaluation["status"] != "feasible":
            raise RuntimeError("label root unexpectedly rejected: "+evaluation["reason"])
        values[root.root_id] = evaluation["values"][f"{env.config.horizon:g}"]
        print(json.dumps({"context": str(output), "root": root.root_id, "seconds": evaluation["seconds"]}), flush=True)
    hold = next(r for r in roots if "HOLD" in r.labels)
    labels = [{"root_id": r.root_id, "value": values[r.root_id],
               "delta_vs_HOLD": delta_value(values[r.root_id], values[hold.root_id])} for r in roots]
    write_json(output/"labels.json", {"information_scope": "observed_arrived_cohort_FCFS_until_original_horizon",
                                     "rows": labels})
    record = {"schema": "fixed-terminal-stage4-context-v1", **metadata, "root_time": env.time,
              "status": "completed", "root_count": len(roots), "degenerate": len(roots) == 1,
              "feature_dim": len(FEATURE_NAMES), "config": asdict(env.config), "prefix": "joint_window",
              "feature_seconds": feature_seconds, "generation_seconds": pool_meta["generation_seconds"],
              "pool_metadata": pool_meta, "scenario_group": [metadata["seed"], metadata["load"], metadata["soc"]]}
    write_json(output/"context.json", record)
    return record


def generate_scenario_data(output, split, seed, load, soc, *, config=DEFAULT_CONFIG, scenario=None, times=SNAPSHOT_TIMES):
    validate_split(split, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = FixedTerminalEnv(config, scenario or generate_scenario(config, seed, load, soc))
    times = tuple(float(t) for t in times)
    if tuple(sorted(set(times))) != times or any(t < 0 or t >= config.horizon or t/config.dt != round(t/config.dt) for t in times):
        raise ValueError("snapshot times must be sorted unique pre-cutoff simulation ticks")
    write_json(output/"frozen_input.json", {"split": split, "seed": seed, "load": load, "soc": soc,
                                          "scenario": asdict(env.scenario), "config": asdict(config), "times": times})
    next_decision, records = 0., []
    for target in times:
        while env.time < target-1e-9:
            if env.time >= next_decision-1e-9:
                apply_scheduler(env, "joint_window")
                next_decision += 5.
            env.step()
        records.append(generate_context(env, output/f"t{target:g}", {"split": split, "seed": seed, "load": load, "soc": soc}))
    write_json(output/"generation.json", {"status": "completed", "contexts": records, "prefix_decisions_before_snapshot": True})
    return records


def load_contexts(directory, expected_split, *, training=False):
    if training and expected_split not in ("train", "val", "pilot"):
        raise ValueError("training loader must never read test data")
    contexts = []
    for path in sorted(Path(directory).rglob("context.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["split"] != expected_split:
            raise ValueError("mixed/unexpected split directory")
        validate_split(metadata["split"], metadata["seed"])
        if metadata["status"] != "completed":
            raise ValueError("incomplete context")
        base = path.parent
        features = json.loads((base/"features.json").read_text(encoding="utf-8"))
        if features["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("feature schema mismatch")
        labels = json.loads((base/"labels.json").read_text(encoding="utf-8"))
        roots = json.loads((base/"roots.json").read_text(encoding="utf-8"))
        ids = [r["root_id"] for r in roots["roots"]]
        if len(set(ids)) != len(ids) or ids != [r["root_id"] for r in features["rows"]] or ids != [r["root_id"] for r in labels["rows"]]:
            raise ValueError("root/feature/label alignment mismatch")
        contexts.append({"metadata": metadata, "path": str(base), "roots": roots["roots"],
                         "pool_metadata": roots["pool_metadata"], "features": [r["values"] for r in features["rows"]],
                         "values": [r["value"] for r in labels["rows"]],
                         "targets": [r["delta_vs_HOLD"]["delta_J"] for r in labels["rows"]]})
    if not contexts:
        raise ValueError("no completed contexts found")
    keys = [(c["metadata"]["seed"], c["metadata"]["load"], c["metadata"]["soc"], c["metadata"]["root_time"]) for c in contexts]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate context")
    return contexts
