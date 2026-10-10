"""Read-only runtime probe for the fixed P3 neighbor actor path.

The probe performs forward, auxiliary-loss, backward, and gradient-clipping
work with a fixed model state.  It never constructs an optimizer, applies an
optimizer update, saves a checkpoint, runs an environment, calls a planner,
queries a teacher, or reports action-quality metrics.  The official nine-cycle
loader may read its archived inputs while rebuilding the already saved
resource tensors; that load is outside the timed probe.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from terminal_marl.algo.intent_neighbor_actor import IntentNeighborActor
from terminal_marl.algo.joint_ppo_v4_2 import hierarchical_bc_loss
from terminal_marl.scripts.diagnose_p3_intent_neighbor import measure, measure_pairs, smoke_inputs
from terminal_marl.training.p3_neighbor_witness_dataset import (
    WitnessTargets,
    auxiliary_loss,
    combine_witness_targets,
    load_witness_targets,
)
from terminal_marl.training.p3_v26_train_dataset import load_nine_cycle_train


PROTOCOL = "p3_neighbor_runtime_profile_20261010_v1"
MODEL_SEED = 91911
AUX_SEED = 91912
HELD_SEED = 893511
DEFAULT_UPDATES = 16
DEFAULT_WARMUPS = 2
DEFAULT_ACTOR_BATCH = 128
DEFAULT_AUX_BATCH = 64
DEFAULT_WALL_CAP = 180.0
DEFAULT_GATE_MODE = "first_feasible_union"


class ProbeError(RuntimeError):
    """Expected fail-closed probe error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {"pid": os.getpid(), "cpu_affinity": None, "cpu_number": None,
                            "rss_bytes": None, "peak_rss_bytes": None}
    try:
        info["cpu_affinity"] = sorted(int(value) for value in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        import psutil  # type: ignore
        process = psutil.Process(os.getpid())
        info["cpu_affinity"] = [int(value) for value in process.cpu_affinity()]
        info["cpu_number"] = int(process.cpu_num())
        info["rss_bytes"] = int(process.memory_info().rss)
    except (ImportError, OSError, RuntimeError):
        pass
    try:
        import resource
        info["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError):
        pass
    return info


def rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError, RuntimeError):
        return None


def input_sha_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        result.append({"path": str(path), "exists": path.is_file(),
                       "sha256": sha256_file(path) if path.is_file() else None})
    return result


def target_support(target: WitnessTargets) -> dict[str, int]:
    return {
        "rows": int(len(target.state_index)),
        "known_pairs": int(target.known.sum()),
        "blocked_pairs": int((target.known & target.blocked.bool()).sum()),
        "scene_known": int(target.scene_known.sum()),
        "scene_blocked": int((target.scene_known & target.scene_blocked.bool()).sum()),
    }


def split_indices(paired: Any, target: WitnessTargets, held_seed: int) -> tuple[torch.Tensor, ...]:
    seeds = torch.tensor([int(context["seed"]) for context in paired.original.contexts])
    held = seeds == int(held_seed)
    require(torch.equal(seeds[target.state_index], target.seeds),
            "auxiliary split identity mismatch")
    target_held = target.seeds == int(held_seed)
    return torch.where(~held)[0], torch.where(held)[0], torch.where(~target_held)[0], torch.where(target_held)[0]


def actor_batches(pool: torch.Tensor, size: int, count: int,
                  generator: torch.Generator) -> list[torch.Tensor]:
    require(pool.ndim == 1 and len(pool) > 0, "empty probe batch order")
    require(size >= 1 and count >= 1, "invalid probe batch shape")
    permutation = torch.randperm(len(pool), generator=generator)
    offset = 0
    batches = []
    for _ in range(count):
        if offset == len(pool):
            permutation = torch.randperm(len(pool), generator=generator)
            offset = 0
        local = permutation[offset:min(offset + size, len(pool))]
        require(len(local) > 0, "empty probe actor batch")
        batches.append(pool[local])
        offset += len(local)
    return batches


def target_batches(target: WitnessTargets, size: int, count: int,
                   generator: torch.Generator) -> list[WitnessTargets]:
    require(len(target.state_index) > 0, "empty auxiliary target set")
    return [target.take(torch.randint(len(target.state_index), (size,), generator=generator))
            for _ in range(count)]


def elapsed(start_wall: float, start_process: float) -> dict[str, float]:
    return {"wall_seconds": time.perf_counter() - start_wall,
            "process_seconds": time.process_time() - start_process}


def actor_loss_forward(model: IntentNeighborActor, paired: Any,
                       batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    started_wall = time.perf_counter(); started_process = time.process_time()
    gate, proposal = model(*(tensor[batch] for tensor in paired.new))
    loss, _ = hierarchical_bc_loss(gate, proposal,
                                   paired.original.gate[batch], paired.original.proposal[batch])
    return loss, elapsed(started_wall, started_process)


def aux_loss_forward(model: IntentNeighborActor, paired: Any,
                     target: WitnessTargets) -> tuple[torch.Tensor, dict[str, float]]:
    started_wall = time.perf_counter(); started_process = time.process_time()
    pair, scene = model.queried_scores(*(tensor[target.state_index] for tensor in paired.new),
                                       target.proposal, target.pad)
    loss = auxiliary_loss(pair, scene, target)
    return loss, elapsed(started_wall, started_process)


def timed_phase(model: IntentNeighborActor, paired: Any,
                warmup_actor_batches: list[torch.Tensor], warmup_aux_batches: list[WitnessTargets],
                timed_actor_batches: list[torch.Tensor], timed_aux_batches: list[WitnessTargets],
                warmups: int, updates: int,
                wall_cap: float, phase_start: float) -> dict[str, Any]:
    require(len(warmup_actor_batches) == warmups and len(warmup_aux_batches) == warmups
            and len(timed_actor_batches) == updates and len(timed_aux_batches) == updates,
            "probe batch count differs")
    timings = []
    peak = rss_bytes()
    backward_calls = 0
    all_batches = list(zip(warmup_actor_batches, warmup_aux_batches, [False] * warmups))
    all_batches += list(zip(timed_actor_batches, timed_aux_batches, [True] * updates))
    for step, (actor_batch, aux_batch, timed) in enumerate(all_batches):
        require(time.perf_counter() - phase_start <= wall_cap,
                "probe wall budget exhausted")
        model.train()
        model.zero_grad(set_to_none=True)
        whole_wall = time.perf_counter(); whole_process = time.process_time()
        actor_loss, actor_time = actor_loss_forward(model, paired, actor_batch)
        aux_loss, aux_time = aux_loss_forward(model, paired, aux_batch)
        backward_start_wall = time.perf_counter(); backward_start_process = time.process_time()
        (actor_loss + aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        backward_time = elapsed(backward_start_wall, backward_start_process)
        backward_calls += 1
        current_rss = rss_bytes()
        if current_rss is not None:
            peak = max(peak or current_rss, current_rss)
        record = {
            "round": step,
            "timed": timed,
            "actor_forward": actor_time,
            "aux_forward": aux_time,
            "backward_clip": backward_time,
            "wall_seconds": time.perf_counter() - whole_wall,
            "process_seconds": time.process_time() - whole_process,
        }
        if timed:
            timings.append(record)
    return {"warmups": warmups, "timed_backward_passes": updates, "backward_calls": backward_calls,
            "timings": timings, "peak_rss_bytes": peak}


@torch.no_grad()
def timed_actor_mark(model: IntentNeighborActor, paired: Any, indices: torch.Tensor) -> dict[str, Any]:
    model.eval()
    started_wall = time.perf_counter(); started_process = time.process_time()
    _ = measure(model, paired, indices)
    timing = elapsed(started_wall, started_process)
    return {"rows": int(len(indices)), "batches": (len(indices) + 31) // 32, **timing}


@torch.no_grad()
def timed_pair_mark(model: IntentNeighborActor, paired: Any,
                    target: WitnessTargets) -> dict[str, Any]:
    model.eval()
    started_wall = time.perf_counter(); started_process = time.process_time()
    _ = measure_pairs(model, paired, target)
    timing = elapsed(started_wall, started_process)
    return {"rows": int(len(target.state_index)), "batches": (len(target.state_index) + 63) // 64,
            **target_support(target), **timing}


def mark_measurements(model: IntentNeighborActor, paired: Any, train: torch.Tensor,
                      held: torch.Tensor, old_train: WitnessTargets,
                      old_held: WitnessTargets, expanded_train: WitnessTargets,
                      expanded_held: WitnessTargets) -> dict[str, Any]:
    return {
        "actor_train": timed_actor_mark(model, paired, train),
        "actor_held": timed_actor_mark(model, paired, held),
        "old_train_pairs": timed_pair_mark(model, paired, old_train),
        "old_held_pairs": timed_pair_mark(model, paired, old_held),
        "expanded_train_pairs": timed_pair_mark(model, paired, expanded_train),
        "expanded_held_pairs": timed_pair_mark(model, paired, expanded_held),
        "reference_held_pairs": timed_pair_mark(model, paired, old_held),
    }


def worker_probe(payload: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    worker_started = time.perf_counter()
    info = runtime_info()
    paired = payload["paired"]
    old_target = payload["old_target"]
    expanded_target = payload["expanded_target"]
    held_seed = int(payload["held_seed"])
    warmups = int(payload["warmups"]); updates = int(payload["updates"])
    actor_batch_size = int(payload["actor_batch_size"]); aux_batch_size = int(payload["aux_batch_size"])
    wall_cap = float(payload["wall_cap"])
    torch.manual_seed(MODEL_SEED)
    model = IntentNeighborActor(gate_mode=DEFAULT_GATE_MODE)
    initial_hash = state_sha(model)
    train, held, old_train_index, old_held_index = split_indices(paired, old_target, held_seed)
    _, _, expanded_train_index, expanded_held_index = split_indices(paired, expanded_target, held_seed)
    old_train = old_target.take(old_train_index); old_held = old_target.take(old_held_index)
    expanded_train = expanded_target.take(expanded_train_index); expanded_held = expanded_target.take(expanded_held_index)
    actor_generator = torch.Generator().manual_seed(MODEL_SEED)
    warmup_actor_batches = actor_batches(train, actor_batch_size, warmups, actor_generator)
    timed_actor_generator = torch.Generator().manual_seed(MODEL_SEED)
    timed_actor_batches = actor_batches(train, actor_batch_size, updates, timed_actor_generator)
    old_warmup_generator = torch.Generator().manual_seed(AUX_SEED)
    old_timed_generator = torch.Generator().manual_seed(AUX_SEED)
    expanded_warmup_generator = torch.Generator().manual_seed(AUX_SEED)
    expanded_timed_generator = torch.Generator().manual_seed(AUX_SEED)
    old_warmup_batches = target_batches(old_train, aux_batch_size, warmups, old_warmup_generator)
    old_timed_batches = target_batches(old_train, aux_batch_size, updates, old_timed_generator)
    expanded_warmup_batches = target_batches(expanded_train, aux_batch_size, warmups,
                                             expanded_warmup_generator)
    expanded_timed_batches = target_batches(expanded_train, aux_batch_size, updates,
                                            expanded_timed_generator)
    train_set = set(int(value) for value in train.tolist())
    held_set = set(int(value) for value in held.tolist())
    require(train_set.isdisjoint(held_set), "actor train/held indices overlap")
    require(all(set(int(value) for value in batch.tolist()) <= train_set
                for batch in warmup_actor_batches + timed_actor_batches),
            "actor batch contains held or out-of-train index")
    expected_aux = torch.randint(len(old_train.state_index), (aux_batch_size,),
                                 generator=torch.Generator().manual_seed(AUX_SEED))
    require(torch.equal(old_timed_batches[0].state_index, old_train.state_index[expected_aux]),
            "auxiliary batch does not match torch.randint replacement sequence")
    phase_start = time.perf_counter()
    old_phase = timed_phase(model, paired, warmup_actor_batches, old_warmup_batches,
                            timed_actor_batches, old_timed_batches, warmups, updates,
                            wall_cap, phase_start)
    expanded_phase = timed_phase(model, paired, warmup_actor_batches, expanded_warmup_batches,
                                 timed_actor_batches, expanded_timed_batches, warmups, updates,
                                 wall_cap, phase_start)
    require(time.perf_counter() - worker_started <= wall_cap,
            "probe wall budget exhausted before mark measurements")
    mark = mark_measurements(model, paired, train, held, old_train, old_held,
                             expanded_train, expanded_held)
    require(time.perf_counter() - worker_started <= wall_cap,
            "probe wall budget exhausted after mark measurements")
    final_hash = state_sha(model)
    require(initial_hash == final_hash, "probe model weights changed")
    info_after = runtime_info()
    return {
        "status": "COMPLETE",
        "runtime": {"before": info, "after": info_after},
        "held_seed": held_seed,
        "supports": {
            "actor_train_rows": int(len(train)), "actor_held_rows": int(len(held)),
            "old_train": target_support(old_train), "old_held": target_support(old_held),
            "expanded_train": target_support(expanded_train),
            "expanded_held": target_support(expanded_held),
        },
        "model_state": {"initial_sha256": initial_hash, "final_sha256": final_hash,
                        "unchanged": True},
        "old_phase": old_phase,
        "expanded_phase": expanded_phase,
        "mark_measurements": mark,
        "backward_calls": int(old_phase["backward_calls"] + expanded_phase["backward_calls"]),
        "optimizer_updates": 0,
        "elapsed_seconds": time.perf_counter() - worker_started,
    }


def synthetic_payload(held_seed: int) -> tuple[Any, WitnessTargets, WitnessTargets, dict[str, Any]]:
    paired, witness = smoke_inputs(supplemental=False)
    for index, context in enumerate(paired.original.contexts):
        context["seed"] = int(held_seed) if index % 3 == 0 else int(held_seed) + 1
    witness.seeds = torch.tensor([int(context["seed"]) for context in paired.original.contexts])
    return paired, witness, witness, {
        "kind": "synthetic",
        "source": "terminal_marl.scripts.diagnose_p3_intent_neighbor.smoke_inputs",
        "source_sha256": sha256_file(PROJECT_ROOT / "terminal_marl/scripts/diagnose_p3_intent_neighbor.py"),
    }


def build_real_payload(args: argparse.Namespace) -> tuple[Any, WitnessTargets, WitnessTargets, dict[str, Any]]:
    require(len(args.train_directories) == 3 and len(args.additional_train_directories) == 3
            and len(args.supplemental_train_directories) == 3,
            "exactly three directories required for each nine-cycle group")
    paired = load_nine_cycle_train(args.train_directories, args.additional_train_directories,
                                   args.supplemental_train_directories)
    old = load_witness_targets(Path(args.old_witness_directory), paired)
    supplemental = load_witness_targets(Path(args.supplemental_witness_directory), paired)
    prefix = load_witness_targets(Path(args.prefix_witness_directory), paired)
    reference = combine_witness_targets((old, supplemental))
    expanded = combine_witness_targets((reference, prefix))
    require(len(reference.state_index) == 473,
            f"old+supplemental auxiliary target count differs: {len(reference.state_index)}")
    require(len(expanded.state_index) == 1270,
            f"expanded auxiliary target count differs: {len(expanded.state_index)}")
    source_paths = []
    for directories in (args.train_directories, args.additional_train_directories,
                        args.supplemental_train_directories):
        for value in directories:
            for name in ("protocol.json", "result.json"):
                source_paths.append(Path(value) / name)
    source_paths += [Path(args.old_witness_directory) / "protocol.json",
                     Path(args.old_witness_directory) / "result.json",
                     Path(args.old_witness_directory) / "witnesses.jsonl",
                     Path(args.supplemental_witness_directory) / "protocol.json",
                     Path(args.supplemental_witness_directory) / "result.json",
                     Path(args.supplemental_witness_directory) / "witnesses.jsonl",
                     Path(args.prefix_witness_directory) / "protocol.json",
                     Path(args.prefix_witness_directory) / "result.json",
                     Path(args.prefix_witness_directory) / "witnesses.jsonl"]
    return paired, reference, expanded, {
        "kind": "nine_cycle_train",
        "source_sha256": input_sha_manifest(source_paths),
        "train_groups": {
            "original": [str(Path(value)) for value in args.train_directories],
            "additional": [str(Path(value)) for value in args.additional_train_directories],
            "supplemental": [str(Path(value)) for value in args.supplemental_train_directories],
        },
        "target_rows": {"old_reference": len(reference.state_index), "expanded": len(expanded.state_index)},
    }


def protocol_payload(args: argparse.Namespace, source: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    code_paths = [
        PROJECT_ROOT / "terminal_marl/scripts/diagnose_p3_intent_neighbor.py",
        PROJECT_ROOT / "terminal_marl/training/p3_v26_train_dataset.py",
        PROJECT_ROOT / "terminal_marl/training/p3_neighbor_witness_dataset.py",
        PROJECT_ROOT / "terminal_marl/algo/intent_neighbor_actor.py",
    ]
    return {
        "protocol": PROTOCOL, "smoke": smoke, "held_seed": int(args.held_seed),
        "model_seed": MODEL_SEED, "aux_batch_seed": AUX_SEED,
        "gate_mode": DEFAULT_GATE_MODE, "architecture": IntentNeighborActor.architecture,
        "observation": "existing V2.6 159+N50",
        "warmups": int(args.warmups), "timed_backward_passes_per_group": int(args.updates),
        "actor_batch_size": int(args.actor_batch_size), "auxiliary_batch_size": int(args.aux_batch_size),
        "wall_cap_seconds": float(args.wall_cap), "workers": int(args.workers),
        "groups": {"old_reference": "old+supplemental", "expanded": "old+supplemental+prefix"},
        "input_source": source,
        "probe_script_sha256": sha256_file(Path(__file__).resolve()),
        "loader_and_model_code_sha256": input_sha_manifest(code_paths),
        "updates": 0, "optimizer_updates": 0,
        "backward_calls_expected_per_worker": int(2 * (args.warmups + args.updates)),
        "backward_calls_expected_total": int(args.workers * 2 * (args.warmups + args.updates)),
        "optimizer_constructed": False, "checkpoint_written": False,
        "environment_resets": 0, "environment_steps": 0, "teacher_queries": 0,
        "planner_calls": 0, "q_accessed": False,
        "runtime_snapshot_calls": 0,
        "loader_snapshot_reads": "official load_nine_cycle_train/load_resource_train reads archived input snapshots for 54D resource reconstruction; outside timed probe",
        "output_must_be_new": True,
        "recommended_real_output_root": "/home/user/training_outputs/terminal/p3_neighbor_runtime_probe_20261010",
        "metrics": "timing_and_support_counts_only; no loss/accuracy/action metrics",
    }


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if output.exists():
        raise ProbeError(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    smoke = bool(args.smoke)
    run_started = time.perf_counter()
    try:
        if smoke:
            paired, old_target, expanded_target, source = synthetic_payload(int(args.held_seed))
        else:
            paired, old_target, expanded_target, source = build_real_payload(args)
        protocol = protocol_payload(args, source, smoke=smoke)
        write_json(output / "protocol.json", protocol)
        payload = {"paired": paired, "old_target": old_target, "expanded_target": expanded_target,
                   "held_seed": int(args.held_seed), "warmups": int(args.warmups),
                   "updates": int(args.updates), "actor_batch_size": int(args.actor_batch_size),
                   "aux_batch_size": int(args.aux_batch_size), "wall_cap": float(args.wall_cap)}
        started = time.perf_counter()
        ctx = mp.get_context("spawn")
        if int(args.workers) == 1:
            worker_results = [worker_probe(payload)]
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=ctx) as executor:
                worker_results = list(executor.map(worker_probe, [payload] * int(args.workers)))
        require(all(result.get("status") == "COMPLETE" for result in worker_results),
                "worker probe did not complete")
        result = {
            "status": "SMOKE_OK" if smoke else "COMPLETE",
            "protocol": PROTOCOL, "workers": int(args.workers),
            "worker_results": worker_results,
            "updates": 0, "optimizer_updates": 0,
            "backward_calls": sum(int(item["backward_calls"]) for item in worker_results),
            "wall_seconds": time.perf_counter() - started,
            "environment_resets": 0, "environment_steps": 0,
            "teacher_queries": 0, "planner_calls": 0, "q_accessed": False,
        }
        write_json(output / "result.json", result)
        return 0
    except Exception as error:
        write_json(output / "result.json", {
            "status": "INCOMPLETE", "protocol": PROTOCOL, "error": repr(error),
            "updates": 0, "optimizer_updates": 0, "environment_resets": 0,
            "environment_steps": 0, "teacher_queries": 0, "planner_calls": 0,
            "q_accessed": False, "wall_seconds": time.perf_counter() - run_started,
        })
        return 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1)
    result.add_argument("--held-seed", type=int, default=HELD_SEED)
    result.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    result.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    result.add_argument("--actor-batch-size", type=int, default=DEFAULT_ACTOR_BATCH)
    result.add_argument("--aux-batch-size", type=int, default=DEFAULT_AUX_BATCH)
    result.add_argument("--wall-cap", type=float, default=DEFAULT_WALL_CAP)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--train-directories", nargs=3, default=[])
    result.add_argument("--additional-train-directories", nargs=3, default=[])
    result.add_argument("--supplemental-train-directories", nargs=3, default=[])
    result.add_argument("--old-witness-directory")
    result.add_argument("--supplemental-witness-directory")
    result.add_argument("--prefix-witness-directory")
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.smoke:
        required = (args.old_witness_directory, args.supplemental_witness_directory,
                    args.prefix_witness_directory)
        require(all(required), "real probe requires all witness directories")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
