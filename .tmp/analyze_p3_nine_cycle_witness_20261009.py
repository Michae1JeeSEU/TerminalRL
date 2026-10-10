"""Read-only analysis of old and supplemental P3 neighbor witness artifacts.

The script uses the official nine-cycle and witness loaders, then checks the
serialized witness rows and held-cycle split identities.  It does not collect
witnesses, call ``witness_state``, reset or step an environment, query a
teacher, or run a learner.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ROOT = (
    Path(r"D:\7_codespace\training_outputs\terminal")
    if sys.platform.startswith("win")
    else Path("/home/user/training_outputs/terminal")
)
DEFAULT_OLD_WITNESS = PROJECT_ROOT / ".tmp" / "p3_neighbor_witness_train_20261009"
SEEDS = (893511, 893512, 893513, 893601, 893602, 893603, 893604, 893605, 893606)
OLD_SEEDS = SEEDS[:6]
SUPPLEMENTAL_SEEDS = SEEDS[6:]


class CheckFailure(RuntimeError):
    """A direct serialized-artifact or loader check mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def train_directories(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    root = root.expanduser()
    old_root = root / "p3_v25_neutral_course_v2_20261008"
    independent_root = root / "p3_v26_resources_20261009"
    supplemental_root = root / "p3_v26_supplemental_20261009"
    original = [old_root / name for name in ("train_flow6_mixed", "train_dense24_fixed", "train_dense24_mixed")]
    independent = [independent_root / f"train_{index:02d}" for index in (1, 2, 3)]
    supplemental = [supplemental_root / f"train_{index:02d}" for index in (1, 2, 3)]
    return original, independent, supplemental


def witness_file_summary(
    directory: Path,
    paired,
    targets,
    *,
    name: str,
    expected_seeds: tuple[int, ...],
    expected_records: int,
    expected_selected_ids: list[str] | None = None,
) -> dict:
    protocol_path = directory / "protocol.json"
    result_path = directory / "result.json"
    witnesses_path = directory / "witnesses.jsonl"
    require(protocol_path.is_file() and result_path.is_file() and witnesses_path.is_file(),
            f"missing {name} witness artifact under {directory}")
    protocol = read_json(protocol_path)
    result = read_json(result_path)
    rows = read_jsonl(witnesses_path)
    row_ids = [row.get("decision_id") for row in rows]
    selected_ids = list(protocol.get("selected_ids", []))
    require(row_ids and len(row_ids) == len(set(row_ids)), f"{name} witness decision IDs are not unique")
    require(len(rows) == expected_records, f"{name} record count {len(rows)} != expected {expected_records}")
    require(selected_ids == row_ids, f"{name} protocol selected IDs differ from witness rows")
    if expected_selected_ids is not None:
        require(selected_ids == expected_selected_ids,
                f"{name} protocol selected IDs differ from existing loader-check selection")
    require(result.get("status") == "COMPLETE" and result.get("error") is None,
            f"{name} result status/error differs: {result.get('status')}/{result.get('error')}")
    require(int(result.get("states", -1)) == len(rows),
            f"{name} result states {result.get('states')} != record count {len(rows)}")
    require(protocol.get("smoke") is False, f"{name} witness is marked smoke")
    declared_seeds = {int(item["seed"]) for item in protocol.get("inputs", [])}
    require(declared_seeds == set(expected_seeds),
            f"{name} protocol input seeds differ")
    require(set(int(row.get("seed")) for row in rows) == set(expected_seeds),
            f"{name} witness row seeds differ")
    witness_sha = sha256_file(witnesses_path)
    require(result.get("witness_sha256") == witness_sha, f"{name} result witness SHA differs")

    target_indices = [int(value) for value in targets.state_index.detach().cpu().tolist()]
    target_ids = {paired.original.ids[index] for index in target_indices}
    query_by_seed = Counter(int(value) for value in targets.seeds.detach().cpu().tolist())
    record_by_seed = Counter(int(row["seed"]) for row in rows)
    plan_sum = 0
    destination_total = 0
    candidate_total = 0
    candidate_consistent = 0
    candidate_infeasible = 0
    pair_uid_checks = 0
    empty_pair_destinations = 0
    pair_approved_checks = 0
    destination_record_ids = set()
    query_delta_by_record = {}
    query_delta_sums = Counter()
    query_delta_nonempty = 0
    for row in rows:
        decision_id = row["decision_id"]
        plan_calls = int(row.get("plan_calls", -1))
        require(plan_calls >= 0, f"{name} negative plan_calls for {decision_id}")
        plan_sum += plan_calls
        delta = row.get("query_counter_delta", {})
        require(isinstance(delta, dict), f"{name} query_counter_delta is not an object for {decision_id}")
        query_delta_by_record[decision_id] = delta
        if delta:
            query_delta_nonempty += 1
        for key, value in delta.items():
            require(isinstance(value, (int, float)) and not isinstance(value, bool),
                    f"{name} query_counter_delta value is not numeric for {decision_id}:{key}")
            query_delta_sums[key] += value
        for candidate in row.get("candidates", []):
            candidate_total += 1
            destinations = candidate.get("destinations", [])
            destination_total += len(destinations)
            if destinations:
                destination_record_ids.add(decision_id)
                require(candidate.get("global_reason") is None,
                        f"{name} destination candidate has global_reason: {decision_id}")
            else:
                require(not bool(candidate.get("original_feasible")),
                        f"{name} empty destination candidate is feasible: {decision_id}")
                require(candidate.get("global_reason") and
                        candidate.get("global_reason") == candidate.get("original_reason"),
                        f"{name} empty destination rejection differs: {decision_id}")
            feasible = any(bool(destination.get("feasible")) for destination in destinations)
            require(feasible == bool(candidate.get("original_feasible")),
                    f"{name} destination feasibility differs from original: {decision_id}")
            if not feasible:
                candidate_infeasible += 1
                if destinations:
                    require(destinations[-1].get("reason") == candidate.get("original_reason"),
                            f"{name} final destination reason differs from original: {decision_id}")
            candidate_consistent += 1
            for destination in destinations:
                pairs = destination.get("pairs", [])
                traffic_uids = list(row.get("traffic_uids", []))
                if pairs:
                    require([pair.get("uid") for pair in pairs] == traffic_uids,
                            f"{name} pair UID order differs: {decision_id}")
                    pair_uid_checks += 1
                else:
                    empty_pair_destinations += 1
                    require(not (destination.get("plan_hash") is not None and traffic_uids),
                            f"{name} nonempty plan_hash has no pair rows: {decision_id}")
                if destination.get("feasible"):
                    require(all(pair.get("approved") is True for pair in pairs),
                            f"{name} feasible destination has unapproved pair: {decision_id}")
                    pair_approved_checks += len(pairs)

    require(plan_sum == int(result.get("plan_calls", -1)),
            f"{name} row plan_calls sum {plan_sum} != result {result.get('plan_calls')}")
    require(plan_sum == len(target_indices),
            f"{name} row plan_calls sum {plan_sum} != target queries {len(target_indices)}")
    require(target_ids == {row_id for row_id in destination_record_ids},
            f"{name} target IDs differ from records with destinations")
    require(set(record_by_seed) == set(expected_seeds), f"{name} record seed set differs")
    require(set(query_by_seed) <= set(expected_seeds), f"{name} target seed set differs")

    if name == "old":
        expected_protocol = "p3_train_neighbor_witness_20261009_v1"
        expected_budget = dict(max_states=144, max_plan_calls=1728, wall_seconds=3600)
    else:
        expected_protocol = "p3_supplemental_train_neighbor_witness_20261009_v1"
        expected_budget = dict(max_states=72, max_plan_calls=864, wall_seconds=1800)
        require(protocol.get("source_group") == "supplemental", "supplemental source_group differs")
    require(protocol.get("protocol") == expected_protocol, f"{name} protocol differs")
    for key, value in expected_budget.items():
        require(protocol.get(key) == value, f"{name} protocol {key} differs: {protocol.get(key)} != {value}")
    elapsed_seconds = result.get("elapsed_seconds")
    require(isinstance(elapsed_seconds, (int, float)) and not isinstance(elapsed_seconds, bool)
            and 0 <= elapsed_seconds <= expected_budget["wall_seconds"],
            f"{name} elapsed_seconds exceeds protocol wall budget: {elapsed_seconds}")
    require(len(rows) <= expected_budget["max_states"],
            f"{name} record count exceeds protocol max_states: {len(rows)}")
    require(plan_sum <= expected_budget["max_plan_calls"],
            f"{name} plan_calls exceeds protocol max_plan_calls: {plan_sum}")
    require(protocol.get("resets") == 0 and protocol.get("steps") == 0 and
            protocol.get("Q_accessed") is False and protocol.get("original_labels_unchanged") is True,
            f"{name} protocol boundary fields differ")

    return dict(
        protocol_file=repo_relative(protocol_path),
        protocol_file_sha256=sha256_file(protocol_path),
        result_file=repo_relative(result_path),
        result_file_sha256=sha256_file(result_path),
        witnesses_file=repo_relative(witnesses_path),
        witnesses_file_sha256=witness_sha,
        protocol=protocol.get("protocol"),
        status=result.get("status"),
        error=result.get("error"),
        elapsed_seconds=elapsed_seconds,
        budget={key: protocol.get(key) for key in ("max_states", "max_plan_calls", "wall_seconds")},
        budget_within_limits=True,
        records=len(rows),
        record_unique_decision_ids=len(set(row_ids)),
        selected_ids=len(selected_ids),
        target_states=len(target_ids),
        queries=len(target_indices),
        record_queries_by_seed=dict(sorted(record_by_seed.items())),
        target_queries_by_seed=dict(sorted(query_by_seed.items())),
        plan_calls_sum=plan_sum,
        all_destinations=destination_total,
        candidate_total=candidate_total,
        candidate_consistency_checks=candidate_consistent,
        candidate_infeasible=candidate_infeasible,
        destination_record_count=len(destination_record_ids),
        pair_uid_order_checks=pair_uid_checks,
        empty_pair_destinations=empty_pair_destinations,
        feasible_pair_approved_checks=pair_approved_checks,
        known_pairs=int(targets.known.sum()),
        blocked_pairs=int((targets.blocked.bool() & targets.known).sum()),
        known_scene=int(targets.scene_known.sum()),
        blocked_scene=int((targets.scene_blocked.bool() & targets.scene_known).sum()),
        query_counter_delta=dict(
            by_record=query_delta_by_record,
            field_sums=dict(sorted(query_delta_sums.items())),
            nonempty_records=query_delta_nonempty,
        ),
    )


def split_checks(paired, witness, seeds: tuple[int, ...], split_function) -> list[dict]:
    reports = []
    actor_seed_by_index = [int(context["seed"]) for context in paired.original.contexts]
    witness_seed_by_index = [int(value) for value in witness.seeds.detach().cpu().tolist()]
    for seed in seeds:
        train, held, witness_train, witness_held = split_function(paired, witness, seed)
        actor_train_seeds = [actor_seed_by_index[index] for index in train.tolist()]
        actor_held_seeds = [actor_seed_by_index[index] for index in held.tolist()]
        aux_train_seeds = [witness_seed_by_index[index] for index in witness_train.tolist()]
        aux_held_seeds = [witness_seed_by_index[index] for index in witness_held.tolist()]
        require(all(value != seed for value in actor_train_seeds) and all(value == seed for value in actor_held_seeds),
                f"actor split leakage for seed {seed}")
        require(all(value != seed for value in aux_train_seeds) and all(value == seed for value in aux_held_seeds),
                f"auxiliary split leakage for seed {seed}")
        require(not set(train.tolist()) & set(held.tolist()) and
                not set(witness_train.tolist()) & set(witness_held.tolist()),
                f"split overlap for seed {seed}")
        reports.append(dict(
            held_seed=seed,
            actor_train_rows=len(train),
            actor_held_rows=len(held),
            auxiliary_train_queries=len(witness_train),
            auxiliary_held_queries=len(witness_held),
            actor_train_has_held_seed=False,
            auxiliary_train_has_held_seed=False,
        ))
    return reports


def run_check(root: Path, old_witness_directory: Path, supplemental_witness_directory: Path,
              loader_check_path: Path) -> dict:
    from terminal_marl.scripts.diagnose_p3_intent_neighbor import split
    from terminal_marl.training.p3_neighbor_witness_dataset import combine_witness_targets, load_witness_targets
    from terminal_marl.training.p3_v26_train_dataset import load_nine_cycle_train

    original, independent, supplemental = train_directories(root)
    paired = load_nine_cycle_train(original, independent, supplemental)
    require(len(paired.original.ids) == 3149 and len(set(paired.original.ids)) == 3149,
            "official nine-cycle loader did not return 3149 unique IDs")
    require(int(paired.original.gate.sum()) == 577 and
            int((paired.original.gate == 0).sum()) == 2572,
            "official nine-cycle labels differ from 577 RELEASE/2572 WAIT")
    require(tuple(paired.new[0].shape) == (3149, 159) and
            tuple(paired.new[1].shape) == (3149, 20, 50) and
            tuple(paired.new[2].shape) == (3149, 20),
            "official nine-cycle tensor shapes differ")

    old_witness_directory = old_witness_directory.expanduser()
    supplemental_witness_directory = supplemental_witness_directory.expanduser()
    loader_check_path = loader_check_path.expanduser()
    require(loader_check_path.is_file(), f"missing existing PASS loader-check: {loader_check_path}")
    loader_check = read_json(loader_check_path)
    require(loader_check.get("status") == "PASS", "existing loader-check is not PASS")
    loader_selection = loader_check.get("supplemental_selection", {})
    expected_supplemental_ids = list(loader_selection.get("selected_ids", []))
    require(loader_selection.get("actual_states") == 65 and len(expected_supplemental_ids) == 65,
            "existing loader-check supplemental selection is not 65 states")
    old_targets = load_witness_targets(old_witness_directory, paired)
    supplemental_targets = load_witness_targets(supplemental_witness_directory, paired)
    combined = combine_witness_targets((old_targets, supplemental_targets))
    old_report = witness_file_summary(old_witness_directory, paired, old_targets, name="old",
                                      expected_seeds=OLD_SEEDS, expected_records=127)
    supplemental_report = witness_file_summary(supplemental_witness_directory, paired, supplemental_targets,
                                               name="supplemental", expected_seeds=SUPPLEMENTAL_SEEDS,
                                               expected_records=65,
                                               expected_selected_ids=expected_supplemental_ids)
    combined_query_by_seed = Counter(int(value) for value in combined.seeds.detach().cpu().tolist())
    split_report = split_checks(paired, combined, SEEDS, split)

    source_path = Path(__file__).resolve()
    return dict(
        status="PASS",
        source_file=repo_relative(source_path),
        source_file_sha256=sha256_file(source_path),
        root=str(root),
        loader_check=dict(
            path=repo_relative(loader_check_path),
            sha256=sha256_file(loader_check_path),
            status=loader_check.get("status"),
            supplemental_selected_states=loader_selection.get("actual_states"),
            supplemental_selected_ids_match=True,
        ),
        execution_scope="official read-only loaders and serialized witness inspection; no collect/witness_state",
        no_collect=True,
        no_witness_state=True,
        no_environment_reset_step=True,
        no_teacher_query=True,
        no_optimizer=True,
        paired=dict(
            rows=3149,
            unique_decision_ids=3149,
            release=577,
            wait=2572,
            labels_preserved_by_official_loader=True,
            shapes=dict(ego=(3149, 159), traffic=(3149, 20, 50), mask=(3149, 20)),
        ),
        old=old_report,
        supplemental=supplemental_report,
        combined=dict(
            queries=len(combined.state_index),
            target_states=len(set(int(value) for value in combined.state_index.detach().cpu().tolist())),
            queries_by_seed=dict(sorted(combined_query_by_seed.items())),
            known_pairs=int(combined.known.sum()),
            blocked_pairs=int((combined.blocked.bool() & combined.known).sum()),
            known_scene=int(combined.scene_known.sum()),
            blocked_scene=int((combined.scene_blocked.bool() & combined.scene_known).sum()),
            split=split_report,
        ),
    )


def write_once(path: Path, report: dict) -> None:
    path = path.expanduser()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--witness-directory", type=Path, default=DEFAULT_OLD_WITNESS)
    parser.add_argument("--supplemental-witness-directory", type=Path, default=None)
    parser.add_argument("--loader-check", type=Path,
                        default=PROJECT_ROOT / ".tmp" / "p3_nine_cycle_loader_check_20261009.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    supplemental_directory = args.supplemental_witness_directory or (
        args.root / "p3_neighbor_supplemental_witness_20261009" / "three_cycle"
    )
    if args.output.expanduser().exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    try:
        report = run_check(args.root, args.witness_directory, supplemental_directory, args.loader_check)
        write_once(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "source_file_sha256": report["source_file_sha256"]}, ensure_ascii=False))
        return 0
    except Exception as error:
        failure = dict(
            status="FAIL",
            source_file=repo_relative(Path(__file__).resolve()),
            source_file_sha256=sha256_file(Path(__file__).resolve()),
            error=repr(error),
            traceback_tail=traceback.format_exc().splitlines()[-10:],
            execution_scope="official read-only loaders and serialized witness inspection; no collect/witness_state",
            no_collect=True,
            no_witness_state=True,
            no_environment_reset_step=True,
            no_teacher_query=True,
            no_optimizer=True,
        )
        try:
            write_once(args.output, failure)
        except FileExistsError:
            pass
        print(json.dumps({"status": "FAIL", "output": str(args.output), "error": repr(error)}, ensure_ascii=False),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
