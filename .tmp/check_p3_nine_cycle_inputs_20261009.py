"""Read-only validation of the nine P3 train-cycle inputs.

This checker intentionally calls only the official data loaders and the
offline supplemental selector.  It does not collect, query a teacher, reset
or step an environment, create witness states, or train an optimizer.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LOCAL_ROOT = (
    Path(r"D:\7_codespace\training_outputs\terminal")
    if sys.platform.startswith("win")
    else Path("/home/user/training_outputs/terminal")
)
DEFAULT_WITNESS_DIRECTORY = PROJECT_ROOT / ".tmp" / "p3_neighbor_witness_train_20261009"

SOURCE_FILES = (
    "terminal_marl/scripts/diagnose_p3_intent_neighbor.py",
    "terminal_marl/training/p3_neighbor_witness_data.py",
    "terminal_marl/training/p3_neighbor_witness_dataset.py",
    "terminal_marl/training/p3_v26_train_dataset.py",
    "terminal_marl/training/p3_v26_resource_inputs.py",
    "terminal_marl/algo/intent_neighbor_actor.py",
)

SEED_ORDER = (893511, 893512, 893513, 893601, 893602, 893603, 893604, 893605, 893606)
EXPECTED_COUNTS = {
    893511: dict(group="original", scenario="flow6_mixed", actor=30, release=23, wait=7, normal=5, emergency=1),
    893512: dict(group="original", scenario="dense24_fixed", actor=721, release=94, wait=627, normal=20, emergency=4),
    893513: dict(group="original", scenario="dense24_mixed", actor=547, release=87, wait=460, normal=14, emergency=10),
    893601: dict(group="independent", scenario="flow6_mixed", actor=18, release=18, wait=0, normal=6, emergency=0),
    893602: dict(group="independent", scenario="dense24_fixed", actor=571, release=75, wait=496, normal=18, emergency=6),
    893603: dict(group="independent", scenario="dense24_mixed", actor=371, release=75, wait=296, normal=19, emergency=5),
    893604: dict(group="supplemental", scenario="medium12_mixed", actor=57, release=37, wait=20, normal=12, emergency=0),
    893605: dict(group="supplemental", scenario="dense24_fixed", actor=542, release=89, wait=453, normal=18, emergency=6),
    893606: dict(group="supplemental", scenario="dense24_mixed", actor=292, release=79, wait=213, normal=19, emergency=5),
}


class CheckFailure(RuntimeError):
    """A direct input/check mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def float32_array(value, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if shape is not None:
        require(array.shape == shape, f"expected float32 shape {shape}, got {array.shape}")
    return array


def traffic_array(row: dict) -> tuple[np.ndarray, np.ndarray]:
    traffic = float32_array(row.get("traffic", []))
    if traffic.size == 0:
        traffic = traffic.reshape(0, 50)
    elif traffic.ndim == 1:
        require(traffic.size % 50 == 0, "traffic vector is not divisible by N50")
        traffic = traffic.reshape(-1, 50)
    require(traffic.ndim == 2 and traffic.shape[1] == 50, "traffic is not N x 50")
    mask = np.asarray(row.get("traffic_mask", []), dtype=np.bool_)
    require(mask.shape == (len(traffic),), "traffic mask length differs from traffic")
    require(np.isfinite(traffic).all() and mask.all(), "non-finite or non-real traffic row")
    return traffic, mask


def build_specs(root: Path) -> list[dict]:
    specs = []
    for group, relative, seeds, scenarios, row_file in (
        (
            "original",
            "p3_v25_neutral_course_v2_20261008",
            (893511, 893512, 893513),
            ("flow6_mixed", "dense24_fixed", "dense24_mixed"),
            "context_observations.jsonl",
        ),
        (
            "independent",
            "p3_v26_resources_20261009",
            (893601, 893602, 893603),
            ("flow6_mixed", "dense24_fixed", "dense24_mixed"),
            "resource_context_observations.jsonl",
        ),
        (
            "supplemental",
            "p3_v26_supplemental_20261009",
            (893604, 893605, 893606),
            ("medium12_mixed", "dense24_fixed", "dense24_mixed"),
            "resource_context_observations.jsonl",
        ),
    ):
        for seed, scenario in zip(seeds, scenarios):
            specs.append(
                dict(
                    group=group,
                    seed=seed,
                    scenario=scenario,
                    directory=root / relative / (
                        f"train_{seed - 893600:02d}" if group != "original" else f"train_{scenario}"
                    ),
                    row_file=row_file,
                )
            )
    # The old directory names are the scenario names; all newer directories
    # use train_01/train_02/train_03.
    specs[0]["directory"] = root / "p3_v25_neutral_course_v2_20261008" / "train_flow6_mixed"
    specs[1]["directory"] = root / "p3_v25_neutral_course_v2_20261008" / "train_dense24_fixed"
    specs[2]["directory"] = root / "p3_v25_neutral_course_v2_20261008" / "train_dense24_mixed"
    for index in range(3, 6):
        specs[index]["directory"] = root / "p3_v26_resources_20261009" / f"train_{index - 2:02d}"
    for index in range(6, 9):
        specs[index]["directory"] = root / "p3_v26_supplemental_20261009" / f"train_{index - 5:02d}"
    return specs


def result_outcome_counts(directory: Path) -> tuple[dict, dict]:
    result = read_json(directory / "result.json")
    rollout = result.get("rollout", result)
    outcomes = dict(rollout.get("outcome_counts", {}))
    require(set(outcomes) <= {"completed", "emergency_landed"},
            f"unexpected outcome keys for {directory}: {sorted(outcomes)}")
    normal = int(outcomes.get("completed", 0))
    emergency = int(outcomes.get("emergency_landed", 0))
    require("requested" in rollout and normal + emergency == int(rollout["requested"]),
            f"outcome total differs from requested for {directory}: "
            f"{normal}+{emergency}!={rollout.get('requested')}")
    failed = sum(int(outcomes.get(key, 0)) for key in ("failed", "failed_energy", "failure"))
    diverted = sum(int(outcomes.get(key, 0)) for key in ("diverted", "upstream_divert", "upstream_diverted"))
    unresolved = int(rollout.get("unresolved", 0))
    fdu = dict(F=failed, D=diverted, U=unresolved)
    return dict(normal=normal, emergency=emergency, FDU=fdu, raw=outcomes), result


def load_input_rows(specs: list[dict]) -> tuple[dict[str, dict], list[dict], dict[int, dict]]:
    rows_by_id: dict[str, dict] = {}
    summaries = []
    rows_by_seed: dict[int, dict] = {}
    for spec in specs:
        directory = spec["directory"]
        path = directory / f"seed_{spec['seed']}" / spec["row_file"]
        require(path.is_file(), f"missing input file: {path}")
        rows = read_jsonl(path)
        expected = EXPECTED_COUNTS[spec["seed"]]
        require(len(rows) == expected["actor"], f"seed {spec['seed']} row count {len(rows)} != {expected['actor']}")
        local_ids = set()
        for row in rows:
            did = row.get("decision_id")
            require(isinstance(did, str) and did not in local_ids and did not in rows_by_id,
                    f"duplicate or missing decision ID: {did!r}")
            require(int(row["seed"]) == spec["seed"] and row.get("actor_mask") is True,
                    f"wrong seed or actor mask for {did}")
            float32_array(row.get("original_observation"), shape=(90,))
            traffic_array(row)
            local_ids.add(did)
            rows_by_id[did] = row
        outcomes, result = result_outcome_counts(directory)
        require(outcomes["normal"] == expected["normal"] and outcomes["emergency"] == expected["emergency"],
                f"seed {spec['seed']} N/E differs: {outcomes['normal']}/{outcomes['emergency']}")
        require(not any(outcomes["FDU"].values()), f"seed {spec['seed']} has nonzero F/D/U: {outcomes['FDU']}")
        rows_by_seed[spec["seed"]] = {"rows": rows, "spec": spec, "outcomes": outcomes}
        summary = dict(
            group=spec["group"],
            seed=spec["seed"],
            scenario=spec["scenario"],
            directory=str(directory),
            source_file=repo_relative(path),
            source_file_sha256=sha256_file(path),
            rows=len(rows),
            normal=outcomes["normal"],
            emergency=outcomes["emergency"],
            FDU=outcomes["FDU"],
            result_status=result.get("status"),
        )
        if spec["group"] != "original":
            context_path = directory / f"seed_{spec['seed']}" / "context_observations.jsonl"
            require(context_path.is_file(), f"missing archived context file: {context_path}")
            summary["context_file"] = repo_relative(context_path)
            summary["context_file_sha256"] = sha256_file(context_path)
        summaries.append(summary)
    require(len(rows_by_id) == 3149, f"input row total {len(rows_by_id)} != 3149")
    return rows_by_id, summaries, rows_by_seed


def check_resource_context_prefixes(rows_by_seed: dict[int, dict]) -> list[dict]:
    checks = []
    for seed, payload in rows_by_seed.items():
        spec = payload["spec"]
        if spec["group"] == "original":
            continue
        directory = spec["directory"] / f"seed_{seed}"
        resource_rows = {row["decision_id"]: row for row in payload["rows"]}
        context_rows = {row["decision_id"]: row for row in read_jsonl(directory / "context_observations.jsonl")}
        require(set(resource_rows) == set(context_rows), f"resource/context IDs differ for seed {seed}")
        for did, resource in resource_rows.items():
            context = context_rows[did]
            require(
                np.array_equal(float32_array(resource["ego"][:105]), float32_array(context["ego"], shape=(105,))),
                f"resource first105 differs from context for {did}",
            )
            resource_traffic, resource_mask = traffic_array(resource)
            context_traffic, context_mask = traffic_array(context)
            require(np.array_equal(resource_traffic, context_traffic) and np.array_equal(resource_mask, context_mask),
                    f"resource/context traffic differs for {did}")
        checks.append(dict(seed=seed, rows=len(resource_rows), first105_and_context_traffic_equal=True))
    require(len(checks) == 6, f"resource/context checks covered {len(checks)} seeds, expected 6")
    return checks


def compare_loaded_rows(paired, rows_by_id: dict[str, dict], rows_by_seed: dict[int, dict]) -> dict:
    original = paired.original
    ego, traffic, mask = (tensor.detach().cpu().numpy() for tensor in paired.new)
    require(tuple(original.x.shape) == (3149, 90), f"original tensor shape {tuple(original.x.shape)}")
    require(ego.shape == (3149, 159), f"new ego shape {ego.shape}")
    require(traffic.shape == (3149, 20, 50), f"new traffic shape {traffic.shape}")
    require(mask.shape == (3149, 20), f"new mask shape {mask.shape}")
    require(np.isfinite(original.x.detach().cpu().numpy()).all(), "non-finite original observations")
    require(np.isfinite(ego).all() and np.isfinite(traffic).all() and mask.dtype == np.bool_,
            "non-finite new tensors or non-bool mask")
    loaded_ids = list(original.ids)
    require(len(loaded_ids) == len(set(loaded_ids)) == 3149, "loaded decision IDs are not unique")
    loaded_index = {did: index for index, did in enumerate(loaded_ids)}
    require(set(loaded_ids) == set(rows_by_id), "loader decision IDs differ from raw inputs")

    release_by_seed = Counter()
    wait_by_seed = Counter()
    preserved_rows = 0
    mask_rows = 0
    for did, row in rows_by_id.items():
        index = loaded_index[did]
        seed = int(row["seed"])
        loaded_original = original.x[index].detach().cpu().numpy()
        raw_original = float32_array(row["original_observation"], shape=(90,))
        require(np.array_equal(loaded_original, raw_original), f"original observation changed for {did}")
        raw_gate = int(row["teacher_gate"])
        raw_proposal = -1 if row.get("teacher_proposal") is None else int(row["teacher_proposal"])
        require(int(original.gate[index]) == raw_gate and int(original.proposal[index]) == raw_proposal,
                f"teacher label changed for {did}")
        if raw_gate:
            release_by_seed[seed] += 1
        else:
            wait_by_seed[seed] += 1

        raw_traffic, raw_mask = traffic_array(row)
        loaded_mask = mask[index]
        require(len(raw_traffic) <= loaded_mask.shape[0], f"traffic width overflow for {did}")
        require(np.array_equal(traffic[index, : len(raw_traffic)], raw_traffic), f"traffic changed for {did}")
        require(np.array_equal(loaded_mask[: len(raw_mask)], raw_mask), f"traffic mask changed for {did}")
        require(not loaded_mask[len(raw_mask) :].any() and int(loaded_mask.sum()) == int(raw_mask.sum()),
                f"traffic mask dropped or added neighbors for {did}")

        raw_ego = float32_array(row["ego"])
        if rows_by_seed[seed]["spec"]["group"] == "original":
            require(raw_ego.shape == (105,) and np.array_equal(ego[index, :105], raw_ego),
                    f"old 105D prefix changed for {did}")
        else:
            require(raw_ego.shape == (159,) and np.array_equal(ego[index], raw_ego),
                    f"resource159D observation changed for {did}")
        preserved_rows += 1
        mask_rows += 1
    require(sum(release_by_seed.values()) == 577 and sum(wait_by_seed.values()) == 2572,
            f"loaded gate counts differ: {sum(release_by_seed.values())}/{sum(wait_by_seed.values())}")
    per_seed = []
    for seed in SEED_ORDER:
        expected = EXPECTED_COUNTS[seed]
        actual = dict(
            group=expected["group"],
            seed=seed,
            scenario=expected["scenario"],
            actor=expected["actor"],
            release=release_by_seed[seed],
            wait=wait_by_seed[seed],
            normal=rows_by_seed[seed]["outcomes"]["normal"],
            emergency=rows_by_seed[seed]["outcomes"]["emergency"],
            F=rows_by_seed[seed]["outcomes"]["FDU"]["F"],
            D=rows_by_seed[seed]["outcomes"]["FDU"]["D"],
            U=rows_by_seed[seed]["outcomes"]["FDU"]["U"],
        )
        require(actual == dict(
            group=expected["group"], seed=seed, scenario=expected["scenario"], actor=expected["actor"],
            release=expected["release"], wait=expected["wait"], normal=expected["normal"],
            emergency=expected["emergency"], F=0, D=0, U=0,
        ), f"per-seed counts differ for {seed}: {actual}")
        per_seed.append(actual)
    return dict(
        rows=preserved_rows,
        mask_rows=mask_rows,
        IDs_unique=True,
        labels_by_decision_id_preserved=True,
        old6_105D_prefix_traffic_mask_preserved=True,
        resource6_159D_loaded=True,
        resource6_first105_context_checked=True,
        finite_real_mask=True,
        shapes=dict(original=(3149, 90), new_ego=(3149, 159), new_traffic=(3149, 20, 50), new_mask=(3149, 20)),
        gate_totals=dict(release=577, wait=2572),
        per_seed=per_seed,
        comparison=dict(
            json_arrays_cast_to_float32=True,
            exact_after_float32_cast=True,
            atol=0.0,
            rtol=0.0,
            loader_resource_reconstruction_uses_official_atol=1e-7,
            loader_resource_reconstruction_uses_official_rtol=1e-6,
        ),
    )


def check_witness_records(witness_directory: Path, paired, witness_targets) -> dict:
    protocol_path = witness_directory / "protocol.json"
    result_path = witness_directory / "result.json"
    witness_file = witness_directory / "witnesses.jsonl"
    require(protocol_path.is_file() and result_path.is_file() and witness_file.is_file(),
            f"missing witness artifact under {witness_directory}")
    protocol = read_json(protocol_path)
    result = read_json(result_path)
    records = read_jsonl(witness_file)
    record_ids = [record.get("decision_id") for record in records]
    require(len(records) == 127 and len(set(record_ids)) == 127,
            f"witness JSONL records/unique IDs differ: {len(records)}/{len(set(record_ids))}")
    selected_ids = list(protocol.get("selected_ids", []))
    require(len(selected_ids) == 127 and len(set(selected_ids)) == 127,
            f"protocol selected IDs differ: {len(selected_ids)}/{len(set(selected_ids))}")
    require(selected_ids == record_ids, "protocol selected IDs differ from witness JSONL order")
    require(int(result.get("states", -1)) == 127, f"witness result states {result.get('states')} != 127")
    require(result.get("status") == "COMPLETE" and result.get("error") is None,
            f"witness result status/error differs: {result.get('status')}/{result.get('error')}")

    target_indices = [int(value) for value in witness_targets.state_index.detach().cpu().tolist()]
    target_record_ids = {paired.original.ids[index] for index in target_indices}
    destination_record_ids = set()
    zero_target_records = []
    reason_counts = Counter()
    all_destinations = 0
    for record in records:
        candidates = record.get("candidates", [])
        destination_count = sum(len(candidate.get("destinations", [])) for candidate in candidates)
        all_destinations += destination_count
        if destination_count:
            destination_record_ids.add(record["decision_id"])
            continue
        require(int(record.get("teacher_gate")) == 0, f"zero-target record is not WAIT: {record['decision_id']}")
        require(int(record.get("plan_calls", -1)) == 0, f"zero-target plan calls nonzero: {record['decision_id']}")
        reasons = []
        candidate_summary = []
        for candidate in candidates:
            global_reason = candidate.get("global_reason")
            original_reason = candidate.get("original_reason")
            require(global_reason and global_reason == original_reason,
                    f"zero-target reason mismatch: {record['decision_id']}")
            reasons.append(global_reason)
            candidate_summary.append(dict(joint=candidate.get("joint"), destinations=candidate.get("destinations", [])))
        require(reasons, f"zero-target record has no candidates: {record['decision_id']}")
        require(len(set(reasons)) == 1, f"zero-target reasons differ: {record['decision_id']}")
        reason = reasons[0]
        reason_counts[reason] += 1
        zero_target_records.append(dict(
            seed=record.get("seed"),
            decision_id=record.get("decision_id"),
            interface=record.get("interface"),
            teacher_gate=record.get("teacher_gate"),
            plan_calls=record.get("plan_calls"),
            candidates=candidate_summary,
            global_reason=reason,
        ))
    require(len(destination_record_ids) == 121, f"records with destinations {len(destination_record_ids)} != 121")
    require(len(zero_target_records) == 6, f"zero-target record count {len(zero_target_records)} != 6")
    require(reason_counts == Counter({"final_capacity": 5, "lz_unavailable": 1}),
            f"zero-target reason counts differ: {dict(reason_counts)}")
    require(target_record_ids == destination_record_ids,
            "witness target state IDs differ from records with at least one destination")
    require(all_destinations == len(witness_targets.state_index) == 317,
            f"destination/query counts differ: {all_destinations}/{len(witness_targets.state_index)}")
    witness_sha = sha256_file(witness_file)
    require(result.get("witness_sha256") == witness_sha, "witness result SHA differs from file SHA")
    return dict(
        protocol_file=repo_relative(protocol_path),
        protocol_file_sha256=sha256_file(protocol_path),
        result_file=repo_relative(result_path),
        result_file_sha256=sha256_file(result_path),
        witnesses_file=repo_relative(witness_file),
        witnesses_file_sha256=witness_sha,
        jsonl_records=127,
        jsonl_unique_decision_ids=127,
        protocol_selected_ids=127,
        result_states=127,
        target_states=121,
        queries=317,
        records_with_destinations=121,
        zero_target_records=zero_target_records,
        zero_target_reason_counts=dict(reason_counts),
        all_destinations=all_destinations,
        target_ids_match_destination_record_ids=True,
    )


def run_check(root: Path, witness_directory: Path) -> dict:
    from terminal_marl.training.p3_neighbor_witness_data import candidate_indices, selected_inputs
    from terminal_marl.training.p3_neighbor_witness_dataset import load_witness_targets
    from terminal_marl.training.p3_v26_train_dataset import load_nine_cycle_train

    specs = build_specs(root.expanduser())
    rows_by_id, input_summary, rows_by_seed = load_input_rows(specs)
    resource_context_checks = check_resource_context_prefixes(rows_by_seed)
    original_directories = [item["directory"] for item in specs if item["group"] == "original"]
    independent_directories = [item["directory"] for item in specs if item["group"] == "independent"]
    supplemental_directories = [item["directory"] for item in specs if item["group"] == "supplemental"]
    paired = load_nine_cycle_train(original_directories, independent_directories, supplemental_directories)
    loaded = compare_loaded_rows(paired, rows_by_id, rows_by_seed)
    loaded["resource_context_checks"] = resource_context_checks

    witness_directory = witness_directory.expanduser()
    witness_targets = load_witness_targets(witness_directory, paired)
    witness_state_indices = witness_targets.state_index.detach().cpu().tolist()
    witness_seeds = sorted(set(int(value) for value in witness_targets.seeds.detach().cpu().tolist()))
    witness_query_count = len(witness_state_indices)
    witness_state_count = len(set(int(value) for value in witness_state_indices))
    require(witness_query_count == 317, f"old witness query count {witness_query_count} != 317")
    require(witness_state_count == 121, f"old witness target state count {witness_state_count} != 121")
    require(witness_seeds == list(SEED_ORDER[:6]), f"old witness seeds differ: {witness_seeds}")
    witness = check_witness_records(witness_directory, paired, witness_targets)
    witness.update(dict(
        protocol_directory=str(witness_directory),
        parent_seeds=witness_seeds,
        old6_queries_and_states_checked=True,
        new3_witness_query_calls=0,
        new3_witness_states=0,
        new3_witness_loaded=False,
    ))

    selected, selection_inputs = selected_inputs(supplemental_directories, supplemental=True)
    selected_ids = [item[2]["decision_id"] for item in selected]
    require(len(selected_ids) == len(set(selected_ids)), "supplemental selected IDs are not unique")
    require(len(selected_ids) <= 72, f"supplemental selection count {len(selected_ids)} exceeds 72")
    candidate_checks = 0
    for _episode, _entry, row, label in selected:
        candidates = candidate_indices(label)
        audited = {int(item["joint_index"]) for item in label["teacher_audit"]}
        require(set(candidates) <= audited, f"selected candidate outside original teacher prefix: {row['decision_id']}")
        candidate_checks += len(candidates)
    selected_by_seed = Counter(int(item[2]["seed"]) for item in selected)
    selection_sha = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    supplemental_selection = dict(
        expected_states=65,
        expected_by_seed={"893604": 17, "893605": 24, "893606": 24},
        actual_states=len(selected_ids),
        actual_by_seed={str(seed): selected_by_seed[seed] for seed in (893604, 893605, 893606)},
        max_states=72,
        selected_ids=selected_ids,
        selected_ids_sha256=selection_sha,
        candidate_prefix_checks=candidate_checks,
        selected_inputs_summary=selection_inputs,
        collection_or_witness_state_calls=0,
    )

    source_hashes = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"missing source file: {path}")
        source_hashes[relative] = sha256_file(path)
    source_path = Path(__file__).resolve()
    report = dict(
        status="PASS",
        source_file=repo_relative(source_path),
        source_file_sha256=sha256_file(source_path),
        source_hashes=source_hashes,
        execution_scope="read-only loader/selection; no collect/witness_state",
        no_q=True,
        no_environment_reset_step=True,
        no_teacher_query=True,
        no_optimizer=True,
        input_root=str(root),
        input_files=input_summary,
        paired=loaded,
        old_witness=witness,
        supplemental_selection=supplemental_selection,
    )
    return report


def write_report(path: Path, report: dict) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--witness-directory", type=Path, default=DEFAULT_WITNESS_DIRECTORY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_check(args.root, args.witness_directory)
        write_report(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "source_file_sha256": report["source_file_sha256"]}, ensure_ascii=False))
        return 0
    except Exception as error:
        report = {
            "status": "FAIL",
            "source_file": repo_relative(Path(__file__).resolve()),
            "source_file_sha256": sha256_file(Path(__file__).resolve()),
            "error": repr(error),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
            "execution_scope": "read-only loader/selection; no collect/witness_state",
            "no_q": True,
            "no_environment_reset_step": True,
            "no_teacher_query": True,
            "no_optimizer": True,
        }
        try:
            write_report(args.output, report)
        except Exception as write_error:
            print(f"failed to write FAIL report: {write_error!r}", file=sys.stderr)
        print(json.dumps({"status": "FAIL", "output": str(args.output), "error": repr(error)}, ensure_ascii=False),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
