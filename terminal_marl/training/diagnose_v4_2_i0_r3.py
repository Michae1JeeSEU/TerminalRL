"""Read-only R3 stopping, scenario, and distance attribution; never qualifies data."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..core.observations import LOW_OBS_V2_1
from .run_v4_2_v2_1_formal_i0_r3 import STRATA, _config, _coverage, _reference_ids, _verify_protocol


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def distance_attribution(training: list[dict], reference: list[dict], *, same_proposal: bool) -> dict:
    """Remove identity coordinates only as a diagnostic, never alter actor inputs."""
    train = np.asarray([row["observation_v2_1"] for row in training], dtype=np.float64)
    keep = np.ones(88, dtype=bool)
    for name in ("mission", "gate"):
        keep[LOW_OBS_V2_1.slices[name]] = False
    distances, reduced, contributions = [], [], Counter()
    for row in reference:
        candidates = np.asarray([r["teacher_gate"] == 1 and (
            not same_proposal or r["teacher_joint_index"] == row["teacher_joint_index"]
        ) for r in training])
        if not candidates.any():
            raise ValueError("reference has no matching RELEASE candidate")
        squared = (train[candidates] - np.asarray(row["observation_v2_1"])) ** 2
        nearest = squared[np.argmin(squared.sum(axis=1))]
        distances.append(float(np.sqrt(nearest.sum())))
        reduced.append(float(np.sqrt(squared[:, keep].sum(axis=1).min())))
        for name, section in LOW_OBS_V2_1.slices.items():
            contributions[name] += float(nearest[section].sum())
    return {
        "count": len(reference),
        "full_88d": {str(r): float(np.mean(np.asarray(distances) <= r)) for r in (.25, 1.)},
        "without_mission_gate_diagnostic_only": {str(r): float(np.mean(np.asarray(reduced) <= r)) for r in (.25, 1.)},
        "nearest_squared_distance_by_feature": dict(contributions.most_common()),
    }


def run(training_path: Path, audit_path: Path, q_path: Path, consensus_path: Path, output: Path) -> dict:
    protocol, verified = _verify_protocol()
    if not verified["PASS"]:
        raise ValueError("historical protocol integrity failed")
    training, audit, q = map(read_rows, (training_path, audit_path, q_path))
    fn, prop = _reference_ids(json.loads(consensus_path.read_text(encoding="utf-8")))
    cycles = {}
    for name, rows in (("training", training), ("audit", audit)):
        episodes = list(dict.fromkeys(row["episode_id"] for row in rows))
        cycles[name] = []
        for end in range(len(STRATA), len(episodes) + 1, len(STRATA)):
            accepted = set(episodes[:end])
            prefix = [row for row in rows if row["episode_id"] in accepted]
            coverage = _coverage(prefix, protocol)
            cycles[name].append({"episodes": end, "rows": len(prefix),
                                 "C05": coverage["residual_context_cells"]["C05"],
                                 "anchors_PASS": coverage["nominal_anchor_PASS"]})
    configs = {}
    for stratum in STRATA:
        signature = json.dumps(vars(_config(stratum.spec)), sort_keys=True, default=str)
        configs.setdefault(signature, []).append(stratum.name)
    result = {"evidence_type": "newly_run_analysis_of_preexisting_artifacts", "PPO_ALLOWED": False,
              "R3_status_unchanged": "FAIL", "complete_cycle_prefixes": cycles,
              "identical_effective_config_groups": list(configs.values()),
              "shared_FN": distance_attribution(training, [q[i] for i in fn], same_proposal=False),
              "shared_proposal_error": distance_attribution(training, [q[i] for i in prop], same_proposal=True)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("training", "audit", "q", "consensus", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.training, args.audit, args.q, args.consensus, args.output), indent=2))
