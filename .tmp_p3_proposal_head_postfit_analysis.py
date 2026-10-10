from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from terminal_marl.core.checkpoints import load_checkpoint
from terminal_marl.training import k3_gate_feasibility
from terminal_marl.training import train_p3_proposal_head_probe_v1 as runner
from terminal_marl.algo.gate_calibrated_adapter_v4_2 import (
    CALIBRATION_TEMPERATURE,
    TemperatureCalibratedGateResidualAdapterStudentPPOV42,
)


COLLECTION = Path(r"D:\7_codespace\training_outputs\terminal\p3_proposal_head_probe_v1_20260926\collection")
SOURCE = Path(r"D:\7_codespace\training_outputs\terminal\p3_gate_coverage_repair_v2_20260925\fit853401\p3_gate_coverage_repair_v2.pt")
CANDIDATE = Path(r"D:\7_codespace\training_outputs\terminal\p3_proposal_head_probe_v1_20260926\fit\p3_proposal_head_probe_v1.pt")


def _top3(logits: torch.Tensor) -> list[int]:
    return [int(value) for value in torch.argsort(logits, descending=True, stable=True)[:3].tolist()]


def _forward(policy, row):
    obs = torch.as_tensor([row["observation"]], dtype=torch.float32, device=policy.device)
    with torch.no_grad():
        gate, proposal = policy.model.distributions(obs)
        logits = proposal.logits[0].detach().cpu()
        probs = F.softmax(logits, dim=-1)
    certified = [int(value) for value in row["certified_proposals"]]
    q = k3_gate_feasibility.k3_certified_mass(logits, certified, shortlist_k=3)
    top3 = _top3(logits)
    intersection = sorted(set(top3).intersection(certified))
    direct = float(probs[certified].sum().item()) if certified else 0.0
    return {
        "qK3": float(q["certified_mass"]),
        "top3": top3,
        "top3_certified_intersection": intersection,
        "p_certified_single_sample": direct,
    }


def main() -> None:
    rows, collection_report = runner._load_collection(COLLECTION)
    config = runner._build_execution_config()
    source, _, source_sha = runner._load_source_policy(SOURCE, config)
    candidate = TemperatureCalibratedGateResidualAdapterStudentPPOV42(
        runner.OBSERVATION_DIM, temperature=CALIBRATION_TEMPERATURE,
    )
    candidate_payload = load_checkpoint(CANDIDATE, candidate, config, load_optimizer=False)
    candidate.model.eval()
    records = []
    for row in rows:
        source_values = _forward(source, row)
        candidate_values = _forward(candidate, row)
        records.append({
            "seed": int(row["seed"]),
            "split": str(row["split"]),
            "step": int(row["step"]),
            "uid": row.get("uid"),
            "arrival_id": row.get("arrival_id"),
            "phase": row.get("phase"),
            "priority": row.get("priority"),
            "soc": row.get("soc"),
            "certified_proposals": row["certified_proposals"],
            "source": source_values,
            "candidate": candidate_values,
            "top3_changed": source_values["top3"] != candidate_values["top3"],
            "source_covered_candidate_uncovered": bool(
                source_values["top3_certified_intersection"]
                and not candidate_values["top3_certified_intersection"]
            ),
        })
    selected = [
        item for item in records
        if item["certified_proposals"] and item["source"]["qK3"] < 0.01
    ]
    split_summary = {}
    for split in ("train", "validation"):
        subset = [item for item in selected if item["split"] == split]
        split_summary[split] = {
            "rows": len(subset),
            "source_p_certified_single_sample_mean": (
                sum(item["source"]["p_certified_single_sample"] for item in subset) / len(subset)
                if subset else None
            ),
            "candidate_p_certified_single_sample_mean": (
                sum(item["candidate"]["p_certified_single_sample"] for item in subset) / len(subset)
                if subset else None
            ),
            "source_qK3_mean": (
                sum(item["source"]["qK3"] for item in subset) / len(subset)
                if subset else None
            ),
            "candidate_qK3_mean": (
                sum(item["candidate"]["qK3"] for item in subset) / len(subset)
                if subset else None
            ),
        }
    result = {
        "source_sha256": source_sha,
        "candidate_sha256": runner._sha256(CANDIDATE),
        "candidate_payload_source_sha256": candidate_payload["metadata"].get("source_checkpoint_sha256"),
        "rows_sha256": collection_report["rows_sha256"],
        "rows": len(records),
        "selected_low_qK3_nonempty_C_rows": len(selected),
        "selected_split_summary": split_summary,
        "selected_rows": selected,
        "all_rows_top3_changed": sum(item["top3_changed"] for item in records),
        "all_rows_sourcecovered_to_candidateuncovered": sum(
            item["source_covered_candidate_uncovered"] for item in records
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

