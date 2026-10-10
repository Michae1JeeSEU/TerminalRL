"""Create the exclusive, primary-reviewed V7 formal admission report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path("/home/user/projects/terminal")
OUTPUT_ROOT = Path(
    "/home/user/training_outputs/terminal/p3_adaptive_kl_probe_v1_20260926"
)
REPORT = OUTPUT_ROOT / "formal_v7_admission.json"
PROTOCOL = PROJECT_ROOT / "terminal_marl/configs/low_v4_2/p3_adaptive_kl_v7.yaml"
SOURCE_CHECKPOINT_SHA256 = (
    "03c0acb0d561219cbe1dc8dac88cd4df696ade4e4691ff0d4db1d4a72ed5e3ba"
)
PROTOCOL_SHA256 = (
    "d2bbd3a3361abb0589099fd2096b8f404f61c34bf94db10a958de99fc3f14a53"
)
PROBES = (
    (
        0,
        OUTPUT_ROOT / "run0/manifest.json",
        "2f0fbf9e890662eaa1ca7cb5ae5b0eea7712e91df8b3ae785c4a9bb361cf4525",
    ),
    (
        2,
        OUTPUT_ROOT / "run2/manifest.json",
        "e85babb969d33fbd3b691f8bd1505f951a81acbe0b0f1717c8c3643baadcda59",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite existing admission report: {REPORT}")
    actual_protocol_sha = sha256(PROTOCOL)
    if actual_protocol_sha != PROTOCOL_SHA256:
        raise RuntimeError(
            f"protocol SHA mismatch: expected {PROTOCOL_SHA256}, got {actual_protocol_sha}"
        )
    manifests = []
    for run_index, path, expected_sha in PROBES:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"run{run_index} manifest SHA mismatch: expected {expected_sha}, got {actual_sha}"
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED":
            raise RuntimeError(f"run{run_index} probe is not COMPLETED")
        if manifest.get("stage") != "P3_ADAPTIVE_KL_PROBE_V1":
            raise RuntimeError(f"run{run_index} probe stage mismatch")
        if manifest.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError(f"run{run_index} source checkpoint SHA mismatch")
        if manifest.get("reference_soft_settings", {}).get("soft_kl_target") != 0.02:
            raise RuntimeError(f"run{run_index} soft KL target mismatch")
        if manifest.get("reference_soft_settings", {}).get("soft_kl_gain") != 400.0:
            raise RuntimeError(f"run{run_index} soft KL gain mismatch")
        manifests.append({"run_index": run_index, "path": str(path), "sha256": actual_sha})

    payload = {
        "stage": "P3_ADAPTIVE_KL_FORMAL_ADMISSION_V7",
        "FORMAL_P3_ALLOWED": True,
        "P3_QUALIFIED": False,
        "P4_ALLOWED": False,
        "initial_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "adaptive_probe_manifests": manifests,
        "primary_review": {
            "probe_records": 16,
            "updated": 16,
            "skipped_kl": 0,
            "accepted_minibatches": 630,
            "restore_exact": 16,
            "requests": 264,
            "normal": 227,
            "emergency": 37,
            "failed_energy": 0,
            "unresolved": 0,
            "dense24_development_tradeoff": {
                "run0": {
                    "emergency_before_after": [1, 5],
                    "priority_wait_delta_seconds": 15.6374,
                    "low_soc_wait_delta_seconds": 40.9181,
                },
                "run2": {
                    "emergency_before_after": [3, 1],
                    "priority_wait_delta_seconds": 5.8564,
                    "low_soc_wait_delta_seconds": 2.5737,
                },
            },
            "admission_scope": "learning_stall_repair_only",
            "performance_review_required": True,
            "fresh_formal_evaluation_required": True,
            "formal_campaign_not_started": True,
        },
        "fresh_formal_initialization": {
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "fresh_adam": True,
            "probe_checkpoints_not_used_as_initialization": True,
        },
        "no_continuous_monitoring": True,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"created": str(REPORT), "protocol_sha256": actual_protocol_sha,
                      "adaptive_probe_manifests": manifests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
