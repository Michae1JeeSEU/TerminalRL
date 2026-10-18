"""Read-only preflight for the delivered FT-RS-S1-PERF-1.0 specification."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

import yaml

from .provenance import (R0_COMMIT, R0_POLICY_ID, PHYSICAL_PATH, digest,
                         semantic_fingerprint, require_same_semantics, ProvenanceError)

CONTRACT_ID = "FT-RS-S1-PERF-1.0"
CONTRACT_FILE = "rotating_slots_simulation_performance_contract_v1_0.md"
CONFIG_FILE = "performance_acceptance_config_v1_0.yaml"
CATALOG_FILE = "performance_acceptance_catalog_v1_0.json"
# Normative input bytes are fixed by the delivered package, independent of its
# editable MANIFEST.json. A changed specification requires explicit new approval.
PACKAGE_SHA256 = {
    CONTRACT_FILE: "bfa3f21a6c5f74c2bbfb170b70e2bdec6000834a90aa37d108ce4869aaed6441",
    CONFIG_FILE: "1d0c167053d83f6d7a96cad04ea84215d0a06f3744ef597f046c427b65b89bef",
    CATALOG_FILE: "ea82df33048e249cc13525fdc0e3e2ef5a4dd96083decb71b54d647d20da8430",
}
# Derived from the byte-verified R0 export; these are semantic projections, not
# implementation-source hashes. Raw LF/CRLF config bytes are recorded separately.
FROZEN_R0_SEMANTICS = {
    "schema": "ft_rs_semantic_fingerprint_v1_0", "reference_commit": R0_COMMIT,
    "policy_id": R0_POLICY_ID,
    "config_hash": "0c7cce0c0c019d50f7bba5aecd5e8340925d0d16867f8f06eb51e89102343d53",
    "vehicle_profile_hash": "0fdecf03b8dbaffad0c49abffc299e5b63c7f3f34767182b1ca1871ce1a18ea1",
    "energy_profile_hash": "383ff383b191d2e5238d988fee55cb24b6f6b153781b5675bfe26ee6e43a5d02",
    "profile_hash": "34b25de04b038a99a53cc4cbdd33e11aa64e72d44edf4fd5caffbdfa2c2f51e4",
    "effective_rules_hash": "1e453ac8863130dd5faa801fae74d5d9e885df73463de6e176e2a240a1e331a2",
    "candidate_policy_hash": "16f7576fb0fed8a7bd4e235921e1f0c639273fa5e6862db1d403d2a1e32663ac",
    "numerical_policy_hash": "ffaf8abb3fd6d5dfad8c7e4f555c440a10a1cb9e18131f5c1283296aaec86cb1",
    "semantic_sha256": "1621315edfef4c59be2cf1d632ad8d7efecac6fb0bcd55474e4e0169347d4ce6",
}


class ContractInconsistent(ValueError):
    code = "CONTRACT_INCONSISTENT"

    def __init__(self, detail):
        super().__init__(f"{self.code}: {detail}")


def preflight(package_dir, source_root, reference_semantics=None):
    """Validate normative package and actual semantic policy before execution.

    Returns parsed config/catalogue, package hash and source semantic identity.
    Does not modify normative files, historical outputs, or qualification status.
    The optional reference must be the verified frozen R0 semantic projection,
    rather than an optimized implementation's self-generated expected values.
    """
    package_dir, source_root = Path(package_dir), Path(source_root)
    try:
        raw = {name: (package_dir / name).read_bytes() for name in PACKAGE_SHA256}
        hashes = {name: sha256(data).hexdigest() for name, data in raw.items()}
        if hashes != PACKAGE_SHA256:
            raise ContractInconsistent("normative contract/config/catalog bytes differ from v1.0")
        config = yaml.safe_load(raw[CONFIG_FILE])
        catalog = json.loads(raw[CATALOG_FILE])
        markdown = raw[CONTRACT_FILE].decode("utf8")
        if config["contract_id"] != CONTRACT_ID or catalog["contract_id"] != CONTRACT_ID or CONTRACT_ID not in markdown:
            raise ContractInconsistent("contract identity")
        if config["reference"]["commit"] != R0_COMMIT or catalog["reference_commit"] != R0_COMMIT or R0_COMMIT not in markdown:
            raise ContractInconsistent("reference commit")
        if config["reference"]["policy_id"] != R0_POLICY_ID or catalog["policy_id"] != R0_POLICY_ID or R0_POLICY_ID not in markdown:
            raise ContractInconsistent("policy identity")
        expected_ids = [f"{group}{index:02}" for group, count in
                        (("A", 8), ("B", 10), ("C", 10), ("D", 10), ("E", 10), ("F", 8), ("G", 8), ("H", 6))
                        for index in range(1, count + 1)]
        ids = [case["id"] for case in catalog["cases"]]
        markdown_ids = re.findall(r"\| ([A-H]\d\d) ·", markdown)
        if ids != expected_ids or markdown_ids != expected_ids or catalog["count"] != 70 or config["acceptance"]["additional_catalog_count"] != 70:
            raise ContractInconsistent("catalogue count/order/Markdown mapping")
        if any(not c["required"] or c["status"] != "NOT_RUN" for c in catalog["cases"]):
            raise ContractInconsistent("specification requirements mistaken for executed results")
        traffic = config["traffic"]
        if traffic["case_count"] != len(traffic["rates_per_hour"]) * len(traffic["seeds"]) or traffic["oracle_policy_id"] != R0_POLICY_ID:
            raise ContractInconsistent("traffic matrix")
        windows = config["benchmark"]["windows"]
        if len(windows) != 6 or len({w["id"] for w in windows}) != 6:
            raise ContractInconsistent("paired windows")
        assertions = config["frozen_semantic_assertions"]
        physical = yaml.safe_load((source_root / PHYSICAL_PATH).read_bytes())
        checks = {
            "hard_separation_m": physical["safety"]["hard_separation_m"],
            "additional_plan_buffer_m": physical["safety"]["additional_plan_buffer_m"],
            "physics_dt_s": physical["timing"]["physics_sample_dt_s"],
            "scheduler_period_s": physical["timing"]["scheduler_period_s"],
            "default_preview_s": physical["timing"]["default_preview_horizon_s"],
            "normal_touchdown_reserve": physical["energy"]["normal_landing_reserve"],
            "absolute_emergency_trigger_soc": physical["emergency"]["trigger_absolute_soc"],
            "reference_holding_s": physical["emergency"]["trigger_rescue_reference_plus_hold_s"],
            "terminal_peak_budget_s": physical["energy"]["terminal_visit_peak_duration_budget_s"],
            "induced_root_bisections": physical["energy"]["induced_root_bisections"],
            "stable_slots_by_ring": {r["id"]: r["slot_count"] for r in physical["airspace"]["rings"]},
            "ground_pads": physical["pads"]["count"],
        }
        if any(assertions[key] != value for key, value in checks.items()):
            raise ContractInconsistent("physical/numerical assertion mismatch")
        if assertions["final_duration_grid_s"] != list(range(20, 301, 5)):
            raise ContractInconsistent("candidate duration grid")
        current = semantic_fingerprint(source_root)
        expected = dict(FROZEN_R0_SEMANTICS, physical_config_bytes_sha256=current["physical_config_bytes_sha256"])
        if reference_semantics is not None:
            require_same_semantics(expected, reference_semantics)
            expected = reference_semantics
        require_same_semantics(expected, current)
    except ContractInconsistent:
        raise
    except (OSError, ValueError, KeyError, TypeError, StopIteration, SyntaxError) as error:
        raise ContractInconsistent(str(error)) from error
    return {"contract_id": CONTRACT_ID, "config": config, "catalog": catalog,
            "package_files_sha256": hashes, "package_sha256": digest(hashes),
            "semantic_fingerprint": current, "implementation_status": "IMPLEMENTING",
            "performance_status": "NOT_QUALIFIED", "human_status": "HUMAN_PENDING"}
