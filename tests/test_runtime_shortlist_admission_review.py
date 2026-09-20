from terminal_marl.training import review_v4_2_p2_runtime_shortlist_admission_v1 as review


def _manifest():
    records = []
    for controller in ("candidate_k1", "candidate_k3", "rule"):
        for seed in range(6):
            result = {
                "requests": 12, "completed": 12, "gates": {"all": True},
                "no_teacher": True, "requested_logprob_exact": True,
                "runtime_shortlist_checks": {
                    "all": True, "max_candidates_checked": 3 if controller == "candidate_k3" else 1,
                    "fallback_selected": 1 if controller == "candidate_k3" else 0,
                },
            }
            records.append({"controller": controller, "seed": seed, "result": result})
    return {
        "stage": "V4_2_P2_GATE_RUNTIME_SHORTLIST_EVALUATION_V2",
        "status": "COMPLETE", "smoke": False,
        "protocol_sha256": "67e61165ef5c056d1b1ebd0a789cafd17f5e0554d07b29128617df3aa8bf6d5d",
        "records": records,
        "aggregates": {name: {"safety_all_zero": True} for name in ("candidate_k1", "candidate_k3", "rule")},
        "decision": {"all": True, "k3_fallback_selected_nonzero": True,
                     "k3_wait_nonworse": True, "k3_request_to_touchdown_nonworse": True,
                     "at_least_one_effectiveness_metric_strictly_better": True},
        "k3_minus_k1": {"weighted_wait_seconds": -1.0, "request_to_touchdown_seconds": -0.5},
        "RUNTIME_SHORTLIST_FREEZE": True, "PPO_ALLOWED": False, "checkpoint_promotable": False,
    }


def test_admission_review_accepts_only_complete_bounded_effective_evidence():
    gates = review.review_gates(_manifest(), review.protocol())
    assert gates and all(gates.values())


def test_admission_review_fails_closed_on_missing_fallback_or_likelihood():
    manifest = _manifest()
    for other in manifest["records"]:
        if other["controller"] == "candidate_k3":
            other["result"]["runtime_shortlist_checks"]["fallback_selected"] = 0
            other["result"]["requested_logprob_exact"] = False
    gates = review.review_gates(manifest, review.protocol())
    assert not gates["k3_fallback_exercised"]
    assert not gates["requested_likelihood"]
    assert not all(gates.values())
