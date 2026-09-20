from terminal_marl.training import review_v4_2_p2_runtime_shortlist_selection_v1 as review


def test_selection_review_protocol_preserves_failure_and_stops_p2_ppo():
    value = review.protocol()
    assert value["bindings"]["runtime_proposal_shortlist_k"] == 3
    assert value["bindings"]["no_simulator_reset"]
    assert value["bindings"]["no_training"]
    assert value["bindings"]["no_threshold_change"]
    assert value["authorization"]["select_original_k3_candidate_on_pass"]
    assert value["authorization"]["stop_additional_p2_ppo_on_pass"]
    assert value["authorization"]["P3_protocol_preregistration_allowed_on_pass"]
    assert value["authorization"]["P3_ALLOWED"] is False


def test_selection_review_protocol_hash_is_frozen():
    assert review._sha256(review.PROTOCOL) == review.PROTOCOL_SHA
