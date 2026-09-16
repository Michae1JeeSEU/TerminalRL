import numpy as np

from terminal_marl.training.diagnose_v4_2_i0_r3 import distance_attribution


def test_one_hot_identity_can_exclude_identical_control_context():
    left = np.zeros(88)
    right = left.copy()
    left[18], right[19] = 1, 1
    def row(obs, joint=14):
        return {"observation_v2_1": obs.tolist(), "teacher_gate": 1, "teacher_joint_index": joint}
    result = distance_attribution([row(left)], [row(right)], same_proposal=True)
    assert result["full_88d"]["1.0"] == 0
    assert result["without_mission_gate_diagnostic_only"]["0.25"] == 1
    assert result["nearest_squared_distance_by_feature"]["mission"] == 2
