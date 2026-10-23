param(
    [ValidateSet('stress','diagnostic','medium')][string]$Scene = 'stress',
    [ValidateSet('energy_slack','rh_cp_sat')][string]$Method = 'energy_slack'
)
# Each branch is one literal PowerShell python -c launch command. Close the viewer to return.
Push-Location -LiteralPath 'D:\7_codespace\terminal'
try {
    switch ("$Scene/$Method") {
        'stress/energy_slack' {
            & D:\7_codespace\terminal\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_need_v7_stress_600s_20261023\energy_slack\replay_0600')).run()"
        }
        'stress/rh_cp_sat' {
            & D:\7_codespace\terminal\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_need_v7_stress_600s_20261023\rh_cp_sat\replay_0600')).run()"
        }
        'diagnostic/energy_slack' {
            & D:\7_codespace\terminal\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_need_v7_diagnostic_600s_20261023\energy_slack\replay_0600')).run()"
        }
        'diagnostic/rh_cp_sat' {
            & D:\7_codespace\terminal\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_need_v7_diagnostic_600s_20261023\rh_cp_sat\replay_0600')).run()"
        }
        'medium/energy_slack' {
            & D:\7_codespace\terminal\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_need_v7_medium_600s_20261023\energy_slack\replay_0600')).run()"
        }
        'medium/rh_cp_sat' {
            & D:\7_codespace\terminal\venv\Scripts\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r'D:\7_codespace\training_outputs\terminal\stage2_rescue_need_v7_medium_600s_20261023\rh_cp_sat\replay_0600')).run()"
        }
    }
} finally {
    Pop-Location
}
