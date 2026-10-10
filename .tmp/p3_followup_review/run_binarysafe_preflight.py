from pathlib import Path

from terminal_marl.training import evaluate_p3_task_neutral as neutral_eval
from terminal_marl.training import run_p3_followup_diagnostics as runner


LOCAL_OUTPUT_ROOT = Path(r"D:\7_codespace\training_outputs\terminal")
SERVER_PREFIX = "/home/user/training_outputs/terminal/"


def synced_path(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    value = str(path).replace("\\", "/")
    if value.startswith(SERVER_PREFIX):
        candidate = LOCAL_OUTPUT_ROOT / value[len(SERVER_PREFIX):]
        if candidate.is_file():
            return candidate
    return path


neutral_eval._synced_path = synced_path
report = runner._build_preflight()
assert report["status"] == "PREFLIGHT_OK"
assert report["protocol"]["observation_dimension"] == 88
assert report["protocol"]["selection_size"] == 24
assert len(report["protocol"]["continuation_seeds"]) == 4
print("BINARYSAFE_REAL_BUILD_PREFLIGHT_OK")
print(f"status={report['status']}")
print(f"episodes={report['inputs']['episodes']}")
print(f"total_steps={report['inputs']['total_steps']}")
print(f"total_requests={report['inputs']['total_requests']}")
print(f"observation_dimension={report['protocol']['observation_dimension']}")
print(f"selection_size={report['protocol']['selection_size']}")
