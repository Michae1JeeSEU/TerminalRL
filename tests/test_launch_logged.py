from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "server" / "launch_logged.py"


def _run(tmp_path: Path, code: str, prefix_name: str = "run") -> tuple[subprocess.CompletedProcess[str], Path]:
    output_dir = tmp_path / "training_outputs"
    prefix = output_dir / prefix_name
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log-prefix",
            str(prefix),
            "--cwd",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result, prefix


def _wait_for(path: Path) -> str:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text(encoding="utf-8")
        time.sleep(0.02)
    raise AssertionError(f"marker was not created: {path}")


def test_exit_zero_writes_pid_exit_and_log_without_leaf(tmp_path: Path) -> None:
    result, prefix = _run(tmp_path, "print('zero')")

    assert result.returncode == 0
    assert not prefix.is_dir()
    assert _wait_for(Path(f"{prefix}.exit")) == "0"
    pid = Path(f"{prefix}.pid").read_text(encoding="utf-8")
    assert pid.isdigit() and int(pid) > 0
    assert "zero" in Path(f"{prefix}.log").read_text(encoding="utf-8")
    assert "zero" not in Path(f"{prefix}.launcher.log").read_text(encoding="utf-8")


def test_exit_seven_is_recorded_without_shell(tmp_path: Path) -> None:
    result, prefix = _run(tmp_path, "print('seven'); raise SystemExit(7)")

    assert result.returncode == 0
    assert _wait_for(Path(f"{prefix}.exit")) == "7"
    assert "seven" in Path(f"{prefix}.log").read_text(encoding="utf-8")


def test_existing_prefix_markers_are_rejected_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "training_outputs"
    output_dir.mkdir()
    for name, suffix in (("existing-launcher", ".launcher.log"), ("existing-child", ".log")):
        prefix = output_dir / name
        log_path = Path(f"{prefix}{suffix}")
        log_path.write_text("keep", encoding="utf-8")

        result, _ = _run(tmp_path, "print('must not run')", name)

        assert result.returncode != 0
        assert log_path.read_text(encoding="utf-8") == "keep"
        assert not Path(f"{prefix}.pid").exists()
        assert not Path(f"{prefix}.exit").exists()
