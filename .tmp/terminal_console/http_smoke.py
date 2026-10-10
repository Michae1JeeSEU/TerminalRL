from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminal_marl.visualization.console_server import ConsoleSession, build_server


CHECKPOINT = (
    ROOT.parent
    / "training_outputs"
    / "terminal"
    / "p3_normal_advantage_gate_repair_20260928"
    / "fit_smoke"
    / "terminal_gate_repair.pt"
)
EVIDENCE = ROOT / ".tmp" / "terminal_console" / "http_smoke.json"


def http_json(base: str, method: str, path: str, payload=None, token: str | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token is not None:
        headers["X-Terminal-Token"] = token
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return int(error.code), parsed


def wait_for(base: str, predicate, *, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        _, value = http_json(base, "GET", "/api/frames")
        last = value
        if predicate(value):
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for state: {last}")


def frame_summary(value):
    frames = value.get("frames", []) if isinstance(value, dict) else []
    return {
        "status": value.get("status"),
        "error": value.get("error"),
        "frame_count": len(frames),
        "times": [frame.get("time") for frame in frames],
        "steps": [frame.get("step_count") for frame in frames],
        "process_alive": value.get("process_alive"),
    }


def main() -> None:
    session = ConsoleSession(token="http-smoke-token")
    server = build_server(port=0, session=session)
    thread = Thread(target=server.serve_forever, name="terminal-console-http-smoke", daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    evidence = {"port": int(server.server_address[1]), "checkpoint": str(CHECKPOINT.resolve())}
    try:
        status, session_info = http_json(base, "GET", "/api/session")
        assert status == 200
        evidence["session_status"] = status
        evidence["session_profile_ids"] = [item["id"] for item in session_info["profiles"]]

        p3_payload = {
            "mode": "checkpoint",
            "profile": "p3_task_neutral",
            "scenario": "flow6",
            "seed": 17,
            "start_paused": True,
            "max_steps": 2,
            "max_simulation_seconds": 2.0,
            "playback_speed": 0.0,
            "low_checkpoint": str(CHECKPOINT.resolve()),
            "high_checkpoint": None,
            "deterministic": False,
        }
        status, started = http_json(base, "POST", "/api/start", p3_payload, session.token)
        assert status == 200
        evidence["p3_start_status"] = status
        evidence["p3_start"] = {
            "status": started.get("status"),
            "config": started.get("config"),
        }

        initial = wait_for(
            base,
            lambda value: value.get("status") == "paused" and len(value.get("frames", [])) >= 1,
        )
        initial_summary = frame_summary(initial)
        assert initial_summary["frame_count"] == 1
        assert initial_summary["times"] == [0.0]
        assert initial_summary["steps"] == [0]
        time.sleep(0.35)
        still_initial = http_json(base, "GET", "/api/frames")[1]
        still_initial_summary = frame_summary(still_initial)
        assert still_initial_summary["frame_count"] == 1
        assert still_initial_summary["times"] == [0.0]
        assert still_initial_summary["steps"] == [0]
        evidence["p3_initial_paused"] = initial_summary
        evidence["p3_after_350ms"] = still_initial_summary

        status, step_one = http_json(base, "POST", "/api/control", {"action": "step"}, session.token)
        assert status == 200
        evidence["p3_step1_status"] = status
        after_step_one = wait_for(
            base,
            lambda value: value.get("status") == "paused"
            and bool(value.get("frames"))
            and value["frames"][-1].get("step_count") == 1,
        )
        evidence["p3_after_step1"] = frame_summary(after_step_one)

        status, step_two = http_json(base, "POST", "/api/control", {"action": "step"}, session.token)
        assert status == 200
        evidence["p3_step2_status"] = status
        completed = wait_for(
            base,
            lambda value: value.get("status") == "completed" and not value.get("process_alive"),
        )
        evidence["p3_after_step2"] = frame_summary(completed)
        assert evidence["p3_after_step2"]["steps"][-1] == 2
        metadata = completed.get("metadata", {})
        evidence["p3_metadata"] = {
            key: metadata.get(key)
            for key in (
                "checkpoint",
                "low_checkpoint",
                "checkpoint_sha256",
                "profile",
                "scenario",
                "seed",
                "deterministic",
                "shortlist_k",
                "observation_dim",
                "weights_unchanged",
                "optimizer_loaded",
                "diagnostic_step_limit",
            )
        }
        assert metadata.get("profile") == "p3_task_neutral"
        assert metadata.get("scenario") == "flow6"
        assert metadata.get("seed") == 17
        assert metadata.get("deterministic") is False
        assert metadata.get("shortlist_k") == 3
        assert metadata.get("observation_dim") == 88
        assert metadata.get("weights_unchanged") is True
        assert metadata.get("checkpoint") == str(CHECKPOINT.resolve())

        invalid_payload = dict(p3_payload)
        invalid_payload["low_checkpoint"] = str((CHECKPOINT.parent / "missing.pt").resolve())
        status, invalid_start = http_json(base, "POST", "/api/start", invalid_payload, session.token)
        assert status == 200
        invalid = wait_for(base, lambda value: value.get("status") == "error" and not value.get("process_alive"))
        evidence["invalid_checkpoint_start_status"] = status
        evidence["invalid_checkpoint_start"] = {
            "response": invalid_start,
            "status": invalid.get("status"),
            "error": invalid.get("error"),
        }
        assert "checkpoint does not exist" in str(invalid.get("error"))
        assert "fixed_rule" not in str(invalid.get("error"))

        rule_payload = {
            "mode": "fixed_rule",
            "scenario": "default",
            "seed": 0,
            "max_steps": 2,
            "max_simulation_seconds": 2.0,
            "playback_speed": 0.0,
            "start_paused": False,
        }
        status, rule_start = http_json(base, "POST", "/api/start", rule_payload, session.token)
        assert status == 200
        rule_done = wait_for(
            base,
            lambda value: value.get("status") == "completed" and not value.get("process_alive"),
        )
        evidence["rule_reset"] = {
            "start_status": status,
            "start_response": rule_start,
            "result": frame_summary(rule_done),
        }
        assert rule_done.get("status") == "completed"
    finally:
        try:
            server.shutdown()
            server.server_close()
        finally:
            session.close()
            thread.join(timeout=5.0)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
