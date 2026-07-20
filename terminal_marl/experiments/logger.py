"""Structured JSON/CSV logging for reproducible terminal-airspace rollouts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


def current_git_commit(cwd: Path | None = None) -> str | None:
    """Return the local commit without requiring a remote repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


class ExperimentLogger:
    """Collect rollout traces and write the standard experiment artifact set."""

    STEP_FIELDS = (
        "step",
        "time",
        "landed_count",
        "active_count",
        "outer_occupancy",
        "inner_occupancy",
        "final_occupancy",
        "conflict_count",
        "raw_pair_conflict_samples",
        "pair_opportunities",
        "normalized_conflict_rate",
        "safety_override_count",
        "total_waiting_time",
        "normalized_waiting",
        "normalized_congestion",
    )
    AIRSPACE_FIELDS = (
        "step",
        "time",
        "outer_radius",
        "inner_radius",
        "outer_height",
        "inner_height",
        "outer_speed",
        "inner_speed",
        "ring_headway",
        "final_headway",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        config_name: str,
        experiment_mode: str,
        policy: str,
        seed: int,
        traffic_demand_level: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "config_name": config_name,
            "experiment_mode": experiment_mode,
            "policy": policy,
            "seed": int(seed),
            "traffic_demand_level": traffic_demand_level,
            "git_commit": current_git_commit(self.output_dir),
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        self.step_metrics: list[dict[str, Any]] = []
        self.airspace_trace: list[dict[str, Any]] = []
        self.safety_events: list[dict[str, Any]] = []
        self.agent_events: list[dict[str, Any]] = []
        self._last_phase: dict[int, int] = {}

    def record_step(self, env: Any) -> None:
        rings = sorted(env.config.rings, key=lambda ring: ring.radius, reverse=True)
        outer, inner = rings[0], rings[-1]
        metrics = env.metrics()
        self.step_metrics.append(
            {
                "step": env.step_count,
                "time": env.time,
                "landed_count": sum(aircraft.finished for aircraft in env.aircraft),
                "active_count": sum(not aircraft.finished for aircraft in env.aircraft),
                "outer_occupancy": len(env.vtac.queues[outer.index]),
                "inner_occupancy": len(env.vtac.queues[inner.index]),
                "final_occupancy": len(env.vtac.queues[0]),
                "conflict_count": env.nmac_count + env.collision_count,
                "raw_pair_conflict_samples": metrics["raw_pair_conflict_samples"],
                "pair_opportunities": metrics["pair_opportunities"],
                "normalized_conflict_rate": metrics["normalized_conflict_rate"],
                "safety_override_count": env.safety_override_count,
                "total_waiting_time": metrics["total_waiting_time"],
                "normalized_waiting": metrics["normalized_waiting"],
                "normalized_congestion": metrics["normalized_congestion"],
            }
        )
        state = env.airspace.state_vector()
        self.airspace_trace.append(
            dict(
                zip(
                    self.AIRSPACE_FIELDS,
                    [env.step_count, env.time, *[float(value) for value in state]],
                )
            )
        )
        for uid, approval in {
            **env.last_release_approval,
            **env.last_landing_approval,
        }.items():
            if approval.get("decision") == "HOLD":
                self.safety_events.append(
                    {
                        "step": env.step_count,
                        "time": env.time,
                        "uav_id": uid,
                        "target_stage": approval.get("target_stage"),
                        "decision": approval.get("decision"),
                        "reason": approval.get("reason"),
                        "violations": "|".join(approval.get("violations", [])),
                    }
                )
        for aircraft in env.aircraft:
            phase = int(aircraft.phase)
            previous = self._last_phase.get(aircraft.uid)
            if previous is None or previous != phase:
                self.agent_events.append(
                    {
                        "step": env.step_count,
                        "time": env.time,
                        "uav_id": aircraft.uid,
                        "event": "spawn" if previous is None else "phase_change",
                        "from_phase": "" if previous is None else previous,
                        "to_phase": phase,
                        "landing_delay": (
                            ""
                            if aircraft.landing_delay is None
                            else aircraft.landing_delay
                        ),
                    }
                )
            self._last_phase[aircraft.uid] = phase

    @staticmethod
    def _write_csv(
        path: Path,
        rows: Iterable[dict[str, Any]],
        default_fields: Iterable[str],
    ) -> None:
        row_list = list(rows)
        fields = list(default_fields)
        for row in row_list:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row_list)

    def finalize(self, metrics: dict[str, Any]) -> dict[str, Any]:
        summary = {**self.metadata, "metrics": metrics}
        (self.output_dir / "episode_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_csv(
            self.output_dir / "step_metrics.csv",
            self.step_metrics,
            self.STEP_FIELDS,
        )
        self._write_csv(
            self.output_dir / "airspace_trace.csv",
            self.airspace_trace,
            self.AIRSPACE_FIELDS,
        )
        self._write_csv(
            self.output_dir / "safety_events.csv",
            self.safety_events,
            ("step", "time", "uav_id", "target_stage", "decision", "reason", "violations"),
        )
        self._write_csv(
            self.output_dir / "agent_events.csv",
            self.agent_events,
            ("step", "time", "uav_id", "event", "from_phase", "to_phase", "landing_delay"),
        )
        return summary
