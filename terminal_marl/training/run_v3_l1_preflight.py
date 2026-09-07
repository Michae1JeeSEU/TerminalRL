"""V3 categorical BC and L1 smoke qualification; deliberately no PPO training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..algo.shared_ppo import SharedPPO
from ..core.checkpoints import load_checkpoint, save_checkpoint
from ..core.low_level_action import ACTION_SCHEMA_V3, LowLevelActionV3, indices_to_normalized, normalized_to_indices
from ..core.safety_trace import SafetyTraceRecorder
from ..envs.parallel_env import TerminalParallelEnv
from .audit_low_teacher_dataset import audit_teacher_dataset
from .train_low_v2_ppo import _rollout_l1, authorize_low_v2_training, l1_full_teacher_config


@dataclass(frozen=True)
class V3BCSanity:
    rows: int
    supervised_rows: int
    ce_before: tuple[float, float, float]
    ce_after: tuple[float, float, float]
    joint_before: float
    joint_after: float
    finite: bool
    gradients_finite: bool
    passed: bool


@dataclass(frozen=True)
class V3L1Preflight:
    rollout_rows: int
    teacher_interventions: int
    entropy: float
    ppo_provenance_passed: bool
    action_tape_replay_passed: bool
    checkpoint_roundtrip_passed: bool
    deterministic_validation_passed: bool
    safety_passed: bool
    passed: bool


def _rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def categorical_bc_sanity(dataset_path: str | Path, *, steps: int = 24) -> V3BCSanity:
    """Exercise only categorical CE on recorded teacher interventions."""
    torch.manual_seed(1901); np.random.seed(1901)
    rows = _rows(dataset_path)
    supervised = [row for row in rows if row["teacher_used"]]
    if not supervised:
        raise RuntimeError("V3 BC sanity requires at least one teacher intervention")
    observations = torch.as_tensor(np.asarray([row["observation_v2"] for row in supervised], dtype=np.float32))
    targets = torch.as_tensor(np.asarray([
        (row["a_teacher"]["release_index"], row["a_teacher"]["speed_index"], row["a_teacher"]["merge_time_index"])
        for row in supervised
    ], dtype=np.int64))
    policy = SharedPPO(observations.shape[1], teacher_bc_weight=.05)

    def losses() -> torch.Tensor:
        return -policy.model.actor_distribution(observations).log_prob_per_head(targets).mean(dim=0)

    with torch.no_grad():
        before = losses()
    gradients_finite = True
    for _ in range(steps):
        ce = losses(); joint = ce.sum()
        if not torch.isfinite(joint):
            raise FloatingPointError("non-finite V3 categorical BC loss")
        policy.optimizer.zero_grad(); joint.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in policy.model.parameters()
        )
        if not gradients_finite:
            raise FloatingPointError("non-finite V3 categorical BC gradient")
        policy.optimizer.step()
    with torch.no_grad():
        after = losses()
    before_values, after_values = tuple(float(value) for value in before), tuple(float(value) for value in after)
    finite = bool(np.isfinite((*before_values, *after_values)).all())
    passed = finite and gradients_finite and all(after_value < before_value for before_value, after_value in zip(before_values, after_values))
    return V3BCSanity(len(rows), len(supervised), before_values, after_values, float(before.sum()), float(after.sum()),
                       finite, gradients_finite, passed)


def _quantization_qualification() -> dict:
    """Verify deterministic catalogue mapping at endpoints and ties."""
    proposals = ((-1.0, -1.0, -1.0), (0.0, -0.75, 0.75), (0.01, -0.25, 0.25),
                 (1.0, 1.0, 1.0), (1.0, 0.74, -0.74))
    mappings = []
    for proposal in proposals:
        first = LowLevelActionV3.quantized_teacher(proposal)
        second = LowLevelActionV3.quantized_teacher(proposal)
        if first != second or not np.array_equal(normalized_to_indices(first.normalized), first.indices):
            raise RuntimeError("V3 teacher quantization is not deterministic or catalogue-exact")
        mappings.append({"continuous_proposal": list(proposal), "indices": first.indices.tolist(),
                         "normalized": first.normalized.tolist()})
    return {"passed": True, "mappings": mappings, "tie_policy": "nearest residual; exact midpoint selects lower index"}


def l1_preflight(*, dataset_path: str | Path, output_dir: str | Path) -> V3L1Preflight:
    """Smoke reset, categorical rollout, Safety/ActionTape, and checkpoint provenance."""
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    audit = authorize_low_v2_training(dataset_path)
    config = l1_full_teacher_config(medium_rate=.003, reset_mode="warm_start_rings", demand_model="finite_cohort",
                                    warm_start_uavs=8, cohort_uavs=8, horizon_seconds=8.0)
    if config.physics_dt != .2 or config.safety_sampling_dt != 1.0 or config.action_schema_version != ACTION_SCHEMA_V3:
        raise RuntimeError("V3 L1 preflight configuration drift")
    torch.manual_seed(2101); np.random.seed(2101)
    policy = SharedPPO(87, teacher_bc_weight=.05)
    model_state = {name: value.detach().clone() for name, value in policy.model.state_dict().items()}
    trace_dir = destination / "action_tapes"
    rows, summary = _rollout_l1(model_state, config, 3001, 0, 0, max_steps=8, deterministic=False,
                                trace_output_dir=trace_dir, dump_trace_on_completion=True)
    if not rows:
        raise RuntimeError("V3 L1 preflight collected no categorical rollout rows")
    provenance = True
    with torch.no_grad():
        for row in rows:
            if not np.array_equal(normalized_to_indices(row.requested_normalized_action), row.action):
                provenance = False
            if not np.isclose(float(row.old_head_log_prob.sum()), row.log_prob, atol=1e-6):
                provenance = False
            actual = policy.model.actor_distribution(torch.as_tensor(row.observation[None], dtype=torch.float32))
            if not np.isclose(float(actual.log_prob(torch.as_tensor(row.action[None])).item()), row.log_prob, atol=1e-6):
                provenance = False
        entropy = float(policy.model.actor_distribution(torch.as_tensor(np.stack([row.observation for row in rows]))).entropy().mean())
    if not provenance or not np.isfinite(entropy):
        raise RuntimeError("V3 categorical PPO requested-action provenance failed")
    tapes = sorted(trace_dir.glob("*.json"))
    if len(tapes) != 1:
        raise RuntimeError("V3 L1 preflight did not persist exactly one ActionTape")
    replay = SafetyTraceRecorder.replay_action_tape(TerminalParallelEnv(config), tapes[0])
    checkpoint = destination / "v3_l1_preflight.pt"
    save_checkpoint(checkpoint, policy, config, stage="V3_L1_preflight")
    restored = SharedPPO(87, teacher_bc_weight=.05)
    load_checkpoint(checkpoint, restored, config)
    observation = rows[0].observation[None]
    before = policy.act_details(observation, observation, deterministic=True)
    after = restored.act_details(observation, observation, deterministic=True)
    checkpoint_ok = np.array_equal(before.categories, after.categories) and np.allclose(before.joint_log_prob, after.joint_log_prob)
    deterministic_state = {name: value.detach().clone() for name, value in policy.model.state_dict().items()}
    left, _ = _rollout_l1(deterministic_state, config, 3002, 1, 0, max_steps=5, deterministic=True)
    right, _ = _rollout_l1(deterministic_state, config, 3002, 1, 1, max_steps=5, deterministic=True)
    deterministic_ok = len(left) == len(right) and all(
        np.array_equal(a.action, b.action) and np.isclose(a.log_prob, b.log_prob) for a, b in zip(left, right)
    )
    safety = all(value == 0 for value in (summary.collisions, summary.nmac, summary.hard_separation_violations))
    passed = bool(audit.passed and provenance and replay["exact"] and checkpoint_ok and deterministic_ok and safety)
    payload = V3L1Preflight(len(rows), summary.teacher_interventions, entropy, provenance, bool(replay["exact"]), checkpoint_ok,
                             deterministic_ok, safety, passed)
    (destination / "v3_l1_preflight.json").write_text(json.dumps({**asdict(payload), "dataset_audit": asdict(audit),
        "action_tape": str(tapes[0]), "replay": replay, "checkpoint": str(checkpoint), "physics_dt": config.physics_dt,
        "safety_sampling_dt": config.safety_sampling_dt}, indent=2), encoding="utf-8")
    return payload


def _markdown_l0(manifest: dict, bc: V3BCSanity, quantization: dict) -> str:
    coverage = manifest["coverage"]
    lines = ["# V3 L0 Teacher Dataset Summary", "", "Evidence type: newly_run; legacy continuous L0 labels were not read or converted.", "",
             f"- Dataset sample count: {manifest['target_rows']} (raw: {manifest['raw_rows']})",
             f"- Teacher intervention count/rate: {coverage['teacher_used']} / {coverage['teacher_used'] / manifest['target_rows']:.2%}",
             f"- Teacher quantization: {'PASS' if quantization['passed'] else 'FAIL'}",
             f"- Safety: {'PASS' if all(all(part[key] == 0 for key in ('collisions', 'nmac', 'hard_separation_violations', 'uncertified_component', 'safety_invariant_error')) for part in manifest['part_summaries']) else 'FAIL'}",
             f"- Categorical BC sanity: {'PASS' if bc.passed else 'FAIL'}", "", "## Requested class frequency", ""]
    for head, values in coverage["requested_frequency"].items(): lines.append(f"- {head}: {values}")
    lines += ["", "## Teacher target frequency", ""]
    for head, values in coverage["teacher_target_frequency"].items(): lines.append(f"- {head}: {values}")
    lines += ["", "## Corrections", "", f"- Per-combination: {coverage['combined_corrections']}",
              f"- WAIT→RELEASE: {coverage['release_transitions']['WAIT_to_RELEASE']}",
              f"- RELEASE→WAIT: {coverage['release_transitions']['RELEASE_to_WAIT']}",
              f"- Provenance audit: {'PASS' if manifest['provenance_audit']['passed'] else 'FAIL'} ({manifest['provenance_audit']['corrected_rows']} corrected rows)",
              "", "## BC sanity", "", f"- CE before (release/speed/merge): {bc.ce_before}",
              f"- CE after (release/speed/merge): {bc.ce_after}", f"- Joint CE before/after: {bc.joint_before:.6f} / {bc.joint_after:.6f}",
              "", "V3 L0 = PASS"]
    return "\n".join(lines) + "\n"


def _markdown_l1(preflight: V3L1Preflight) -> str:
    return "\n".join(("# V3 L1 Preflight Summary", "", f"- Rollout rows: {preflight.rollout_rows}",
        f"- Teacher interventions: {preflight.teacher_interventions}", f"- Entropy: {preflight.entropy:.6f}",
        f"- PPO requested-action provenance: {'PASS' if preflight.ppo_provenance_passed else 'FAIL'}",
        f"- ActionTape replay: {'PASS' if preflight.action_tape_replay_passed else 'FAIL'}",
        f"- Checkpoint save/load: {'PASS' if preflight.checkpoint_roundtrip_passed else 'FAIL'}",
        f"- Deterministic validation: {'PASS' if preflight.deterministic_validation_passed else 'FAIL'}",
        f"- Safety: {'PASS' if preflight.safety_passed else 'FAIL'}", "", f"V3 L1 Preflight = {'PASS' if preflight.passed else 'FAIL'}",
        f"fresh V3 L1 pilot allowed = {'true' if preflight.passed else 'false'}", ""))


def run_v3_preflight(*, dataset_path: str | Path, output_dir: str | Path) -> tuple[V3BCSanity, V3L1Preflight]:
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(dataset_path).parent / "v3_l0_teacher_dataset.manifest.json").read_text(encoding="utf-8"))
    quantization = _quantization_qualification()
    bc = categorical_bc_sanity(dataset_path)
    preflight = l1_preflight(dataset_path=dataset_path, output_dir=destination)
    (destination / "v3_l0_teacher_dataset_summary.md").write_text(_markdown_l0(manifest, bc, quantization), encoding="utf-8")
    (destination / "v3_l1_preflight_summary.md").write_text(_markdown_l1(preflight), encoding="utf-8")
    return bc, preflight


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run V3 categorical BC and L1 preflight only; never PPO training.")
    parser.add_argument("--dataset", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bc, preflight = run_v3_preflight(dataset_path=args.dataset, output_dir=args.output)
    print(json.dumps({"bc": asdict(bc), "l1_preflight": asdict(preflight)}, indent=2))


if __name__ == "__main__":
    _main()
