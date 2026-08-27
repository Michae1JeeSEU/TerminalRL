"""Training entry points with dependency-scoped, lazy public exports.

Calibration and data-collection stages intentionally have a smaller runtime
dependency set than PPO.  Importing ``terminal_marl.training`` must therefore
not require PyTorch before C0/L0 can establish their prerequisites.
"""
from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CurriculumStage": (".train_low_level_ppo", "CurriculumStage"),
    "train_shared_low_level": (".train_low_level_ppo", "train_shared_low_level"),
    "assess_a5_gate": (".evaluate_low_level", "assess_a5_gate"),
    "compare_fixed_rule_and_low_rl": (".evaluate_low_level", "compare_fixed_rule_and_low_rl"),
    "TeacherDatasetCohort": (".build_low_teacher_dataset", "TeacherDatasetCohort"),
    "TeacherDatasetMatrixSummary": (".build_low_teacher_dataset", "TeacherDatasetMatrixSummary"),
    "TeacherDatasetSummary": (".build_low_teacher_dataset", "TeacherDatasetSummary"),
    "build_low_teacher_dataset": (".build_low_teacher_dataset", "build_low_teacher_dataset"),
    "build_low_teacher_dataset_matrix": (".build_low_teacher_dataset", "build_low_teacher_dataset_matrix"),
    "LowV2TrainingPreflight": (".train_low_v2_ppo", "LowV2TrainingPreflight"),
    "authorize_low_v2_training": (".train_low_v2_ppo", "authorize_low_v2_training"),
    "prepare_low_v2_training": (".train_low_v2_ppo", "prepare_low_v2_training"),
    "StudentQualificationRecord": (".eval_low_v2_student", "StudentQualificationRecord"),
    "StudentQualificationReport": (".eval_low_v2_student", "StudentQualificationReport"),
    "qualify_student_records": (".eval_low_v2_student", "qualify_student_records"),
    "TeacherDatasetAudit": (".audit_low_teacher_dataset", "TeacherDatasetAudit"),
    "audit_teacher_dataset": (".audit_low_teacher_dataset", "audit_teacher_dataset"),
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
