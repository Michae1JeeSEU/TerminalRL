from pathlib import Path
body=Path('.tmp/commit_p3_annotation_launch.py').read_text(encoding='utf8')
body=body.replace("== 'dbc2dd3296522530c1fa52d9f814aa0848d557f6'", ".startswith('66199f')")
files=['terminal_marl/algo/trajectory_set_actor.py','terminal_marl/algo/trajectory_set_ppo.py',
 'terminal_marl/core/resource_context_observation.py','terminal_marl/training/p3_v26_resource_inputs.py',
 'terminal_marl/training/p3_v26_course_data.py','terminal_marl/training/p3_v25_course_data.py',
 'terminal_marl/training/p3_v25_runtime.py','terminal_marl/training/p3_v25_rollout.py',
 'terminal_marl/training/p3_v25_training.py','terminal_marl/scripts/diagnose_p3_v25_train_generalization.py',
 'tests/test_resource_context_observation.py','tests/test_p3_v25_train_generalization.py',
 'tests/test_p3_v25_runtime.py','tests/test_trajectory_set_ppo.py','tests/test_p3_v26_course_data.py',
 'docs/evidence/v4_2/p3_v25_generalization_repair_20261009.md']
body=body.replace("files = ['docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']",'files = '+repr(files))
body=body.replace('docs: record complete annotation repair launch and continuation',
                  'fix: expose spatial pad resources and prepare independent P3 train cycles')
exec(compile(body,__file__,'exec'))
