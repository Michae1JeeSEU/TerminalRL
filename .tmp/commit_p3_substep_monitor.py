from pathlib import Path
body=Path('.tmp/commit_p3_annotation_launch.py').read_text(encoding='utf8')
body=body.replace("== 'dbc2dd3296522530c1fa52d9f814aa0848d557f6'", ".startswith('cd80513')")
body=body.replace("files = ['docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']",
    "files = ['terminal_marl/core/runtime_separation.py','terminal_marl/envs/parallel_env.py','tests/test_executed_substep_separation.py','docs/evidence/v4_2/p3_v25_nominal_curve_fresh_training_20261008.md']")
body=body.replace('docs: record complete annotation repair launch and continuation',
                  'fix: measure separation over actual executor substeps')
exec(compile(body,__file__,'exec'))
