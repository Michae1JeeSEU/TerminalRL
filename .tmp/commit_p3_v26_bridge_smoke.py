from pathlib import Path
body=Path('.tmp/commit_p3_annotation_launch.py').read_text(encoding='utf8')
body=body.replace("== 'dbc2dd3296522530c1fa52d9f814aa0848d557f6'", ".startswith('ea0ed80')")
body=body.replace("files = ['docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']",
 "files = ['terminal_marl/scripts/evaluate_p3_v25_student.py','tests/test_p3_v25_runtime.py','docs/evidence/v4_2/p3_v25_generalization_repair_20261009.md']")
body=body.replace("assert not git('diff', '--cached', '--name-only').strip()",
 "assert set(git('diff', '--cached', '--name-only').decode().splitlines()) == {'terminal_marl/scripts/evaluate_p3_v25_student.py','tests/test_p3_v25_runtime.py','docs/evidence/v4_2/p3_v25_generalization_repair_20261009.md'}")
body=body.replace('docs: record complete annotation repair launch and continuation',
                  'fix: record actual set observation schema and verify resource rollout bridge')
exec(compile(body,__file__,'exec'))
