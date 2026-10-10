from pathlib import Path
body=Path('.tmp/commit_p3_annotation_launch.py').read_text(encoding='utf8')
body=body.replace("== 'dbc2dd3296522530c1fa52d9f814aa0848d557f6'", ".startswith('b5bb744')")
body=body.replace("files = ['docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']",
 "files = ['terminal_marl/configs/p3_v25_course.py','terminal_marl/training/p3_v26_course_data.py','terminal_marl/training/p3_v26_supplemental_data.py','tests/test_p3_v26_supplemental_data.py','docs/evidence/v4_2/p3_v25_generalization_repair_20261009.md']")
body=body.replace('docs: record complete annotation repair launch and continuation',
                  'feat: declare bounded medium and dense supplemental training cycles')
exec(compile(body,__file__,'exec'))
