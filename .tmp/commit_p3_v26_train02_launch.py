from pathlib import Path
body=Path('.tmp/commit_p3_annotation_launch.py').read_text(encoding='utf8')
body=body.replace("== 'dbc2dd3296522530c1fa52d9f814aa0848d557f6'", ".startswith('74c1dd5')")
body=body.replace("files = ['docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']",
 "files = ['docs/evidence/v4_2/p3_v25_generalization_repair_20261009.md']")
body=body.replace('docs: record complete annotation repair launch and continuation',
                  'docs: accept resource train flow cycle and record dense fixed launch')
exec(compile(body,__file__,'exec'))
