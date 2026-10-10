from pathlib import Path
body=Path('.tmp/commit_p3_annotation_launch.py').read_text(encoding='utf8')
body=body.replace("== 'dbc2dd3296522530c1fa52d9f814aa0848d557f6'", ".startswith('49994cb')")
body=body.replace("files = ['docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']",
    "files = ['terminal_marl/scripts/diagnose_p3_v25_train_generalization.py','terminal_marl/scripts/diagnose_p3_v25_lz_alias.py','tests/test_p3_v25_train_generalization.py','tests/test_p3_v25_lz_alias.py','docs/evidence/v4_2/p3_v25_generalization_repair_20261009.md']")
body=body.replace('docs: record complete annotation repair launch and continuation',
                  'diagnose: isolate P3 cross-cycle overfit and missing pad context')
exec(compile(body,__file__,'exec'))
