from pathlib import Path
body=Path('.tmp/commit_p3_annotation_launch.py').read_text(encoding='utf8')
body=body.replace("== 'dbc2dd3296522530c1fa52d9f814aa0848d557f6'", ".startswith('8901bfe')")
files=['terminal_marl/algo/intent_neighbor_actor.py','terminal_marl/training/p3_neighbor_witness_data.py',
       'terminal_marl/training/p3_neighbor_witness_dataset.py','terminal_marl/scripts/diagnose_p3_intent_neighbor.py',
       'tests/test_intent_neighbor_actor.py','tests/test_p3_neighbor_witness_data.py','tests/test_p3_neighbor_witness_dataset.py',
       'docs/evidence/v4_2/p3_neighbor_observability_repair_20261009.md',
       'docs/evidence/v4_2/p3_v25_generalization_repair_20261009.md']
body=body.replace("files = ['docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']",'files = '+repr(files))
body=body.replace('docs: record complete annotation repair launch and continuation',
                  'feat: add train-only native blocker supervision and relational actor diagnostic')
exec(compile(body,__file__,'exec'))
