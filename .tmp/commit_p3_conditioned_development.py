from pathlib import Path
body=Path('.tmp/commit_p3_ordered_development.py').read_text(encoding='utf8')
body=body.replace("startswith('4d60da1')", "startswith('bfe8034')")
body=body.replace('terminal_marl/algo/ordered_proposal_actor.py','terminal_marl/algo/conditioned_set_actor.py')
body=body.replace('tests/test_ordered_proposal_actor.py','tests/test_conditioned_set_actor.py')
body=body.replace('feat: test shared ordered proposal representation on complete train cycles',
                  'feat: test ego-conditioned neighbor encoding before aggregation')
exec(compile(body,__file__,'exec'))
