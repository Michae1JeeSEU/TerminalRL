from pathlib import Path
body=Path('.tmp/commit_p3_refined_student_result.py').read_text(encoding='utf8')
body=body.replace(".startswith('e340a89')", ".startswith('0ee4574')")
body=body.replace('docs: accept complete refined student cycle and freeze Q evaluation',
                  'docs: record complete first independent Q collection')
exec(compile(body,__file__,'exec'))
