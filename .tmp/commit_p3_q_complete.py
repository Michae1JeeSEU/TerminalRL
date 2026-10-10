from pathlib import Path
body=Path('.tmp/commit_p3_refined_student_result.py').read_text(encoding='utf8')
body=body.replace(".startswith('e340a89')", ".startswith('12b73d4')")
body=body.replace('docs: accept complete refined student cycle and freeze Q evaluation',
                  'docs: accept complete Q cohort for frozen candidate evaluation')
exec(compile(body,__file__,'exec'))
