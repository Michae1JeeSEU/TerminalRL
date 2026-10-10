"""Only the failed development mixed cycle, unchanged actor/seeds, new output."""
from pathlib import Path
body=Path('.tmp/run_p3_v25_course_student.py').read_text(encoding='utf8')
body=body.replace("choices=['smoke','flow6_mixed','dense24_fixed','dense24_mixed']", "choices=['dense24_mixed']")
body=body.replace('p3_v25_neutral_course_20261008', 'p3_v25_neutral_course_v2_20261008')
body=body.replace('docs/evidence/v4_2/p3_v25_neutral_course_v2_20261008.md',
                  'docs/evidence/v4_2/p3_v25_nominal_curve_fresh_training_20261008.md')
body=body.replace("out=base+'/student_initial/'+a.case", "out=base+'/student_runtime_refined/'+a.case")
body=body.replace("q(base+'/student_initial')", "q(base+'/student_runtime_refined')")
body=body.replace("'terminal_marl/core/safety_filter.py'", "'terminal_marl/core/safety_filter.py','terminal_marl/core/runtime_traffic_safety.py'")
# This is an explicitly paired development rerun, never an untouched Q seed.
body=body.replace('assert not collisions', "assert collisions, 'failed original development evidence must exist'")
exec(compile(body,__file__,'exec'))
