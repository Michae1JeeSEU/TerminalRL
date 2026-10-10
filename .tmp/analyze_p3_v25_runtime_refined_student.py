from pathlib import Path
body=Path('.tmp/analyze_p3_v25_course_v2_student.py').read_text(encoding='utf8')
# The v2 analyzer itself compiles the common student analyzer with a new root.
body=body.replace("exec(compile(body,__file__,'exec'))", "body=body.replace('student_initial','student_runtime_refined')\nexec(compile(body,__file__,'exec'))")
exec(compile(body,__file__,'exec'))
