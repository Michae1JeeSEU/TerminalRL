from pathlib import Path
body=Path('.tmp/run_p3_v26_resources.py').read_text(encoding='utf8')
body=body.replace('p3_v26_resources_20261009','p3_v26_supplemental_20261009')
body=body.replace('p3_v26_course_data','p3_v26_supplemental_data')
body=body.replace("tests=['tests/test_p3_v26_supplemental_data.py'",
                  "tests=['tests/test_p3_v26_course_data.py','tests/test_p3_v26_supplemental_data.py'")
body=body.replace("files=[", "files=['terminal_marl/training/p3_v26_course_data.py',")
body=body.replace('892871','892872').replace('893600+index','893603+index')
exec(compile(body,__file__,'exec'))
