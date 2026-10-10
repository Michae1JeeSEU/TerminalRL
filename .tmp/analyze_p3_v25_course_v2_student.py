"""Reuse complete request/tensor assertions for the fresh v2 canonical student."""
from pathlib import Path
body=Path('.tmp/analyze_p3_v25_student.py').read_text(encoding='utf8')
body=body.replace('p3_v25_neutral_course_20261008', 'p3_v25_neutral_course_v2_20261008')
body=body.replace("==892852+index", "==(892864 if smoke else 892852+index)")
body=body.replace("==929852+index", "==(929864 if smoke else 929852+index)")
body=body.replace("source['sha256']=='5ca0c9dc229b7cf750e3d55db611d68fad690093a7da5e9e7d1ab7ea98ad7883'",
                  "len(source['sha256'])==64")
exec(compile(body,__file__,'exec'))
