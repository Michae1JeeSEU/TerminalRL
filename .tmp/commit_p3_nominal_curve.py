from pathlib import Path
body=Path('.tmp/commit_p3_annotation_launch.py').read_text(encoding='utf8')
body=body.replace('dbc2dd3296522530c1fa52d9f814aa0848d557f6','f60a4aab081f5a30c18fbc32e944e16aec518fdf')
body=body.replace("files = ['docs/evidence/v4_2/p3_v25_neutral_course_20261008.md']",'files = '+repr([
    'docs/evidence/v4_2/p3_v25_neutral_course_20261008.md','docs/requirements.md','docs/technical_spec.md',
    'terminal_marl/core/config.py','terminal_marl/core/priority.py','terminal_marl/core/safety_filter.py',
    'terminal_marl/configs/p3_v25_course.py','terminal_marl/training/p3_v25_course_data.py',
    'terminal_marl/scripts/diagnose_p3_teacher_context.py','tests/test_nominal_holder_curve_certificate.py']))
body=body.replace('docs: record complete annotation repair launch and continuation','fix: certify nominal holder curves and include internal waiting')
exec(compile(body,__file__,'exec'))
