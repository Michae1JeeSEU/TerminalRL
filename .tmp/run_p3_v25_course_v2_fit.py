"""Same fixed fitter, fresh corrected-contract input directories only."""
from pathlib import Path
body=Path('.tmp/run_p3_v25_course_fit.py').read_text(encoding='utf8')
body=body.replace('p3_v25_neutral_course_20261008', 'p3_v25_neutral_course_v2_20261008')
# The fresh cycle is complete in its original directory; there is no repair arm.
body=body.replace('train_dense24_fixed_annotation_v2', 'train_dense24_fixed')
# The evidence source name did not change with the output root.
body=body.replace('docs/evidence/v4_2/p3_v25_neutral_course_v2_20261008.md',
                  'docs/evidence/v4_2/p3_v25_nominal_curve_fresh_training_20261008.md')
exec(compile(body,__file__,'exec'))
