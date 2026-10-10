"""Apply the verified complete-row checks to the fresh v2 collection contract."""
from pathlib import Path
body=Path('.tmp/analyze_p3_v25_course_data.py').read_text(encoding='utf8')
body=body.replace("'train_dense24_mixed','train_dense24_fixed_annotation_v2'", "'train_dense24_mixed'")
body=body.replace("repaired=args.case=='train_dense24_fixed_annotation_v2'", 'repaired=False')
body=body.replace('p3_v25_neutral_course_20261008', 'p3_v25_neutral_course_v2_20261008')
body=body.replace("180 if smoke else 21600 if repaired or args.case=='train_dense24_mixed' else 7200", '180 if smoke else 21600')
body=body.replace("config=new_data.build_config(scenario)", "assert new_data.PROTOCOL=='p3_v25_neutral_course_data_20261008_v2'\nassert new_data.SMOKE_SEED==892863 and [r[0] for r in new_data.TRAIN_CASES]==[893511,893512,893513]\nconfig=new_data.build_config(scenario)")
exec(compile(body,__file__,'exec'))
