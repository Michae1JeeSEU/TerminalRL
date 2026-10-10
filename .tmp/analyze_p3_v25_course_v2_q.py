"""Same complete-record validation, explicitly using the evaluation decoder."""
from pathlib import Path
body=Path('.tmp/analyze_p3_v25_course_data.py').read_text(encoding='utf8')
body=body.replace("'train_dense24_mixed','train_dense24_fixed_annotation_v2'", "'train_dense24_mixed'")
body=body.replace("repaired=args.case=='train_dense24_fixed_annotation_v2'", 'repaired=False')
body=body.replace('p3_v25_neutral_course_20261008','p3_v25_neutral_course_v2_20261008')
body=body.replace('new_data.TRAIN_CASES','new_data.Q_CASES')
body=body.replace("180 if smoke else 21600 if repaired or args.case=='train_dense24_mixed' else 7200",'21600')
body=body.replace("partition=new_data.DEVELOPMENT_PARTITION if smoke else 'train'", "assert not smoke\npartition='Q'")
body=body.replace('config=new_data.build_config(scenario)',
    "assert new_data.PROTOCOL=='p3_v25_neutral_course_data_20261008_v2'\nassert [v[0] for v in new_data.Q_CASES]==[895501,895502,895503]\nconfig=new_data.build_config(scenario)")
body=body.replace('q.decode_evaluation_row(value,seed=seed,partition=partition) if smoke else new_data.decode_row(value)',
                  'q.decode_evaluation_row(value,seed=seed,partition=partition)')
for case in ('flow6_mixed','dense24_fixed','dense24_mixed'):
    body=body.replace('train_'+case,'Q_'+case)
body=body.replace("/args.case", "/('Q_substep_monitor_dense24_fixed' if args.case=='Q_dense24_fixed' else args.case)")
exec(compile(body,__file__,'exec'))
