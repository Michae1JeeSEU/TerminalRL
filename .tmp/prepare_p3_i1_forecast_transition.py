"""Primary-authored pending transition; execute only with collection core idle."""
from pathlib import Path
p=Path('terminal_marl/training/p3_v25_imitation_qualification.py');s=p.read_text(encoding='utf8')
assert "PROTOCOL = 'p3_v25_fresh_i1_20261008_v1'" in s
s=s.replace('from ..configs.p3_integrated_repair import build_config','from ..configs.p3_integrated_repair import build_config, FORECAST_PROFILE')
s=s.replace('fit_arm, load_paired, synthetic, measure','fit_arm, synthetic, measure')
s=s.replace("PROTOCOL = 'p3_v25_fresh_i1_20261008_v1'", "PROTOCOL = 'p3_v25_fresh_i1_20261008_v2_forecast'\nGAP_MODE = 'committed_arrivals_v1'")
s=s.replace("and protocol['protocol'] == PROTOCOL, 'incomplete or wrong Q cohort')", "and protocol['protocol'] == PROTOCOL\n                and protocol['profile'] == FORECAST_PROFILE\n                and protocol['merge_gap_prediction_mode'] == GAP_MODE, 'incomplete or wrong Q cohort')")
marker='def fit(root, output, *, q_directories=(), smoke=False):'
assert marker in s
s=s.replace(marker,"""def load_training_inputs(root):
    # Lazy import avoids a dependency cycle with the shared reconstruction.
    from .p3_v25_forecast_data import load_train
    base = Path(root)/'p3_v25_forecast_train_20261008'
    return load_train([base/name for name in ('train_A','train_B','train_C')])


"""+marker)
s=s.replace('synthetic() if smoke else load_paired(root)','synthetic() if smoke else load_training_inputs(root)')
s=s.replace("observation_schema=VERSION, Q_row_access=False, frozen_Q_metadata=cohort", "observation_schema=VERSION, profile=FORECAST_PROFILE, merge_gap_prediction_mode=GAP_MODE,\n        Q_row_access=False, frozen_Q_metadata=cohort")
s=s.replace("protocol = dict(protocol=PROTOCOL, stage='collect_Q', seed=seed, partition=partition,", "protocol = dict(protocol=PROTOCOL, stage='collect_Q', seed=seed, partition=partition,\n        profile=FORECAST_PROFILE, merge_gap_prediction_mode=GAP_MODE,")
s=s.replace('build_config(requests=requests, warm_start=warm, high_soc=high)', 'build_config(requests=requests, warm_start=warm, high_soc=high, forecast_merge_gap=True)')
s=s.replace("report['reconstruction'] = rebuild_q(episode, seed=seed, partition=partition)", "report['reconstruction'] = rebuild_context(episode, seed=seed, partition=partition, expected_gap_mode=GAP_MODE)")
p.write_text(s,encoding='utf8')
p=Path('tests/test_p3_v25_imitation_qualification.py');s=p.read_text(encoding='utf8').replace("'load_paired'","'load_training_inputs'")
s=s.replace("dict(protocol=q.PROTOCOL,partition='Q',seed=seed)","dict(protocol=q.PROTOCOL,partition='Q',seed=seed,profile=q.FORECAST_PROFILE,merge_gap_prediction_mode=q.GAP_MODE)")
s=s.replace("assert seed == protocol['seed'] == q.SMOKE_SEED", "assert seed == protocol['seed'] == q.SMOKE_SEED\n        assert config.merge_gap_prediction_mode == protocol['merge_gap_prediction_mode'] == q.GAP_MODE")
p.write_text(s,encoding='utf8')
print('Prepared I1 transition applied; must test and update protocol before any Q or fit.')
