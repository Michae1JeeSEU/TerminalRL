from terminal_marl.training import p3_v25_fit_curve as curve


def test_fixed_curve_reproduces_prefix_and_retains_all_points(tmp_path):
    result=curve.run(tmp_path/'unused',tmp_path/'smoke',smoke=True)
    assert result['status']=='CURVE_COMPLETE' and result['smoke']
    fit=result['fits'][0]
    assert fit['updates']==3 and [c['update'] for c in fit['curve']]==[2,3]
    assert fit['curve'][0]['reference_prefix_exact']
    assert fit['final']==fit['curve'][-1]['metrics']
    assert (tmp_path/'smoke/seed_17/diagnostic_final.pt').exists()
