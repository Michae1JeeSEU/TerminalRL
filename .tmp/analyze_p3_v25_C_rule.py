import argparse
from collections import Counter
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);assert not out.exists()
root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_student_dev_20261008')
case=root/'C_rule_same_seed';decisions=Counter();reasons=Counter();interfaces=Counter();rows=0
with (case/'blockers.jsonl').open(encoding='utf8') as stream:
    for line in stream:
        row=json.loads(line);rows+=1;assert row['ticks']==rows
        for release in row['releases']:
            decisions[release['decision']]+=1;reasons[str(release['primary_blocker'])]+=1
            interfaces[release['interface']]+=1
        if a.smoke:break
summary=dict(status='SCHEMA_SMOKE_OK' if a.smoke else 'FULL_ANALYSIS_COMPLETE',rows=rows,
             decisions=decisions,reasons=reasons,interfaces=interfaces)
if not a.smoke:
    result=json.loads((case/'result.json').read_text());summary['result']=result
    assert rows==result['ticks'] and row['outcomes']==result['outcomes']
    energy={x['arrival_id']:x for x in result['energy_outcomes']}
    assert len(energy)==len(result['energy_outcomes'])==24
    summary['rule_total_energy']=sum(x['energy_consumed'] for x in energy.values())
    summary['rule_minimum_soc']=min(x['soc'] for x in energy.values())
    summary['outcomes']=Counter(result['outcomes'].values());comparisons={}
    for name in ('C_det','C_sto'):
        other=json.loads((root/name/'result.json').read_text())
        protocol=json.loads((root/name/'protocol.json').read_text())
        physics=dict(protocol['physics_config'])
        # The two existing serializers spell enum DICT KEYS differently;
        # retain and compare every physical value after this exact mapping.
        aliases={'RingRole.OUTER':'outer','RingRole.MIDDLE':'middle','RingRole.INNER':'inner'}
        physics['rings']={aliases.get(k,k):v for k,v in physics['rings'].items()}
        assert physics==result['config']
        assert protocol['seed']==result['env_seed'] and protocol['policy_seed']==result['policy_seed']
        other_energy={x['arrival_id']:x for x in other['energy_outcomes']}
        assert set(other_energy)==set(energy)
        differences=[abs(energy[aid]['energy_remaining']+energy[aid]['energy_consumed']-
            other_energy[aid]['energy_remaining']-other_energy[aid]['energy_consumed']) for aid in energy]
        assert max(differences)<1e-6,'exogenous arrival initial energy differs'
        comparisons[name]=dict(ticks=other['ticks'],outcomes=Counter(other['outcomes'].values()),
            reward=other['reward_total'],total_energy=sum(x['energy_consumed'] for x in other_energy.values()),
            max_initial_energy_difference=max(differences),
            per_arrival=[dict(arrival_id=i,rule=energy[i]['outcome'],student=other_energy[i]['outcome'],
                rule_time=energy[i]['time'],student_time=other_energy[i]['time'],
                rule_soc=energy[i]['soc'],student_soc=other_energy[i]['soc']) for i in sorted(energy)])
    summary['same_config_seed_arrival_comparison']=comparisons
out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=summary['status'],rows=rows,output=str(out))))
