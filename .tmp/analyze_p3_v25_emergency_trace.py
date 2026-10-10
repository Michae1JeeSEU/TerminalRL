import json
import math
from collections import Counter
from pathlib import Path

root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_student_dev_20261008')
reports={}
for case in ('B_det','B_sto'):
    result=json.loads((root/case/'result.json').read_text())
    targets={str(r['uid']):r for r in result['energy_outcomes'] if r['arrival_id'] in (17,22)}
    stats={u:dict(outcome=o,counts=Counter(),reasons=Counter(),approvals=[],wait_p=[],wait_streaks=[],
                  first_tick=None,last_tick=None,streak_start=None,streak_len=0) for u,o in targets.items()}
    for line in (root/case/'steps.jsonl').open():
        r=json.loads(line)
        for uid,s in stats.items():
            action=r['tape']['actions'].get(uid)
            if action is None: continue
            tick=r['tick']; s['last_tick']=tick
            if s['first_tick'] is None: s['first_tick']=tick
            actor=r['actor_mask'][uid]
            release=action[0]>0
            s['counts']['actor' if actor else 'forced']+=1
            if actor: s['counts']['RELEASE' if release else 'WAIT']+=1
            if actor and not release:
                s['wait_p'].append(math.exp(r['likelihoods'][uid]['gate']))
                if not s['streak_len']: s['streak_start']=tick
                s['streak_len']+=1
            elif s['streak_len']:
                s['wait_streaks'].append([s['streak_start'],tick-1,s['streak_len']])
                s['streak_len']=0
            trace=r['release_telemetry'].get(uid)
            if actor and release:
                if trace is None: s['counts']['RELEASE_without_normal_release_telemetry']+=1
                elif trace.get('approval_time') is not None:
                    s['approvals'].append(dict(tick=tick,interface=trace['interface'],rank=trace.get('runtime_shortlist_rank'),
                        request=action,applied_speed=trace.get('applied_speed_residual'),
                        applied_timing=trace.get('applied_merge_time_residual'),
                        scheduled_delay=trace.get('scheduled_delay')))
                else: s['reasons'][str(trace.get('primary_blocker'))]+=1
    for s in stats.values():
        if s['streak_len']: s['wait_streaks'].append([s['streak_start'],s['last_tick'],s['streak_len']])
        s['longest_wait_streaks']=sorted(s.pop('wait_streaks'),key=lambda v:v[2],reverse=True)[:5]
        values=s.pop('wait_p');s['chosen_WAIT_probability']=dict(count=len(values),minimum=min(values),
             mean=sum(values)/len(values),maximum=max(values)) if values else None
        s.pop('streak_start');s.pop('streak_len')
    reports[case]=dict(ticks=result['ticks'],reward=result['reward_total'],
        total_energy=sum(r['energy_consumed'] for r in result['energy_outcomes']),targets=stats)
print(json.dumps(reports,indent=2))
