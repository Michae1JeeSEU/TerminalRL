import json
import math
from collections import Counter
from pathlib import Path
import sys

root=Path('D:/7_codespace/training_outputs/terminal/p3_v25_student_dev_20261008')/sys.argv[1]
result=json.loads((root/'result.json').read_text())
counts=Counter(); reasons=Counter(); interfaces=Counter(); ranks=Counter(); last=None; reward=0.
for tick,line in enumerate((root/'steps.jsonl').read_text().splitlines()):
    row=json.loads(line)
    assert row['tick']==tick
    if last is not None: assert last==row['pre_hash']
    last=row['post_hash'];reward+=row['reward'];counts['ticks']+=1
    assert set(row['tape']['actions'])==set(row['actor_mask'])==set(row['likelihoods'])
    for uid,actor in row['actor_mask'].items():
        action=row['tape']['actions'][uid]; lp=row['likelihoods'][uid]; trace=row['action_traces'][uid]
        assert trace['teacher_used'] is False and trace['teacher_corrected_action'] is None
        assert trace['requested_low_action']==action
        assert all(math.isfinite(v) for v in (lp['gate'],lp['joint']))
        gate=int(action[0]); assert gate in (0,1)
        assert lp['proposal'] is None if gate==0 else math.isfinite(lp['proposal'])
        assert abs(lp['joint']-lp['gate']-(lp['proposal'] or 0))<1e-6
        if not actor:
            assert action==[0,0,0] and lp['joint']==lp['gate']==0
            counts['forced']+=1
        else:
            counts['actor']+=1;counts['RELEASE' if gate else 'WAIT']+=1
            counts['shield_actor']+=bool(trace['shield_used'])
        if gate:
            shortlist=row['tape']['shortlists'][uid]
            assert len(shortlist)==3 and shortlist[0]==action
            rank=trace['runtime_shortlist_rank'];ranks[str(rank)]+=1
        telemetry=row['release_telemetry'].get(uid)
        if actor and telemetry:
            reasons[str(telemetry['primary_blocker'])]+=1
            interfaces[telemetry['interface']]+=1
assert counts['ticks']==result['ticks']
for key in ('actor','RELEASE','WAIT','shield_actor'): assert counts[key]==result['counts'].get(key,0)
assert abs(reward-result['reward_total'])<1e-6
print(json.dumps(dict(result=result,checked_counts=counts,reasons=reasons,interfaces=interfaces,ranks=ranks),indent=2))
