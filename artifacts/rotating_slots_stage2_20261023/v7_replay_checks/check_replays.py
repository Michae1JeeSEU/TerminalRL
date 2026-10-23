"""Read-only v7 load/seek and production viewer offscreen evidence."""
from pathlib import Path
import bisect, collections, gc, json, sys, traceback
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from terminal_marl.core.rotating_slots.replay import ReplayBundle,_plain
from terminal_marl.visualization.rotating_slots_replay import ReplayController,RotatingSlotsReplayViewer
from terminal_marl.visualization.rotating_slots_controller import _recorded_rescue_status,_INVISIBLE
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent
ROOT=Path('D:/7_codespace/training_outputs/terminal')
PILOT='--pilot' in sys.argv
RESULT={'human_acceptance':'HUMAN_PENDING','visual_review':'HUMAN_PENDING','python':sys.executable,'viewer_head_observed_by_parent':'217c1f4','experiment_frozen_commit_prefix':'4da','viewer_note':'Current viewer contains subsequent UI refactor; frozen simulation results unchanged.','selection_definition':'actual event timestamp with sampled frame >=1 WAIT/RESERVED and >=1 executing FINAL_NORMAL plan','bundles':[],'launch_commands':[],'missing_event_evidence':[]}
def save():
 (OUT/('pilot.json' if PILOT else 'check.json')).write_text(json.dumps(RESULT,indent=2,ensure_ascii=False),encoding='utf-8')
def statuses(c,f):
 return {str(a['id']):_recorded_rescue_status(_plain(a),f,c._plans) for a in f['aircraft']}
def normals(c,f):
 return [str(a['id']) for a in f['aircraft'] if a.get('status') in ('FINAL_NORMAL','VERTICAL_DESCENT') and c._plans.get(a.get('trajectory_hash')) is not None and c._plans[a['trajectory_hash']].kind=='FINAL_NORMAL']
def describe(c,t):
 s=c.seek(t); f=c.sampled_state(); raw=_plain(f['aircraft']); aircraft=c.aircraft_at(); pads=c.pads_at(); rescue=statuses(c,f)
 counts={k:sum(v==k for v in rescue.values()) for k in ('WAIT','RESERVED','ACTIVE')}
 ground=[{'pad_id':p['id'],'aircraft_id':p['occupant_id'],'service_end':p['service_end'],'service_remaining':p['service_remaining']} for p in pads if p['occupant_id'] is not None]
 future=[{'pad_id':p['id'],**{k:r[k] for k in ('owner_id','commitment','maneuver_kind','maneuver_start','interval_start','interval_end')}} for p in pads for r in p['reservations'] if r['future_interval']]
 n=normals(c,f); visible=sum(a.get('status') not in _INVISIBLE for a in raw)
 counts.update(record_aircraft=len(raw),controller_aircraft=len(aircraft),record_visible=visible,snapshot_uavs=len(s.uavs),record_service=sum(a.get('status') in ('SERVICE','LANDED_SERVICE') for a in raw),snapshot_service=s.service_count,snapshot_completed=s.completed_count,active_normal_finals=len(n),ground_service=len(ground),future_pad_reservations=len(future))
 labels=[{'aircraft_id':str(a['id']),'uid':c.uid_by_id[str(a['id'])],'status':a.get('status'),'soc':a['soc'],'rescue_label':rescue[str(a['id'])],'warning':a.get('emergency',{}).get('warning'),'declared':a.get('emergency',{}).get('declared')} for a in aircraft if rescue[str(a['id'])] is not None]
 return {'requested_time':t,'controller_time':c.time,'sampled_time':f['time'],'counts':counts,'checks':{'controller_count_matches':len(raw)==len(aircraft),'visible_count_matches':len(s.uavs)==visible,'service_count_matches':s.service_count==len(ground),'time_matches':s.time==c.time,'sample_not_after_query':f['time']<=c.time},'status_counts':dict(collections.Counter(a.get('status') for a in aircraft)),'rescue_labels':labels,'active_normal_final_ids':n,'physical_ground_service':ground,'future_pad_reservations':future,'hud_lines':c.hud_lines()}
def adjacent(c,event):
 t=float(event['time']); i=bisect.bisect_left(c._sample_times,t); frames=[]
 for j in (i-1,i):
  if 0<=j<len(c._sample_times):
   f=c.bundle.data['states'][j]; a=next((a for a in f['aircraft'] if str(a['id'])==str(event['aircraft_id'])),None)
   frames.append({'time':c._sample_times[j],'aircraft':None if a is None else {'id':a['id'],'status':a.get('status'),'soc':a.get('soc'),'emergency':_plain(a.get('emergency'))}})
 return {'event':event,'exact_frame_present':t in c._sample_times,'adjacent_frames':frames}
def coexist(c,events,scene):
 for t in sorted({float(e['time']) for e in events}):
  if scene in ('stress','diagnostic') and t<240: continue
  f=c.bundle.data['states'][max(0,bisect.bisect_right(c._sample_times,t)-1)]; r=statuses(c,f)
  if any(s in ('WAIT','RESERVED') for s in r.values()) and normals(c,f):
   return {'time':t,'sampled_time':f['time'],'event':next(e for e in events if e['time']==t),'selected_id':next(k for k,v in r.items() if v in ('WAIT','RESERVED'))}
 return None
def render(c,scene,method,targets):
 shots=[]; viewer=None
 try:
  c.seek(targets[0][1]); viewer=RotatingSlotsReplayViewer(c,off_screen=True); viewer._plotter.window_size=(1600,860)
  for label,t,identity,event in targets:
   c.select(identity); obs=describe(c,t); viewer.draw(); viewer.set_view('3d',vertical_exaggeration=1.); viewer.draw()
   p=OUT/f'{scene}_{method}_{label}.png'; viewer._plotter.screenshot(str(p))
   with Image.open(p) as im: im.load(); dimensions=im.size
   shots.append({'path':str(p),'label':label,'time':c.time,'selected_id':identity,'event':event,'image_load':'PASS','size':dimensions,'observation':obs})
   print('IMAGE',p.name,f'{c.time:.6f}',dimensions,flush=True)
 finally:
  if viewer is not None: viewer.close()
 return shots
for scene in (('stress',) if PILOT else ('stress','diagnostic','medium')):
 for method in (('energy_slack',) if PILOT else ('energy_slack','rh_cp_sat')):
  path=ROOT/f'stage2_rescue_need_v7_{scene}_600s_20261023'/method/'replay_0600'; b={'scene':scene,'method':method,'path':str(path),'queries':[],'screenshots':[]}; RESULT['bundles'].append(b)
  print('START',scene,method,flush=True); bundle=c=None
  try:
   bundle=ReplayBundle.load(path); c=ReplayController(bundle); events=_plain(bundle.data['events'])
   declarations=[e for e in events if e['type']=='emergency_declared']; starts=[e for e in events if e['type']=='maneuver_started' and e.get('details',{}).get('maneuver_kind')=='FINAL_EMERGENCY']; first=declarations[0] if declarations else None
   audit_path=path.parent/'audit_0600.json'; audit=json.loads(audit_path.read_text(encoding='utf-8'))
   b.update(load='PASS',time_bounds=bundle.time_bounds,states_count=len(bundle.data['states']),code_commit=bundle.index.get('code_commit'),human_acceptance=bundle.index.get('human_acceptance'),audit_path=str(audit_path),audit_status=audit.get('status'),audit_issues=audit.get('issues',[]),first_emergency=first,first_rescue_start=starts[0] if starts else None)
   for issue in audit.get('issues',[]):
    if issue.get('code')=='EVENT_FRAME_MISSING':
     matching=[e for e in events if e['type']==issue['event_type'] and e['time']==issue['time']]
     RESULT['missing_event_evidence'].append({'scene':scene,'method':method,'audit_path':str(audit_path),'audit_issue':issue,'matching_events':[adjacent(c,e) for e in matching]})
   times=[0.,240.,480.,600.]
   if first: times += [first['time'],min(600.,first['time']+1)]
   if starts: times += [starts[0]['time'],min(600.,starts[0]['time']+1)]
   b['queries']=[describe(c,t) for t in sorted(set(times))]; b['coexistence']=coexist(c,events,scene)
   RESULT['launch_commands'].append({'scene':scene,'method':method,'powershell':'& D:\\7_codespace\\terminal\\venv\\Scripts\\python.exe -c "from terminal_marl.visualization.rotating_slots_replay import ReplayController, RotatingSlotsReplayViewer; RotatingSlotsReplayViewer(ReplayController(r\''+str(path)+'\')).run()"'})
   if (scene,method) in (('stress','energy_slack'),('medium','rh_cp_sat')):
    targets=[]; co=b['coexistence']
    if co: targets.append(('coexist_wait_reserved_normal',co['time'],co['selected_id'],co['event']))
    if starts:
     start=starts[0]; targets.append(('rescue_start_plus001',min(600.,start['time']+.01),start['aircraft_id'],start))
     touch=next((e for e in events if e['type']=='touchdown' and e.get('aircraft_id')==start['aircraft_id'] and e['time']>=start['time']),None)
     if touch: targets.append(('rescue_touchdown_plus001',min(600.,touch['time']+.01),touch['aircraft_id'],touch))
    targets.append(('end600',600.,co['selected_id'] if co else None,None)); b['screenshots']=render(c,scene,method,targets); b['render']='PASS'
   print('DONE',scene,method,len(b['queries']),flush=True)
  except Exception:
   b['error']=traceback.format_exc(); print(b['error'],flush=True); save(); raise
  finally:
   bundle=c=None; gc.collect(); save()
RESULT['all_query_checks_pass']=all(all(q['checks'].values()) for b in RESULT['bundles'] for q in b['queries'])
RESULT['image_count']=sum(len(b['screenshots']) for b in RESULT['bundles'])
if not PILOT:
 shots=[(b,s) for b in RESULT['bundles'] for s in b['screenshots']]; contact=Image.new('RGB',(1600,900*len(shots)),'white')
 for i,(b,s) in enumerate(shots):
  with Image.open(s['path']) as im: contact.paste(im.convert('RGB'),(0,i*900+40))
  ImageDraw.Draw(contact).text((12,i*900+10),f"{b['scene']} / {b['method']} / {s['label']} t={s['time']:.3f}s | HUMAN_PENDING",fill='black')
 cp=OUT/'contactsheet.png'; contact.save(cp); RESULT['contactsheet']=str(cp)
save()
print(json.dumps({'query_checks':RESULT['all_query_checks_pass'],'image_count':RESULT['image_count'],'missing_event_count':len(RESULT['missing_event_evidence'])}),flush=True)
