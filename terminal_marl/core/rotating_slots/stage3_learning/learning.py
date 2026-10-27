"""One guarded formal CPU training run using separated approved preference pairs."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from .features import SCHEMA_ID, SCHEMA_V2
from .model import ARCHITECTURE_ID, FeatureNormalizer, build_model, fingerprint_compatible, cost_aware_scores

SEED = 20261023
LABEL_POLICY = 'bounded_reference_supported_pair_v1_0'
PAIR_SOURCES = {'supported_vs_certified_single_invalid','bounded_conditional_certified_objective_preference'}
JOINT_LABEL_POLICY = 'bounded_joint_contribution_v2'
JOINT_PAIR_SOURCES = {'observed_bounded_joint_contribution_preference',
    'observed_bounded_joint_contribution_measured_cost_preference'}


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
    os.replace(tmp,path)


def _read(value):
    return json.loads(Path(value).read_text(encoding='utf-8-sig')) if isinstance(value,(str,Path)) else value


def _groups(dataset, fingerprint, *, schema_id=SCHEMA_ID):
    """Reject sealed/test/development input rather than consulting it for selection."""
    if not fingerprint_compatible(dataset.get('fingerprint',{}),fingerprint):
        raise ValueError('dataset fingerprint mismatch')
    contexts = dataset.get('contexts',[])
    if any(c.get('split') not in ('train','validation') for c in contexts):
        raise ValueError('training input may contain only train/validation contexts; keep test sealed')
    by_id = {}
    roots = {}
    for c in contexts:
        if not fingerprint_compatible(c.get('fingerprint',{}),fingerprint):
            raise ValueError('context fingerprint mismatch')
        if c['context_id'] in by_id:
            raise ValueError('duplicate context_id')
        root = c['ancestry_root_id']
        if root in roots and roots[root] != c['split']:
            raise ValueError('ancestry crosses training split')
        roots[root] = c['split']
        features = {f['descriptor_id']:f for f in c['features']}
        if len(features) != len(c['features']):
            raise ValueError('duplicate descriptor_id in context')
        by_id[c['context_id']] = dict(context=c,features=features,pairs=[],facts={})
        for fact in c.get('facts',[]):
            if not fingerprint_compatible(fact.get('fingerprint',{}),fingerprint):
                raise ValueError('fact fingerprint mismatch')
            if fact.get('context_id') != c['context_id'] or fact.get('ancestry_root_id') != root:
                raise ValueError('fact context or ancestry mismatch')
            did = fact['descriptor_id']
            if did in features:
                by_id[c['context_id']]['facts'][did] = fact
    for p in dataset.get('preference_pairs',[]):
        if p.get('split') not in ('train','validation'):
            raise ValueError('sealed/test pair found in training input')
        sources = PAIR_SOURCES | ({'equal_quality_incremental_preference','equal_quality_measured_cost_preference'} if schema_id == SCHEMA_V2 else set())
        future = schema_id == SCHEMA_V2 and fingerprint.get('label_policy_id') == JOINT_LABEL_POLICY
        if p.get('label_source') not in sources | (JOINT_PAIR_SOURCES if future else set()) or not p.get('evidence_refs'):
            raise ValueError('unapproved preference pair')
        if p.get('objective_id') != fingerprint.get('objective_id'):
            raise ValueError('preference objective mismatch')
        group = by_id.get(p.get('context_id'))
        if group is None:
            raise ValueError('pair without context')
        c = group['context']
        if future:
            if p.get('label_policy_id') == LABEL_POLICY:
                provenance=c.get('migration_provenance',{})
                if (c['split']!='train' or c.get('feature_state_semantics')!='CURRENT_STATE_V2' or
                        not provenance.get('source_fingerprint') or
                        provenance.get('source_feature_state_semantics')!='CURRENT_STATE_V2' or
                        p.get('label_source') not in sources):
                    raise ValueError('legacy pair requires explicit old-current train migration')
            elif p.get('label_policy_id') == JOINT_LABEL_POLICY:
                if c.get('feature_state_semantics')!='planning_effective_state_v1':
                    raise ValueError('new pair requires projected future feature state')
                if p.get('label_source') in JOINT_PAIR_SOURCES:
                    # Reconstruct the approved pair from the exact four archived
                    # inclusion/removal solves, selected facts and increments.
                    from .labeling import joint_contribution_pairs, canonical_solution_id
                    solutions=c.get('contribution_solutions',[])
                    if any(s.get('solution_id')!=canonical_solution_id(s) for s in solutions):
                        raise ValueError('contribution solve content identity mismatch')
                    approved=joint_contribution_pairs(c.get('descriptor_records',[]),c.get('facts',[]),solutions,
                        c.get('incremental_contribution_evidence',[]),root_id=c['ancestry_root_id'],split=c['split'],fingerprint=fingerprint)
                    if p not in approved or len(p['evidence_refs'])!=4 or len(set(p['evidence_refs']))!=4:
                        raise ValueError('unverified four-solve contribution evidence')
                elif p.get('label_source') not in PAIR_SOURCES:
                    raise ValueError('unapproved new-policy preference source')
            else:
                raise ValueError('unapproved preference pair policy')
        elif p.get('label_policy_id') != LABEL_POLICY:
            raise ValueError('unapproved preference pair')
        if p['split'] != c['split'] or p['ancestry_root_id'] != c['ancestry_root_id']:
            raise ValueError('preference split/ancestry mismatch')
        a,b = p['preferred_descriptor'],p['other_descriptor']
        if a not in group['features'] or b not in group['features'] or a == b:
            raise ValueError('invalid preference descriptor')
        pair = (a,b)
        if pair not in group['pairs']:
            group['pairs'].append(pair)
    return list(by_id.values())


def auxiliary_targets(fact, *, schema_id=SCHEMA_ID):
    """Missing, unattempted, timeout and background conflicts are never fake negatives."""
    outcome = fact.get('outcome')
    validity = 1. if outcome == 'SINGLE_VALID' else 0. if outcome == 'SINGLE_INVALID_CERTIFIED' else None
    if fact.get('validity_label_mask') is False:
        validity = None
    wall = fact.get('wall_s')
    cost = None
    if outcome not in ('TIMEOUT','NOT_ATTEMPTED') and not fact.get('censored',False) and wall is not None and math.isfinite(wall) and wall >= 0:
        cost = math.log1p(wall*1000.)
    if schema_id == SCHEMA_V2 and (not fact.get('cost_complete') or fact.get('cost_scope') != 'GENERATOR_ENERGY_TERMINAL_VALIDATION_V2'):
        cost = None
    return validity,cost


def _loss(model, groups, normalizer, rng=None, rank_only=False):
    import torch
    from torch.nn import functional as F
    rank_losses, total_losses = [], []
    for group in groups:
        pairs = list(group['pairs'])
        v2 = getattr(normalizer,'schema_id',SCHEMA_ID) == SCHEMA_V2
        if not pairs and (rank_only or not v2):
            continue
        if rng is not None and len(pairs) > 64:
            pairs = rng.sample(pairs,64)
        # Validation uses all approved pairs. Forward is chunked to ready-size 64.
        ids = list(group['features'])
        predictions = {}
        for start in range(0,len(ids),64):
            chunk = ids[start:start+64]
            out = model([normalizer.transform(group['features'][did]) for did in chunk])
            predictions.update(zip(chunk,out.unbind(0)))
        rank = torch.stack([F.softplus(-(predictions[a][0]-predictions[b][0])) for a,b in pairs]).mean() if pairs else next(iter(predictions.values()))[0]*0.
        if pairs:
            rank_losses.append(rank)
        valid_losses,cost_losses = [],[]
        if not rank_only:
            for did,fact in group['facts'].items():
                v,c = auxiliary_targets(fact,schema_id=getattr(normalizer,'schema_id',SCHEMA_ID))
                if v is not None:
                    valid_losses.append(F.binary_cross_entropy_with_logits(predictions[did][1],torch.tensor(v)))
                if c is not None:
                    cost_losses.append(F.huber_loss(predictions[did][2],torch.tensor(c),delta=1.))
        zero = rank*0.
        total_losses.append(rank+.1*(torch.stack(valid_losses).mean() if valid_losses else zero)+.1*(torch.stack(cost_losses).mean() if cost_losses else zero))
    if not total_losses:
        return None,None
    return torch.stack(total_losses).mean(),torch.stack(rank_losses).mean() if rank_losses else None


def _fixed_config(config):
    if config is None:
        return
    if isinstance(config,(str,Path)):
        import yaml
        config = yaml.safe_load(Path(config).read_text(encoding='utf-8-sig'))
    training = config.get('training',{})
    expected = dict(seed=SEED,formal_runs_max=1,optimizer='AdamW',learning_rate=.001,weight_decay=.0001,
        max_epochs=30,batch_context_groups_max=16,pairs_per_group_per_batch_max=64,gradient_clip_norm=1.,early_stopping_patience=5)
    if any(training.get(k) != v for k,v in expected.items()):
        raise ValueError('training configuration differs from frozen contract')
    model = config.get('model',{})
    if model.get('architecture_id') != ARCHITECTURE_ID:
        raise ValueError('model configuration differs from frozen contract')


def train_model(dataset_path, output_dir, fingerprint, *, config=None, deadline=None, schema_id=SCHEMA_ID):
    """Return DATA_INSUFFICIENT before torch/guard when either split lacks rank labels.

    deadline is the campaign's absolute Unix wall deadline, never a new budget.
    A parent-directory exclusive guard survives failure and forbids a second run.
    """
    output = Path(output_dir)
    if (output.parent/'heldout_access.json').exists():
        raise ValueError('heldout access already recorded; formal training forbidden')
    _fixed_config(config)
    dataset, binding = _read(dataset_path), _read(fingerprint)
    fingerprint = binding.get('fingerprint',binding)
    groups = _groups(dataset,fingerprint,schema_id=schema_id)
    train = eligible_training_groups(groups,schema_id=schema_id)
    validation = [g for g in groups if g['context']['split'] == 'validation' and g['pairs']]
    output.mkdir(parents=True,exist_ok=True)
    counts = dict(train_pairs=sum(len(g['pairs']) for g in train),validation_pairs=sum(len(g['pairs']) for g in validation),
        train_groups=len(train),validation_groups=len(validation))
    if not train or not validation or not counts['train_pairs'] or not counts['validation_pairs']:
        result = dict(status='DATA_INSUFFICIENT',formal_training_started=False,**counts)
        atomic_json(output/'model_card.json',result)
        return result
    if deadline is not None and time.time() >= deadline:
        result = dict(status='BUDGET_LIMITED',formal_training_started=False,**counts)
        atomic_json(output/'model_card.json',result)
        return result
    # All train contexts may contribute visible moments; validation contributes none.
    normalizer = FeatureNormalizer(schema_id=schema_id).fit([g['context'] for g in groups if g['context']['split'] == 'train'])
    cost_statistics = fit_cost_statistics(train) if schema_id == SCHEMA_V2 else None
    if schema_id == SCHEMA_V2 and not cost_statistics['count']:
        result = dict(status='DATA_INSUFFICIENT',reason='NO_COMPLETE_V2_TRAIN_COST',formal_training_started=False,**counts)
        atomic_json(output/'model_card.json',result)
        return result
    guard = output.parent/'formal_training.guard.json'
    descriptor = os.open(guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w',encoding='utf8') as handle:
        json.dump(dict(seed=SEED,output=str(output.resolve()),started_unix=time.time()),handle)
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model = build_model(schema_id)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[]; best=float('inf'); best_state=None; best_epoch=None; patience=0; limited=False
    start = time.time()
    for epoch in range(30):
        model.train(); order=list(train); rng.shuffle(order); train_values=[]
        for offset in range(0,len(order),16):
            if deadline is not None and time.time() >= deadline:
                limited=True; break
            optimizer.zero_grad(set_to_none=True)
            loss,_ = _loss(model,order[offset:offset+16],normalizer,rng)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            train_values.append(float(loss.detach()))
        if limited:
            break
        model.eval()
        with torch.no_grad():
            rank_values=[]
            for offset in range(0,len(validation),16):
                if deadline is not None and time.time() >= deadline:
                    limited=True; break
                _,rank = _loss(model,validation[offset:offset+16],normalizer,rank_only=True)
                rank_values.extend([float(rank)]*len(validation[offset:offset+16]))
        if limited:
            break
        val = math.fsum(rank_values)/len(rank_values)
        if not math.isfinite(val):
            raise ValueError('nonfinite validation loss')
        history.append(dict(epoch=epoch+1,train_loss=math.fsum(train_values)/len(train_values),validation_rank_loss=val,elapsed_s=time.time()-start))
        temp = output/'training_history.jsonl.tmp'
        temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history),encoding='utf8')
        os.replace(temp,output/'training_history.jsonl')
        if val < best:
            best=val; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1; patience=0
        else:
            patience+=1
        if patience >= 5:
            break
    result = dict(status='BUDGET_LIMITED' if limited else 'TRAINED_PILOT',formal_training_started=True,
        seed=SEED,architecture_id=ARCHITECTURE_ID,parameter_count=sum(p.numel() for p in model.parameters()),
        epochs_completed=len(history),best_epoch=best_epoch,validation_rank_loss=best if best_state else None,
        elapsed_s=time.time()-start,model_frozen=best_state is not None,**counts)
    if best_state is not None:
        payload = dict(architecture_id=ARCHITECTURE_ID,schema_id=schema_id,fingerprint=fingerprint,
            state_dict=best_state,normalizer=normalizer.as_dict(),selection_metric='validation_rank_loss',best_epoch=best_epoch)
        if schema_id == SCHEMA_V2:
            model.load_state_dict(best_state)
            result['cost_diagnostics'] = cost_diagnostics(model,validation_cost_groups(groups),normalizer,cost_statistics)
            payload['cost_statistics'] = cost_statistics
        tmp = output/'model.pt.tmp'; torch.save(payload,tmp); os.replace(tmp,output/'model.pt')
        result['model_hash']=hashlib.sha256((output/'model.pt').read_bytes()).hexdigest()
        atomic_json(output/'feature_normalizer.json',normalizer.as_dict())
    atomic_json(output/'model_card.json',dict(result,fingerprint=fingerprint,heldout_accessed=False))
    return result


def fit_cost_statistics(groups):
    values = [c for g in groups for f in g['facts'].values()
              for _,c in [auxiliary_targets(f,schema_id=SCHEMA_V2)] if c is not None]
    mean = math.fsum(values)/len(values) if values else 0.
    return dict(fit_split='train',count=len(values),mean=mean,
        std=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values)) if values else 0.)


def eligible_training_groups(groups,*,schema_id=SCHEMA_ID):
    """Admit rank groups and v2 complete-cost groups; validity alone never admits."""
    return [g for g in groups if g['context']['split']=='train' and (g['pairs'] or
        (schema_id==SCHEMA_V2 and any(auxiliary_targets(f,schema_id=schema_id)[1] is not None for f in g['facts'].values())))]


def validation_cost_groups(groups):
    """Cost diagnostics include cost-only validation groups, independent of rank selection."""
    return [g for g in groups if g['context']['split'] == 'validation' and
            any(auxiliary_targets(f,schema_id=SCHEMA_V2)[1] is not None for f in g['facts'].values())]


def cost_diagnostics(model,groups,normalizer,statistics):
    """Descriptive log1p-ms MAE and train-mean baseline, no acceptance threshold."""
    import torch
    rows = []
    with torch.inference_mode():
        for index,group in enumerate(groups):
            for did,fact in group['facts'].items():
                _,target = auxiliary_targets(fact,schema_id=SCHEMA_V2)
                if target is not None:
                    vector = model([normalizer.transform(group['features'][did])])[0].tolist()
                    predicted = vector[2]
                    adjusted = cost_aware_scores([did],[vector],statistics)[did]
                    rows.append(dict(context_id=group.get('context',{}).get('context_id',str(index)),descriptor_id=did,
                        target_log1p_ms=target,predicted_log1p_ms=predicted,preference_score=vector[0],
                        cost_adjusted_score=adjusted,cost_penalty=vector[0]-adjusted,
                        cost_penalty_clipped=abs((predicted-statistics['mean'])/max(statistics['std'],1.e-6))>3.))
    by_context={}
    for r in rows:by_context.setdefault(r['context_id'],[]).append(r)
    changed=sum(sorted(values,key=lambda r:-r['preference_score'])!=sorted(values,key=lambda r:-r['cost_adjusted_score']) for values in by_context.values())
    return dict(count=len(rows),mae_log1p_ms=math.fsum(abs(r['target_log1p_ms']-r['predicted_log1p_ms']) for r in rows)/len(rows) if rows else None,
        constant_train_mean_mae_log1p_ms=math.fsum(abs(r['target_log1p_ms']-statistics['mean']) for r in rows)/len(rows) if rows else None,
        predictions=rows,ordering_scope='MEASURED_COST_VALIDATION_EXAMPLES_ONLY',covered_context_count=len(by_context),
        comparable_order_context_count=sum(len(v)>1 for v in by_context.values()),ordering_changed_contexts=changed,
        penalty_min=min((r['cost_penalty'] for r in rows),default=None),penalty_max=max((r['cost_penalty'] for r in rows),default=None),
        penalty_clipped_count=sum(r['cost_penalty_clipped'] for r in rows))


