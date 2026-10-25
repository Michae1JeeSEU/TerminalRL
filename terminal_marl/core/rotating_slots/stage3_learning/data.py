"""Portable Stage3 ancestry, snapshots and offline sampling (no model import)."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
import random

from ..stage2_campaign import atomic_json, digest, load_json


def connected_splits(records):
    """Reject transitive root/manifest/checkpoint leakage, including policy branches."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    for row in records:
        keys = [(k, str(row[k])) for k in ("ancestry_root_id", "manifest_lineage_id", "checkpoint_parent_id") if row.get(k)]
        for x in keys[1:]: parent[find(x)] = find(keys[0])
    groups = defaultdict(set)
    for row in records:
        key = ("ancestry_root_id", str(row["ancestry_root_id"]))
        groups[find(key)].add(row["split"])
        if row.get("seed") == 71 and row["split"] != "development":
            raise ValueError("seed71 is development only")
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError("connected ancestry component crosses splits")
    return {str(k): next(iter(v)) for k, v in groups.items()}


def scan_used_seeds(paths, *, exclusions=None):
    """Read protocol/manifest metadata only; never use outcome values to select seeds."""
    found = {11, 23, 47, 71, 101}
    allowed = {"seed", "preferred_seed", "initial_seed", "manifest_seed", "seeds"}
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in allowed:
                    for s in item if isinstance(item, list) else [item]:
                        if isinstance(s, int): found.add(s)
                elif isinstance(item, (dict, list)): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    scanned = []
    for path in sorted(set(Path(p).resolve() for p in paths)):
        if path.is_file() and path.suffix.lower() in (".json", ".yaml", ".yml"):
            try:
                raw=path.read_bytes()
                sentinel=("SYNTHETIC_BYTE_COMPARISON_ONLY:"+path.name).encode("utf-8")
                if raw in (sentinel,sentinel+b"\n",sentinel+b"\r\n"):
                    if exclusions is not None:
                        exclusions.append(dict(path=str(path),reason="EXACT_SYNTHETIC_BYTE_COMPARISON_SENTINEL",
                            sha256=sha256(raw).hexdigest()))
                    continue
                if path.suffix == ".json": value = load_json(path)
                else:
                    import yaml
                    value = yaml.safe_load(path.read_text(encoding="utf-8"))
                visit(value); scanned.append(str(path))
            except (OSError, ValueError):
                raise ValueError("unreadable collision-scan input: " + str(path))
    return found, scanned


def freeze_split(plan, used_seeds=(), *, scan_inputs=(), seed_scan_exclusions=()):
    used = set(used_seeds) | {71}
    rows = []
    for planned in plan["roots"]:
        seed = planned["preferred_seed"]
        if seed in used:
            seed = 7301
            while seed in used: seed += 1
        used.add(seed)
        rows.append(dict(planned, seed=seed, ancestry_root_id=planned["root_id"],
            manifest_lineage_id="manifest:" + planned["root_id"], checkpoint_parent_id=None,
            initial_state_kind="SYNTHETIC_INITIAL"))
    connected_splits(rows)
    if len(rows) != 24 or {s:sum(r["split"] == s for r in rows) for s in ("train","validation","test")} != {"train":16,"validation":4,"test":4}:
        raise ValueError("formal split must contain exactly 24 roots (16/4/4)")
    value = dict(schema="stage3_split_v1", roots=rows, collision_scan_inputs=list(scan_inputs),
        used_seeds_before_freeze=sorted(set(used_seeds)), frozen_before_outcomes=True)
    if seed_scan_exclusions:value["seed_scan_exclusions"]=list(seed_scan_exclusions)
    value["split_manifest_id"] = digest(value)
    return value


def make_root(root, output, binding):
    """Materialize the existing fixture law with frozen independent streams."""
    from ..config import load_service_profile
    from ..demand import ArrivalManifest
    from ..stage2_campaign import busy_fixture, save_checkpoint
    from ..stage2_planning import NormalSchedulerCallback
    directory = Path(output) / "roots" / root["root_id"]
    directory.mkdir(parents=True, exist_ok=True)
    manifest = ArrivalManifest.generate(root["seed"], root["arrival_rate_per_hour"], 3600., allow_new_seed=True)
    world = busy_fixture(manifest, config=load_service_profile("service_half_v1"),
        scenario_profile=root["scenario_profile"], rescue_rule_profile="rescue_need_v7", initial_seed=root["seed"])
    identity = checkpoint_identity(binding, root, manifest)
    scheduler = NormalSchedulerCallback("rh_cp_sat", planning_profile="downstream_service_v8")
    checkpoint = directory / "initial.checkpoint.json"
    save_checkpoint(checkpoint, world, scheduler, identity)
    initial = dict(aircraft=[a.to_dict() for a in world.aircraft.values()], slots=world.slots.to_dict(), pads=world.pads.to_dict())
    atomic_json(directory / "manifest.json", manifest.to_dict())
    atomic_json(directory / "initial_state.json", initial)
    row = dict(root, checkpoint_path=str(checkpoint), identity=identity, snapshot_id=root["root_id"]+":t0",
        initial_state_hash=digest(initial), manifest_hash=digest(manifest.to_dict()),
        raw_snapshot_path=str(directory / "initial_state.json"))
    atomic_json(directory / "root.json", row)
    return row


def checkpoint_identity(binding, root, manifest):
    from ..world import STAGE2_POLICY_ID
    return dict(policy_id=STAGE2_POLICY_ID, source_hash=binding["source_hash"],
        base_config_hash=binding["fingerprint"]["physical_config_hash"], overlay_sha256=binding["contract_hash"],
        manifest_sha256=digest(manifest.to_dict()),service_profile="service_half_v1",
        scenario_profile=root["scenario_profile"], rescue_rule_profile="rescue_need_v7",planning_profile="downstream_service_v8")


def restore_root(row, scheduler):
    from ..stage2_campaign import restore_checkpoint
    return restore_checkpoint(row["checkpoint_path"], scheduler, row["identity"])


def validate_independent_initial_states(rows):
    seen = {}
    for row in rows:
        state_hash = row["initial_state_hash"]
        if state_hash in seen:
            raise ValueError("identical t0 states are not independent roots: " + seen[state_hash] + ", " + row["root_id"])
        seen[state_hash] = row["root_id"]
    connected_splits(rows)


class StratifiedProbe:
    """One seeded probe opportunity in every five offline selections.

    Select a stratum uniformly, then a descriptor within it. The actual
    conditional probability is saved, rather than reporting a false global 0.2.
    """
    def __init__(self, seed=20261024): self.rng = random.Random(seed)
    def select(self, descriptors, selected, opportunity_index):
        if opportunity_index % 5 != 4 or not descriptors:
            return selected, dict(offline_probe=False, probe_opportunity_fraction=.2, selection_probability=1.)
        strata = defaultdict(list)
        for d in descriptors:
            get = (lambda k, default=None: d.get(k, default)) if isinstance(d, dict) else (lambda k, default=None: getattr(d,k,default))
            depth = get("chain_depth", 0 if get("parent_prefix_id") is None else 1)
            strata[(get("source_ring"),get("target_kind"),get("start_offset_s",0)>0,depth)].append(d)
        keys = sorted(strata, key=repr); key = keys[self.rng.randrange(len(keys))]
        group = strata[key]; result = group[self.rng.randrange(len(group))]
        return result, dict(offline_probe=True, probe_opportunity_fraction=.2,
            selection_probability=1./len(keys)/len(group), stratum=list(key))


class PortableObserver:
    """Raw feature/fact streams remain physically separate from derived labels."""
    def __init__(self, directory):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.records = []
    def record(self, value):
        value = value.as_dict() if hasattr(value,"as_dict") else dict(value)
        self.records.append(value)
        kind = value.get("record_type", "trace")
        name = "features.jsonl" if kind == "FeaturesBeforeQuery" else "descriptors.jsonl" if kind == "WorkDescriptor" else "facts.jsonl"
        with (self.directory / name).open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
    __call__ = record


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def auxiliary_labels(facts):
    rows=[]
    for fact in facts:
        outcome=fact.get("outcome")
        validity = 1. if outcome=="SINGLE_VALID" else 0. if outcome=="SINGLE_INVALID_CERTIFIED" else None
        if fact.get("validity_label_mask") is False:validity=None
        wall = fact.get('wall_s')
        complete = not fact.get("censored",False) and outcome not in ("TIMEOUT","NOT_ATTEMPTED") and isinstance(wall,(int,float)) and math.isfinite(wall) and wall >= 0
        if fact.get('cost_scope') is not None:
            complete = complete and fact.get('cost_complete',False)
        rows.append(dict(descriptor_id=fact["descriptor_id"], validity=validity,validity_mask=validity is not None,
            log1p_cost_ms=math.log1p(fact["wall_s"]*1000.) if complete else None,cost_mask=complete,censored=fact.get("censored",False)))
    return rows


def merge_v2_datasets(datasets, fingerprint):
    """Merge ancestry-safe datasets; migrate known v1 fields, never infer new context.

    Conflicting/repeated context IDs are rejected rather than choosing a feature
    observation silently. Legacy partial costs stay explicitly masked for v2.
    """
    from copy import deepcopy
    from .features import SCHEMA_ID, SCHEMA_V2, migrate_v1_feature_to_v2
    from .model import fingerprint_compatible
    contexts, pairs, roots, seen = [], [], {}, set()
    for dataset in datasets:
        if not fingerprint_compatible(dataset.get('fingerprint',{}),fingerprint):
            raise ValueError('dataset fingerprint mismatch')
        for source in dataset.get('contexts',[]):
            context = deepcopy(source)
            if context['split'] not in ('train','validation'):
                raise ValueError('keep test sealed')
            root = context['ancestry_root_id']
            if root in roots and roots[root] != context['split']:
                raise ValueError('ancestry crosses split')
            roots[root] = context['split']
            if context['context_id'] in seen:
                raise ValueError('duplicate context observation; retain precise feature reference')
            seen.add(context['context_id'])
            context['features'] = [migrate_v1_feature_to_v2(f) if f.get('schema_id') == SCHEMA_ID else f for f in context['features']]
            if any(f.get('schema_id') != SCHEMA_V2 for f in context['features']):
                raise ValueError('unsupported feature schema')
            for fact in context.get('facts',[]):
                if fact.get('cost_scope') != 'GENERATOR_ENERGY_TERMINAL_VALIDATION_V2':
                    fact['cost_complete'] = False
                    fact['legacy_cost_boundary'] = fact.get('cost_boundary','GENERATOR_PLUS_OBSERVED_SINGLE_VEHICLE_ENERGY_EXCLUDES_JOINT_VALIDATION')
            contexts.append(context)
        pairs.extend(deepcopy(dataset.get('preference_pairs',[])))
    return dict(schema='stage3_training_dataset_v2',feature_schema_id=SCHEMA_V2,fingerprint=dict(fingerprint),contexts=contexts,preference_pairs=pairs)


def migrate_v1_dataset(dataset,fingerprint,*,expected_source_fingerprint=None):
    """Explicit authorized semantic migration with original binding preserved.

    This changes feature/cost schema bindings, not evidence/ancestry/objectives.
    Caller supplies the frozen source binding when it is independently known.
    """
    from copy import deepcopy
    from .features import SCHEMA_ID,SCHEMA_V2,migrate_v1_feature_to_v2
    from .model import fingerprint_compatible
    original=dataset.get('fingerprint',{})
    if expected_source_fingerprint is not None and not fingerprint_compatible(original,expected_source_fingerprint):
        raise ValueError('source dataset fingerprint mismatch')
    if original.get('objective_id') != fingerprint.get('objective_id'):
        raise ValueError('migration cannot change objective semantics')
    result=deepcopy(dataset)
    result.update(schema='stage3_training_dataset_v2',feature_schema_id=SCHEMA_V2,fingerprint=dict(fingerprint),
        migration_provenance=dict(source_fingerprint=original,policy='V1_KNOWN_NUMERIC_FIELDS_NEW_CONTEXT_MASKED'))
    for c in result.get('contexts',[]):
        if not fingerprint_compatible(c.get('fingerprint',original),original):
            raise ValueError('source context fingerprint mismatch')
        c['migration_provenance']=dict(source_fingerprint=c.get('fingerprint',original))
        c['fingerprint']=dict(fingerprint)
        c['features']=[migrate_v1_feature_to_v2(f) if f.get('schema_id')==SCHEMA_ID else f for f in c['features']]
        for fact in c.get('facts',[]):
            if not fingerprint_compatible(fact.get('fingerprint',original),original):
                raise ValueError('source fact fingerprint mismatch')
            fact['source_fingerprint']=fact.get('fingerprint',original)
            fact['fingerprint']=dict(fingerprint)
            fact['cost_complete']=False
            fact['legacy_cost_boundary']=fact.get('cost_boundary','GENERATOR_PLUS_OBSERVED_SINGLE_VEHICLE_ENERGY_EXCLUDES_JOINT_VALIDATION')
    for pair in result.get('preference_pairs',[]):
        if pair.get('objective_id') != fingerprint.get('objective_id'):
            raise ValueError('migration pair objective mismatch')
    return merge_v2_datasets([result],fingerprint)


def query_observations(records):
    """Exact feature-ref join; repeated descriptor queries remain distinct rows.

    Integration must explicitly select an observation when building a preference
    context. This function never overwrites observations by descriptor ID.
    """
    features = {r['features_before_query_ref']:r for r in records if r.get('record_type') == 'FeaturesBeforeQuery'}
    return [dict(fact=f,features=features.get(f.get('features_before_query_ref')))
            for f in records if f.get('record_type') == 'FactualResult']


def summarize_coverage(descriptors,facts,supported_descriptor_ids=(),*,support_kind="OBSERVED_CERTIFIED_ANCESTRY"):
    """Distinct descriptor coverage; proposals, queries and physical work stay separate."""
    by_id={d["descriptor_id"]:d for d in descriptors}
    support=set(supported_descriptor_ids)
    evidence={}
    for fact in facts:
        if fact.get("outcome") in ("SINGLE_VALID","SINGLE_INVALID_CERTIFIED") and fact.get("validity_label_mask") is not False:
            evidence[fact["descriptor_id"]]=fact["outcome"]
    attempted={f["descriptor_id"] for f in facts if f.get("record_type","FactualResult")=="FactualResult" and f.get("outcome")!="NOT_ATTEMPTED" and not f.get("censored",False)}
    queries=[f for f in facts if f.get("record_type","FactualResult")=="FactualResult"]
    def counts(ids):
        ids=set(ids)
        return dict(proposed_descriptors=len(ids),attempted_descriptors=len(ids&attempted),
            single_valid_descriptors=sum(evidence.get(d)=="SINGLE_VALID" for d in ids),
            certified_single_invalid_descriptors=sum(evidence.get(d)=="SINGLE_INVALID_CERTIFIED" for d in ids),
            supported_descriptors=len(ids&support),
            materialization_queries=sum(f["descriptor_id"] in ids for f in queries),
            physical_attempts=sum(f.get("physical_attempts",0) for f in queries if f["descriptor_id"] in ids))
    ring_groups=defaultdict(list);stage_groups=defaultdict(list);joint_groups=defaultdict(list)
    for did,descriptor in by_id.items():
        ring=descriptor.get("source_ring","UNKNOWN");stage=descriptor.get("construction_stage","UNKNOWN")
        ring_groups[ring].append(did);stage_groups[stage].append(did);joint_groups[(ring,stage)].append(did)
    return dict(support_kind=support_kind,counts=counts(by_id),
        by_ring={ring:counts(ids) for ring,ids in sorted(ring_groups.items())},
        by_stage={stage:counts(ids) for stage,ids in sorted(stage_groups.items())},
        by_ring_stage=[dict(source_ring=ring,construction_stage=stage,**counts(ids)) for (ring,stage),ids in sorted(joint_groups.items())],
        supported_unmapped_descriptor_ids=sorted(support-set(by_id)))
