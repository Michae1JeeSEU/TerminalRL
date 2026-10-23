"""Lazy CPU set MLP and checkpoint-backed advisor.

Importing this module does not import torch; OFF remains dependency independent.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
from .features import SCHEMA_ID, QUERY_DIM, SET_DIMS, SET_LIMITS, FeatureInputError

ARCHITECTURE_ID = 'cc_set_mlp64_v1_0'


def _validate_feature(feature):
    if feature.get('schema_id') != SCHEMA_ID:
        raise FeatureInputError('INPUT_OOD', 'feature schema')
    for name, rows in [('query',[feature['query']]), *feature['sets'].items()]:
        dim = QUERY_DIM if name == 'query' else SET_DIMS[name]
        if name != 'query' and len(rows) > SET_LIMITS[name]:
            raise FeatureInputError('ENTITY_LIMIT')
        for row in rows:
            if len(row) != dim or not all(math.isfinite(float(v)) for v in row):
                raise FeatureInputError('INPUT_OOD', 'shape or nonfinite')
            if any(v not in (0.,1.) for v in row[1::2]):
                raise FeatureInputError('INPUT_OOD', 'missing indicator')


class FeatureNormalizer:
    """Moments are fitted only on observed numeric values from training contexts."""
    def __init__(self, statistics=None):
        self.statistics = statistics

    def fit(self, contexts):
        values = {k:[[] for _ in range(d//2)] for k,d in [('query',QUERY_DIM),*SET_DIMS.items()]}
        if not contexts or any(c.get('split') != 'train' for c in contexts):
            raise ValueError('normalizer fit requires train contexts only')
        for context in contexts:
            for f in context['features']:
                _validate_feature(f)
                for k, rows in [('query',[f['query']]),*f['sets'].items()]:
                    for r in rows:
                        for j in range(len(r)//2):
                            if r[2*j+1]:
                                values[k][j].append(float(r[2*j]))
        self.statistics = {}
        for k,columns in values.items():
            self.statistics[k] = []
            for column in columns:
                mean = math.fsum(column)/len(column) if column else 0.
                variance = math.fsum((x-mean)**2 for x in column)/len(column) if column else 0.
                self.statistics[k].append(dict(mean=mean,std=max(math.sqrt(variance),1e-6),count=len(column)))
        return self

    def transform(self, feature):
        _validate_feature(feature)
        if self.statistics is None:
            raise ValueError('normalizer is not fitted')
        result = copy.deepcopy(feature)
        for k, rows in [('query',[result['query']]),*result['sets'].items()]:
            for r in rows:
                for j,s in enumerate(self.statistics[k]):
                    r[2*j] = (r[2*j]-s['mean'])/s['std'] if r[2*j+1] else 0.
        return result

    def as_dict(self):
        return dict(schema_id=SCHEMA_ID,fit_split='train',statistics=self.statistics)

    @classmethod
    def from_dict(cls, value):
        if value.get('schema_id') != SCHEMA_ID or value.get('fit_split') != 'train':
            raise ValueError('incompatible normalizer')
        return cls(value['statistics'])


def build_model():
    """Construct exactly one frozen 64x2 four-set architecture."""
    import torch
    from torch import nn

    class CandidateSetMLP(nn.Module):
        def __init__(self):
            super().__init__()
            def encoder(dim):
                return nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
            self.query_encoder = encoder(QUERY_DIM)
            self.entity_encoders = nn.ModuleDict({k:encoder(d) for k,d in SET_DIMS.items()})
            self.score_head = nn.Sequential(nn.Linear(64+4*(64*2+1),128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
            if sum(p.numel() for p in self.parameters()) > 250000:
                raise ValueError('architecture parameter cap exceeded')

        @staticmethod
        def pool(encoded, mask, limit):
            # Empty sets have zero mean, zero max and zero count, including padding.
            mask = mask.bool()
            count = mask.sum(dim=-1,keepdim=True)
            mean = (encoded*mask.unsqueeze(-1)).sum(dim=-2)/count.clamp(min=1)
            maximum = encoded.masked_fill(~mask.unsqueeze(-1),float('-inf')).amax(dim=-2) if encoded.shape[-2] else torch.zeros_like(mean)
            maximum = torch.where(count > 0,maximum,torch.zeros_like(maximum))
            return torch.cat((mean,maximum,count.to(encoded.dtype)/limit),dim=-1)

        def forward(self, features):
            if not features:
                return torch.empty((0,3),dtype=torch.float32,device='cpu')
            for f in features:
                _validate_feature(f)
            query = torch.tensor([f['query'] for f in features],dtype=torch.float32,device='cpu')
            chunks = [self.query_encoder(query)]
            for name,dim in SET_DIMS.items():
                size = max((len(f['sets'][name]) for f in features),default=0)
                data = torch.zeros((len(features),size,dim),dtype=torch.float32,device='cpu')
                mask = torch.zeros((len(features),size),dtype=torch.bool,device='cpu')
                for i,f in enumerate(features):
                    rows = f['sets'][name]
                    if rows:
                        data[i,:len(rows)] = torch.tensor(rows,dtype=torch.float32,device='cpu')
                        mask[i,:len(rows)] = True
                chunks.append(self.pool(self.entity_encoders[name](data),mask,SET_LIMITS[name]))
            return self.score_head(torch.cat(chunks,dim=-1))
    return CandidateSetMLP().cpu()


def fingerprint_compatible(actual, expected):
    """A collected NO_MODEL is replaced with the frozen hash at evaluation."""
    a, b = dict(actual), dict(expected)
    a.pop('model_id',None); b.pop('model_id',None)
    return a == b


class LearnedAdvisor:
    def __init__(self, checkpoint, fingerprint=None, *, evaluation=False):
        self.path = Path(checkpoint)
        if self.path.is_dir():
            self.path = self.path/'model.pt'
        self.fingerprint = fingerprint
        self.evaluation = evaluation
        self._model = self._normalizer = self._metadata = None

    def _load(self):
        if not self.path.is_file():
            raise FeatureInputError('MODEL_MISSING')
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        data = torch.load(self.path,map_location='cpu',weights_only=True)
        if data.get('architecture_id') != ARCHITECTURE_ID or data.get('schema_id') != SCHEMA_ID:
            raise FeatureInputError('FINGERPRINT_MISMATCH')
        if self.fingerprint is None or not fingerprint_compatible(data['fingerprint'],self.fingerprint):
            raise FeatureInputError('FINGERPRINT_MISMATCH')
        self._model = build_model()
        self._model.load_state_dict(data['state_dict'],strict=True)
        self._model.eval()
        self._normalizer = FeatureNormalizer.from_dict(data['normalizer'])
        self._metadata = dict(model_hash=hashlib.sha256(self.path.read_bytes()).hexdigest(),architecture_id=ARCHITECTURE_ID)

    def rank(self, features, budget=None):
        from .advisor import RankResult
        if self._model is None:
            self._load()
        if len(features) > 64:
            raise FeatureInputError('INPUT_OOD','ready batch exceeds 64')
        import torch
        normalized = [self._normalizer.transform(f) for f in features]
        with torch.inference_mode():
            prediction = self._model(normalized).tolist()
        ids = [f['descriptor_id'] for f in features]
        if len(set(ids)) != len(ids):
            raise FeatureInputError('ILLEGAL_DESCRIPTOR_ID')
        if not all(math.isfinite(x) for r in prediction for x in r):
            raise FeatureInputError('NONFINITE_SCORE')
        scores = {did:r[0] for did,r in zip(ids,prediction)}
        ordered = tuple(sorted(ids,key=lambda did:(-scores[did],ids.index(did))))
        return RankResult(ordered_ids=ordered,model_metadata=self._metadata,scores=scores)

