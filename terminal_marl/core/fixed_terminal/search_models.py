"""One frozen gradient-boost expand classifier, fitted only on train traces."""
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss

from .learning_data import write_json
from .search_data import file_hash, LABEL_SCOPE
from .search_features import FEATURE_NAMES, FEATURE_DIM, SCHEMA as FEATURE_SCHEMA
from terminal_marl.configs.fixed_terminal_stage1 import DEFAULT_CONFIG
from terminal_marl.configs.fixed_terminal_stage7 import TRAINING, EXPAND_THRESHOLD, scenario_jobs, SNAPSHOT_TIMES, TRACE_WIDTHS
from dataclasses import asdict

MODEL_SCHEMA = "fixed-terminal-stage7-expand-model-v1"
MANIFEST_SCHEMA = "fixed-terminal-stage7-model-manifest-v1"
MODEL_FILE = "gradient_boost_expand.joblib"
NAME = "gradient_boost_expand"


def _matrix(rows):
    x = np.asarray(rows, dtype=np.float64)
    if len(rows) == 0:
        return np.empty((0, FEATURE_DIM), dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
        raise ValueError("feature schema/dimension/finite check failed")
    return x


class FrozenSearchPolicy:
    name, kind = NAME, "expand"

    def __init__(self, model, manifest=None):
        self.model, self.manifest = model, manifest

    def predict_expand(self, feature_rows):
        x = _matrix(feature_rows)
        if not len(x):
            return []
        probabilities = self.model.predict_proba(x)
        classes = list(self.model.classes_)
        result = probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(x))
        if result.shape != (len(x),) or not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
            raise ValueError("model produced invalid expand probabilities")
        return result.astype(float).tolist()

    @classmethod
    def from_directory(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "completed"
                or manifest.get("name") != NAME or manifest.get("kind") != "expand"
                or manifest.get("feature_schema") != FEATURE_SCHEMA or manifest.get("feature_names") != list(FEATURE_NAMES)
                or manifest.get("feature_dim") != FEATURE_DIM or manifest.get("training") != TRAINING
                or manifest.get("trace_widths") != list(TRACE_WIDTHS)
                or manifest.get("threshold") != EXPAND_THRESHOLD or manifest.get("test_read") is not False
                or manifest.get("frozen") is not True):
            raise ValueError("frozen search manifest schema/protocol mismatch")
        if set(manifest["files"]) != {MODEL_FILE, "protocol.json", "training.json"}:
            raise ValueError("model dependencies incomplete")
        for name, digest in manifest["files"].items():
            if Path(name).name != name or file_hash(directory/name) != digest:
                raise ValueError("model dependency hash mismatch")
        payload = joblib.load(directory/MODEL_FILE)
        if (payload["schema"] != MODEL_SCHEMA or payload["kind"] != "expand" or payload["name"] != NAME
                or payload["feature_names"] != list(FEATURE_NAMES) or payload["feature_schema"] != FEATURE_SCHEMA
                or payload["training"] != TRAINING):
            raise ValueError("serialized search model schema/kind mismatch")
        if payload.get("trace_widths") != list(TRACE_WIDTHS):
            raise ValueError("serialized training trace budgets mismatch")
        model = payload["model"]
        if not isinstance(model, (GradientBoostingClassifier, DummyClassifier)) or model.n_features_in_ != FEATURE_DIM:
            raise ValueError("unexpected model type or input dimension")
        if isinstance(model, GradientBoostingClassifier) and any(model.get_params()[k] != v for k, v in TRAINING.items()):
            raise ValueError("serialized model parameters differ from protocol")
        return cls(model, manifest)


def validate_training_groups(train, val, smoke=False):
    for contexts, split in ((train, "train"), (val, "val")):
        wanted = "pilot" if smoke else split
        if not contexts or any(c["metadata"]["split"] != wanted for c in contexts):
            raise ValueError("train/val split mismatch; test is never accepted")
        if not smoke:
            expected = {(j["seed"], j["load"], j["soc"], time) for j in scenario_jobs(split) for time in SNAPSHOT_TIMES}
            actual = {(c["metadata"]["seed"], c["metadata"]["load"], c["metadata"]["soc"], c["metadata"]["root_time"]) for c in contexts}
            if actual != expected or len(contexts) != len(expected):
                raise ValueError("formal training context inventory is incomplete or duplicated")
            if any(c["metadata"]["config"] != json.loads(json.dumps(asdict(DEFAULT_CONFIG))) or c["metadata"]["smoke_only"] for c in contexts):
                raise ValueError("formal physical config or smoke provenance mismatch")
        _matrix([row for c in contexts for row in c["features"]])
        if any(type(y) is not int or y not in (0, 1) for c in contexts for y in c["targets"]):
            raise ValueError("invalid binary expansion label")
    if not smoke and {tuple(c["metadata"]["scenario_group"]) for c in train} & {tuple(c["metadata"]["scenario_group"]) for c in val}:
        raise ValueError("complete scenario leakage between train/val")


def _metrics(policy, contexts):
    x = [row for c in contexts for row in c["features"]]
    y = np.asarray([v for c in contexts for v in c["targets"]], dtype=int)
    p = np.asarray(policy.predict_expand(x))
    if not len(y):
        return {"nodes": 0, "positive": 0, "negative": 0, "positive_fraction": None, "accuracy": None,
                "precision": None, "recall": None, "brier": None}
    predicted = (p >= EXPAND_THRESHOLD).astype(int)
    return {"nodes": len(y), "positive": int(y.sum()), "negative": int(len(y)-y.sum()),
            "positive_fraction": float(y.mean()), "accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)), "brier": float(brier_score_loss(y, p))}


def train_model(train, val, output, *, smoke=False):
    validate_training_groups(train, val, smoke)
    x = _matrix([row for c in train for row in c["features"]])
    y = np.asarray([v for c in train for v in c["targets"]], dtype=int)
    if not len(y):
        raise ValueError("no expandable training nodes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    one_class = len(np.unique(y)) == 1
    model = DummyClassifier(strategy="constant", constant=int(y[0])) if one_class else GradientBoostingClassifier(**TRAINING)
    tick = perf_counter()
    model.fit(x, y)  # Exactly one fit; validation never selects or modifies a model.
    fit_seconds = perf_counter()-tick
    policy = FrozenSearchPolicy(model)
    protocol = {"schema": MODEL_SCHEMA, "kind": "expand", "name": NAME, "training": TRAINING,
                "feature_schema": FEATURE_SCHEMA, "feature_names": FEATURE_NAMES, "feature_dim": FEATURE_DIM,
                "threshold": EXPAND_THRESHOLD, "test_read": False, "fit_count": 1, "smoke_only": smoke,
                "trace_widths": list(TRACE_WIDTHS), "training_context_schema": "fixed-terminal-stage7-context-v2",
                "label_scope": LABEL_SCOPE, "validation_use": "report_only; no_threshold_parameter_or_checkpoint_selection",
                "train": [{"path": c["path"], "context_sha256": c["context_hash"], "scenario_group": c["metadata"]["scenario_group"], "time": c["metadata"]["root_time"]} for c in train],
                "val": [{"path": c["path"], "context_sha256": c["context_hash"], "scenario_group": c["metadata"]["scenario_group"], "time": c["metadata"]["root_time"]} for c in val]}
    payload = {"schema": MODEL_SCHEMA, "kind": "expand", "name": NAME, "feature_names": list(FEATURE_NAMES),
               "feature_schema": FEATURE_SCHEMA, "training": TRAINING, "trace_widths": list(TRACE_WIDTHS), "model": model}
    joblib.dump(payload, output/MODEL_FILE)
    reloaded = FrozenSearchPolicy(joblib.load(output/MODEL_FILE)["model"])
    check_rows = [row for c in train+val for row in c["features"]]
    roundtrip = policy.predict_expand(check_rows) == reloaded.predict_expand(check_rows)
    if not roundtrip:
        raise RuntimeError("frozen model roundtrip changed probabilities")
    metrics = {"train": _metrics(policy, train), "val": _metrics(policy, val)}
    training = {"status": "completed", "protocol": protocol, "fit_seconds": fit_seconds, "metrics": metrics,
                "model_class": type(model).__name__, "one_class_dummy": one_class, "roundtrip_predictions_equal": roundtrip}
    write_json(output/"protocol.json", protocol)
    write_json(output/"training.json", training)
    manifest = {"schema": MANIFEST_SCHEMA, "status": "completed", "frozen": True, "test_read": False,
                "name": NAME, "kind": "expand", "feature_schema": FEATURE_SCHEMA, "feature_names": FEATURE_NAMES,
                "feature_dim": FEATURE_DIM, "training": TRAINING, "threshold": EXPAND_THRESHOLD,
                "trace_widths": list(TRACE_WIDTHS), "training_context_schema": "fixed-terminal-stage7-context-v2",
                "smoke_only": smoke, "model_class": type(model).__name__, "fit_count": 1,
                "metrics": metrics, "roundtrip_predictions_equal": roundtrip,
                "splits": {"train": sorted({c["metadata"]["seed"] for c in train}), "val": sorted({c["metadata"]["seed"] for c in val})},
                "producer": "terminal_marl.scripts.train_fixed_terminal_stage7",
                "files": {name: file_hash(output/name) for name in (MODEL_FILE, "protocol.json", "training.json")}}
    write_json(output/"manifest.json", manifest)
    FrozenSearchPolicy.from_directory(output)
    return manifest
