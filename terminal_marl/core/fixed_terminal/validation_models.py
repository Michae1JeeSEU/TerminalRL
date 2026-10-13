"""Train-only stage5 scorers and TOTAL-budget validation checkpoint selection.

Boosting checkpoints are trusted, locally produced joblib artifacts. Their
manifest/hash is checked before loading; this is not a safe unpickler for files
from an untrusted source. Torch checkpoints use weights_only=True.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import platform
import numpy as np
import torch
from torch import nn

from .learning_data import load_contexts, write_json
from .learning_features import FEATURE_NAMES
from .learning_models import Standardizer, MLP, root_objects, true_pairs, weighted_training_pairs
from .learning_scheduler import prediction_order
from .reference import value_comparison_key
from terminal_marl.configs.fixed_terminal_stage5 import (
    TRAIN_SEEDS, VAL_SEEDS, PILOT_SEEDS, MODEL_NAMES, MODEL_FILES, TRAINING, BOOSTING)

SCHEMA = "fixed-terminal-stage5-model-manifest-v1"
PRODUCER = "terminal_marl.core.fixed_terminal.validation_models"


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def total_budget_indices(context, ranked_indices, budget=5):
    """Deduplicate anchors first, then fill the TOTAL budget in ranking order."""
    ids = [r["root_id"] for r in context["roots"]]
    if len(set(ids)) != len(ids) or sorted(ranked_indices) != list(range(len(ids))):
        raise ValueError("unique roots and a complete ranking permutation required")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("positive integer TOTAL budget required")
    selected = []
    for key in context["pool_metadata"]["anchors"]:
        if key not in ids:
            raise ValueError("anchor not present in candidate pool")
        index = ids.index(key)
        if index not in selected:
            selected.append(index)
    if len(selected) > budget:
        raise ValueError("TOTAL budget cannot exclude mandatory anchors")
    for index in ranked_indices:
        if len(selected) == min(budget, len(ids)):
            break
        if index not in selected:
            selected.append(index)
    return selected


def ordinal_targets(context):
    keys = [value_comparison_key(value) for value in context["values"]]
    levels = sorted(set(keys))
    ranks = {key: i / (len(levels)-1) if len(levels) > 1 else 0.
             for i, key in enumerate(levels)}
    return [ranks[key] for key in keys]


class ValidationScorer:
    def __init__(self, name, model, inputs, targets=None):
        if name not in MODEL_NAMES:
            raise ValueError("unknown stage5 scorer")
        torch.set_num_threads(1)
        self.name = name
        self.kind = "value" if name == "value" else "rank"
        self.model, self.inputs, self.targets = model, inputs, targets
        if name != "gradient_boost":
            self.model.cpu().eval()

    def predict(self, feature_rows):
        if not feature_rows:
            return []
        values = np.asarray(feature_rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
            raise ValueError("finite93-dimensional features required")
        x = self.inputs.transform(values)
        if self.name == "gradient_boost":
            output = self.model.predict(x).reshape(-1, 1)
        else:
            with torch.no_grad():
                output = self.model(torch.from_numpy(x)).cpu().numpy()
            if self.kind == "value":
                output = self.targets.inverse(output)
        if not np.isfinite(output).all():
            raise ValueError("nonfinite prediction")
        return output.tolist()

    def save(self, path, protocol):
        payload = {"schema": "fixed-terminal-stage5-scorer-v1", "name": self.name,
                   "kind": self.kind, "feature_names": list(FEATURE_NAMES),
                   "inputs": self.inputs.serializable(),
                   "targets": self.targets.serializable() if self.targets else None,
                   "protocol": protocol}
        if self.name == "gradient_boost":
            import joblib
            joblib.dump({**payload, "model": self.model}, path)
        else:
            torch.save({**payload, "state_dict": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix == ".joblib":
            # Only accept the explicit local training manifest before unpickling.
            manifest = json.loads((path.parent/"manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("schema") != SCHEMA or manifest.get("producer") != PRODUCER or
                    manifest.get("status") != "completed" or not manifest.get("frozen") or
                    manifest.get("files", {}).get(path.name) != file_hash(path)):
                raise ValueError("trusted local boosting manifest/hash required")
            import joblib
            data = joblib.load(path)
        else:
            data = torch.load(path, map_location="cpu", weights_only=True)
        name = data["name"]
        if (data["schema"] != "fixed-terminal-stage5-scorer-v1" or
                data["feature_names"] != list(FEATURE_NAMES) or name not in MODEL_NAMES):
            raise ValueError("stage5 model schema mismatch")
        if name == "gradient_boost":
            if path.suffix != ".joblib":
                raise ValueError("boosting requires the checked local joblib artifact")
            model = data["model"]
        else:
            model = nn.Linear(len(FEATURE_NAMES), 1) if name == "linear_rank" else MLP(7 if name == "value" else 1)
            model.load_state_dict(data["state_dict"])
        def scaler(row):
            return Standardizer(np.asarray(row["mean"]), np.asarray(row["scale"])) if row else None
        result = cls(name, model, scaler(data["inputs"]), scaler(data["targets"]))
        result.protocol = data["protocol"]
        return result


def validation_metrics(contexts, scorer, budget=5):
    hits, informative = 0, 0
    for context in contexts:
        keys = [value_comparison_key(value) for value in context["values"]]
        if len(set(keys)) < 2:
            continue
        roots = root_objects(context)
        ordered = prediction_order(roots, scorer.predict(context["features"]), scorer.kind)
        by_id = {r.root_id: i for i, r in enumerate(roots)}
        chosen = total_budget_indices(context, [by_id[key] for key in ordered], budget)
        hits += max(keys[i] for i in chosen) == max(keys)
        informative += 1
    return {"informative_contexts": informative, "best_key_retained": hits,
            "retention_at_total_budget": hits/informative if informative else None,
            "total_budget": budget}


def selection_key(retention, loss):
    # Iterations are visited in increasing order; strict improvement keeps earliest ties.
    return retention if retention is not None else 0., -loss


def validate_groups(train, val, smoke=False):
    if not train or not val:
        raise ValueError("nonempty train and val required")
    for group, split, seeds in ((train, "train", TRAIN_SEEDS), (val, "val", VAL_SEEDS)):
        for context in group:
            meta = context["metadata"]
            if smoke:
                if meta["split"] != "pilot" or meta["seed"] not in PILOT_SEEDS:
                    raise ValueError("stage5 smoke accepts explicit pilot5001 only")
            elif meta["split"] != split or meta["seed"] not in seeds:
                raise ValueError("stage5 train/val split violation; test never accepted")
            n = len(context["features"])
            if not n or any(len(context[key]) != n for key in ("roots", "values", "targets")):
                raise ValueError("candidate/label alignment mismatch")
            for key, width in (("features", 93), ("targets", 7)):
                values = np.asarray(context[key], dtype=np.float64)
                if values.shape != (n, width) or not np.isfinite(values).all():
                    raise ValueError("finite feature/target arrays required")
            total_budget_indices(context, list(range(n)), TRAINING["checkpoint_budget"])
    if not smoke and {c["metadata"]["seed"] for c in train} & {c["metadata"]["seed"] for c in val}:
        raise ValueError("train/val seed leakage")


def _load_pilot(directory):
    """Stage5 pilot5001 only; never masquerade as stage4's pilot4001."""
    contexts = []
    for path in sorted(Path(directory).rglob("context.json")):
        def read(name):
            return json.loads((path.parent/name).read_text(encoding="utf-8"))
        meta = read("context.json")
        if meta["split"] != "pilot" or meta["seed"] not in PILOT_SEEDS or meta["status"] != "completed":
            raise ValueError("stage5 smoke accepts completed pilot5001 only")
        features, labels, roots = read("features.json"), read("labels.json"), read("roots.json")
        ids = [r["root_id"] for r in roots["roots"]]
        if (features["feature_names"] != list(FEATURE_NAMES) or len(set(ids)) != len(ids) or
                ids != [r["root_id"] for r in features["rows"]] or ids != [r["root_id"] for r in labels["rows"]]):
            raise ValueError("pilot feature/root/label alignment mismatch")
        contexts.append({"metadata": meta, "path": str(path.parent), "roots": roots["roots"],
                         "pool_metadata": roots["pool_metadata"],
                         "features": [r["values"] for r in features["rows"]],
                         "values": [r["value"] for r in labels["rows"]],
                         "targets": [r["delta_vs_HOLD"]["delta_J"] for r in labels["rows"]]})
    return contexts


def _loss(scorer, contexts):
    if scorer.name == "gradient_boost":
        return float(np.mean([np.mean((np.asarray(scorer.predict(c["features"])).ravel()-ordinal_targets(c))**2)
                              for c in contexts]))
    with torch.no_grad():
        if scorer.kind == "value":
            x = torch.from_numpy(scorer.inputs.transform([r for c in contexts for r in c["features"]]))
            y = torch.from_numpy(scorer.targets.transform([r for c in contexts for r in c["targets"]]))
            return float(nn.functional.mse_loss(scorer.model(x), y))
        losses = []
        for c in contexts:
            pairs = true_pairs(c)
            if pairs:
                prediction = scorer.model(torch.from_numpy(scorer.inputs.transform(c["features"]))).flatten()
                losses.append(nn.functional.softplus(torch.stack([prediction[j]-prediction[i] for i, j in pairs])).mean())
        return float(torch.stack(losses).mean()) if losses else 0.


def _inventory(contexts):
    return {"contexts": len(contexts), "candidates": sum(len(c["features"]) for c in contexts),
            "non_tie_pairs": sum(len(true_pairs(c)) for c in contexts),
            "informative": sum(bool(true_pairs(c)) for c in contexts),
            "seeds": sorted({c["metadata"]["seed"] for c in contexts}),
            "sources": [str(c["path"]) for c in contexts]}


def train_models(train, val, output, *, smoke=False):
    """Train each frozen model once, with validation-only checkpoint selection."""
    validate_groups(train, val, smoke)
    import sklearn
    import joblib
    from sklearn.ensemble import GradientBoostingRegressor
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    inputs = Standardizer.fit([r for c in train for r in c["features"]])
    targets = Standardizer.fit([r for c in train for r in c["targets"]])
    epochs = TRAINING["smoke_iterations"] if smoke else TRAINING["max_epochs"]
    trees = TRAINING["smoke_iterations"] if smoke else BOOSTING["n_estimators"]
    source_commit = os.environ.get("TERMINAL_SOURCE_COMMIT")
    protocol = {"schema": "fixed-terminal-stage5-training-v1", "smoke_only": smoke, "test_read": False,
                "model_names": MODEL_NAMES, "model_files": MODEL_FILES, "training": TRAINING,
                "boosting": BOOSTING, "epochs": epochs, "trees": trees, "feature_names": FEATURE_NAMES,
                "train": _inventory(train), "val": _inventory(val), "inputs": inputs.serializable(),
                "value_targets": targets.serializable(), "source_commit": source_commit,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": str(torch.__version__), "sklearn": sklearn.__version__, "joblib": joblib.__version__},
                "checkpoint_selection": "max_val_informative_best_key_retention_TOTAL_B5_anchors_first; min_model_val_loss; earliest_iteration",
                "no_informative_val": "retention_null; selection_uses0_then_loss_then_earliest",
                "rank_loss": "non_tie_pair_logistic; context_equal_weight; uniform_pair_minibatches256",
                "boost_target": "within_context_lex_key_ordinal_0_to1; ties_equal; single_level0; context_total_weight1",
                "label_scope": "observed_arrived_cohort_FCFS_finite_reference; not_expected_Q_or_action_imitation"}
    write_json(output/"protocol.json", protocol)
    x = torch.from_numpy(inputs.transform([r for c in train for r in c["features"]]))
    y = torch.from_numpy(targets.transform([r for c in train for r in c["targets"]]))
    pairs, weights = weighted_training_pairs(train)
    weights = torch.tensor(np.asarray(weights)*len(pairs), dtype=torch.float32)
    selected, verification_predictions = {}, {}
    for name in MODEL_NAMES:
        curve, best_key, best_model = [], None, None
        if name == "gradient_boost":
            model = GradientBoostingRegressor(**{**BOOSTING, "n_estimators": trees})
            ordinal = [v for c in train for v in ordinal_targets(c)]
            sample_weight = [1/len(c["features"]) for c in train for _ in c["features"]]
            model.fit(x.numpy(), ordinal, sample_weight=sample_weight)
            # staged_predict is the fitted ensemble's first1..T trees; no refitting.
            all_val_x = inputs.transform([r for c in val for r in c["features"]])
            staged = model.staged_predict(all_val_x)
            iterations = trees
        else:
            torch.manual_seed(TRAINING["seed"])
            rng = np.random.default_rng(TRAINING["seed"])
            model = nn.Linear(len(FEATURE_NAMES), 1) if name == "linear_rank" else MLP(7 if name == "value" else 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
            iterations = epochs
        for iteration in range(1, iterations+1):
            if name == "gradient_boost":
                stage_predictions = next(staged)
                candidate = deepcopy(model)
                candidate.estimators_ = candidate.estimators_[:iteration]
                candidate.train_score_ = candidate.train_score_[:iteration]
                candidate.n_estimators = candidate.n_estimators_ = iteration
                scorer = ValidationScorer(name, candidate, inputs)
                if not np.array_equal(candidate.predict(all_val_x), stage_predictions):
                    raise RuntimeError("saved boosting prefix differs from staged_predict")
            else:
                model.train()
                n = len(x) if name == "value" else len(pairs)
                order = rng.permutation(n)
                for start in range(0, n, TRAINING["batch_size"]):
                    indices = order[start:start+TRAINING["batch_size"]]
                    optimizer.zero_grad()
                    if name == "value":
                        loss = nn.functional.mse_loss(model(x[indices]), y[indices])
                    else:
                        first, second = [pairs[i][0] for i in indices], [pairs[i][1] for i in indices]
                        loss = (nn.functional.softplus(model(x[second]).flatten()-model(x[first]).flatten())*weights[indices]).mean()
                    loss.backward()
                    optimizer.step()
                scorer = ValidationScorer(name, model, inputs, targets if name == "value" else None)
            metrics = validation_metrics(val, scorer, TRAINING["checkpoint_budget"])
            val_loss = _loss(scorer, val)
            row = {"iteration": iteration, "iteration_unit": "tree" if name == "gradient_boost" else "epoch",
                   "train_loss": _loss(scorer, train), "val_loss": val_loss, **metrics}
            curve.append(row)
            key = selection_key(metrics["retention_at_total_budget"], val_loss)
            if best_key is None or key > best_key:
                best_key, best_model = key, deepcopy(scorer)
                selected[name] = dict(row)
            with (output/f"{name}_curve.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        path = output/MODEL_FILES[name]
        best_model.save(path, {**protocol, "selected_iteration": selected[name]["iteration"], "name": name})
        verification_predictions[name] = best_model.predict(val[0]["features"])
        selected[name]["training_status"] = "no_pairs_no_updates" if name in ("linear_rank", "mlp_rank") and not pairs else "completed_with_updates"
        write_json(output/f"{name}_training.json", {"selected": selected[name], "curve": curve})
    write_json(output/"training.json", {"status": "completed", "models": selected, "protocol": protocol})
    manifest = {"schema": SCHEMA, "producer": PRODUCER, "status": "completed", "frozen": True,
                "smoke_only": smoke, "test_read": False, "model_seed": TRAINING["seed"], "checkpoint_budget": 5,
                "source_commit": source_commit, "versions": protocol["versions"],
                "train_contexts": len(train), "val_contexts": len(val),
                "feature_names": list(FEATURE_NAMES), "files": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
                "models": {name: {"file": MODEL_FILES[name], "sha256": file_hash(output/MODEL_FILES[name]),
                                  "kind": "value" if name == "value" else "rank", "selected_iteration": selected[name]["iteration"]}
                           for name in MODEL_NAMES}, "roundtrip_predictions_equal": False}
    write_json(output/"manifest.json", manifest)
    for name in MODEL_NAMES:
        if ValidationScorer.load(output/MODEL_FILES[name]).predict(val[0]["features"]) != verification_predictions[name]:
            raise RuntimeError("saved model predictions differ after load: "+name)
    manifest["roundtrip_predictions_equal"] = True
    write_json(output/"manifest.json", manifest)
    return {"status": "completed", "models": selected, "manifest": str(output/"manifest.json")}


def train_validation_models(stage4_root, output, *, smoke=False):
    root = Path(stage4_root)/"data"
    loader = _load_pilot if smoke else lambda path: load_contexts(path, path.name, training=True)
    train, val = loader(root/"train"), loader(root/"val")
    if not smoke:
        for contexts, seeds in ((train, TRAIN_SEEDS), (val, VAL_SEEDS)):
            expected = {(seed, load, soc, time) for seed in seeds
                        for load in ("low", "medium", "high") for soc in ("high", "medium", "low")
                        for time in (0., 5., 15., 90.)}
            actual = {(c["metadata"]["seed"], c["metadata"]["load"], c["metadata"]["soc"],
                       c["metadata"]["root_time"]) for c in contexts}
            if actual != expected or len(contexts) != len(expected):
                raise ValueError("complete frozen stage4 train/val context coverage required")
    return train_models(train, val, output, smoke=smoke)
