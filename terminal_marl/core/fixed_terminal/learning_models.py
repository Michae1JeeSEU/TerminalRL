"""CPU MLP value regression and context-balanced pairwise logistic ranking."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from torch import nn
from .learning_features import FEATURE_NAMES
from .learning_scheduler import prediction_order, shortlist
from .reference import value_comparison_key, action_key
from .learning_data import write_json
from terminal_marl.configs.fixed_terminal_stage4 import TRAINING, SPLIT_SEEDS


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values):
        values = np.asarray(values, dtype=np.float64)
        mean, scale = values.mean(axis=0), values.std(axis=0)
        return cls(mean, np.where(scale < 1e-8, 1., scale))

    def transform(self, values):
        return ((np.asarray(values, dtype=np.float64)-self.mean)/self.scale).astype(np.float32)

    def inverse(self, values):
        return np.asarray(values, dtype=np.float64)*self.scale+self.mean

    def serializable(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "fit_scope": "train_only"}


class MLP(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))

    def forward(self, x):
        return self.layers(x)


class Scorer:
    def __init__(self, kind, model, inputs, targets=None):
        torch.set_num_threads(1)
        self.kind, self.model, self.inputs, self.targets = kind, model.cpu().eval(), inputs, targets

    def predict(self, values):
        torch.set_num_threads(1)
        with torch.no_grad():
            output = self.model(torch.from_numpy(self.inputs.transform(values))).cpu().numpy()
        output = self.targets.inverse(output) if self.kind == "value" else output
        if not np.isfinite(output).all():
            raise ValueError("nonfinite predictions")
        return output.tolist()

    def save(self, path, protocol):
        torch.save({"kind": self.kind, "feature_names": list(FEATURE_NAMES), "state_dict": self.model.state_dict(),
                    "inputs": self.inputs.serializable(), "targets": self.targets.serializable() if self.targets else None,
                    "protocol": protocol}, path)

    @classmethod
    def load(cls, path):
        torch.set_num_threads(1)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("model feature schema mismatch")
        model = MLP(7 if data["kind"] == "value" else 1)
        model.load_state_dict(data["state_dict"])
        def scaler(row):
            return Standardizer(np.asarray(row["mean"]), np.asarray(row["scale"])) if row else None
        result = cls(data["kind"], model, scaler(data["inputs"]), scaler(data["targets"]))
        result.protocol = data["protocol"]
        return result


def root_objects(context):
    return [SimpleNamespace(root_id=r["root_id"], plans=tuple(SimpleNamespace(aircraft_id=p["aircraft_id"], wait_cycles=tuple(p["wait_cycles"])) for p in r["plans"])) for r in context["roots"]]


def true_pairs(context):
    keys = [value_comparison_key(v) for v in context["values"]]
    return [(i, j) if keys[i] > keys[j] else (j, i)
            for i in range(len(keys)) for j in range(i+1, len(keys)) if keys[i] != keys[j]]


def weighted_training_pairs(contexts):
    """Global root indices; every informative context has equal total weight."""
    pairs, weights, offset = [], [], 0
    informative = sum(bool(true_pairs(c)) for c in contexts)
    for context in contexts:
        local = true_pairs(context)
        pairs.extend((i+offset, j+offset) for i, j in local)
        weights.extend([1/(len(local)*informative)]*len(local) if local else [])
        offset += len(context["features"])
    return pairs, weights


def context_metrics(context, method, scorer=None):
    roots = root_objects(context)
    values = context["values"]
    keys = [value_comparison_key(v) for v in values]
    predictions = scorer.predict(context["features"]) if scorer else None
    if method == "beam_topk":
        order = list(context["pool_metadata"]["beam_order"])
        # All roots not in beam order (e.g. FCFS anchor) follow deterministically.
        order += [r.root_id for r in sorted(roots, key=lambda r: action_key(r.plans)) if r.root_id not in order]
    else:
        order = prediction_order(roots, predictions, method.split("_")[0])
    by_id = {r.root_id: i for i, r in enumerate(roots)}
    ranked = [by_id[key] for key in order]
    best = max(keys)
    selected = shortlist(roots, context["pool_metadata"], method, predictions)
    best_filtered = max(keys[by_id[r.root_id]] for r in selected)
    index = min((by_id[r.root_id] for r in selected if keys[by_id[r.root_id]] == best_filtered), key=lambda i: action_key(roots[i].plans))
    oracle_index = min((i for i, k in enumerate(keys) if k == best), key=lambda i: action_key(roots[i].plans))
    positions = {idx: rank for rank, idx in enumerate(ranked)}
    pairs = true_pairs(context)
    return {"context": context["path"], "root_count": len(roots), "degenerate": len(roots) == 1,
            "informative": len(set(keys)) > 1, "non_tie_pairs": len(pairs),
            "pair_correct": sum(positions[i] < positions[j] for i, j in pairs),
            "top1_best_hit": keys[ranked[0]] == best, "top3_best_hit": any(keys[i] == best for i in ranked[:3]),
            "filtered_root_count": len(selected), "filtered_chosen": roots[index].root_id,
            "full_best_root": roots[oracle_index].root_id, "comparison": "tie" if best_filtered == best else "loss",
            "raw_J_regret_full_minus_filtered": [a-b for a, b in zip(values[oracle_index]["J"], values[index]["J"])],
            "raw_component_regret_full_minus_filtered": {k: values[oracle_index]["components"][k]-values[index]["components"][k] for k in values[index]["components"]},
            "raw_delta_component_absolute_errors": np.abs(np.asarray(predictions)-np.asarray(context["targets"])).tolist() if scorer and scorer.kind == "value" else None}


def offline_metrics(contexts, method, scorer=None):
    rows = [context_metrics(c, method, scorer) for c in contexts]
    groups = {}
    for group in ("all", "degenerate", "nondegenerate", "informative"):
        selected = [r for r in rows if group == "all" or (not r["degenerate"] if group == "nondegenerate" else r[group])]
        errors = [e for r in selected for e in (r["raw_delta_component_absolute_errors"] or [])]
        pairs = sum(r["non_tie_pairs"] for r in selected)
        groups[group] = {"contexts": len(selected), "non_tie_pairs": pairs,
                         "pair_accuracy": sum(r["pair_correct"] for r in selected)/pairs if pairs else None,
                         "top1_best_hit": sum(r["top1_best_hit"] for r in selected)/len(selected) if selected else None,
                         "top3_best_recall": sum(r["top3_best_hit"] for r in selected)/len(selected) if selected else None,
                         "filtered_vs_full": {k: sum(r["comparison"] == k for r in selected) for k in ("win", "tie", "loss")},
                         "raw_delta_J_MAE": np.mean(errors, axis=0).tolist() if errors else None}
    return {"method": method, "groups": groups, "contexts": rows,
            "oracle_scope": "same_finite_candidate_pool_exact_observed_FCFS_values_not_global_optimum"}


def validate_training_groups(train, val, smoke=False):
    if smoke:
        if any(c["metadata"]["split"] != "pilot" for c in train+val):
            raise ValueError("smoke may only use pilot contexts")
        return
    for group, split in ((train, "train"), (val, "val")):
        if any(c["metadata"]["split"] != split or c["metadata"]["seed"] not in SPLIT_SEEDS[split] for c in group):
            raise ValueError("train/val split violated; test is never accepted")
    if {c["metadata"]["seed"] for c in train} & {c["metadata"]["seed"] for c in val}:
        raise ValueError("scenario seed leakage")


def _loss(scorer, contexts):
    with torch.no_grad():
        if scorer.kind == "value":
            x = torch.from_numpy(scorer.inputs.transform([r for c in contexts for r in c["features"]]))
            y = torch.from_numpy(scorer.targets.transform([r for c in contexts for r in c["targets"]]))
            return float(nn.functional.mse_loss(scorer.model(x), y))
        losses = []
        for context in contexts:
            pairs = true_pairs(context)
            if pairs:
                prediction = scorer.model(torch.from_numpy(scorer.inputs.transform(context["features"]))).flatten()
                losses.append(nn.functional.softplus(torch.stack([prediction[j]-prediction[i] for i, j in pairs])).mean())
        return float(torch.stack(losses).mean()) if losses else 0.


def train_models(train, val, output, *, epochs=300, smoke=False):
    validate_training_groups(train, val, smoke)
    if not 1 <= epochs <= TRAINING["max_epochs"] or (not smoke and epochs != TRAINING["max_epochs"]):
        raise ValueError("formal training uses the frozen300 epochs; shorter runs are pilot smoke only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    inputs = Standardizer.fit([row for c in train for row in c["features"]])
    targets = Standardizer.fit([row for c in train for row in c["targets"]])
    protocol = {**TRAINING, "epochs": epochs, "smoke_only": smoke, "feature_names": FEATURE_NAMES,
                "train_contexts": len(train), "val_contexts": len(val), "test_read": False,
                "train_candidate_count": sum(len(c["features"]) for c in train),
                "rank_train_non_tie_pair_count": sum(len(true_pairs(c)) for c in train),
                "rank_train_informative_context_count": sum(bool(true_pairs(c)) for c in train),
                "train_seeds": sorted({c["metadata"]["seed"] for c in train}), "val_seeds": sorted({c["metadata"]["seed"] for c in val}),
                "checkpoint_selection": "maximum_val_informative_candidate_top3_best_key_recall; then_minimum_val_loss; then_earliest_epoch",
                "no_informative_val_fallback": "recall0_then_val_loss_then_earliest_epoch",
                "pair_loss": "non_tie_pairs_logistic; each_context_equal_total_weight; uniform_pair_minibatches256",
                "inputs": inputs.serializable(), "value_targets": targets.serializable(),
                "interpretation": "observed_cohort_FCFS_value_acceleration_not_action_imitation_or_expected_future_Q"}
    write_json(output/"protocol.json", protocol)
    results = {}
    for kind in ("value", "rank"):
        torch.manual_seed(TRAINING["seed"])
        rng = np.random.default_rng(TRAINING["seed"])
        scorer = Scorer(kind, MLP(7 if kind == "value" else 1), inputs, targets if kind == "value" else None)
        optimizer = torch.optim.Adam(scorer.model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"])
        all_x = [row for c in train for row in c["features"]]
        x = torch.from_numpy(inputs.transform(all_x))
        if kind == "value":
            y = torch.from_numpy(targets.transform([row for c in train for row in c["targets"]]))
            n = len(x)
        else:
            pairs, weights = weighted_training_pairs(train)
            n = len(pairs)
            weights = torch.tensor(np.asarray(weights)*n, dtype=torch.float32)
        best, curve = None, []
        for epoch in range(1, epochs+1):
            scorer.model.train()
            for start in range(0, n, TRAINING["batch_size"]):
                if start == 0:
                    order = rng.permutation(n)
                indices = order[start:start+TRAINING["batch_size"]]
                optimizer.zero_grad()
                if kind == "value":
                    loss = nn.functional.mse_loss(scorer.model(x[indices]), y[indices])
                else:
                    first = [pairs[i][0] for i in indices]
                    second = [pairs[i][1] for i in indices]
                    difference = scorer.model(x[second]).flatten()-scorer.model(x[first]).flatten()
                    loss = (nn.functional.softplus(difference)*weights[indices]).mean()
                loss.backward()
                optimizer.step()
            scorer.model.eval()
            validation = offline_metrics(val, kind+"_topk", scorer)
            recall = validation["groups"]["informative"]["top3_best_recall"]
            val_loss = _loss(scorer, val)
            key = (recall if recall is not None else 0., -val_loss)
            row = {"epoch": epoch, "train_loss": _loss(scorer, train), "val_loss": val_loss,
                   "val_informative_top3_best_recall": recall, "val_informative_contexts": validation["groups"]["informative"]["contexts"]}
            curve.append(row)
            if best is None or key > best:
                best = key
                results[kind] = {"best_epoch": epoch, **row}
                scorer.save(output/f"{kind}.pt", {**protocol, "selected_epoch": epoch, "kind": kind})
            with (output/f"{kind}_curve.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        results[kind]["training_status"] = "no_pairs_no_updates" if kind == "rank" and n == 0 else "completed_with_updates"
        write_json(output/f"{kind}_training.json", {"selected": results[kind], "curve": curve})
    write_json(output/"training.json", {"status": "completed", "models": results, "protocol": protocol})
    return results
