"""Frozen stage5 validation budgets; the physical configuration is unchanged."""

TRAIN_SEEDS = (4101, 4102, 4103)
VAL_SEEDS = (4201,)
TEST_SEEDS = (5301, 5302, 5303)
PILOT_SEEDS = (5001,)
MODEL_NAMES = ("linear_rank", "mlp_rank", "gradient_boost", "value")
MODEL_FILES = {name: name + (".joblib" if name == "gradient_boost" else ".pt")
               for name in MODEL_NAMES}
BUDGETS = (3, 5, 8)
POOL_LIMIT = 14
LARGE_POOL_LIMIT = 32
SNAPSHOT_TIMES = (0., 15., 90.)
TRAINING = {"hidden_sizes": (64, 64), "learning_rate": .001, "weight_decay": .0001,
            "batch_size": 256, "max_epochs": 300, "seed": 5401, "threads": 1,
            "checkpoint_budget": 5, "smoke_iterations": 3}
BOOSTING = {"n_estimators": 100, "max_depth": 3, "learning_rate": .05,
            "random_state": 5401, "loss": "squared_error"}
ONLINE_ARMS = [
    {"name": "beam_b3", "method": "beam", "budget": 3},
    {"name": "beam_b5", "method": "beam", "budget": 5},
    {"name": "beam_b8", "method": "beam", "budget": 8},
    {"name": "full_reference", "method": "full", "budget": None},
    {"name": "rolling20_b5", "method": "rolling", "budget": 5, "rolling_seconds": 20.},
    {"name": "rolling60_b5", "method": "rolling", "budget": 5, "rolling_seconds": 60.},
    {"name": "linear_rank_b5", "method": "linear_rank", "budget": 5},
    {"name": "mlp_rank_b5", "method": "mlp_rank", "budget": 5},
    {"name": "gradient_boost_b5", "method": "gradient_boost", "budget": 5},
    {"name": "value_b5", "method": "value", "budget": 5},
    {"name": "hybrid_b5", "method": "hybrid", "budget": 5},
]


def scenario_jobs(split):
    """New test/pilot scenarios, including one rotated medium/medium per seed."""
    if split not in ("test", "pilot"):
        raise ValueError("stage5 scenario generation supports test or pilot only")
    seeds = TEST_SEEDS if split == "test" else PILOT_SEEDS
    jobs = [{"split": split, "seed": seed, "load": load, "soc": soc, "variant": "standard"}
            for seed in seeds for load in ("low", "medium", "high")
            for soc in ("high", "medium", "low")]
    jobs.extend({"split": split, "seed": seed, "load": "medium", "soc": "medium",
                 "variant": "initial_rotated"} for seed in seeds)
    return jobs
