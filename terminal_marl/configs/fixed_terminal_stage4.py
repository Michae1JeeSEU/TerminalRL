"""Frozen stage4 protocol; physical configuration is inherited unchanged."""
from terminal_marl.configs.fixed_terminal_stage3 import REFERENCE_SETTINGS

SPLIT_SEEDS = {"train": (4101, 4102, 4103), "val": (4201,), "test": (4301, 4302, 4303), "pilot": (4001,)}
SNAPSHOT_TIMES = (0., 5., 15., 90.)
METHODS = ("fcfs", "rolling_horizon", "full_reference", "beam_topk", "value_topk", "rank_topk")
POOL_KEEP = 12
TOP_K = 3
TRAINING = {"hidden_sizes": (64, 64), "learning_rate": .001, "weight_decay": .0001,
            "batch_size": 256, "max_epochs": 300, "seed": 4401, "threads": 1}


def jobs(split):
    return [{"split": split, "seed": seed, "load": load, "soc": soc}
            for seed in SPLIT_SEEDS[split] for load in ("low", "medium", "high")
            for soc in ("high", "medium", "low")]


def validate_split(split, seed):
    if split not in SPLIT_SEEDS or seed not in SPLIT_SEEDS[split]:
        raise ValueError("scenario seed is not in the preregistered split")
