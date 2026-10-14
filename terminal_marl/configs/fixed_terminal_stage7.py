"""Frozen partial-plan search protocol; physical settings remain in stage1."""
from terminal_marl.configs.fixed_terminal_stage3 import REFERENCE_SETTINGS

TRAIN_SEEDS = (4101, 4102, 4103)
VAL_SEEDS = (4201,)
TEST_SEEDS = (7301, 7302, 7303)
PILOT_SEEDS = (7001,)
SPLIT_SEEDS = {"train": TRAIN_SEEDS, "val": VAL_SEEDS, "test": TEST_SEEDS, "pilot": PILOT_SEEDS}
MODEL_SEED = 7401
BUDGETS = (3, 5, 8)
REFERENCE_WIDTH = 128
TRACE_WIDTHS = (3, 5, 8, REFERENCE_WIDTH)
POOL_LIMIT = 14
ROOT_KEEP = 12
SNAPSHOT_TIMES = (0., 5., 15., 90.)
PANEL_TIMES = (0., 60.)
EXPAND_THRESHOLD = .5
EPSILON_NORMAL = 0.
COMPUTE_REDUCTION_TARGET = .10
TRAINING = {"n_estimators": 100, "max_depth": 3, "learning_rate": .05, "random_state": MODEL_SEED}
ONLINE_ARMS = [
    {"name": "beam_b3", "method": "beam", "budget": 3, "adaptive": False, "model_name": None},
    {"name": "beam_b5", "method": "beam", "budget": 5, "adaptive": False, "model_name": None},
    {"name": "beam_b8", "method": "beam", "budget": 8, "adaptive": False, "model_name": None},
    {"name": "beam_adaptive", "method": "beam", "budget": 3, "adaptive": True, "model_name": None},
    {"name": "full_reference", "method": "full", "budget": REFERENCE_WIDTH, "adaptive": False, "model_name": None},
    {"name": "learned_pruning", "method": "learned_pruning", "budget": 8, "adaptive": True, "model_name": "gradient_boost_expand"},
    {"name": "learned_beam_width", "method": "learned_beam_width", "budget": 8, "adaptive": True, "model_name": "gradient_boost_expand"},
    {"name": "hybrid", "method": "hybrid", "budget": 8, "adaptive": True, "model_name": "gradient_boost_expand"},
]
ARMS = tuple(a["name"] for a in ONLINE_ARMS)


def validate_split(split, seed):
    if split not in SPLIT_SEEDS or seed not in SPLIT_SEEDS[split]:
        raise ValueError("scenario seed outside frozen stage7 split")


def scenario_jobs(split):
    if split not in SPLIT_SEEDS:
        raise ValueError("unknown split")
    jobs = [{"split": split, "seed": seed, "load": load, "soc": soc, "variant": "standard"}
            for seed in SPLIT_SEEDS[split] for load in ("low", "medium", "high")
            for soc in ("high", "medium", "low")]
    if split in ("test", "pilot"):
        jobs.extend({"split": split, "seed": seed, "load": "medium", "soc": "medium", "variant": variant}
                    for seed in SPLIT_SEEDS[split] for variant in ("initial_sparse", "initial_dense"))
    return jobs


def panel_scenario_jobs(split):
    if split not in ("test", "pilot"):
        raise ValueError("panels support test/pilot only")
    profiles = {("low", "high"), ("medium", "medium"), ("high", "low")}
    return [j for j in scenario_jobs(split) if j["variant"] != "standard" or (j["load"], j["soc"]) in profiles]


def panel_jobs(split):
    return [{**job, "time": time} for job in panel_scenario_jobs(split) for time in PANEL_TIMES]
