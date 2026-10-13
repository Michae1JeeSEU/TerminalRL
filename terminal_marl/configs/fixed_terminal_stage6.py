"""Frozen stage6 search budgets; all physical configuration stays in stage1."""

TEST_SEEDS = (6301, 6302, 6303)
PILOT_SEEDS = (6001,)
POOL_LIMIT = 14
BUDGETS = (3, 5, 8)
PANEL_TIMES = (0., 60.)
EPSILON_NORMAL = 0.0
COMPUTE_REDUCTION_TARGET = 0.10  # Descriptive threshold, not significance.

ONLINE_ARMS = [
    {"name": "beam_b3", "method": "beam", "budget": 3, "adaptive": False, "model_name": None},
    {"name": "beam_b5", "method": "beam", "budget": 5, "adaptive": False, "model_name": None},
    {"name": "beam_b8", "method": "beam", "budget": 8, "adaptive": False, "model_name": None},
    {"name": "full_reference", "method": "full", "budget": None, "adaptive": False, "model_name": None},
    {"name": "beam_adaptive", "method": "beam", "budget": 3, "adaptive": True, "model_name": None},
    {"name": "rank_adaptive", "method": "mlp_rank", "budget": 3, "adaptive": True, "model_name": "mlp_rank"},
    {"name": "gb_adaptive", "method": "gradient_boost", "budget": 3, "adaptive": True, "model_name": "gradient_boost"},
    {"name": "hybrid_adaptive", "method": "hybrid", "budget": 3, "adaptive": True, "model_name": "mlp_rank"},
]
ARMS = tuple(arm["name"] for arm in ONLINE_ARMS)


def scenario_jobs(split):
    """27 standard + six initial-occupancy variants for the three test seeds."""
    if split not in ("test", "pilot"):
        raise ValueError("stage6 scenarios support test or pilot only")
    seeds = TEST_SEEDS if split == "test" else PILOT_SEEDS
    jobs = [{"split": split, "seed": seed, "load": load, "soc": soc, "variant": "standard"}
            for seed in seeds for load in ("low", "medium", "high")
            for soc in ("high", "medium", "low")]
    jobs.extend({"split": split, "seed": seed, "load": "medium", "soc": "medium", "variant": variant}
                for seed in seeds for variant in ("initial_sparse", "initial_dense"))
    return jobs


def panel_scenario_jobs(split):
    """15 test / five pilot scenarios; each job evaluates both PANEL_TIMES."""
    profiles = {("low", "high"), ("medium", "medium"), ("high", "low")}
    return [job for job in scenario_jobs(split)
            if job["variant"] != "standard" or (job["load"], job["soc"]) in profiles]


def panel_jobs(split):
    """State inventory only; the dispatcher groups these by scenario."""
    return [{**job, "time": time} for job in panel_scenario_jobs(split) for time in PANEL_TIMES]
