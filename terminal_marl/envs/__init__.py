"""Load legacy environments only on explicit named access."""
from importlib import import_module


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = ".parallel_env" if name == "TerminalParallelEnv" else ".high_level_wrapper"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

__all__ = ["TerminalParallelEnv", "DynamicHighLevelWrapper", "FrozenLowLevelPolicy", "RuleLowLevelAdapter"]
