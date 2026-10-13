"""Fixed-terminal arrival scheduling research with lazy legacy compatibility."""
from importlib import import_module


def __getattr__(name):
    modules = {"SimConfig": ".core.config", "RingRole": ".core.config",
               "TerminalParallelEnv": ".envs.parallel_env"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

__all__ = ["SimConfig", "RingRole", "TerminalParallelEnv"]
