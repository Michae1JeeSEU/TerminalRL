"""Inject the inactive candidate only into this local test process."""
from pathlib import Path
import importlib.util
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
name = 'terminal_marl.core.polynomial_trajectory'
spec = importlib.util.spec_from_file_location(name, root / '.tmp/p3_polynomial_candidate_20261005.py')
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
import terminal_marl.core
setattr(terminal_marl.core, 'polynomial_trajectory', module)
from terminal_marl.core.trajectory_time_parameterization import prime_quintic_bounds
assert prime_quintic_bounds is module.prime_quintic_bounds
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
