"""Optional, read-only live visualization for terminal-MARL rollouts.

Importing this package never imports PyVista.  Ordinary training and headless
evaluation therefore remain independent of visualization dependencies.
"""

from .config import VisualizationConfig
from .snapshot import build_world_render_snapshot

__all__ = ["VisualizationConfig", "build_world_render_snapshot"]
