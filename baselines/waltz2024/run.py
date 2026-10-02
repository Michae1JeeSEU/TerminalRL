"""Command-line entry point for the Waltz et al. (2024) UAM wrapper."""

try:
    from .runner import main
except ImportError:  # direct ``python run.py`` execution
    from runner import main


if __name__ == "__main__":
    raise SystemExit(main())
