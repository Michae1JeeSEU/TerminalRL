"""Create a compact server-training upload package and manifest."""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path


DEFAULT_PACKAGE_NAME = "terminal_server_training_package.zip"
DEFAULT_MANIFEST_NAME = "SERVER_UPLOAD_MANIFEST.txt"

INCLUDE_ROOTS = ("terminal_marl", "docs", "scripts/server")
OPTIONAL_ROOT_FILES = ("requirements.txt", "pyproject.toml", "README.md", "AGENTS.md")
TRAINING_TESTS = (
    "tests/test_training_config_loader.py",
    "tests/test_evaluate_high_level_policy_smoke.py",
    "tests/test_pack_for_server.py",
    "tests/test_reward_diagnostics.py",
    "tests/test_experiment_matrix.py",
    "tests/test_train_high_level_ppo_smoke.py",
    "tests/test_high_level_ppo_policy_eval.py",
)
EXCLUDE_PATTERNS = (
    ".git/*",
    "runs/*",
    "__pycache__/*",
    "*.pyc",
    ".pytest_cache/*",
    "*.zip",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.log",
    "events.out.tfevents*",
)


def _is_excluded(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    parts = normalized.split("/")
    if any(part in {"__pycache__", ".pytest_cache", ".git", "runs"} for part in parts):
        return True
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in EXCLUDE_PATTERNS)


def collect_server_files(project_root: str | Path = ".") -> list[Path]:
    """Return relative paths that should be uploaded to a Linux server."""

    root = Path(project_root)
    files: list[Path] = []
    for include_root in INCLUDE_ROOTS:
        base = root / include_root
        if base.is_file() and not _is_excluded(include_root):
            files.append(Path(include_root))
        elif base.is_dir():
            for path in base.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(root)
                    if not _is_excluded(str(relative)):
                        files.append(relative)
    for test in TRAINING_TESTS:
        if (root / test).is_file() and not _is_excluded(test):
            files.append(Path(test))
    for filename in OPTIONAL_ROOT_FILES:
        if (root / filename).is_file():
            files.append(Path(filename))
    return sorted(set(files), key=lambda path: path.as_posix())


def write_manifest(files: list[Path], manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Server upload manifest",
        "# Upload these project-relative files/directories for simulation-only high-level PPO training.",
        "",
        "Required top-level content:",
        "- terminal_marl/",
        "- docs/",
        "- scripts/server/",
        "- selected tests/",
        "- requirements.txt or pyproject.toml/README if present",
        "",
        "Files:",
        *[path.as_posix() for path in files],
        "",
    ]
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def pack_for_server(
    *,
    project_root: str | Path = ".",
    output_zip: str | Path = DEFAULT_PACKAGE_NAME,
    manifest_path: str | Path = DEFAULT_MANIFEST_NAME,
) -> dict[str, str | int]:
    """Write a zip package plus ``SERVER_UPLOAD_MANIFEST.txt``."""

    root = Path(project_root)
    files = collect_server_files(root)
    manifest = write_manifest(files, root / manifest_path)
    package = root / output_zip
    package.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(root / relative, relative.as_posix())
        archive.write(manifest, Path(manifest_path).as_posix())
    return {
        "package": str(package),
        "manifest": str(manifest),
        "file_count": len(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-zip", type=Path, default=Path(DEFAULT_PACKAGE_NAME))
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST_NAME))
    args = parser.parse_args()
    print(pack_for_server(
        project_root=args.project_root,
        output_zip=args.output_zip,
        manifest_path=args.manifest,
    ))


if __name__ == "__main__":
    main()

