"""Local-only stage6 delivery ZIP with explicit scope and completion checks."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import zipfile


DOC_FILES = (
    "AGENTS.md", "README.md", "docs/README.md", "docs/CURRENT_STATUS.md",
    "docs/execution_plan.md", "docs/research_positioning.md", "docs/requirements.md",
    "docs/design_spec.md", "docs/technical_spec.md", "docs/fixed_terminal_problem_definition.md",
    "docs/fixed_terminal_learning_problem_definition.md",
    "docs/fixed_terminal_reconstruction_plan_stage1.md",
    "docs/fixed_terminal_stage2_traditional_baseline_plan.md",
    "docs/fixed_terminal_stage3_learning_problem_definition_plan.md",
    "docs/fixed_terminal_stage4_learning_scheduler_plan.md",
    "docs/fixed_terminal_stage5_learning_validation_plan.md",
    "docs/arrival_restructure/stage5_learning_validation.md",
    "docs/experiments/fixed_terminal_stage5_protocol.md",
)
SOURCE_FILES = (
    "terminal_marl/__init__.py", "terminal_marl/core/__init__.py",
    "terminal_marl/envs/__init__.py",
    "terminal_marl/envs/fixed_terminal_env.py",
    "requirements-stage2.txt", "requirements-stage4.txt", "requirements-stage5.txt",
)
SOURCE_GLOBS = (
    "terminal_marl/core/fixed_terminal/*.py", "terminal_marl/configs/fixed_terminal_stage*.py",
    "terminal_marl/scripts/*fixed_terminal_stage*.py", "tests/test_fixed_terminal*.py",
)
from terminal_marl.scripts.summarize_fixed_terminal_stage6 import SUMMARY_COUNTS
DOC_FILES += ("docs/fixed_terminal_stage6_learning_assisted_scheduling_plan.md",
              "docs/arrival_restructure/stage6_learning_assisted_scheduling.md",
              "docs/experiments/fixed_terminal_stage6_protocol.md")



def read_json(path):
    def invalid(value):
        raise ValueError("non-finite JSON: " + value)
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid)


def local_head(repository):
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def safe_file(path, root):
    path, root = Path(path), Path(root)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("source escapes allowed root: " + str(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("symlink source is not supported: " + str(part))
        if part == root:
            break
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def tree_files(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("required ordinary directory absent: " + str(root))
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in included tree: " + str(path))
        if "__pycache__" in path.relative_to(root).parts or path.suffix == ".pyc":
            continue
        if path.is_file():
            yield safe_file(path, root)


def completion_checks(repository, results, stage5, final_commit):
    if re.fullmatch(r"[0-9a-f]{40}", final_commit) is None:
        raise ValueError("--final-commit must be a full lowercase SHA")
    if local_head(repository) != final_commit:
        raise ValueError("local HEAD differs from --final-commit")
    from terminal_marl.scripts.summarize_fixed_terminal_stage6 import summarize
    actual = summarize(results)
    saved = read_json(safe_file(results / "summary/summary.json", results))
    # Replay aggregation from raw files, without executing models or simulation.
    def portable(record):
        # Syncthing relocates the same evidence from the server to local Windows.
        record = json.loads(json.dumps(record))
        record.pop("root", None)
        for row in record.get("episode_rows", []) + record.get("same_state", {}).get("rows", []):
            row.pop("source", None)
        return record
    if portable(saved) != portable(actual) or actual["counts"] != SUMMARY_COUNTS or actual["status"] != "completed":
        raise ValueError("formal stage6 summary is missing/incomplete or differs from raw")
    list(tree_files(stage5 / "models"))
    from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import frozen_models
    current = frozen_models(stage5 / "models", smoke=False)
    for phase in ("episodes", "panels"):
        for key in ("frozen_models_before", "frozen_models_after"):
            recorded = actual["execution"][phase][key]
            if recorded["manifest_sha256"] != current["manifest_sha256"] or recorded["manifest"] != current["manifest"]:
                raise ValueError("stage5 model bytes differ from stage6 evaluation freeze")
    return {"episode_jobs": 264, "panel_jobs": 15, "summary_counts": SUMMARY_COUNTS,
            "manifest_sha256": current["manifest_sha256"], "model_hashes_unchanged": True}


def collect_sources(repository, results, stage5):
    entries = {}
    def add(path, root, prefix=""):
        path = safe_file(path, root)
        name = (Path(prefix) / path.relative_to(root)).as_posix()
        if name in entries:
            raise ValueError("duplicate archive entry: " + name)
        entries[name] = path
    for name in DOC_FILES + SOURCE_FILES:
        add(repository / name, repository)
    for pattern in SOURCE_GLOBS:
        for source in sorted(repository.glob(pattern)):
            add(source, repository)
    for source in tree_files(repository / "docs/evidence/fixed_terminal_stage6"):
        add(source, repository)
    for source in tree_files(results):
        add(source, results, "results/stage6")
    model_root = stage5 / "models"
    manifest_path = safe_file(model_root / "manifest.json", model_root)
    manifest = read_json(manifest_path)
    add(manifest_path, stage5, "inputs/stage5")
    for name in sorted(manifest["files"]):
        if (not isinstance(name, str) or name in ("", ".", "..")
                or Path(name).name != name or any(char in name for char in ("/", "\\", ":"))):
            raise ValueError("unsafe frozen model manifest filename: " + repr(name))
        add(safe_file(model_root / name, model_root), stage5, "inputs/stage5")
    preflight = repository / "tmp_stage6_preflight"
    if preflight.is_dir():
        for source in sorted(preflight.rglob("*")):
            if source.is_file() and (source.suffix == ".log" or source.name in ("validation.json", "core_validation.json")):
                add(source, preflight, "preflight")
    return dict(sorted(entries.items()))


def bundle_readme(commit):
    return f"""# 阶段6文档、冻结模型与结果包

源码分支：codex/arrival-restructure；最终提交：{commit}。
阅读顺序：docs/CURRENT_STATUS.md → docs/experiments/fixed_terminal_stage6_protocol.md
→ docs/arrival_restructure/stage6_learning_assisted_scheduling.md → docs/evidence/fixed_terminal_stage6/。
results/stage6保存全部本阶段原始输出及严格summary；inputs/stage5/models保存四冻结模型、manifest及其files引用的协议/训练记录/曲线等辅助文件，满足冻结模型完整校验；不包含未被manifest引用的阶段5原始结果。
当前fixed核心/config/scripts/tests、必要依赖、用户阶段1–6计划随包保存。未包含阶段4/5全部raw；未收录历史证据可在原仓库阅读。

模型只排序初始TOTAL B3；扩搜与最终执行依赖确定性层。闭合证书仅针对已到达机群FCFS continuation，不是长期Q、未来到达或全J最优证明。
模型不需重新训练；环境按requirements-stage5.txt和模型manifest配置。各phase不得重叠，panel单worker，数值库threads=1。
使用新的输出目录重现，禁止覆盖原始结果：
python -m terminal_marl.scripts.run_fixed_terminal_stage6 --phase episodes --models inputs/stage5/models --output reproduction/results --workers 20 --execute
python -m terminal_marl.scripts.run_fixed_terminal_stage6 --phase panels --models inputs/stage5/models --output reproduction/results --workers 1 --execute
python -m terminal_marl.scripts.summarize_fixed_terminal_stage6 --root reproduction/results --output reproduction/results/summary
来源提交通过TERMINAL_SOURCE_COMMIT传入；实际版本、环境、四模型hash和计时范围见原始产物。

MANIFEST.json列内容文件SHA256和bytes，不自包含自身hash。包外validation.json记录CRC/逐entry检查，zip.sha256记录完整ZIP哈希。
"""


def package(repository, results_root, stage5_root, output_dir, final_commit, build=False):
    repository, results, stage5, output = map(Path, (repository, results_root, stage5_root, output_dir))
    target = output / f"terminal_stage6_documents_models_results_{final_commit[:7]}.zip"
    # Dry-run reports missing completion evidence without creating any directory.
    try:
        checks = completion_checks(repository, results, stage5, final_commit)
        entries = collect_sources(repository, results, stage5)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        if build:
            raise
        return {"status": "not_ready", "reason": str(exc), "writes": False,
                "scope": ["current docs/fixed-terminal source", "results/stage6", "inputs/stage5/models", "preflight"],
                "target": str(target)}
    if not build:
        return {"status": "ready", "writes": False, "checks": checks, "target": str(target),
                "source_files": len(entries), "entries": list(entries)}
    if output.resolve().is_relative_to(results.resolve()) or output.resolve().is_relative_to(stage5.resolve()):
        raise ValueError("delivery output must not be inside an input tree")
    for destination in (target, output / "validation.json", output / "zip.sha256"):
        if destination.exists():
            raise FileExistsError("refusing to overwrite delivery: " + str(destination))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "fixed-terminal-stage6-bundle-v1", "final_commit": final_commit, "files": {}}
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, source in entries.items():
            digest, size = sha256(), 0
            with source.open("rb") as incoming, archive.open(name, "w", force_zip64=True) as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    outgoing.write(chunk)
            manifest["files"][name] = {"sha256": digest.hexdigest(), "bytes": size}
        readme = bundle_readme(final_commit).encode("utf-8")
        archive.writestr("BUNDLE_README.md", readme)
        manifest["files"]["BUNDLE_README.md"] = {"sha256": sha256(readme).hexdigest(), "bytes": len(readme)}
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        archive.writestr("MANIFEST.json", manifest_bytes)
    expected = {**manifest["files"], "MANIFEST.json": {"sha256": sha256(manifest_bytes).hexdigest(), "bytes": len(manifest_bytes)}}
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError("ZIP CRC or entry inventory failed")
        for name, specification in expected.items():
            digest, size = sha256(), 0
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != specification["sha256"] or size != specification["bytes"]:
                raise ValueError("ZIP entry verification failed: " + name)
    with target.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result = {"status": "completed", "final_commit": final_commit, "zip": str(target),
              "bytes": target.stat().st_size, "entries": len(expected), "sha256": digest.hexdigest(),
              "crc_valid": True, "entry_hashes_valid": True, "checks": checks}
    with (output / "validation.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    with (output / "zip.sha256").open("x", encoding="ascii") as stream:
        stream.write(f"{digest.hexdigest()}  {target.name}\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results-root", "stage5-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--final-commit", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = package(Path(__file__).resolve().parents[2], args.results_root, args.stage5_root,
                         args.output_dir, args.final_commit, args.build)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in ("ready", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
