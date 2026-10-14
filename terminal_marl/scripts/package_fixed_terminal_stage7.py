"""Local-only stage7 delivery ZIP with explicit scope and completion checks."""
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
    "docs/fixed_terminal_stage6_learning_assisted_scheduling_plan.md",
    "docs/arrival_restructure/stage6_learning_assisted_scheduling.md",
    "docs/experiments/fixed_terminal_stage6_protocol.md",
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
from terminal_marl.scripts.summarize_fixed_terminal_stage7 import SUMMARY_COUNTS
DOC_FILES += ("docs/fixed_terminal_stage7_learning_to_search_plan.md",
              "docs/arrival_restructure/stage7_learning_to_search.md",
              "docs/experiments/fixed_terminal_stage7_protocol.md")



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


def completion_checks(repository, results, models, final_commit):
    if re.fullmatch(r"[0-9a-f]{40}", final_commit) is None:
        raise ValueError("--final-commit must be a full lowercase SHA")
    if local_head(repository) != final_commit:
        raise ValueError("local HEAD differs from --final-commit")
    from terminal_marl.scripts.summarize_fixed_terminal_stage7 import summarize
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
        raise ValueError("formal stage7 summary is missing/incomplete or differs from raw")
    list(tree_files(models))
    from terminal_marl.scripts.evaluate_fixed_terminal_stage7 import frozen_models
    current = frozen_models(models, smoke=False)
    for phase in ("episodes", "panels"):
        for key in ("frozen_models_before", "frozen_models_after"):
            recorded = actual["execution"][phase][key]
            if recorded["manifest_sha256"] != current["manifest_sha256"] or recorded["manifest"] != current["manifest"]:
                raise ValueError("stage7 model bytes differ from evaluation freeze")
    return {"episode_jobs": 264, "panel_jobs": 15, "summary_counts": SUMMARY_COUNTS,
            "manifest_sha256": current["manifest_sha256"], "model_hashes_unchanged": True}


def collect_sources(repository, results, models):
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
    for source in tree_files(repository / "docs/evidence/fixed_terminal_stage7"):
        add(source, repository)
    for source in tree_files(results):
        add(source, results, "results/stage7")
    model_root = models
    manifest_path = safe_file(model_root / "manifest.json", model_root)
    manifest = read_json(manifest_path)
    add(manifest_path, models, "inputs/stage7/models")
    for name in sorted(manifest["files"]):
        if (not isinstance(name, str) or name in ("", ".", "..")
                or Path(name).name != name or any(char in name for char in ("/", "\\", ":"))):
            raise ValueError("unsafe frozen model manifest filename: " + repr(name))
        add(safe_file(model_root / name, model_root), models, "inputs/stage7/models")
    preflight = repository / "tmp_stage7_preflight"
    if preflight.is_dir():
        for source in sorted(preflight.rglob("*")):
            if source.is_file() and (source.suffix == ".log" or source.name in ("validation.json", "core_validation.json")):
                add(source, preflight, "preflight")
    return dict(sorted(entries.items()))


def bundle_readme(commit):
    return f"""# 阶段7文档、冻结模型与结果包

源码分支：codex/arrival-restructure；最终提交：{commit}。
阅读顺序：docs/CURRENT_STATUS.md → docs/experiments/fixed_terminal_stage7_protocol.md
→ docs/arrival_restructure/stage7_learning_to_search.md → docs/evidence/fixed_terminal_stage7/。
results/stage7保存本阶段全部原始数据、partial节点trace、训练/validation分析、264episodes、15paneljobs及严格summary。
inputs/stage7/models保存一次训练冻结的GradientBoostingClassifier、manifest及其files引用的全部辅助文件。
用户阶段7计划、当前fixed核心/config/scripts/tests、必要依赖及阶段1–6历史计划随包保存。

学习控制partial计划展开、剪枝和beam宽度；最终执行仍由独立确定性层验证。full_reference是固定catalogue上width128、top12+HOLD/liveFCFS的有限参考，不是全域oracle。
证书仅针对当前已到达机群FCFS闭合正常完成上界，不能证明全J最优或长期Q。
并行episodes与单worker隔离panel不可重叠；数值库threads=1。完整wall包含catalogue、partial搜索、特征、推断、累计unique rollout、最终验证、应急及bookkeeping；模型加载/IO/环境step单列。
模型只fit一次，threshold=.5，seed7401。不得根据test结果调参或重训。

MANIFEST.json列内容文件SHA256和bytes，不自包含自身hash。包外validation.json记录CRC/逐entry检查，zip.sha256记录完整ZIP哈希。
"""


def package(repository, results_root, models_root, output_dir, final_commit, build=False):
    repository, results, models, output = map(Path, (repository, results_root, models_root, output_dir))
    target = output / f"terminal_stage7_documents_models_results_{final_commit[:7]}.zip"
    # Dry-run reports missing completion evidence without creating any directory.
    try:
        checks = completion_checks(repository, results, models, final_commit)
        entries = collect_sources(repository, results, models)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        if build:
            raise
        return {"status": "not_ready", "reason": str(exc), "writes": False,
                "scope": ["current docs/fixed-terminal source", "results/stage7", "inputs/stage7/models", "preflight"],
                "target": str(target)}
    if not build:
        return {"status": "ready", "writes": False, "checks": checks, "target": str(target),
                "source_files": len(entries), "entries": list(entries)}
    if output.resolve().is_relative_to(results.resolve()) or output.resolve().is_relative_to(models.resolve()):
        raise ValueError("delivery output must not be inside an input tree")
    for destination in (target, output / "validation.json", output / "zip.sha256"):
        if destination.exists():
            raise FileExistsError("refusing to overwrite delivery: " + str(destination))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "fixed-terminal-stage7-bundle-v1", "final_commit": final_commit, "files": {}}
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
    for name in ("results-root", "models-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--final-commit", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = package(Path(__file__).resolve().parents[2], args.results_root, args.models_root,
                         args.output_dir, args.final_commit, args.build)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in ("ready", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
