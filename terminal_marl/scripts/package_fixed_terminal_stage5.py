"""Local-only stage5 delivery ZIP with explicit scope and completion checks."""
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
SUMMARY_COUNTS = dict(episodes=330, generations=30, base_pool_contexts=90,
                      large_pool_contexts=9, timing_panels=24, timing_rows=264, continuations=9)


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


def completion_checks(repository, results, final_commit):
    if re.fullmatch(r"[0-9a-f]{40}", final_commit) is None:
        raise ValueError("--final-commit must be a full lowercase SHA")
    if local_head(repository) != final_commit:
        raise ValueError("local HEAD differs from --final-commit")
    states = {}
    for phase, count in (("test", 360), ("continuation", 9)):
        state = read_json(safe_file(results / f"execution_{phase}.json", results))
        jobs = state.get("jobs", [])
        if (state.get("status") != "completed" or state.get("exit_code") != 0 or
                state.get("counts") != dict(total=count, pending=0, running=0, completed=count, failed=0) or
                len(jobs) != count or len({j.get("name") for j in jobs}) != count or
                any(j.get("status") != "completed" or j.get("returncode") != 0 for j in jobs) or
                state.get("model_hashes_unchanged") is not True):
            raise ValueError(f"{phase} phase is incomplete or inconsistent")
        states[phase] = state
    summary = read_json(safe_file(results / "summary/summary.json", results))
    if (summary.get("schema") != "fixed-terminal-stage5-summary-v1" or
            summary.get("status") != "completed" or summary.get("allow_incomplete") is not False or
            any(summary.get("counts", {}).get(k) != v for k, v in SUMMARY_COUNTS.items())):
        raise ValueError("formal stage5 summary is missing/incomplete")
    # Reject links before reading model bytes through the existing frozen-artifact validator.
    list(tree_files(results / "models"))
    from terminal_marl.scripts.evaluate_fixed_terminal_stage5 import frozen_models
    current = frozen_models(results / "models", smoke=False)
    for key in ("frozen_models_before", "frozen_models_after"):
        recorded = states["test"].get(key) or {}
        if (recorded.get("manifest_sha256") != current["manifest_sha256"] or
                recorded.get("manifest") != current["manifest"]):
            raise ValueError("current models differ from evaluation freeze: " + key)
    return {"test_jobs": 360, "continuation_jobs": 9, "summary_counts": SUMMARY_COUNTS,
            "manifest_sha256": current["manifest_sha256"], "model_hashes_unchanged": True}


def collect_sources(repository, results, stage4):
    entries = {}

    def add(path, root, prefix=""):
        path = safe_file(path, root)
        name = (Path(prefix) / path.relative_to(root)).as_posix()
        if name in entries:
            raise ValueError("duplicate archive entry: " + name)
        entries[name] = path

    def tree(path, root, prefix=""):
        for source in tree_files(path):
            add(source, root, prefix)

    for name in DOC_FILES + SOURCE_FILES:
        add(repository / name, repository)
    for pattern in SOURCE_GLOBS:
        for source in sorted(repository.glob(pattern)):
            add(source, repository)
    tree(repository / "docs/evidence/fixed_terminal_stage5", repository)
    tree(results, results, "results/stage5")
    for split in ("train", "val"):
        tree(stage4 / "data" / split, stage4, "inputs/stage4")
    tree(stage4 / "models", stage4, "inputs/stage4")
    add(stage4 / "summary/offline_test.json", stage4, "inputs/stage4")
    for load in ("low", "medium", "high"):
        for soc in ("high", "medium", "low"):
            directory = stage4 / "data/test" / f"{load}-{soc}-4303"
            for name in ("frozen_input.json", "generation.json"):
                add(directory / name, stage4, "inputs/stage4")
            tree(directory / "t15", stage4, "inputs/stage4")
    preflight = repository / "tmp_stage5_preflight"
    add(preflight / "stage4_diagnostics.json", preflight, "preflight")
    # Small inspection records only; omit smoke model/data/rollout copies.
    for path in sorted(preflight.rglob("*")):
        relative = path.relative_to(preflight)
        if any(p in ("models", "data", "rollouts", "__pycache__") for p in relative.parts):
            continue
        if path.is_file() and (path.suffix == ".log" or path.name in ("validation.json", "core_validation.json")):
            add(path, preflight, "preflight")
    return dict(sorted(entries.items()))


def bundle_readme(commit):
    return f"""# 阶段5文档、模型与结果包

源码分支：codex/arrival-restructure；最终文档/源码提交：{commit}。
本阶段比较当前可见机群联合批次的闭合FCFS参考筛选及TOTAL计算预算；不是长期Q或全局最优证明。

阅读顺序：docs/CURRENT_STATUS.md → docs/experiments/fixed_terminal_stage5_protocol.md
→ docs/arrival_restructure/stage5_learning_validation.md → docs/evidence/fixed_terminal_stage5/。
完整结果：results/stage5/summary/summary.json；执行状态与模型冻结记录：results/stage5/execution_test.json、execution_continuation.json、models/manifest.json。
inputs/stage4/包含固定训练/验证数据、旧模型和九个诊断快照；preflight/是流程/诊断记录，不替代正式效果结果。
历史阶段用户计划仅作上下文，未收旧162场完整episode。保留文档原相对链接，未收录的历史证据可在原仓库阅读。

正式base版本为NumPy2.5.1、SciPy1.18.1、torch2.13.0、scikit-learn1.9.1；实际实验安装torch2.13.0+cu130且仅用CPU。requirements-stage5.txt独立列出这些base版本，安装/重现以冻结模型manifest实际环境为准。本地预检的torch2.5.1/NumPy2.4.4仅作接口测试。
在解压根目录配置该环境后，可直接评价包内冻结模型，无需先重新训练。使用新输出目录，不覆盖原始数据：
python -m terminal_marl.scripts.run_fixed_terminal_stage5 --phase test --models results/stage5/models --output reproduction/results --workers 20 --execute
python -m terminal_marl.scripts.run_fixed_terminal_stage5 --phase continuation --stage4-root inputs/stage4 --output reproduction/results --workers 20 --execute
python -m terminal_marl.scripts.summarize_fixed_terminal_stage5 --root reproduction/results --output reproduction/results/summary
若需复现训练，另用新目录：
python -m terminal_marl.scripts.train_fixed_terminal_stage5 --stage4-root inputs/stage4 --output reproduction/models
来源提交通过环境变量TERMINAL_SOURCE_COMMIT传入；运行环境和实际模型hash见原始manifest。

MANIFEST.json列出每个内容文件的SHA256和bytes；清单不自包含其自身hash。
外侧validation.json记录CRC/逐entry检查，zip.sha256记录完整ZIP的SHA256。
"""


def package(repository, results_root, stage4_root, output_dir, final_commit, build=False):
    repository, results, stage4, output = map(Path, (repository, results_root, stage4_root, output_dir))
    target = output / f"terminal_stage5_documents_models_results_{final_commit[:7]}.zip"
    # Dry-run reports missing completion evidence without creating any directory.
    try:
        checks = completion_checks(repository, results, final_commit)
        entries = collect_sources(repository, results, stage4)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        if build:
            raise
        return {"status": "not_ready", "reason": str(exc), "writes": False,
                "scope": ["current docs/fixed-terminal source", "results/stage5", "inputs/stage4", "preflight"],
                "target": str(target)}
    if not build:
        return {"status": "ready", "writes": False, "checks": checks, "target": str(target),
                "source_files": len(entries), "entries": list(entries)}
    if output.resolve().is_relative_to(results.resolve()) or output.resolve().is_relative_to(stage4.resolve()):
        raise ValueError("delivery output must not be inside an input tree")
    for destination in (target, output / "validation.json", output / "zip.sha256"):
        if destination.exists():
            raise FileExistsError("refusing to overwrite delivery: " + str(destination))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "fixed-terminal-stage5-bundle-v1", "final_commit": final_commit, "files": {}}
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
    for name in ("results-root", "stage4-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--final-commit", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = package(Path(__file__).resolve().parents[2], args.results_root, args.stage4_root,
                         args.output_dir, args.final_commit, args.build)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in ("ready", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
