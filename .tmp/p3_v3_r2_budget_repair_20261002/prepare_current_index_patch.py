"""Stageable CURRENT top only: preserve every pre-existing lower dirty edit."""
import difflib
from pathlib import Path
import subprocess

workspace = Path(__file__).resolve().parents[2]
name = "docs/CURRENT_STATUS.md"
head = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=workspace).decode("utf-8")
working = (workspace / name).read_text(encoding="utf-8")
marker = "## 2026-10-02 已批准首阶段修复与新增运动学阻塞（独立块）"
if head.count(marker) != 1 or working.count(marker) != 1:
    raise RuntimeError("CURRENT marker is not unique")
desired = working.split(marker, 1)[0] + marker + head.split(marker, 1)[1]
patch = "".join(difflib.unified_diff(
    head.splitlines(keepends=True), desired.splitlines(keepends=True),
    fromfile="a/" + name, tofile="b/" + name,
))
target = Path(__file__).resolve().parent / "current_status_top_only.patch"
target.write_bytes(patch.encode("utf-8"))
print(patch)
