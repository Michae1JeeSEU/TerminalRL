"""Guarded prepare-only launcher for the P3 blocker-ranking CV.

The manifest is the only source of the Python argv.  The default action is a
local dry-run; ``--execute`` remains blocked until the code worker's source
freeze is accepted and exact source hashes are written into the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / ".tmp" / "p3_blocker_ranking_cv_proposed_20261010.json"
DRYRUN_PATH = REPO / ".tmp" / "launch_p3_blocker_ranking_cv_20261010.dryrun.json"
PRIOR_MANIFEST_PATH = REPO / ".tmp" / "p3_neighbor_ring_reuse_cv_proposed_20261010.json"
SSH_TARGET = "lab-server"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"approval_pending", "registered"}:
        raise RuntimeError("manifest status is not approval_pending or registered")
    return manifest


def value_after(argv: list[str], option: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise RuntimeError(f"expected exactly one value for {option}")
    return argv[positions[0] + 1]


def values_after(argv: list[str], option: str, count: int) -> list[str]:
    positions = [index for index, token in enumerate(argv) if token == option]
    if len(positions) != 1:
        raise RuntimeError(f"expected exactly one {option} flag")
    start = positions[0] + 1
    values = argv[start : start + count]
    if len(values) != count or any(not value or value.startswith("-") for value in values):
        raise RuntimeError(f"{option} does not have {count} complete path values")
    if any(not value.startswith("/") for value in values):
        raise RuntimeError(f"{option} contains a non-absolute path")
    return values


def verify_exact_argv(argv: list[str], output: str) -> None:
    prior = json.loads(PRIOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    prior_argv = list(dict(prior["command"])["argv"])
    output_positions = [index for index, token in enumerate(prior_argv) if token == "--output"]
    if len(output_positions) != 1 or output_positions[0] + 1 >= len(prior_argv):
        raise RuntimeError("prior ring-reuse argv has no unique output option")
    output_index = output_positions[0]
    expected = prior_argv[:output_index] + ["--blocker-ranking", "--output", output]
    if argv != expected:
        raise RuntimeError("candidate argv differs from prior argv beyond output replacement and --blocker-ranking")


def collect_source_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    source = dict(manifest.get("source_freeze") or {})
    if source.get("status") != "READY":
        raise RuntimeError("source freeze is not READY")
    expected = source.get("source_hashes_expected_local")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("source freeze has no accepted hash list")
    normalized: dict[str, str] = {}
    for path, digest in expected.items():
        if not isinstance(path, str) or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise RuntimeError(f"invalid source hash entry: {path!r}")
        normalized[path] = digest.lower()
    return normalized


def verify_local_hashes(source_hashes: dict[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for relative_path, expected in source_hashes.items():
        path = REPO / relative_path
        if not path.is_file():
            mismatches.append(f"missing:{relative_path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[relative_path] = digest
        if digest.lower() != expected.lower():
            mismatches.append(f"sha:{relative_path}:{digest}!={expected}")
    if mismatches:
        raise RuntimeError("local source hash check failed: " + "; ".join(mismatches))
    return actual


def validate_frozen_argv(
    manifest: dict[str, Any], *, execute: bool
) -> tuple[list[str], str, str, int, str, dict[str, str], str]:
    command = dict(manifest["command"])
    argv = list(command["argv"])
    if not argv or argv[0] != command["python"]:
        raise RuntimeError("manifest argv does not begin with manifest python")
    if any("?" in token or "..." in token for token in argv):
        raise RuntimeError("manifest argv contains a placeholder")
    for option in ("--train-directories", "--additional-train-directories", "--supplemental-train-directories"):
        values_after(argv, option, 3)
    blocker_positions = [index for index, token in enumerate(argv) if token == "--blocker-ranking"]
    if len(blocker_positions) != 1:
        raise RuntimeError("expected exactly one --blocker-ranking flag")
    if blocker_positions[0] + 1 >= len(argv) or argv[blocker_positions[0] + 1] != "--output":
        raise RuntimeError("--blocker-ranking must immediately precede --output")
    output = value_after(argv, "--output")
    candidate = dict(manifest["candidate"])
    if output != candidate["output_remote"]:
        raise RuntimeError("argv output does not equal manifest candidate output")
    verify_exact_argv(argv, output)

    prefix_witness = value_after(argv, "--prefix-witness-directory")
    if not prefix_witness.endswith("/full"):
        raise RuntimeError("prefix witness path is not the validated full directory")
    remote_prefix = shlex.split(command["remote_shell_prefix"])
    if not remote_prefix or remote_prefix[0] != "env":
        raise RuntimeError("manifest remote shell prefix is not an env command")
    rendered_tokens = remote_prefix + argv
    rendered = shlex.join(rendered_tokens)
    roundtrip = shlex.split(rendered)
    if roundtrip != rendered_tokens:
        raise RuntimeError("rendered command shlex roundtrip changed tokens")
    actual_python_argv = roundtrip[len(remote_prefix) :]
    if actual_python_argv != argv:
        raise RuntimeError("rendered Python argv differs from frozen manifest argv")

    protocol = dict(manifest["protocol"])
    timeout = int(command["outer_timeout_seconds"])
    if timeout != 4200 or int(protocol["outer_timeout_seconds"]) != 4200:
        raise RuntimeError("outer timeout is not fixed at 4200 seconds")
    if int(command["workers"]) != 4 or int(protocol["wall_seconds_per_fold_arm"]) != 1200:
        raise RuntimeError("worker or fold budget changed")

    witness = value_after(argv, "--witness-directory")
    marker = "/.tmp/"
    if marker not in witness:
        raise RuntimeError("witness path does not identify the synced project root")
    remote_project = witness.split(marker, 1)[0]
    if not remote_project.startswith("/"):
        raise RuntimeError("remote project root is not absolute")

    source_status = str(dict(manifest["source_freeze"]).get("status"))
    if source_status == "READY":
        source_hashes = collect_source_hashes(manifest)
        local_hashes = verify_local_hashes(source_hashes)
    elif source_status == "NOT_READY":
        if execute:
            raise RuntimeError("execution blocked until source_freeze.status is READY")
        source_hashes = {}
        local_hashes = {}
    else:
        raise RuntimeError(f"unsupported source freeze status: {source_status}")
    return argv, rendered, remote_project, timeout, output, local_hashes, source_status


def quote(value: str) -> str:
    return shlex.quote(value)


def hash_check_line(source_hashes: dict[str, str]) -> str:
    entries = [f"{digest}  {path}" for path, digest in source_hashes.items()]
    return "printf '%s\\n' " + " ".join(quote(entry) for entry in entries) + " | sha256sum -c -"


def build_remote_script(
    *,
    rendered_command: str,
    remote_project: str,
    output: str,
    timeout: int,
    source_hashes: dict[str, str],
    source_status: str,
) -> tuple[str, dict[str, str]]:
    parent = output.rsplit("/", 1)[0]
    sibling_paths = {
        "log": output + ".log",
        "exit": output + ".exit",
        "pid": output + ".pid",
        "root_pid": output + ".root.pid",
        "wrapper_pid": output + ".wrapper.pid",
    }
    lines = ["set -eu", f"cd {quote(remote_project)}"]
    if source_status == "READY":
        lines.append(hash_check_line(source_hashes))
    else:
        lines.extend([
            "echo 'source freeze NOT_READY; launch is blocked' >&2",
            "exit 1",
        ])
    lines.extend([
        "for proc in /proc/[0-9]*; do",
        "  [ -r \"$proc/comm\" ] || continue",
        "  comm=$(cat \"$proc/comm\" 2>/dev/null || true)",
        "  case \"$comm\" in python|python3|python3.*) ;; *) continue ;; esac",
        "  if tr '\\0' '\\n' < \"$proc/cmdline\" 2>/dev/null | grep -Fxq 'terminal_marl.scripts.diagnose_p3_intent_neighbor'; then",
        "    echo 'existing terminal_marl.scripts.diagnose_p3_intent_neighbor process' >&2",
        "    exit 1",
        "  fi",
        "done",
        f"test ! -e {quote(output)}",
        f"test ! -e {quote(output + '/protocol.json')}",
        f"test ! -e {quote(output + '/result.json')}",
    ])
    for path in sibling_paths.values():
        lines.append(f"test ! -e {quote(path)}")
    wrapper_body = "\n".join([
        "set -u",
        f"cd {quote(remote_project)}",
        f"timeout {timeout} {rendered_command} > {quote(sibling_paths['log'])} 2>&1",
        "code=$?",
        f"printf '%s\\n' \"$code\" > {quote(sibling_paths['exit'])}",
        "exit \"$code\"",
    ])
    lines.extend([
        f"mkdir -p {quote(parent)}",
        f"nohup sh -c {quote(wrapper_body)} > /dev/null 2>&1 < /dev/null &",
        "pid=$!",
        f"printf '%s\\n' \"$pid\" > {quote(sibling_paths['pid'])}",
        f"printf 'pid=%s\\n' \"$pid\"",
    ])
    return "\n".join(lines), sibling_paths


def validate_wrapper_body(
    *, remote_script: str, remote_project: str, rendered_command: str, timeout: int, sibling_paths: dict[str, str]
) -> dict[str, Any]:
    marker = "nohup sh -c "
    suffix = " > /dev/null 2>&1 < /dev/null &"
    starts = [match.start() for match in re.finditer(re.escape(marker), remote_script)]
    if len(starts) != 1:
        raise RuntimeError("expected exactly one nohup wrapper command")
    start = starts[0]
    end = remote_script.find(suffix, start)
    if end < 0:
        raise RuntimeError("nohup wrapper has no expected redirection suffix")
    tokens = shlex.split(remote_script[start : end + len(suffix)])
    if tokens[:3] != ["nohup", "sh", "-c"]:
        raise RuntimeError("nohup wrapper does not use sh -c")
    body_lines = tokens[3].splitlines()
    expected = [
        "set -u",
        f"cd {quote(remote_project)}",
        f"timeout {timeout} {rendered_command} > {quote(sibling_paths['log'])} 2>&1",
        "code=$?",
        f"printf '%s\\n' \"$code\" > {quote(sibling_paths['exit'])}",
        "exit \"$code\"",
    ]
    if body_lines != expected:
        raise RuntimeError(f"wrapper body mismatch: {body_lines!r}")
    timeout_count = sum(f"timeout {timeout} " in line for line in remote_script.splitlines())
    if timeout_count != 1:
        raise RuntimeError(f"expected one timeout {timeout}, found {timeout_count}")
    return {
        "nohup_tokens_prefix": tokens[:3],
        "wrapper_body_lines": body_lines,
        "wrapper_body_line_count": len(body_lines),
        "timeout_occurrences": timeout_count,
        "no_duplicate_module_guard": "proc_comm_and_cmdline_exact_module",
    }


def dryrun_record(
    manifest: dict[str, Any], *, argv: list[str], rendered_command: str, remote_project: str,
    timeout: int, output: str, local_hashes: dict[str, str], remote_script: str,
    sibling_paths: dict[str, str], wrapper_validation: dict[str, Any], source_status: str
) -> dict[str, Any]:
    prefix = shlex.split(dict(manifest["command"])["remote_shell_prefix"])
    roundtrip = shlex.split(rendered_command)
    return {
        "status": "DRY_RUN_ONLY",
        "execution": "NOT_STARTED",
        "manifest": str(MANIFEST_PATH.relative_to(REPO)),
        "manifest_status": manifest["status"],
        "source_freeze_status": source_status,
        "manifest_argv": argv,
        "rendered_command": rendered_command,
        "rendered_tokens": prefix + argv,
        "roundtrip_tokens": roundtrip,
        "roundtrip_python_argv": roundtrip[len(prefix) :],
        "argv_roundtrip_equal": roundtrip[len(prefix) :] == argv,
        "argv_variation_guard": "prior ring-reuse argv plus --blocker-ranking and output replacement",
        "remote_project": remote_project,
        "ssh_target": SSH_TARGET,
        "outer_timeout_seconds": timeout,
        "workers": 4,
        "output": output,
        "sibling_paths_guarded_without_move_or_delete": sibling_paths,
        "source_hashes_local_verified": local_hashes,
        "remote_script": remote_script,
        "wrapper_validation": wrapper_validation,
        "execute_requires_explicit_flag": True,
        "execute_blocked_until_source_ready": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the guarded script through ssh lab-server")
    args = parser.parse_args()
    manifest = load_manifest()
    values = validate_frozen_argv(manifest, execute=args.execute)
    argv, rendered, remote_project, timeout, output, local_hashes, source_status = values
    source_hashes = collect_source_hashes(manifest) if source_status == "READY" else {}
    remote_script, sibling_paths = build_remote_script(
        rendered_command=rendered,
        remote_project=remote_project,
        output=output,
        timeout=timeout,
        source_hashes=source_hashes,
        source_status=source_status,
    )
    wrapper_validation = validate_wrapper_body(
        remote_script=remote_script,
        remote_project=remote_project,
        rendered_command=rendered,
        timeout=timeout,
        sibling_paths=sibling_paths,
    )
    if not args.execute:
        record = dryrun_record(
            manifest,
            argv=argv,
            rendered_command=rendered,
            remote_project=remote_project,
            timeout=timeout,
            output=output,
            local_hashes=local_hashes,
            remote_script=remote_script,
            sibling_paths=sibling_paths,
            wrapper_validation=wrapper_validation,
            source_status=source_status,
        )
        DRYRUN_PATH.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": record["status"],
            "artifact": str(DRYRUN_PATH.relative_to(REPO)),
            "argv_roundtrip_equal": record["argv_roundtrip_equal"],
            "source_freeze_status": source_status,
            "outer_timeout_seconds": timeout,
            "workers": 4,
            "execute": False,
        }, ensure_ascii=False))
        return 0
    return subprocess.run(["ssh", SSH_TARGET, remote_script], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
