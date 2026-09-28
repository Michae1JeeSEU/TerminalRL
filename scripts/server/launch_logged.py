"""Launch an argv command with exclusive lifecycle markers and separate logs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path


def _paths(prefix: Path) -> tuple[Path, Path, Path, Path]:
    base = str(prefix)
    return (
        Path(base + ".launcher.log"),
        Path(base + ".log"),
        Path(base + ".pid"),
        Path(base + ".exit"),
    )


def _write(path: Path, text: str, mode: str = "x") -> None:
    with path.open(mode, encoding="utf-8", newline="") as stream:
        stream.write(text)


def _trace(log: Path) -> None:
    with log.open("a", encoding="utf-8", newline="") as stream:
        traceback.print_exc(file=stream)


def _watch(prefix: Path, cwd: Path, command: list[str]) -> int:
    launcher_log, child_log, pid, exit_marker = _paths(prefix)
    try:
        with child_log.open("x", encoding="utf-8", newline="") as stream:
            try:
                child = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=stream,
                    start_new_session=True,
                )
            except BaseException:
                _trace(launcher_log)
                return 1
            _write(pid, str(child.pid))
            code = child.wait()
        _write(exit_marker, str(code))
        return code
    except BaseException:
        _trace(launcher_log)
        return 1


def _launch(prefix: Path, cwd: Path, command: list[str]) -> int:
    launcher_log, child_log, pid, exit_marker = _paths(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (launcher_log, child_log, pid, exit_marker) if path.exists()]
    if existing:
        raise FileExistsError("existing launcher marker(s): " + ", ".join(map(str, existing)))
    _write(launcher_log, "")
    watch = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--watch",
        "--log-prefix",
        str(prefix),
        "--cwd",
        str(cwd),
        "--",
        *command,
    ]
    try:
        with launcher_log.open("a", encoding="utf-8", newline="") as stream:
            launcher = subprocess.Popen(
                watch,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                start_new_session=True,
            )
    except BaseException:
        _trace(launcher_log)
        return 1
    print(f"launcher_pid={launcher.pid}")
    print(f"log={child_log}")
    return 0


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--log-prefix", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        command = list(args.command)
        if command[:1] == ["--"]:
            command.pop(0)
        if not command:
            raise ValueError("a command is required after --")
        if args.watch:
            return _watch(Path(args.log_prefix), Path(args.cwd), command)
        return _launch(Path(args.log_prefix), Path(args.cwd), command)
    except BaseException:
        if args.watch:
            _trace(Path(f"{args.log_prefix}.launcher.log"))
        else:
            print(f"launch failed: {traceback.format_exc().rstrip()}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
