import zipfile
from pathlib import Path

from scripts.server.pack_for_server import collect_server_files, pack_for_server


def test_pack_for_server_creates_manifest_and_zip(tmp_path: Path) -> None:
    (tmp_path / "terminal_marl").mkdir()
    (tmp_path / "terminal_marl" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "terminal_marl" / "module.pyc").write_text("skip", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "technical_spec.md").write_text("spec", encoding="utf-8")
    (tmp_path / "scripts" / "server").mkdir(parents=True)
    (tmp_path / "scripts" / "server" / "README_SERVER_TRAINING.md").write_text(
        "server", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_training_config_loader.py").write_text(
        "def test_x(): pass", encoding="utf-8"
    )
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    summary = pack_for_server(project_root=tmp_path)
    assert Path(summary["package"]).is_file()
    assert Path(summary["manifest"]).is_file()
    files = collect_server_files(tmp_path)
    assert Path("terminal_marl/__init__.py") in files
    assert Path("terminal_marl/module.pyc") not in files
    with zipfile.ZipFile(summary["package"]) as archive:
        names = set(archive.namelist())
    assert "SERVER_UPLOAD_MANIFEST.txt" in names
    assert "requirements.txt" in names

