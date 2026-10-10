from pathlib import Path
import subprocess
import zipfile


TREE = "9db2cec12f962dffceced5a1a649269be57c4ced"
ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "staged_9db2cec12f962dffceced5a1a649269be57c4ced_binarysafe.zip"
DEST = ROOT / "staged_9db2cec12f962dffceced5a1a649269be57c4ced_binarysafe"


if ARCHIVE.exists() or DEST.exists():
    raise SystemExit("binary-safe archive destination already exists")

entries = subprocess.check_output(["git", "ls-tree", "-r", "-z", TREE])
with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as output:
    for record in entries.split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = header.split()
        if kind != b"blob":
            continue
        data = subprocess.check_output(["git", "cat-file", "blob", object_id.decode("ascii")])
        name = raw_path.decode("utf-8")
        info = zipfile.ZipInfo(name)
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = int(mode, 8) << 16
        output.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)

with zipfile.ZipFile(ARCHIVE) as source:
    source.extractall(DEST)

print(f"archive={ARCHIVE}")
print(f"extract={DEST}")
print(f"entries={len(source.namelist())}")
