"""CPU-only verification of archived single-layer precision and raw L1 timing."""

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def extract_regular(archive, destination):
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert sum(member.size for member in members) < 50_000_000
        for member in members:
            name = Path(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source:
                target.write_bytes(source.read())


def verify():
    for name, digest in json.loads((ROOT / "SHA256.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        extract_regular(ROOT / "reproduction-tools.tar.gz", work)
        manifest = json.loads((work / "tools/manifest.json").read_text())
        assert len(manifest) == 4
        for name, digest in manifest.items():
            assert hashlib.sha256((work / "tools" / name).read_bytes()).hexdigest() == digest, name
        packages = sorted(ROOT.glob("fixed-l1-*-evidence.tar.gz"))
        assert len(packages) == 8
        for path in packages:
            shutil.copy2(path, work / path.name)
        subprocess.run(
            [sys.executable, str(work / "recheck.py")],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        result = json.loads((work / "RECHECK.json").read_text())
        assert result == json.loads((ROOT / "RECHECK.json").read_text())
        assert result["full_matrix_pass"] and result["validated_cases"] == 8
        assert sum(len(case["L1_samples_us"]) for case in result["cases"]) == 40
    return {"status": "PASS_SAVED_EIGHT_CASE_PRECISION_AND_L1", "cases": 8, "windows": 40, "NPU_replayed": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
