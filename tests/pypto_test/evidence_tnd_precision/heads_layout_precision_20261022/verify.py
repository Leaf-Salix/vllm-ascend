"""仅用标准库复核封存的128K精度结果，不运行模型或NPU。"""

import hashlib
import json
import runpy
import sys
import tarfile
import tempfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    for line in (here / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert Path(name).name == name
        assert hashlib.sha256((here / name).read_bytes()).hexdigest() == digest, name
    case = "heads-precision-128k-b24-v1"
    with tempfile.TemporaryDirectory(prefix="heads-precision-") as temporary:
        root = Path(temporary)
        with tarfile.open(here / "tools.tar.gz") as archive:
            archive.extractall(root, filter="data")
        target = root / ("evidence-" + case)
        target.mkdir()
        with tarfile.open(here / (case + "-evidence.tar.gz")) as archive:
            archive.extractall(target, filter="data")
        previous = sys.argv
        try:
            sys.argv = [str(root / "recheck.py"), case]
            runpy.run_path(str(root / "recheck.py"), run_name="__main__")
        finally:
            sys.argv = previous
        assert json.loads((root / "RECHECK.json").read_text()) == json.loads((here / "RECHECK.json").read_text())
    print("PASS: 128K/B24 precision, 48 Native A/B boundaries, 12 extra replays; no new timing")


if __name__ == "__main__":
    main()
