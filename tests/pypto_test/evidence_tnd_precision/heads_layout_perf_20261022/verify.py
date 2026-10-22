"""仅用CPU重算三任务的原始计时、身份和按性能筛选的精度资格。"""

import hashlib
import json
import runpy
import sys
import tarfile
import tempfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    sums = {}
    for line in (here / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert Path(name).name == name
        assert hashlib.sha256((here / name).read_bytes()).hexdigest() == digest, name
        sums[name] = digest
    cases = ("heads-8k-b16-v1", "heads-8k-b16-v2", "heads-128k-b24-v1")
    assert all(case + "-evidence.tar.gz" in sums for case in cases)
    with tempfile.TemporaryDirectory(prefix="heads-perf-verify-") as temporary:
        root = Path(temporary)
        with tarfile.open(here / "tools.tar.gz") as archive:
            archive.extractall(root, filter="data")
        for case in cases:
            target = root / ("evidence-" + case)
            target.mkdir()
            with tarfile.open(here / (case + "-evidence.tar.gz")) as archive:
                archive.extractall(target, filter="data")
        previous = sys.argv
        try:
            sys.argv = [str(root / "recheck.py"), *cases]
            runpy.run_path(str(root / "recheck.py"), run_name="__main__")
        finally:
            sys.argv = previous
        actual = json.loads((root / "RECHECK.json").read_text())
        assert actual == json.loads((here / "RECHECK.json").read_text())
    print("PASS: 3 tasks, 108 event samples, 45 raw L1 windows; 8K exact, 128K accuracy skipped")


if __name__ == "__main__":
    main()
