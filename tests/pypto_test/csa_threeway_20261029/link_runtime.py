# SPDX-License-Identifier: Apache-2.0
"""仅在新快照绑定已验收Native二进制；原环境和bank只读。"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("frozen_native", type=Path)
    args = parser.parse_args()
    native = args.frozen_native / "vllm_ascend"
    binaries = list(native.glob("*.so"))
    assert len(binaries) == 2 and any(p.name.startswith("vllm_ascend_C") for p in binaries)
    for arm in ("native", "ours", "nalinaly"):
        if not (args.cohort / arm).is_dir():
            continue
        root = args.cohort / arm / "source/vllm_ascend"
        rows = []
        for item in binaries:
            destination = root / item.name
            assert not destination.exists()
            destination.symlink_to(item.resolve())
            rows.append({"name": item.name, "target": str(item.resolve()), "bytes": item.stat().st_size})
        for name in ("_build_info.py", "_version.py"):
            if (native / name).is_file():
                shutil.copy2(native / name, root / name)
        (root.parent.parent / "runtime-links.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
