# SPDX-License-Identifier: Apache-2.0
"""Freeze the independently built native-wait tools before device validation."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    baseline = json.loads((here / "bsh_baseline_manifest.json").read_text())
    files, repositories = {}, {}
    for name in ("pypto", "ptoas"):
        root = Path(baseline[name]["snapshot"]).resolve()
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if head != baseline[name]["commit"]:
            raise ValueError(f"{name} no longer uses the frozen BSH source base")
        paths = subprocess.check_output(
            [
                "git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard",
                "--", "src", "include", "lib", "python", "tools",
            ],
            text=True,
        ).splitlines()
        selected = {
            root / path
            for path in paths
            if Path(path).suffix in (".py", ".pyi", ".cpp", ".h", ".hpp", ".td", ".inc")
        }
        if name == "pypto":
            for pattern in ("pypto_core*.so", "_torch_npu*.so"):
                matches = list((root / "python/pypto").glob(pattern))
                if len(matches) != 1:
                    raise ValueError(f"Missing or ambiguous own-worktree build: {root}/{pattern}")
                selected.add(matches[0])
        else:
            binary = root / "build/bin/ptoas"
            if not binary.is_file():
                raise ValueError("PTOAS build must finish before freezing")
            selected.add(binary)
        for path in sorted(selected):
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        repositories[name] = {
            "root": str(root),
            "base_commit": head,
            "patch": subprocess.check_output(
                ["git", "-C", str(root), "diff", "--", "src", "include", "lib", "python"], text=True
            ),
        }
    result = {
        "purpose": "Both paired sides use these tools; only TND requests the native 150-tick wait attribute",
        "build_provenance": "Built from each isolated worktree; no binaries copied from the BSH tooltrees",
        "repositories": repositories,
        "files": files,
    }
    destination = here / "native_toolchain_manifest.json"
    with destination.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(destination, "files", len(files))


if __name__ == "__main__":
    main()
