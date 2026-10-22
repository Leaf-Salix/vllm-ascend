# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the exact v122 reference from a Git revision and two saved files."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def materialize(output):
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    manifest = json.loads((here / "frozen_v122.json").read_text())
    # Resolve and verify all bytes before creating any output.
    contents = {}
    for name, expected in manifest["files"].items():
        overlay = here / "reference_overlay" / (name + ".source")
        if name in manifest["overlay"]:
            value = overlay.read_bytes()
        else:
            value = subprocess.check_output(
                ["git", "show", f"{manifest['revision']}:{manifest['package']}/{name}"], cwd=repo
            )
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError(f"Frozen v122 reconstruction mismatch: {name}")
        contents[name] = value
    output.mkdir(parents=True, exist_ok=False)
    for name, value in contents.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    materialize(args.output.resolve())
    print(args.output.resolve())
