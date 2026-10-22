# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the eight-case frozen baseline without relying on ignored runs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def materialize(output):
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    manifest = json.loads((here / "baseline.json").read_text())
    contents = {}
    for name, expected in manifest["files"].items():
        if name in manifest["overlay"]:
            value = (here / "baseline_overlay" / (name + ".source")).read_bytes()
        else:
            value = subprocess.check_output(
                ["git", "show", f"{manifest['revision']}:{manifest['package']}/{name}"], cwd=repo
            )
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError(f"Frozen eight-case baseline mismatch: {name}")
        contents[name] = value
    output.mkdir(parents=True, exist_ok=False)
    for name, value in contents.items():
        (output / name).write_bytes(value)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    materialize(parser.parse_args().output.resolve())
