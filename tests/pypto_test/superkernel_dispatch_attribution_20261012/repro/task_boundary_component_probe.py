# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Attribute the ordinary-path regression by rebuilding one original component.

These diagnostics intentionally support unmarked tasks only. They rebuild
original component source against current headers; no old binary is mixed in.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import task_boundary_hot_path_probe as hot
import task_boundary_normal_state_probe as normal


BASELINE = "ef6812891575eaf2a15051fc9abc659f5415c2e2"
COMPONENTS = {
    "core": "src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp",
    "cpu": "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp",
}


def changes(root, component):
    result = hot.changes(root)
    relative = COMPONENTS[component]
    path = root / relative
    original = subprocess.check_output(["git", "show", f"{BASELINE}:{relative}"], cwd=root, text=True)
    result[path] = (path.read_text(), original)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--component", choices=tuple(COMPONENTS), required=True)
    args, remaining = parser.parse_known_args()
    if args.action == "run":
        assert remaining[remaining.index("--modes") + 1 :] == ["plain"], "ordinary-only diagnostic"
    normal.changes = lambda root: changes(root, args.component)
    sys.argv = [sys.argv[0], args.action, "--root", str(args.root), *remaining]
    normal.main()


if __name__ == "__main__":
    main()
