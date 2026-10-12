# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Build or run the state-prefetch runtime against unchanged task images."""

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from simpler_setup import platform_info, runtime_builder, runtime_compiler
from simpler_setup.runtime_builder import RuntimeBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.root.resolve()
    RuntimeBuilder._CACHE_DIR = root / "runtime/cache"
    RuntimeBuilder._LIB_DIR = root / "runtime/lib"
    if args.action == "build":
        source = Path(os.environ["SCALAR_OVERLAP_ROOT"]) / "simpler-state-prefetch-20261012"
        # The editable install resolves setup modules to the active checkout.
        # Build from the isolated source root without changing that installation.
        with ExitStack() as stack:
            for module in (runtime_builder, runtime_compiler, platform_info):
                stack.enter_context(patch.object(module, "PROJECT_ROOT", source))
            builder = RuntimeBuilder(platform="a2a3")
            assert builder.runtime_root == source
            assert builder._runtime_compiler.project_root == source
            print(f"BUILD_SOURCE={builder.runtime_root}", flush=True)
            binaries = builder.get_binaries("tensormap_and_ringbuffer", build=True)
        serialized = {k: str(v) for k, v in asdict(binaries).items()}
        (root / "binaries.json").write_text(json.dumps(serialized, indent=2))
        return
    sys.path.insert(0, str(Path(os.environ["SCALAR_OVERLAP_ROOT"]) / "pypto/tests/st/runtime/kernel"))
    import task_boundary_acceptance as acceptance  # noqa: PLC0415

    sys.argv = [sys.argv[0], *remaining]
    acceptance.main()


if __name__ == "__main__":
    main()
