# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Build matching private packet-layout runtime and task binaries, then replay."""

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from simpler_setup import kernel_compiler, platform_info, runtime_builder, runtime_compiler
from simpler_setup.runtime_builder import RuntimeBuilder


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", choices=("build", "compile", "run", "diagnostics"))
    parser.add_argument("--root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.root.resolve()
    private = Path(os.environ["SCALAR_OVERLAP_ROOT"])
    source = private / "simpler-compact-dispatch-20261012"
    RuntimeBuilder._CACHE_DIR = root / "runtime/cache"
    RuntimeBuilder._LIB_DIR = root / "runtime/lib"
    # The private packet layout requires matching includes during both compile
    # and load. Never run the frozen normal task images with this runtime.
    with ExitStack() as stack:
        for module in (runtime_builder, runtime_compiler, platform_info, kernel_compiler):
            stack.enter_context(patch.object(module, "PROJECT_ROOT", source))
        assert kernel_compiler.KernelCompiler().project_root == source
        if args.action == "build":
            builder = RuntimeBuilder(platform="a2a3")
            assert builder.runtime_root == source
            assert builder._runtime_compiler.project_root == source
            print(f"BUILD_SOURCE={source}", flush=True)
            binaries = builder.get_binaries("tensormap_and_ringbuffer", build=True)
            serialized = {k: str(v) for k, v in asdict(binaries).items()}
            (root / "binaries.json").write_text(json.dumps(serialized, indent=2))
            return
        sys.path.insert(0, str(private / "pypto-compact-dispatch-20261012/tests/st/runtime/kernel"))
        if args.action == "diagnostics":
            import pytest  # noqa: PLC0415

            raise SystemExit(pytest.main(remaining))
        import task_boundary_acceptance as acceptance  # noqa: PLC0415

        sys.argv = [sys.argv[0], args.action, *remaining]
        acceptance.main()


if __name__ == "__main__":
    main()
