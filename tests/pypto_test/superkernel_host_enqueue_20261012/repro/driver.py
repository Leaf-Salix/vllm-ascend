# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Add host enqueue clocks to the unchanged fixed-chain replay protocol."""

import argparse
import os
import sys
from pathlib import Path

from simpler_setup.runtime_builder import RuntimeBuilder

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--compact", action="store_true")
args, remaining = parser.parse_known_args()
if args.compact:
    root = Path(os.environ["SCALAR_OVERLAP_ROOT"]) / "artifacts/boundary_compact_dispatch_20261012"
    RuntimeBuilder._CACHE_DIR = root / "runtime/cache"
    RuntimeBuilder._LIB_DIR = root / "runtime/lib"
import task_boundary_host_enqueue as acceptance  # noqa: E402

sys.argv = [sys.argv[0], *remaining]
acceptance.main()
