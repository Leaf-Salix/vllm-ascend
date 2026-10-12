# SPDX-License-Identifier: Apache-2.0
"""Isolated TMR experiment: unroll the gated task's Scalar argument preparation."""

import argparse
import difflib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import tmr_csa_full_128k_b16_demo as demo
from simpler_setup.runtime_builder import RuntimeBuilder
from tmr_csa_score_fair_compile import PACKAGES
from tmr_csa_score_scalar_plan_demo import check_metadata_contract


def build(runtime_root):
    source = (
        Path(os.environ["SCALAR_OVERLAP_ROOT"])
        / "simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp"
    )
    original = source.read_text()
    changed = original
    for count in ("tensor_count", "scalar_count"):
        marker = f"                for (int32_t i = 0; i < {count}; i++) {{"
        assert changed.count(marker) == 1
        changed = changed.replace(marker, "#pragma unroll 4\n" + marker)
    runtime_root.mkdir(parents=True, exist_ok=True)
    relative = "src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp"
    (runtime_root / "candidate.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                changed.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    )
    try:
        source.write_text(changed)
        binaries = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=True)
    finally:
        assert source.read_text() == changed, "concurrent executor edit"
        source.write_text(original)
    (runtime_root / "binaries.json").write_text(
        json.dumps({key: str(value) for key, value in asdict(binaries).items()}, indent=2) + "\n"
    )
    print("Candidate built separately; original runtime source restored", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args, remaining = parser.parse_known_args()
    root = args.runtime_root.resolve()
    original_dirs = RuntimeBuilder._CACHE_DIR, RuntimeBuilder._LIB_DIR
    RuntimeBuilder._CACHE_DIR, RuntimeBuilder._LIB_DIR = root / "cache", root / "lib"
    try:
        if args.build:
            assert not remaining, remaining
            build(root)
        else:
            assert (root / "binaries.json").is_file()
            sys.argv = [sys.argv[0], *remaining]
            demo.PACKAGES.update(PACKAGES)
            with check_metadata_contract():
                demo.main()
    finally:
        RuntimeBuilder._CACHE_DIR, RuntimeBuilder._LIB_DIR = original_dirs


if __name__ == "__main__":
    main()
