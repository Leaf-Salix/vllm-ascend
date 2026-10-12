# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Bounded experiment: reuse clean context for explicitly synchronous entries."""

import argparse
import difflib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import task_boundary_acceptance as acceptance
from pypto.backend import pto_backend
from simpler_setup.runtime_builder import RuntimeBuilder


def changes(root):
    base = root / "src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler"
    path = base / "scheduler_types.h"
    old = path.read_text()
    anchor = "    uint8_t pad0_;                          // offset 39: alignment padding"
    assert old.count(anchor) == 1
    new = old.replace(anchor, "    uint8_t ordinary_context_clean;         // offset 39: clean dual-slot bitmask")
    result = {path: (old, new)}
    path = base / "scheduler_dispatch.cpp"
    old = path.read_text()
    start = old.index("    auto &overlap = payload.global_context;")
    end = old.index("\n    if (to_pending) {", start)
    section = old[start:end]
    section = section.replace("    const uint8_t member = 1u << static_cast<uint32_t>(subslot);\n", "")
    prefix = """    const uint8_t member = 1u << static_cast<uint32_t>(subslot);
    const auto &source_contract = *slot_state.payload;
    // Bits 3..5 are a compiler proof that the member never writes overlap state.
    // Legacy entries have no proof and retain their per-dispatch initialization.
    const bool ordinary =
        (source_contract.scalar_prefix_tagged_args & (member << 3)) != 0 &&
        ((source_contract.scalar_prefix | source_contract.runtime_barrier_tail |
          source_contract.runtime_compiled_poll) & member) == 0;
    const uint8_t buffer_bit = 1u << buf_idx;
    if (!ordinary || (core_exec_state.ordinary_context_clean & buffer_bit) == 0) {
"""
    suffix = """        if (ordinary) core_exec_state.ordinary_context_clean |= buffer_bit;
        else core_exec_state.ordinary_context_clean &= ~buffer_bit;
    }
"""
    new = old[:start] + prefix + "".join("    " + line for line in section.splitlines(keepends=True)) + suffix + old[end:]
    result[path] = (old, new)
    return result


def build(output):
    root = Path(os.environ["SCALAR_OVERLAP_ROOT"]) / "simpler"
    edited = changes(root)
    (output / "candidate.patch").write_text("".join(
        "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                    fromfile=f"a/{path.relative_to(root)}", tofile=f"b/{path.relative_to(root)}"))
        for path, (old, new) in edited.items()))
    try:
        for path, (_, new) in edited.items():
            path.write_text(new)
        binaries = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=True)
    finally:
        for path, (old, new) in edited.items():
            assert path.read_text() == new
            path.write_text(old)
    (output / "binaries.json").write_text(json.dumps({k: str(v) for k, v in asdict(binaries).items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", choices=("build", "compile", "run", "diagnostics"))
    parser.add_argument("--root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "compile":
        original = pto_backend._generate_single_chip

        def with_proof(*positional, **keywords):
            files = original(*positional, **keywords)
            for name, code in files.items():
                if name.startswith("orchestration/") and name.endswith(".cpp"):
                    assert ".set_scalar_prefix" not in code and ".set_runtime_barrier_tail" not in code
                    anchor = "params_t0.set_allow_early_resolve(true);"
                    assert code.count(anchor) == 1
                    files[name] = code.replace(anchor, "params_t0.set_scalar_prefix_tagged_args_mask(56);\n                " + anchor)
            return files

        with patch.object(pto_backend, "_generate_single_chip", with_proof):
            acceptance.artifact(root / "compiled", "plain", compile_now=True)
        return
    RuntimeBuilder._CACHE_DIR = root / "runtime/cache"
    RuntimeBuilder._LIB_DIR = root / "runtime/lib"
    if args.action == "build":
        build(root)
    elif args.action == "diagnostics":
        import pytest  # noqa: PLC0415 -- build and timing do not use pytest

        raise SystemExit(pytest.main(remaining))
    else:
        sys.argv = [sys.argv[0], *remaining]
        acceptance.main()


if __name__ == "__main__":
    main()
