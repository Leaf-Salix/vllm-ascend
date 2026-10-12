# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Read source descriptors for ready and gated local-boundary entries."""

import argparse
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import task_boundary_acceptance as acceptance
import task_boundary_inline_payload_probe as inline
import task_boundary_normal_state_probe as normal
from pypto.backend import pto_backend


def changes(root):
    result = inline.changes(root)
    base = root / "src/a2a3/runtime/tensormap_and_ringbuffer"
    path = base / "runtime/scheduler/scheduler_dispatch.cpp"
    old, new = result[path]
    anchor = """        int n = 0;
        for (int32_t i = 0; i < payload.tensor_count; i++) {
            dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
        }
        for (int32_t i = 0; i < payload.scalar_count; i++) {
            dispatch_payload.args[n++] = payload.scalars[i];
        }
"""
    assert new.count(anchor) == 1
    replacement = """        const uint8_t member = 1u << static_cast<uint32_t>(subslot);
        bool source_args = (payload.scalar_prefix_tagged_args & member) != 0 &&
            (payload.scalar_prefix_tagged_args & SCALAR_PREFIX_SOURCE_ARGS_CAPABILITY) != 0;
#if SIMPLER_DFX
        source_args = source_args && chip_swimlane_level_ == ChipSwimlaneLevel::DISABLED &&
            !is_pmu_enabled() && !is_dump_args_enabled();
#endif
        if (source_args) {
            dispatch_payload.args[0] = reinterpret_cast<uint64_t>(&payload);
        } else {
""" + "".join("    " + line for line in anchor.splitlines(keepends=True)) + "        }\n"
    result[path] = (old, new.replace(anchor, replacement))

    path = base / "aicore/aicore_executor.cpp"
    old, new = result[path]
    source_setup = """!ordinary && !chip_swimlane_enabled && !pmu_enabled && !dump_args_enabled &&
                              (exec_payload->global_context.scalar_prefix & SCALAR_PREFIX_SOURCE_ARGS) != 0"""
    anchor = "            bool source_args = false;"
    assert new.count(anchor) == 1
    new = new.replace(anchor, "            bool source_args = " + source_setup + ";")
    anchor = "                source_args = " + source_setup + ";\n"
    assert new.count(anchor) == 1
    new = new.replace(anchor, "")
    result[path] = (old, new)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", choices=("build", "compile", "run", "diagnostics"))
    parser.add_argument("--root", type=Path, required=True)
    args, _ = parser.parse_known_args()
    original = pto_backend._generate_single_chip

    def ready_source(*positional, **keywords):
        files = original(*positional, **keywords)
        anchor = """    auto* __source_payload = (__pypto_args_word & uint64_t{4}) != 0
        ? reinterpret_cast<__gm__ char*>(reinterpret_cast<__gm__ DispatchPayload*>(
              reinterpret_cast<__gm__ char*>(args) - offsetof(DispatchPayload, args))->src_payload)
        : nullptr;
"""
        replacement = """    __gm__ char* __source_payload = nullptr;
    if ((__pypto_args_word & uint64_t{4}) != 0) {
        auto* __dispatch = reinterpret_cast<__gm__ DispatchPayload*>(
            reinterpret_cast<__gm__ char*>(args) - offsetof(DispatchPayload, args));
        const uint64_t __source_address = __dispatch->src_payload;
        __source_payload = reinterpret_cast<__gm__ char*>(
            __source_address != 0 ? __source_address : static_cast<uint64_t>(args[0]));
    }
"""
        for name, source in files.items():
            if name.startswith("kernels/") and anchor in source:
                assert source.count(anchor) == 1
                files[name] = source.replace(anchor, replacement)
        return files

    with patch.object(pto_backend, "_generate_single_chip", ready_source):
        if args.action == "compile":
            root = args.root.resolve()
            source = root.parent / "boundary_normal_state_20261012/compiled/plain"
            destination = root / "compiled/plain"
            assert not destination.exists()
            shutil.copytree(source, destination)
            acceptance.artifact(root / "compiled", "early", compile_now=True)
            return
        normal.changes = changes
        normal.main()


if __name__ == "__main__":
    main()
