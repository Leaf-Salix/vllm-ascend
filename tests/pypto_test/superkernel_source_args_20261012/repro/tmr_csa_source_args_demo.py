# SPDX-License-Identifier: Apache-2.0
"""Isolated TMR experiment: wrappers read gated arguments from TaskPayload."""

import argparse
import difflib
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from pypto.backend import pto_backend
from pypto.pypto_core import ir as ir_core
from simpler_setup.runtime_builder import RuntimeBuilder

SOURCE_CAPABILITY = 0x80
SOURCE_CONTEXT = 8
SOURCE_TAG = 4


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def runtime_changes(root):
    base = "src/a2a3/runtime/tensormap_and_ringbuffer"
    executor = root / base / "aicore/aicore_executor.cpp"
    scheduler = root / base / "runtime/scheduler/scheduler_dispatch.cpp"
    ex = original_ex = executor.read_text()
    sc = original_sc = scheduler.read_text()
    marker = "            bool scalar_prefix_handoff = exec_payload->global_context.predecessor_context != 0;"
    ex = replace_once(ex, marker, marker + "\n            bool source_args = false;")
    start = ex.index("                int32_t tensor_count =")
    end = ex.index("                // An idle core", start)
    loop = ex[start:end]
    ex = (
        ex[:start]
        + (
            "                // Only wrappers explicitly advertising the source ABI may skip args stores.\n"
            "                source_args = !chip_swimlane_enabled && !pmu_enabled && !dump_args_enabled &&\n"
            f"                    (exec_payload->global_context.scalar_prefix & {SOURCE_CONTEXT}u) != 0;\n"
            "                if (!source_args) {\n"
            + "".join("    " + line for line in loop.splitlines(keepends=True))
            + "                }\n"
        )
        + ex[end:]
    )
    marker = "            // PMU window brackets kernel execution."
    ex = replace_once(
        ex,
        marker,
        (
            "            if (source_args) {\n"
            "                call_args = reinterpret_cast<__gm__ int64_t *>(\n"
            f"                    reinterpret_cast<uint64_t>(call_args) | {SOURCE_TAG}u);\n"
            "            }\n\n" + marker
        ),
    )
    marker = """    overlap.scalar_prefix = (slot_state.payload->scalar_prefix_tagged_args & member) != 0 ?
                               SCALAR_PREFIX_TAGGED_ARGS : 0u;"""
    sc = replace_once(
        sc,
        marker,
        marker
        + (
            "\n    // High metadata bit is an internal compiler capability, not a dispatch policy.\n"
            "    if (overlap.scalar_prefix != 0 &&\n"
            f"        (slot_state.payload->scalar_prefix_tagged_args & {SOURCE_CAPABILITY}u) != 0) {{\n"
            f"        overlap.scalar_prefix |= {SOURCE_CONTEXT}u;\n"
            "    }"
        ),
    )
    return {executor: (original_ex, ex), scheduler: (original_sc, sc)}


def build(runtime_root):
    root = Path(os.environ["SCALAR_OVERLAP_ROOT"]) / "simpler"
    changes = runtime_changes(root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "candidate.patch").write_text(
        "".join(
            "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    changed.splitlines(keepends=True),
                    fromfile=f"a/{path.relative_to(root)}",
                    tofile=f"b/{path.relative_to(root)}",
                )
            )
            for path, (original, changed) in changes.items()
        )
    )
    try:
        for path, (_, changed) in changes.items():
            path.write_text(changed)
        binaries = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=True)
    finally:
        for path, (original, changed) in changes.items():
            assert path.read_text() == changed, f"concurrent edit: {path}"
            path.write_text(original)
    (runtime_root / "binaries.json").write_text(
        json.dumps({key: str(value) for key, value in asdict(binaries).items()}, indent=2) + "\n"
    )


@contextmanager
def source_wrappers():
    original_wrapper = pto_backend._generate_kernel_wrapper
    original_generate = pto_backend._generate_single_chip
    audit = {}

    def wrapper(func, *args, **kwargs):
        text = original_wrapper(func, *args, **kwargs)
        if not pto_backend._function_uses_ops(func, pto_backend._LOCAL_TASK_WAIT_OPS):
            return text
        tensors = [p for p in func.params if isinstance(p.type, ir_core.TensorType)]
        scalars = [p for p in func.params if isinstance(p.type, (ir_core.ScalarType, ir_core.CommCtxType))]
        pre, entry = text.split("// --- Kernel entry point ---", 1)
        entry = replace_once(entry, "~uint64_t{1}", "~uint64_t{5}")
        marker = "    args = reinterpret_cast<__gm__ int64_t*>(__pypto_args_word & ~uint64_t{5});\n"
        entry = replace_once(
            entry,
            marker,
            marker
            + (
                "    static_assert(alignof(DispatchPayload) >= 8 && offsetof(DispatchPayload, args) % 8 == 0);\n"
                "    auto* __source_payload = (__pypto_args_word & uint64_t{4}) != 0\n"
                "        ? reinterpret_cast<__gm__ char*>(reinterpret_cast<__gm__ DispatchPayload*>(\n"
                "              reinterpret_cast<__gm__ char*>(args) - offsetof(DispatchPayload, args))->src_payload)\n"
                "        : nullptr;\n"
            ),
        )
        if scalars:
            entry = replace_once(
                entry,
                "        : nullptr;\n",
                "        : nullptr;\n"
                + (
                    "    auto* __source_scalars = __source_payload != nullptr\n"
                    "        ? reinterpret_cast<__gm__ uint64_t*>(__source_payload + TASKPAYLOAD_SCALARS_OFFSET)\n"
                    f"        : reinterpret_cast<__gm__ uint64_t*>(args + {len(tensors)});\n"
                ),
            )
        for i, param in enumerate(tensors):
            old = f"reinterpret_cast<__gm__ Tensor*>(args[{i}])"
            new = (
                "(__source_payload != nullptr\n"
                "        ? reinterpret_cast<__gm__ Tensor*>(__source_payload + TASKPAYLOAD_TENSORS_OFFSET"
                f" + {i} * TASKPAYLOAD_TENSOR_STRIDE)\n"
                f"        : {old})"
            )
            entry = replace_once(entry, old, new)
        for j, _ in enumerate(scalars):
            entry = replace_once(entry, f"args[{len(tensors) + j}]", f"__source_scalars[{j}]")
        audit[func.name] = {"tensors": len(tensors), "scalars": len(scalars)}
        return '#include "dispatch_payload.h"\n' + pre + "// --- Kernel entry point ---" + entry

    def generate(*args, **kwargs):
        files = original_generate(*args, **kwargs)
        for name, text in files.items():
            if name.startswith("orchestration/") and name.endswith(".cpp"):
                text = text.replace(
                    ".set_scalar_prefix_tagged_args(true)",
                    f".set_scalar_prefix_tagged_args_mask({SOURCE_CAPABILITY | 7})",
                )

                def cap(match):
                    return f".set_scalar_prefix_tagged_args_mask({int(match[1]) | SOURCE_CAPABILITY})"

                text = re.sub(r"\.set_scalar_prefix_tagged_args_mask\((\d+)\)", cap, text)
                files[name] = text
        return files

    with (
        patch.object(pto_backend, "_generate_kernel_wrapper", wrapper),
        patch.object(pto_backend, "_generate_single_chip", generate),
    ):
        yield audit


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("action", choices=("build", "compile", "diagnostics", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.action == "compile":
        assert not remaining
        import tmr_csa_full_128k_b16_compile as compiler
        from tmr_csa_score_fair_compile import PACKAGES

        compiler.PACKAGES.update(PACKAGES)
        with source_wrappers() as audit:
            compiler.compile_variant("early_score", root / "compiled/early_score")
        (root / "wrapper_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        assert len(audit) == 73, len(audit)
        return

    RuntimeBuilder._CACHE_DIR = root / "runtime/cache"
    RuntimeBuilder._LIB_DIR = root / "runtime/lib"
    if args.action == "build":
        assert not remaining
        build(root / "runtime")
    elif args.action == "diagnostics":
        import pytest

        with source_wrappers() as audit:
            result = pytest.main(remaining)
        (root / "diagnostic_wrapper_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        raise SystemExit(result)
    else:
        import tmr_csa_full_128k_b16_demo as demo
        from tmr_csa_score_fair_compile import PACKAGES
        from tmr_csa_score_scalar_plan_demo import check_metadata_contract

        assert (root / "runtime/binaries.json").is_file()
        sys.argv = [sys.argv[0], *remaining]
        demo.PACKAGES.update(PACKAGES)
        with check_metadata_contract():
            demo.main()


if __name__ == "__main__":
    main()
