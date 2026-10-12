# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Diagnostic-only real V/M completion marker for the current TMR protocol."""

import argparse
import difflib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from pypto.backend import pto_backend
from simpler_setup.runtime_builder import RuntimeBuilder


def build(root):
    simpler = Path(os.environ["SCALAR_OVERLAP_ROOT"]) / "simpler"
    path = simpler / "src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp"
    original = path.read_text()
    old = """        wait_started_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
        pipe_barrier(PIPE_ALL);
        compute_completed_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;"""
    new = """        wait_started_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
        if (trace_addr != 0) {
            if ((ctx->late_handoff_scratch & (uint64_t{1} << 40)) != 0) {
                wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
            } else {
                wait_flag(PIPE_M, PIPE_S, EVENT_ID6);
            }
        }
        compute_completed_at = trace_addr != 0 ? get_sys_cnt_aicore() : 0;
        pipe_barrier(PIPE_ALL);"""
    assert original.count(old) == 1
    changed = original.replace(old, new)
    (root / "runtime.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                changed.splitlines(keepends=True),
                fromfile=f"a/{path.relative_to(simpler)}",
                tofile=f"b/{path.relative_to(simpler)}",
            )
        )
    )
    try:
        path.write_text(changed)
        binaries = RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=True)
    finally:
        assert path.read_text() == changed
        path.write_text(original)
    (root / "binaries.json").write_text(
        json.dumps({k: str(v) for k, v in asdict(binaries).items()}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", choices=("build", "test"))
    parser.add_argument("--root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    RuntimeBuilder._CACHE_DIR = root / "runtime/cache"
    RuntimeBuilder._LIB_DIR = root / "runtime/lib"
    if args.action == "build":
        build(root)
        return

    original = pto_backend._generate_kernel_wrapper
    audit = {}

    def wrapper(func, *positional, **keywords):
        code = original(func, *positional, **keywords)
        if func.name != "boundary_probe_producer":
            return code
        cube = func.func_type.name == "AIC"
        engine = "M" if cube else "V"
        head, body = code.split("// --- ptoas-generated code ---", 1)
        assert not re.search(rf"(?:set|wait)_flag\(PIPE_{engine}, PIPE_S, EVENT_ID6\)", code)
        body, count = re.subn(
            r"(?m)^(\s*)pypto_set_next_task_start\(",
            rf"\1set_flag(PIPE_{engine}, PIPE_S, EVENT_ID6);\n\1pypto_set_next_task_start(",
            body,
        )
        assert count == 1, count
        if not cube:
            marker = "    // Forward to ptoas-generated function\n"
            assert body.count(marker) == 1
            body = body.replace(
                marker, "    __async_ctx->late_handoff_scratch |= uint64_t{1} << 40;\n" + marker
            )
        audit[engine] = {"marker": f"{engine}->S event6", "original_events_unchanged": True}
        return head + "// --- ptoas-generated code ---" + body

    import pytest  # noqa: PLC0415 -- build mode does not need the test runner

    with patch.object(pto_backend, "_generate_kernel_wrapper", wrapper):
        result = pytest.main(remaining)
    (root / "marker_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
