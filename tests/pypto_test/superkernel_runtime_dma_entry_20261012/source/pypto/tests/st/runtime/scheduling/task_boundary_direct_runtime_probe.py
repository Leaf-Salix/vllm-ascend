# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Verify that runtime DMA uses the real core type after weak-symbol coalescing."""

import re
from contextlib import contextmanager

import task_boundary_runtime_poll_probe as runtime_probe
from pypto.backend import pto_backend


@contextmanager
def direct_notification():
    original = pto_backend._generate_kernel_wrapper

    def wrapper(func, *args, **kwargs):
        code = original(func, *args, **kwargs)
        scratch = dict(func.attrs).get("task_boundary_scratch_base")
        if scratch is None or "void pypto_poll_local_tail(" not in code:
            return code
        helper_start = code.index("void pypto_poll_local_tail(")
        helper_end = code.index("\n}\n", helper_start)
        events = re.findall(r"_flag\([^;]*EVENT_ID([0-7])", code[helper_start:helper_end])
        assert events and set(events) == {"7"}, "direct prototype requires the reserved default events"
        marker = "void pypto_set_next_task_start(__gm__ int64_t* args) {"
        assert code.count(marker) == 1
        start = code.index(marker) + len(marker)
        end = code.index("\n}", start)
        # Only this experiment uses the old runtime-owned scratch contract.
        # The normal compiler remains on metadata-only RuntimeCompiledPollClean.
        body = f"""
    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    ctx->late_handoff_scratch = {scratch};
    ctx->async_tail_kind = static_cast<uint32_t>(AsyncTailKind::RuntimePollable);"""
        return code[:start] + body + code[end:]

    try:
        pto_backend._generate_kernel_wrapper = wrapper
        yield
    finally:
        pto_backend._generate_kernel_wrapper = original


@contextmanager
def instrument(mode, cube):
    with runtime_probe.instrument(mode, cube), direct_notification():
        yield


def validate(*args):
    result = runtime_probe.validate(*args)
    trace, tile_rows = args[4], args[6]
    expected = 16 if tile_rows == 128 else 32
    for block in result["polled_after_first_miss"]:
        actual = (trace[block][28] >> 8) & 48
        assert actual == expected, f"runtime selected wrong core-specific helper: {actual} != {expected}"
    result["runtime_aic_verified_blocks"] = result["polled_after_first_miss"] if expected == 16 else []
    result["runtime_aiv_verified_blocks"] = result["polled_after_first_miss"] if expected == 32 else []
    return result


if __name__ == "__main__":
    runtime_probe.probe.instrument = instrument
    runtime_probe.probe.validate = validate
    runtime_probe.probe.main()
