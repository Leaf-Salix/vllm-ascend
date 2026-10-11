# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Validate callback inlining or a final dispatch check before notification DMA."""

import argparse
import sys
from contextlib import contextmanager

import task_boundary_runtime_poll_probe as runtime_probe
from pypto.backend import pto_backend


def make_instrument(variant):
    @contextmanager
    def instrument(mode, cube):
        original = pto_backend._compiled_late_poll

        def callback(func):
            helper, entry = original(func)
            if not helper:
                return helper, entry
            if variant == "inline":
                return helper.replace(
                    "__attribute__((noinline)) void pypto_poll_local_tail",
                    "__attribute__((always_inline)) inline void pypto_poll_local_tail",
                ), entry
            assert variant == "recheck"
            marker = "    auto* scratch = reinterpret_cast<"
            assert helper.count(marker) == 1
            check = """    uint32_t dispatched;
    asm volatile("MOV %0, DATA_MAIN_BASE" : "=l"(dispatched) : : "memory");
    if (dispatched != ctx->reg_task_id && dispatched != 0x7ffffffdu) return;
"""
            return helper.replace(marker, check + marker), entry

        try:
            pto_backend._compiled_late_poll = callback
            with runtime_probe.instrument(mode, cube):
                yield
        finally:
            pto_backend._compiled_late_poll = original

    return instrument


def validate(*args):
    result = runtime_probe.validate(*args)
    producer_trace = args[4]
    result["late_without_notice"] = [
        handoff for handoff in result["late_handoffs"] if producer_trace[handoff["producer"]][28] & 256 == 0
    ]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=("inline", "recheck"), default="inline")
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    runtime_probe.probe.instrument = make_instrument(options.variant)
    runtime_probe.probe.validate = validate
    runtime_probe.probe.main()
