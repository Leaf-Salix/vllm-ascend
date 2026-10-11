# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Validate free-event allocation and event-free fallback around explicit flags."""

import argparse
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import task_boundary_explicit_event_probe as explicit_probe
from pypto.backend import pto_backend


def make_instrument(event_id):
    @contextmanager
    def instrument(mode, cube):
        original = pto_backend._run_ptoas
        original_events = pto_backend._apply_task_boundary_events

        def observe_blocking_callback(func_name, helper, ptoas_code):
            helper, contract = original_events(func_name, helper, ptoas_code)
            if event_id == "exhausted" and func_name == "late_producer" and mode != "once":
                # Diagnostic-only marker, no timings are claimed for this
                # probe. The existing trace reserves word 8 for such markers.
                assert "PYPTO_TASK_BOUNDARY_BLOCKING_POLL_V1" in contract
                helper = helper.replace("__gm__ int64_t*)", "__gm__ int64_t* args)")
                assert helper.count("    pipe_barrier(PIPE_ALL);") == 1
                helper = helper.replace(
                    "    pipe_barrier(PIPE_ALL);",
                    "    auto* ctx = reinterpret_cast<__gm__ GlobalContext*>("
                    "args[SPMD_GLOBAL_CONTEXT_INDEX]);\n"
                    "    auto* tail_trace = reinterpret_cast<__gm__ uint64_t*>(ctx->scalar_overlap_trace);\n"
                    "    tail_trace[-16] = 1;\n"
                    "    dcci(tail_trace - 16, SINGLE_CACHE_LINE, CACHELINE_OUT);\n"
                    "    pipe_barrier(PIPE_ALL);",
                )
            return helper, contract

        def compile_events(pto_path, output_path, ptoas_flags=None):
            source = Path(pto_path)
            if source.name == "late_producer.pto":
                text = source.read_text()
                # The fixture has no previous user of this pair. Keep all
                # original data dependencies and add a balanced manual pair.
                assert "_flag[<PIPE_MTE2>, <PIPE_MTE3>" not in text
                marker = "    call @pypto_set_next_task_start("
                assert text.count(marker) == 1 and "%__notice_" not in text
                if event_id == "static":
                    events = (
                        "    pto.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID7>]\n"
                        "    pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID7>]\n"
                    )
                elif event_id == "exhausted":
                    # All eight IDs are possible across the existing blocks.
                    # The backend must preserve this pair without borrowing it.
                    block = re.findall(r"(%\w+): i32", text)[0]
                    events = f"""    %__notice_block = arith.index_cast {block} : i32 to index
    %__notice_seven = arith.constant 7 : index
    %__notice_id = arith.andi %__notice_block, %__notice_seven : index
    pto.set_flag_dyn [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_MTE3>, %__notice_id]
    pto.wait_flag_dyn [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_MTE3>, %__notice_id]
"""
                else:
                    # The first i32 entry argument is this fixture's SPMD block
                    # index. Even/odd blocks exercise both alternatives on NPU.
                    block = re.findall(r"(%\w+): i32", text)[0]
                    events = f"""    %__notice_block = arith.index_cast {block} : i32 to index
    %__notice_zero = arith.constant 0 : index
    %__notice_one = arith.constant 1 : index
    %__notice_seven = arith.constant 7 : index
    %__notice_bit = arith.andi %__notice_block, %__notice_one : index
    %__notice_even = arith.cmpi eq, %__notice_bit, %__notice_zero : index
    %__notice_id = arith.select %__notice_even, %__notice_zero, %__notice_seven : index
    pto.set_flag_dyn [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_MTE3>, %__notice_id]
    pto.wait_flag_dyn [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_MTE3>, %__notice_id]
"""
                text = text.replace(marker, events + marker)
                if mode == "once":
                    # The reference wrapper and orchestration omit the polling
                    # callback, so its compiler input requests no notification.
                    text = text.replace(", pto.runtime_late_handoff", "")
                source.write_text(text)
            return original(pto_path, output_path, ptoas_flags)

        try:
            pto_backend._run_ptoas = compile_events
            pto_backend._apply_task_boundary_events = observe_blocking_callback
            with explicit_probe.instrument(mode, cube):
                yield
        finally:
            pto_backend._run_ptoas = original
            pto_backend._apply_task_boundary_events = original_events

    return instrument


def validate_exhausted(*args):
    result = explicit_probe.runtime_probe.validate(*args)
    assert not result["polled_after_first_miss"], "blocking fallback must not submit notification DMA"
    # Freshly cleared for each eager input. Replays keep these diagnostic
    # markers, so their counts mean "seen since reset", not calls per replay.
    result["blocking_callbacks"] = [i for i, row in enumerate(args[4][:-1]) if row[8] == 1]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--event-id", choices=("static", "dynamic", "exhausted"), default="dynamic")
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    explicit_probe.runtime_probe.probe.instrument = make_instrument(options.event_id)
    explicit_probe.runtime_probe.probe.validate = (
        validate_exhausted if options.event_id == "exhausted" else explicit_probe.runtime_probe.validate
    )
    explicit_probe.runtime_probe.probe.main()
