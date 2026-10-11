# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Use Scalar-written UB notification tokens in the bounded AIV handoff probe.

Only this known small graph is compiled: its body never uses S->MTE3 event7.
General kernels require a matching PTOAS reservation before using this variant.
"""

import argparse
import sys
from contextlib import contextmanager

import task_boundary_runtime_poll_probe as runtime_probe
from pypto import ir
from pypto.backend import pto_backend


def make_instrument(words):
    @contextmanager
    def instrument(mode, cube):
        original = pto_backend._compiled_late_poll

        def scalar_token_poll(func):
            helper, entry = original(func)
            if not helper or func.func_type == ir.FunctionType.AIC:
                return helper, entry
            old = "    copy_gm_to_ubuf(scratch, source, 0, 1, 2, 0, 0);"
            assert helper.count(old) == 1
            stores = """#pragma unroll
    for (int i = 0; i < 8; ++i) scalar_scratch[i] = expected;"""
            if words == 1:
                # Only value[0] participates in the notification protocol.
                # The remaining scratch bytes are reserved and never inspected.
                stores = "    scalar_scratch[0] = expected;"
            replacement = (
                "    auto* scalar_scratch = reinterpret_cast<volatile __ubuf__ uint64_t*>(scratch);\n"
                f"{stores}\n"
                "    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);\n"
                "    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);"
            )
            return helper.replace(old, replacement), entry

        try:
            pto_backend._compiled_late_poll = scalar_token_poll
            with runtime_probe.instrument(mode, cube):
                yield
        finally:
            pto_backend._compiled_late_poll = original

    return instrument


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--words", type=int, choices=(1, 8), default=8)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    runtime_probe.probe.instrument = make_instrument(options.words)
    runtime_probe.probe.validate = runtime_probe.validate
    runtime_probe.probe.main()
