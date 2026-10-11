# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Measure a small internal callback router without changing task authors' API."""

from contextlib import contextmanager

import task_boundary_runtime_poll_probe as runtime_probe
from pypto.backend import pto_backend


def rewrite_entry(code):
    if "void pypto_poll_local_tail(" not in code:
        return code
    signature = (
        'extern "C" __aicore__ __attribute__((always_inline)) '
        "void kernel_entry(__gm__ int64_t* args)\n{\n"
    )
    assert code.count(signature) == 1
    prefix, body = code.split(signature)
    callback_end = "        pypto_poll_local_tail(tail_args);\n        return;\n    }\n"
    assert body.count(callback_end) == 1 and body.endswith("}\n")
    index = body.index(callback_end) + len(callback_end)
    callback, task = body[:index], body[index:]
    # Keep the full ordinary body, its argument-tag decode, Wait, modes and
    # Scalar operands together. The runtime ABI and both tag values stay intact.
    return (
        prefix
        + "static __aicore__ __attribute__((noinline)) "
        + "void pypto_execute_local_task(__gm__ int64_t* args)\n{\n"
        + task
        + signature
        + callback
        + "    pypto_execute_local_task(args);\n}\n"
    )


@contextmanager
def thin_entry():
    original = pto_backend._generate_kernel_wrapper

    def wrapper(*args, **kwargs):
        return rewrite_entry(original(*args, **kwargs))

    try:
        pto_backend._generate_kernel_wrapper = wrapper
        yield
    finally:
        pto_backend._generate_kernel_wrapper = original


@contextmanager
def instrument(mode, cube):
    # Move diagnostic writes with their original body as well; inserting them
    # into the new router would observe the notification callback as a task.
    with runtime_probe.instrument(mode, cube), thin_entry():
        yield


if __name__ == "__main__":
    runtime_probe.probe.instrument = instrument
    runtime_probe.probe.validate = runtime_probe.validate
    runtime_probe.probe.main()
