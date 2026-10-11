# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Supply direct DMA resources through original task metadata, without Set stores."""

import re
from contextlib import contextmanager

import task_boundary_direct_runtime_probe as direct_probe
import task_boundary_runtime_poll_probe as runtime_probe
from pypto import ir
from pypto.backend import pto_backend


def with_direct_resource(func, scratch):
    attrs = dict(func.attrs)
    attrs["task_boundary_direct_scratch"] = scratch
    return ir.Function(
        func.name,
        list(zip(func.params, func.param_directions)),
        func.return_types,
        func.body,
        func.span,
        type=func.func_type,
        level=func.level,
        role=func.role,
        attrs=attrs,
        requires_runtime_binding=func.requires_runtime_binding,
        ir_stage=func.ir_stage,
    )


@contextmanager
def clean_direct_notification():
    original_wrapper = pto_backend._generate_kernel_wrapper
    original_orch = pto_backend._codegen_core.generate_orchestration
    resources = {}

    def wrapper(func, *args, **kwargs):
        code = original_wrapper(func, *args, **kwargs)
        scratch = dict(func.attrs).get("task_boundary_scratch_base")
        if scratch is None or "void pypto_poll_local_tail(" not in code:
            return code
        start = code.index("void pypto_poll_local_tail(")
        end = code.index("\n}\n", start)
        events = re.findall(r"_flag\([^;]*EVENT_ID([0-7])", code[start:end])
        # Explicit events remain unchanged. Non-seven allocations keep their
        # existing compiled callback and retain exactly the same eligibility.
        if events and set(events) == {"7"} and 0 <= scratch <= 65534 * 64 and scratch % 64 == 0:
            resources[func.name] = scratch
        return code

    def orchestration(program, orch_func, *args, **kwargs):
        functions = [
            with_direct_resource(func, resources[func.name]) if func.name in resources else func
            for func in program.functions.values()
        ]
        program = ir.Program(functions, program.name, program.span)
        return original_orch(program, orch_func, *args, **kwargs)

    try:
        pto_backend._generate_kernel_wrapper = wrapper
        pto_backend._codegen_core.generate_orchestration = orchestration
        yield
    finally:
        pto_backend._generate_kernel_wrapper = original_wrapper
        pto_backend._codegen_core.generate_orchestration = original_orch


@contextmanager
def instrument(mode, cube):
    with runtime_probe.instrument(mode, cube), clean_direct_notification():
        yield


if __name__ == "__main__":
    runtime_probe.probe.instrument = instrument
    runtime_probe.probe.validate = direct_probe.validate
    runtime_probe.probe.main()
